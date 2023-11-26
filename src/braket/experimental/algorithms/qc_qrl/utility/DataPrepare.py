import pandas as pd
import numpy as np
import re
import random
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os.path
# from utility.BruteForceSearch import expansion, Product, Reaction
from .BruteForceSearch import expansion, Product, Reaction

import logging
log = logging.getLogger()
log.setLevel('INFO')


class Prepare(object):

    def __init__(self, data_file, fp_size=255, buyable_file=None, retrain=False, path=None, name=None):
        # parse file
        self.name = None
        self.path = path

        self.middle = None
        self.buyable2 = None
        self.target = None
        self.df = None
        self.selected2_reactant_uniques = None
        self.selected2_product_uniques = None
        self.fp_size = fp_size
        self.k = None
        self.exist = False

        file_type = data_file.split('.')[-1]

        if name is None:
            self.name = data_file.split('/')[-1].split('.')[0]
        else:
            self.name = name

        if path is None:
            self.path = data_file.split('.')[0].replace(self.name, '')

        if retrain is False:
            if (os.path.isfile(self.path+'Deadend.npy')) and (os.path.isfile(self.path+'buyable.npy')) and (os.path.isfile(self.path+'smiles_dictionary.npy')) and (os.path.isfile(self.path+'reactions_dictionary.npy')) and (os.path.isfile(self.path+'target_product.npy')):
                logging.info("Files are present.")
                self.exist = True
                pass
            else:
                if file_type == 'xlsx' or file_type == 'csv':
                    logging.info("parse {} file!".format(file_type))
                else:
                    logging.error(
                        "file type {} not supported! only support xlsx,csv".format(file_type))
                    raise Exception("file type not supported!")

                if buyable_file is not None:
                    buyable_file_type = buyable_file.split('.')[-1]
                    if buyable_file_type == 'xlsx' or buyable_file_type == 'csv':
                        logging.info("parse {} file!".format(buyable_file_type))
                    else:
                        logging.error(
                            "file type {} not supported! only support xlsx,csv".format(buyable_file_type))
                        raise Exception("file type not supported!")
                    buyable = pd.read_excel(buyable_file, engine='openpyxl')
                    self.buyable = buyable['buyable'].tolist()
                else:
                    self.buyable = None

                # read uspto-50k.xlsx
                self._read(data_file)
        else:
            if file_type == 'xlsx' or file_type == 'csv':
                logging.info("parse {} file!".format(file_type))
            else:
                logging.error(
                    "file type {} not supported! only support xlsx,csv".format(file_type))
                raise Exception("file type not supported!")

            if buyable_file is not None:
                buyable_file_type = buyable_file.split('.')[-1]
                if buyable_file_type == 'xlsx' or buyable_file_type == 'csv':
                    logging.info("parse {} file!".format(buyable_file_type))
                else:
                    logging.error(
                        "file type {} not supported! only support xlsx,csv".format(buyable_file_type))
                    raise Exception("file type not supported!")
                buyable = pd.read_excel(buyable_file, engine='openpyxl')
                self.buyable = buyable['buyable'].tolist()
            else:
                self.buyable = None

            # read uspto-50k.xlsx
            self._read(data_file)

    def _read(self, file):
        df = pd.read_excel(file, engine='openpyxl')
        listr = np.array(df['reactant'].tolist())
        listp = np.array(df['product'].tolist())
        listc = np.array(df['category'].tolist())

        # Treat the reactants by splitting punctuation
        punc = "."
        punc_re = '|'.join(re.escape(x) for x in punc)
        list11 = list()
        n = 0
        for i in range(len(listr)):
            tokens = re.sub(punc_re, lambda x: ' ' + x.group() + ' ', listr[i])
            tokens = tokens.split()
            num = tokens.count('.')
            if num > n:
                n = num
        for i in range(len(listr)):
            tokens = re.sub(punc_re, lambda x: ' ' + x.group() + ' ', listr[i])
            tokens = tokens.split()
            for k in range(n):
                if '.' in tokens:
                    tokens.remove('.')
                else:
                    tokens.append('none')
            list11.append(tokens)

        # Determine the amount of reactants in all reactions
        k = 0
        for j in range(len(listr)):
            c = list11[j]
            if len(c) > k:
                k = len(c)
        logging.info("There are at most {} reactants in one reaction!".format(k))
        self.k = k

        # Convert the resulting delimited data into a dataframe
        df1 = pd.DataFrame(list11, columns=['reactant'+str(i) for i in range(1, k+1)])
        list22 = listp.tolist()
        list33 = listc.tolist()
        df2 = pd.DataFrame(list22, columns=['product'])
        df3 = pd.DataFrame(list33, columns=['category'])
        df = pd.concat([df1, df2], axis=1, join='inner')
        df = pd.concat([df, df3], axis=1, join='inner')

        # for i in range(1, k+1):
        #     exec('list{} = {}'.format(i, df['reactant'+str(i)].tolist()))
        names = locals()
        for i in range(1, k+1):
            names['list%s' % i] = df['reactant'+str(i)].tolist()

        # names = locals()
        total_list = names.get('list' + str(1))
        for i in range(2, k+1):
            total_list.extend(names.get('list' + str(i)))

        reactant_uniques = set(total_list)
        len(reactant_uniques)  # 52990-reactants

        list5 = df['product'].tolist()
        product_uniques = set(list5)
        len(product_uniques)  # 49673-products

        # Using the non-repetitive set of reactants and products to get intermediate reactants (molecules contained in
        # both the reactants and the products), then reaction substrates and final products can be derived.
        middle = [x for x in reactant_uniques if x in product_uniques]  # 1965-intermediate reactants
        buyable = [y for y in reactant_uniques if y not in middle]  # 51025-reaction substrates

        # By querying the location of the intermediate products, the synthesis reactions of more than two steps are
        # screened.
        c = pd.DataFrame(columns=['reactant'+str(i) for i in range(1, k+1)] + ['product', 'category'])
        d = pd.DataFrame(columns=['reactant'+str(i) for i in range(1, k+1)] + ['product', 'category'])

        logging.info("Start picking dataset.")
        k1, k2 = 0, 0
        for i in tqdm(range(len(df))):
            for j in range(len(middle)):
                if middle[j] == df.iloc[i, k]:          # If intermediate reactant in products, we can store
                    c.loc[len(c)] = df.loc[i].tolist()  # intermediate reactions.
                    k1 += 1
                po = False
                for w in range(k):
                    if middle[j] == df.iloc[i, w]:
                        po = True
                if po is True:                          # If intermediate reactant in reactants, we can store
                    d.loc[len(d)] = df.loc[i].tolist()  # final reactions.
                    k2 += 1
            # print(i, k1, k2)
        logging.info("There are {} intermediate reactions.".format(k1))
        logging.info("There are {} final reactions.".format(k2))

        c.to_excel(self.path+"middle_reaction.xlsx", index=False)
        d.to_excel(self.path+"final_reaction.xlsx", index=False)
        logging.info("Intermediate reactions are saved!".format(k1))
        logging.info("Final reactions are saved!".format(k2))
        selected = pd.concat([c, d], axis=0, join='inner')
        # Remove the data with the same product and reaction category.
        selected2 = selected.drop_duplicates(['product', 'category'], keep='first', inplace=False)  # 5083
        selected2.to_excel(self.path+"selected2.xlsx", index=False)

        # for i in range(1, k+1):
        #     exec('listt{} = {}'.format(i, selected2['reactant'+str(i)].tolist()))

        names = locals()
        for i in range(1, k+1):
            names['listt%s' % i] = selected2['reactant'+str(i)].tolist()

        # names = locals()
        total_list2 = names.get('listt' + str(1))
        for i in range(2, k+1):
            total_list2.extend(names.get('listt' + str(i)))

        selected2_reactant_uniques = set(total_list2)
        len(selected2_reactant_uniques)  # 6195-all unique reactants

        list55 = selected['product'].tolist()
        selected2_product_uniques = set(list55)  # 5048-all unique products

        middle2 = [x for x in selected2_product_uniques if x in selected2_reactant_uniques]  # 1964intermediate reactant
        buyable2 = [y for y in selected2_reactant_uniques if y not in middle2]  # 4231 substrates
        target2 = [z for z in selected2_product_uniques if z not in middle2]  # 3084 target_product
        buyable2.pop(buyable2.index('none'))  # 4230
        logging.info("Storing data!")

        self.middle = middle2
        self.buyable2 = buyable2
        self.target = target2
        self.selected2_reactant_uniques = selected2_reactant_uniques
        self.selected2_product_uniques = selected2_product_uniques
        logging.info("Preparing is done!")

    def generate_files(self):
        if self.exist is False:
            self._generate_reactions_dictionary()
            self._generate_smiles_dictionary()
            self._generate_target()
            self._generate_buyalbe()
            self._generate_Deadend()
            logging.info("All files are generated!")
        else:
            logging.info("All files are generated!")

    def _generate_reactions_dictionary(self):
        """
        The generation of the reactions_dictionary is realized by the following code:
        df is the preprocessed data set read from the molecular synthesis database,
        which contains three columns: product, category, reactant,
        represent product, reaction category and composition respectively;
        The data format of the processed reactions_dictionary file is as follows:
        {product1:{category1:[reactant1,reactant1,reactant1...],category2:[reactant1,reactant1,reactant1...]...}}，
        That is, a combination of nested dictionaries and lists. """
        file1 = {}
        df = pd.read_excel(self.path+"selected2.xlsx", engine='openpyxl')
        logging.info("Start generating files!")
        for i in tqdm(range(len(df))):
            if df.loc[i]['product'] in file1.keys():
                file1[df.loc[i]['product']][df.loc[i]['category']] = []
            else:
                file1[df.loc[i]['product']] = {df.loc[i]['category']: []}
            for n in range(1, self.k+1):
                if df.loc[i]['reactant'+str(n)] != 'none':
                    file1[df.loc[i]['product']][df.loc[i]['category']].append(df.loc[i]['reactant'+str(n)])
        file_path = self.path + 'reactions_dictionary.npy'
        np.save(file_path, file1)
        logging.info("reactions_dictionary is saved!")

    def _generate_smiles_dictionary(self):
        """
        The smiles_dictionary is used to record the molecular code of each molecule. The molecular code is implemented by
        the Morgan fingerprint coding API in the rdkit library. The generation of the file is realized by the
        following code:
        temp is a list, which saves the SMILES codes of all molecules in the database.
        The data format of the processed file2 file is as follows:
        {smiles1:fingerprint1,smiles2:fingerprint2,...} """

        temp = list(self.selected2_reactant_uniques)
        temp.extend(list(self.selected2_product_uniques))
        temp.remove('none')
        temp = set(temp)

        def mol_to_fp(mol, FINGERPRINT_SIZE=16384, FP_rad=3):
            if mol is None:
                return np.zeros((FINGERPRINT_SIZE,), dtype=np.int)
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, FP_rad, nBits=FINGERPRINT_SIZE,
                                                                  useChirality=True, useFeatures=False), dtype=np.int32)

        def get_feature_vec(smiles, FINGERPRINT_SIZE, FP_rad):
            if not smiles:
                return np.zeros((FINGERPRINT_SIZE,), dtype=np.int)
            return mol_to_fp(Chem.MolFromSmiles(smiles), FINGERPRINT_SIZE, FP_rad)

        file2 = {}
        for i in temp:
            file2[i] = get_feature_vec(i, self.fp_size, 3)
        # file_path = self.path + 'smiles_dictionary' + '-' + str(self.fp_size) + '.npy'
        file_path = self.path + 'smiles_dictionary.npy'
        np.save(file_path, file2)
        logging.info("smiles_dictionary is saved!")

    def _generate_target(self):
        """
        The target_product file is used to record all molecular SMILES codes that need to be trained for retrosynthesis,
        extracted from the database, and saved in the following format:
        [smiles1,smiles2,...] """
        file3 = self.target
        file_path = self.path + 'target_product.npy'
        np.save(file_path, file3)
        logging.info("target_product file is saved!")

    def _generate_buyalbe(self):
        """
        The buyalbe file is used to record the SMILES codes of all the smallest commercially available synthetic
        substrate molecules, file is stored in the following format:
        [smiles1,smiles2,...] """

        if self.buyable is None:
            # 暂时使用随机
            logging.info("Temporarily random sampling.")
            self.buyable = random.sample(self.buyable2, int(0.5*len(self.buyable2)))
            file_path = self.path + 'buyable.npy'
            np.save(file_path, self.buyable)
        else:
            file_path = self.path + 'buyable.npy'
            np.save(file_path, self.buyable)
        logging.info("buyalbe file is saved!")

    def _generate_Deadend(self):
        """
        Deadend files are used to record SMILES codes for all non-commercial minimal synthetic substrate molecules.
        File is saved in the following formats:
        [smiles1,smiles2,...] """
        Deadend = self.buyable2.copy()
        for i in self.buyable:
            Deadend.remove(i)
        file_path = self.path + 'Deadend.npy'
        np.save(file_path, Deadend)
        logging.info("Deadend file is saved!")

    def generate_ground_truth(self):
        """
        The ground_truth file is used to record the true least cost value and its path obtained by the brute force
        search algorithm for the molecular results of all target products.
        File is saved in the following formats:
        {smiles1:{'cost':cost1, 'path':[smiles11,smiles12,...]},smiles2:{'cost':cost2, 'path':[smiles21,smiles22,...]},...}
        """
        if os.path.isfile(self.path + 'ground_truth.npy'):
            logging.info("File is present.")
            pass
        else:
            ground_truth = {}
            target_product = np.load(self.path+'target_product.npy').tolist()
            for i in tqdm(range(len(target_product))):
                target1 = Product(target_product[i])
                expansion(target1, self.path)
                index = target1.cost.index(min(target1.cost))
                ground_truth[target_product[i]] = {'cost': min(target1.cost), 'path': target1.temp[index]}

            file_path = self.path + 'ground_truth.npy'
            np.save(file_path, ground_truth)
            logging.info("ground_truth file is saved!")
