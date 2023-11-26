import numpy as np
import copy


class Reaction(object):
    def __init__(self, name):
        self.name = name
        self.reactant = []
        self.cost = []
        self.last_reactant = None
        self.path = []
        self.temp = []

    def add(self, product):
        self.reactant.append(product)
        product.last_reaction = self
        product.depth = 1 + self.last_reactant.depth

    def reverse(self):
        self.last_reactant.cost.append(self.cost)

    def calculate(self):
        num = 1
        for n in self.cost:
            num = num * len(n)
        temp = np.zeros(num).tolist()
        num_t = num
        for i in range(len(self.cost)):
            lth = len(self.cost[i])
            num_t = int(num_t / lth)  # 2  1  1
            for t in range(int(num / (lth * num_t))):
                for j in range(lth):  # 3  2  1
                    for k in range(num_t):  # 2  1  1
                        temp[k + j * num_t + t * lth * num_t] += self.cost[i][j]
        for i in range(num):
            temp[i] += 1
        self.cost = temp
        # if self.last_reactant.last_reaction is not None:
        #     self.reverse()
        self.reverse()

    def pathway(self):
        # if len(self.path) == 1:
        #     self.temp = copy.deepcopy(self.path[0])
        #     self.temp[0].insert(0, self.last_reactant.name)
        #     self.last_reactant.path.append(self.temp)
        # else:
        num = 1
        for n in self.path:
            num = num * len(n)
        temp = [[] for _ in range(num)]
        num_t = num
        for i in range(len(self.path)):
            lth = len(self.path[i])
            num_t = int(num_t / lth)  # 2  1  1
            for t in range(int(num / (lth * num_t))):
                for j in range(lth):  # 3  2  1
                    for k in range(num_t):  # 2  1  1
                        temp[k + j * num_t + t * lth * num_t] += self.path[i][j]
        self.temp = copy.deepcopy(temp)
        for i in self.temp:
            i.insert(0, self.last_reactant.name)
        self.last_reactant.path.append(self.temp)


class Product(object):
    def __init__(self, name):
        self.name = name
        self.reaction = []
        self.cost = []
        self.last_reaction = None
        self.depth = 0
        self.path = []
        self.temp = []

    def add(self, reaction):
        self.reaction.append(reaction)
        reaction.last_reactant = self

    def reverse(self):
        if self.last_reaction is not None:
            self.last_reaction.cost.append(self.cost)
            # if self.last_reaction.last_reactant.last_reaction is not None:
            #     self.last_reaction.reverse()

    def calculate(self):
        num = len(self.cost)
        if num == 1:
            temp = self.cost[0]
            self.cost = temp
        else:
            temp = copy.deepcopy(self.cost[0])
            for i in self.cost[1:]:
                temp += i
            # self.cost = np.array(temp, dtype=float).tolist()
            self.cost = temp

    def pathway(self):
        if len(self.path) == 1:
            self.temp = copy.deepcopy(self.path[0])
            if self.last_reaction is not None:
                self.last_reaction.path.append(self.temp)
        else:
            temp = []
            for i in self.path:
                for j in i:
                    temp.append(j)
            self.temp = copy.deepcopy(temp)
            if self.last_reaction is not None:
                self.last_reaction.path.append(self.temp)


def expansion(target, input_data_path):
    reactions_dictionary = np.load(input_data_path+'reactions_dictionary.npy', allow_pickle=True).item()
    Deadend = np.load(input_data_path+'Deadend.npy').tolist()
    buyable = np.load(input_data_path+'buyable.npy').tolist()
    if target.name not in reactions_dictionary.keys():
        if target.name in buyable:
            target.cost.append(0.0)
        elif target.name in Deadend:
            target.cost.append(100.0)
        else:
            target.cost.append(100.0)
        target.last_reaction.path.append([[target.name]])
        return
    elif target.depth == 9:
        target.cost.append(10.0)
        target.last_reaction.path.append([[target.name]])
        return

    num_r = 0
    for r in reactions_dictionary[target.name].keys():
        num_r += 1
        target.add(Reaction(r))
        num_p = 0
        for p in reactions_dictionary[target.name][r]:
            num_p += 1
            target.reaction[num_r - 1].add(Product(p))
            expansion(target.reaction[num_r - 1].reactant[num_p - 1], input_data_path)
            target.reaction[num_r - 1].reactant[num_p - 1].reverse()
        target.reaction[num_r - 1].calculate()
        target.reaction[num_r - 1].pathway()
    target.calculate()
    target.pathway()

