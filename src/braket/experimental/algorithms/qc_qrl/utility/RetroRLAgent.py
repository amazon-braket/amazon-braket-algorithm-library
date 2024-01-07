import numpy as np
import random
import sys
import copy
from .RetroGateModel import RetroRLModel

import torch
import torch.nn as nn
import torch.nn.functional as F
# from deepquantum.gates.qcircuit import Circuit as dqCircuit

from braket.circuits import Circuit as bkCircuit
# import deepquantum.gates.qoperator as op
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.local.local_job import LocalQuantumJob
from braket.aws import AwsQuantumJob
from braket.jobs.image_uris import Framework, retrieve_image
from braket.jobs.config import OutputDataConfig

import math
import logging
import time
import datetime
import pickle
import os

log = logging.getLogger()
log.setLevel('INFO')

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
current_path = os.path.realpath(os.path.dirname(__file__))

class RetroRLAgent:
    def __init__(self, build_model=False, method=None, load_path=None, agent_name=None, **param):
        self.model_name = param['model_name']
        init_method = list(param["init_param"].keys())
        init_param = param["init_param"]
        retro_rl_model = RetroRLModel(data=None, method=init_method, **init_param)
        if load_path != None:
            logging.info(f"try to load agent from {load_path} for pathway function...")
            logging.info("load data...")
            self.cost1 = {}
            self.cost2 = {}
            self.method = method
            self._load_data(input_data_path=load_path)
            # self.NN = model['nn_model']
            model_param={}
            method = 'retro-qrl'
            model_param[method] = {}
            model_param[method]['n_qubits'] = [8]
            model_param[method]['device'] = ['local']
            model_param[method]['framework'] = ['pennylane']
            model_param[method]['shots'] = [100]
            model_param[method]['layers'] = [1]

            retro_rl_model.build_model(**model_param)
            self.NN = retro_rl_model.get_model(method,self.model_name)['nn_model']
            self.NN.load_state_dict(torch.load(f"{load_path}/{agent_name}"))
            self.NN.eval()
        else:
            logging.info(f"initial a new agent...")
            self.param = param
            self.train_mode = param["train_mode"]
            # self.model_path = param["model_path"]
            self.model_name = param["model_name"]
            self.episodes = param["episodes"]
            self.epoches = param["epoches_per_episode"]
            self.method = method
            # load data
            logging.info("load data...")
            self._load_data()
            logging.info(f"build_model is {build_model}")

            self.job = None
            self.s3_save_path = None
            # try to update s3_path
            from datetime import datetime

            currentDay = datetime.now().day
            currentMonth = datetime.now().month
            currentYear = datetime.now().year

            current_time = f"{currentYear}{currentMonth}{currentDay}"
            self.save_name = f"{self.model_name}_agent_{current_time}.pt"
            try:
                s3_data_path = self.param['s3_data_path']
                self.s3_save_path = f"{s3_data_path}/{self.save_name}"
            except:
                self.s3_save_path = None

            if method == 'retro-qrl' or method == 'retro-rl':
                self.depth = 1
                self.maxdepth = 10
                self.cost1 = {}
                self.cost2 = {}
                self.is_model = False
                self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
                self.layer2 = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}}
                self.data = None
                self.updates = 0
                self.epsilon = 0.2
                self.loss_fn = torch.nn.MSELoss()

                if build_model != False:
                    # self.name = model['model_name']
                    # self.NN = model['nn_model']
                    model_param={}
                    # method = 'retro-qrl'
                    # model_param[method] = {}
                    # model_param[method]['n_qubits'] = [8]
                    # model_param[method]['device'] = ['local']
                    # model_param[method]['framework'] = ['pennylane']
                    # model_param[method]['shots'] = [100]
                    # model_param[method]['layers'] = [1]

                    model_param = param["model_param"]

                    logging.info(f"model_param is {model_param}")
                    retro_rl_model.build_model(**model_param)
                    self.NN = retro_rl_model.get_model(method,self.model_name)['nn_model']
                    self.opt = torch.optim.SGD(self.NN.parameters(), lr=0.001 / (1 + 2 * math.sqrt(self.updates)))

                self.tocost = {}
                self.avtocost = []
                self.lossv = []

    def _load_data(self, input_data_path=None):
        if input_data_path == None:
            input_data_path = self.param['data_path']
        self.file1 = np.load(f'{input_data_path}/reactions_dictionary.npy', allow_pickle=True).item()
        self.file2 = np.load(f'{input_data_path}/smiles_dictionary.npy', allow_pickle=True).item()
        self.file3 = np.load(f'{input_data_path}/target_product.npy').tolist()
        self.deadend = np.load(f'{input_data_path}/Deadend.npy').tolist()
        self.buyable = np.load(f'{input_data_path}/buyable.npy').tolist()
        # self.file1 = np.load('reactions_dictionary.npy', allow_pickle=True).item()
        # self.file2 = np.load('smiles_dictionary.npy', allow_pickle=True).item()
        # self.file3 = np.load('target_product.npy').tolist()
        # self.deadend = np.load('deadend.npy').tolist()
        # self.buyable = np.load('buyable.npy').tolist()

    def get_job(self):
        return self.job

    def get_job_arn(self):
        return self.job.arn

    def get_job_state(self):
        return self.job.state()

    def create_job(self, arn):
        self.job = AwsQuantumJob(arn)

    def game_job(self):
        device_name = self.model_name.split('_')[1]
        # model_name = self.name
        train_mode = self.param["train_mode"]

        # if self.method == 'retro-rl':
        #     self.game()
        #     return

        if train_mode == 'local-instance':
            self.game(self.episodes)
        else:
            device = None
            if device_name == 'sv1':
                device = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            elif device_name == 'aspen-m-3':
                device = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
            elif device_name == 'local':
                device = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            elif device_name == 'aria-2':
                device ="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2"
            else:
                device = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

            print(f"Going to run {device_name} mode")

            input_data = {}
            input_data['data'] = self.param['data_path']
            s3_data_path = self.param['s3_data_path']

            region = AwsSession().region
            image_uri = retrieve_image(Framework.PL_PYTORCH, region)

            interface = 'torch'

            def define_hyperparameters(interface):
                hyperparameters = {
                    "method": self.method,
                    "model_name": self.model_name,
                    # "model_path": self.model_path.split('/')[-1],
                    # Number of tasks per iteration = 2 * (num_nodes + num_edges) * p + 1
                    "p": "2",
                    # Maximum number of simultaneous tasks allowed
                    "max_parallel": "10",
                    # Number of total optimization iterations, including those from previous checkpoint (if any)
                    "num_iterations": "5",
                    # Step size / learning rate for gradient descent
                    "stepsize": "0.1",
                    # Shots for each circuit execution
                    "shots": self.model_name.split("_")[3],
                    "interface": interface,
                    "train_mode": train_mode,
                    # Parameters for reinforcement learning
                    "episodes": self.episodes,
                }
                return hyperparameters

            hyperparameters = define_hyperparameters(interface)

            job = None
            if train_mode == "local-job":
                job = LocalQuantumJob.create(
                    device=device,
                    source_module=f"{current_path}",
                    # Any unique name works. Note 50-character limit in job name
                    # (comment out to use default naming)
                    job_name="retrorl-job-" + device_name + "-" + interface + "-" + str(int(time.time())),
                    image_uri=image_uri,
                    # Relative to the source_module
                    entry_point="retrorl_algorithm_script",
                    # general parameters
                    hyperparameters=hyperparameters,
                    input_data=input_data,
                    output_data_config=OutputDataConfig(s3_data_path),
                )
            elif train_mode == "hybrid-job":
                if device_name == 'aspen-m-3' or device_name == 'aria-2':
                    t = datetime.datetime.utcfromtimestamp(time.time())
                    # print(t)
                    if t.hour == 5 or t.hour == 17:
                        if t.minute >= 50:
                            time.sleep(900)
                    job = AwsQuantumJob.create(
                        device=device,
                        source_module="hybridjobs",
                        job_name="retrorl-job-" + device_name + "-" + interface + "-" + str(int(time.time())),
                        image_uri=image_uri,
                        # Relative to the source_module
                        entry_point="hybridjobs.retrorl_algorithm_script",
                        copy_checkpoints_from_job=None,
                        # general parameters
                        hyperparameters=hyperparameters,
                        input_data=s3_data_path,
                        wait_until_complete=False,
                    )
                else:
                    job = AwsQuantumJob.create(
                        device=device,
                        source_module="hybridjobs",
                        job_name="retrorl-job-" + device_name + "-" + interface + "-" + str(int(time.time())),
                        image_uri=image_uri,
                        # Relative to the source_module
                        entry_point="hybridjobs.retrorl_algorithm_script",
                        copy_checkpoints_from_job=None,
                        # general parameters
                        hyperparameters=hyperparameters,
                        input_data=s3_data_path,
                        wait_until_complete=False,
                    )

            self.job = job

    def game(self, episodes):
        # for episode in range(1, 301):
        for episode in range(1, episodes):
            logging.info(f'episode {episode}')
            episodecost = 0
            for name in self.file3:
                self.layer[1] = [name]
                namecost = 0
                while self.depth < 10:
                    if self.layer[self.depth] == []:
                        # print(name)
                        self.depth = 10
                    else:
                        for name in self.layer[self.depth]:
                            rm, minv, _ = self.choosereaction(name, self.file1, self.file2, self.deadend, self.buyable)
                            namecost += minv
                            episodecost += minv
                            if rm:
                                for m in rm:
                                    self.add_child(m)
                        self.depth += 1
                # if name in self.tocost.keys():
                #     self.tocost[name].append(namecost)
                # else:
                #     self.tocost[name] = [namecost]
                self.renew()
            with torch.no_grad():
                avc = float(episodecost) / len(self.file3)
            self.avtocost.append(avc)
            self.merge()
            # if episode % 30 == 0:
            if episode % 1 == 0:
                self.updates += 1
                self.is_model = True
                logging.info(f"epsiode {episode} averate cost {avc} start training for {self.epoches} epoches...")
                self.train(self.file1, self.file2)
            if episode % 100 == 0:
                if self.epsilon - 0.05 > 0:
                    self.epsilon -= 0.03
                else:
                    self.epsilon = 0
                self.cost1 = {}
                self.cost2 = {}

    def renew(self):
        self.depth = 1
        self.layer = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
        self.layer2 = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}}
        self.data = None

    def choosereaction(self, name, file1, file2, deadend, buyable):
        minv = sys.maxsize
        mink = None
        if name not in file1.keys():
            return None, 0, None
        if np.random.random() > self.epsilon:  # name产物， r反应， rm反应物list， m反应物
            for r in file1[name].keys():
                rm = file1[name][r]
                tempv = 1
                if not self.is_model:
                    for m in rm:
                        if (m, 9 - self.depth) in self.cost1.keys():
                            rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                            tempv = tempv + rmv
                        else:
                            if m in buyable:
                                rmv = 0
                            elif m in deadend:
                                rmv = 100
                            elif self.depth == 9:
                                rmv = 10
                            else:
                                rmv = np.random.randint(1, 101)
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                            tempv = tempv + rmv
                else:
                    for m in rm:
                        if (m, 9 - self.depth) in self.cost1.keys():
                            rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                            tempv = tempv + rmv
                        else:
                            if m in buyable:
                                rmv = 0
                            elif m in deadend:
                                rmv = 100
                            elif self.depth == 9:
                                rmv = 10
                            else:
                                fp = torch.tensor(file2[m], dtype=torch.float)
                                depth = torch.tensor([self.depth], dtype=torch.float)
                                fp = torch.cat([fp, depth])
                                if self.method == 'retro-qrl':
                                    fp = fp.reshape(1, -1)
                                    fp = nn.functional.normalize(fp)
                                # print(f"choose reaction 1 type of forward tensor {fp.dtype}")
                                rmv = self.NN.forward(fp)[0]
                            if (m, 9 - self.depth) in self.cost2.keys():
                                self.cost2[m, 9 - self.depth].append(rmv)
                            else:
                                self.cost2[m, 9 - self.depth] = [rmv]
                            tempv = tempv + rmv
                if tempv < minv:
                    minv = tempv
                    mink = r
        else:
            mink = random.sample(file1[name].keys(), 1)[0]
            rm = file1[name][mink]
            tempv = 1
            if not self.is_model:
                for m in rm:
                    if (m, 9 - self.depth) in self.cost1.keys():
                        rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                        tempv = tempv + rmv
                    else:
                        if m in buyable:
                            rmv = 0
                        elif m in deadend:
                            rmv = 100
                        elif self.depth == 9:
                            rmv = 10
                        else:
                            rmv = np.random.randint(1, 101)
                        if (m, 9 - self.depth) in self.cost2.keys():
                            self.cost2[m, 9 - self.depth].append(rmv)
                        else:
                            self.cost2[m, 9 - self.depth] = [rmv]
                        tempv = tempv + rmv
            else:
                for m in rm:
                    if (m, 9 - self.depth) in self.cost1.keys():
                        rmv = sum(self.cost1[m, 9 - self.depth]) / len(self.cost1[m, 9 - self.depth])
                        tempv = tempv + rmv
                    else:
                        if m in buyable:
                            rmv = 0
                        elif m in deadend:
                            rmv = 100
                        elif self.depth == 9:
                            rmv = 10
                        else:
                            fp = torch.tensor(file2[m], dtype=torch.float)
                            depth = torch.tensor([self.depth], dtype=torch.float)
                            fp = torch.cat([fp, depth])
                            if self.method == 'retro-qrl':
                                fp = fp.reshape(1, -1)
                                fp = nn.functional.normalize(fp)
                            # print(f"choose reaction else type of forward tensor {fp.dtype}")
                            rmv = self.NN.forward(fp)[0]
                        if (m, 9 - self.depth) in self.cost2.keys():
                            self.cost2[m, 9 - self.depth].append(rmv)
                        else:
                            self.cost2[m, 9 - self.depth] = [rmv]
                        tempv = tempv + rmv
            minv = tempv

        return file1[name][mink], minv, mink

    def add_child(self, m):
        self.layer[self.depth + 1].append(m)

    def merge(self):
        self.cost1.update(self.cost2)

    def train(self, file1, file2):
        # for epoch in range(50):
        for epoch in range(self.epoches):
            start = time.time()
            y = []
            x = []
            # OLD: temp = random.sample(self.cost1.keys(), 128)
            temp = random.sample(self.cost1.keys(), 32)
            for i in temp:
                y.append(self.cost1[i][0])
                fp = np.array(file2[i[0]])
                depth = i[1]
                fp = np.append(fp, depth)
                x.append(fp)
            x = np.array(x)
            if self.method == 'retro-qrl':
                y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
                # x = torch.tensor(x, dtype=torch.float)+0j
                x = torch.tensor(x, dtype=torch.float64)
            else:
                y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
                x = torch.tensor(x, dtype=torch.float)
            if self.method == 'retro-qrl':
                x = nn.functional.normalize(x)
            # x = x.reshape(128,1,-1)
            dtype = x.dtype
            # print(f"train type of forward tensor {dtype} with size {x.size()}")
            # output = []
            # for sample in x:
            #     output.append(self.NN.forward(sample).reshape(-1,1))
            # print(f"output is {output}")
            # loss = self.loss_fn(np.array(output), y)

            # # ① 使用 qml.QNode(self.qlcircuit, dev, interface='torch')
            # output = self.NN.forward(x[0]).reshape(-1, 1)
            # for sample in x[1:]:
            #     output2 = self.NN.forward(sample).reshape(-1, 1)
            #     output = torch.cat([output, output2], axis=0)
            # #

            # ② 使用 qml.qnn.TorchLayer(qnode, weight_shapes)
            if self.method == 'retro-qrl':
                output = self.NN.forward(x).reshape(-1, 1)
            #
            else:
                output = self.NN.forward(x)
            # print(f"before calculate loss, with input x {x} with size {x.size()}")
            # print(f"before calculate loss, with label y {y} with size {y.size()}")
            # print(f"before calculate loss, with output {output} with size {output.size()}")
            loss = self.loss_fn(output, y)
            # print(f"y is {y}")
            # print(f"output is {output}")
            # print(f"loss is {loss}")
            # print(f"self.NN.weights.dtype is {self.NN.weights.dtype}")
            # if self.method == 'retro-qrl':
            #     for i in self.NN.parameters():
            #         print(i)
            # print(f"append loss data")
            self.lossv.append(loss.data)
            # print(f"zere grad for opt")
            self.opt.zero_grad()
            # print(f"loss backward")
            loss.backward()
            # print(f"opt step")
            self.opt.step()
            end = time.time()
            # print(f'finish epoch {epoch} for {(end - start) / 60} minutes')

    def add_child2(self, index, m):
        self.layer2[self.depth + 1][index] = m

    def pathway(self, name):
        self.renew()
        self.is_model = True
        self.epsilon = 0
        self.layer2[1]['1'] = name
        namecost = 0
        while self.depth < 10:
            if self.layer2[self.depth] == {}:
                self.depth = 10
            else:
                for key in self.layer2[self.depth].keys():
                    rm, minv, r = self.choosereaction(self.layer2[self.depth][key], self.file1, self.file2,
                                                      self.deadend, self.buyable)
                    namecost += minv
                    if rm:
                        for m in range(len(rm)):
                            index = key + str(m + 1)
                            self.add_child2(index, rm[m])
                self.depth += 1

        self.depth = 0
        data = self.DataS(name)
        data.add_img()
        # url = self.smiles2url(name)
        # data.data["img"] = url
        data.add_child()
        self.expansion1(name, data.data)
        self.data = data.data
        return namecost, data.data

    class DataS(object):

        def __init__(self, name):
            self.data = {
                "id": name
            }

        def add_child(self):
            self.data["children"] = []

        def add_img(self):
            self.data["img"] = None

    def expansion1(self, name, data):
        rm, minv, r = self.choosereaction(name, self.file1, self.file2, self.deadend, self.buyable)
        self.depth += 1
        if self.depth == 10:
            return
        if rm:
            tempr = self.DataS(str(r))
            tempr.add_child()
            data["children"].append(copy.deepcopy(tempr.data))
            for i in rm:
                self.expansion2(i, data["children"][0])

    def expansion2(self, name, data):
        rm, minv, r = self.choosereaction(name, self.file1, self.file2, self.deadend, self.buyable)
        tempm = self.DataS(name)
        tempm.add_img()
        # url = self.smiles2url(name)
        # tempm.data["img"] = url
        if rm:
            if self.depth < 9:
                tempm.add_child()
        data["children"].append(copy.deepcopy(tempm.data))
        if rm:
            self.expansion1(name, data["children"][data["children"].index(copy.deepcopy(tempm.data))])

    # def smiles2url(self, name):
    #     input_data_path = self.param['data_path']
    #     # smiles_map = np.load(f'{input_data_path}/smiles/smiles_map.npy', allow_pickle=True).item()
    #     smiles_map = np.load(f'{input_data_path}/smiles_map.npy', allow_pickle=True).item()
    #     index = smiles_map[name]
    #     url = "https://web-demo-test2.s3.us-west-2.amazonaws.com/data/smiles/" + str(index) + ".svg"

    #     return url

    def save(self, path=None):
        save_path = None
        save_name = self.save_name

        if path != None:
            save_path = os.path.join(path, save_name)
        else:
            save_path = os.path.join(".", save_name)


        logging.info(f"{self.s3_save_path}")

        # with open(save_path, "wb") as f:
        #     pickle.dump(self, f)

        torch.save(self.NN.state_dict(), save_path)

        logging.info(f"finish save {save_name}")

        return save_path, save_name

    def save_nn(self, version, path=None):
        save_path = None
        save_name = f"{self.model_name}_{version}"

        if path != None:
            save_path = os.path.join(path, save_name)
        else:
            save_path = os.path.join(".", save_name)

        # with open(save_path, "wb") as f:
        #     pickle.dump(self, f)
        torch.save(self.NN.state_dict(), save_path)
        logging.info(f"finish save agent{save_name}")
        return save_path, save_name

    def load_nn(self, filename):
        # with open(filename, "rb") as f:
        #     return pickle.load(f)  # nosec
        self.NN.load_state_dict(torch.load(filename))

    # @classmethod
    # def load(cls, filename):
    #     # initialize for pathway: inference function
    #     self._load_data(load_path)
    #         self.cost1 = {}
    #         self.cost2 = {}
    #     self.cost
    #     return RetroRLAgent(load_path)

    def get_parameter_num(self):
        total_trainable_params = sum(
            p.numel() for p in self.NN.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        return total_trainable_params

# tree =Tree()
# tree.game()

# np.save('tocost.npy', tree.tocost)
# np.save('lossv.npy', tree.lossv)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(range(0,len(tree.avtocost)),tree.avtocost,label='lossv')
# plt.figure(2)
# plt.plot(range(0,len(tree.lossv)),tree.lossv,label='lossv')
# plt.savefig('avtocost.svg',dpi=300,format='svg')
