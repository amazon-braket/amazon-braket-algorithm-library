import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from deepquantum.gates.qcircuit import Circuit as dqCircuit
import pennylane as qml
from braket.circuits import Circuit as bkCircuit
# import deepquantum.gates.qoperator as op

import pickle
import logging
import time
import os

log = logging.getLogger()
log.setLevel('INFO')


class Model(nn.Module):
    def __init__(self, inputsize, middlesize, outputsize):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.value_fc1 = nn.Linear(in_features=inputsize, out_features=middlesize)
        self.value_fc2 = nn.Linear(in_features=middlesize, out_features=outputsize)

    def forward(self, state):
        v = self.relu(self.value_fc1(state))
        v = self.value_fc2(v)
        return v


class CirModel(nn.Module):

    def __init__(self, n_qubits, device='local', framework='dq', shots=1000, layers=1, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        self.device = device
        self.framework = framework
        self.shots = shots
        self.dev = None

        he_std = gain * 5 ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.n_qubits = n_qubits
        self.n_layers = layers

        # # weights for old implementation
        # self.weights = nn.Parameter(nn.init.uniform_(torch.empty(6 * self.n_qubits), a=0.0, b=2 * np.pi) * init_std)
        # weights for pennlylane implementation
        # self.weights = nn.Parameter(nn.init.uniform_(torch.empty((n_layers, self.n_qubits)), a=0.0, b=2 * np.pi) * init_std)
        # global dev
        if framework == 'pennylane':
            # self.dev, _ = self._pl_def()
            self.weights = nn.Parameter(nn.init.uniform_(torch.empty((self.n_layers, self.n_qubits), dtype=torch.float64), a=0.0, b=2 * np.pi) * init_std)
            # ②
            self.weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
            self.dev, _ = self._pl_def()
            self.my_qnode = qml.QNode(self.qlcircuit, self.dev)
            self.pl_layer = qml.qnn.TorchLayer(self.my_qnode, self.weight_shapes)
            self.pl_layer.weights = self.weights
            self.pl_layer.qnode_weights['weights'] = self.weights

    def forward(self, x):
        #
        # if x.ndim == 2:
        #
        #     assert x.shape[0] == 1 and x.shape[1] == 2**(self.n_qubits)
        #     is_batch = False
        #     x = x.view([2]*self.n_qubits)
        # elif x.ndim == 3:
        #
        #     assert x.shape[1] == 1 and x.shape[2] == 2**(self.n_qubits)
        #     is_batch = True
        #     x = x.view([ x.shape[0] ]+[2]*self.n_qubits)
        # else:
        #
        #     raise ValueError("input x dimension error!")

        rst = None
        if self.device == 'local' and self.framework == 'turingq-dq':
            # wires_lst = list(range(self.n_qubits))
            # cir = dqCircuit(self.n_qubits)
            # cir.XYZLayer(wires_lst, w[0:3 * self.n_qubits])
            # cir.ring_of_cnot(wires_lst)
            # cir.YZYLayer(wires_lst, w[3 * self.n_qubits:6 * self.n_qubits])

            # x = cir.TN_contract_evolution(x, batch_mod=is_batch)

            # x0 = torch.clone(x).conj()
            # x = op.PauliZ(self.n_qubits, 0).TN_contract(x, batch_mod=is_batch)
            # s = x.shape
            # if is_batch == True:
            #     x = x.reshape(s[0], -1, 1)
            #     x0 = x0.reshape(s[0], 1, -1)
            # else:
            #     x = x.reshape(-1, 1)
            #     x0 = x0.reshape(1, -1)

            # rst = (x0 @ x).real
            # rst = rst.squeeze(-1)
            raise ValueError(f"device {self.device} for framework {self.framework} not implemented yet!")
        elif self.framework == 'pennylane':
            # self.dev, _ = self._pl_def()
            # self.my_qnode = qml.QNode(self.qlcircuit, self.dev)
            # self.pl_layer = qml.qnn.TorchLayer(self.my_qnode, self.weight_shapes)
            # self.pl_layer.weights = self.weights
            # self.pl_layer.qnode_weights['weights'] = self.weights
            rst = self.pl_layer(x)
            #

            # # ①
            # self.pl_layer = qml.QNode(self.qlcircuit, dev, interface='torch')
            # rst = self.pl_layer(x, self.weights)
            # #
            return rst

        elif self.device == 'local' and self.framework == 'aws-braket':
            cir = bkCircuit()
            cir.z(0)
            for i in range(1, self.n_qubits):
                cir.i(i)
            M = torch.tensor(cir.to_unitary()).type(dtype=torch.complex64)

            w = self.weights
            cir2 = bkCircuit()
            for which_q in range(0, self.n_qubits):
                cir2.ry(which_q, w[0+6*which_q])
                cir2.rz(which_q, w[1+6*which_q])
                cir2.ry(which_q, w[2+6*which_q])
                if which_q < (self.n_qubits-1):
                    cir2.cnot(which_q, which_q + 1)
                else:
                    cir2.cnot(which_q, 0)
                cir2.ry(which_q, w[3+6*which_q])
                cir2.rz(which_q, w[4+6*which_q])
                cir2.ry(which_q, w[5+6*which_q])
            unitary = torch.tensor(cir2.to_unitary(), requires_grad = True).type(dtype=torch.complex64)
            print(f"unitary {unitary} with size {unitary.size()}")
            print(f"x {x} with size {x.size()}")

            if x.shape[0] == 1:
                out = unitary @ x.T
                rst = (out.conj().T @ M @ out).real
            else:
                out = unitary @ x.T
                rst = (out.conj().T @ M @ out).diag().real
                rst = rst.reshape(-1,1)
        else:
            raise ValueError(f"device {self.device} for framework {self.framework} not implemented yet!")

        return rst
    
    def _pl_def(self):
        optimizer = None
        if self.device == 'local':
            # dev = qml.device("braket.local.qubit", wires=self.n_qubits)
            dev = qml.device("lightning.qubit", wires=self.n_qubits)
        elif self.device == 'sv1':
            dev = qml.device("braket.aws.qubit", 
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", 
            shots=self.shots,
            wires=self.n_qubits)
        elif self.device == 'aspen-m-3':
            dev = qml.device("braket.aws.qubit", 
            device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
            shots=self.shots,
            wires=self.n_qubits)
        elif self.device == 'aria-2':
            dev = qml.device("braket.aws.qubit", 
            device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2",
            shots=self.shots,
            wires=self.n_qubits)

        return dev, optimizer
    
    # @qml.qnode(dev, gradient_fn=optimizer)
    def qlcircuit(self, inputs, weights):
        # weights = self.weights
        # print(f"inputs size is {inputs.size()} and weights size is {weights.size()}")
        qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), pad_with=0, normalize=True)
        for i in range(self.n_layers):
            tempweight = weights[i].reshape(1, -1)
            qml.BasicEntanglerLayers(tempweight, wires=range(self.n_qubits), rotation=qml.RY)
        # qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RZ)
        # qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.expval(qml.PauliZ(wires=0))
    

class RetroRLModel:
    def __init__(self, method=None, **param):

        self.param = param
        # self.name = f"retrorl_{self.mol_data.name}_model"
        self.name = f"retrorl_model"
        # initial variables
        self.model_info = {}
        self.model = {}

        for mt in method:
            self.model_info[f"{mt}"] = {}
            self.model[f"{mt}"] = {}

            if mt == "retro-rl":
                logging.info(
                    "initial reinforcement learning for retrosynthetic-planning")
                for param in self.param[mt]["param"]:
                    self.model_info[mt][param] = set()
            elif mt == "retro-qrl":
                for param in self.param[mt]["param"]:
                    self.model_info[mt][param] = set()
                logging.info(
                    "initial quantum reinforcement learning for retrosynthetic-planning")

    def build_model(self, **param):

        for method, config in param.items():
            model_param = config
            if method == "retro-rl":
                self._build_retrorl_model(**model_param)
            elif method == "retro-qrl":
                self._build_retroqrl_model(**model_param)

        # self.NN = Model(inputsize, middlesize, outputsize)

    
    def _build_retrorl_model(self, **model_param):
        for inputsize in model_param["inputsize"]:
            for middlesize in model_param["middlesize"]:
                for outputsize in model_param["outputsize"]:
                    start = time.time()
                    model_name = f"{inputsize}_{middlesize}_{outputsize}"
                    # check availability
                    if model_name in self.model["retro-rl"].keys():
                        logging.info(
                            f"duplicate model !! pass !! inputsize {inputsize}, middlesize {middlesize}, outputsize {outputsize}")
                        continue
                    else:
                        self._update_model_info([inputsize, middlesize, outputsize], ["inputsize", "middlesize", "outputsize"], "retro-rl")

                    NN_model = Model(inputsize, middlesize, outputsize)

                    end = time.time()

                    self.model["retro-rl"][model_name] = {}
                    self.model["retro-rl"][model_name]["model_name"]= model_name
                    self.model["retro-rl"][model_name]["version"]= str(int(time.time()))
                    self.model["retro-rl"][model_name]["nn_model"] = NN_model

                    logging.info(
                        f"Construct model for inputsize:{inputsize},middlesize:{middlesize},outputsize:{outputsize} {(end-start)/60} min")

    def _build_retroqrl_model(self, **model_param):
        for n_qubits in model_param["n_qubits"]:
            for device in model_param["device"]:
                for framework in model_param["framework"]:
                    for shots in model_param["shots"]:
                        for layers in model_param["layers"]:
                            model_name = f"{n_qubits}_{device}_{framework}_{shots}_{layers}"
                            # check availability
                            if model_name in self.model["retro-qrl"].keys():
                                logging.info(
                                    f"duplicate model !! pass !! n_qubits {n_qubits}, device {device}, framework {framework}, shots {shots}, layers {layers}")
                                continue
                            else:
                                self._update_model_info([n_qubits, device, framework, shots, layers], ["n_qubits", "device", "framework", "shots", "layers"], "retro-qrl")

                            start = time.time()

                            NN_model = CirModel(n_qubits, device, framework, shots, layers)

                            end = time.time()

                            self.model["retro-qrl"][model_name] = {}
                            self.model["retro-qrl"][model_name]["model_name"]= model_name
                            self.model["retro-qrl"][model_name]["version"]= str(int(time.time()))
                            self.model["retro-qrl"][model_name]["nn_model"] = NN_model

                            logging.info(
                                f"Construct model for n_qubits:{n_qubits},device:{device},framework:{framework},layers:{layers} {(end-start)/60} min")

    def _update_model_info(self, values, names, method):
        for value, name in zip(values, names):
            self.model_info[method][name].add(value)

    def clear_model(self, method):
        for mt in method:
            self.model_info[f"{mt}"] = {}
            self.model[f"{mt}"] = {}

        return 0

    def describe_model(self):

        # information for model
        for method, info in self.model_info.items():
            logging.info(f"method: {method}")
            # param_len = len(info.keys())
            # if method == "pre-calc":
            #     logging.info(
            #         "The model_name should be {M}_{D}_{A}_{hubo_qubo_val}")
            for param, value in info.items():
                logging.info("param: {}, value {}".format(param, value))

        return self.model_info

    def get_model(self, method, model_name):

        return self.model[method][model_name]

    def save(self, version, path=None):
        save_path = None
        save_name = f"{self.name}_{version}.pickle"

        if path != None:
            save_path = os.path.join(path, save_name)
        else:
            save_path = os.path.join(".", save_name)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        logging.info(f"finish save {save_name}")
        return save_path

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)  # nosec