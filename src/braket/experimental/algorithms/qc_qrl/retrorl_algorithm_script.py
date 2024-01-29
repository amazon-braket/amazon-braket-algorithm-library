import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# from deepquantum.gates.qcircuit import Circuit as dqCircuit
from braket.circuits import Circuit as bkCircuit
from braket.tracking import Tracker
from braket.jobs import save_job_checkpoint

# import deepquantum.gates.qoperator as op
import boto3

import os
import math
import logging
import time
import json

from platform import python_version

os.environ["PYTHONUNBUFFERED"] = "1"

print("Current Python Version-", python_version())
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from utility.RetroGateModel import RetroRLModel
from utility.RetroRLAgent import RetroRLAgent

t = Tracker().start()

input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
output_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]
job_name = os.environ["AMZN_BRAKET_JOB_NAME"]  # noqa
checkpoint_dir = os.environ["AMZN_BRAKET_CHECKPOINT_DIR"]  # noqa
hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]

# Read the hyperparameters
with open(hp_file, "r") as f:
    hyperparams = json.load(f)
print(hyperparams)

p = int(hyperparams["p"])
# seed = int(hyperparams["seed"])
max_parallel = int(hyperparams["max_parallel"])
num_iterations = int(hyperparams["num_iterations"])
stepsize = float(hyperparams["stepsize"])
shots = int(hyperparams["shots"])
pl_interface = hyperparams["interface"]

model_name = hyperparams["model_name"]
# model_path = hyperparams["model_path"]
method = hyperparams["method"]
train_mode = hyperparams["train_mode"]
episodes = int(hyperparams["episodes"])

# to store files in a list
list = []

# dirs=directories
# print(input_dir)
# print(device_arn)
# print(file_dir)
# for (root, dirs, file) in os.walk(input_dir):
#     print(f'root {root} dirs {dirs} file {file}')

# file1 = np.load(f'{input_data_path}/reactions_dictionary.npy', allow_pickle=True).item()
# file2 = np.load(f'{input_data_path}/smiles_dictionary.npy', allow_pickle=True).item()
# file3 = np.load(f'{input_data_path}/target_product.npy').tolist()
# deadend = np.load(f'{input_data_path}/Deadend.npy').tolist()
# buyable = np.load(f'{input_data_path}/buyable.npy').tolist()

if "copy_checkpoints_from_job" in hyperparams:
    copy_checkpoints_from_job = hyperparams["copy_checkpoints_from_job"].split("/", 2)[-1]
else:
    copy_checkpoints_from_job = None

try:
    input_data_path = f"{input_dir}/input"
    os.system(f"ls -alh {input_data_path}")
    # retro_rl_model = RetroRLModel.load(f'{input_data_path}/{model_path}')
except Exception as e1:
    print(f"error e1 {e1}")
    try:
        # Second solution
        input_data_path = f"{input_dir}/data"
        os.system(f"ls -alh {input_data_path}")
        # retro_rl_model = RetroRLModel.load(f'{input_data_path}/{model_path}')
    except Exception as e2:
        print(f"error e1 {e1}")
        print(f"error e2 {e2}")
        try:
            # Second solution
            input_data_path = f"/opt/ml/input/data/input"
            os.system(f"ls -alh {input_data_path}")
            # retro_rl_model = RetroRLModel.load(f'{input_data_path}/{model_path}')
        except Exception as e3:
            # Handle both solutions failing
            print(f"error e1 {e1}")
            print(f"error e2 {e2}")
            print(f"error e3 {e3}")
            print(
                f"Can't find data in {input_dir}/input or {input_dir}/data or {input_data_path}!!!"
            )
else:
    # Code to execute when the first solution succeeds
    print(f"Found data in {input_data_path}!!")


# retro_model = retro_rl_model.get_model(method, model_name)

agent_param = {}
# initial the RetroRLModel object
init_param = {}
method = ["retro-rl", "retro-qrl"]

for mt in method:
    if mt == "retro-rl":
        init_param[mt] = {}
        init_param[mt]["param"] = ["inputsize", "middlesize", "outputsize"]
    elif mt == "retro-qrl":
        init_param[mt] = {}
        init_param[mt]["param"] = ["n_qubits", "device", "framework", "shots", "layers"]

# retro_rl_model = RetroRLModel(data=None, method=method, **init_param)
agent_param["init_param"] = init_param

model_param = {}
method = "retro-qrl"
model_param[method] = {}
model_param[method]["n_qubits"] = [int(model_name.split("_")[0])]
# model_param[method]['device'] = ['local', 'sv1', 'aspen-m-3', 'aria-2']
model_param[method]["device"] = [model_name.split("_")[1]]
model_param[method]["framework"] = [model_name.split("_")[2]]
# model_param[method]['shots'] = [100,1000]
model_param[method]["shots"] = [int(model_name.split("_")[3])]
# model_param[method]['layers'] = [1,2,3]
model_param[method]["layers"] = [int(model_name.split("_")[4])]

agent_param["model_param"] = model_param

agent_param["data_path"] = input_data_path
agent_param["train_mode"] = train_mode
agent_param["model_name"] = model_name
# agent_param["model_path"] = model_path
agent_param["episodes"] = episodes

# retro_rl_agent = RetroRLAgent(retro_model, method, **agent_param)
retro_rl_agent = RetroRLAgent(True, method, **agent_param)

# retro_rl_agent.game(path=input_data_path)
retro_rl_agent.game(episodes)

#
save_path, save_name = retro_rl_agent.save(path=input_data_path)
s3 = os.environ["AMZN_BRAKET_OUT_S3_BUCKET"]

s3_client = boto3.client("s3")

s3_client.upload_file(save_path, s3, f"data/{save_name}")

# os.system(f"aws s3 cp {save_path} s3://{s3}/data/{save_name}")
# AWS_REGION = "us-west-1"
# S3_BUCKET_NAME = "amazon-braket-us-west-1-493904798517"
# s3_client = boto3.client("s3", region_name=AWS_REGION)
# s3_client.upload_file(save_path, S3_BUCKET_NAME, f'data/{save_name}')
# job_name = os.environ["AMZN_BRAKET_JOB_NAME"]
# save_job_checkpoint(
#     checkpoint_data={"data": f"data for checkpoint from {job_name}"},
#     checkpoint_file_suffix="checkpoint-1",
#     )
# if retro_rl_agent.name.split('_')[1] == 'aspen-m2':
#     AWS_REGION = "us-west-1"
#     S3_BUCKET_NAME = "amazon-braket-us-west-1-493904798517"
#     s3_client = boto3.client("s3", region_name=AWS_REGION)
#     s3_client.upload_file(save_path, S3_BUCKET_NAME, f'data/{save_name}')
# else:
#     AWS_REGION = "us-west-1"
#     S3_BUCKET_NAME = "amazon-braket-us-west-1-493904798517"
#     s3_client = boto3.client("s3", region_name=AWS_REGION)
#     s3_client.upload_file(save_path, S3_BUCKET_NAME, f'data/{save_name}')
