# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import json
import os
import time
import sys
import ast

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.image_uris import Framework, retrieve_image
from braket.jobs import load_job_checkpoint, save_job_checkpoint, save_job_result
from braket.tracking import Tracker

import pennylane as qml

# Given an integer n, returns n random numbers such that the sum is greater than or equal to 0 and less than upper_total
def generate_random_numbers(n, upper_total):
    random.seed(0)
    random_numbers = [random.uniform(0, upper_total) for _ in range(n)]
    
    total = sum(random_numbers)
    
    while total < 0 or total >= upper_total:
        random_numbers = [random.uniform(0, upper_total) for _ in range(n)]
        total = sum(random_numbers)
    
    return sorted(random_numbers)


def init_pl_device(device_arn, n_qubits, shots, max_parallel):
    return qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        wires=n_qubits,
        shots=shots,
        # Set s3_destination_folder=None to output task results to a default folder
        s3_destination_folder=None,
        parallel=True,
        max_parallel=max_parallel,
        # poll_timeout_seconds=30,
    )

def linear_combination(inputs:str, weights, bias, ancilla, n_qubits):
    for qubit in range(len(inputs)):
        if(inputs[qubit]=='1'):
            qml.PauliX(qubit)
    
    for qubit in range(len(inputs)):
        
        qml.CRY(phi=2*weights[qubit], wires=[qubit, ancilla])
    
    qml.RY(2*bias, wires=ancilla)

def activation_function(inputs:str, weights, bias, ancilla, output, n_qubits):
    linear_combination(inputs, weights, bias, ancilla, n_qubits)
    
    qml.CY(wires=[ancilla, output])
    qml.RZ(phi=-np.pi/2, wires=ancilla)
    
    for qubit in range(len(inputs)):
        qml.CRY(phi=-2*weights[qubit], wires=[qubit, ancilla])  # note '-(minus)'
        
    qml.RY(-2*bias, wires=ancilla)  # note '-(minus)'

def quantum_neuron(inputs:str, weights, bias, n_qubits, dev):
    ancilla = len(inputs) # ID of an ancilla qubit
    output = len(inputs) + 1   # ID of an output qubit
    
    theta = np.inner(np.array(list(inputs), dtype=int), np.array(weights)) + bias   # linear comination with numpy
    theta = theta.item()   # Convert numpy array to native python float-type
    
    @qml.qnode(dev)
    def af_circuit():
        activation_function(inputs, weights, bias, ancilla, output, n_qubits)
        
        return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
    
    ### start of post-processing ###
    sample = af_circuit()
    sample = sample.T
    sample = (1 - sample.numpy()) / 2

    adopted_sample = sample[sample[:,ancilla] == 0]

    count_0 = len(adopted_sample[adopted_sample[:,output] == 0])
    count_1 = len(adopted_sample[adopted_sample[:,output] == 1])

    p_0 = count_0 / (count_0 + count_1)
    
    q_theta = np.arccos(np.sqrt(p_0))
    ### end of post-processing ###
    
    return theta, q_theta

def main():
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
    
    n_inputs = int(hyperparams["n_inputs"])
    shots = int(hyperparams["shots"])
    interface = hyperparams["interface"]
    max_parallel = int(hyperparams["max_parallel"])
    
    n_qubits = n_inputs+2 # +2: ancilla and output qubit
    
    inputs_list = [format(i, f'0{n_inputs}b') for i in range(2**n_inputs)]
    bias = 0.05  # constant
    weights = generate_random_numbers(n_inputs, np.pi/2-bias)
    
    if "copy_checkpoints_from_job" in hyperparams:
        copy_checkpoints_from_job = hyperparams["copy_checkpoints_from_job"].split("/", 2)[-1]
    else:
        copy_checkpoints_from_job = None
    
    # Run quantum neuron circuit
    dev = init_pl_device(device_arn, n_qubits, shots, max_parallel)
    theta_list = []
    q_theta_list = []
    
    for i in range(2**n_inputs):
        theta, q_theta = quantum_neuron(inputs_list[i], weights, bias, n_qubits, dev)
        
        theta_list.append(theta)
        q_theta_list.append(q_theta)
    
    save_job_result({"theta_list": theta_list, "q_theta_list": q_theta_list, "task summary": t.quantum_tasks_statistics(), "estimated cost": t.qpu_tasks_cost() + t.simulator_tasks_cost()})

if __name__ == "__main__":
    main()