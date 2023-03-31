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

import pennylane as qml
import numpy as np

from qn.quantum_neuron import (
    generate_random_numbers,
    linear_combination,
    quantum_neuron,
    activation_function
)

def test_linear_combination():
    for n_inputs in range(3, 6):
        print()
        n_qubits = n_inputs+2 # +2: ancilla and output qubit
        bias = 0.05  # constant
        weights = generate_random_numbers(n_inputs, np.pi/2-bias)
        ancilla = len(weights) # ID of an ancilla qubit
        
        inputs_list = [format(i, f'0{n_inputs}b') for i in range(2**n_inputs)]
        input_to_test = 5   # I picked this at random.
        
        dev = qml.device("braket.local.qubit", wires=n_qubits, shots=100000)

        @qml.qnode(dev)
        def lc_circuit():
            linear_combination(inputs_list[input_to_test], weights, bias, ancilla, n_qubits)
            return qml.expval(qml.PauliZ(ancilla))
        
        print(qml.draw(lc_circuit, decimals=2)())
        
        z_expected_value = lc_circuit()
        print(f'z_expected_value: {z_expected_value}\n')

        theta = np.inner(np.array(list(inputs_list[input_to_test]), dtype=int), np.array(weights)) + bias   # linear comination with numpy
        theta = theta.item()   # Convert numpy array to native python float-type
        # print(f'theta: {theta}')
        # print(f'RY(theta): {qml.matrix(qml.RY(phi=theta, wires=0))}')
        # print(f'shape of RY(theta): {qml.matrix(qml.RY(phi=theta, wires=0)).shape}')
        
        theoritical_z_expected_value = np.cos(theta)**2 - np.sin(theta)**2  # Z expected value of Ry(2*theta)|0>
        print(f'theoritical_z_expected_value: {z_expected_value}')
        

        np.testing.assert_almost_equal(z_expected_value, theoritical_z_expected_value, decimal=2)
    
def test_quantum_neuron():
    for n_inputs in range(3, 6):
        print()
        n_qubits = n_inputs+2 # +2: ancilla and output qubit
        bias = 0.05  # constant
        weights = generate_random_numbers(n_inputs, np.pi/2-bias)

        inputs_list = [format(i, f'0{n_inputs}b') for i in range(2**n_inputs)]
        input_to_test = 5   # I picked this at random.
        
        dev = qml.device("braket.local.qubit", wires=n_qubits, shots=100000)
        
        theta, q_theta = quantum_neuron(inputs_list[input_to_test], weights, bias, n_qubits, dev)
        print(f'q_theta: {q_theta}')
       
        expected_q_theta = np.arctan(np.tan(theta)**2)
        print(f'expected_q_theta: {expected_q_theta}')
    
        np.testing.assert_almost_equal(q_theta, expected_q_theta, decimal=2)
        