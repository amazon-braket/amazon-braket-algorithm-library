import pennylane as qml
import numpy as np

from qn.quantum_neuron import (
    generate_random_numbers,
    linear_combination,
    quantum_neuron,
    activation_function
)

# WIP
'''
def test_linear_combination():
    for n_inputs in range(3, 7):
        n_qubits = n_inputs+2 # +2: ancilla and output qubit
        
        inputs_list = [format(i, f'0{n_inputs}b') for i in range(2**n_inputs)]
        bias = 0.05  # constant
        weights = generate_random_numbers(n_inputs, np.pi/2-bias)
        ancilla = len(weights) # ID of an ancilla qubit
        
        dev = qml.device("braket.local.qubit", wires=n_qubits, shots=100)

        @qml.qnode(dev)
        def lc_circuit(inputs, weights, bias, ancilla, n_qubits):
            linear_combination(inputs, weights, bias, ancilla, n_qubits)
            return qml.expval(qml.PauliZ(ancilla))
        
        output = lc_circuit(inputs_list[1], weights, bias)
        print(output)

        # expected_output = np.cos(weights[0]) * np.cos(weights[2]) * np.cos(bias) - np.sin(weights[0]) * np.sin(weights[2]) * np.cos(bias)

        # np.testing.assert_almost_equal(output, expected_output, decimal=6)
    
    
# def test_activation_function():

# def test_quantum_neuron():
'''
