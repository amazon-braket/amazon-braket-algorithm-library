import numpy as np
import pytest
from braket.circuits import Circuit, ResultType, Qubit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.hadamard_test import hadamard_test_circuit


def test_hadamard_test_real():
    unitary = Circuit().h(0)
    ancilla = Qubit(0)
    
    test_circuit = hadamard_test_circuit(ancilla, unitary, component='real')
    test_circuit.measure(ancilla)
    
    device = LocalSimulator()
    task = device.run(test_circuit, shots=1000)
    
    probs = task.result().measurement_probabilities
    p_zero = probs.get('0', 0)
    real_part = 2 * p_zero - 1
    
    assert np.isclose(real_part, 1/np.sqrt(2), atol=0.1)


def test_hadamard_test_imaginary():
    unitary = Circuit().s(0)
    ancilla = Qubit(0)
    
    test_circuit = hadamard_test_circuit(ancilla, unitary, component='imaginary')
    test_circuit.measure(ancilla)
    
    device = LocalSimulator()
    task = device.run(test_circuit, shots=10000)

    probs = task.result().measurement_probabilities
    p_zero = probs.get('0', 0)
    imag_part = 2 * p_zero - 1

    assert np.isclose(imag_part, 0.0, atol=0.1)


def test_hadamard_test_identity():
    unitary = Circuit().i(0)
    ancilla = Qubit(0)
    
    real_circuit = hadamard_test_circuit(ancilla, unitary, component='real')
    imag_circuit = hadamard_test_circuit(ancilla, unitary, component='imaginary')
    real_circuit.measure(ancilla)
    imag_circuit.measure(ancilla)
    
    device = LocalSimulator()
    
    real_task = device.run(real_circuit, shots=1000)
    real_probs = real_task.result().measurement_probabilities
    p_zero_real = real_probs.get('0', 0)
    real_part = 2 * p_zero_real - 1
    
    imag_task = device.run(imag_circuit, shots=1000)
    imag_probs = imag_task.result().measurement_probabilities
    p_zero_imag = imag_probs.get('0', 0)
    imag_part = 2 * p_zero_imag - 1
    
    assert np.isclose(real_part, 1.0, atol=0.1)
    assert np.isclose(imag_part, 0.0, atol=0.1)

@pytest.mark.xfail(raises=ValueError)
def test_hadamard_test_invalid_component():
    unitary = Circuit().h(0)
    ancilla = Qubit(0)
    hadamard_test_circuit(ancilla, unitary, component='invalid')
