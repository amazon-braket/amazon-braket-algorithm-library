import numpy as np
import pytest
from braket.circuits import Circuit, ResultType
from braket.devices import LocalSimulator

from braket.experimental.algorithms.hadamard_test.hadamard_test import hadamard_test


def test_hadamard_test_real():
    unitary = Circuit().h(0)
    
    test_circuit = hadamard_test(unitary, phase='real')
    test_circuit.measure(0)
    
    device = LocalSimulator()
    task = device.run(test_circuit, shots=1000)
    
    counts = task.result().measurement_counts
    p_zero = counts.get('0', 0) / 1000
    real_part = 2 * p_zero - 1
    
    assert np.isclose(real_part, 1/np.sqrt(2), atol=0.1)


def test_hadamard_test_imaginary():
    unitary = Circuit().s(0)
    
    test_circuit = hadamard_test(unitary, phase='imaginary')
    test_circuit.measure(0)
    
    device = LocalSimulator()
    task = device.run(test_circuit, shots=10000)
    
    counts = task.result().measurement_counts
    p_zero = counts.get('0', 0) / 10000
    imag_part = 2 * p_zero - 1
    
    assert np.isclose(imag_part, 0.0, atol=0.1)


def test_hadamard_test_identity():
    unitary = Circuit().i(0)
    
    real_circuit = hadamard_test(unitary, phase='real')
    imag_circuit = hadamard_test(unitary, phase='imaginary')
    real_circuit.measure(0)
    imag_circuit.measure(0)
    
    device = LocalSimulator()
    
    real_task = device.run(real_circuit, shots=1000)
    real_counts = real_task.result().measurement_counts
    p_zero_real = real_counts.get('0', 0) / 1000
    real_part = 2 * p_zero_real - 1
    
    imag_task = device.run(imag_circuit, shots=1000)
    imag_counts = imag_task.result().measurement_counts
    p_zero_imag = imag_counts.get('0', 0) / 1000
    imag_part = 2 * p_zero_imag - 1
    
    assert np.isclose(real_part, 1.0, atol=0.1)
    assert np.isclose(imag_part, 0.0, atol=0.1)


@pytest.mark.parametrize("phase", ["real", "imaginary"])
def test_hadamard_test_shots_0(phase):
    unitary = Circuit().h(0)
    test_circuit = hadamard_test(unitary, phase=phase)
    test_circuit.add_result_type(ResultType.Probability(target=[0]))
    
    device = LocalSimulator()
    task = device.run(test_circuit, shots=0)
    
    p_zero = task.result().values[0][0]
    
    if phase == 'real':
        assert np.isclose(2 * p_zero - 1, 1/np.sqrt(2), atol=0.1)
    else:
        assert np.isclose(2 * p_zero - 1, 0.0, atol=0.1)


@pytest.mark.xfail(raises=ValueError)
def test_hadamard_test_invalid_phase():
    unitary = Circuit().h(0)
    hadamard_test(unitary, phase='invalid')
