import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.auxilary_functions.random_circuit import random_circuit


def test_random_circuit_returns_circuit():
    circuit = random_circuit(num_qubits=3, num_gates=5, seed=1)
    assert isinstance(circuit, Circuit)


def test_random_circuit_instruction_count():
    circuit = random_circuit(num_qubits=3, num_gates=5, seed=1)
    assert len(circuit.instructions) == num_gates


def test_random_circuit_consistency_with_seed():
    circuit1 = random_circuit(num_qubits=3, num_gates=5, seed=1)
    circuit2 = random_circuit(num_qubits=3, num_gates=5, seed=1)
    assert circuit1 == circuit2


def test_random_circuit_variability_without_seed():
    circuit1 = random_circuit(num_qubits=3, num_gates=5)
    circuit2 = random_circuit(num_qubits=3, num_gates=5)
    assert circuit1 != circuit2
