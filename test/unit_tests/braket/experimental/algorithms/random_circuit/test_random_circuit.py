import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.random_circuit import (
    filter_gate_set,
    random_circuit,
    run_random_circuit,
)


def test_filter_gate_set_returns_classes():
    gate_classes = filter_gate_set(1)
    assert isinstance(gate_classes, list)
    assert all(isinstance(cls, type) for cls in gate_classes)


def test_filter_gate_set_max_qubits():
    max_qubits = 2
    gate_classes = filter_gate_set(max_qubits)
    assert all(cls.fixed_qubit_count() <= max_qubits for cls in gate_classes)


def test_random_circuit_returns_circuit():
    circuit = random_circuit(3, 5, 1)
    assert isinstance(circuit, Circuit)


def test_random_circuit_instruction_count():
    num_gates = 5
    circuit = random_circuit(3, num_gates, 1)
    assert len(circuit.instructions) == num_gates


def test_random_circuit_consistency_with_seed():
    circuit1 = random_circuit(3, 5, 1, seed=123)
    circuit2 = random_circuit(3, 5, 1, seed=123)
    assert circuit1 == circuit2


def test_random_circuit_variability_without_seed():
    circuit1 = random_circuit(3, 5, 1)
    circuit2 = random_circuit(3, 5, 1)
    assert circuit1 != circuit2


def test_get_random_circuit_results():
    local_simulator = LocalSimulator()
    circuit = random_circuit(2, 3, 1, seed=20)
    circuit.probability()

    result = run_random_circuit(circuit, local_simulator, shots=0).result()
    measured_probabilities = result.values[0].tolist()
    expected_probabilities = [0.5, 0.5, 0, 0]
    assert measured_probabilities == pytest.approx(
        expected_probabilities, rel=1e-5
    ), "The measured probabilities are not within the expected tolerance."
