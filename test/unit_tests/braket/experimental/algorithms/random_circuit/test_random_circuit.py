import pytest

from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.random_circuit import get_filtered_gates, random_circuit, get_random_circuit_results, run_random_circuit


def test_get_filtered_gates_returns_classes():
    gate_classes = get_filtered_gates(1)
    assert isinstance(gate_classes, list)
    assert all(isinstance(cls, type) for cls in gate_classes)


def test_get_filtered_gates_min_qubits():
    min_qubits = 2
    gate_classes = get_filtered_gates(min_qubits)
    assert all(cls.fixed_qubit_count() >= min_qubits for cls in gate_classes)


def test_random_circuit_returns_circuit():
    circuit = random_circuit(3, 5, 1)
    assert isinstance(circuit, Circuit)


def test_random_circuit_instruction_count():
    num_instructions = 5
    circuit = random_circuit(3, num_instructions, 1)
    assert len(circuit.instructions) == num_instructions


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
    r_circuit = random_circuit(2, 3, 1, seed=20)
    task = run_random_circuit(r_circuit, local_simulator, shots=1000)
    random_result = get_random_circuit_results(task)

    expected_results = {'00': 0.958, '10': 0.042}

    for key in expected_results:
        assert key in random_result, f"The key {key} is not  in results"
        assert random_result[key] == pytest.approx(expected_results[key], rel=0.05), f"The result for {key} is not within the expected tolerance."
