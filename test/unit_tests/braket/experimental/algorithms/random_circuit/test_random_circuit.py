from braket.circuits import Circuit

from braket.experimental.algorithms.random_circuit import get_filtered_gates, random_circuit


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
