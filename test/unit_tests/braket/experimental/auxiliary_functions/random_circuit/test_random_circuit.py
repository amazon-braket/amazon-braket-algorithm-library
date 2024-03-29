from braket.circuits import Circuit
from braket.circuits.gates import XY, CNot, CPhaseShift, H, Rx, Ry, Rz, S, T

from braket.experimental.auxiliary_functions import random_circuit


def test_random_circuit_returns_circuit():
    circuit = random_circuit(num_qubits=3, num_gates=5, seed=1)
    assert isinstance(circuit, Circuit)


def test_random_circuit_instruction_count():
    num_gates = 5
    circuit = random_circuit(num_qubits=3, num_gates=num_gates, seed=1)
    assert len(circuit.instructions) == num_gates


def test_random_circuit_consistency_with_seed():
    circuit1 = random_circuit(num_qubits=3, num_gates=5, seed=1)
    circuit2 = random_circuit(num_qubits=3, num_gates=5, seed=1)
    assert circuit1 == circuit2


def test_random_circuit_variability_without_seed():
    circuit1 = random_circuit(num_qubits=3, num_gates=5)
    circuit2 = random_circuit(num_qubits=3, num_gates=5)
    assert circuit1 != circuit2


def test_custom_gate_set():
    gate_set = [CNot, H, S, T, Rx, Ry, Rz, XY, CPhaseShift]
    circuit = random_circuit(num_qubits=3, num_gates=5, gate_set=gate_set, seed=1)

    gate_from_gate_set = []
    for instr in circuit.instructions:
        gate_class = instr.operator.__class__
        gate_from_gate_set.append(gate_class in gate_set)

    assert all(gate_from_gate_set)
