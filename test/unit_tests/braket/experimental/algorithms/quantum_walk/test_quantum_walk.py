import numpy as np
import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from quantum_walk import qft_conditional_add_1, quantum_walk


def test_quantum_walk_4_nodes_4_steps():
    local_simulator = LocalSimulator()
    quantum_walk_circuit = quantum_walk(4, 4)
    task = local_simulator.run(quantum_walk_circuit, shots=1000)
    result = task.result()
    counts = result.measurement_counts

    assert set(counts.keys()) == {"001"}

    assert np.isclose(list(counts.values()), [1000]).all()


def test_quantum_walk_4_nodes_1_step():
    local_simulator = LocalSimulator()
    quantum_walk_circuit = quantum_walk(4, 1)
    shots = 10000
    task = local_simulator.run(quantum_walk_circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts

    assert set(counts.keys()) == {"010", "111"}

    assert np.isclose(np.array(list(counts.values())) / shots, [0.5, 0.5], rtol=1e-02).all()


@pytest.mark.parametrize(
    "num_qubits, a", [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
)
def test_qft_conditional_add_1(num_qubits, a):
    qc = Circuit()
    local_simulator = LocalSimulator()

    # Step 0: Set the 0th qubit to 1 so that we are adding 1
    qc.x(0)
    # Step 1: Encode the number a into the circuit
    binary_a = bin(abs(a))[2:]
    for i, bit in enumerate(reversed(binary_a)):
        if bit == "1":
            qc.x(1 + i)

    # Step 2: Apply the adder and get the results
    qc.add_circuit(qft_conditional_add_1(num_qubits))
    result = local_simulator.run(qc, shots=1000).result()
    counts = result.measurement_counts

    # Step 3: Make sure the measurement outcome corresponds to the addition a+1
    values = np.asarray([int(v[::-1], 2) for v in counts.keys()])
    counts = np.asarray(list(counts.values()))
    c = values[np.argmax(counts)]

    assert np.mod(a + 1 - c, 2**num_qubits) == 0


@pytest.mark.parametrize(
    "num_qubits, a, action",
    [
        (3, 0, "+"),
        (3, 1, "+"),
        (3, 2, "+"),
        (3, 3, "+"),
        (3, 4, "+"),
        (3, 5, "+"),
        (3, 6, "+"),
        (3, 7, "+"),
        (3, 0, "-"),
        (3, 1, "-"),
        (3, 2, "-"),
        (3, 3, "-"),
        (3, 4, "-"),
        (3, 5, "-"),
        (3, 6, "-"),
        (3, 7, "-"),
    ],
)
def test_qft_conditional_minus_1(num_qubits, a, action):
    qc = Circuit()
    local_simulator = LocalSimulator()

    # Step 0: Do nothing to the 0th qubit to 1 so that we are subtracting 1
    if action == "-":
        qc.x(0)

    # Step 1: Encode the number a into the circuit
    binary_a = bin(abs(a))[2:]
    for i, bit in enumerate(reversed(binary_a)):
        if bit == "1":
            qc.x(1 + i)

    # Step 2: Add or subtract 1 conditionally
    qc.add_circuit(qft_conditional_add_1(num_qubits))
    result = local_simulator.run(qc, shots=1000).result()
    counts = result.measurement_counts

    # Step 3: Make sure the measurement outcome corresponds to the addition a+b
    values = np.asarray([int(v[1:][::-1], 2) for v in counts.keys()])
    counts = np.asarray(list(counts.values()))
    c = values[np.argmax(counts)]

    if action == "-":
        assert np.mod(a - 1 - c, 2**num_qubits) == 0
    else:  # action == "+"
        assert np.mod(a + 1 - c, 2**num_qubits) == 0
