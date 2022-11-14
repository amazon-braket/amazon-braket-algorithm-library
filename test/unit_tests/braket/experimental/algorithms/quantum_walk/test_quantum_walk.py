import numpy as np
import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_walk.quantum_walk import (
    qft_conditional_add_1,
    quantum_walk,
    run_quantum_walk,
)


def test_quantum_walk_4_nodes_4_steps():
    local_simulator = LocalSimulator()
    quantum_walk_circuit = quantum_walk(4, 4)
    task = local_simulator.run(quantum_walk_circuit, shots=1000)
    result = task.result()
    counts = result.measurement_counts

    assert set(counts.keys()) == {"001"}

    assert np.isclose(list(counts.values()), [1000]).all()


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
def test_qft_conditional_add_1(num_qubits, a, action):
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


def test_value_error_num_nodes():
    try:
        quantum_walk(3)
    except Exception as e:
        assert str(e) == ("The number of nodes has to be 2^n for integer n.")


def test_run_quantum_walk():
    local_sim = LocalSimulator()
    qc = quantum_walk(4, 2)
    out = run_quantum_walk(qc, local_sim)

    assert out["circuit"] == qc
