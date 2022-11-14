import numpy as np
from braket.devices import LocalSimulator

from braket.experimental.algorithms.grover_search.grover_search import build_oracle, grover_search


def test_grover_search():
    solution = "000"
    n_qubits = len(solution)

    oracle = build_oracle(solution)
    circuit = grover_search(oracle, n_qubits=n_qubits, n_reps=1)

    local_simulator = LocalSimulator()
    task = local_simulator.run(circuit, shots=1000)
    result = task.result()
    probabilities = result.values[0]

    assert np.argmax(probabilities) == 0
