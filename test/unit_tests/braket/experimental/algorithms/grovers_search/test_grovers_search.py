import numpy as np
import pytest
from braket.devices import LocalSimulator

from braket.experimental.algorithms.grovers_search import build_oracle, grovers_search


@pytest.mark.parametrize("solution", ["00", "000", "0000", "00000"])
def test_grovers_search_solution(solution):
    n_qubits = len(solution)

    oracle = build_oracle(solution)
    circuit = grovers_search(oracle, n_qubits=n_qubits, n_reps=1)

    local_simulator = LocalSimulator()
    task = local_simulator.run(circuit, shots=1000)
    result = task.result()
    probabilities = result.values[0]

    assert np.argmax(probabilities) == 0


@pytest.mark.parametrize("decompose_ccnot", [True, False])
def test_grovers_search_solution_decompose_ccnot(decompose_ccnot):
    solution = "000"
    n_qubits = len(solution)

    oracle = build_oracle(solution, decompose_ccnot=decompose_ccnot)
    circuit = grovers_search(oracle, n_qubits=n_qubits, n_reps=1, decompose_ccnot=decompose_ccnot)

    local_simulator = LocalSimulator()
    task = local_simulator.run(circuit, shots=1000)
    result = task.result()
    probabilities = result.values[0]

    assert np.argmax(probabilities) == 0
