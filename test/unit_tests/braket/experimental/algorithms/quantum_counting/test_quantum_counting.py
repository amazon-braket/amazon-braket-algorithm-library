# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import math

from braket.devices import LocalSimulator

from braket.experimental.algorithms.grovers_search import build_oracle
from braket.experimental.algorithms.quantum_counting import (
    get_quantum_counting_results,
    quantum_counting_circuit,
    run_quantum_counting,
)


def test_single_solution():
    """Oracle '101' marks 1 of 8 states. Verify M_estimate is close to 1."""
    solution = "101"
    oracle = build_oracle(solution)
    n_search = len(solution)
    n_counting = 4

    circ = quantum_counting_circuit(oracle, n_search, n_counting)
    task = run_quantum_counting(circ, LocalSimulator(), shots=1000)
    results = get_quantum_counting_results(task, n_search, n_counting)

    assert abs(results["M_estimate"] - 1.0) < 1.5
    assert results["N_total"] == 8


def test_two_qubit_search():
    """Oracle '11' marks 1 of 4 states. Verify M_estimate is close to 1."""
    solution = "11"
    oracle = build_oracle(solution)
    n_search = len(solution)
    n_counting = 4

    circ = quantum_counting_circuit(oracle, n_search, n_counting)
    task = run_quantum_counting(circ, LocalSimulator(), shots=1000)
    results = get_quantum_counting_results(task, n_search, n_counting)

    assert abs(results["M_estimate"] - 1.0) < 1.5
    assert results["N_total"] == 4


def test_circuit_structure():
    """Verify the circuit has the expected qubit layout."""
    solution = "01"
    oracle = build_oracle(solution)
    n_search = 2
    n_counting = 3

    circ = quantum_counting_circuit(oracle, n_search, n_counting)

    # Circuit should use at least counting + search qubits
    assert circ.qubit_count >= n_counting + n_search


def test_results_dict_keys():
    """Verify output dict has expected keys."""
    solution = "11"
    oracle = build_oracle(solution)
    n_search = 2
    n_counting = 3

    circ = quantum_counting_circuit(oracle, n_search, n_counting)
    task = run_quantum_counting(circ, LocalSimulator(), shots=100)
    results = get_quantum_counting_results(task, n_search, n_counting)

    expected_keys = {"measurement_counts", "theta_estimate", "M_estimate", "N_total"}
    assert set(results.keys()) == expected_keys


def test_theta_in_valid_range():
    """Verify theta estimate is in [0, pi/2]."""
    solution = "101"
    oracle = build_oracle(solution)
    n_search = 3
    n_counting = 4

    circ = quantum_counting_circuit(oracle, n_search, n_counting)
    task = run_quantum_counting(circ, LocalSimulator(), shots=1000)
    results = get_quantum_counting_results(task, n_search, n_counting)

    assert 0 <= results["theta_estimate"] <= math.pi / 2


def test_m_estimate_in_valid_range():
    """Verify M_estimate is in [0, N]."""
    solution = "10"
    oracle = build_oracle(solution)
    n_search = 2
    n_counting = 4

    circ = quantum_counting_circuit(oracle, n_search, n_counting)
    task = run_quantum_counting(circ, LocalSimulator(), shots=1000)
    results = get_quantum_counting_results(task, n_search, n_counting)

    assert 0 <= results["M_estimate"] <= results["N_total"]


def test_decompose_ccnot():
    """Verify circuit works with decomposed Toffoli gates."""
    solution = "11"
    oracle = build_oracle(solution, decompose_ccnot=True)
    n_search = 2
    n_counting = 3

    circ = quantum_counting_circuit(oracle, n_search, n_counting, decompose_ccnot=True)
    task = run_quantum_counting(circ, LocalSimulator(), shots=1000)
    results = get_quantum_counting_results(task, n_search, n_counting)

    assert abs(results["M_estimate"] - 1.0) < 1.5
