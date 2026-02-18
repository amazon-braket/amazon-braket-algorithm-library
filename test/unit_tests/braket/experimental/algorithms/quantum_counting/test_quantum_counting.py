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

import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_counting import quantum_counting as qc


# ============================================================
# Oracle / matrix construction tests
# ============================================================


def test_oracle_matrix_single_marked():
    """Oracle should flip the diagonal entry of the marked state to -1."""
    oracle = qc.build_oracle_matrix(2, [3])
    expected = np.diag([1, 1, 1, -1])
    np.testing.assert_array_equal(oracle, expected)


def test_oracle_matrix_multiple_marked():
    """Oracle with multiple marked states should flip all corresponding entries."""
    oracle = qc.build_oracle_matrix(2, [0, 2])
    expected = np.diag([-1, 1, -1, 1])
    np.testing.assert_array_equal(oracle, expected)


def test_oracle_matrix_no_marked():
    """Oracle with no marked states should be identity."""
    oracle = qc.build_oracle_matrix(2, [])
    expected = np.eye(4)
    np.testing.assert_array_equal(oracle, expected)


def test_diffusion_matrix():
    """Diffusion matrix for 1-qubit should be [[0, 1], [1, 0]] (X gate)."""
    diffusion = qc.build_diffusion_matrix(1)
    expected = np.array([[0, 1], [1, 0]])
    np.testing.assert_array_almost_equal(diffusion, expected)


def test_grover_matrix_unitarity():
    """Grover matrix should be unitary."""
    grover = qc.build_grover_matrix(2, [1])
    product = grover @ grover.T
    np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)


# ============================================================
# Circuit construction tests
# ============================================================


def test_quantum_counting_circuit_construction():
    """Circuit should have expected gate structure."""
    counting_qubits = [0, 1]
    search_qubits = [2]
    marked_states = [0]
    grover = qc.build_grover_matrix(1, marked_states)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    # The circuit should have instructions (Hadamard + controlled unitaries + QFT)
    assert len(circ.instructions) > 0
    # Should have result types (probability)
    assert len(circ.result_types) > 0


def test_inverse_qft_for_counting_2_qubits():
    """Inverse QFT on 2 qubits should produce correct gate count."""
    qubits = [0, 1]
    qft_circ = qc.inverse_qft_for_counting(qubits)

    # 2-qubit inverse QFT: 1 SWAP + 1 CPHASE + 2 H = 4 instructions
    assert len(qft_circ.instructions) == 4


# ============================================================
# End-to-end counting tests
# ============================================================


def test_count_1_of_4():
    """Quantum counting should estimate M ≈ 1 for 1 marked item out of 4."""
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [3]
    grover = qc.build_grover_matrix(2, marked_states)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=1000)

    results = qc.get_quantum_counting_results(task, counting_qubits, search_qubits, verbose=True)

    # Best estimate should be close to 1
    assert results["best_estimate"] is not None
    assert results["search_space_size"] == 4
    assert abs(results["best_estimate"] - 1.0) < 1.5


def test_count_0_of_4():
    """Quantum counting should estimate M ≈ 0 when no items are marked."""
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = []
    grover = qc.build_grover_matrix(2, marked_states)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=1000)

    results = qc.get_quantum_counting_results(task, counting_qubits, search_qubits, verbose=True)

    # With no marked items, the most common measurement should yield M ≈ 0
    assert results["best_estimate"] is not None
    assert abs(results["best_estimate"] - 0.0) < 0.5


def test_count_4_of_4():
    """Quantum counting should estimate M ≈ 4 when all items are marked."""
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [0, 1, 2, 3]
    grover = qc.build_grover_matrix(2, marked_states)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=1000)

    results = qc.get_quantum_counting_results(task, counting_qubits, search_qubits, verbose=True)

    # With all items marked, M should be ≈ 4
    assert results["best_estimate"] is not None
    assert abs(results["best_estimate"] - 4.0) < 0.5


def test_count_2_of_8():
    """Quantum counting should estimate M ≈ 2 for 2 marked items out of 8."""
    counting_qubits = [0, 1, 2, 3, 4]
    search_qubits = [5, 6, 7]
    marked_states = [2, 5]
    grover = qc.build_grover_matrix(3, marked_states)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=2000)

    results = qc.get_quantum_counting_results(task, counting_qubits, search_qubits, verbose=True)

    # Best estimate should be close to 2
    assert results["best_estimate"] is not None
    assert results["search_space_size"] == 8
    assert abs(results["best_estimate"] - 2.0) < 1.5


def test_oracle_invalid_state_raises():
    """Building oracle with out-of-range state should raise ValueError."""
    try:
        qc.build_oracle_matrix(2, [4])  # 4 is out of range for 2 qubits (max=3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_counting_results_have_correct_keys():
    """Results dict should contain all expected keys."""
    counting_qubits = [0, 1, 2]
    search_qubits = [3]
    grover = qc.build_grover_matrix(1, [0])

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, grover)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=100)

    results = qc.get_quantum_counting_results(task, counting_qubits, search_qubits)

    expected_keys = {
        "measurement_counts",
        "counting_register_results",
        "phases",
        "estimated_counts",
        "best_estimate",
        "search_space_size",
    }
    assert set(results.keys()) == expected_keys
