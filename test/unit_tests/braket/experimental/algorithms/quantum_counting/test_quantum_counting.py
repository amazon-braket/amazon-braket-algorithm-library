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

from unittest.mock import MagicMock

import numpy as np

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.quantum_counting import quantum_counting as qc

# Oracle / circuit construction tests


def test_oracle_circuit_single_marked():
    """Oracle circuit should flip the phase of the marked state."""
    oracle = qc.build_oracle_circuit(2, [3])
    assert len(oracle.instructions) > 0
    assert oracle.qubit_count >= 2


def test_oracle_circuit_multiple_marked():
    """Oracle with multiple marked states should compose individual oracles."""
    oracle = qc.build_oracle_circuit(2, [0, 2])
    oracle_single = qc.build_oracle_circuit(2, [0])
    assert len(oracle.instructions) > len(oracle_single.instructions)


def test_oracle_circuit_no_marked():
    """Oracle with no marked states should be an empty circuit."""
    oracle = qc.build_oracle_circuit(2, [])
    assert len(oracle.instructions) == 0


def test_grover_circuit_construction():
    """Grover circuit should combine oracle and diffusion."""
    grover = qc.build_grover_circuit(2, [1])
    assert len(grover.instructions) > 0
    assert grover.qubit_count >= 2


def test_grover_circuit_unitarity():
    """Grover circuit unitary should be unitary."""
    grover = qc.build_grover_circuit(2, [1])
    u = grover.to_unitary()
    product = u @ u.conj().T
    np.testing.assert_array_almost_equal(product, np.eye(len(u)), decimal=10)


# Circuit construction tests


def test_quantum_counting_circuit_construction():
    """Circuit should have expected gate structure."""
    counting_qubits = [0, 1]
    search_qubits = [2]
    marked_states = [0]

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    assert len(circ.instructions) > 0
    assert len(circ.result_types) > 0


def test_inverse_qft_for_counting_2_qubits():
    """Inverse QFT on 2 qubits should produce correct gate count."""
    qubits = [0, 1]
    qft_circ = qc.inverse_qft_for_counting(qubits)

    # 2-qubit inverse QFT: 1 SWAP + 1 CPHASE + 2 H = 4 instructions
    assert len(qft_circ.instructions) == 4


# End-to-end counting tests


def _expected_M(n_counting: int, M_true: int, N: int) -> float:
    """Compute the expected M estimate given finite counting-qubit precision.

    With n_counting qubits the QPE can only resolve phases in multiples of
    1/2^n_counting. This helper finds the closest representable phase to the
    true Grover angle and returns the corresponding M estimate, which is what
    an ideal (noiseless, infinite-shot) quantum counting run would produce.
    """
    if M_true == 0:
        return 0.0
    if M_true == N:
        return float(N)
    import math

    theta = math.asin(math.sqrt(M_true / N))  # Grover angle
    exact_phase = theta / math.pi
    # Closest representable phase with n_counting bits
    y = round(exact_phase * (2**n_counting))
    approx_phase = y / (2**n_counting)
    return N * (math.sin(math.pi * approx_phase) ** 2)


def test_count_1_of_4():
    """Quantum counting should estimate M ≈ 1 for 1 marked item out of 4.

    With 4 counting qubits and N=4, the discretization error gives
    M_expected ≈ 1.235 (from the closest phase 3/16 to the exact 1/6).
    We allow a tolerance of 0.5 around the discretized expected value.
    """
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [3]
    N = 2 ** len(search_qubits)
    M_exp = _expected_M(len(counting_qubits), len(marked_states), N)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=1000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert count_estimates["search_space_size"] == N
    assert abs(count_estimates["best_estimate"] - M_exp) < 0.5


def test_count_0_of_4():
    """Quantum counting should estimate M ≈ 0 when no items are marked.

    With no marked items, the Grover operator is the identity (up to global
    phase). QPE should measure phase 0, giving M = 0 exactly.
    """
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = []

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=1000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert abs(count_estimates["best_estimate"] - 0.0) < 0.01


def test_count_4_of_4():
    """Quantum counting should estimate M ≈ 4 when all items are marked.

    With all items marked, the oracle flips every amplitude equally, so
    QPE resolves the phase exactly and M = N.
    """
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [0, 1, 2, 3]
    N = 2 ** len(search_qubits)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=1000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert abs(count_estimates["best_estimate"] - float(N)) < 0.01


def test_count_3_of_4():
    """Quantum counting should estimate M ≈ 3 for 3 marked items out of 4.

    Complements the 1-of-4 test by verifying a non-trivial fraction (3/4).
    With 4 counting qubits, the discretization error is small enough that
    the estimate should be within 0.5 of the expected value.
    """
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [0, 1, 3]
    N = 2 ** len(search_qubits)
    M_exp = _expected_M(len(counting_qubits), len(marked_states), N)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=1000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert abs(count_estimates["best_estimate"] - M_exp) < 0.5


def test_count_2_of_8():
    """Quantum counting should estimate M ≈ 2 for 2 marked items out of 8.

    With 5 counting qubits and N=8, the discretization error is small.
    """
    counting_qubits = [0, 1, 2, 3, 4]
    search_qubits = [5, 6, 7]
    marked_states = [2, 5]
    N = 2 ** len(search_qubits)
    M_exp = _expected_M(len(counting_qubits), len(marked_states), N)

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=2000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert count_estimates["search_space_size"] == N
    assert abs(count_estimates["best_estimate"] - M_exp) < 0.5


def test_count_with_marked_initial_state():
    """QPE should estimate M correctly when search register starts in a marked state.

    Instead of the standard uniform superposition |s⟩ = H^⊗n|0⟩^⊗n, we
    prepare the search register directly in a marked state |β⟩ = |11⟩.
    Since |β⟩ is a superposition of both Grover eigenstates (with equal
    amplitude), QPE still resolves the correct phase and produces the same
    M estimate as the standard |s⟩ initialization.

    This validates that the algorithm works for initial states within the
    Grover eigenspace beyond just the uniform superposition.
    """
    counting_qubits = [0, 1, 2, 3]
    search_qubits = [4, 5]
    marked_states = [3]  # Mark |11⟩
    N = 2 ** len(search_qubits)
    M_exp = _expected_M(len(counting_qubits), len(marked_states), N)

    # Build custom QPE circuit with marked-state initialization
    circ = Circuit()

    # QPE: Hadamard on counting qubits
    circ.h(counting_qubits)

    # Prepare |β⟩ = |11⟩ (marked state) instead of uniform superposition
    circ.x(search_qubits)

    # Controlled-G^(2^k) using gate-level circuit
    for ii, qubit in enumerate(reversed(counting_qubits)):
        power = 2 ** ii
        circ.add_circuit(
            qc.controlled_grover_circuit(qubit, search_qubits, marked_states, power)
        )

    # Inverse QFT on counting qubits
    circ.add_circuit(qc.inverse_qft_for_counting(counting_qubits))

    # Measurement on counting register
    circ.probability(counting_qubits)

    device = LocalSimulator()
    task = device.run(circ, shots=1000)

    count_estimates = qc.get_quantum_counting_results(
        task, counting_qubits, search_qubits, verbose=True
    )

    assert count_estimates["best_estimate"] is not None
    assert abs(count_estimates["best_estimate"] - M_exp) < 0.5


def test_oracle_invalid_state_raises():
    """Building oracle with out-of-range state should raise ValueError."""
    try:
        qc.build_oracle_circuit(2, [4])  # 4 is out of range for 2 qubits (max=3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_counting_estimates_have_correct_keys():
    """Counting estimates dict should contain all expected keys."""
    counting_qubits = [0, 1, 2]
    search_qubits = [3]
    marked_states = [0]

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = device.run(circ, shots=100)

    count_estimates = qc.get_quantum_counting_results(task, counting_qubits, search_qubits)

    expected_keys = {
        "measurement_counts",
        "counting_register_results",
        "phases",
        "estimated_counts",
        "best_estimate",
        "search_space_size",
    }
    assert set(count_estimates.keys()) == expected_keys


def test_get_quantum_counting_results_empty_counts():
    """Test get_quantum_counting_results with empty measurement counts."""
    mock_task = MagicMock()
    mock_result = MagicMock()
    mock_result.measurement_counts = {}
    mock_task.result.return_value = mock_result

    counting_qubits = [0, 1]
    search_qubits = [2]

    count_estimates = qc.get_quantum_counting_results(mock_task, counting_qubits, search_qubits)

    assert count_estimates["measurement_counts"] == {}
    assert count_estimates["counting_register_results"] == {}
    assert count_estimates["phases"] == []
    assert count_estimates["estimated_counts"] == []
    assert count_estimates["best_estimate"] is None


def test_run_quantum_counting():
    """run_quantum_counting should run the circuit and return a task."""
    counting_qubits = [0, 1, 2]
    search_qubits = [3]
    marked_states = [0]

    circ = Circuit()
    circ = qc.quantum_counting_circuit(circ, counting_qubits, search_qubits, marked_states)

    device = LocalSimulator()
    task = qc.run_quantum_counting(circ, device, shots=100)

    count_estimates = qc.get_quantum_counting_results(task, counting_qubits, search_qubits)

    assert count_estimates["best_estimate"] is not None
    assert count_estimates["search_space_size"] == 2


def test_controlled_grover_circuit_construction():
    """Controlled Grover circuit should produce a non-empty circuit."""
    control = 0
    search_qubits = [1, 2]
    marked_states = [1]

    circ = qc.controlled_grover_circuit(control, search_qubits, marked_states)
    assert len(circ.instructions) > 0


def test_controlled_grover_circuit_power():
    """Controlled Grover circuit with power > 1 should have more gates."""
    control = 0
    search_qubits = [1, 2]
    marked_states = [1]

    circ_p1 = qc.controlled_grover_circuit(control, search_qubits, marked_states, power=1)
    circ_p2 = qc.controlled_grover_circuit(control, search_qubits, marked_states, power=2)
    assert len(circ_p2.instructions) > len(circ_p1.instructions)
