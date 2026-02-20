"""Quantum Counting Algorithm implementation using the Amazon Braket SDK.

The quantum counting algorithm combines Grover's search operator with Quantum
Phase Estimation (QPE) to count the number of marked items in an unstructured
search space of N = 2^n elements.

Reference:
    G. Brassard, P. Høyer, and A. Tapp, "Quantum Counting",
    Proceedings of ICALP 1998. arXiv:quant-ph/9805082
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from braket.circuits import Circuit
from braket.circuits.qubit_set import QubitSetInput
from braket.experimental.algorithms.grovers_search.grovers_search import amplify, build_oracle


def build_oracle_circuit(
    n_qubits: int,
    marked_states: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Build an oracle circuit that flips the phase of marked states.

    Composes individual oracle circuits from grovers_search.build_oracle
    for each marked state. Each call flips the phase of one basis state,
    and their composition marks all specified states.

    Args:
        n_qubits (int): Number of qubits in the search register.
        marked_states (List[int]): Indices of marked computational-basis states.
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gates.

    Returns:
        Circuit: Oracle circuit that flips the phase of all marked states.

    Raises:
        ValueError: If a marked state index is out of range.
    """
    dim = 2**n_qubits
    for state in marked_states:
        if state < 0 or state >= dim:
            raise ValueError(
                f"Marked state {state} is out of range for {n_qubits} qubits "
                f"(must be in [0, {dim - 1}])."
            )

    oracle_circ = Circuit()
    for state in marked_states:
        bitstring = format(state, f"0{n_qubits}b")
        oracle_circ.add_circuit(build_oracle(bitstring, decompose_ccnot))
    return oracle_circ


def build_grover_circuit(
    n_qubits: int,
    marked_states: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Build the Grover operator G = D · O as a circuit.

    Uses circuit primitives from grovers_search: build_oracle for the phase
    oracle and amplify for the diffusion operator (H · oracle_0 · H).

    Note:
        The MCZ ancilla decomposition introduces a global phase of -1
        relative to the ideal Grover matrix.  This is accounted for
        in get_quantum_counting_results during phase correction.

    Args:
        n_qubits (int): Number of qubits in the search register.
        marked_states (List[int]): Indices of marked states.
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gates.

    Returns:
        Circuit: Circuit implementing the Grover operator G.
    """
    oracle_circ = build_oracle_circuit(n_qubits, marked_states, decompose_ccnot)
    diffusion_circ = amplify(n_qubits, decompose_ccnot)

    grover_circ = Circuit()
    grover_circ.add_circuit(oracle_circ)
    grover_circ.add_circuit(diffusion_circ)
    return grover_circ


def inverse_qft_for_counting(qubits: QubitSetInput) -> Circuit:
    """Inverse Quantum Fourier Transform applied to the given qubits.

    Args:
        qubits (QubitSetInput): Qubits on which to apply the inverse QFT.

    Returns:
        Circuit: Circuit implementing the inverse QFT.
    """
    qft_circ = Circuit()
    num_qubits = len(qubits)

    # SWAP gates to reverse qubit order
    for i in range(math.floor(num_qubits / 2)):
        qft_circ.swap(qubits[i], qubits[-i - 1])

    # Controlled phase rotations + Hadamard
    for k in reversed(range(num_qubits)):
        for j in reversed(range(1, num_qubits - k)):
            angle = -2 * math.pi / (2 ** (j + 1))
            qft_circ.cphaseshift(qubits[k + j], qubits[k], angle)
        qft_circ.h(qubits[k])

    return qft_circ


def controlled_grover(
    control: int, target_qubits: QubitSetInput, grover_unitary: np.ndarray
) -> Circuit:
    """Apply a controlled Grover operator.

    Applies the controlled-U gate where U is the Grover operator, with the
    given control qubit and target qubits.

    Args:
        control (int): Index of the control qubit.
        target_qubits (QubitSetInput): Indices of target (search and ancilla) qubits.
     """
    circ = Circuit()

    # Build controlled unitary: |0><0| ⊗ I + |1><1| ⊗ U
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    id_matrix = np.eye(len(grover_unitary))
    controlled_matrix = np.kron(p0, id_matrix) + np.kron(p1, grover_unitary)

    targets = [control] + list(target_qubits)
    circ.unitary(matrix=controlled_matrix, targets=targets)

    return circ


def quantum_counting_circuit(
    counting_circ: Circuit,
    counting_qubits: QubitSetInput,
    search_qubits: QubitSetInput,
    marked_states: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Create the full quantum counting circuit with result types.

    Builds the Grover operator as a circuit from grovers_search primitives
    (build_oracle + amplify), extracts its unitary, and applies QPE.

    Args:
        counting_circ (Circuit): Initial circuit (may contain setup operations).
        counting_qubits (QubitSetInput): Qubits for the counting (precision) register.
        search_qubits (QubitSetInput): Qubits for the search register.
        marked_states (List[int]): Indices of marked computational-basis states.
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gates.

    Returns:
        Circuit: The complete quantum counting circuit with result types.
    """
    counting_circ.add_circuit(
        quantum_counting(
            counting_qubits, search_qubits, marked_states, decompose_ccnot
        )
    )
    return counting_circ.probability(counting_qubits)


def quantum_counting(
    counting_qubits: QubitSetInput,
    search_qubits: QubitSetInput,
    marked_states: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Build the core quantum counting circuit using QPE on the Grover operator.

    Constructs the Grover operator as a gate-level circuit from grovers_search
    primitives (build_oracle for the phase oracle, amplify for the diffusion
    operator), extracts its unitary via Circuit.to_unitary(), and applies
    controlled-G^(2^k) for QPE.

    Note:
        The MCZ ancilla decomposition in grovers_search introduces a global
        phase of -1 on the Grover operator relative to the ideal matrix.
        This shifts QPE phase estimates by 0.5. The correction is applied
        in get_quantum_counting_results.

    The circuit structure:
      1. Apply H to all counting qubits
      2. Apply H to all search qubits (prepare uniform superposition |s>)
      3. Apply controlled-G^(2^k) for each counting qubit k
      4. Apply inverse QFT to counting qubits

    Args:
        counting_qubits (QubitSetInput): Qubits for the counting (precision) register.
        search_qubits (QubitSetInput): Qubits for the search register.
        marked_states (List[int]): Indices of marked computational-basis states.
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gates.

    Returns:
        Circuit: Circuit implementing the quantum counting algorithm.
    """
    n_search = len(search_qubits)

    # Build the Grover operator circuit from grovers_search primitives
    grover_circ = build_grover_circuit(n_search, marked_states, decompose_ccnot)
    grover_unitary = grover_circ.to_unitary()

    # Determine ancilla qubits introduced by the circuit decomposition
    n_ancilla = grover_circ.qubit_count - n_search
    ancilla_qubits = [
        max(list(counting_qubits) + list(search_qubits)) + 1 + i
        for i in range(n_ancilla)
    ]
    all_grover_qubits = list(search_qubits) + ancilla_qubits

    qc_circ = Circuit()

    # Hadamard on counting qubits
    qc_circ.h(counting_qubits)

    # Hadamard on search qubits (prepare |s>)
    qc_circ.h(search_qubits)

    # Controlled-G^(2^k)
    for ii, qubit in enumerate(reversed(counting_qubits)):
        power = 2**ii
        g_power = np.linalg.matrix_power(grover_unitary, power)
        qc_circ.add_circuit(controlled_grover(qubit, all_grover_qubits, g_power))

    # Inverse QFT on counting qubits
    qc_circ.add_circuit(inverse_qft_for_counting(counting_qubits))

    return qc_circ



def get_quantum_counting_results(
    task,
    counting_qubits: QubitSetInput,
    search_qubits: QubitSetInput,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Post-process quantum counting results to estimate the number of marked items.

    After measuring the counting qubits, the most likely outcome y gives
    an estimate of the phase φ = y / 2^t. The number of marked items M is:
        M = N · sin²(π · φ)
    where N = 2^n_search.

    Args:
        task (QuantumTask): The task holding quantum counting results.
        counting_qubits (QubitSetInput): Qubits of the counting register.
        search_qubits (QubitSetInput): Qubits of the search register.
        verbose (bool): If True, print detailed results (default False).

    Returns:
        Dict[str, Any]: Aggregate measurement results including:
            - measurement_counts: raw measurement counts
            - counting_register_results: counts collapsed to the counting register
            - phases: estimated phases φ for each measured bitstring
            - estimated_counts: estimated M for each measured bitstring
            - best_estimate: best estimate of M (from most frequent outcome)
            - search_space_size: N = 2^n_search
    """
    result = task.result()
    measurement_counts = result.measurement_counts
    n_counting = len(counting_qubits)
    n_search = len(search_qubits)
    N = 2**n_search

    # Aggregate results on counting register (trace out search qubits)
    counting_register_results: Dict[str, int] = {}
    if measurement_counts:
        for key in measurement_counts.keys():
            counting_bits = key[:n_counting]
            counting_register_results[counting_bits] = (
                counting_register_results.get(counting_bits, 0) + measurement_counts[key]
            )

    # Convert counting register bitstrings to phase estimates and M estimates
    # The MCZ ancilla decomposition introduces a global phase of -1 on the
    # Grover operator, shifting QPE phase estimates by 0.5. We correct by
    # computing: corrected_phase = |raw_phase - 0.5|
    phases, estimated_counts = _get_counting_estimates(counting_register_results, n_counting, N)

    # Best estimate from most frequent outcome
    if counting_register_results:
        best_key = max(counting_register_results, key=counting_register_results.get)
        best_y = int(best_key, 2)
        raw_phase = best_y / (2**n_counting)
        best_phase = abs(raw_phase - 0.5)
        best_M = N * (np.sin(np.pi * best_phase) ** 2)
    else:
        best_key = None
        best_M = None

    aggregate_results = {
        "measurement_counts": measurement_counts,
        "counting_register_results": counting_register_results,
        "phases": phases,
        "estimated_counts": estimated_counts,
        "best_estimate": best_M,
        "search_space_size": N,
    }

    if verbose:
        print(f"Search space size N = {N}")
        sorted_cr = sorted(
            counting_register_results.items(), key=lambda x: x[1], reverse=True
        )
        print("\nCounting register distribution (top outcomes):")
        for bitstring, count in sorted_cr[:6]:
            y_val = int(bitstring, 2)
            raw_ph = y_val / (2**n_counting)
            ph = abs(raw_ph - 0.5)
            m_est = N * (np.sin(np.pi * ph) ** 2)
            print(f"  |{bitstring}>: {count} counts  ->  phase = {ph:.4f},  M ~ {m_est:.4f}")
        if len(sorted_cr) > 6:
            print(f"  ... ({len(sorted_cr) - 6} more outcomes)")
        print(f"\nBest estimate of M: {best_M}")

    return aggregate_results


def _get_counting_estimates(
    counting_register_results: Dict[str, int],
    n_counting: int,
    N: int,
) -> Tuple[List[float], List[float]]:
    """Convert counting register bitstrings to phase and count estimates.

    Args:
        counting_register_results (Dict[str, int]): Measurement results on the counting register.
        n_counting (int): Number of counting qubits.
        N (int): Search space size.

    Returns:
        Tuple[List[float], List[float]]: (phases, estimated_counts) lists, one entry per unique
            measured bitstring with counts > 0.
    """
    phases = []
    estimated_counts = []

    for bitstring in counting_register_results:
        y = int(bitstring, 2)
        raw_phase = y / (2**n_counting)
        # Correct for -1 global phase from MCZ decomposition
        phase = abs(raw_phase - 0.5)
        M_est = N * (np.sin(np.pi * phase) ** 2)
        phases.append(phase)
        estimated_counts.append(M_est)

    return phases, estimated_counts
