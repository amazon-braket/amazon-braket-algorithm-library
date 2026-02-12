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

from braket.circuits import Circuit, circuit
from braket.circuits.qubit_set import QubitSetInput
from braket.devices import Device
from braket.tasks import QuantumTask



def build_oracle_matrix(n_qubits: int, marked_states: List[int]) -> np.ndarray:
    """Build a diagonal oracle matrix that flips the phase of marked states.

    The oracle acts as O|x> = -|x> for marked x, and O|x> = |x> otherwise.

    Args:
        n_qubits (int): Number of qubits in the search register.
        marked_states (List[int]): Indices of marked computational-basis states.

    Returns:
        np.ndarray: The 2^n × 2^n diagonal oracle matrix.

    Raises:
        ValueError: If a marked state index is out of range.
    """
    dim = 2**n_qubits
    oracle = np.eye(dim)
    for state in marked_states:
        if state < 0 or state >= dim:
            raise ValueError(
                f"Marked state {state} is out of range for {n_qubits} qubits "
                f"(must be in [0, {dim - 1}])."
            )
        oracle[state, state] = -1
    return oracle


def build_diffusion_matrix(n_qubits: int) -> np.ndarray:
    """Build the Grover diffusion matrix D = 2|s><s| - I.

    |s> = H^{⊗n}|0>^{⊗n} is the uniform superposition state.

    Args:
        n_qubits (int): Number of qubits in the search register.

    Returns:
        np.ndarray: The 2^n × 2^n diffusion matrix.
    """
    dim = 2**n_qubits
    s = np.ones(dim) / np.sqrt(dim)
    return 2 * np.outer(s, s) - np.eye(dim)


def build_grover_matrix(n_qubits: int, marked_states: List[int]) -> np.ndarray:
    """Build the Grover operator G = D · O.

    Args:
        n_qubits (int): Number of qubits in the search register.
        marked_states (List[int]): Indices of marked states.

    Returns:
        np.ndarray: The 2^n × 2^n Grover operator matrix.
    """
    oracle = build_oracle_matrix(n_qubits, marked_states)
    diffusion = build_diffusion_matrix(n_qubits)
    return diffusion @ oracle


@circuit.subroutine(register=True)
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


@circuit.subroutine(register=True)
def controlled_grover(
    control: int, target_qubits: QubitSetInput, grover_unitary: np.ndarray
) -> Circuit:
    """Apply a controlled Grover operator.

    Applies the controlled-U gate where U is the Grover operator, with the
    given control qubit and target qubits.

    Args:
        control (int): Index of the control qubit.
        target_qubits (QubitSetInput): Indices of target (search) qubits.
        grover_unitary (np.ndarray): The Grover operator matrix.

    Returns:
        Circuit: Circuit implementing the controlled Grover operator.
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
    grover_matrix: np.ndarray,
) -> Circuit:
    """Create the full quantum counting circuit with result types.

    Builds the quantum counting circuit comprising:
      1. Hadamard gates on all counting and search qubits
      2. Controlled-G^(2^k) for each counting qubit k
      3. Inverse QFT on the counting qubits
      4. Probability result type on all qubits

    Args:
        counting_circ (Circuit): Initial circuit (may contain setup operations).
        counting_qubits (QubitSetInput): Qubits for the counting (precision) register.
        search_qubits (QubitSetInput): Qubits for the search register.
        grover_matrix (np.ndarray): The Grover operator matrix G.

    Returns:
        Circuit: The complete quantum counting circuit with result types.
    """
    return counting_circ.quantum_counting(
        counting_qubits, search_qubits, grover_matrix
    ).probability()


@circuit.subroutine(register=True)
def quantum_counting(
    counting_qubits: QubitSetInput,
    search_qubits: QubitSetInput,
    grover_matrix: np.ndarray,
) -> Circuit:
    """Build the core quantum counting circuit using QPE on the Grover operator.

    The circuit structure:
      1. Apply H to all counting qubits
      2. Apply H to all search qubits (prepare uniform superposition |s>)
      3. Apply controlled-G^(2^k) for each counting qubit k
      4. Apply inverse QFT to counting qubits

    Args:
        counting_qubits (QubitSetInput): Qubits for the counting (precision) register.
        search_qubits (QubitSetInput): Qubits for the search register.
        grover_matrix (np.ndarray): The Grover operator matrix G.

    Returns:
        Circuit: Circuit implementing the quantum counting algorithm.
    """
    qc_circ = Circuit()

    # Hadamard on counting qubits
    qc_circ.h(counting_qubits)

    # Hadamard on search qubits (prepare |s>)
    qc_circ.h(search_qubits)

    # Controlled-G^(2^k)
    n_counting = len(counting_qubits)
    for ii, qubit in enumerate(reversed(counting_qubits)):
        power = 2**ii
        g_power = np.linalg.matrix_power(grover_matrix, power)
        qc_circ.controlled_grover(qubit, search_qubits, g_power)

    # Inverse QFT on counting qubits
    qc_circ.inverse_qft_for_counting(counting_qubits)

    return qc_circ


def run_quantum_counting(
    circuit: Circuit,
    device: Device,
    shots: int = 1000,
) -> QuantumTask:
    """Run the quantum counting circuit on the given device.

    Args:
        circuit (Circuit): The quantum counting circuit.
        device (Device): Braket device backend.
        shots (int): Number of measurement shots (default 1000).

    Returns:
        QuantumTask: Task from running the quantum counting circuit.
    """
    return device.run(circuit, shots=shots)


def get_quantum_counting_results(
    task: QuantumTask,
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
                counting_register_results.get(counting_bits, 0)
                + measurement_counts[key]
            )

    # Convert counting register bitstrings to phase estimates and M estimates
    phases, estimated_counts = _get_counting_estimates(counting_register_results, n_counting, N)

    # Best estimate from most frequent outcome
    if counting_register_results:
        best_key = max(counting_register_results, key=counting_register_results.get)
        best_y = int(best_key, 2)
        best_phase = best_y / (2**n_counting)
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
        print(f"Measurement counts: {measurement_counts}")
        print(f"Counting register results: {counting_register_results}")
        print(f"Phase estimates: {phases}")
        print(f"Estimated item counts: {estimated_counts}")
        print(f"Best estimate of M: {best_M}")
        print(f"Search space size N: {N}")

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
        phase = y / (2**n_counting)
        M_est = N * (np.sin(np.pi * phase) ** 2)
        phases.append(phase)
        estimated_counts.append(M_est)

    return phases, estimated_counts
