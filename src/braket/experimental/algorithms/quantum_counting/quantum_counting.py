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

# Quantum Counting Algorithm: Amazon Braket Algorithm Library
#
# The Quantum Counting Algorithm (Brassard, Hoyer, Mosca, Tapp 1998) combines
# Grover's Oracle with Quantum Phase Estimation to estimate how many solutions
# exist for a search problem. Given an oracle O_f that marks M solutions out of
# N = 2^n possible inputs, the algorithm estimates M = N * sin^2(theta), where
# theta is obtained via phase estimation on the Grover operator G = D * O_f.

import math
from typing import Any, Dict, List

from braket.circuits import Circuit
from braket.devices import Device
from braket.tasks import QuantumTask

from braket.experimental.algorithms.quantum_fourier_transform.quantum_fourier_transform import iqft


def _controlled_z(circuit: Circuit, control: int, target: int) -> Circuit:
    """Apply a controlled-Z gate between control and target qubits."""
    circuit.h(target)
    circuit.cnot(control, target)
    circuit.h(target)
    return circuit


def _get_mcz_ancilla_count(n_qubits: int) -> int:
    """Return the number of ancilla qubits needed for MCZ on n_qubits.

    Args:
        n_qubits (int): Number of qubits in the MCZ gate.

    Returns:
        int: Number of ancilla qubits required.
    """
    if n_qubits <= 3:
        return 0
    from braket.experimental.algorithms.grovers_search.grovers_search import multi_control_z

    mcz_circ = multi_control_z(n_qubits, False)
    return mcz_circ.qubit_count - n_qubits


def _multi_controlled_z(
    circuit: Circuit,
    qubits: List[int],
    ancilla_qubits: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Apply a multi-controlled-Z gate across the given qubits.

    For 2 qubits: CZ gate (no ancilla needed).
    For 3 qubits: H + CCX + H decomposition (no ancilla needed).
    For 4+ qubits: uses Grover's multi_control_z with provided ancilla.

    Args:
        circuit (Circuit): Circuit to add the gate to.
        qubits (list): List of qubit indices participating in the MCZ.
        ancilla_qubits (list): Pre-allocated ancilla qubit indices (reused across calls).
        decompose_ccnot (bool): Whether to decompose Toffoli gates.

    Returns:
        Circuit: Circuit with multi-controlled-Z added.
    """
    n = len(qubits)
    if n < 2:
        return circuit

    if n == 2:
        return _controlled_z(circuit, qubits[0], qubits[1])

    if n == 3:
        circuit.h(qubits[2])
        if decompose_ccnot:
            circuit.ccnot_decomposed(qubits[0], qubits[1], qubits[2])
        else:
            circuit.ccnot(qubits[0], qubits[1], qubits[2])
        circuit.h(qubits[2])
        return circuit

    from braket.experimental.algorithms.grovers_search.grovers_search import multi_control_z

    mcz_circ = multi_control_z(n, decompose_ccnot)
    n_ancilla = mcz_circ.qubit_count - n
    target_mapping = {i: qubits[i] for i in range(n)}
    for i in range(n_ancilla):
        target_mapping[n + i] = ancilla_qubits[i]
    circuit.add_circuit(mcz_circ, target_mapping=target_mapping)
    return circuit


def _apply_controlled_oracle(
    circuit: Circuit,
    control_qubit: int,
    search_qubits: List[int],
    oracle: Circuit,
    ancilla_qubits: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Apply oracle phase flip controlled on the control qubit.

    Reconstructs the oracle pattern by analyzing X gates to determine
    which qubits are zero-marked, then applies MCZ including the control.

    Args:
        circuit (Circuit): Circuit to add gates to.
        control_qubit (int): Control qubit from counting register.
        search_qubits (list): Search register qubit indices.
        oracle (Circuit): The uncontrolled oracle circuit.
        ancilla_qubits (list): Pre-allocated ancilla qubit indices.
        decompose_ccnot (bool): Whether to decompose Toffoli gates.

    Returns:
        Circuit: Circuit with controlled oracle added.
    """
    n = len(search_qubits)

    # Analyze the oracle to find which qubits get X gates (zero-marked positions)
    x_qubits_in_oracle = set()
    for instruction in oracle.instructions:
        if instruction.operator.name == "X":
            target_idx = int(instruction.target[0])
            if target_idx < n:
                x_qubits_in_oracle.add(target_idx)

    # Map oracle's qubit indices to our search_qubits
    x_positions = [search_qubits[i] for i in range(n) if i in x_qubits_in_oracle]

    # Apply X gates on zero-marked qubits
    for q in x_positions:
        circuit.x(q)

    # MCZ on all search qubits + control qubit
    mcz_qubits = [control_qubit] + list(search_qubits)
    _multi_controlled_z(circuit, mcz_qubits, ancilla_qubits, decompose_ccnot)

    # Undo X gates
    for q in x_positions:
        circuit.x(q)

    return circuit


def _apply_controlled_diffusion(
    circuit: Circuit,
    control_qubit: int,
    search_qubits: List[int],
    ancilla_qubits: List[int],
    decompose_ccnot: bool = False,
) -> Circuit:
    """Apply the diffusion operator controlled on the control qubit.

    The controlled diffusion applies (I - 2|s><s|) when control=|1>
    and identity when control=|0>. It decomposes as:
        H^n -> X^n -> MCZ(control + search) -> X^n -> H^n

    Args:
        circuit (Circuit): Circuit to add gates to.
        control_qubit (int): Control qubit from counting register.
        search_qubits (list): Search register qubit indices.
        ancilla_qubits (list): Pre-allocated ancilla qubit indices.
        decompose_ccnot (bool): Whether to decompose Toffoli gates.

    Returns:
        Circuit: Circuit with controlled diffusion added.
    """
    for q in search_qubits:
        circuit.h(q)
    for q in search_qubits:
        circuit.x(q)

    mcz_qubits = [control_qubit] + list(search_qubits)
    _multi_controlled_z(circuit, mcz_qubits, ancilla_qubits, decompose_ccnot)

    for q in search_qubits:
        circuit.x(q)
    for q in search_qubits:
        circuit.h(q)

    return circuit


def quantum_counting_circuit(
    oracle: Circuit,
    n_search_qubits: int,
    n_counting_qubits: int,
    decompose_ccnot: bool = False,
) -> Circuit:
    """Build the full Quantum Counting circuit.

    Combines Grover's oracle with Quantum Phase Estimation to estimate the
    number of solutions M to a search problem. The Grover operator G has
    eigenvalues e^{+/- 2i*theta} where sin^2(theta) = M/N. QPE estimates
    theta, from which M can be computed.

    Qubit layout:
        - Counting qubits: [0, 1, ..., t-1]
        - Search qubits: [t, t+1, ..., t+n-1]
        - Ancilla qubits (if needed): [t+n, ...]

    Args:
        oracle (Circuit): Oracle circuit that marks solution states with a phase flip.
            Can be built using build_oracle from braket.experimental.algorithms.grovers_search.
        n_search_qubits (int): Number of search (data) qubits.
        n_counting_qubits (int): Number of counting (precision) qubits. More qubits
            give higher precision in estimating the number of solutions.
        decompose_ccnot (bool): Whether to decompose Toffoli gates into
            single-qubit and CNOT gates. Default False.

    Returns:
        Circuit: The quantum counting circuit.
    """
    t = n_counting_qubits
    n = n_search_qubits

    counting_qubits = list(range(t))
    search_qubits = list(range(t, t + n))

    # Pre-allocate fixed ancilla qubits for MCZ operations.
    # MCZ on (1 control + n search) qubits may need ancilla.
    mcz_size = 1 + n
    n_ancilla = _get_mcz_ancilla_count(mcz_size)
    ancilla_qubits = list(range(t + n, t + n + n_ancilla))

    circ = Circuit()

    # Step 1: Apply Hadamard to all counting and search qubits
    for q in counting_qubits:
        circ.h(q)
    for q in search_qubits:
        circ.h(q)

    # Step 2: Apply controlled-G^(2^k) for each counting qubit k
    # Counting qubit k (starting from the last) controls 2^k applications of G.
    for k, counting_qubit in enumerate(reversed(counting_qubits)):
        n_applications = 2**k
        for _ in range(n_applications):
            # Controlled oracle: O_f conditioned on counting qubit
            _apply_controlled_oracle(
                circ, counting_qubit, search_qubits, oracle,
                ancilla_qubits, decompose_ccnot,
            )
            # Controlled diffusion: (I - 2|s><s|) conditioned on counting qubit
            _apply_controlled_diffusion(
                circ, counting_qubit, search_qubits,
                ancilla_qubits, decompose_ccnot,
            )
            # Phase correction: the library's Grover operator G_lib = D*O has
            # eigenvalues -e^{±2iθ} (extra minus sign vs. the standard convention
            # G_std = (2|s><s|-I)*O with eigenvalues e^{±2iθ}). A Z gate on the
            # control qubit flips the sign of the |1> component, converting
            # controlled-G_lib into controlled-G_std so that QPE recovers the
            # standard phase θ/π directly.
            circ.z(counting_qubit)

    # Step 3: Apply inverse QFT on counting qubits
    circ.iqft(counting_qubits)

    return circ


def run_quantum_counting(
    circuit: Circuit,
    device: Device,
    shots: int = 1000,
) -> QuantumTask:
    """Submit the quantum counting circuit for execution.

    Args:
        circuit (Circuit): Quantum counting circuit (from quantum_counting_circuit).
        device (Device): Braket device backend (e.g., LocalSimulator()).
        shots (int): Number of measurement shots. Default 1000.

    Returns:
        QuantumTask: Task from running the circuit.
    """
    task = device.run(circuit, shots=shots)
    return task


def get_quantum_counting_results(
    task: QuantumTask,
    n_search_qubits: int,
    n_counting_qubits: int,
) -> Dict[str, Any]:
    """Extract and process quantum counting results.

    Converts measurement outcomes on the counting register to an estimate
    of M, the number of solutions. The most frequent measurement outcome y
    gives theta = y * pi / 2^t, and M = N * sin^2(theta).

    The algorithm produces two peaks at y and 2^t - y (corresponding to
    +theta and -theta). Both are handled, and the best estimate is returned.

    Args:
        task (QuantumTask): Completed quantum task.
        n_search_qubits (int): Number of search qubits (n). N = 2^n.
        n_counting_qubits (int): Number of counting qubits (t).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - measurement_counts: Raw measurement counts from the counting register.
            - theta_estimate: Estimated theta value.
            - M_estimate: Estimated number of solutions.
            - N_total: Total number of possible inputs (2^n).
    """
    result = task.result()
    measurement_counts = result.measurement_counts

    t = n_counting_qubits
    n = n_search_qubits
    N = 2**n

    # Aggregate counts on counting register only (first t qubits)
    counting_counts = {}
    for bitstring, count in measurement_counts.items():
        counting_bits = bitstring[:t]
        counting_counts[counting_bits] = counting_counts.get(counting_bits, 0) + count

    # Find the most frequent measurement outcome
    most_frequent = max(counting_counts, key=counting_counts.get)
    y = int(most_frequent, 2)

    # Compute theta from measurement outcome: theta = y * pi / 2^t
    if y == 0:
        theta = 0.0
        M_estimate = 0.0
    else:
        theta = y * math.pi / (2**t)
        M_estimate = N * math.sin(theta) ** 2

        # Handle the two-peak phenomenon: y and 2^t - y both appear.
        # Pick the estimate closer to an integer (M must be integer in exact cases).
        y_alt = 2**t - y
        theta_alt = y_alt * math.pi / (2**t)
        M_alt = N * math.sin(theta_alt) ** 2

        if abs(M_alt - round(M_alt)) < abs(M_estimate - round(M_estimate)):
            theta = theta_alt
            M_estimate = M_alt

    return {
        "measurement_counts": counting_counts,
        "theta_estimate": theta,
        "M_estimate": M_estimate,
        "N_total": N,
    }
