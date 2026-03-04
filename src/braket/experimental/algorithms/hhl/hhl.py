"""Harrow-Hassidim-Lloyd (HHL) Algorithm for Solving Linear Systems of Equations.

The HHL algorithm is a quantum algorithm for solving systems of linear equations
of the form Ax = b. Given an N x N Hermitian matrix A and a unit vector b, the
algorithm produces a quantum state |x> proportional to A^{-1}|b>.

For certain classes of problems (sparse, well-conditioned matrices), and when only
summary statistics of the solution are needed (e.g., <x|M|x> for some operator M),
HHL can offer a speedup over classical methods. However, the overall advantage
depends on the efficiency of state preparation and readout.

This implementation provides a simplified version of HHL suitable for small systems
(2x2 matrices), illustrating the core concepts:
1. State preparation: encode |b> into a quantum state
2. Quantum Phase Estimation (QPE): decompose |b> in the eigenbasis of A
3. Controlled rotation: apply the eigenvalue inversion C/lambda
4. Inverse QPE: uncompute the eigenvalue register
5. Measurement: post-select on the ancilla qubit

References:
    [1] A. W. Harrow, A. Hassidim, S. Lloyd, "Quantum algorithm for linear systems
        of equations", Phys. Rev. Lett. 103, 150502 (2009). arXiv:0811.3171
    [2] Wikipedia: https://en.wikipedia.org/wiki/HHL_algorithm
"""

from typing import Any, Dict, List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_braket_provider.providers.adapter import to_braket

from braket.circuits import Circuit, circuit
from braket.circuits.qubit_set import QubitSetInput
from braket.devices import Device
from braket.experimental.algorithms.quantum_fourier_transform.quantum_fourier_transform import (
    iqft,
    qft,
)
from braket.tasks import QuantumTask


# Public API
@circuit.subroutine(register=True)
def hhl_circuit(
    matrix: np.ndarray,
    b_vector: np.ndarray,
    num_clock_qubits: int = 2,
    scaling_factor: Optional[float] = None,
) -> Circuit:
    """Construct the full HHL circuit for solving Ax = b.

    The circuit uses:
    - 1 input qubit for encoding |b>
    - num_clock_qubits clock qubits for QPE
    - 1 ancilla qubit for eigenvalue inversion (post-selection)

    Qubit layout:
    - Clock qubits: 0 to num_clock_qubits - 1
    - Input qubit: num_clock_qubits
    - Ancilla qubit: num_clock_qubits + 1

    Args:
        matrix (np.ndarray): A 2x2 Hermitian matrix A.
        b_vector (np.ndarray): A normalized 2-element vector b.
        num_clock_qubits (int): Number of clock qubits for QPE (default: 2).
        scaling_factor (Optional[float]): Scaling factor for Hamiltonian simulation.
            If None, automatically computed from the eigenvalues.

    Returns:
        Circuit: The complete HHL circuit.

    Raises:
        ValueError: If matrix is not 2x2 Hermitian or b_vector is invalid.
    """
    _validate_hermitian_2x2(matrix)

    # Normalize b_vector
    b_norm = np.linalg.norm(b_vector)
    if b_norm < 1e-10:
        raise ValueError("b_vector must be non-zero")
    b_normalized = b_vector / b_norm

    # Compute eigenvalues for scaling
    eigenvalues, _ = np.linalg.eigh(matrix)

    # Determine scaling factor if not provided
    if scaling_factor is None:
        # Map the full range of eigenvalues into the QPE register
        max_eigenval = max(abs(ev) for ev in eigenvalues)
        num_states = 2**num_clock_qubits
        # Scale so that eigenvalues map to distinct, well-separated QPE states
        # using the eigenvalue range to ensure all eigenvalues are representable
        scaling_factor = 2 * np.pi * (num_states - 1) / (max_eigenval * num_states)

    # Define qubit registers
    clock_qubits = list(range(num_clock_qubits))
    input_qubit = num_clock_qubits
    ancilla_qubit = num_clock_qubits + 1

    # Build the circuit
    circ = Circuit()

    # Step 1: State preparation - encode |b> on the input qubit
    circ = _prepare_state_b(circ, input_qubit, b_normalized)

    # Step 2: Quantum Phase Estimation
    qpe_circ = _qpe_for_hhl(clock_qubits, input_qubit, matrix, scaling_factor)
    circ.add(qpe_circ)

    # Step 3: Controlled rotation (eigenvalue inversion)
    rotation_circ = _controlled_rotation(
        clock_qubits, ancilla_qubit, eigenvalues, num_clock_qubits, scaling_factor
    )
    circ.add(rotation_circ)

    # Step 4: Inverse QPE (uncompute clock register)
    inv_qpe_circ = _inverse_qpe_for_hhl(clock_qubits, input_qubit, matrix, scaling_factor)
    circ.add(inv_qpe_circ)

    return circ


def run_hhl(
    circuit: Circuit,
    device: Device,
    shots: int = 1000,
) -> QuantumTask:
    """Run the HHL circuit on the specified device.

    Args:
        circuit (Circuit): The HHL circuit to run.
        device (Device): Braket device backend.
        shots (int): Number of measurement shots (default: 1000).

    Returns:
        QuantumTask: Task from running HHL.
    """
    task = device.run(circuit, shots=shots)
    return task


def get_hhl_results(
    task: QuantumTask,
    matrix: np.ndarray,
    b_vector: np.ndarray,
    num_clock_qubits: int = 2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Post-process results from an HHL run.

    Extracts the solution state by post-selecting on the ancilla qubit
    measuring |1>. The solution |x> is proportional to A^{-1}|b>.

    Args:
        task (QuantumTask): The task containing HHL results.
        matrix (np.ndarray): The original 2x2 matrix A.
        b_vector (np.ndarray): The original vector b.
        num_clock_qubits (int): Number of clock qubits used (default: 2).
        verbose (bool): If True, prints detailed results (default: False).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - measurement_counts: Raw measurement counts
            - post_selected_counts: Counts post-selected on ancilla=1
            - solution_state_probabilities: Probabilities of solution components
            - classical_solution: The exact classical solution for comparison
            - fidelity: Fidelity between quantum and classical solutions
            - success_probability: Probability of ancilla measuring |1>
    """
    result = task.result()
    measurement_counts = result.measurement_counts

    # Compute classical solution for comparison
    b_norm = np.linalg.norm(b_vector)
    b_normalized = b_vector / b_norm
    classical_solution = np.linalg.solve(matrix, b_normalized)
    classical_solution_normalized = classical_solution / np.linalg.norm(classical_solution)

    # Post-select: keep only measurements where ancilla (last qubit) = 1
    # and clock qubits are all 0 (indicating successful uncomputation)
    post_selected_counts = {}
    total_shots = sum(measurement_counts.values())
    success_shots = 0

    for bitstring, count in measurement_counts.items():
        # Bit ordering: clock_qubits | input_qubit | ancilla_qubit
        ancilla_bit = bitstring[-1]  # Last qubit is ancilla
        clock_bits = bitstring[:num_clock_qubits]

        if ancilla_bit == "1" and all(b == "0" for b in clock_bits):
            input_bit = bitstring[num_clock_qubits]
            if input_bit in post_selected_counts:
                post_selected_counts[input_bit] += count
            else:
                post_selected_counts[input_bit] = count
            success_shots += count

    # Compute solution state probabilities from post-selected counts
    solution_state_probs = {}
    if success_shots > 0:
        for state, count in post_selected_counts.items():
            solution_state_probs[state] = count / success_shots
    else:
        solution_state_probs = {"0": 0.0, "1": 0.0}

    success_probability = success_shots / total_shots if total_shots > 0 else 0.0

    # Compute fidelity between quantum result and classical solution
    quantum_probs = np.array(
        [
            solution_state_probs.get("0", 0.0),
            solution_state_probs.get("1", 0.0),
        ]
    )
    classical_probs = np.abs(classical_solution_normalized) ** 2

    # Fidelity F = (sum sqrt(p_i * q_i))^2
    fidelity = (np.sum(np.sqrt(quantum_probs * classical_probs))) ** 2

    aggregate_results = {
        "measurement_counts": measurement_counts,
        "post_selected_counts": post_selected_counts,
        "solution_state_probabilities": solution_state_probs,
        "classical_solution": classical_solution,
        "classical_solution_normalized": classical_solution_normalized,
        "classical_probabilities": classical_probs,
        "fidelity": fidelity,
        "success_probability": success_probability,
        "total_shots": total_shots,
        "success_shots": success_shots,
    }

    if verbose:
        print(f"Matrix A:\n{matrix}")
        print(f"\nVector b: {b_vector}")
        print(f"\nClassical solution x = A^(-1)b: {classical_solution}")
        print(f"Classical solution (normalized): {classical_solution_normalized}")
        print(f"Classical probabilities |x_i|^2: {classical_probs}")
        print(f"\nTotal measurement shots: {total_shots}")
        print(f"Post-selection success shots: {success_shots}")
        print(f"Success probability: {success_probability:.4f}")
        print(f"\nPost-selected counts: {post_selected_counts}")
        print(f"Quantum solution probabilities: {solution_state_probs}")
        print(f"\nFidelity with classical solution: {fidelity:.4f}")

    return aggregate_results


# Private helpers
def _validate_hermitian_2x2(matrix: np.ndarray) -> None:
    """Validate that the input is a 2x2 Hermitian matrix.

    Args:
        matrix (np.ndarray): The matrix to validate.

    Raises:
        ValueError: If the matrix is not 2x2 or not Hermitian.
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Matrix must be 2x2, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("Matrix must be Hermitian (A = A†)")


def _qpe_for_hhl(
    clock_qubits: QubitSetInput,
    input_qubit: int,
    matrix: np.ndarray,
    scaling_factor: float,
) -> Circuit:
    """Quantum Phase Estimation subroutine for HHL.

    Applies the QPE circuit to estimate eigenvalues of the Hermitian matrix A.
    Uses Hamiltonian simulation via e^{iAt} for a 2x2 system.

    Args:
        clock_qubits (QubitSetInput): Clock register qubits.
        input_qubit (int): The input qubit encoding |b>.
        matrix (np.ndarray): The 2x2 Hermitian matrix A.
        scaling_factor (float): Time parameter for Hamiltonian simulation.

    Returns:
        Circuit: QPE circuit.
    """
    circ = Circuit()
    num_clock = len(clock_qubits)

    # Apply Hadamard to clock qubits
    circ.h(clock_qubits)

    # Apply controlled-U^(2^k) operations
    # U = e^{iA * scaling_factor / num_states}
    # For clock qubit k, apply U^(2^k)
    for k, clock_qubit in enumerate(reversed(clock_qubits)):
        power = 2**k
        # Compute U^power = e^{i * A * scaling_factor * power / (2^num_clock)}
        t = scaling_factor * power / (2**num_clock)
        unitary = _compute_hamiltonian_simulation(matrix, t)

        # Decompose controlled unitary into 1q/2q gates and add to circuit
        cu_circ = _decompose_controlled_unitary(unitary, [clock_qubit, input_qubit])
        circ.add_circuit(cu_circ)

    # Apply inverse QFT to clock register using the library's iqft
    circ.add(iqft(clock_qubits))

    return circ


def _compute_hamiltonian_simulation(matrix: np.ndarray, t: float) -> np.ndarray:
    """Compute the unitary e^{iAt} for the Hamiltonian simulation.

    For a 2x2 Hermitian matrix, uses eigendecomposition for exact computation.

    Args:
        matrix (np.ndarray): The Hermitian matrix A.
        t (float): The time parameter.

    Returns:
        np.ndarray: The unitary matrix e^{iAt}.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # e^{iAt} = V * diag(e^{i*lambda_j*t}) * V†
    phases = np.exp(1j * eigenvalues * t)
    unitary = eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
    return unitary


def _decompose_controlled_unitary(
    unitary: np.ndarray,
    targets: List[int],
) -> Circuit:
    """Decompose a controlled-U gate into native 1q/2q gates via Qiskit transpilation.

    Constructs a controlled-U gate from a 2x2 unitary U, then uses
    Qiskit's transpiler to decompose it into U + CX gates, and converts the
    result to a Braket circuit mapped onto the specified target qubits.

    Args:
        unitary (np.ndarray): The 2x2 unitary matrix U.
        targets (List[int]): Target qubit indices [control, target].

    Returns:
        Circuit: Braket circuit implementing the controlled-U with native gates.
    """
    if unitary.shape != (2, 2):
        raise ValueError("Only 2x2 unitaries supported for explicit control construction")

    # Build a Qiskit circuit with the UnitaryGate controlled on 1 qubit
    cu_gate = UnitaryGate(unitary).control(1)
    qc = QuantumCircuit(2)
    qc.append(cu_gate, [0, 1])

    # Transpile into 1q/2q basis gates
    pm = generate_preset_pass_manager(
        optimization_level=2,
        basis_gates=["cx", "u", "id", "rz", "sx", "x"],
    )
    transpiled = pm.run(qc)

    # If transpilation optimized away all gates, return empty circuit
    if transpiled.size() == 0:
        return Circuit()

    # Determine which original Qiskit qubits are actually used in the
    # transpiled circuit, so we can map them to the correct target qubits.
    active_qiskit_indices = sorted(
        {transpiled.qubits.index(q) for instr in transpiled.data for q in instr.qubits}
    )

    # Convert transpiled Qiskit circuit to Braket circuit
    braket_circ = to_braket(transpiled)

    # Map each active Qiskit qubit index to the corresponding target qubit.
    # Qiskit qubit 0 -> targets[0] (control), qubit 1 -> targets[1] (target).
    # The Braket circuit's qubits are numbered 0..n-1 in the same order
    # as the active Qiskit qubits.
    target_mapping = [targets[i] for i in active_qiskit_indices]
    remapped = Circuit()
    remapped.add_circuit(braket_circ, target=target_mapping)
    return remapped


def _controlled_rotation(
    clock_qubits: QubitSetInput,
    ancilla_qubit: int,
    eigenvalues: np.ndarray,
    num_clock_qubits: int,
    scaling_factor: float,
) -> Circuit:
    """Apply controlled rotations to encode C/lambda into the ancilla qubit.

    For each eigenvalue lambda_j, performs a controlled-Ry rotation on the
    ancilla qubit conditioned on the clock register containing |lambda_j>.
    After rotation, the ancilla is in state:
        sqrt(1 - C^2/lambda_j^2)|0> + C/lambda_j|1>

    Measuring |1> on the ancilla post-selects the desired solution.

    Args:
        clock_qubits (QubitSetInput): Clock register qubits.
        ancilla_qubit (int): The ancilla qubit for post-selection.
        eigenvalues (np.ndarray): Eigenvalues of matrix A.
        num_clock_qubits (int): Number of clock qubits.
        scaling_factor (float): Scaling factor for eigenvalue encoding.

    Returns:
        Circuit: Circuit with controlled rotations.
    """
    circ = Circuit()
    num_states = 2**num_clock_qubits

    # Compute the constant C (normalization)
    abs_eigenvalues = np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10])
    if len(abs_eigenvalues) == 0:
        return circ
    c_value = np.min(abs_eigenvalues)

    # For each possible clock register state, apply a controlled rotation
    for clock_state in range(1, num_states):
        # Reconstruct the eigenvalue from the clock state
        reconstructed_eigenval = (2 * np.pi * clock_state) / (scaling_factor * num_states)

        # Compute rotation angle
        ratio = c_value / abs(reconstructed_eigenval)
        ratio = min(ratio, 1.0)
        theta = 2 * np.arcsin(ratio)

        if abs(theta) < 1e-12:
            continue

        # Convert clock_state to binary to determine which clock qubits are |0> vs |1>
        binary_rep = format(clock_state, f"0{num_clock_qubits}b")

        # Apply X gates to select the correct clock state
        for i, bit in enumerate(binary_rep):
            if bit == "0":
                circ.x(clock_qubits[i])

        # Apply multi-controlled Ry using Braket's built-in control mechanism
        _add_multi_controlled_ry(circ, list(clock_qubits), ancilla_qubit, theta)

        # Undo X gates
        for i, bit in enumerate(binary_rep):
            if bit == "0":
                circ.x(clock_qubits[i])

    return circ


def _add_multi_controlled_ry(circ: Circuit, controls: list, target: int, theta: float) -> None:
    """Add a multi-controlled Ry gate to the circuit.

    Uses Braket's built-in control mechanism for the Ry gate.

    Args:
        circ (Circuit): The circuit.
        controls (list): Control qubits.
        target (int): Target qubit.
        theta (float): Rotation angle.
    """
    if len(controls) == 1:
        # Use Braket's built-in controlled Ry
        circ.ry(target, theta, control=controls[0])
    else:
        # For multiple controls, decompose recursively:
        # C^n-Ry(theta) = C^(n-1)-Ry(theta/2) . CNOT . C^(n-1)-Ry(-theta/2) . CNOT . ...
        _add_multi_controlled_ry(circ, controls[1:], target, theta / 2)
        circ.cnot(controls[0], controls[-1])
        _add_multi_controlled_ry(circ, controls[1:], target, -theta / 2)
        circ.cnot(controls[0], controls[-1])
        _add_multi_controlled_ry(circ, controls[:-1], target, theta / 2)


def _inverse_qpe_for_hhl(
    clock_qubits: QubitSetInput,
    input_qubit: int,
    matrix: np.ndarray,
    scaling_factor: float,
) -> Circuit:
    """Inverse QPE subroutine to uncompute the clock register.

    Args:
        clock_qubits (QubitSetInput): Clock register qubits.
        input_qubit (int): The input qubit.
        matrix (np.ndarray): The 2x2 Hermitian matrix A.
        scaling_factor (float): Time parameter for Hamiltonian simulation.

    Returns:
        Circuit: Inverse QPE circuit.
    """
    circ = Circuit()
    num_clock = len(clock_qubits)

    # Apply forward QFT to clock register (inverse of inverse QFT)
    circ.add(qft(clock_qubits))

    # Apply inverse controlled-U^(2^k) operations (in reverse order)
    for k, clock_qubit in enumerate(reversed(clock_qubits)):
        power = 2**k
        t = scaling_factor * power / (2**num_clock)
        # Inverse unitary: (e^{iAt})† = e^{-iAt}
        unitary_inv = _compute_hamiltonian_simulation(matrix, -t)

        # Decompose inverse controlled unitary into 1q/2q gates
        cu_inv_circ = _decompose_controlled_unitary(unitary_inv, [clock_qubit, input_qubit])
        circ.add_circuit(cu_inv_circ)

    # Apply Hadamard to clock qubits
    circ.h(clock_qubits)

    return circ


def _prepare_state_b(circ: Circuit, input_qubit: int, b_vector: np.ndarray) -> Circuit:
    """Prepare the quantum state |b> on the input qubit.

    For a 2-element vector b = [b0, b1], prepares the state:
        |b> = b0|0> + b1|1>

    The vector must be normalized (|b0|^2 + |b1|^2 = 1).

    Args:
        circ (Circuit): The circuit to add state preparation to.
        input_qubit (int): The qubit to prepare the state on.
        b_vector (np.ndarray): The normalized 2-element vector b.

    Returns:
        Circuit: Circuit with state preparation.

    Raises:
        ValueError: If b_vector is not a normalized 2-element vector.
    """
    if len(b_vector) != 2:
        raise ValueError(f"b_vector must have 2 elements, got {len(b_vector)}")

    norm = np.linalg.norm(b_vector)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"b_vector must be normalized, got norm={norm}")

    # Compute the rotation angle to prepare |b> = cos(theta/2)|0> + sin(theta/2)|1>
    theta = 2 * np.arccos(np.clip(np.real(b_vector[0]), -1, 1))

    # Handle the phase if b_vector has complex components
    if np.isreal(b_vector).all():
        if np.real(b_vector[1]) < 0:
            theta = -theta
        circ.ry(input_qubit, theta)
    else:
        # General state preparation for complex amplitudes
        # |b> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
        phi = np.angle(b_vector[1]) - np.angle(b_vector[0])
        circ.ry(input_qubit, theta)
        circ.rz(input_qubit, phi)

    return circ
