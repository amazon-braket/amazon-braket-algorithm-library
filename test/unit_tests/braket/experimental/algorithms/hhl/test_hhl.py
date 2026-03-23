from unittest.mock import MagicMock

import numpy as np
import pytest

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.hhl import hhl as hhl_module


# Test with a simple diagonal 2x2 matrix
def test_hhl_diagonal_matrix():
    """Test HHL with a diagonal Hermitian matrix."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)

    assert circ is not None
    assert circ.qubit_count == 4  # 2 clock + 1 input + 1 ancilla


# Test with a symmetric 2x2 matrix
def test_hhl_symmetric_matrix():
    """Test HHL with a symmetric Hermitian matrix."""
    matrix = np.array([[2, 1], [1, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)

    assert circ is not None
    assert circ.qubit_count == 4


# Test validation: non-Hermitian matrix should raise error
def test_hhl_non_hermitian_raises():
    """Test that a non-Hermitian matrix raises ValueError."""
    matrix = np.array([[1, 2], [3, 4]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    with pytest.raises(ValueError, match="Hermitian"):
        hhl_module.hhl_circuit(matrix, b_vector)


# Test validation: wrong-sized matrix should raise error
def test_hhl_wrong_size_matrix_raises():
    """Test that a non-2x2 matrix raises ValueError."""
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
    b_vector = np.array([1, 0, 0], dtype=complex)

    with pytest.raises(ValueError, match="2x2"):
        hhl_module.hhl_circuit(matrix, b_vector)


# Test validation: unnormalized b_vector should raise error
def test_hhl_zero_b_vector_raises():
    """Test that a zero b_vector raises ValueError."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([0, 0], dtype=complex)

    with pytest.raises(ValueError, match="non-zero"):
        hhl_module.hhl_circuit(matrix, b_vector)


# Test state preparation
def test_state_preparation_basic():
    """Test that state preparation works for basic vectors."""
    from braket.circuits import Circuit

    circ = Circuit()
    b_vector = np.array([1, 0], dtype=float)
    circ = hhl_module._prepare_state_b(circ, 0, b_vector)
    assert circ is not None


# Test state preparation with superposition
def test_state_preparation_superposition():
    """Test state preparation for a superposition vector."""
    from braket.circuits import Circuit

    circ = Circuit()
    b_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=float)
    circ = hhl_module._prepare_state_b(circ, 0, b_vector)
    assert circ is not None


# Test Hamiltonian simulation
def test_hamiltonian_simulation():
    """Test that the Hamiltonian simulation produces a unitary matrix."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    t = 1.0
    unitary = hhl_module._compute_hamiltonian_simulation(matrix, t)

    # Check unitarity: U @ U† = I
    identity = unitary @ unitary.conj().T
    assert np.allclose(identity, np.eye(2), atol=1e-10)


# Test Hamiltonian simulation at t=0 gives identity
def test_hamiltonian_simulation_t0():
    """Test that e^{i*A*0} = I."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    unitary = hhl_module._compute_hamiltonian_simulation(matrix, 0)
    assert np.allclose(unitary, np.eye(2), atol=1e-10)


# Test eigendecomposition
def test_eigendecomposition():
    """Test eigendecomposition of a known matrix."""
    matrix = np.array([[2, 1], [1, 2]], dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Known eigenvalues for [[2,1],[1,2]] are 1 and 3
    assert np.allclose(sorted(eigenvalues), [1, 3], atol=1e-10)


# Test HHL circuit with identity matrix (trivial case)
def test_hhl_identity_matrix():
    """Test HHL with identity matrix: solution should be b itself."""
    matrix = np.array([[1, 0], [0, 1]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)

    assert circ is not None


# Test run_hhl function
def test_run_hhl():
    """Test running HHL on local simulator."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)
    device = LocalSimulator()
    task = hhl_module.run_hhl(circ, device, shots=100)

    result = task.result()
    assert result is not None
    assert result.measurement_counts is not None


# Test get_hhl_results function
def test_get_hhl_results():
    """Test post-processing of HHL results."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)
    device = LocalSimulator()
    task = hhl_module.run_hhl(circ, device, shots=1000)

    results = hhl_module.get_hhl_results(task, matrix, b_vector, num_clock_qubits=2, verbose=True)

    assert "measurement_counts" in results
    assert "post_selected_counts" in results
    assert "solution_state_probabilities" in results
    assert "classical_solution" in results
    assert "fidelity" in results
    assert "success_probability" in results

    # Classical solution for Ax=b with A=diag(1,2), b=[1,0] is x=[1,0]
    classical_sol = results["classical_solution"]
    assert np.allclose(classical_sol, [1, 0], atol=1e-10)


# Test validation of b_vector length
def test_hhl_b_vector_wrong_length_raises():
    """Test that a b_vector with wrong length raises ValueError."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0, 0], dtype=complex)

    with pytest.raises(ValueError):
        hhl_module.hhl_circuit(matrix, b_vector)


# Test with custom scaling factor
def test_hhl_custom_scaling():
    """Test HHL with a custom scaling factor."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2, scaling_factor=np.pi)
    assert circ is not None


# Test with a matrix having negative eigenvalues
def test_hhl_negative_eigenvalues():
    """Test HHL circuit creation for a matrix with negative eigenvalues."""
    # This matrix has eigenvalues -1 and 3
    matrix = np.array([[1, 2], [2, 1]], dtype=complex)

    # For now just test circuit construction
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)
    assert circ is not None


def test_prepare_state_b_complex_vector():
    """Test state preparation with complex amplitudes."""
    circ = Circuit()
    # Normalized complex vector: [1/sqrt(2), i/sqrt(2)]
    b_vector = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)
    circ = hhl_module._prepare_state_b(circ, 0, b_vector)

    # Check if Rz gate is applied (indicates complex path taken)
    assert any(instruction.operator.name == "Rz" for instruction in circ.instructions)


def test_prepare_state_b_negative_component():
    """Test state preparation with negative real component."""
    circ = Circuit()
    # Normalized vector with negative component: [0, -1]
    # This triggers the `if np.real(b_vector[1]) < 0` branch
    b_vector = np.array([0, -1], dtype=float)
    circ = hhl_module._prepare_state_b(circ, 0, b_vector)

    # We can check specific rotation angle if needed,
    # but mainly we care that it runs without error.
    assert circ is not None


def test_prepare_state_b_unnormalized_raises():
    """Test that unnormalized b_vector raises ValueError."""
    circ = Circuit()
    b_vector = np.array([1, 1], dtype=float)  # Norm is sqrt(2)

    with pytest.raises(ValueError, match="normalized"):
        hhl_module._prepare_state_b(circ, 0, b_vector)


def test_hhl_1_clock_qubit():
    """Test HHL with 1 clock qubit."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=1)

    # Check controlled rotation structure usage
    # With 1 clock qubit, it uses _add_controlled_ry
    # We can inspect the circuit instruction count or similar
    assert circ is not None
    # 1 clock + 1 input + 1 ancilla = 3 qubits
    assert circ.qubit_count == 3


def test_hhl_3_clock_qubits():
    """Test HHL with 3 clock qubits (triggering multi-controlled logic)."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=3)

    # With 3 clock qubits, it uses _add_multi_controlled_ry
    assert circ is not None
    # 3 clock + 1 input + 1 ancilla = 5 qubits
    assert circ.qubit_count == 5


def test_get_hhl_results_no_success():
    """Test get_hhl_results when no shots succeed (no post-selection)."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    # Mock task and result
    mock_task = MagicMock()
    mock_result = MagicMock()
    # All measurements have ancilla=0 (failure)
    # Format: clock(2) | input(1) | ancilla(1)
    # e.g., "00" + "0" + "0"
    mock_result.measurement_counts = {"0000": 100}
    mock_task.result.return_value = mock_result

    results = hhl_module.get_hhl_results(mock_task, matrix, b_vector, num_clock_qubits=2)

    assert results["success_shots"] == 0
    assert results["success_probability"] == 0.0
    # Should handle empty dict locally
    assert results["solution_state_probabilities"] == {"0": 0.0, "1": 0.0}


def test_get_hhl_results_multiple_post_selection():
    """Test get_hhl_results aggregating counts correctly."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    # Mock task and result
    mock_task = MagicMock()
    mock_result = MagicMock()
    # Success cases: ancilla=1, clock=00
    # "00" + "0" + "1" -> input 0
    # "00" + "1" + "1" -> input 1
    mock_result.measurement_counts = {
        "0001": 30,  # input 0, success
        "0011": 20,  # input 1, success
        "1100": 50,  # fail
    }
    mock_task.result.return_value = mock_result

    results = hhl_module.get_hhl_results(mock_task, matrix, b_vector, num_clock_qubits=2)

    assert results["success_shots"] == 50
    assert results["post_selected_counts"]["0"] == 30
    assert results["post_selected_counts"]["1"] == 20
    assert results["solution_state_probabilities"]["0"] == 0.6
    assert results["solution_state_probabilities"]["1"] == 0.4


def test_post_selected_counts_accumulation():
    """Test accumulation of post-selected counts for same input bit."""
    matrix = np.array([[1, 0], [0, 2]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)
    num_clock = 2

    mock_task = MagicMock()
    mock_result = MagicMock()
    # To hit line `post_selected_counts[input_bit] += count`,
    # we need multiple bitstrings mapping to same input_bit.
    # Standard format: clock(2) | input(1) | ancilla(1)
    # If we have extra "hidden" bits at the end, get_hhl_results relies on indices.
    # bitstring is opaque.
    # Ancilla is bitstring[-1].
    # Clock is bitstring[:num_clock].
    # Input is bitstring[num_clock].

    # Let's say we have bitstrings "0001A" and "0001B" where both pass.
    # This requires string length > 4.
    mock_result.measurement_counts = {
        "00010": 10,  # "00" clock, input "0", ancilla "0" (fail)
        "00011": 20,  # "00" clock, input "0", ancilla "1" (success)
        # We need another string that is "00" + "0" + ... + "1"
        # Since logic is by index, we can construct arbitrary string.
        # "00" (clock) + "0" (input) + "X" + "1" (ancilla)
        # Length 5. clock=0:2, input=2, ancilla=-1.
        "000X1": 30,  # "00" clock, input "0", ancilla "1"
    }
    mock_task.result.return_value = mock_result

    results = hhl_module.get_hhl_results(mock_task, matrix, b_vector, num_clock_qubits=num_clock)

    # Both "00011" and "000X1" map to input "0".
    # Total count for "0" should be 20 + 30 = 50.
    assert results["success_shots"] == 50
    assert results["post_selected_counts"]["0"] == 50


def test_controlled_rotation_zeros():
    """Test controlled rotation with zero eigenvalues."""
    # If eigenvalues are all "zero" (filtered by 1e-10)
    eigenvalues = np.array([0.0, 1e-11], dtype=float)

    # Should return empty circuit immediately (line 308)
    res_circ = hhl_module._controlled_rotation([0, 1], 2, eigenvalues, 2, 1.0)

    assert len(res_circ.instructions) == 0


def test_add_multi_controlled_ry_base_case():
    """Test _add_multi_controlled_ry with 1 control to hit base case."""
    circ = Circuit()
    controls = [0]
    target = 1
    theta = np.pi

    # Directly call the internal function
    hhl_module._add_multi_controlled_ry(circ, controls, target, theta)

    # Should produce instructions (uses built-in controlled Ry)
    assert len(circ.instructions) > 0


def test_hhl_small_ratio_skip():
    """Test skippng of small rotation angles (ratio < 1e-12)."""
    # Matrix with huge condition number: min=1e-8, max=1e8 -> ratio ~ 1e-16
    # Both eigenvalues > 1e-10 so c_value is valid.
    matrix = np.array([[1e-8, 0], [0, 1e8]], dtype=complex)
    b_vector = np.array([1, 0], dtype=complex)

    # This should trigger the continue statement in _controlled_rotation
    # for the max eigenvalue component (reconstructed ~ 1e8, c ~ 1e-8)
    circ = hhl_module.hhl_circuit(matrix, b_vector, num_clock_qubits=2)
    assert circ is not None


def test_decompose_controlled_unitary_wrong_shape():
    """Test that a non-2x2 unitary raises ValueError in decompose function."""
    unitary = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="Only 2x2 unitaries"):
        hhl_module._decompose_controlled_unitary(unitary, [0, 1])


def test_decompose_controlled_unitary_empty():
    """Test decompose_controlled_unitary with an identity unitary (results in 0 gates)."""
    unitary = np.eye(2, dtype=complex)
    circ = hhl_module._decompose_controlled_unitary(unitary, [0, 1])
    assert len(circ.instructions) == 0
