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
import pytest
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
    eigenvalues, eigenvectors = hhl_module._compute_eigendecomposition(matrix)

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

    results = hhl_module.get_hhl_results(
        task, matrix, b_vector, num_clock_qubits=2, verbose=True
    )

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

    circ = hhl_module.hhl_circuit(
        matrix, b_vector, num_clock_qubits=2, scaling_factor=np.pi
    )
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
