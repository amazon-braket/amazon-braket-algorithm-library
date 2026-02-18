from typing import List

import numpy as np

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.adaptive_shot_allocation.adaptive_allocator_braket_helpers import (
    observable_from_string,
)

"""
Utilities for creating and manipulating quantum circuits.
"""

_localSim = LocalSimulator()


def create_random_state(num_qubits: int = 4) -> Circuit:
    """
    Generate a quantum circuit with random rotations and entanglement.

    Args:
        num_qubits (int): Number of qubits in the circuit

    Returns:
        Circuit: Quantum circuit with random state preparation
    """
    circ = Circuit()

    # First layer of rotations
    for i in range(num_qubits):
        circ.ry(i, np.pi * np.random.rand())

    # Entangling layer
    for i in range(num_qubits - 1):
        circ.cnot(control=i, target=i + 1)

    # Second layer of rotations
    for i in range(num_qubits):
        circ.ry(i, np.pi * np.random.rand())

    return circ


def create_bell_state() -> Circuit:
    """
    Generate a Bell state on the first two qubits.

    Returns:
        Circuit: Quantum circuit preparing a Bell state
    """
    return Circuit().h(0).cnot(control=0, target=1)


def get_exact_expectation(circuit: Circuit, paulis: List[str], coeffs: List[float]) -> float:
    """
    Calculate exact expectation value for a Hamiltonian.

    Args:
        circuit (Circuit): Quantum circuit to measure
        paulis (List[str]): List of Pauli string operators
        coeffs (List[float]): Corresponding coefficients

    Returns:
        float: Exact expectation value
    """
    device = _localSim
    e_exact = 0.0
    for c, p in zip(coeffs, paulis):
        expect_circ = circuit.copy()
        expect_circ.expectation(observable_from_string(p))
        result = device.run(expect_circ, shots=0).result()
        e_exact += c * result.values[0]
    return e_exact


"""
Utilities for allocating measurement shots across different measurement groups.
"""


def get_uniform_shots(num_groups: int, total_shots: int) -> List[int]:
    """
    Generate uniform shot allocation across measurement groups.

    Args:
        num_groups (int): Number of measurement groups
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    shots = [total_shots // num_groups] * num_groups
    remainder = total_shots % num_groups
    for i in range(remainder):
        shots[i] += 1
    return shots


def get_random_shots(num_groups: int, total_shots: int) -> List[int]:
    """
    Generate random shot allocation across measurement groups.

    Args:
        num_groups (int): Number of measurement groups
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    weights = np.random.rand(num_groups)
    shots = np.floor(weights * total_shots / sum(weights)).astype(int)
    remainder = total_shots - sum(shots)
    for i in range(remainder):
        shots[i] += 1
    return shots.tolist()


def get_weighted_shots(cliq: List[List[int]], coeffs: List[float], total_shots: int) -> List[int]:
    """
    Generate weighted shot allocation based on coefficient magnitudes.

    Args:
        cliq (List[List[int]]): List of measurement groups (cliques)
        coeffs (List[float]): Coefficients for each term
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    weights = np.array([sum(np.abs(np.array(coeffs)[c])) for c in cliq])
    shots = np.floor(weights * total_shots / sum(weights)).astype(int)
    remainder = total_shots - sum(shots)
    for i in range(remainder):
        shots[i] += 1
    return shots.tolist()
