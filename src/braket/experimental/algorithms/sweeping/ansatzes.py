import numpy as np

from braket.experimental.algorithms.sweeping.typing import UNITARY_LAYER


def generate_staircase_ansatz(num_qubits: int, num_layers: int) -> list[UNITARY_LAYER]:
    """Generate a staircase ansatz with the specified number of qubits and layers.

    Args:
        num_qubits (int): The number of qubits in the ansatz.
        num_layers (int): The number of layers in the ansatz.

    Returns:
        list[UNITARY_LAYER]: The generated staircase ansatz.
    """
    unitary_layers: list[UNITARY_LAYER] = []

    for i in range(num_layers):
        unitary_layers.append([])

        for j in range(num_qubits - 1):
            unitary_layers[i].append(([j, j + 1], np.eye(4, dtype=np.complex128)))

    return unitary_layers


def generate_brickwall_ansatz(num_qubits: int, num_layers: int) -> list[UNITARY_LAYER]:
    """Generate a brickwall ansatz with the specified number of qubits and layers.

    Args:
        num_qubits (int): The number of qubits in the ansatz.
        num_layers (int): The number of layers in the ansatz.

    Returns:
        list[UNITARY_LAYER]: The generated brickwall ansatz.
    """
    unitary_layers: list[UNITARY_LAYER] = []

    if num_qubits % 2 == 0:
        start_1 = 1
        start_2 = 0
    else:
        start_1 = 0
        start_2 = 1

    for i in range(num_layers):
        unitary_layers.append([])

        for j in range(start_1, num_qubits - 1, 2):
            unitary_layers[i].append(([j, j + 1], np.eye(4, dtype=np.complex128)))
        for j in range(start_2, num_qubits - 1, 2):
            unitary_layers[i].append(([j, j + 1], np.eye(4, dtype=np.complex128)))

    return unitary_layers
