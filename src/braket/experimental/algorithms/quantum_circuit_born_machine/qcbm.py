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


from typing import List, Tuple

import numpy as np
from braket.circuits import Circuit, FreeParameter, circuit
from braket.devices import Device


class QCBM:
    """Quantum circuit Born machine.

    Example: n_layers = 1, n_qubits = 2
    T  : |    0    |    1    |    2    |3|4|Result Types|

    q0 : -Rx(0.667)-Rz(0.783)-Rx(0.257)-C-X-Probability--
                                        | | |
    q1 : -Rx(0.549)-Rz(0.878)-Rx(0.913)-X-C-Probability--

    T  : |    0    |    1    |    2    |3|4|Result Types|
    """

    def __init__(
        self,
        device: Device,
        n_qubits: int,
        n_layers: int,
        target_probabilities: np.ndarray,
        shots: int = 10_000,
    ):
        """Quantum circuit Born machine.

        Consists of `n_layers`, where each layer is a rotation layer (rx, rz, rx)
        followed by an entangling layer of cnot gates.

        Args:
            device (Device): Amazon Braket device to use
            n_qubits (int): Number of qubits
            n_layers (int): Number of layers
            target_probabilities (ndarray): Target probabilities.
            shots (int): Number of shots. Defaults to 10_000.
        """
        if n_qubits <= 1:
            raise ValueError("Number of qubits must be greater than 1.")
        self.device = device
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.neighbors = [(i, (i + 1) % n_qubits) for i in range(n_qubits - 1)]
        self.target_probs = target_probabilities
        self.shots = shots
        self.parameters = [
            [
                [FreeParameter(f"theta_{layer}_{qubit}_{i}") for i in range(3)]
                for qubit in range(n_qubits)
            ]
            for layer in range(n_layers)
        ]
        self.parametric_circuit = self._create_circuit()

    def _create_circuit(self) -> Circuit:
        """Creates a QCBM circuit, and returns the probabilities.

        Returns:
            Circuit: Circuit with parameters fixed to `params`.
        """
        circ = Circuit()
        circ.qcbm_layers(self.neighbors, self.parameters)
        circ.probability()
        return circ

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        """Run and get probability results.

        Args:
            values (np.ndarray): Values for free parameters.

        Returns:
            np.ndarray: Probabilities vector.
        """
        qcbm_original_circuit = self.bound_circuit(values)
        task = self.device.run(qcbm_original_circuit, shots=self.shots)
        qcbm_probs = task.result().values[0]
        return qcbm_probs

    def bound_circuit(self, values: np.ndarray) -> np.ndarray:
        """Get probabilities from the current parameters.

        Args:
            values (ndarray): Parameters for QCBM.

        Returns:
            ndarray: Probabilities.
        """
        # Need to flatten parameters and also parameter.
        flat_values = values.flatten()
        flat_parameters = np.array(self.parameters, dtype=str).flatten()
        bound_values = dict(zip(flat_parameters, flat_values))
        circ = self.parametric_circuit.make_bound_circuit(bound_values)
        return circ

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """Gradient for QCBM via:

        Liu, Jin-Guo, and Lei Wang.
        “Differentiable Learning of Quantum Circuit Born Machine.”
        Physical Review A 98, no. 6 (December 19, 2018): 062324.
        https://doi.org/10.1103/PhysRevA.98.062324.

        Args:
            params (ndarray): Parameters for the rotation gates in the QCBM

        Returns:
            ndarray: Gradient vector
        """

        qcbm_probs = self.get_probabilities(params)

        shift = np.ones_like(params) * np.pi / 2
        shifted_params = np.stack([params + np.diag(shift), params - np.diag(shift)]).reshape(
            2 * len(params), len(params)
        )

        probs = [self.get_probabilities(p) for p in shifted_params]
        probs = np.array(probs).reshape(2, len(params), 2**self.n_qubits)

        grad = np.zeros(len(params))
        for i in range(len(params)):
            grad_pos = _compute_kernel(qcbm_probs, probs[0][i]) - _compute_kernel(
                qcbm_probs, probs[1][i]
            )
            grad_neg = _compute_kernel(self.target_probs, probs[0][i]) - _compute_kernel(
                self.target_probs, probs[1][i]
            )
            grad[i] = grad_pos - grad_neg
        return grad


def _compute_kernel(px: np.ndarray, py: np.ndarray, sigma_list: List[float] = [0.1, 1]) -> float:
    r"""Gaussian radial basis function (RBF) kernel.

    K(x, y) = sum_\sigma exp(-|x-y|^2/(2\sigma^2 ))

    Args:
        px (ndarray): Probability distribution
        py (ndarray): Target probability distribution
        sigma_list (List[float]): Standard deviations of distribution. Defaults to [0.1, 1].

    Returns:
        float: Value of the Gaussian RBF function for kernel(px, py).
    """
    x = np.arange(len(px))
    y = np.arange(len(py))
    K = sum(np.exp(-np.abs(x[:, None] - y[None, :]) ** 2 / (2 * s**2)) for s in sigma_list)
    kernel = px @ K @ py
    return kernel


def mmd_loss(px: np.ndarray, py: np.ndarray, sigma_list: List[float] = [0.1, 1]) -> float:
    r"""Maximum Mean Discrepancy loss (MMD).

    MMD determines if two distributions are equal by looking at the difference between
    their means in feature space.

    MMD(x, y) = | \sum_{j=1}^N \phi(y_j) - \sum_{i=1}^N \phi(x_i) |_2^2

    With a RBF kernel, we apply the kernel trick to expand MMD to

    MMD(x, y) = \sum_{j=1}^N \sum_{j'=1}^N k(y_j, y_{j'})
                + \sum_{i=1}^N \sum_{i'=1}^N k(x_i, x_{i'})
                - 2 \sum_{j=1}^N \sum_{i=1}^N k(y_j, x_i)

    For the RBF kernel, MMD is zero if and only if the distributions are identical.

    Args:
        px (ndarray): Probability distribution
        py (ndarray): Target probability distribution
        sigma_list (List[float]):  Standard deviations of distribution. Defaults to [0.1, 1].

    Returns:
        float: Value of the MMD loss
    """

    mmd_xx = _compute_kernel(px, px, sigma_list)
    mmd_yy = _compute_kernel(py, py, sigma_list)
    mmd_xy = _compute_kernel(px, py, sigma_list)
    return mmd_xx + mmd_yy - 2 * mmd_xy


@circuit.subroutine(register=True)
def qcbm_layers(
    neighbors: List[Tuple[int, int]], parameters: List[List[List[FreeParameter]]]
) -> Circuit:
    """QCBM layers.

    Args:
        neighbors (List[Tuple[int,int]]): List of qubit pairs.
        parameters (List[List[List[FreeParameter]]]): List of FreeParameters. First index is
            n_layers, second is n_qubits, and third is [0,1,2]

    Returns:
        Circuit: QCBM circuit.
    """
    n_layers = len(parameters)
    circ = Circuit()
    circ.rotation_layer(parameters[0])
    for L in range(1, n_layers):
        circ.entangler(neighbors)
        circ.rotation_layer(parameters[L])
    circ.entangler(neighbors)
    return circ


@circuit.subroutine(register=True)
def entangler(neighbors: List[Tuple[int, int]]) -> Circuit:
    """Add CNot gates to circuit.

    Args:
        neighbors (List[Tuple[int,int]]): Neighbors for CNots to connect

    Returns:
        Circuit: CNot entangling layer
    """
    circ = Circuit()
    for i, j in neighbors:
        circ.cnot(i, j)
    return circ


@circuit.subroutine(register=True)
def rotation_layer(parameters: List[List[FreeParameter]]) -> Circuit:
    """Add rotation layers  to circuit.

    Args:
        parameters (List[List[FreeParameter]]): Parameters for rotation layers.

    Returns:
        Circuit: Rotation layer
    """
    circ = Circuit()
    n_qubits = len(parameters)
    for n in range(n_qubits):
        circ.rx(n, parameters[n][0])
        circ.rz(n, parameters[n][1])
        circ.rx(n, parameters[n][2])
    return circ
