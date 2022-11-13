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


from typing import List

import numpy as np
from braket.circuits import Circuit
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
        self, device: Device, n_qubits: int, n_layers: int, data: np.ndarray, shots: int = 10_000
    ):
        """Quantum circuit Born machine.

        Consists of `n_layers`, where each layer is a rotation layer (rx, rz, rx)
        followed by an entangling layer of cnot gates.

        Args:
            device (Device): Amazon Braket device to use
            n_qubits (int): Number of qubits
            n_layers (int): Number of layers
            data (ndarray): Target probabilities
            shots (int): Number of shots. Defaults to 10_000.
        """
        self.device = device
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.neighbors = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        self.data = data  # target probabilities
        self.shots = shots

    def entangler(self, circ: Circuit) -> None:
        """Add CNot gates to circuit.

        Args:
            circ (Circuit): The circuit to add CNots to.
        """
        for i, j in self.neighbors:
            circ.cnot(i, j)

    def rotation_layer(self, circ: Circuit, params: np.ndarray) -> None:
        """Add rotation layers  to circuit.

        Args:
            circ (Circuit): The circuit to add CNots to.
            params (ndarray): Parameters for rotation layers
        """
        for n in range(self.n_qubits):
            circ.rx(n, params[n, 0])
            circ.rz(n, params[n, 1])
            circ.rx(n, params[n, 2])

    def create_circuit(self, params: np.ndarray) -> Circuit:
        """Creates a QCBM circuit, and returns the probabilities

        Args:
            params (ndarray): Parameters for the rotation gates in the circuit,
                length = 3 * n_qubits * n_layers

        Returns:
            Circuit: Circuit with parameters fixed to `params`.
        """
        try:
            params = params.reshape(self.n_layers, self.n_qubits, 3)
        except Exception:
            print(
                "Length of initial parameters was not correct. Expected: "
                + f"{self.n_layers*self.n_qubits*3} but got {len(params)}."
            )
        circ = Circuit()
        self.rotation_layer(circ, params[0])
        for L in range(1, self.n_layers):
            self.entangler(circ)
            self.rotation_layer(circ, params[L])
        self.entangler(circ)
        circ.probability()
        return circ

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        """Get probabilities from a run.

        Args:
            params (np.ndarray): Parameters for QCBM.

        Returns:
            ndarray: Probabilities.
        """
        circ = self.create_circuit(params)
        probs = self.device.run(circ, shots=self.shots).result().values[0]
        return probs

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
        qcbm_probs = self.probabilities(params)
        shift = np.ones_like(params) * np.pi / 2
        shifted_params = np.stack([params + np.diag(shift), params - np.diag(shift)]).reshape(
            2 * len(params), len(params)
        )
        circuits = [self.create_circuit(p) for p in shifted_params]

        try:
            result = self.device.run_batch(circuits, shots=self.shots).results()
        except Exception:
            result = [self.device.run(c, shots=self.shots).result() for c in circuits]

        res = [result[i].values[0] for i in range(len(circuits))]
        res = np.array(res).reshape(2, len(params), 2**self.n_qubits)

        grad = np.zeros(len(params))
        for i in range(len(params)):
            grad_pos = compute_kernel(qcbm_probs, res[0][i]) - compute_kernel(qcbm_probs, res[1][i])
            grad_neg = compute_kernel(self.data, res[0][i]) - compute_kernel(self.data, res[1][i])
            grad[i] = grad_pos - grad_neg
        return grad


def compute_kernel(px: np.ndarray, py: np.ndarray, sigma_list: List[float] = [0.1, 1]) -> float:
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

    mmd_xx = compute_kernel(px, px, sigma_list)
    mmd_yy = compute_kernel(py, py, sigma_list)
    mmd_xy = compute_kernel(px, py, sigma_list)
    return mmd_xx + mmd_yy - 2 * mmd_xy
