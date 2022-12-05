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
from scipy.optimize import minimize

from braket.experimental.algorithms.quantum_circuit_born_machine import QCBM, mmd_loss


def test_mmd_loss():
    loss = mmd_loss(np.zeros(4), np.zeros(4))
    assert np.isclose(loss, 0, rtol=1e-5)


def test_qcbm():
    n_qubits = 2
    n_layers = 2
    data = np.ones(3 * n_layers * n_qubits)
    device = LocalSimulator()
    qcbm = QCBM(device, n_qubits, n_layers, data)
    init_params = np.zeros((n_layers, n_qubits, 3))
    probs = qcbm.get_probabilities(init_params)
    expected = np.array([1, 0, 0, 0])
    assert np.isclose(probs, expected).all()


@pytest.mark.xfail(raises=ValueError)
def test_qcbm_no_qubits():
    n_qubits = 0
    n_layers = 1
    data = np.ones(3 * n_layers * n_qubits)
    device = LocalSimulator()
    QCBM(device, n_qubits, n_layers, data)


def test_qcbm_gradient():
    n_layers = 1
    n_qubits = 2
    n_iterations = 1
    data = np.ones(3 * n_layers * n_qubits)
    device = LocalSimulator()

    init_params = np.random.rand(3 * n_layers * n_qubits)
    qcbm = QCBM(device, n_qubits, n_layers, data)

    result = minimize(
        lambda x: mmd_loss(qcbm.get_probabilities(x), data),
        x0=init_params,
        method="L-BFGS-B",
        jac=lambda x: qcbm.gradient(x),
        options={"maxiter": n_iterations},
    )

    assert result.njev < 4
    assert result.nit == 1
    assert result.status == 1
