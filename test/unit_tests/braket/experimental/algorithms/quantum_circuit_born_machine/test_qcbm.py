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
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_circuit_born_machine.qcbm import QCBM


def test_qcbm():

    n_qubits = 2
    n_layers = 1

    data = np.ones(3 * n_layers * n_qubits)

    device = LocalSimulator()
    qcbm = QCBM(device, n_qubits, n_layers, data)

    expected = Circuit()
    expected.rx(0, 1.0).rz(0, 1.0).rx(0, 1.0)
    expected.rx(1, 1.0).rz(1, 1.0).rx(1, 1.0)
    expected.cnot(0, 1)
    expected.cnot(1, 0)
    expected.probability()

    init_params = np.ones(3 * n_layers * n_qubits)

    circ = qcbm.create_circuit(init_params)

    assert circ == expected
