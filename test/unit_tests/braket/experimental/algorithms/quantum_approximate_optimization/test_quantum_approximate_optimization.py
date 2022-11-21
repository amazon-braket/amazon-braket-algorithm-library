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
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_approximate_optimization import (
    cost_function,
    qaoa,
    run_qaoa_circuit,
)


def test_qaoa():
    n_qubits = 2
    n_layers = 1
    coupling_matrix = np.diag(np.ones(n_qubits - 1), 1)
    circ = qaoa(n_qubits, n_layers, coupling_matrix)
    assert circ.qubit_count == 2


def test_qaoa_evaluate_circuit():
    param = FreeParameter("theta")
    circ = Circuit().rx(0, param).probability()
    device = LocalSimulator()
    shots = 0
    values = [0]
    task = run_qaoa_circuit(device, circ, values, shots)
    result = task.result().values[0]
    assert np.isclose(result[0], 1)


def test_qaoa_cost_function():
    coeffs = [1]
    n_qubits = 2
    n_layers = 1
    coupling_matrix = np.diag(np.ones(n_qubits - 1), 1)
    circ = qaoa(n_qubits, n_layers, coupling_matrix)
    cost = cost_function(np.ones(2), LocalSimulator(), circ, coeffs, [])
    print("loss is", cost)
    assert np.isclose(cost, -0.636827341031835, rtol=1e-6)
