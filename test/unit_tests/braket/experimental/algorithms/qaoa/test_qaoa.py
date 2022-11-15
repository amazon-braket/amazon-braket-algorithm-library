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
from braket.circuits import Circuit, FreeParameter, Observable
from braket.devices import LocalSimulator

from braket.experimental.algorithms.qaoa.qaoa import (
    qaoa, evaluate_circuit, evaluate_loss
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
    task = evaluate_circuit(device, circ, values, shots)
    result = task.result().values[0]
    assert np.isclose(result[0], 1)


def test_qaoa_evaluate_loss():
    circ = Circuit().i(0)
    circ.expectation(observable=Observable.Z(), target=0)
    task = LocalSimulator().run(circ, shots=0)
    js = [1]
    assert np.isclose(evaluate_loss(task, js), 1)
    







# gammas = [0.1, 0.2]
# betas = [0.3, 0.4]
# params = gammas + betas
# J_sub = np.array([[0, 1], [0, 0]])
# N = J_sub.shape[0]
# bitstring_init = -1 * np.ones([N])
# energy_init = np.dot(bitstring_init, np.dot(J_sub, bitstring_init))


# def test_qaoa():
#     n_qubits = 2
#     n_layers = 2
#     ising = np.diag(np.ones(n_qubits - 1), 1)
#     circ = qaoa(n_qubits, n_layers, ising)
#     print(circ)

#     values = np.random.rand(2 * n_layers)
#     print(values)
#     c2 = circ.make_bound_circuit(dict(zip(np.array(list(circ.parameters), dtype=str), values)))

#     print(c2)

#     device = LocalSimulator()
#     device.run(c2, shots=10_000)


# @circuit.subroutine(register=True)
# def qaoa_circuit():

#     circ = qaoa(LocalSimulator(), params, N, J_sub)
#     return circ


# def test_qaoa():
#     circ = qaoa_circuit()
#     assert circ.depth == 6
#     assert circ.qubit_count == 2


# def test_qaoa_optimization():
#     metrics = {
#         "count": 1,  # Elapsed optimization steps
#         "optimal_energy": energy_init,  # Global optimal energy
#         "opt_energies": [],  # Optimal energy at each step
#         "global_energies": [],  # Global optimal energy at each step
#         "optimal_bitstring": bitstring_init,  # Global optimal bitstring
#         "opt_bitstrings": [],  # Optimal bitstring at each step
#         "costs": [],  # Cost (average energy) at each step
#         "res": None,  # Quantum result object
#         "params": [],  # Track parameters
#     }
#     options = {"disp": True, "maxiter": 500}

#     result, metrics = run_qaoa(
#         device=LocalSimulator(),
#         options=options,
#         depth=3,
#         ising=J_sub,
#         n_qubits=N,
#         shots=10,
#         opt_method="Powell",
#         metrics=metrics,
#     )

#     assert result.fun > 0.00 or result.fun < 0.00
#     assert len(result.x) > 0
#     assert metrics["optimal_energy"] > 0.00 or metrics["optimal_energy"] < 0.00




