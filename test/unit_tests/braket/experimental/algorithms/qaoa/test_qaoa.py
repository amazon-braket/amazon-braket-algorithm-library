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
from braket.circuits import circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.qaoa import qaoa

gammas = [0.1, 0.2]
betas = [0.3, 0.4]
params = gammas + betas
J_sub = np.array([[0, 1], [0, 0]])
N = J_sub.shape[0]
bitstring_init = -1 * np.ones([N])
energy_init = np.dot(bitstring_init, np.dot(J_sub, bitstring_init))


@circuit.subroutine(register=True)
def qaoa_circuit():

    circ = qaoa.qaoa(params, LocalSimulator(), N, J_sub)
    return circ


def test_qaoa():
    circ = qaoa_circuit()

    assert circ.depth == 6
    assert circ.qubit_count == 2


def test_qaoa_optimization():
    tracker = {
        "count": 1,  # Elapsed optimization steps
        "optimal_energy": energy_init,  # Global optimal energy
        "opt_energies": [],  # Optimal energy at each step
        "global_energies": [],  # Global optimal energy at each step
        "optimal_bitstring": bitstring_init,  # Global optimal bitstring
        "opt_bitstrings": [],  # Optimal bitstring at each step
        "costs": [],  # Cost (average energy) at each step
        "res": None,  # Quantum result object
        "params": [],  # Track parameters
    }
    options = {"disp": True, "maxiter": 500}

    result, tracker = qaoa.run_qaoa(
        device=LocalSimulator(),
        options=options,
        p=3,
        ising=J_sub,
        n_qubits=N,
        n_shots=10,
        opt_method="Powell",
        tracker=tracker,
    )

    assert result.fun > 0.00 or result.fun < 0.00
    assert len(result.x) > 0
    assert tracker["optimal_energy"] > 0.00 or tracker["optimal_energy"] < 0.00
