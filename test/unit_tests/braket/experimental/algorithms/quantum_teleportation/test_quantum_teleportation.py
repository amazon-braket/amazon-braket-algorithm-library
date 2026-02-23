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

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.quantum_teleportation import (
    get_quantum_teleportation_results,
    quantum_teleportation_circuit,
    run_quantum_teleportation,
)


@pytest.fixture
def device():
    return LocalSimulator()


# teleport |0>, expect P(|0>)=1
def test_teleport_zero(device):
    circ = quantum_teleportation_circuit()
    task = run_quantum_teleportation(circ, device, shots=0)
    result = get_quantum_teleportation_results(task)
    assert np.isclose(result["0"], 1.0)
    assert np.isclose(result["1"], 0.0)


# teleport |1>, expect P(|1>)=1
def test_teleport_one(device):
    circ = quantum_teleportation_circuit(Circuit().x(0))
    task = run_quantum_teleportation(circ, device, shots=0)
    result = get_quantum_teleportation_results(task)
    assert np.isclose(result["0"], 0.0)
    assert np.isclose(result["1"], 1.0)


# teleport |+>, expect 50/50
def test_teleport_plus(device):
    circ = quantum_teleportation_circuit(Circuit().h(0))
    task = run_quantum_teleportation(circ, device, shots=0)
    result = get_quantum_teleportation_results(task)
    assert np.isclose(result["0"], 0.5, atol=1e-6)
    assert np.isclose(result["1"], 0.5, atol=1e-6)


# teleport Ry(pi/3)|0>, verify probabilities match
def test_teleport_arbitrary_state(device):
    angle = np.pi / 3
    circ = quantum_teleportation_circuit(Circuit().ry(0, angle))
    task = run_quantum_teleportation(circ, device, shots=0)
    result = get_quantum_teleportation_results(task)
    # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    assert np.isclose(result["0"], np.cos(angle / 2) ** 2, atol=1e-6)
    assert np.isclose(result["1"], np.sin(angle / 2) ** 2, atol=1e-6)


# teleport |+i>, verify phase is preserved via S†H readout
def test_teleport_plus_i_phase(device):
    # prep |+i> = S H |0>
    teleportation_circuit = quantum_teleportation_circuit(Circuit().h(0).s(0))

    # S†H maps |+i> to |0>, so successful teleportation => P(|0>) ~= 1
    phase_sensitive_circuit = (
        Circuit().add(teleportation_circuit.instructions).si(2).h(2).probability([2])
    )

    task = run_quantum_teleportation(phase_sensitive_circuit, device, shots=0)
    result = get_quantum_teleportation_results(task)

    assert np.isclose(result["0"], 1.0, atol=1e-6)
    assert np.isclose(result["1"], 0.0, atol=1e-6)
