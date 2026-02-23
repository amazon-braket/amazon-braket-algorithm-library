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
from typing import Dict, Optional

import numpy as np

from braket.circuits import Circuit
from braket.devices import Device
from braket.tasks import QuantumTask


def quantum_teleportation_circuit(state_prep_circ: Optional[Circuit] = None) -> Circuit:
    """Create a quantum teleportation circuit using deferred measurement.

    Teleports the state of qubit 0 to qubit 2 via a shared Bell pair.
    Uses CNOT and CZ gates instead of classical feedforward (deferred measurement
    principle), so the circuit can run on any simulator without mid-circuit measurement.

    Qubit layout:
        qubit 0: message qubit (Alice's input)
        qubit 1: Alice's half of Bell pair
        qubit 2: Bob's half of Bell pair (receives teleported state)

    Args:
        state_prep_circ (Optional[Circuit]): Circuit to prepare the state on qubit 0.
            If None, teleports the |0> state.

    Returns:
        Circuit: Quantum teleportation circuit with probability measurement on qubit 2.
    """
    circ = Circuit()

    # Prepare the state to teleport on qubit 0
    if state_prep_circ is not None:
        circ.add_circuit(state_prep_circ)

    # Create Bell pair between qubit 1 and qubit 2
    circ.h(1)
    circ.cnot(1, 2)

    # Bell measurement on qubits 0 and 1
    circ.cnot(0, 1)
    circ.h(0)

    # Deferred corrections (equivalent to classical conditional gates)
    circ.cnot(1, 2)  # if qubit 1 measured |1>: apply X to qubit 2
    circ.cz(0, 2)  # if qubit 0 measured |1>: apply Z to qubit 2

    circ.probability([2])
    return circ


def run_quantum_teleportation(circuit: Circuit, device: Device, shots: int = 1000) -> QuantumTask:
    """Run a quantum teleportation circuit on a device.

    Args:
        circuit (Circuit): Quantum teleportation circuit.
        device (Device): Braket device or simulator.
        shots (int): Number of shots. Defaults to 1000.

    Returns:
        QuantumTask: Quantum task.
    """
    return device.run(circuit, shots=shots)


def get_quantum_teleportation_results(task: QuantumTask) -> Dict[str, float]:
    """Extract teleportation results from a quantum task.

    Args:
        task (QuantumTask): Completed quantum task.

    Returns:
        Dict[str, float]: Probability of Bob's qubit in {"0": P(|0>), "1": P(|1>)}.
    """
    probs = task.result().result_types[0].value
    probs = np.round(probs, 10)
    return {"0": probs[0], "1": probs[1]}
