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

# Quantum Fourier Transform: Amazon Braket Algorithm Library

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, circuit
from braket.devices.device import Device
from braket.tasks.gate_model_quantum_task_result import GateModelQuantumTaskResult


def quantum_fourier_transform(qubits: List[int]) -> Circuit:
    """
    Construct a circuit object corresponding to the Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the QFT.

    Args:
        qubits (List[int]): The list of qubits labels on which to apply the QFT

    Returns:
        Circuit: qft circuit
    """

    qft_circ = Circuit()

    # get number of qubits
    num_qubits = len(qubits)

    for k in range(num_qubits):
        # First add a Hadamard gate
        qft_circ.h(qubits[k])

        # Then apply the controlled rotations, with weights (angles) defined by the distance
        # to the control qubit. Start on the qubit after qubit k, and iterate until the end.
        # When num_qubits==1, this loop does not run.
        for j in range(1, num_qubits - k):
            angle = 2 * math.pi / (2 ** (j + 1))
            qft_circ.cphaseshift(qubits[k + j], qubits[k], angle)

    # Then add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qft_circ.swap(qubits[i], qubits[-i - 1])

    return qft_circ


@circuit.subroutine(register=True)
def qft(qubits: List[int]) -> Circuit:
    """qft circuit.

    Args:
        qubits (List[int]): The list of qubits labels on which to apply the QFT

    Returns:
        Circuit: qft circuit
    """
    return quantum_fourier_transform(qubits)


def inverse_quantum_fourier_transform(qubits: List[int]) -> Circuit:
    """
    Construct a circuit object corresponding to the inverse Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the circuit.

    Args:
        qubits (List[int]): The list of qubits on which to apply the inverse QFT

    Returns:
        Circuit: inverse qft circuit
    """
    # instantiate circuit object
    qft_circ = Circuit()

    # get number of qubits
    num_qubits = len(qubits)

    # First add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qft_circ.swap(qubits[i], qubits[-i - 1])

    # Start on the last qubit and work to the first.
    for k in reversed(range(num_qubits)):

        # Apply the controlled rotations, with weights (angles) defined by the distance
        # to the control qubit. # These angles are the negative of the angle used in the QFT.
        # Start on the last qubit and iterate until the qubit after k.
        # When num_qubits==1, this loop does not run.
        for j in reversed(range(1, num_qubits - k)):
            angle = -2 * math.pi / (2 ** (j + 1))
            qft_circ.cphaseshift(qubits[k + j], qubits[k], angle)

        # Then add a Hadamard gate
        qft_circ.h(qubits[k])

    return qft_circ


@circuit.subroutine(register=True)
def iqft(qubits: List[int]) -> Circuit:
    """Inverse qft circuit.

    Args:
        qubits (List[int]): The list of qubits labels on which to apply the IQFT

    Returns:
        Circuit: inverse qft circuit
    """
    return inverse_quantum_fourier_transform(qubits)


def run_quantum_fourier_transform(
    qubits: List[int],
    n_shots: int,
    device: Device,
    state_prep_circ: Circuit = Circuit(),
    analysis_circ: Circuit = Circuit(),
    inverse: bool = False,
) -> GateModelQuantumTaskResult:
    """Execute QFT algorithm and returns results.

    Args:
        qubits (List[int]): qubit indices
        n_shots (int): number of shots
        device (Device): The requested device (default: LocalSimulator)
        state_prep_circ (Circuit): circuit to be run before qft
        analysis_circ (Circuit): circuit to be run after  qft
        inverse (bool): do the inverse qft

    Returns:
        GateModelQuantumTaskResult: circuit execution result

    """
    circuit = state_prep_circ

    if inverse:
        circuit = circuit.iqft(qubits)
    else:
        circuit = circuit.qft(qubits)

    circuit = circuit + analysis_circ
    circuit.probability()

    task = device.run(circuit, shots=n_shots)

    return task


# def


def plot_bitstrings(probabilities: List[float]) -> None:
    """Plot the measure results.
    Args:
        probabilities (List[float]): Probabilities of measuring each bitstring.
    """
    num_qubits = int(np.log2(len(probabilities)))
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    plt.bar(bitstring_keys, probabilities)
    plt.xlabel("bitstrings")
    plt.ylabel("probability")
    plt.xticks(rotation=90)
    plt.ylim([0, 1])
