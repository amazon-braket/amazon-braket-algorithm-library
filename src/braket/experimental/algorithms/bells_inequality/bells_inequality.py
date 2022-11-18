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
from collections import Counter
from typing import List, Tuple

import numpy as np
from braket.circuits import Circuit, Qubit, circuit
from braket.devices import Device
from braket.tasks import QuantumTask


def create_bell_inequality_circuits(
    qubit0: Qubit = 0,
    qubit1: Qubit = 1,
    angle_A: float = 0,
    angle_B: float = np.pi / 3,
    angle_C: float = 2 * np.pi / 3,
) -> List[Circuit]:
    """Create the three circuits for Bell's inequality. Default angles will give maximum violation
    of Bell's inequality.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.
        angle_A (float): Angle for the first measurement basis A. Defaults to 0.
        angle_B (float): Angle for the second measurement basis B. Defaults to np.pi/3.
        angle_C (float): Angle for the third measurement basis C. Defaults to 2*np.pi/3 to give
            maximum violation of Bell's inequality.

    Returns:
        List[Circuit]: Three circuits circAB, circAC, circBC.
    """
    circAB = bell_singlet_rotated_basis(qubit0, qubit1, angle_A, angle_B)
    circAC = bell_singlet_rotated_basis(qubit0, qubit1, angle_A, angle_C)
    circBC = bell_singlet_rotated_basis(qubit0, qubit1, angle_B, angle_C)
    return [circAB, circAC, circBC]


def run_bell_inequality(
    circuits: List[Circuit],
    device: Device,
    shots: int = 1_000,
) -> List[QuantumTask]:

    """Submit three Bell circuits to a device.

    Args:
        circuits (List[Circuit]): Three Bell inequality circuits in order circAB, circAC, circBC.
        device (Device): Quantum device or simulator.
        shots (int): Number of shots. Defaults to 1_000.

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    tasks = [device.run(circ, shots=shots) for circ in circuits]
    return tasks


def get_bell_inequality_results(
    tasks: List[QuantumTask], verbose: bool = True
) -> Tuple[List[Counter], float, float, float]:
    """Return Bell task results after post-processing.

    Args:
        tasks (List[QuantumTask]): List of quantum tasks.
        verbose (bool): Controls printing of the inequality result. Defaults to True.

    Returns:
        Tuple[List[Counter], float, float, float]: results, pAB, pAC, pBC
    """
    results = [task.result().result_types[0].value for task in tasks]  # probability result type
    prob_same = np.array([d[0] + d[3] for d in results])  # 00 and 11 states
    prob_different = np.array([d[1] + d[2] for d in results])  # 01 and 10 states

    pAB, pAC, pBC = prob_same - prob_different  # Bell probabilities
    bell_ineqality_lhs = np.abs(pAB - pAC) - pBC
    if verbose:
        print(f"P(a,b) = {pAB},P(a,c) = {pAC},P(b,c) = {pBC}")
        print(f"Bell's' inequality: {bell_ineqality_lhs} ≤ 1")
        if bell_ineqality_lhs > 1:
            print("Bell's inequality is violated!")
        else:
            print("Bell's inequality is not violated due to noise.")
    return results, pAB, pAC, pBC


def bell_singlet_rotated_basis(
    qubit0: Qubit, qubit1: Qubit, rotation0: float, rotation1: float
) -> Circuit:
    """Prepare a Bell singlet state in a Rx-rotated meaurement basis.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.
        rotation0 (float): First qubit Rx rotation angle.
        rotation1 (float): Second qubit Rx rotation angle.

    Returns:
        Circuit: the Braket circuit that prepares the Bell circuit.
    """
    circ = Circuit().bell_singlet(qubit0, qubit1)
    if rotation0 != 0:
        circ.rx(qubit0, rotation0)
    if rotation1 != 0:
        circ.rx(qubit1, rotation1)
    circ.probability()
    return circ


@circuit.subroutine(register=True)
def bell_singlet(qubit0: Qubit, qubit1: Qubit) -> Circuit:
    """Prepare a Bell singlet state.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        Circuit: the Braket circuit that prepares the Bell single state.
    """
    return Circuit().x(qubit0).x(qubit1).h(qubit0).cnot(qubit0, qubit1)
