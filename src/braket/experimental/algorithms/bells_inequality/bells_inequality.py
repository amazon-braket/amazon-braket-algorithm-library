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
from braket.circuits import Circuit, Qubit
from braket.devices import Device
from braket.tasks import QuantumTask


def run_bell_tasks(
    device: Device, shots: int = 1_000, qubit0: Qubit = 0, qubit1: Qubit = 1
) -> List[QuantumTask]:
    """Submit three Bell circuits to a device.

    Args:
        device (Device): Quantum device or simulator.
        shots (int): Number of shots. Defaults to 1_000.
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    circAB = bell_singlet_rotated(qubit0, qubit1, 0, np.pi / 3.0)
    circAC = bell_singlet_rotated(qubit0, qubit1, 0, 2 * np.pi / 3.0)
    circBC = bell_singlet_rotated(qubit0, qubit1, np.pi / 3.0, 2 * np.pi / 3.0)
    tasks = [device.run(circ, shots=shots) for circ in [circAB, circAC, circBC]]
    return tasks


def get_bell_results(
    tasks: List[QuantumTask], verbose: bool = True
) -> Tuple[List[Counter], float, float, float]:
    """Return Bell task results after post-processing.

    Args:
        tasks (List[QuantumTask]): List of quantum tasks.
        verbose (bool): Controls printing of the inequality result. Defaults to True.

    Returns:
        Tuple[List[Counter[float]], float, float, float]: results, pAB, pAC, pBC
    """
    results = [task.result().measurement_probabilities for task in tasks]
    prob_same = [d["00"] + d["11"] for d in results]
    prob_different = [d["01"] + d["10"] for d in results]
    # Bell probabilities
    pAB, pAC, pBC = np.array(prob_same) - np.array(prob_different)
    bell_ineqality_lhs = np.abs(pAB - pAC) - pBC
    if verbose:
        print(f"P(a,b) = {pAB},P(a,c) = {pAC},P(b,c) = {pBC}")
        print(f"Bell's' inequality: {bell_ineqality_lhs} ≤ 1")
        if bell_ineqality_lhs > 1:
            print("Bell's inequality is violated!")
            print("Notice that the quantity is not exactly 1.5 as predicted by theory.")
            print("This is may be due to less number shots or the effects of noise on the QPU.")
        else:
            print("Bell's inequality is not violated due to noise.")
    return results, pAB, pAC, pBC


def bell_singlet_rotated(
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
    circ = bell_singlet(qubit0, qubit1)
    if rotation0 != 0:
        circ.rx(qubit0, rotation0)
    if rotation1 != 0:
        circ.rx(qubit1, rotation1)
    return circ


def bell_singlet(qubit0: Qubit, qubit1: Qubit) -> Circuit:
    """Prepare a Bell singlet state.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        Circuit: the Braket circuit that prepares the Bell single state.
    """
    return Circuit().x(qubit0).x(qubit1).h(qubit0).cnot(qubit0, qubit1)
