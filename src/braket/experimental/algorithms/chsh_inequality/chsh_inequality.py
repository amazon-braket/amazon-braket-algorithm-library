from collections import Counter
from typing import List, Tuple

import numpy as np
from braket.circuits import Circuit, Qubit
from braket.devices import Device
from braket.tasks import QuantumTask

from braket.experimental.algorithms.bells_inequality.bells_inequality import (
    bell_singlet_rotated_basis,
)


def create_chsh_inequality_circuits(
    qubit0: Qubit = 0,
    qubit1: Qubit = 1,
    a: float = 0,
    b: float = np.pi / 4,
    c: float = 3 * np.pi / 4,
    d: float = np.pi / 2,
) -> List[Circuit]:
    """Create the four circuits for CHSH inequality. Default angles will give maximum violation of
    the inequality.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.
        a (Float): First basis rotation angle for first qubit
        b (Float): Second basis rotation angle for first qubit
        c (Float): First basis rotation angle for second qubit
        d (Float): Second basis rotation angle for second qubit

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    circ_ab = bell_singlet_rotated_basis(qubit0, qubit1, a, b)
    circ_ac = bell_singlet_rotated_basis(qubit0, qubit1, a, c)
    circ_db = bell_singlet_rotated_basis(qubit0, qubit1, d, b)
    circ_dc = bell_singlet_rotated_basis(qubit0, qubit1, d, c)
    return [circ_ab, circ_ac, circ_db, circ_dc]


def run_chsh_inequality(
    circuits: List[Circuit],
    device: Device,
    shots: int = 1_000,
) -> List[QuantumTask]:

    """Submit four CHSH circuits to a device.

    Args:
        circuits (List[Circuit]): Four CHSH inequality circuits to run.
        device (Device): Quantum device or simulator.
        shots (int): Number of shots. Defaults to 1_000.

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """
    tasks = [device.run(circ, shots=shots) for circ in circuits]
    return tasks


def get_chsh_results(
    tasks: List[QuantumTask], verbose: bool = True
) -> Tuple[List[Counter[float]], float, float, float]:
    """Return CHSH task results after post-processing.

    Args:
        tasks (List[QuantumTask]): List of quantum tasks.
        verbose (bool): Controls printing of the inequality result. Defaults to True.

    Returns:
        Tuple[List[Counter[float]], float, float, float]: results, pAB, pAC, pBC
    """
    results = [task.result().result_types[0].value for task in tasks]
    prob_same = np.array([d[0] + d[3] for d in results])  # 00 and 11 states
    prob_different = np.array([d[1] + d[2] for d in results])  # 01 and 10 states
    pAB, pAC, pDB, pDC = np.array(prob_same) - np.array(prob_different)
    chsh_value = np.abs(pAB - pAC) + np.abs(pDB + pDC)

    if verbose:
        print(f"P(a,b) = {pAB}, P(a,c) = {pAC}, P(d,b) = {pDB}, P(d,c) = {pDC}")
        print(f"\nCHSH inequality: {chsh_value} ≤ 2")

        if chsh_value > 2:
            print("CHSH inequality is violated!")
            print(
                "Notice that the quantity may not be exactly as predicted by Quantum theory. "
                "This is may be due to finite shots or the effects of noise on the QPU."
            )
        else:
            print("CHSH inequality is not violated.")
    return chsh_value, results, pAB, pAC, pDB, pDC
