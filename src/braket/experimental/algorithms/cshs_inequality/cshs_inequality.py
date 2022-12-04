from collections import Counter
from typing import List, Tuple

import numpy as np
from braket.circuits import Circuit, Observable, Qubit
from braket.devices import Device
from braket.tasks import QuantumTask


def submit_cshs_tasks(
    device: Device,
    shots: int = 1_000,
    qubit0: Qubit = 0,
    qubit1: Qubit = 1,
    a: float = 0,
    a_: float = 2 * np.pi / 8,
    b: float = np.pi / 8,
    b_: float = 3 * np.pi / 8,
) -> List[QuantumTask]:
    """Submit four CSHS circuits to a device.

    Args:
        device (Device): Quantum device or simulator.
        shots (int): Number of shots. Defaults to 1_000.
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        List[QuantumTask]: List of quantum tasks.
    """

    print("a:", a)
    print("a_:", a_)
    print("b:", b)
    print("b_:", b_)

    circ_ab = bell_singlet_rotated_basis(qubit0, qubit1, a, b)
    circ_ab_ = bell_singlet_rotated_basis(qubit0, qubit1, a, b_).h(qubit1)
    circ_a_b = bell_singlet_rotated_basis(qubit0, qubit1, a_, b).h(qubit0)
    circ_a_b_ = bell_singlet_rotated_basis(qubit0, qubit1, a_, b_).h(qubit0).h(qubit1)
    print("circ_ab\n:", circ_ab)
    print("circ_ab_\n:", circ_ab_)
    print("circ_a_b\n:", circ_a_b)
    print("circ_a_b_\n:", circ_a_b_)

    tasks = [device.run(circ, shots=shots) for circ in [circ_ab, circ_ab_, circ_a_b, circ_a_b_]]
    return tasks


def get_cshs_results(
    tasks: List[QuantumTask], verbose: bool = True
) -> Tuple[List[Counter[float]], float, float, float]:
    """Return Bell task results after post-processing.

    Args:
        tasks (List[QuantumTask]): List of quantum tasks.
        verbose (bool): Controls printing of the inequality result. Defaults to True.

    Returns:
        Tuple[List[Counter[float]], float, float, float]: results, pAB, pAC, pBC
    """
    results = [task.result().measurement_probabilities for task in tasks]
    # prob_same = [d["00"] + d["11"] for d in results]
    print("measurement_probabilities:", results)
    prob_same = [(d["00"] if "00" in d else 0) + (d["11"] if "11" in d else 0) for d in results]

    prob_different = [
        (d["01"] if "01" in d else 0) + (d["10"] if "10" in d else 0) for d in results
    ]
    print("prob_same:", prob_same)
    print("prob_different:", prob_different)
    # Bell probabilities
    E_ab, E_ab_, E_a_b, E_a_b_ = np.array(prob_same) - np.array(prob_different)
    cshs_value = E_ab - E_ab_ + E_a_b + E_a_b_
    print("cshs_value:", cshs_value)
    cshs_ineqality_lhs = np.abs(cshs_value)
    if verbose:
        print(
            f"E(a,b) = {E_ab},E(a,b') = {E_ab_}, E(a',b) = {E_a_b}, E(a',b') = {E_a_b_}\nBell's' inequality: {cshs_ineqality_lhs} ≤ 2"
        )
        if cshs_ineqality_lhs > 1:
            print("CSHS inequality is violated!")
            print(
                "Notice that the quantity is not exactly 2.82 as predicted by theory."
                "This is may be due to less number shots or the effects of noise on the QPU."
            )
        else:
            print("CSHS inequality is not violated due to noise.")
    return cshs_value, cshs_ineqality_lhs, results, E_ab, E_ab_, E_a_b, E_a_b_


def bell_singlet_rotated_basis(
    qubit0: Qubit, qubit1: Qubit, rotation0: float, rotation1: float
) -> Circuit:
    """Prepare a Bell singlet state in a Ry-rotated meaurement basis.

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
        # circ.rx(qubit0, rotation0)
        circ.ry(qubit0, rotation0)
    if rotation1 != 0:
        # circ.rx(qubit1, rotation1)
        circ.ry(qubit1, rotation1)
    # circ.sample(Observable.Z())
    return circ


def bell_singlet(qubit0: Qubit, qubit1: Qubit) -> Circuit:
    """Prepare a Bell singlet state.

    Args:
        qubit0 (Qubit): First qubit.
        qubit1 (Qubit): Second qubit.

    Returns:
        Circuit: the Braket circuit that prepares the Bell single state.
    """
    # return Circuit().x(qubit0).x(qubit1).h(qubit0).cnot(qubit0, qubit1)
    return Circuit().h(qubit0).cnot(qubit0, qubit1)
