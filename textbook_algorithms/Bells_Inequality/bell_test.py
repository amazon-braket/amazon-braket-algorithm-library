import numpy as np
from braket.circuits import Circuit, Observable


def submit_bell_tasks(device, shots=1_000):
    circAB = bell_singlet_rotated(0, 1, 0, np.pi / 3.0)
    circAC = bell_singlet_rotated(0, 1, 0, 2 * np.pi / 3.0)
    circBC = bell_singlet_rotated(0, 1, np.pi / 3.0, 2 * np.pi / 3.0)
    tasks = [device.run(circ, shots=shots) for circ in [circAB, circAC, circBC]]
    return tasks


def get_results(tasks, verbose=True):
    results = [task.result().measurement_probabilities for task in tasks]
    prob_same = [d["00"] + d["11"] for d in results]
    prob_different = [d["01"] + d["10"] for d in results]
    pAB, pAC, pBC = np.array(prob_same) - np.array(prob_different)
    bell_ineqality_lhs = np.abs(pAB - pAC) - pBC
    if verbose == True:
        print(
            f"P(a,b) = {pAB},P(a,c) = {pAC},P(b,c) = {pBC}\nBell's' inequality: {bell_ineqality_lhs} ≤ 1"
        )
        if bell_ineqality_lhs > 1:
            print("Bell's inequality is violated!")
            print(
                "Notice that the quantity is not exactly 1.5 as predicted by theory. This is may be due to less number shots or the effects of noise on the QPU."
            )
        else:
            print("Bell's inequality is not violated due to noise.")
    return results, pAB, pAC, pBC


def bell_singlet_rotated(qubit0, qubit1, rotation0, rotation1):
    c = bell_singlet(qubit0, qubit1)
    if rotation0 != 0:
        c = c.rx(qubit0, rotation0)
    if rotation1 != 0:
        c = c.rx(qubit1, rotation1)
    c.sample(Observable.Z())
    return c


def bell_singlet(qubit0, qubit1):
    return Circuit().x(qubit0).x(qubit1).h(qubit0).cnot(qubit0, qubit1)
