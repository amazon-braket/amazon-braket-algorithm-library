from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, QubitSetInput, circuit
from braket.tasks import GateModelQuantumTaskResult


def grover_search(
    target_bitstring: str, 
    oracles: Dict[str, Circuit], 
    n_qubits: int = 3, 
    n_reps: int = 1
) -> Circuit:
    """Generate Grover's circuit for a target solution and oracle.

    Args:
        target_bitstring (str): Target solution (e.g., '010')
        oracles (Dict[str, Circuit]): Oracle implementations for each solution
            as quantum circuits
        n_qubits (int): Number of qubits. Defaults to 3.
        n_reps (int): Number of repititions for amplification. Defaults to 1.

    Returns:
        Circuit: Grover's circuit
    """

    grover_circ = Circuit().h(np.arange(n_qubits))
    for _ in range(n_reps):
        grover_circ.add(oracles[target_bitstring])
        amplification = amplify(oracles)
        grover_circ.add(amplification)
    grover_circ.probability()
    return grover_circ


def amplify(oracles: Dict[str, Circuit], n_qubits: int = 3) -> Circuit:
    """
    Perform a single iteration of amplitude amplification.

    Args:
        oracles (Dict[str, Circuit]): oracle implementations for each solution as quantum circuits.
        n_qubits (int): Number of qubits. Defaults to 3.

    Returns:
        Circuit: Amplification circuit.
    """
    circ = Circuit()
    circ.h(np.arange(n_qubits))
    circ.add_circuit(oracles[n_qubits * "0"])
    circ.h(np.arange(n_qubits))
    return circ


def get_oracles() -> Dict[str, Circuit]:
    return {
        "000": Circuit().x([0, 1, 2]).ccz(targets=[0, 1, 2]).x([0, 1, 2]),
        "001": Circuit().x([0, 1]).ccz(targets=[0, 1, 2]).x([0, 1]),
        "010": Circuit().x([0, 2]).ccz(targets=[0, 1, 2]).x([0, 2]),
        "011": Circuit().x([0]).ccz(targets=[0, 1, 2]).x([0]),
        "100": Circuit().x([1, 2]).ccz(targets=[0, 1, 2]).x([1, 2]),
        "101": Circuit().x([1]).ccz(targets=[0, 1, 2]).x([1]),
        "110": Circuit().x([2]).ccz(targets=[0, 1, 2]).x([2]),
        "111": Circuit().ccz(targets=[0, 1, 2]),
    }


def plot_bitstrings(result: GateModelQuantumTaskResult) -> None:
    """Plot the measure results.

    Args:
        result (GateModelQuantumTaskResult): Result from a Braket device.
    """
    num_qubits = len(result.measured_qubits)
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    probs_values = result.values[0]
    plt.bar(bitstring_keys, probs_values)
    plt.xlabel("bitstrings")
    plt.ylabel("probability")
    plt.xticks(rotation=90)


@circuit.subroutine(register=True)
def CCNot(controls: QubitSetInput = [0, 1], target: int = 2) -> Circuit:
    """
    Build CCNOT (Toffoli gate) from H, CNOT, T, Ti.

    Args:
        controls (QubitSetInput): control qubits of CCNot gates
        target (int): target qubit of CCNot gates

    Returns:
        Circuit: CCNot circuit.
    """
    qubit_0, qubit_1 = controls
    circ = Circuit()
    circ.h(target)
    circ.cnot(qubit_1, target)
    circ.ti(target)
    circ.cnot(qubit_0, target)
    circ.t(target)
    circ.cnot(qubit_1, target)
    circ.ti(target)
    circ.cnot(qubit_0, target)
    circ.t(target)
    circ.h(target)
    circ.t(qubit_1)
    circ.cnot(qubit_0, qubit_1)
    circ.t(qubit_0)
    circ.ti(qubit_1)
    circ.cnot(qubit_0, qubit_1)
    return circ


@circuit.subroutine(register=True)
def ccz(targets: QubitSetInput = [0, 1, 2]) -> Circuit:
    """
    Build CCZ from H and CCNOT.

    Args:
        targets (QubitSetInput): Target qubits

    Returns:
        Circuit: CCZ circuit.
    """
    return Circuit().h(targets[2]).CCNot([targets[0], targets[1]], targets[2]).h(targets[2])
