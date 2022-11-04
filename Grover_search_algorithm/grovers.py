import numpy as np
import matplotlib.pyplot as plt

from braket.circuits import Circuit, circuit


def grover(item, oracles, n_qubits=3, n_reps=1):
    """ Put together individual modules of Grover algorithm

    Args:
        item (str): target solution (e.g., '010')
        oracle (Dict[str, Circuit]): oracle implementations for each solution
            as quantum circuits
        n_qubits (int): number of qubits
        n_reps (int): number of repetitions for amplification
    """
    grover_circ = Circuit().h(np.arange(n_qubits))
    for _ in range(n_reps):
        grover_circ.add(oracles[item])
        amplification = amplify(oracles)
        grover_circ.add(amplification)
    grover_circ.probability()
    return grover_circ


def amplify(oracles, n_qubits=3):
    """ Amplitude amplification. amplify is a function that does a single
    iteration of amplitude amplification shown in Figure 1 of Ref[1].

    Args:
        oracle (Dict[str, Circuit]): oracle implementations for each solution
            as quantum circuits.
        n_qubits (int): number of qubits
    """
    circ = Circuit()
    circ.h(np.arange(n_qubits))
    circ.add_circuit(oracles['000'])
    circ.h(np.arange(n_qubits))
    return circ


def oracles():
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


def plot_bitstrings(result):
    """ Plot the measure results

    Args:
        result (GateModelQuantumTaskResult): Execution result from a Braket
            device
    """
    num_qubits = len(result.measured_qubits)
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [
        format_bitstring.format(ii) for ii in range(2 ** num_qubits)
        ]

    # plot probabalities
    probs_values = result.values[0]
    plt.bar(bitstring_keys, probs_values)
    plt.xlabel("bitstrings")
    plt.ylabel("probability")
    plt.xticks(rotation=90)


@circuit.subroutine(register=True)
def ccz(targets=[0, 1, 2]):
    """
    implementation of three-qubit gate CCZ
    The quantum circuit for each marked state is based on Table 1 of Ref [1].

    Args:
        targets (List[int]): target qubits of ccz gates
    """
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        dtype=complex,
    )
    return Circuit().unitary(
        matrix=matrix, targets=targets, display_name="CCZ"
        )


@circuit.subroutine(register=True)
def CCNot(controls=[0, 1], target=2):
    """
    build CCNOT (Toffoli gate) from H, CNOT, T, Ti

    Args:
        controls (List[int]): control qubits of CCNot gates
        target (int): target qubit of CCNot gates
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
def ccz_ionq(controls=[0, 1], target=2):
    """
    build CCZ from H and CCNOT

    Args:
        controls (List[int]): control qubits of CCNot gates
        target (int): target qubit of CCNot gates
    """
    return Circuit().h(target).CCNot(controls, target).h(target)
