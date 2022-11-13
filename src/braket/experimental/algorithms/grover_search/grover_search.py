from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, QubitSetInput, circuit
from braket.tasks import GateModelQuantumTaskResult


def grover_search(
    target_bitstring: str, 
    n_qubits: int = 3, 
    n_reps: int = 1,
    decompose_ccnot: bool = False
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
    oracle = get_oracle(target_bitstring, decompose_ccnot)
    n_qubits = len(target_bitstring)

    grover_circ = Circuit().h(np.arange(n_qubits))
    for _ in range(n_reps):
        grover_circ.add(oracle)
        amplification = amplify(n_qubits, decompose_ccnot)
        grover_circ.add(amplification)
    grover_circ.probability(range(n_qubits))
    return grover_circ


def amplify(n_qubits: int, decompose_ccnot: bool) -> Circuit:
    """
    Perform a single iteration of amplitude amplification.

    Args:
        oracles (Dict[str, Circuit]): oracle implementations for each solution as quantum circuits.
        n_qubits (int): Number of qubits.

    Returns:
        Circuit: Amplification circuit.
    """
    oracle = get_oracle(n_qubits * "0", decompose_ccnot)
    circ = Circuit()
    circ.h(np.arange(n_qubits))
    circ.add_circuit(oracle)
    circ.h(np.arange(n_qubits))
    return circ


def plot_bitstrings(probabilities) -> None:
    """Plot the measure results.

    Args:
        result (GateModelQuantumTaskResult): Result from a Braket device.
    """
    num_qubits = int(np.log2(len(probabilities)))
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    plt.bar(bitstring_keys, probabilities)
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


def multi_control_not_constructor(
        n_qubit: int, 
        decompose_ccnot: bool, 
        outermost_call: bool = True,
    ):
    if n_qubit==1:
        n_ancilla = 1
        circ = Circuit().cnot(0, 1)
        return circ, n_ancilla
    elif n_qubit==2:
        n_ancilla = 1
        if decompose_ccnot:
            circ = Circuit().CCNot([0,1],2)
        else:
            circ = Circuit().ccnot(0,1,2)
        return circ, n_ancilla
    else:
        n_ancilla = 0
        nq1 = n_qubit // 2
        nq2 = n_qubit - nq1 
        
        circ1, na1 = multi_control_not_constructor(nq1, decompose_ccnot, outermost_call=False)
        circ2, na2 = multi_control_not_constructor(nq2, decompose_ccnot, outermost_call=False)
        
        circ = Circuit()

        qd1 = list(range(0,nq1))
        qa1 = list(range(n_qubit+n_ancilla, n_qubit+n_ancilla+na1))
#         print(qd1)
#         print(qa1)        
        circ.add_circuit(circ1, target=qd1+qa1) 
        n_ancilla += na1
        
#         print('-' * 20)
        
        qd2 = list(range(nq1,nq1+nq2))
        qa2 = list(range(n_qubit+n_ancilla, n_qubit+n_ancilla+na2))
#         print(qd2)
#         print(qa2)        
        circ.add_circuit(circ2, target=qd2+qa2) 
        n_ancilla += na2

        q0, q1, q2 = qa1[-1], qa2[-1], n_qubit+n_ancilla
        if decompose_ccnot:
            circ.CCNot([q0,q1],q2)
        else:
            circ.ccnot(q0, q1, q2)
        n_ancilla += 1
        
#         print('='*50)
        if outermost_call:
            circ.add_circuit(circ2.adjoint(), target=qd2+qa2)
            circ.add_circuit(circ1.adjoint(), target=qd1+qa1) 
 
        return circ, n_ancilla

    
def multi_control_not(n_qubit: int, decompose_ccnot: bool):
    mcx_circ, _ = multi_control_not_constructor(n_qubit, decompose_ccnot, outermost_call=True)
    return mcx_circ

    
def multi_control_z(n_qubit: int, decompose_ccnot: bool):    
    mcz_circ = multi_control_not(n_qubit, decompose_ccnot)
    z_target = mcz_circ.qubit_count - 1

    circ = Circuit()
    circ.x(z_target).h(z_target).add_circuit(mcz_circ).h(z_target).x(z_target)
    
    return circ


def get_oracle(solution: str, decompose_ccnot: bool):
    x_idx = [i for i,s in enumerate(solution) if s=='0']
    
    circ = Circuit()
    n_qubit = len(solution)
    mcz = multi_control_z(n_qubit, decompose_ccnot)
    circ.x(x_idx).add_circuit(mcz).x(x_idx)
    return circ



