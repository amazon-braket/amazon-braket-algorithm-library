from typing import Tuple

from braket.circuits import Circuit, circuit


def grovers_search(
    oracle: Circuit, n_qubits: int, n_reps: int = 1, decompose_ccnot: bool = False
) -> Circuit:
    """Generate Grover's circuit for a target solution and oracle.

    Args:
        oracle (Circuit): Oracle circuit for a solution.
        n_qubits (int): Number of data qubits.
        n_reps (int): Number of repititions for amplification. Defaults to 1.
        decompose_ccnot (bool): To decompose CCNOT (Toffoli) gate in the circuit.

    Returns:
        Circuit: Grover's circuit
    """
    grover_circ = Circuit().h(range(n_qubits))
    for _ in range(n_reps):
        grover_circ.add(oracle)
        amplification = amplify(n_qubits, decompose_ccnot)
        grover_circ.add(amplification)
    grover_circ.probability(range(n_qubits))
    return grover_circ


def build_oracle(solution: str, decompose_ccnot: bool = False) -> Circuit:
    """Oracle circuit of a given solution.

    Args:
        solution (str): Target solution (e.g., '010')
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gate in the circuit.

    Returns:
        Circuit: Oracle circuit
    """
    x_idx = [i for i, s in enumerate(solution) if s == "0"]

    circ = Circuit()
    n_qubit = len(solution)
    mcz = multi_control_z(n_qubit, decompose_ccnot)
    circ.x(x_idx).add_circuit(mcz).x(x_idx)
    return circ


def amplify(n_qubits: int, decompose_ccnot: bool) -> Circuit:
    """Perform a single iteration of amplitude amplification.

    Args:
        n_qubits (int): Number of data qubits.
        decompose_ccnot (bool): Whether to decompose CCNOT (Toffoli) gate in the circuit.

    Returns:
        Circuit: Amplification circuit.
    """
    oracle = build_oracle(n_qubits * "0", decompose_ccnot)
    circ = Circuit()
    circ.h(range(n_qubits))
    circ.add_circuit(oracle)
    circ.h(range(n_qubits))
    return circ


@circuit.subroutine(register=True)
def ccnot_decomposed(control_1: int, control_2: int, target: int) -> Circuit:
    """Build CCNOT (Toffoli gate) from H, CNOT, T, Ti.

    Args:
        control_1 (int): control qubit 1 of CCNot gate
        control_2 (int): control qubit 2 of CCNot gate
        target (int): target qubit of CCNot gate

    Returns:
        Circuit: CCNot circuit.
    """
    qubit_0, qubit_1 = control_1, control_2
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
    is_outermost_call: bool = True,
) -> Tuple[Circuit, int]:
    """Recursive constructor of a multi-contol Not circuit (generalized Toffoli gate).
    Ref: https://arxiv.org/abs/1904.01671

    Args:
        n_qubit (int): Number of qubits.
        decompose_ccnot (bool): To decompose CCNOT (Toffoli) gate in the circuit.
        is_outermost_call (bool):  Whether the call is the outermost call from external functions.

    Returns:
        Tuple[Circuit, int]:  the multi-contol Not circuit and the number of ancilla in the circuit
    """
    if n_qubit == 1:
        n_ancilla = 1
        circ = Circuit().cnot(0, 1)
        return circ, n_ancilla
    elif n_qubit == 2:
        n_ancilla = 1
        if decompose_ccnot:
            circ = Circuit().ccnot_decomposed(0, 1, 2)
        else:
            circ = Circuit().ccnot(0, 1, 2)
        return circ, n_ancilla
    else:
        n_ancilla = 0
        nq1 = n_qubit // 2
        nq2 = n_qubit - nq1

        circ1, na1 = multi_control_not_constructor(nq1, decompose_ccnot, is_outermost_call=False)
        circ2, na2 = multi_control_not_constructor(nq2, decompose_ccnot, is_outermost_call=False)

        circ = Circuit()

        qd1 = list(range(nq1))
        qa1 = list(range(n_qubit + n_ancilla, n_qubit + n_ancilla + na1))
        circ.add_circuit(circ1, target=qd1 + qa1)
        n_ancilla += na1

        qd2 = list(range(nq1, nq1 + nq2))
        qa2 = list(range(n_qubit + n_ancilla, n_qubit + n_ancilla + na2))
        circ.add_circuit(circ2, target=qd2 + qa2)
        n_ancilla += na2

        q0, q1, q2 = qa1[-1], qa2[-1], n_qubit + n_ancilla
        if decompose_ccnot:
            circ.ccnot_decomposed(q0, q1, q2)
        else:
            circ.ccnot(q0, q1, q2)
        n_ancilla += 1

        if is_outermost_call:
            circ.add_circuit(circ2.adjoint(), target=qd2 + qa2)
            circ.add_circuit(circ1.adjoint(), target=qd1 + qa1)

        return circ, n_ancilla


def multi_control_not(n_qubit: int, decompose_ccnot: bool) -> Circuit:
    """Multi-control Not circuit.

    Args:
        n_qubit (int): Number of qubits.
        decompose_ccnot (bool): To decompose CCNOT (Toffoli) gate in the circuit.

    Returns:
        Circuit:  multi-contol Not circuit
    """
    mcx_circ, _ = multi_control_not_constructor(n_qubit, decompose_ccnot, is_outermost_call=True)
    return mcx_circ


def multi_control_z(n_qubit: int, decompose_ccnot: bool) -> Circuit:
    """Multi-control Z circuit.

    Args:
        n_qubit (int): Number of qubits.
        decompose_ccnot (bool): To decompose CCNOT (Toffoli) gate in the circuit.

    Returns:
        Circuit:  multi-contol Z circuit
    """
    mcz_circ = multi_control_not(n_qubit, decompose_ccnot)
    z_target = mcz_circ.qubit_count - 1

    circ = Circuit()
    circ.x(z_target).h(z_target).add_circuit(mcz_circ).h(z_target).x(z_target)

    return circ
