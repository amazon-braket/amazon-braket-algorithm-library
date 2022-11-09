from collections import Counter

import numpy as np
from braket.circuits import Circuit


def deutsch_josza_gate(case: str, num_qubits: int) -> Circuit:
    circ = Circuit()
    if case == "balanced":
        b = np.random.randint(1, 2**num_qubits)
        b_str = format(b, f"0{num_qubits}b")
        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                circ.x(qubit)

        for qubit in range(num_qubits):
            circ.cnot(qubit, num_qubits)

        for qubit in range(len(b_str)):
            if b_str[qubit] == "1":
                circ.x(qubit)

    elif case == "constant":
        output = np.random.randint(2)
        if output == 1:
            circ.x(num_qubits)
    return circ


def dj_algorithm(deutsch_josza_gate: Circuit, num_qubits: int):
    dj_circuit = Circuit()
    dj_circuit.x(num_qubits)
    dj_circuit.h(num_qubits)
    dj_circuit.h(range(num_qubits))
    dj_circuit.add(deutsch_josza_gate)  # add the deutsch_josza_gate
    dj_circuit.h(range(num_qubits))
    return dj_circuit


def marginalize_measurements(measurement_counts: Counter) -> Counter:
    """Remove the last qubit measurement counts.
    Args:
        measurement_counts (Counter): Measurement counts from circuit.
    Returns:
        Counter: Measurement counts from without the last qubit.
    """
    new_dict = {}
    for k, v in measurement_counts.items():
        if k[:-1] not in new_dict:
            new_dict[k[:-1]] = v
        else:
            new_dict[k[:-1]] += v
    return Counter(new_dict)
