from typing import Any, Dict

import numpy as np
from braket.circuits import Circuit
from braket.devices import Device


def qft(num_qubits: int, inverse: bool = False) -> Circuit:
    """Creates the quantum Fourier transform circuit and its inverse.

    Args:
        num_qubits (int): Number of qubits in the circuit
        inverse (bool): If true return the inverse of the circuit. Default is False

    Returns:
        Circuit: Circuit object that implements the quantum Fourier transform or its inverse
    """

    qc = Circuit()
    N = num_qubits - 1

    if not inverse:
        qc.h(N)
        for n in range(1, N + 1):
            qc.cphaseshift(N - n, N, 2 * np.pi / 2 ** (n + 1))

        for i in range(1, N):
            qc.h(N - i)
            for n in range(1, N - i + 1):
                qc.cphaseshift(N - (n + i), N - i, 2 * np.pi / 2 ** (n + 1))
        qc.h(0)

    else:  # The inverse of the quantum Fourier transform
        qc.h(0)
        for i in range(N - 1, 0, -1):
            for n in range(N - i, 0, -1):
                qc.cphaseshift(N - (n + i), N - i, -2 * np.pi / 2 ** (n + 1))
            qc.h(N - i)

        for n in range(N, 0, -1):
            qc.cphaseshift(N - n, N, -2 * np.pi / 2 ** (n + 1))

        qc.h(N)

    return qc


def qft_conditional_add_1(num_qubits: int) -> Circuit:
    """Creates the quantum circuit that conditionally add +1 or -1 using:

    1) The first qubit to control if add 1 or subtract 1: when the first qubit is 0, we add 1 from
    the number, and when the first qubit is 1, we subtract 1 from the number.

    2) The second register with `num_qubits` qubits to save the result.

    Args:
        num_qubits (int): Number of qubits that saves the result.

    Returns:
        Circuit: Circuit object that implements the circuit that conditionally add +1 or -1.
    """

    qc = Circuit()
    qc.add(qft(num_qubits), target=range(1, num_qubits + 1))

    # add \pm 1 with control phase gates
    for i in range(num_qubits):
        qc.cphaseshift01(control=0, target=num_qubits - i, angle=2 * np.pi / 2 ** (num_qubits - i))
        qc.cphaseshift(control=0, target=num_qubits - i, angle=-2 * np.pi / 2 ** (num_qubits - i))

    qc.add(qft(num_qubits, inverse=True), target=range(1, num_qubits + 1))

    return qc


def quantum_walk(n_nodes: int, num_steps: int = 1) -> Circuit:
    """Creates the quantum random walk circuit.

    Args:
        n_nodes (int): The number of nodes in the graph
        num_steps (int): The number of steps for the quantum walk. Default is 1

    Returns:
        Circuit: Circuit object that implements the quantum random walk algorithm

    Raises:
        If `np.log2(n_nodes)` is not an integer, a value error will be raised.
    """

    n = np.log2(n_nodes)  # number of qubits for the graph

    if float(n).is_integer():
        n = int(n)
    else:
        raise ValueError("The number of nodes has to be 2^n for integer n.")

    qc = Circuit()
    for _ in range(num_steps):
        qc.h(0)
        qc.add_circuit(qft_conditional_add_1(n))
        qc.x(0)  # flip the coin after the shift

    return qc


def run_quantum_walk(
    circ: Circuit,
    device: Device,
    shots: int = 1000,
) -> Dict[str, Any]:
    """Function to run quantum random walk algorithm and return measurement counts.

    Args:
        circ (Circuit): Quantum random walk circuit
        device (Device): Braket device backend
        shots (int): Number of measurement shots. Default is 1000.

    Returns:
        Dict[str, Any]: measurements and results from running Quantum Phase Estimation
    """

    # Add results_types
    circ.probability()

    # get total number of qubits
    num_qubits = circ.qubit_count

    # Run the circuit with all zeros input.
    # The query_circuit subcircuit generates the desired input from all zeros.
    task = device.run(circ, shots=shots)

    result = task.result()

    # get output probabilities (see result_types above)
    probs_values = result.values[0]

    # get measurement results
    measurement_counts = result.measurement_counts

    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    quantum_walk_measurement_counts = {}
    for key, val in measurement_counts.items():
        node = int(key[1:][::-1], 2)
        if node in quantum_walk_measurement_counts:
            quantum_walk_measurement_counts[node] += val / shots
        else:
            quantum_walk_measurement_counts[node] = val / shots

    output = {
        "circuit": circ,
        "task_metadata": result.task_metadata,
        "measurements": result.measurements,
        "measured_qubits": result.measured_qubits,
        "measurement_counts": measurement_counts,
        "measurement_probabilities": result.measurement_probabilities,
        "probs_values": probs_values,
        "bitstring_keys": bitstring_keys,
        "quantum_walk_measurement_counts": quantum_walk_measurement_counts,
    }

    return output
