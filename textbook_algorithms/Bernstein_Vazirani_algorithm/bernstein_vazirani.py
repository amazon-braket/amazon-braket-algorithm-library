from collections import Counter

import matplotlib.pyplot as plt
from braket.circuits import Circuit


def bernstein_vazirani_circuit(hidden_string: str = "011") -> Circuit:
    """Bernstein–Vazirani circuit on a hidden string.

    Args:
        hidden_string (str): Hidden bitstring. Defaults to "011".

    Returns:
        Circuit: Bernstein–Vazirani circuit
    """

    num_qubits = len(hidden_string)

    bv_circuit = Circuit()
    bv_circuit.h(num_qubits)
    bv_circuit.z(num_qubits)
    bv_circuit.h(range(num_qubits))

    for q in range(num_qubits):
        if hidden_string[q] == "0":
            bv_circuit.i(q)
        else:
            bv_circuit.cnot(q, num_qubits)

    bv_circuit.h(range(num_qubits))
    return bv_circuit


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


def plot_bitstrings(counts: Counter, title: str = None):
    """Plot the measure results

    Args:
        counts (Counter): Measurement counts.
        title (str): Title for the plot.
    """
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.title(title)
    plt.xticks(rotation=90)
