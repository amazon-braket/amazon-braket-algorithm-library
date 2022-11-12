# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from collections import Counter

import matplotlib.pyplot as plt
from braket.circuits import Circuit, Observable


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

    for q in range(num_qubits):
        bv_circuit.sample(Observable.Z(), q)
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


def plot_bitstrings(counts: Counter, title: str = None) -> None:
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
