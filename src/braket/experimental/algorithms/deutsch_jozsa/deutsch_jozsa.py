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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, circuit
from braket.tasks import QuantumTask


def constant_oracle(n_qubits: int) -> Circuit:
    """Constant oracle circuit.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        Circuit: Constant oracle circuit
    """
    if n_qubits < 1:
        raise ValueError(f"Number of qubits must be greater than 0. Recieved {n_qubits}")

    circ = Circuit().i(range(n_qubits))
    rand_output = np.random.randint(0, 2)
    if rand_output == 0:
        circ.i(n_qubits)
    elif rand_output == 1:
        circ.x(n_qubits)
    return circ


def balanced_oracle(n_qubits: int) -> Circuit:
    """Balanced oracle circuit.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        Circuit: Balanced oracle circuit
    """
    if n_qubits < 1:
        raise ValueError(f"Number of qubits must be greater than 0. Recieved {n_qubits}")

    # generate a random array of 0s and 1s to figure out where to place x gates
    random_num = np.random.randint(2, size=n_qubits)

    circuit = Circuit()

    # place initial x gates
    for qubit in range(n_qubits):
        if random_num[qubit] == 1:
            circuit.x(qubit)

    # place cnot gates
    for qubit in range(n_qubits):
        circuit.cnot(control=qubit, target=n_qubits)

    # place final x gates
    for qubit in range(n_qubits):
        if random_num[qubit] == 1:
            circuit.x(qubit)

    return circuit


def deutsch_jozsa_circuit(oracle: Circuit, n_qubits: int) -> Circuit:
    """Deutsch-Jozsa circuit.

    Args:
        oracle (Circuit): Constant or balanced oracle circuit.
        n_qubits (int): Number of qubits.

    Returns:
        Circuit: The Deutsch-Jozsa circuit and result types.
    """
    circuit = Circuit()
    circuit.deutsch_jozsa(oracle, n_qubits)
    circuit.probability(range(n_qubits))
    return circuit


@circuit.subroutine(register=True)
def deutsch_jozsa(oracle: Circuit, n_qubits: int) -> Circuit:
    """Deutsch-Jozsa subroutine.

    Args:
        oracle (Circuit): Constant or balanced oracle circuit.
        n_qubits (int): Number of qubits.

    Returns:
        Circuit: The Deutsch-Jozsa circuit.
    """
    circuit = Circuit()
    circuit.h(range(n_qubits))
    circuit.x(n_qubits).h(n_qubits)
    circuit.add_circuit(oracle, range(n_qubits + n_qubits))
    circuit.h(range(n_qubits))
    return circuit


def get_deutsch_jozsa_results(task: QuantumTask) -> Dict[str, float]:
    """Return the probabilities and corresponding bitstrings.

    Args:
        task (QuantumTask): Quantum task to process.

    Returns:
        Dict[str, float]: Results as a dictionary of bitstrings
    """
    probabilities = task.result().result_types[0].value
    probabilities = np.round(probabilities, 10)  # round off floating-point errors
    num_qubits = int(np.log2(len(probabilities)))
    binary_strings = [format(i, "b").zfill(num_qubits) for i in range(2**num_qubits)]
    return dict(zip(binary_strings, probabilities))


def plot_bitstrings(probabilities: Dict[str, float], title: str = None) -> None:
    """Plot the measurement results.

    Args:
        probabilities (Dict[str, float]): Measurement probabilities.
        title (str): Title for the plot.
    """
    plt.bar(probabilities.keys(), probabilities.values())
    plt.xlabel("bitstrings")
    plt.ylabel("probabilities")
    plt.title(title)
    plt.xticks(rotation=90)
