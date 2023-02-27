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
        raise ValueError(f"Number of qubits must be greater than 0. Received {n_qubits}")

    circ = Circuit().i(range(n_qubits))
    rand_output = np.random.randint(0, 2)
    if rand_output == 0:
        circ.i(n_qubits)
    else:
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
        raise ValueError(f"Number of qubits must be greater than 0. Received {n_qubits}")

    # generate a random array of 0s and 1s to figure out where to place x gates
    random_num = np.random.randint(2, size=n_qubits)

    circ = Circuit()

    # place initial x gates
    for qubit in range(n_qubits):
        if random_num[qubit] == 1:
            circ.x(qubit)

    # place cnot gates
    for qubit in range(n_qubits):
        circ.cnot(control=qubit, target=n_qubits)

    # place final x gates
    for qubit in range(n_qubits):
        if random_num[qubit] == 1:
            circ.x(qubit)

    return circ


def deutsch_jozsa_circuit(oracle: Circuit) -> Circuit:
    """Deutsch-Jozsa circuit.

    Args:
        oracle (Circuit): Constant or balanced oracle circuit.

    Returns:
        Circuit: The Deutsch-Jozsa circuit and result types.
    """
    n_qubits = oracle.qubit_count - 1
    circ = Circuit()
    circ.deutsch_jozsa(oracle)
    circ.probability(range(n_qubits))
    return circ


@circuit.subroutine(register=True)
def deutsch_jozsa(oracle: Circuit) -> Circuit:
    """Deutsch-Jozsa subroutine.

    Args:
        oracle (Circuit): Constant or balanced oracle circuit.

    Returns:
        Circuit: The Deutsch-Jozsa circuit.
    """
    n_qubits = oracle.qubit_count - 1
    circ = Circuit()
    circ.h(range(n_qubits))
    circ.x(n_qubits)
    circ.h(n_qubits)
    circ.add_circuit(oracle)
    circ.h(range(n_qubits))
    return circ


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
