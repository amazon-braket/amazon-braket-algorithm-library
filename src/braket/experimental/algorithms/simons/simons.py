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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from braket.circuits import Circuit
from braket.devices import Device
from braket.tasks import QuantumTask
from sympy import Matrix


def simons_oracle(secret_string: str) -> Circuit:
    """Quantum circuit implementing a particular oracle for Simon's problem.

    In the quantum setting, we first copy the input register into some
    ancillary qubits: |x>|0> -> |x>|x>.

    We then perform the quantum analog of XOR, which means we apply an X gate
    to the kth qubit whenever the kth bit of `string` is 1. However, we only
    apply this X gate when the flag qubit is also |1>. Thus, our X gate becomes
    a CNOT gate between the flag qubit on the input register, and the kth qubit
    on the output.

    Args:
        secret_string (str): the secret string

    Returns:
        Circuit: Circuit object that implements the oracle
    """
    # Find the index of the first 1 in secret_string, to be used as the flag bit
    flag_bit = secret_string.find("1")

    length_string = len(secret_string)

    circ = Circuit()
    # First copy the first n qubits, so that |x>|0> -> |x>|x>
    for i in range(length_string):
        circ.cnot(i, i + length_string)

    # If flag_bit=-1, secret_string is the all-zeros string, and we do nothing else.
    if flag_bit != -1:
        # Now apply the XOR with secret_string whenever the flag bit is 1.
        for index, bit_value in enumerate(secret_string):

            if bit_value not in ["0", "1"]:
                raise ValueError(
                    "Incorrect char '" + bit_value + "' in the secret string:" + secret_string
                )

            # XOR with secret_string whenever the flag bit is 1.
            # In terms of gates, XOR means we apply an X gate only
            # whenever the corresponding bit in secret_string is 1.
            # Applying this X only when the flag qubit is 1 means this is a CNOT gate.
            if bit_value == "1":
                circ.cnot(flag_bit, index + length_string)
    return circ


def simons_algorithm(oracle: Circuit) -> Circuit:
    """Build the circuit associated with Simon's algorithm.

    Args:
        oracle (Circuit): The oracle encoding the secret string

    Returns:
        Circuit: circuit associated with Simon's algorithm
    """
    nb_base_qubits = int(oracle.qubit_count / 2)
    return Circuit().h(range(nb_base_qubits)).add(oracle).h(range(nb_base_qubits))


def run_simons_algorithm(
    oracle: Circuit, device: Device, shots: Optional[int] = None
) -> QuantumTask:
    """Function to run Simon's algorithm and return the secret string.

    Args:
        oracle (Circuit): The oracle encoding the secret string
        device (Device): Braket device backend
        shots (Optional[int]) : Number of measurement shots (default is None).
            The default number of shots is set to twice the arity of the oracle.
            shots must be a strictly positive integer.

    Returns:
        QuantumTask: Task for Simon's algorithm.
    """
    if shots is None:
        shots = 2 * oracle.qubit_count
    if shots <= 0:
        raise ValueError("shots must be a strictly positive integer.")

    circ = simons_algorithm(oracle)
    circ.probability()

    task = device.run(circ, shots=shots)

    return task


def get_simons_algorithm_results(task: QuantumTask) -> Dict[str, Any]:
    """Get and print classically post-processed results from Simon's algorithm execution.

    Args:
        task (QuantumTask): Task for Simon's algorithm.

    Returns:
        Dict[str, Any]: Dict containing the secret string and marginalized output states
    """

    task_result = task.result()

    results = {
        "measurements": task_result.measurements,
        "measured_qubits": task_result.measured_qubits,
        "measurement_counts": task_result.measurement_counts,
        "measurement_probabilities": task_result.measurement_probabilities,
    }
    result_string, traced_measurement_counts = _get_secret_string(results["measurement_counts"])

    output = {
        "secret_string": result_string,
        "traced_measurement_counts": traced_measurement_counts,
    }

    print("Result string:", result_string)

    return output


def _get_secret_string(measurement_counts: Counter) -> Tuple[str, Counter]:
    """Classical post-processing to recover the secret string.

    The measurement counter contains k bitstrings which correspond to k equations:
        z_k . s = 0 mod 2
    where k.j = k_1*j_1 + ... + k_n*j_n with + the XOR operator
    and s the secret string

    Args:
        measurement_counts (Counter): Counter with all measured bistrings

    Returns:
        Tuple[str, Counter]: the secret string and the marginalized output states
    """
    nb_base_qubits = len(list(measurement_counts.keys())[0]) // 2

    traced_results = Counter()
    for bitstring, count in measurement_counts.items():
        traced_results.update({bitstring[:nb_base_qubits]: count})

    if len(traced_results.keys()) < nb_base_qubits:
        raise RuntimeError(
            "System will be underdetermined. Minimum "
            + str(nb_base_qubits)
            + " bistrings needed, but only "
            + str(len(traced_results.keys()))
            + " returned. Please rerun Simon's algorithm."
        )
    bitstring_matrix = np.vstack([np.array([*key], dtype=int) for key in traced_results]).T
    nb_rows, nb_columns = bitstring_matrix.shape

    # Construct the augmented matrix
    augmented_matrix = Matrix(np.hstack([bitstring_matrix, np.eye(nb_rows, dtype=int)]))

    # Perform row reduction, working modulo 2. We use the iszerofunc property of rref
    # to perform the Gaussian elimination over the finite field.
    reduced_matrix = augmented_matrix.rref(iszerofunc=lambda x: x % 2 == 0)

    # Apply helper function to the matrix to treat fractions as modular inverse:
    final_reduced_matrix = reduced_matrix[0].applyfunc(lambda x: x.as_numer_denom()[0] % 2)

    # Extract the kernel of M from the remaining columns of the last row, when s is nonzero.
    if all(value == 0 for value in final_reduced_matrix[-1, :nb_columns]):
        result_string = "".join(str(e) for e in final_reduced_matrix[-1, nb_columns:])
    else:  # Otherwise, the sub-matrix will be full rank, so just set s=0...0
        result_string = "0" * nb_rows

    return result_string, traced_results
