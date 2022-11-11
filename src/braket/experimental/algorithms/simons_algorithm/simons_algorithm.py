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
"""
Implementation of the Simon's Algorithm in Amazon Braket
"""
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
from braket.circuits import Circuit, circuit
from braket.devices import Device
from sympy import Matrix
from sympy.core.numbers import Integer, Rational


@circuit.subroutine(register=True)
def simons_oracle(secret_s: str) -> Circuit:
    """
    Quantum circuit implementing a particular oracle for Simon's problem. Details of this
    implementation are explained in an example notebook readable in the Amazon Braket example
    https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/Simons_Algorithm/Simons_Algorithm.ipynb
    Args:
        secret_s (str): the secret string

    Returns:
        Circuit: Circuit object that implements the oracle
    """
    # Find the index of the first 1 in s, to be used as the flag bit
    flag_bit = secret_s.find("1")

    length_string = len(secret_s)

    circ = Circuit()
    # First copy the first n qubits, so that |x>|0> -> |x>|x>
    for i in range(length_string):
        circ.cnot(i, i + length_string)

    # If flag_bit=-1, s is the all-zeros string, and we do nothing else.
    if flag_bit != -1:
        # Now apply the XOR with s whenever the flag bit is 1.
        for index, bit_value in enumerate(secret_s):

            if bit_value not in ["0", "1"]:
                raise ValueError(
                    "Incorrect char '" + bit_value + "' in secret string s:" + secret_s
                )

            # XOR with s whenever the flag bit is 1.
            # In terms of gates, XOR means we apply an X gate only
            # whenever the corresponding bit in s is 1.
            # Applying this X only when the flag qubit is 1 means this is a CNOT gate.
            if bit_value == "1":
                circ.cnot(flag_bit, index + length_string)
    return circ


def simons_algorithm(oracle: Circuit) -> Circuit:
    """
    Build the circuit associated with Simon's algorithm.
    Args:
        oracle (Circuit): The oracle encoding the secret string
    Returns:
        Circuit: circuit associated with Simon's algorithm
    """
    nb_base_qubits = int(oracle.qubit_count / 2)
    return Circuit().h(range(nb_base_qubits)).add(oracle).h(range(nb_base_qubits))


def run_simons_algorithm(
    oracle: Circuit, device: Device, shots: Optional[int] = None
) -> Dict[str, Any]:
    """
    Function to run Simon's algorithm and return the secret string.
    Args:
        oracle (Circuit): The oracle encoding the secret string
        device (Device): Braket device backend
        shots (Optional[int]) : Number of measurement shots (default is None).
            The default number of shots is set to twice the arity of the oracle.
            0 shots results in no measurement.
    Returns:
        Dict[str, Any]: measurements and results from running Simon's algorithm
    """
    circ = simons_algorithm(oracle)
    circ.probability()

    task = device.run(circ, shots=2 * oracle.qubit_count if shots is None else shots)

    result = task.result()

    out = {
        "circuit": circ,
        "task_metadata": result.task_metadata,
        "measurements": result.measurements,
        "measured_qubits": result.measured_qubits,
        "measurement_counts": result.measurement_counts,
        "measurement_probabilities": result.measurement_probabilities,
    }

    return out


def get_simons_algorithm_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get and print classically post-processed results from Simon's algorithm execution
    Args:
        results (Dict[str, Any]): result Dict from run_simons_algorithm()
    Returns:
        Dict[str, Any]: Dict containing the secret string and marginalized output states
    """
    result_s, traced_measurement_counts = _get_secret_string(results["measurement_counts"])

    out = {
        "secret_string": result_s,
        "traced_measurement_counts": traced_measurement_counts,
    }

    print("Result string:", result_s)

    return out


def _get_secret_string(measurement_counts: Counter) -> Tuple[str, Counter]:
    """
    Classical post-processing to recover the secret string.

    The measurement counter contains k bitstrings which correspond to k equations:
        z_k . s = 0 mod 2
    where k.j = k_1*j_1 + ... + k_n*j_n with + the XOR operator
    and s the secret string

    Args:
        measurement_counts (Counter): Counter with all measured bistrings
    Returns:
        Tuple[str, Counter]: the secret string and the marginalized output states
    """
    nb_base_qubits = int(len(list(measurement_counts.keys())[0]) / 2)

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

    # Helper function to treat fractions as modular inverse:
    def mod2(bit: Rational) -> Integer:
        return bit.as_numer_denom()[0] % 2

    # Apply our helper function to the matrix
    final_reduced_matrix = reduced_matrix[0].applyfunc(mod2)

    # Extract the kernel of M from the remaining columns of the last row, when s is nonzero.
    if all(value == 0 for value in final_reduced_matrix[-1, :nb_columns]):
        result_s = "".join(str(e) for e in final_reduced_matrix[-1, nb_columns:])

    # Otherwise, the sub-matrix will be full rank, so just set s=0...0
    else:
        result_s = "0" * nb_rows

    return result_s, traced_results
