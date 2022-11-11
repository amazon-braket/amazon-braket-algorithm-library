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
from typing import Any, Dict, Optional

import numpy as np
from braket.circuits import Circuit, circuit
from braket.devices import Device
from sympy import Matrix


@circuit.subroutine(register=True)
def simons_oracle(secret_s: str) -> Circuit:
    """
    Quantum circuit implementing a particular oracle for Simon's problem. Details of this implementation are
    explained in the Simons Algorithm examples notebook:
    https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/Simons_Algorithm/Simons_Algorithm.ipynb
    Args:
        secret_s (str): the secret string

    Returns:
        Circuit: Circuit object that implements the oracle
    """
    # Find the index of the first 1 in s, to be used as the flag bit
    flag_bit = secret_s.find("1")

    n = len(secret_s)

    circ = Circuit()
    # First copy the first n qubits, so that |x>|0> -> |x>|x>
    for i in range(n):
        circ.cnot(i, i + n)

    # If flag_bit=-1, s is the all-zeros string, and we do nothing else.
    if flag_bit != -1:
        # Now apply the XOR with s whenever the flag bit is 1.
        for index, bit_value in enumerate(secret_s):

            if bit_value not in ["0", "1"]:
                raise Exception(
                    "Incorrect char '" + bit_value + "' in secret string s:" + secret_s
                )

            # XOR with s whenever the flag bit is 1.
            # In terms of gates, XOR means we apply an X gate only whenever the corresponding bit in s is 1.
            # Applying this X only when the flag qubit is 1 means this is a CNOT gate.
            if bit_value == "1":
                circ.cnot(flag_bit, index + n)
    return circ


def simons_algorithm(oracle: Circuit) -> Circuit:
    n = int(oracle.qubit_count / 2)
    return Circuit().h(range(n)).add(oracle).h(range(n))


def run_simons_algorithm(
    oracle: Circuit, device: Device, shots: Optional[int] = None
) -> Dict[str, Any]:
    """
    Function to run Simon's algorithm and return the secret string.
    Args:
        oracle (Circuit): The oracle encoding the secret string
        device (Device): Braket device backend
        shots (int) : Number of measurement shots (default is None).
            The default number of shots is set to twice the arity of the oracle.
            0 shots results in no measurement.
    Returns:
        Dict[str, Any]: measurements and results from running Simon's algorithm
    """
    circuit = simons_algorithm(oracle)
    circuit.probability()

    task = device.run(circuit, shots=2 * oracle.qubit_count if shots is None else shots)

    result = task.result()

    out = {
        "circuit": circuit,
        "task_metadata": result.task_metadata,
        "measurements": result.measurements,
        "measured_qubits": result.measured_qubits,
        "measurement_counts": result.measurement_counts,
        "measurement_probabilities": result.measurement_probabilities,
    }

    return out

def get_simons_algorithm_results(results: Dict[str, Any]) -> Dict[str, Any]:

    result_s, traced_measurement_counts = _get_secret_string(results["measurement_counts"])

    out = {
        "secret_string": result_s,
        "traced_measurement_counts": traced_measurement_counts,
    }

    print("Result string:", result_s)

    return out

def _get_secret_string(measurement_counts: Counter):
    n = int(len(list(measurement_counts.keys())[0]) / 2)

    traced_results = Counter()
    for bitstring, count in measurement_counts.items():
        traced_results.update({bitstring[:n]: count})

    if len(traced_results.keys()) < n:
        raise RuntimeError(
            "System will be underdetermined. Minimum "
            + str(n)
            + " bistrings needed, but only "
            + str(len(traced_results.keys()))
            + " returned. Please rerun Simon's algorithm."
        )
    M = np.vstack([np.array([*key], dtype=int) for key in traced_results]).T

    # Construct the augmented matrix
    M_I = Matrix(np.hstack([M, np.eye(M.shape[0], dtype=int)]))

    # Perform row reduction, working modulo 2. We use the iszerofunc property of rref
    # to perform the Gaussian elimination over the finite field.
    M_I_rref = M_I.rref(iszerofunc=lambda x: x % 2 == 0)

    # In row reduced echelon form, we can end up with a solution outside of the finite field {0,1}.
    # Thus, we need to revert the matrix back to this field by treating fractions as a modular inverse.
    # Since the denominator will always be odd (i.e. 1 mod 2), it can be ignored.

    # Helper function to treat fractions as modular inverse:
    def mod2(x):
        return x.as_numer_denom()[0] % 2

    # Apply our helper function to the matrix
    M_I_final = M_I_rref[0].applyfunc(mod2)

    # Extract the kernel of M from the remaining columns of the last row, when s is nonzero.
    if all(value == 0 for value in M_I_final[-1, : M.shape[1]]):
        result_s = "".join(str(c) for c in M_I_final[-1, M.shape[1] :])

    # Otherwise, the sub-matrix will be full rank, so just set s=0...0
    else:
        result_s = "0" * M.shape[0]

    return result_s, traced_results