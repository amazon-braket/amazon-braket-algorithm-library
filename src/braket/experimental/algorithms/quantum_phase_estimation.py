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

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from braket.circuits import Circuit, circuit
from braket.devices import Device


@circuit.subroutine(register=True)
def quantum_phase_estimation(
    precision_qubits: List[int],
    query_qubits: List[int],
    unitary_apply_func: Callable,
) -> Circuit:
    """
    Creates the Quantum Phase Estimation circuit using:
      1) The first register for precision
      2) The second register for query qubits which hosts the eigenstate
         and should already be prepared in its initial state
      3) A function that applies a controlled unitary circuit. This function accepts a control qubit
         and the target qubits on which to apply the unitary. Quantum Phase Estimation will
         repeatedly apply this function for the target qubits. This is a necessary input because the
         controlled unitary needs to be defined in terms of available gates for a given QPU.

    Args:
        precision_qubits (list): Qubits defining the precision register
        query_qubits (list) : Qubits defining the query register
        unitary_apply_func (Callable): Function that applies the desired controlled unitary to
        a provided circuit using provided control and target qubits

    Returns:
        Circuit: Circuit object that implements the Quantum Phase Estimation algorithm
    """
    quantum_phase_estimation_circuit = Circuit()

    # Apply Hadamard across precision register
    quantum_phase_estimation_circuit.h(precision_qubits)

    # Apply controlled unitaries C-U(2^k). Start with the last precision_qubit, end with the first
    for ii, qubit in enumerate(reversed(precision_qubits)):
        # If the control qubit is zero, the identity gate is used and there is no need to call apply
        if qubit:
            for _ in range(2**ii):
                unitary_apply_func(
                    quantum_phase_estimation_circuit, qubit, query_qubits
                )

    # Apply inverse qft to the precision_qubits
    quantum_phase_estimation_circuit.inverse_qft(precision_qubits)

    return quantum_phase_estimation_circuit


def run_quantum_phase_estimation(
    circuit: Circuit,
    precision_qubits: List[int],
    query_qubits: List[int],
    device: Device,
    items_to_keep: int = None,
    shots: int = 1000,
) -> Dict[str, Any]:
    """
    Function to run Quantum Phase Estimation algorithm and return measurement counts.

    Args:
        circuit (Circuit): Quantum Phase Estimation circuit
        precision_qubits (List): Qubits defining the precision register
        query_qubits (List) : Qubits defining the query register
        device (Device): Braket device backend
        items_to_keep (int) : Number of items to return, topmost measurement counts for
                              precision register (default to None which means all)
        shots (int) : Number of measurement shots (default is 1000).
                      0 shots results in no measurement.

    Returns:
        dict: measurements and results from running Quantum Phase Estimation
    """

    # Add desired results_types
    circuit.probability()

    # get total number of qubits
    num_qubits = len(precision_qubits) + len(query_qubits)

    # Run the circuit with all zeros input.
    # The query_circuit subcircuit generates the desired input from all zeros.
    task = device.run(circuit, shots=shots)

    # get result for this task
    result = task.result()

    # get metadata
    metadata = result.task_metadata

    # get output probabilities (see result_types above)
    probs_values = result.values[0]

    # get measurement results
    measurements = result.measurements
    measured_qubits = result.measured_qubits
    measurement_counts = result.measurement_counts
    measurement_probabilities = result.measurement_probabilities

    # bitstrings
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    # quantum phase estimation postprocessing
    phases_decimal, precision_results_dict = _get_quantum_phase_estimation_phases(
        measurement_counts, precision_qubits, items_to_keep
    )

    if not phases_decimal and not precision_results_dict:
        eigenvalues = None
    else:
        eigenvalues = [np.exp(2 * np.pi * 1j * phase) for phase in phases_decimal]

    # aggregate results
    out = {
        "circuit": circuit,
        "task_metadata": metadata,
        "measurements": measurements,
        "measured_qubits": measured_qubits,
        "measurement_counts": measurement_counts,
        "measurement_probabilities": measurement_probabilities,
        "probs_values": probs_values,
        "bitstring_keys": bitstring_keys,
        "precision_results_dict": precision_results_dict,
        "phases_decimal": phases_decimal,
        "eigenvalues": eigenvalues,
    }

    return out


# helper function to convert binary fractional to decimal
# reference: https://www.geeksforgeeks.org/convert-binary-fraction-decimal/
def _binary_to_decimal(binary: str) -> float:
    """
    Helper function to convert binary string (example: '01001') to decimal

    Args:
        binary (str): value to convert to decimal fraction

    Returns:
        float: decimal value
    """

    fracDecimal = 0

    # Convert fractional part of binary to decimal equivalent
    twos = 2

    for ii in range(len(binary)):
        fracDecimal += (ord(binary[ii]) - ord("0")) / twos
        twos *= 2.0

    # return fractional part
    return fracDecimal


# helper function for postprocessing based on measurement shots
def _get_quantum_phase_estimation_phases(
    measurement_counts: Counter, precision_qubits: List[int], items_to_keep: int = 1
) -> Tuple[List[float], Dict[str, int]]:
    """
    Get Quantum Phase Estimates phase estimate from measurement_counts for given number
    of precision qubits

    Args:
        measurement_counts: measurement results from a device run
        precision_qubits: List of qubits corresponding to precision_qubits. Currently assumed to be
                          a list of integers corresponding to the indices of the qubits.
        items_to_keep: number of items to return (topmost measurement counts for precision register)

    Returns:
        list: decimal phase estimates
        dict: precision results
    """
    # Aggregate the results (i.e., ignore/trace out the query register qubits):
    if not measurement_counts:
        return None, None
    # First get bitstrings with corresponding counts for precision qubits only
    bitstrings_precision_register = [
        key[: len(precision_qubits)] for key in measurement_counts.keys()
    ]

    # Then keep only the unique strings
    bitstrings_precision_register_set = set(bitstrings_precision_register)
    # Cast as a list for later use
    bitstrings_precision_register_list = list(bitstrings_precision_register_set)

    # Now create a new dict to collect measurement results on the precision_qubits. Keys are given
    # by the measurement count substrings on the register qubits. Initialize the counts to zero.
    precision_results_dict = {key: 0 for key in bitstrings_precision_register_list}

    # Loop over all measurement outcomes
    for key in measurement_counts.keys():
        # Save the measurement count for this outcome
        counts = measurement_counts[key]
        # Generate the corresponding shortened key (supported only on the precision_qubits register)
        count_key = key[: len(precision_qubits)]
        # Add these measurement counts to the corresponding key in our new dict
        precision_results_dict[count_key] += counts

    # Get topmost values only
    c = Counter(precision_results_dict)
    topmost = c.most_common(items_to_keep)
    # get decimal phases from bitstrings for topmost bitstrings
    phases_decimal = [_binary_to_decimal(item[0]) for item in topmost]

    return phases_decimal, precision_results_dict


def get_quantum_phase_estimation_results(out: Dict[str, Any]):
    """
    Function to postprocess dictionary returned by run_quantum_phase_estimation
    and pretty print results

    Args:
        out (dict): Results associated with quantum phase estimation run as produced by
        run_quantum_phase_estimation
    """

    # unpack results
    circuit = out["circuit"]
    measurement_counts = out["measurement_counts"]
    bitstring_keys = out["bitstring_keys"]
    probs_values = out["probs_values"]
    precision_results_dict = out["precision_results_dict"]
    phases_decimal = out["phases_decimal"]
    eigenvalues = out["eigenvalues"]

    # print the circuit
    print(f"Printing circuit: {circuit}")

    # print measurement results
    print(f"Measurement counts: {measurement_counts}")

    # plot probabalities
    plt.bar(bitstring_keys, probs_values)
    plt.xlabel("bitstrings")
    plt.ylabel("probability")
    plt.xticks(rotation=90)

    if not eigenvalues:
        eigenvalue_estimates = None
    else:
        eigenvalue_estimates = np.round(eigenvalues, 5)

    # print results
    print(f"Results in precision register: {precision_results_dict}")
    print(f"Quantum phase estimation phase estimates: {phases_decimal}")
    print(f"Quantum phase estimation eigenvalue estimates: {eigenvalue_estimates}")


# TODO: Add to qft module once available
# inverse QFT
@circuit.subroutine(register=True)
def inverse_qft(qubits: List[int]) -> Circuit:
    """
    Construct a circuit object corresponding to the inverse Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the circuit.

    Args:
        qubits (list): Qubits on which to apply the inverse Quantum Fourier Transform

    Returns:
        Circuit: Circuit object that implements the inverse Quantum Fourier Transform algorithm
    """
    # Instantiate circuit object
    qft_circuit = Circuit()

    # Fet number of qubits
    num_qubits = len(qubits)

    # First add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qft_circuit.swap(qubits[i], qubits[-i - 1])

    # Start on the last qubit and work to the first.
    for k in reversed(range(num_qubits)):

        # Apply the controlled rotations, with weights (angles) defined by the distance to the
        # control qubit. These angles are the negative of the angle used in the QFT.
        # Start on the last qubit and iterate until the qubit after k.
        # When num_qubits==1, this loop does not run.
        for j in reversed(range(1, num_qubits - k)):
            angle = -2 * math.pi / (2 ** (j + 1))
            qft_circuit.cphaseshift(qubits[k + j], qubits[k], angle)

        # Then add a Hadamard gate
        qft_circuit.h(qubits[k])

    return qft_circuit
