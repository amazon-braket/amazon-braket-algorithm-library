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

import numpy as np
from braket.circuits import Circuit, circuit
from braket.circuits.qubit_set import QubitSetInput
from braket.devices import Device
from braket.tasks import QuantumTask


def quantum_phase_estimation_circuit(
    quantum_phase_estimation_circ: Circuit,
    precision_qubits: QubitSetInput,
    query_qubits: QubitSetInput,
    unitary_apply_func: Callable,
) -> Circuit:
    """Adds result type to quantum phase estimation

    Args:
        quantum_phase_estimation_circ (Circuit): Circuit with query bits prepared
        precision_qubits (QubitSetInput): Qubits defining the precision register
        query_qubits (QubitSetInput) : Qubits defining the query register
        unitary_apply_func (Callable): Function that applies the desired controlled unitary to a
            provided circuit using provided control and target qubits

    Returns:
        Circuit: Implements the Quantum Phase Estimation algorithm with result type
    """

    return quantum_phase_estimation_circ.quantum_phase_estimation(
        precision_qubits, query_qubits, unitary_apply_func
    ).probability()


@circuit.subroutine(register=True)
def quantum_phase_estimation(
    precision_qubits: QubitSetInput,
    query_qubits: QubitSetInput,
    unitary_apply_func: Callable,
) -> Circuit:
    """Creates the Quantum Phase Estimation circuit using:

    1) The first register for precision.
    2)The second register for query qubits which hosts the eigenstate and should already be prepared
    in its initial state. 3) A function that applies a controlled unitary circuit. This function
    accepts a control qubit and the target qubits on which to apply the unitary. Quantum Phase
    Estimation will repeatedly apply this function for the target qubits. This is a necessary input
    because the controlled unitary needs to be defined in terms of available gates for a given QPU.

    Example:
        >>> def cnot_apply_func(circ, control_qubit, query_qubits):
        ...    circ.qpe_cnot_unitary(control_qubit, query_qubits)
        >>> circ = Circuit().h([2])
        >>> circ.quantum_phase_estimation([0, 1], [2], cnot_apply_func)
        >>> print(circ)
        T  : |0|1| 2  |3|     4      |5|
        q0 : -H---SWAP---PHASE(-1.57)-H-
                  |      |
        q1 : -H-C-SWAP-H-C--------------
                |
        q2 : -H-X-----------------------
        T  : |0|1| 2  |3|     4      |5|

    Args:
        precision_qubits (QubitSetInput): Qubits defining the precision register
        query_qubits (QubitSetInput) : Qubits defining the query register
        unitary_apply_func (Callable): Function that applies the desired controlled unitary to a
            provided circuit using provided control and target qubits

    Returns:
        Circuit: Circuit object that implements the Quantum Phase Estimation algorithm
    """
    quantum_phase_estimation_circ = Circuit()

    quantum_phase_estimation_circ.h(precision_qubits)

    # Apply controlled unitaries C-U(2^k). Start with the last precision_qubit, end with the first
    for ii, qubit in enumerate(reversed(precision_qubits)):
        if qubit:
            for _ in range(2**ii):
                unitary_apply_func(quantum_phase_estimation_circ, qubit, query_qubits)

    quantum_phase_estimation_circ.inverse_qft(precision_qubits)

    return quantum_phase_estimation_circ


def run_quantum_phase_estimation(
    circuit: Circuit,
    device: Device,
    shots: int = 1000,
) -> QuantumTask:
    """Function to run Quantum Phase Estimation algorithm and return measurement counts.

    Args:
        circuit (Circuit): Quantum Phase Estimation circuit
        device (Device): Braket device backend
        shots (int) : Number of measurement shots (default is 1000).

    Returns:
        QuantumTask: Task from running Quantum Phase Estimation
    """

    task = device.run(circuit, shots=shots)

    return task


def get_quantum_phase_estimation_results(
    task: QuantumTask,
    precision_qubits: QubitSetInput,
    query_qubits: QubitSetInput,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Function to postprocess results returned by run_quantum_phase_estimation and pretty print
    results.

    Args:
        task (QuantumTask): The task which holds the results for the quantum phase estimation run
        precision_qubits (QubitSetInput): Qubits defining the precision register
        query_qubits (QubitSetInput) : Qubits defining the query register
        verbose (bool) : If True, prints aggregate results (default is False)

    Returns:
        Dict[str, Any]: aggregate measurement results
    """

    result = task.result()
    metadata = result.task_metadata
    probs_values = result.values[0]
    measurements = result.measurements
    measured_qubits = result.measured_qubits
    measurement_counts = result.measurement_counts
    measurement_probabilities = result.measurement_probabilities

    num_qubits = len(precision_qubits) + len(query_qubits)
    format_bitstring = "{0:0" + str(num_qubits) + "b}"
    bitstring_keys = [format_bitstring.format(ii) for ii in range(2**num_qubits)]

    if not measurement_counts:
        phases_decimal = result.result_types[0].value
        precision_results_dict = None
    else:
        phases_decimal, precision_results_dict = _get_quantum_phase_estimation_phases(
            measurement_counts, precision_qubits
        )

    eigenvalues = [np.exp(2 * np.pi * 1j * phase) for phase in phases_decimal]
    eigenvalue_estimates = np.round(eigenvalues, 5)

    aggregate_results = {
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
        "eigenvalue_estimates": eigenvalue_estimates,
    }

    if verbose:
        print(f"Measurement counts: {measurement_counts}")
        print(f"Results in precision register: {precision_results_dict}")
        print(f"Quantum phase estimation phase estimates: {phases_decimal}")
        print(f"Quantum phase estimation eigenvalue estimates: {eigenvalue_estimates}")

    return aggregate_results


def _binary_to_decimal(binary: str) -> float:
    """Helper function to convert binary string (example: '01001') to decimal.

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


def _get_quantum_phase_estimation_phases(
    measurement_counts: Counter, precision_qubits: QubitSetInput
) -> Tuple[List[float], Dict[str, int]]:
    """Get Quantum Phase Estimates phase estimate from measurement_counts for given number of
    precision qubits.

    Args:
        measurement_counts (Counter) : measurement results from a device run
        precision_qubits (QubitSetInput): Qubits defining the precision register

    Returns:
        Tuple[List[float], Dict[str, int]]: decimal phase estimates, precision results
    """

    # Aggregate the results (i.e., ignore/trace out the query register qubits):
    # First get bitstrings with corresponding counts for precision qubits only
    bitstrings_precision_register = [
        key[: len(precision_qubits)] for key in measurement_counts.keys()
    ]

    # Now create a new dict to collect measurement results on the precision_qubits. Keys are given
    # by the measurement count substrings on the register qubits. Initialize the counts to zero.
    precision_results_dict = {key: 0 for key in set(bitstrings_precision_register)}

    # Loop over all measurement outcomes
    for key in measurement_counts.keys():
        # Save the measurement count for this outcome
        counts = measurement_counts[key]
        # Generate the corresponding shortened key (supported only on the precision_qubits register)
        count_key = key[: len(precision_qubits)]
        # Add these measurement counts to the corresponding key in our new dict
        precision_results_dict[count_key] += counts

    # get decimal phases from bitstrings
    phases_decimal = [_binary_to_decimal(item[0]) for item in precision_results_dict]

    return phases_decimal, precision_results_dict


# TODO: Add to qft module once available
# inverse QFT
@circuit.subroutine(register=True)
def inverse_qft(qubits: QubitSetInput) -> Circuit:
    """Construct a circuit object corresponding to the inverse Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the circuit.

    Args:
        qubits (QubitSetInput): Qubits on which to apply the inverse Quantum Fourier Transform

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
