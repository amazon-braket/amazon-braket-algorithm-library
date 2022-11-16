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
from fractions import Fraction
from typing import Any, Dict, List, Optional

import numpy as np
from braket.circuits import Circuit, circuit
from braket.circuits.qubit_set import QubitSetInput
from braket.devices import Device


@circuit.subroutine(register=True)
def shors_algorithm(integer_N: int, integer_a: int) -> Circuit:
    """
    Creates the circuit for Shor's algorithm.
      1) Based on integer N, calculate number of counting qubits for the first register
      2) Setup same number of auxiliary qubits for the second register
         and apply modular exponentian function
      3) Apply inverse_QFT

    Args:
        integer_N (int) : The integer N to be factored
        integer_a (int) : Any integer 'a' that satisfies 1 < a < N and gcd(a, N) = 1.

    Returns:
        Circuit: Circuit object that implements the Shor's algorithm
    """

    # validate the inputs
    if integer_N < 1 or integer_N % 2 == 0:
        raise ValueError("The input N needs to be an odd integer greater than 1.")
    if integer_a >= integer_N or math.gcd(integer_a, integer_N) != 1:
        raise ValueError('The integer "a" needs to satisfy 1 < a < N and gcd(a, N) = 1.')

    # calculate number of qubits needed
    n = int(np.ceil(np.log2(integer_N)))
    m = n

    counting_qubits = [*range(n)]
    aux_qubits = [*range(n, n + m)]

    shors_circuit = Circuit()

    # Initialize counting and aux qubits
    shors_circuit.h(counting_qubits)
    shors_circuit.x(aux_qubits[0])

    # Apply modular exponentiation
    shors_circuit.modular_exponentiation_amod15(counting_qubits, aux_qubits, integer_a)

    # Apply inverse QFT
    shors_circuit.inverse_qft_noswaps(counting_qubits)

    return shors_circuit


def run_shors_algorithm(
    circuit: Circuit,
    device: Device,
    shots: Optional[int] = 1000,
) -> Dict[str, Any]:
    """
    Function to run Shor's algorithm and return measurement counts.

    Args:
        circuit (Circuit): Shor's algorithm circuit
        device (Device): Braket device backend
        shots (Optional[int]) : Number of measurement shots (default is 1000).
            0 shots results in no measurement.

    Returns:
        Dict[str, Any]: measurements and results from running Shors's algorithm
    """

    task = device.run(circuit, shots=shots)

    result = task.result()

    out = {
        "measurements": result.measurements,
        "measured_qubits": result.measured_qubits,
        "measurement_counts": result.measurement_counts,
        "measurement_probabilities": result.measurement_probabilities,
    }

    return out


@circuit.subroutine(register=True)
def inverse_qft_noswaps(qubits: QubitSetInput) -> Circuit:
    """
    Construct a circuit object corresponding to the inverse Quantum Fourier Transform (QFT)
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
    # for i in range(math.floor(num_qubits / 2)):
    #     qft_circuit.swap(qubits[i], qubits[-i - 1])

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


@circuit.subroutine(register=True)
def modular_exponentiation_amod15(
    counting_qubits: QubitSetInput, aux_qubits: QubitSetInput, integer_a: int
) -> Circuit:
    """
    Construct a circuit object corresponding the modular exponentiation of a^x Mod 15

    Args:
        counting_qubits (QubitSetInput): Qubits defining the counting register
        aux_qubits (QubitSetInput) : Qubits defining the auxilary register
        integer_a (int) : Any integer that satisfies 1 < a < N and gcd(a, N) = 1.
    Returns:
        Circuit: Circuit object that implements the modular exponentiation of a^x Mod 15
    """

    # Instantiate circuit object
    mod_exp_amod15 = Circuit()

    for x in counting_qubits:
        r = 2**x
        if integer_a not in [2, 7, 8, 11, 13]:
            raise ValueError("integer 'a' must be 2,7,8,11 or 13")
        for iteration in range(r):
            if integer_a in [2, 13]:
                mod_exp_amod15.cswap(x, aux_qubits[0], aux_qubits[1])
                mod_exp_amod15.cswap(x, aux_qubits[1], aux_qubits[2])
                mod_exp_amod15.cswap(x, aux_qubits[2], aux_qubits[3])
            if integer_a in [7, 8]:
                mod_exp_amod15.cswap(x, aux_qubits[2], aux_qubits[3])
                mod_exp_amod15.cswap(x, aux_qubits[1], aux_qubits[2])
                mod_exp_amod15.cswap(x, aux_qubits[0], aux_qubits[1])
            if integer_a == 11:
                mod_exp_amod15.cswap(x, aux_qubits[1], aux_qubits[3])
                mod_exp_amod15.cswap(x, aux_qubits[0], aux_qubits[2])
            if integer_a in [7, 11, 13]:
                for q in aux_qubits:
                    mod_exp_amod15.cnot(x, q)

    return mod_exp_amod15


def get_factors_from_results(
    results: Dict[str, Any],
    integer_N: int,
    integer_a: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Function to postprocess dictionary returned by run_shors_algorithm
        and pretty print results

    Args:
        results (Dict[str, Any]): Results associated with quantum phase estimation run as produced
            by run_shors_algorithm
        integer_N (int) : The integer to be factored
        integer_a (int) : Any integer that satisfies 1 < a < N and gcd(a, N) = 1.
        verbose (bool) : If True, prints aggregate results (default is False)
    Returns:
        Dict[str, Any]: Factors of the integer N
    """

    # unpack results
    measurement_counts = results["measurement_counts"]

    # get phases
    phases_decimal = _get_phases(measurement_counts)

    r_guesses = []
    factors = []
    if verbose:
        print(f"Number of Measured phases (s/r) : {len(phases_decimal)}")
    for phase in phases_decimal:
        if verbose:
            print(f"\nFor phase {phase} :")
        r = (Fraction(phase).limit_denominator(integer_N)).denominator
        r_guesses.append(r)
        if verbose:
            print(f"Estimate for r is : {r}")
        factor = [
            math.gcd(integer_a ** (r // 2) - 1, integer_N),
            math.gcd(integer_a ** (r // 2) + 1, integer_N),
        ]
        factors.append(factor[0])
        factors.append(factor[1])
        if verbose:
            print(f"Factors are : {factor[0]} and {factor[1]}")
    factors_set = set(factors)
    factors_set.discard(1)
    factors_set.discard(integer_N)
    if verbose:
        print(f"\n\nNon-trivial factors found are : {factors_set}")

    aggregate_results = {"guessed_factors": factors_set}

    return aggregate_results


def _get_phases(measurement_counts: Counter) -> List[float]:
    """
    Get phase estimate from measurement_counts using top half qubits

    Args:
        measurement_counts (Counter) : measurement results from a device run
    Returns:
        List[float] : decimal phase estimates
    """

    # Aggregate the results (i.e., ignore/trace out the query register qubits):
    if not measurement_counts:
        return None

    # First get bitstrings with corresponding counts for counting qubits only (top half)
    num_counting_qubits = int(len(list(measurement_counts.keys())[0]) / 2)

    bitstrings_precision_register = [key[:num_counting_qubits] for key in measurement_counts.keys()]

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
        count_key = key[:num_counting_qubits]
        # Add these measurement counts to the corresponding key in our new dict
        precision_results_dict[count_key] += counts

    phases_decimal = [_binary_to_decimal(item) for item in precision_results_dict.keys()]

    return phases_decimal


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
