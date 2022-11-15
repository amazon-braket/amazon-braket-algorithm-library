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

## Quantum Fourier Transform: Amazon Braket Algorithm Library

## See https://en.wikipedia.org/wiki/Quantum_Fourier_transform
## See tutorial at https://github.com/aws/amazon-braket-examples/blob/main/examples/advanced_circuits_algorithms/Quantum_Fourier_Transform/Quantum_Fourier_Transform.ipynb

# general imports
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

# AWS imports: Import Braket SDK modules
from braket.circuits import Circuit


# @circuit.subroutine(register=True)
def quantum_fourier_transform(qubits):
    """
    Construct a circuit object corresponding to the Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the QFT.

    Args:
        qubits (list): The list of qubits labels on which to apply the QFT
    """

    qftcirc = Circuit()

    if isinstance(qubits, int):
        qubits = list(range(qubits))

    # get number of qubits
    num_qubits = len(qubits)

    for k in range(num_qubits):
        # First add a Hadamard gate
        qftcirc.h(qubits[k])

        # Then apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
        # Start on the qubit after qubit k, and iterate until the end.  When num_qubits==1, this loop does not run.
        for j in range(1, num_qubits - k):
            angle = 2 * math.pi / (2 ** (j + 1))
            qftcirc.cphaseshift(qubits[k + j], qubits[k], angle)

    # Then add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qftcirc.swap(qubits[i], qubits[-i - 1])

    return qftcirc


# @circuit.subroutine(register=True)
def inverse_quantum_fourier_transform(qubits):
    """
    Construct a circuit object corresponding to the inverse Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.  Does not use recursion to generate the circuit.

    Args:
        qubits (list): The list of qubits on which to apply the inverse QFT
    """
    # instantiate circuit object
    qftcirc = Circuit()

    if isinstance(qubits, int):
        qubits = list(range(qubits))

    # get number of qubits
    num_qubits = len(qubits)

    # First add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qftcirc.swap(qubits[i], qubits[-i - 1])

    # Start on the last qubit and work to the first.
    for k in reversed(range(num_qubits)):

        # Apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
        # These angles are the negative of the angle used in the QFT.
        # Start on the last qubit and iterate until the qubit after k.
        # When num_qubits==1, this loop does not run.
        for j in reversed(range(1, num_qubits - k)):
            angle = -2 * math.pi / (2 ** (j + 1))
            qftcirc.cphaseshift(qubits[k + j], qubits[k], angle)

        # Then add a Hadamard gate
        qftcirc.h(qubits[k])

    return qftcirc


def run_quantum_fourier_transform(
    qubits, nshots=1000, device=None, state_prep_cir=None, analysis_cir=None, doInverse=False
):
    """Execute QFT algorithm and returns results.

    Args:
        device                The requested device (default: LocalSimulator)
        qubits                (int or list)  number of qubits or a list of qubits
        state_perp_cir        (circuit) to be run before qft
        analysis_cir          (circuit) to be run after  qft
        doInverse             (bool) do the inverse qft
    """

    if isinstance(qubits, int):
        qubits = list(range(qubits))

    if not state_prep_cir:
        circuit = Circuit()
    else:
        circuit = state_prep_cir

    if doInverse:
        circuit = circuit + inverse_quantum_fourier_transform(qubits)
    else:
        circuit = circuit + quantum_fourier_transform(qubits)

    if analysis_cir:
        circuit = circuit + analysis_cir

    if device == None:
        device = braket.devices.LocalSimulator()

    results = device.run(circuit, shots=nshots).result()

    return results


def postprocess_qft_results(result, verbose=False):
    """
    Function to postprocess results returned by run_qpe

    Args:
        out (dict): Results/information associated with QPE run as produced by run_quantum_fourier_transform
    """

    if verbose:
        print(result.result_types)

    probs = []

    if len(result.measured_qubits) < 6:

        n = len(result.measured_qubits)

        # bitstrings
        format_bitstring = "{0:0" + str(n) + "b}"
        bitstring_keys = [format_bitstring.format(ii) for ii in range(2**n)]

        for key in bitstring_keys:
            if key in result.measurement_probabilities:
                probs.append(result.measurement_probabilities[key])
            else:
                probs.append(0.0)

        xlocs = [ii for ii in range(2**n)]

        plt.bar(xlocs, probs)

        plt.ylabel("probabilities")
        plt.xlabel("bitstrings")

        plt.xticks(xlocs, labels=bitstring_keys, rotation=90)
        plt.ylim([0, 1])

        plt.show()

        return

    plt.bar(result.measurement_counts.keys(), result.measurement_counts.values())
    plt.xticks(rotation=90)
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.show()


### QFT IMPLEMENTATION 2:
def qft_alt2(num_qubits, inverse=False):
    """
    Construct a circuit object corresponding to the Quantum Fourier Transform (QFT)
    algorithm over a fixed number of qubits.  Does not use recursion to generate the QFT.

    Args:
        num_qubits (int): The number of qubits on which to apply the QFT.
    """
    qc = Circuit()
    N = num_qubits - 1

    if inverse == False:
        # First add a Hadamard gate
        qc.h(N)

        # Then apply the controlled rotations, with weights (angles) defined by the distance to the control qubit.
        # Start on the qubit after qubit k, and iterate until the end.
        for n in range(1, N + 1):
            qc.cphaseshift(N - n, N, 2 * np.pi / 2 ** (n + 1))

        # repeat structure for all lower values of i < N
        for i in range(1, N):
            qc.h(N - i)
            for n in range(1, N - i + 1):
                qc.cphaseshift(N - (n + i), N - i, 2 * np.pi / 2 ** (n + 1))
        qc.h(0)
    else:
        qc.h(0)
        for i in range(N - 1, 0, -1):
            for n in range(N - i, 0, -1):
                qc.cphaseshift(N - (n + i), N - i, -2 * np.pi / 2 ** (n + 1))
            qc.h(N - i)

        for n in range(N, 0, -1):
            qc.cphaseshift(N - n, N, -2 * np.pi / 2 ** (n + 1))

        qc.h(N)

    return qc
