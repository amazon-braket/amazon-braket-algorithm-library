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


from typing import Tuple

import numpy as np
from braket.circuits import Circuit, Observable
from braket.devices import Device, LocalSimulator
from braket.tasks import GateModelQuantumTaskResult


def init_states(n_qubits: int) -> Circuit:
    """initalize the state.

    Args:
        n_qubits (int): Number of qubits

    Returns:
        Circuit: Inital state
    """
    return Circuit().h(range(n_qubits)).x(n_qubits).h(n_qubits)


def constant_oracle(n_qubits: int) -> Circuit:
    """Constant oracle circuit.

    Args:
        n_qubits (int): Number of qubits

    Returns:
        Circuit: Constant oracle circuit
    """

    circuit = Circuit()

    # set output to either 0 or 1
    rand_output = np.random.randint(0, 2)

    # if output qubit is 0, apply i gate to not change value
    if rand_output == 0:
        circuit.i(n_qubits)
    # if output is 1, apply x gate to change value to 0
    if rand_output == 1:
        circuit.x(n_qubits)
    return circuit


def balanced_oracle(n_qubits: int) -> Circuit:
    """Balanced oracle circuit.

    Args:
        n_qubits (int): Number of qubits

    Returns:
        Circuit: Balanced oracle circuit
    """
    circuit = Circuit()

    # generate a random array of 0s and 1s to figure out where to place x gates
    random_num = np.random.randint(2, size=n_qubits)

    # place x gates
    for qubit in range(len(random_num)):
        if random_num[qubit] == 1:
            circuit.x(qubit)

    # place cnot gates
    for qubit in range(n_qubits):
        circuit.cnot(control=qubit, target=n_qubits)

    # place x gates
    for qubit in range(len(random_num)):
        if random_num[qubit] == 1:
            circuit.x(qubit)

    return circuit


def random_oracle(n_qubits: int) -> Circuit:
    """Random oracle circuit.

    Args:
        n_qubits (int): Number of qubits

    Returns:
        Circuit: Random oracle circuit
    """
    circuit = Circuit()

    # create a random array of 0s and 1s to determine where to place the gates
    random_array = np.random.randint(2, size=n_qubits)

    # define the single gate set

    # define the multiple qubit gate set
    multiple_gate_set = ["cnot", "ccnot"]

    # create a list combining single and multiple qubit gates to randomly choose from
    choice_gate_set = ["single_gate_set", "multiple_gate_set"]

    # random oracle circuit generator

    # implement all the gate options
    for qubit in range(len(random_array)):
        # randomly choose to apply a single qubit or multiple qubit gate
        random_gate_set = np.random.choice(choice_gate_set, p=[0.30, 0.70])

        # if single qubit gate then implement x gate accordingly
        if random_gate_set == "single_gate_set":
            if random_array[qubit] == 1:
                circuit.x(qubit)

        # if multiple qubit gate then implement cnot and ccnot gates
        if random_gate_set == "multiple_gate_set":
            # randomly choose to implement a cnot or ccnot gate
            random_gate_m = np.random.choice(multiple_gate_set)
            if random_gate_m == "cnot":
                if random_array[qubit] == 0:
                    # randomly choose where the target qubit is
                    targetf = np.random.randint(n_qubits + 1)
                    if qubit != targetf:
                        circuit.cnot(control=qubit, target=targetf)
                else:
                    # randomly choose where the target qubit is
                    targetf = np.random.randint(n_qubits + 1)
                    circuit.x(qubit)
                    if qubit != targetf:
                        circuit.cnot(control=qubit, target=targetf)
                    circuit.x(qubit)
            if random_gate_m == "ccnot":
                # randomly choose where the first and second controls are
                control1 = np.random.randint(n_qubits + 1)
                control2 = np.random.randint(n_qubits + 1)
                # randomly choose where the target qubit is
                targetf = np.random.randint(n_qubits + 1)
                if control1 != control2 and control1 != targetf and control2 != targetf:
                    circuit.ccnot(control1, control2, targetf)

    return circuit


def deutsch_jozsa_algorithm(
    oracle: Circuit, n_qubits: int, device: Device = None, shots: int = 10_000
) -> Tuple[Circuit, GateModelQuantumTaskResult]:
    """General Deutsch-Jozsa algorithm.

    Args:
        oracle (Circuit): Oracle circuit
        n_qubits (int): Number of qubits.
        device (Device, optional): Device to run on. Defaults to LocalSimulator.
        shots (int, optional): Number of shots. Defaults to 10_000.

    Returns:
        Circuit: the Deutsch-Jozsa circuit
    """
    if device is None:
        device = LocalSimulator()

    # define the output qubit
    output_qubit = n_qubits

    # create circuit and initialize states
    circuit = Circuit().h(range(n_qubits)).x(output_qubit).h(output_qubit)

    # add the oracle circuit
    circuit.add_circuit(oracle, range(n_qubits + output_qubit))

    # place the h-gates again
    circuit.h(range(n_qubits))

    # measure the results
    for qubit in range(n_qubits):
        circuit.sample(observable=Observable.Z(), target=qubit)

    # Designate the device being used as the local simulator, feel free to use another device
    task = device.run(circuit, shots=shots)
    # Retrieve the result
    result = task.result()
    # print the measurement probabilities
    print("Measurement results:\n", result.measurement_probabilities)

    return circuit, result


def classical_generator(
    n_qubits: int, random_oracle: Circuit, device: Device = None, shots: int = 1
) -> Circuit:
    """Run initialized states through circuit and add random_oracle


    Args:
        n_qubits (int): Number of qubits.
        random_oracle (Circuit): Oracle circuit.
        device (Device): Device to run on. Defaults to LocalSimulator.
        shots (int): Number of shots. Defaults to 1.

    Returns:
        Circuit: Random_oracle
    """
    if device is None:
        device = LocalSimulator()

    output_qubit = n_qubits
    n = n_qubits
    matrix_bitout = construct_matrix(n)
    measurement_results_classical = []

    for row in matrix_bitout:
        circuit = build_dj_circuit(random_oracle, output_qubit, n, row)
        task = device.run(circuit, shots=shots)
        result = task.result()

        # store results in a measurement variable
        measurement_classical = result.measurement_probabilities
        measurement_list = list(measurement_classical.keys())[-1]
        measurement_results_classical.append(measurement_list)

        # classical checker circuits
        classical_results = np.array(measurement_results_classical)
        final_rc = []
        for row in classical_results:
            result = int(row[-1])
            final_rc.append(result)

        total_rc = len(final_rc)
        count_0 = 0
        count_1 = 0

        for val in final_rc:
            if val == 0:
                count_0 += 1
            elif val == 1:
                count_1 += 1

    if total_rc == count_0:
        print("This is a constant function")

    elif count_0 == total_rc / 2 and count_1 == total_rc / 2:
        print("This is a balanced function")
    else:
        print("This is neither a constant nor balanced function")

    print("These are the final outputs from the random circuits", final_rc)
    return random_oracle


def construct_matrix(n: int) -> np.ndarray:
    """Construct a matrix.

    Args:
        n (int): log2(size of matrix.

    Returns:
        ndarray: Matrix
    """
    matrix = []
    for i in range(2**n):
        # convert integer to bitstring and into an integer
        bit_str = [int(x) for x in str(format(i, "b"))]
        # if len of bitstring less than n then pad with zeros
        if len(bit_str) < n:
            num_zero = n - len(bit_str)
            bit_str = np.pad(bit_str, (num_zero, 0), "constant")
        matrix.append(bit_str)

    return np.array(matrix)


def build_dj_circuit(random_oracle: Circuit, output_qubit: int, row: np.ndarray) -> Circuit:
    """Construct full Deustch-Jozsa circuit.

    Args:
        random_oracle (Circuit): Random oracle circuit.
        output_qubit (int): Output qubit.
        row (np.ndarray): Row vector.

    Returns:
        Circuit: Deustch-Jozsa circuit.
    """
    circuit = Circuit()
    # retrieve the bit string that initializes the qubits for this iteration
    qubit_info = row.tolist()
    # creating initial states of our classical circuit
    for qubit in range(len(qubit_info)):
        if qubit_info[qubit] == 1:
            circuit.x(qubit)
        else:
            circuit.i(qubit)
    circuit.add_circuit(random_oracle, range(output_qubit))
    circuit.sample(observable=Observable.Z(), target=output_qubit)
    return circuit


def quantum_generator(
    n_qubits: int,
    random_oracle: Circuit,
    device: Device = None,
    shots: int = 100,
) -> Circuit:
    """Run initialized states through circuit and add random_oracle


    Args:
        n_qubits (int): Number of qubits.
        random_oracle (Circuit): Oracle circuit.
        device (Device): Device to run on. Defaults to LocalSimulator.
        shots (int): Number of shots. Defaults to 100.

    Returns:
        Circuit: Final circuit used for Deutsch-Jozsa algorithm.
    """
    if device is None:
        device = LocalSimulator()
    output_qubit = n_qubits

    # create circuit and initialize states
    circuit = build_quantum_dj_circuit(n_qubits, random_oracle, output_qubit)

    device = LocalSimulator()
    task = device.run(circuit, shots=shots)
    result = task.result()
    measurement_quantum = result.measurement_probabilities
    print("Measurement results:\n", result.measurement_probabilities)

    measurement_keys = list(measurement_quantum.keys())
    quantum_results = np.array(measurement_keys)

    all_measurements = []
    for row in quantum_results:
        func = [int(val) for val in row[:-1]]
        all_measurements.append(func)

    num_outputs = len(all_measurements)
    type_func = sum(0 if sum(row) == 0 else 1 for row in all_measurements)
    print_results(num_outputs, type_func)
    return circuit


def print_results(num_outputs: int, type_func: int):
    """Print the predictions.

    Args:
        num_outputs (int): Number of outputs.
        type_func (int): Type of function.
    """
    if type_func == 0:
        print("This is a constant function")
    elif type_func == num_outputs:
        print("This is a balanced function")
    else:
        print("This is neither a balanced or constant function")


def build_quantum_dj_circuit(n_qubits: int, random_oracle: Circuit, output_qubit: int) -> Circuit:
    """Build the full circuit.

    Args:
        n_qubits (int): Number of qubits
        random_oracle (Circuit): Random oracle circuit.
        output_qubit (int): Output qubit.

    Returns:
        Circuit: Full circuit with measurements.
    """
    circuit = Circuit().h(range(n_qubits)).x(output_qubit).h(output_qubit)
    # add the random oracle to this circuit
    circuit.add_circuit(random_oracle, range(n_qubits))
    # place the h-gates again
    circuit.h(range(n_qubits))

    # measure the results
    for qubit in range(n_qubits):
        circuit.sample(observable=Observable.Z(), target=qubit)
    return circuit
