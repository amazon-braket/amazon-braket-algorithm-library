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


from typing import List

import numpy as np
from braket.circuits import Circuit, FreeParameter, Observable, circuit
from braket.devices import Device
from braket.tasks import QuantumTask


def cost_function(
    values: np.ndarray,
    device: Device,
    circ: Circuit,
    coeffs: np.ndarray,
    cost_history: List[float],
    shots: int = 0,
) -> float:
    """Cost function and append to loss history list.

    Args:
        values (ndarray): Values for the parameters.
        device (Device): Braket device to run on.
        circ (Circuit): QAOA circuit to run.
        coeffs (ndarray): The coefficients of the cost Hamiltonian.
        cost_history (List[float]): History of cost evaluations.
        shots (int): Number of shots. Defaults to 0.

    Returns:
        float: The cost function value
    """
    task = run_qaoa_circuit(device, circ, values, shots=shots)
    cost = get_cost(task, coeffs)
    cost_history.append(cost)
    return cost


def run_qaoa_circuit(device: Device, circ: Circuit, values: np.ndarray, shots: int) -> QuantumTask:
    """Evaluate a QAOA circuit with parameters=values.

    Args:
        device (Device): Braket device to run on.
        circ (Circuit): QAOA circuit to run.
        values (np.ndarray): Values for the parameters.
        shots (int): Number of shots.

    Returns:
        QuantumTask: The Braket task to run.
    """
    fixed_circuit = circ.make_bound_circuit(
        dict(zip(np.array(list(circ.parameters), dtype=str), values))
    )
    task = device.run(fixed_circuit, shots=shots)
    return task


def get_cost(task: QuantumTask, coeffs: np.ndarray) -> float:
    """Evaluate the cost function from a QAOA task.

    Args:
        task (QuantumTask): QAOA task.
        coeffs (np.ndarray): The coefficients of the cost Hamiltonian.

    Returns:
        float: Loss function value.
    """
    exp_vals = task.result().result_types
    cost = sum(c * s.value for c, s in zip(coeffs, exp_vals))
    return cost


def qaoa(n_qubits: int, n_layers: int, ising: np.ndarray) -> Circuit:
    """QAOA template.

    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers. Defaults to 1.
        ising (ndarray): Ising interaction matrix.

    Returns:
        Circuit: The parameteric QAOA Circuit
    """

    gammas = [FreeParameter(f"gamma_{p}") for p in range(n_layers)]
    betas = [FreeParameter(f"beta_{p}") for p in range(n_layers)]

    circ = Circuit()
    circ.h(range(n_qubits))  # prepare |+> state
    for gamma, beta in zip(gammas, betas):
        circ.cost_layer(gamma, ising)
        circ.driver_layer(beta, n_qubits)

    # add Result types
    idx = ising.nonzero()
    for qubit_pair in zip(idx[0], idx[1]):
        # get interaction strength from Ising matrix
        circ.expectation(observable=Observable.Z() @ Observable.Z(), target=qubit_pair)
    return circ


@circuit.subroutine(register=True)
def driver_layer(beta: float, n_qubits: int) -> Circuit:
    """Returns circuit for driver Hamiltonian U(Hb, beta).

    Args:
        beta (float): Rotation angle to apply parameterized rotation around x
        n_qubits (int): number of qubits to apply rx gate

    Returns:
        Circuit: Circuit object that implements evolution with driver Hamiltonian
    """
    return Circuit().rx(range(n_qubits), 2 * beta)


@circuit.subroutine(register=True)
def cost_layer(
    gamma: float,
    ising: np.ndarray,
) -> Circuit:
    """Returns circuit for evolution with cost Hamiltonian.

    Args:
        gamma (float): Rotation angle to apply parameterized rotation around z
        ising (np.ndarray): Ising matrix

    Returns:
        Circuit: Circuit for evolution with cost Hamiltonian
    """
    circ = Circuit()
    # get all non-zero entries (edges) from Ising matrix
    idx = ising.nonzero()
    edges = list(zip(idx[0], idx[1]))
    # apply ZZ gate for every edge (with corresponding interaction strength)
    for qubit_pair in edges:
        # get interaction strength from Ising matrix
        interaction_strength = ising[qubit_pair[0], qubit_pair[1]]
        circ.decomposed_zz_gate(qubit_pair[0], qubit_pair[1], gamma * interaction_strength)
    return circ


@circuit.subroutine(register=True)
def decomposed_zz_gate(qubit0: int, qubit1: int, gamma: float) -> Circuit:
    """Return a circuit implementing exp(-i gamma Z_i Z_j) using CNOT gates if ZZ not supported.

    Args:
        qubit0 (int): Index value for the controlling qubit for CNOT gate
        qubit1 (int): Index value for the target qubit for CNOT gate
        gamma (float): Rotation angle to apply parameterized rotation around z

    Returns:
        Circuit: Circuit object that implements ZZ gate using CNOT gates
    """
    circ_zz = Circuit()
    circ_zz.cnot(qubit0, qubit1).rz(qubit1, gamma).cnot(qubit0, qubit1)
    return circ_zz
