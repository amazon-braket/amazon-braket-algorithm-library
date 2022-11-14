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

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit, circuit
from braket.devices import LocalSimulator
from scipy.optimize import minimize


def _zzgate(q1: int, q2: int, gamma: float) -> Circuit:
    """function that returns a circuit implementing exp(-i gamma Z_i Z_j)
    using CNOT gates if ZZ not supported

    Args:
        q1 (int): Index value for the controlling qubit for CNOT gate
        q2 (int): Index value for the target qubit for CNOT gate
        gamma (float): Rotation angle to apply parameterized rotation around z

    Returns:
        Circuit: Circuit object that implements ZZ gate using CNOT gates
    """

    # get a circuit
    circ_zz = Circuit()

    # construct decomposition of ZZ
    circ_zz.cnot(q1, q2).rz(q2, gamma).cnot(q1, q2)

    return circ_zz


def _driver(beta: float, n_qubits: int) -> Circuit:
    """Returns circuit for driver Hamiltonian U(Hb, beta)

    Args:
        beta (float): Rotation angle to apply parameterized rotation around x
        n_qubits (int): number of qubits to apply rx gate

    Returns:
        Circuit: Circuit object that implements evolution with driver Hamiltonian
    """
    return Circuit().rx(range(n_qubits), 2 * beta)


def _cost_circuit(
    gamma: float,  ising: np.ndarray, device: Union[AwsDevice, LocalSimulator]
) -> Circuit:
    """Returns circuit for evolution with cost Hamiltonian

    Args:
        gamma (float): Rotation angle to apply parameterized rotation around z
        n_qubits (int): number of qubits
        ising (np.ndarray): Ising matrix
        device (Union[AwsDevice, LocalSimulator]): AwsDevice or LocalSimulator to run the circuit on

    Returns:
        Circuit: Circuit for evolution with cost Hamiltonian
    """
    # instantiate circuit object
    circ = Circuit()

    # get all non-zero entries (edges) from Ising matrix
    idx = ising.nonzero()
    edges = list(zip(idx[0], idx[1]))

    # apply ZZ gate for every edge (with corresponding interaction strength)
    for qubit_pair in edges:
        # get interaction strength from Ising matrix
        int_strength = ising[qubit_pair[0], qubit_pair[1]]
        # for Rigetti we decompose ZZ using CNOT gates
        if isinstance(device, AwsDevice) and device.provider_name == "Rigetti":
            gate = _zzgate(qubit_pair[0], qubit_pair[1], gamma * int_strength)
            circ.add(gate)
        # classical simulators and IonQ support ZZ gate
        else:
            gate = Circuit().zz(qubit_pair[0], qubit_pair[1], angle=2 * gamma * int_strength)
            circ.add(gate)

    return circ


@circuit.subroutine(register=True)
def qaoa(
    params: List[float], device: Union[AwsDevice, LocalSimulator], n_qubits: int, ising: np.ndarray
) -> Circuit:
    """function to return full QAOA circuit;
    depends on device as ZZ implementation depends on gate set of backend

    Args:
        params (List[float]): Parameters for the rotation gates
        device (Union[AwsDevice, LocalSimulator]): AwsDevice or LocalSimulator to run the circuit on
        n_qubits (int): number of qubits
        ising (np.ndarray): Ising matrix

    Returns:
        Circuit: QAOA circuit
    """

    # initialize qaoa circuit with first Hadamard layer: for minimization start in |->
    circ = Circuit()
    circ.x(range(0, n_qubits))
    circ.h(range(0, n_qubits))

    # setup two parameter families
    circuit_length = int(len(params) / 2)
    gammas = params[:circuit_length]
    betas = params[circuit_length:]

    # add QAOA circuit layer blocks
    for mm in range(circuit_length):
        circ.add(_cost_circuit(gammas[mm], n_qubits, ising, device))
        circ.add(_driver(betas[mm], n_qubits))

    return circ


def _objective_function(
    params: List[float],
    device: Union[AwsDevice, LocalSimulator],
    ising: np.ndarray,
    n_qubits: int,
    n_shots: int,
    tracker: Dict[str, Any],
    verbose: bool = False,
) -> float:
    """objective function takes a list of variational parameters as input,
    and returns the cost associated with those parameters

    Args:
        params (List[float]): Parameters for the rotation gates
        device (Union[AwsDevice, LocalSimulator]): AwsDevice or LocalSimulator to run the circuit on
        ising (np.ndarray): Ising matrix
        n_qubits (int): number of qubits
        n_shots (int): Number of shots
        tracker (Dict[str, Any]): Tracker to keep track of results
        verbose (bool): Enable or disable verbose logging

    Returns:
        float: Final average energy cost
    """

    # get a quantum circuit instance from the parameters
    qaoa_circuit = qaoa(params, device, n_qubits, ising)

    # classically simulate the circuit
    # execute the correct device.run call depending on whether the backend is local or cloud based
    if isinstance(device, LocalSimulator):
        task = device.run(qaoa_circuit, shots=n_shots)
    else:
        task = device.run(qaoa_circuit, shots=n_shots, poll_timeout_seconds=3 * 24 * 60 * 60)

    # get result for this task
    result = task.result()

    # convert results (0 and 1) to ising (-1 and 1)
    meas_ising = result.measurements
    meas_ising[meas_ising == 0] = -1

    # get all energies (for every shot): (n_shots, 1) vector
    all_energies = np.diag(np.dot(meas_ising, np.dot(ising, np.transpose(meas_ising))))

    # find minimum and corresponding classical string
    energy_min = np.min(all_energies)
    tracker["opt_energies"].append(energy_min)
    optimal_string = meas_ising[np.argmin(all_energies)]
    tracker["opt_bitstrings"].append(optimal_string)

    # store optimal (classical) result/bitstring
    if energy_min < tracker["optimal_energy"]:
        tracker.update({"optimal_energy": energy_min})
        tracker.update({"optimal_bitstring": optimal_string})

    # store global minimum
    tracker["global_energies"].append(tracker["optimal_energy"])

    # energy expectation value
    energy_expect = np.sum(all_energies) / n_shots

    # update tracker
    tracker.update({"count": tracker["count"] + 1, "res": result})
    tracker["costs"].append(energy_expect)
    tracker["params"].append(params)

    return energy_expect


def run_qaoa(
    device: Union[AwsDevice, LocalSimulator],
    options: Dict[str, Any],
    p: int,
    ising: np.ndarray,
    n_qubits: int,
    n_shots: int,
    opt_method: str,
    tracker: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """function to run QAOA algorithm for given, fixed circuit depth p

    Args:
        device (Union[AwsDevice, LocalSimulator]): AwsDevice or LocalSimulator to run the circuit on
        options (Dict[str, Any]): options for classical optimization
        p (int): circuit depth for QAOA
        ising (np.ndarray): Ising matrix
        n_qubits (int): Number of qubits
        n_shots (int): Number of shots
        opt_method (str): Optimization technique
        tracker (Dict[str, Any]): Tracker to keep track of results
        verbose (bool): Enable or disable logging

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Results and tracker output

    """
    print("Training in progress...")

    # randomly initialize variational parameters within appropriate bounds
    gamma_initial = np.random.uniform(0, 2 * np.pi, p).tolist()
    beta_initial = np.random.uniform(0, np.pi, p).tolist()
    params0 = np.array(gamma_initial + beta_initial)

    # set bounds for search space
    bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
    bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
    bnds = bnds_gamma + bnds_beta

    tracker["params"].append(params0)

    # run classical optimization (example: method='Nelder-Mead')
    result = minimize(
        _objective_function,
        params0,
        args=(device, ising, n_qubits, n_shots, tracker),
        options=options,
        method=opt_method,
        bounds=bnds,
    )

    return result, tracker


def get_qaoa_results(result: Dict[str, Any], tracker: Dict[str, Any]) -> None:
    """Function to postprocess dictionary returned by run_qaoa and pretty print results

    Args:
        result (Dict[str, Any]): Results associated with QAOA optimization
        tracker (Dict[str, Any]): Results tracked by tracker

    """

    print("Optimal energy:", tracker["optimal_energy"])
    print("Optimal classical bitstring:", tracker["optimal_bitstring"])
    result_energy = result.fun
    print("Final average energy (cost):", result_energy)
    result_angle = result.x
    print("Final angles:", result_angle)
