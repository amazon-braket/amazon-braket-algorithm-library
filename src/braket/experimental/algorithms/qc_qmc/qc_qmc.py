import multiprocessing as mp
import os
from typing import Callable, List, Tuple

import numpy as np
import pennylane as qml
from openfermion.linalg.givens_rotations import givens_decomposition_square

from braket.experimental.algorithms.qc_qmc.classical_qmc import (
    ChemicalProperties,
    greens_pq,
    hartree_fock_energy,
    imag_time_propogator,
    local_energy,
    propagate_walker,
    reortho,
)

np.seterr(divide="ignore", invalid="ignore")  # ignore divide by zero


def qc_qmc(
    num_walkers: int,
    num_steps: int,
    dtau: float,
    quantum_evaluations_every_n_steps: int,
    trial: np.ndarray,
    prop: ChemicalProperties,
    trial_state_circuit: Callable,
    dev: qml.Device,
    max_pool: int = 8,
) -> Tuple[List[float], List[float]]:
    """Quantum assisted Auxiliary-Field Quantum Monte Carlo.

    Args:
        num_walkers (int): Number of walkers.
        num_steps (int): Number of (imaginary) time steps
        dtau (float): Increment of each time step
        quantum_evaluations_every_n_steps (int): How often to evaluate the energy using quantum
        trial (ndarray): Trial wavefunction.
        prop (ChemicalProperties): Chemical properties.
        trial_state_circuit (Callable): quantum trial state as a pennylane quantum function
        dev (qml.Device): Pennylane device to run circuits on.
        max_pool (int): Max workers. Defaults to 8.

    Returns:
        Tuple[List[float], List[float]]: quantum and classical energies
    """
    e_hf = hartree_fock_energy(trial, prop)
    walkers = [trial] * num_walkers
    weights = [1.0] * num_walkers

    inputs = [
        (
            num_steps,
            quantum_evaluations_every_n_steps,
            dtau,
            trial,
            prop,
            e_hf,
            walker,
            weight,
            trial_state_circuit,
            dev,
        )
        for walker, weight in zip(walkers, weights)
    ]

    # parallelize with multiprocessing
    with mp.Pool(max_pool) as pool:
        results = list(pool.map(q_full_imag_time_evolution_wrapper, inputs))

    local_energies, weights, nums, denoms = map(np.array, zip(*results))

    energies = np.real(np.average(local_energies, weights=weights, axis=0))

    # post-processing to include quantum energy evaluations
    # this will have many np.nans, but it's okay
    quantum_energies = np.real((weights * nums).mean(0) / (weights * denoms).mean(0))
    for q_step in range(0, num_steps, quantum_evaluations_every_n_steps):
        energies[q_step] = quantum_energies[q_step]
    quantum_energies = quantum_energies[~np.isnan(quantum_energies)]  # remove nans
    return quantum_energies, energies


def q_full_imag_time_evolution_wrapper(args: Tuple) -> Callable:
    return q_full_imag_time_evolution(*args)


def q_full_imag_time_evolution(
    num_steps: int,
    quantum_evaluations_every_n_steps: int,
    dtau: float,
    trial: np.ndarray,
    prop: ChemicalProperties,
    e_shift: float,
    walker: np.ndarray,
    weight: float,
    trial_state_circuit: Callable,
    dev: qml.Device,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Imaginary time evolution of a single walker.

    Args:
        num_steps (int): number of time steps
        quantum_evaluations_every_n_steps (int): between how many steps to do a quantum evaluation
        dtau (float): imaginary time step size
        trial (ndarray): trial state as np.ndarray, e.g., for h2 HartreeFock state, it is
            np.array([[1,0], [0,1], [0,0], [0,0]])
        prop (ChemicalProperties): Chemical properties.
        e_shift (float): Reference energy, i.e. Hartree-Fock energy
        walker (ndarray): normalized walker state as np.ndarray, others are the same as trial
        weight (float): weight for sampling.
        trial_state_circuit (Callable): quantum trial state
        dev (qml.Device): `qml.device('lightning.qubit', wires=wires)` for simulator;
            or `qml.device('braket.aws.qubit', device_arn=device_arn, wires=wires, shots=shots)`
            for quantum device;

    Returns:
        Tuple[List[float],List[float],List[float],List[float]]: energy_list, weights, qs, cs
    """
    # random seed for mutliprocessing
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

    energy_list, weights, qs, cs = [], [], [], []
    for time in range(num_steps):
        # If the time step is in the quantum times, evaluate the energy with quantum
        if time % quantum_evaluations_every_n_steps == 0:
            # if time * dtau in quantum_times:
            e_loc, num, denom, walker, weight = imag_time_propogator_qaee(
                dtau, trial, walker, weight, prop, e_shift, trial_state_circuit, dev
            )
        else:  # otherwise, do classical energy
            e_loc, walker, weight = imag_time_propogator(dtau, trial, walker, weight, prop, e_shift)
            num = 0
            denom = 0
        energy_list.append(e_loc)
        weights.append(weight)
        qs.append(num)
        cs.append(denom)
    return energy_list, weights, qs, cs


def imag_time_propogator_qaee(
    dtau: float,
    trial: np.ndarray,
    walker: np.ndarray,
    weight: float,
    prop: ChemicalProperties,
    e_shift: float,
    trial_state_circuit: Callable,
    dev: qml.Device,
) -> Tuple[float, float, float, np.ndarray, float]:
    """Imaginary time propogator with quantum energy evaluations.

    Args:
        dtau (float): imaginary time step size
        trial (ndarray): trial state as np.ndarray, e.g., for h2 HartreeFock state,
            it is np.array([[1,0], [0,1], [0,0], [0,0]])
        walker (ndarray): normalized walker state as np.ndarray, others are the same as trial
        weight (float): weight for sampling.
        prop (ChemicalProperties): Chemical properties.
        e_shift (float): Reference energy, i.e. Hartree-Fock energy
        trial_state_circuit (Callable): quantum trial state
        dev (qml.Device): Pennylane device

    Returns:
        Tuple[float, float, float, ndarray, float]: propogatpr results
        e_loc: local energy
        e_loc_q / c_ovlp: numerator
        q_ovlp / c_ovlp: denominator for evaluation of total energy
        new_walker: new walker for the next time step
        new_weight: new weight for the next time step
    """
    # First compute the bias force using the expectation value of L operators
    num_spin_orbitals, num_electrons = trial.shape
    num_fields = len(prop.v_gamma)
    np.identity(num_spin_orbitals)
    # compute the overlap integral
    ovlp = np.linalg.det(trial.transpose().conj() @ walker)

    trial_up = trial[::2, ::2]
    trial_down = trial[1::2, 1::2]
    walker_up = walker[::2, ::2]
    walker_down = walker[1::2, 1::2]
    green_funcs = [greens_pq(trial_up, walker_up), greens_pq(trial_down, walker_down)]
    e_loc = local_energy(prop.h1e, prop.eri, green_funcs, prop.nuclear_repulsion)

    # Quantum-assisted energy evaluation
    # compute the overlap between qtrial state and walker
    c_ovlp = np.linalg.det(trial.transpose().conj() @ walker)
    q_ovlp = amplitude_estimate(walker, trial_state_circuit, dev)
    e_loc_q = (
        local_energy_quantum(
            walker, q_ovlp, prop.h_chem, prop.lambda_l, prop.u_l, trial_state_circuit, dev
        )
        + q_ovlp * prop.nuclear_repulsion
    )

    # update the walker
    x = np.random.normal(0.0, 1.0, size=num_fields)
    new_walker = propagate_walker(
        x, prop.v_0, prop.v_gamma, prop.mf_shift, dtau, trial, walker, green_funcs
    )

    # Define the I operator and find new weight
    new_ovlp = np.linalg.det(trial.transpose().conj() @ new_walker)
    arg = np.angle(new_ovlp / ovlp)
    new_weight = weight * np.exp(-dtau * (np.real(e_loc) - e_shift)) * np.max([0.0, np.cos(arg)])

    numerator = e_loc_q / c_ovlp
    denominator = q_ovlp / c_ovlp
    return e_loc, numerator, denominator, new_walker, new_weight


def local_energy_quantum(  # noqa: C901
    walker: np.ndarray,
    ovlp: float,
    one_body: np.ndarray,
    lambda_l: np.ndarray,
    u_l: np.ndarray,
    trial_state_circuit: Callable,
    dev: qml.device,
) -> complex:
    r"""This function estimates the integral $\\langle \\Psi_Q|H|\\phi_l\rangle$ with rotated basis.

    Args:
        walker (ndarray): np.ndarray; matrix representation of the walker state, not necessarily
            orthonormalized.
        ovlp (float): amplitude between walker and the quantum trial state
        one_body (ndarray): (corrected) one-body term in the second quantized hamiltonian
            written in chemist's notation. This term is assumed to be diagonal in the current
            implementation, but should be rather straight forward to generalize if it's not.
        lambda_l (ndarray): eigenvalues of Cholesky vectors
        u_l (ndarray): eigenvectors of Cholesky vectors
        trial_state_circuit (Callable): quantum trial state
        dev (qml.device): `qml.device('lightning.qubit', wires=wires)` for simulator;
            or `qml.device('braket.aws.qubit', device_arn=device_arn, wires=wires, shots=shots)`
            for quantum device;

    Returns:
        complex: energy
    """
    energy = 0.0 + 0.0j
    num_qubits, num_particles = walker.shape

    # one-body term assuming diagonal form already
    Id = np.identity(num_qubits)
    dictionary = {}
    for i in range(num_qubits):
        dictionary[i] = pauli_estimate(walker, trial_state_circuit, Id, [i], dev)
        for j in range(i + 1, num_qubits):
            dictionary[(i, j)] = pauli_estimate(walker, trial_state_circuit, Id, [i, j], dev)

    for i in range(num_qubits):
        expectation_value = 0.5 * (ovlp - dictionary.get(i))
        energy += one_body[i, i] * expectation_value

    # Cholesky decomposed two-body term
    for lamb, u_matrix in zip(lambda_l, u_l):
        # define a dictionary to store all the expectation values
        if np.count_nonzero(np.round(u_matrix - np.diag(np.diagonal(u_matrix)), 7)) == 0:
            new_dict = dictionary
        else:
            new_dict = {}
            for i in range(num_qubits):
                new_dict[i] = pauli_estimate(walker, trial_state_circuit, u_matrix, [i], dev)
                for j in range(i, num_qubits):
                    new_dict[(i, j)] = pauli_estimate(
                        walker, trial_state_circuit, u_matrix, [i, j], dev
                    )

        for i in range(num_qubits):
            for j in range(i, num_qubits):
                if i == j:
                    expectation_value = 0.5 * (ovlp - new_dict.get(i))
                else:
                    expectation_value = 0.5 * (
                        ovlp - new_dict.get(i) - new_dict.get(j) + new_dict.get((i, j))
                    )
                energy += 0.5 * lamb[i] * lamb[j] * expectation_value
    return energy


def givens_block_circuit(givens: Tuple) -> None:
    r"""This function defines the Givens rotation circuit from a single givens tuple.

    Args:
        givens (Tuple): (i, j, \theta, \varphi)
    """
    (i, j, theta, varphi) = givens

    qml.RZ(-varphi, wires=j)
    qml.CNOT(wires=[j, i])

    # implement the cry rotation
    qml.RY(theta, wires=j)
    qml.CNOT(wires=[i, j])
    qml.RY(-theta, wires=j)
    qml.CNOT(wires=[i, j])

    qml.CNOT(wires=[j, i])


def prepare_slater_circuit(circuit_description: List[Tuple]) -> None:
    """Creating Givens rotation circuit to prepare arbitrary Slater determinant.

    Args:
        circuit_description (List[Tuple]): list of tuples containing Givens rotation
            (i, j, theta, phi) in reversed order.
    """

    for parallel_ops in circuit_description:
        for givens in parallel_ops:
            qml.adjoint(givens_block_circuit)(givens)


def circuit_first_half(q_state: np.ndarray) -> None:
    """Construct the first half of the vacuum reference circuit.

    Args:
        q_state (ndarray): orthonormalized walker state
    """
    num_qubits, num_particles = q_state.shape
    qml.Hadamard(wires=0)

    for i in range(1, num_particles):
        qml.CNOT(wires=[0, i])

    complement = np.ones((num_qubits, num_qubits - num_particles))
    w_matrix, _ = reortho(np.hstack((q_state, complement)))
    decomposition, diagonal = givens_decomposition_square(w_matrix.T)
    circuit_description = list(reversed(decomposition))

    for i in range(len(diagonal)):
        qml.RZ(np.angle(diagonal[i]), wires=i)

    prepare_slater_circuit(circuit_description)


def circuit_second_half_real(q_state: np.ndarray, trial_state_circuit: Callable) -> None:
    """Construct the second half of the vacuum reference circuit (for real expectation values)

    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
    """
    num_qubits, num_particles = q_state.shape
    qml.adjoint(trial_state_circuit)()

    for i in range(1, num_particles)[::-1]:
        qml.CNOT(wires=[0, i])
    qml.Hadamard(wires=0)


def circuit_second_half_imag(q_state: np.ndarray, trial_state_circuit: Callable) -> None:
    """Construct the second half of the vacuum reference circuit (for imaginary expectation values)
    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
    """
    num_qubits, num_particles = q_state.shape
    qml.adjoint(trial_state_circuit)()

    for i in range(1, num_particles)[::-1]:
        qml.CNOT(wires=[0, i])

    qml.S(wires=0)
    qml.S(wires=0)
    qml.S(wires=0)
    qml.Hadamard(wires=0)


def amplitude_real(q_state: np.ndarray, trial_state_circuit: Callable) -> None:
    """Construct the the vacuum reference circuit for measuring amplitude real part
    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
    """
    circuit_first_half(q_state)
    circuit_second_half_real(q_state, trial_state_circuit)


def amplitude_imag(q_state: np.ndarray, trial_state_circuit: Callable) -> None:
    """Construct the the vacuum reference circuit for measuring amplitude imaginary part
    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
    """
    circuit_first_half(q_state)
    circuit_second_half_imag(q_state, trial_state_circuit)


def amplitude_estimate(
    q_state: np.ndarray, trial_state_circuit: Callable, dev: qml.device
) -> np.complex128:
    """This function computes the amplitude between walker state and quantum trial state.

    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
        dev (qml.device): `qml.device('lightning.qubit', wires=wires)` for simulator;
            or `qml.device('braket.aws.qubit', device_arn=device_arn, wires=wires, shots=shots)`
            for quantum device;
    Returns:
        complex128: amplitude
    """
    num_qubits, num_particles = q_state.shape

    @qml.qnode(dev, interface=None, diff_method=None)
    def __compute_real(q_state, trial_state_circuit):
        amplitude_real(q_state, trial_state_circuit)
        return qml.probs(range(num_qubits))

    probs_values = __compute_real(q_state, trial_state_circuit)
    real = probs_values[0] - probs_values[int(2**num_qubits / 2)]

    @qml.qnode(dev, interface=None, diff_method=None)
    def __compute_imag(q_state, trial_state_circuit):
        amplitude_imag(q_state, trial_state_circuit)
        return qml.probs(range(num_qubits))

    probs_values = __compute_imag(q_state, trial_state_circuit)
    imag = probs_values[0] - probs_values[int(2**num_qubits / 2)]

    return real + 1.0j * imag


def u_circuit(u_matrix: np.ndarray) -> None:
    """Construct circuit to perform unitary transformation U.

    Args:
        u_matrix (ndarray): unitary
    """

    decomposition, diagonal = givens_decomposition_square(u_matrix)
    circuit_description = list(reversed(decomposition))

    for i in range(len(diagonal)):
        qml.RZ(np.angle(diagonal[i]), i)

    if circuit_description != []:
        prepare_slater_circuit(circuit_description)


def pauli_real(
    q_state: np.ndarray, trial_state_circuit: Callable, u_matrix: np.ndarray, pauli: List[int]
) -> Callable:
    """Construct the the vacuum reference circuit for measuring expectation value
        of a pauli real part
    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
        u_matrix (ndarray): unitary transformation to change the Pauli into Z basis
        pauli (List[int]): list that stores the position of the Z gate, e.g., [0,1]
            represents 'ZZII'.

    Returns:
        Callable: pennylane circuit
    """
    circuit_first_half(q_state)

    u_circuit(u_matrix)
    for i in pauli:
        qml.PauliZ(wires=i)

    qml.adjoint(u_circuit)(u_matrix)
    circuit_second_half_real(q_state, trial_state_circuit)


def pauli_imag(
    q_state: np.ndarray, trial_state_circuit: Callable, u_matrix: np.ndarray, pauli: List[int]
) -> Callable:
    """Construct the the vacuum reference circuit for measuring expectation value
        of a pauli imaginary part
    Args:
        q_state (ndarray): orthonormalized walker state
        trial_state_circuit (Callable): quantum trial state
        u_matrix (ndarray): unitary transformation to change the Pauli into Z basis
        pauli (List[int]): list that stores the position of the Z gate, e.g., [0,1]
            represents 'ZZII'.

    Returns:
        Callable: pennylane circuit
    """
    circuit_first_half(q_state)

    u_circuit(u_matrix)
    for i in pauli:
        qml.PauliZ(wires=i)

    qml.adjoint(u_circuit)(u_matrix)
    circuit_second_half_imag(q_state, trial_state_circuit)


def pauli_estimate(
    q_state: np.ndarray,
    trial_state_circuit: Callable,
    u_matrix: np.ndarray,
    pauli: List[int],
    dev: qml.device,
) -> float:
    """This function returns the expectation value of $\\langle \\Psi_q_state|pauli|\\phi_l\rangle$.
    Args:
        q_state (ndarray): np.ndarray; matrix representation of the walker state, not necessarily
            orthonormalized.
        trial_state_circuit (Callable): circuit unitary to prepare the quantum trial state
        u_matrix (ndarray): eigenvector of Cholesky vectors, $L = U \\lambda U^{\\dagger}$
        pauli (List[int]): list of 0 and 1 as the representation of a Pauli string,
            e.g., [0,1] represents 'ZZII'.
        dev (qml.device): `qml.device('lightning.qubit', wires=wires)` for simulator;
            or `qml.device('braket.aws.qubit', device_arn=device_arn, wires=wires, shots=shots)`
            for quantum device;

    Returns:
        float: expectation value
    """
    num_qubits, num_particles = q_state.shape

    @qml.qnode(dev, interface=None, diff_method=None)
    def __compute_real(q_state, trial_state_circuit, u_matrix, pauli):
        pauli_real(q_state, trial_state_circuit, u_matrix, pauli)
        return qml.probs(range(num_qubits))

    probs_values = __compute_real(q_state, trial_state_circuit, u_matrix, pauli)
    real = probs_values[0] - probs_values[int(2**num_qubits / 2)]

    @qml.qnode(dev, interface=None, diff_method=None)
    def __compute_real(q_state, trial_state_circuit, u_matrix, pauli):
        pauli_imag(q_state, trial_state_circuit, u_matrix, pauli)
        return qml.probs(range(num_qubits))

    probs_values = __compute_real(q_state, trial_state_circuit, u_matrix, pauli)
    imag = probs_values[0] - probs_values[int(2**num_qubits / 2)]

    return real + 1.0j * imag
