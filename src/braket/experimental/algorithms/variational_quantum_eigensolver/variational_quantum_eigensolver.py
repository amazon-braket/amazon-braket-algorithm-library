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

import re
from functools import reduce
from itertools import combinations, pairwise
from typing import Any

import numpy as np
from openfermion import FermionOperator, hermitian_conjugated, jordan_wigner
from scipy.optimize import minimize

from braket.circuits import Circuit, FreeParameter, FreeParameterExpression, Instruction, Observable
from braket.devices import Device

# A Hamiltonian is a mapping from Pauli strings to real coefficients, e.g.
# {"II": -1.05, "ZI": 0.39, "XX": 0.18}. The i-th character acts on qubit i.
Hamiltonian = dict[str, float]

CHEMICAL_ACCURACY = 1.6e-3  # Hartree

_PAULI_OBSERVABLES = {"X": Observable.X, "Y": Observable.Y, "Z": Observable.Z}
_PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


def hardware_efficient_ansatz(
    n_qubits: int, n_layers: int, initial_state: list[int] | None = None
) -> Circuit:
    """Hardware-efficient ansatz made of RY rotations and a linear chain of CNOTs.

    Each layer applies an RY rotation to every qubit followed by CNOTs between
    neighbouring qubits. A final layer of RY rotations is appended, so the circuit has
    n_qubits * (n_layers + 1) free parameters named theta_0, theta_1, ...

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of entangling layers.
        initial_state (list[int] | None): Computational basis state to start from, e.g.
            the Hartree-Fock state [1, 1, 0, 0]. Defaults to the all-zero state.

    Returns:
        Circuit: The parametric ansatz circuit.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative.")
    if initial_state is not None and len(initial_state) != n_qubits:
        raise ValueError("initial_state must have one entry per qubit.")

    circ = Circuit()
    if initial_state is not None:
        for qubit, bit in enumerate(initial_state):
            if bit:
                circ.x(qubit)

    parameter_index = 0
    for layer in range(n_layers + 1):
        for qubit in range(n_qubits):
            circ.ry(qubit, FreeParameter(f"theta_{parameter_index}"))
            parameter_index += 1
        if layer < n_layers:
            for qubit in range(n_qubits - 1):
                circ.cnot(qubit, qubit + 1)
    return circ


def pauli_rotation(pauli_string: str, angle: float | FreeParameterExpression) -> Circuit:
    """Circuit implementing exp(-i * angle / 2 * P) for a Pauli string P.

    Each qubit in the support of P is rotated into the Z basis, the parity of the
    support is accumulated with a CNOT ladder, an RZ rotation is applied and the
    ladder and basis changes are undone.

    Args:
        pauli_string (str): Pauli string, e.g. "XZY". The i-th character acts on qubit i.
        angle (float | FreeParameterExpression): Rotation angle.

    Returns:
        Circuit: The Pauli rotation circuit.
    """
    _validate_hamiltonian({pauli_string: 1.0})
    support = [q for q, p in enumerate(pauli_string) if p != "I"]
    circ = Circuit()
    if not support:
        return circ.i(range(len(pauli_string)))

    def change_basis(sign: int) -> None:
        for qubit in support:
            if pauli_string[qubit] == "X":
                circ.h(qubit)
            elif pauli_string[qubit] == "Y":
                circ.rx(qubit, sign * np.pi / 2)

    change_basis(1)
    for control, target in pairwise(support):
        circ.cnot(control, target)
    circ.rz(support[-1], angle)
    for control, target in reversed(list(pairwise(support))):
        circ.cnot(control, target)
    change_basis(-1)
    return circ


def pauli_exponential_ansatz(
    generators: list[Hamiltonian], initial_state: list[int] | None = None
) -> Circuit:
    """Ansatz prod_k exp(-i * theta_k * G_k) for Hermitian generators G_k.

    Each generator is a sum of Pauli strings G_k = sum_j c_j P_j and is implemented as
    the product of Pauli rotations exp(-i * theta_k * c_j * P_j). This is exact when the
    Pauli strings of a generator commute, as is the case for fermionic excitations.

    Args:
        generators (list[Hamiltonian]): Hermitian generators, one per free parameter.
        initial_state (list[int] | None): Computational basis state to start from, e.g.
            the Hartree-Fock state [1, 1, 0, 0]. Defaults to the all-zero state.

    Returns:
        Circuit: The parametric ansatz circuit with parameters theta_0, theta_1, ...
    """
    if not generators:
        raise ValueError("At least one generator is required.")
    for generator in generators:
        _validate_hamiltonian(generator)
    n_qubits = len(next(iter(generators[0])))
    if any(len(p) != n_qubits for generator in generators for p in generator):
        raise ValueError("All generators must act on the same number of qubits.")
    if initial_state is not None and len(initial_state) != n_qubits:
        raise ValueError("initial_state must have one entry per qubit.")

    circ = Circuit()
    if initial_state is not None:
        for qubit, bit in enumerate(initial_state):
            if bit:
                circ.x(qubit)
    for index, generator in enumerate(generators):
        theta = FreeParameter(f"theta_{index}")
        for pauli_string, coefficient in generator.items():
            if set(pauli_string) != {"I"}:
                circ.add_circuit(pauli_rotation(pauli_string, 2 * coefficient * theta))
    return circ


def uccsd_generators(n_qubits: int, n_electrons: int) -> list[Hamiltonian]:
    """Jordan-Wigner encoded generators of the UCCSD ansatz.

    Spin orbitals are ordered with alternating spin (even indices spin up, odd indices
    spin down), as in OpenFermion, and the Hartree-Fock state occupies the first
    n_electrons spin orbitals. Only spin-conserving single and double excitations are
    included. Each excitation T gives the Hermitian generator G = i(T - T^dagger), so that
    exp(-i * theta * G) = exp(theta * (T - T^dagger)).

    Args:
        n_qubits (int): Number of spin orbitals (qubits).
        n_electrons (int): Number of electrons.

    Returns:
        list[Hamiltonian]: One generator per single and double excitation.
    """
    if not 0 < n_electrons < n_qubits:
        raise ValueError("n_electrons must be between 1 and n_qubits - 1.")
    occupied = range(n_electrons)
    virtual = range(n_electrons, n_qubits)

    excitations = [((a, 1), (i, 0)) for i in occupied for a in virtual if i % 2 == a % 2] + [
        ((a, 1), (b, 1), (j, 0), (i, 0))
        for i, j in combinations(occupied, 2)
        for a, b in combinations(virtual, 2)
        if sorted([i % 2, j % 2]) == sorted([a % 2, b % 2])
    ]

    generators = []
    for excitation in excitations:
        operator = FermionOperator(excitation)
        generator = jordan_wigner(1j * (operator - hermitian_conjugated(operator)))
        generators.append(hamiltonian_from_qubit_operator(generator, n_qubits))
    return generators


def hamiltonian_from_qubit_operator(qubit_operator: Any, n_qubits: int) -> Hamiltonian:
    """Convert an OpenFermion QubitOperator into a Hamiltonian dictionary.

    Args:
        qubit_operator (Any): An object with a `terms` attribute mapping tuples of
            (qubit, pauli) pairs to coefficients, such as an OpenFermion QubitOperator.
        n_qubits (int): Number of qubits the Hamiltonian acts on.

    Returns:
        Hamiltonian: Mapping from Pauli strings to real coefficients.
    """
    hamiltonian = {}
    for term, coefficient in qubit_operator.terms.items():
        if not np.isclose(np.imag(coefficient), 0):
            raise ValueError("Hamiltonian coefficients must be real.")
        pauli = ["I"] * n_qubits
        for qubit, operator in term:
            if qubit >= n_qubits:
                raise ValueError(f"Term {term} acts outside of {n_qubits} qubits.")
            pauli[qubit] = operator
        pauli_string = "".join(pauli)
        hamiltonian[pauli_string] = hamiltonian.get(pauli_string, 0.0) + float(np.real(coefficient))
    return hamiltonian


def hamiltonian_matrix(hamiltonian: Hamiltonian) -> np.ndarray:
    """Dense matrix representation of a Hamiltonian.

    Args:
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.

    Returns:
        np.ndarray: The 2**n x 2**n Hamiltonian matrix, with qubit 0 as the most
        significant bit.
    """
    _validate_hamiltonian(hamiltonian)
    return sum(
        coefficient * reduce(np.kron, [_PAULI_MATRICES[p] for p in pauli_string])
        for pauli_string, coefficient in hamiltonian.items()
    )


def exact_ground_state_energy(hamiltonian: Hamiltonian) -> float:
    """Lowest eigenvalue of a Hamiltonian obtained by exact diagonalization.

    Args:
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.

    Returns:
        float: The ground state energy.
    """
    return float(np.linalg.eigvalsh(hamiltonian_matrix(hamiltonian))[0])


def group_commuting_terms(hamiltonian: Hamiltonian) -> list[list[str]]:
    """Greedily partition the non-identity terms into qubit-wise commuting groups.

    Terms in the same group can be measured simultaneously from a single circuit.

    Args:
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.

    Returns:
        list[list[str]]: Groups of qubit-wise commuting Pauli strings.
    """
    _validate_hamiltonian(hamiltonian)
    groups = []
    terms = sorted((p for p in hamiltonian if set(p) != {"I"}), key=lambda p: -abs(hamiltonian[p]))
    for pauli_string in terms:
        for group in groups:
            if all(_qubit_wise_commute(pauli_string, member) for member in group):
                group.append(pauli_string)
                break
        else:
            groups.append([pauli_string])
    return groups


def vqe_circuits(ansatz: Circuit, hamiltonian: Hamiltonian, shots: int = 0) -> list[Circuit]:
    """Circuits that measure the energy of the ansatz state.

    For shots=0 a single circuit with an expectation result type per term is returned.
    For shots>0 one circuit per qubit-wise commuting group is returned, with basis
    rotations appended so that all terms of the group are read out in the Z basis.

    Args:
        ansatz (Circuit): Parametric circuit preparing the trial state.
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.
        shots (int): Number of shots. Defaults to 0.

    Returns:
        list[Circuit]: The measurement circuits.
    """
    _validate_hamiltonian(hamiltonian)
    n_qubits = len(next(iter(hamiltonian)))
    if ansatz.qubits and max(int(q) for q in ansatz.qubits) >= n_qubits:
        raise ValueError("The ansatz acts on more qubits than the Hamiltonian.")
    if ansatz.result_types:
        raise ValueError("The ansatz must not contain result types.")

    base = ansatz.copy()
    unused_qubits = sorted(set(range(n_qubits)) - {int(q) for q in ansatz.qubits})
    if unused_qubits:
        base.i(unused_qubits)

    if shots == 0:
        circ = base.copy()
        for pauli_string in hamiltonian:
            if set(pauli_string) != {"I"}:
                observable, targets = _pauli_observable(pauli_string)
                circ.expectation(observable=observable, target=targets)
        return [circ]

    circuits = []
    for group in group_commuting_terms(hamiltonian):
        circ = base.copy()
        for qubit in range(n_qubits):
            basis = next((p[qubit] for p in group if p[qubit] != "I"), "I")
            if basis != "I":
                for gate in _PAULI_OBSERVABLES[basis]().basis_rotation_gates:
                    circ.add_instruction(Instruction(gate, qubit))
        circuits.append(circ)
    return circuits


def evaluate_energy(
    device: Device,
    ansatz: Circuit,
    hamiltonian: Hamiltonian,
    parameter_values: np.ndarray,
    shots: int = 0,
) -> float:
    """Estimate the energy of the ansatz state for the given parameter values.

    Args:
        device (Device): Braket device to run on.
        ansatz (Circuit): Parametric circuit preparing the trial state.
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.
        parameter_values (np.ndarray): Values of the ansatz parameters, ordered as
            returned by `get_parameter_names`.
        shots (int): Number of shots per measurement circuit. Defaults to 0.

    Returns:
        float: The estimated energy.
    """
    names = get_parameter_names(ansatz)
    if len(parameter_values) != len(names):
        raise ValueError(f"Expected {len(names)} parameter values, got {len(parameter_values)}.")
    bindings = dict(zip(names, (float(v) for v in parameter_values)))
    circuits = [c.make_bound_circuit(bindings) for c in vqe_circuits(ansatz, hamiltonian, shots)]

    energy = sum(c for p, c in hamiltonian.items() if set(p) == {"I"})
    terms = [p for p in hamiltonian if set(p) != {"I"}]
    if not terms:
        return float(energy)
    if shots == 0:
        result = device.run(circuits[0], shots=0).result()
        return float(energy + sum(hamiltonian[p] * v for p, v in zip(terms, result.values)))

    tasks = [device.run(circ, shots=shots) for circ in circuits]
    for group, task in zip(group_commuting_terms(hamiltonian), tasks):
        result = task.result()
        columns = {int(q): i for i, q in enumerate(result.measured_qubits)}
        signs = 1 - 2 * np.asarray(result.measurements)
        for pauli_string in group:
            support = [columns[q] for q, p in enumerate(pauli_string) if p != "I"]
            energy += hamiltonian[pauli_string] * np.mean(np.prod(signs[:, support], axis=1))
    return float(energy)


def run_vqe(
    device: Device,
    hamiltonian: Hamiltonian,
    ansatz: Circuit,
    initial_parameters: np.ndarray,
    shots: int = 0,
    optimizer: str = "COBYLA",
    optimizer_options: dict | None = None,
) -> dict[str, Any]:
    """Run the variational quantum eigensolver.

    The ansatz parameters are optimized classically to minimize the energy
    <psi(theta)|H|psi(theta)> evaluated on the quantum device.

    Args:
        device (Device): Braket device to run on.
        hamiltonian (Hamiltonian): Mapping from Pauli strings to coefficients.
        ansatz (Circuit): Parametric circuit preparing the trial state.
        initial_parameters (np.ndarray): Starting parameter values, ordered as returned by
            `get_parameter_names`.
        shots (int): Number of shots per measurement circuit. Defaults to 0 (exact).
        optimizer (str): Any method accepted by `scipy.optimize.minimize`.
            Defaults to "COBYLA".
        optimizer_options (dict | None): Options passed to the optimizer.

    Returns:
        dict[str, Any]: Dictionary with the optimal energy, optimal parameters, the
        energy at every evaluation and the raw `scipy.optimize.OptimizeResult`.
    """
    energy_history = []

    def cost(parameter_values: np.ndarray) -> float:
        energy = evaluate_energy(device, ansatz, hamiltonian, parameter_values, shots)
        energy_history.append(energy)
        return energy

    optimizer_result = minimize(
        cost,
        np.asarray(initial_parameters, dtype=float),
        method=optimizer,
        options=optimizer_options,
    )
    return {
        "optimal_energy": float(optimizer_result.fun),
        "optimal_parameters": optimizer_result.x,
        "energy_history": energy_history,
        "optimizer_result": optimizer_result,
    }


def get_vqe_results(
    vqe_result: dict[str, Any], exact_energy: float | None = None, verbose: bool = True
) -> dict[str, Any]:
    """Summarize the output of `run_vqe`.

    Args:
        vqe_result (dict[str, Any]): Output of `run_vqe`.
        exact_energy (float | None): Reference energy to compare against, if known.
        verbose (bool): Whether to print the summary. Defaults to True.

    Returns:
        dict[str, Any]: The VQE energy, number of energy evaluations and, if a reference
        energy is given, the absolute error and whether chemical accuracy is reached.
    """
    summary = {
        "vqe_energy": vqe_result["optimal_energy"],
        "n_evaluations": len(vqe_result["energy_history"]),
    }
    if exact_energy is not None:
        error = abs(vqe_result["optimal_energy"] - exact_energy)
        summary.update(
            exact_energy=exact_energy,
            absolute_error=error,
            chemical_accuracy=bool(error < CHEMICAL_ACCURACY),
        )
    if verbose:
        print(f"VQE energy:          {summary['vqe_energy']:.6f}")
        print(f"Energy evaluations:  {summary['n_evaluations']}")
        if exact_energy is not None:
            print(f"Exact energy:        {exact_energy:.6f}")
            print(f"Absolute error:      {summary['absolute_error']:.2e}")
            print(f"Chemical accuracy:   {summary['chemical_accuracy']}")
    return summary


def get_parameter_names(circ: Circuit) -> list[str]:
    """Names of the free parameters of a circuit in natural sort order (theta_2 < theta_10).

    Args:
        circ (Circuit): A parametric circuit.

    Returns:
        list[str]: The sorted parameter names.
    """

    def natural_key(name: str) -> list:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

    return sorted((p.name for p in circ.parameters), key=natural_key)


def _validate_hamiltonian(hamiltonian: Hamiltonian) -> None:
    if not hamiltonian:
        raise ValueError("The Hamiltonian must contain at least one term.")
    lengths = {len(p) for p in hamiltonian}
    if len(lengths) != 1:
        raise ValueError("All Pauli strings must have the same length.")
    if not all(set(p) <= set("IXYZ") for p in hamiltonian):
        raise ValueError("Pauli strings may only contain the characters I, X, Y and Z.")


def _qubit_wise_commute(a: str, b: str) -> bool:
    return all(p == "I" or q == "I" or p == q for p, q in zip(a, b))


def _pauli_observable(pauli_string: str) -> tuple[Observable, list[int]]:
    targets = [q for q, p in enumerate(pauli_string) if p != "I"]
    observable = reduce(
        lambda a, b: a @ b, [_PAULI_OBSERVABLES[pauli_string[q]]() for q in targets]
    )
    return observable, targets
