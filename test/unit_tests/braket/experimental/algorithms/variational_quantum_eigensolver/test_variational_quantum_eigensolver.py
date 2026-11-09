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

from functools import reduce

import numpy as np
import openfermion as of
import pytest
from scipy.linalg import expm

from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from braket.experimental.algorithms.variational_quantum_eigensolver import (
    evaluate_energy,
    exact_ground_state_energy,
    get_parameter_names,
    get_vqe_results,
    group_commuting_terms,
    hamiltonian_from_qubit_operator,
    hamiltonian_matrix,
    hardware_efficient_ansatz,
    pauli_exponential_ansatz,
    pauli_rotation,
    run_vqe,
    uccsd_generators,
    vqe_circuits,
)

PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def _pauli_matrix(pauli_string):
    return reduce(np.kron, [PAULIS[p] for p in pauli_string])


def _h2_hamiltonian():
    molecule = of.MolecularData(
        [("H", (0, 0, 0)), ("H", (0, 0, 0.7414))], "sto-3g", 1, description="0.7414"
    )
    molecule.load()
    qubit_operator = of.jordan_wigner(of.get_fermion_operator(molecule.get_molecular_hamiltonian()))
    return molecule, hamiltonian_from_qubit_operator(qubit_operator, molecule.n_qubits)


def test_hardware_efficient_ansatz():
    circ = hardware_efficient_ansatz(3, 2, initial_state=[1, 0, 1])
    assert circ.qubit_count == 3
    assert get_parameter_names(circ) == [f"theta_{i}" for i in range(9)]
    assert [instr.operator.name for instr in circ.instructions[:2]] == ["X", "X"]
    assert sum(instr.operator.name == "CNot" for instr in circ.instructions) == 4


def test_hardware_efficient_ansatz_no_layers():
    circ = hardware_efficient_ansatz(2, 0)
    assert len(circ.parameters) == 2
    assert all(instr.operator.name == "Ry" for instr in circ.instructions)


@pytest.mark.parametrize(
    "n_qubits, n_layers, initial_state",
    [(0, 1, None), (2, -1, None), (2, 1, [1, 0, 0])],
)
def test_hardware_efficient_ansatz_invalid(n_qubits, n_layers, initial_state):
    with pytest.raises(ValueError):
        hardware_efficient_ansatz(n_qubits, n_layers, initial_state)


@pytest.mark.parametrize("pauli_string", ["X", "Y", "Z", "XY", "ZX", "XZY", "IYI", "YIIX"])
def test_pauli_rotation(pauli_string):
    angle = 0.73
    circ = Circuit().i(range(len(pauli_string))).add_circuit(pauli_rotation(pauli_string, angle))
    expected = expm(-1j * angle / 2 * _pauli_matrix(pauli_string))
    assert np.allclose(circ.to_unitary(), expected)


def test_pauli_rotation_identity():
    circ = pauli_rotation("II", 0.5)
    assert np.allclose(circ.to_unitary(), np.eye(4))


def test_pauli_exponential_ansatz():
    generators = [{"XY": 0.5, "YX": -0.5}, {"ZI": 1.0, "II": 2.0}]
    circ = pauli_exponential_ansatz(generators, initial_state=[1, 0])
    assert get_parameter_names(circ) == ["theta_0", "theta_1"]

    theta = [0.3, -0.2]
    bound = circ.make_bound_circuit(dict(zip(get_parameter_names(circ), theta)))
    expected = np.eye(4)
    for t, generator in zip(theta, generators):
        matrix = sum(c * _pauli_matrix(p) for p, c in generator.items() if set(p) != {"I"})
        expected = expm(-1j * t * matrix) @ expected
    expected = expected @ np.kron(PAULIS["X"], PAULIS["I"])
    assert np.allclose(bound.to_unitary(), expected)


@pytest.mark.parametrize(
    "generators, initial_state",
    [([], None), ([{"X": 1.0}, {"XX": 1.0}], None), ([{"XX": 1.0}], [1])],
)
def test_pauli_exponential_ansatz_invalid(generators, initial_state):
    with pytest.raises(ValueError):
        pauli_exponential_ansatz(generators, initial_state)


def test_uccsd_generators():
    generators = uccsd_generators(4, 2)
    assert len(generators) == 3
    for generator in generators:
        matrix = hamiltonian_matrix(generator)
        assert np.allclose(matrix, matrix.conj().T)
    assert len(uccsd_generators(6, 2)) == 4 + 4


@pytest.mark.parametrize("n_qubits, n_electrons", [(4, 0), (4, 4)])
def test_uccsd_generators_invalid(n_qubits, n_electrons):
    with pytest.raises(ValueError):
        uccsd_generators(n_qubits, n_electrons)


def test_hamiltonian_from_qubit_operator():
    qubit_operator = (
        of.QubitOperator("X0 Z2", 0.5) + of.QubitOperator("", -1.0) + of.QubitOperator("Y1", 0.25)
    )
    hamiltonian = hamiltonian_from_qubit_operator(qubit_operator, 3)
    assert hamiltonian == {"XIZ": 0.5, "III": -1.0, "IYI": 0.25}


@pytest.mark.parametrize(
    "qubit_operator, n_qubits",
    [(of.QubitOperator("X0", 1j), 1), (of.QubitOperator("X3", 1.0), 2)],
)
def test_hamiltonian_from_qubit_operator_invalid(qubit_operator, n_qubits):
    with pytest.raises(ValueError):
        hamiltonian_from_qubit_operator(qubit_operator, n_qubits)


def test_hamiltonian_matrix_matches_openfermion():
    molecule, hamiltonian = _h2_hamiltonian()
    qubit_operator = of.jordan_wigner(of.get_fermion_operator(molecule.get_molecular_hamiltonian()))
    expected = of.get_sparse_operator(qubit_operator, molecule.n_qubits).toarray()
    assert np.allclose(hamiltonian_matrix(hamiltonian), expected)
    assert np.isclose(exact_ground_state_energy(hamiltonian), molecule.fci_energy)


def test_exact_ground_state_energy():
    assert np.isclose(exact_ground_state_energy({"XX": 1.0, "ZZ": 1.0}), -2)
    assert np.isclose(exact_ground_state_energy({"Z": 1.0, "I": 0.5}), -0.5)


@pytest.mark.parametrize("hamiltonian", [{}, {"X": 1.0, "XX": 1.0}, {"XA": 1.0}])
def test_invalid_hamiltonian(hamiltonian):
    with pytest.raises(ValueError):
        exact_ground_state_energy(hamiltonian)


def test_group_commuting_terms():
    hamiltonian = {"II": 1.0, "ZI": 0.1, "IZ": 0.2, "ZZ": 0.3, "XX": 0.4, "YY": 0.5, "XI": 0.05}
    groups = group_commuting_terms(hamiltonian)
    assert sorted(p for group in groups for p in group) == sorted(set(hamiltonian) - {"II"})
    for group in groups:
        for a in group:
            for b in group:
                assert all(p == "I" or q == "I" or p == q for p, q in zip(a, b))
    assert len(groups) == 3


def test_vqe_circuits_exact():
    ansatz = hardware_efficient_ansatz(2, 1)
    circuits = vqe_circuits(ansatz, {"II": 1.0, "ZI": 0.5, "XX": 0.5, "YY": 0.5})
    assert len(circuits) == 1
    assert len(circuits[0].result_types) == 3


def test_vqe_circuits_sampled():
    ansatz = hardware_efficient_ansatz(2, 1)
    circuits = vqe_circuits(ansatz, {"ZI": 0.5, "XX": 0.5, "YY": 0.5}, shots=100)
    assert len(circuits) == 3
    assert all(not circ.result_types for circ in circuits)


def test_vqe_circuits_pads_unused_qubits():
    ansatz = Circuit().ry(0, FreeParameter("theta_0"))
    circuits = vqe_circuits(ansatz, {"ZZ": 1.0}, shots=10)
    assert circuits[0].qubit_count == 2


def test_vqe_circuits_ansatz_too_large():
    with pytest.raises(ValueError):
        vqe_circuits(hardware_efficient_ansatz(3, 1), {"ZZ": 1.0})


def test_vqe_circuits_rejects_result_types():
    ansatz = Circuit().h(0).probability()
    with pytest.raises(ValueError):
        vqe_circuits(ansatz, {"Z": 1.0})


def test_evaluate_energy_exact_matches_statevector():
    hamiltonian = {"III": 0.3, "ZIZ": 0.5, "XYI": -0.4, "IYX": 0.2, "ZZZ": 0.1}
    ansatz = hardware_efficient_ansatz(3, 2)
    values = np.random.default_rng(0).uniform(-np.pi, np.pi, len(ansatz.parameters))
    bound = ansatz.make_bound_circuit(dict(zip(get_parameter_names(ansatz), values)))
    state = bound.to_unitary()[:, 0]
    expected = np.real(state.conj() @ hamiltonian_matrix(hamiltonian) @ state)
    energy = evaluate_energy(LocalSimulator(), ansatz, hamiltonian, values)
    assert np.isclose(energy, expected)


@pytest.mark.parametrize(
    "preparation, pauli_string",
    [
        (Circuit().i(0), "Z"),
        (Circuit().h(0), "X"),
        (Circuit().rx(0, -np.pi / 2), "Y"),
        (Circuit().h(0).cnot(0, 1), "XX"),
        (Circuit().h(0).cnot(0, 1).s(0).s(1), "YY"),
    ],
)
def test_evaluate_energy_sampled_eigenstates(preparation, pauli_string):
    # Each preparation is a +1 eigenstate of the Pauli string, so every shot gives +1.
    ansatz = preparation.add_circuit(Circuit().rz(0, FreeParameter("theta_0")))
    hamiltonian = {pauli_string: 1.0, "I" * len(pauli_string): 0.25}
    energy = evaluate_energy(LocalSimulator(), ansatz, hamiltonian, [0.0], shots=100)
    assert np.isclose(energy, 1.25)


def test_evaluate_energy_sampled_close_to_exact():
    _, hamiltonian = _h2_hamiltonian()
    ansatz = pauli_exponential_ansatz(uccsd_generators(4, 2), initial_state=[1, 1, 0, 0])
    values = [0.0, 0.0, -0.113]
    exact = evaluate_energy(LocalSimulator(), ansatz, hamiltonian, values)
    sampled = evaluate_energy(LocalSimulator(), ansatz, hamiltonian, values, shots=20000)
    assert np.isclose(sampled, exact, atol=0.02)


@pytest.mark.parametrize("shots", [0, 100])
def test_evaluate_energy_constant_hamiltonian(shots):
    energy = evaluate_energy(
        LocalSimulator(), hardware_efficient_ansatz(2, 1), {"II": -0.5}, [0.1] * 4, shots
    )
    assert energy == -0.5


def test_evaluate_energy_wrong_number_of_parameters():
    with pytest.raises(ValueError):
        evaluate_energy(LocalSimulator(), hardware_efficient_ansatz(1, 0), {"Z": 1.0}, [0, 1])


def test_run_vqe_single_qubit():
    ansatz = hardware_efficient_ansatz(1, 0)
    result = run_vqe(LocalSimulator(), {"Z": 1.0, "X": 1.0}, ansatz, [0.1])
    assert np.isclose(result["optimal_energy"], -np.sqrt(2), atol=1e-4)
    assert len(result["energy_history"]) == result["optimizer_result"].nfev
    assert result["optimal_parameters"].shape == (1,)


def test_run_vqe_h2_uccsd_reaches_fci():
    molecule, hamiltonian = _h2_hamiltonian()
    generators = uccsd_generators(molecule.n_qubits, molecule.n_electrons)
    ansatz = pauli_exponential_ansatz(generators, initial_state=[1, 1, 0, 0])
    result = run_vqe(
        LocalSimulator(), hamiltonian, ansatz, np.zeros(len(generators)), optimizer="BFGS"
    )
    assert np.isclose(result["energy_history"][0], molecule.hf_energy)
    assert np.isclose(result["optimal_energy"], molecule.fci_energy, atol=1e-6)


def test_get_vqe_results(capsys):
    vqe_result = {"optimal_energy": -1.1370, "energy_history": [-1.0, -1.1, -1.1370]}
    summary = get_vqe_results(vqe_result, exact_energy=-1.1373)
    assert summary["n_evaluations"] == 3
    assert np.isclose(summary["absolute_error"], 3e-4)
    assert summary["chemical_accuracy"]
    assert "Chemical accuracy" in capsys.readouterr().out

    summary = get_vqe_results(vqe_result, verbose=False)
    assert set(summary) == {"vqe_energy", "n_evaluations"}
    assert capsys.readouterr().out == ""


def test_get_vqe_results_outside_chemical_accuracy():
    vqe_result = {"optimal_energy": -1.10, "energy_history": [-1.10]}
    assert not get_vqe_results(vqe_result, exact_energy=-1.1373, verbose=False)["chemical_accuracy"]


def test_get_parameter_names_natural_order():
    circ = Circuit()
    for name in ["theta_10", "theta_2", "theta_1"]:
        circ.rx(0, FreeParameter(name))
    assert get_parameter_names(circ) == ["theta_1", "theta_2", "theta_10"]
