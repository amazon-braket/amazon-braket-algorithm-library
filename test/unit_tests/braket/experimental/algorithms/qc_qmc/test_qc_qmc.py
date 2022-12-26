from unittest.mock import patch

import pennylane as qml
import pytest
from pennylane import numpy as np

from braket.experimental.algorithms.qc_qmc.classical_qmc import (
    chemistry_preparation,
    classical_qmc,
    full_imag_time_evolution_wrapper,
    hartree_fock_energy,
)
from braket.experimental.algorithms.qc_qmc.qc_qmc import q_full_imag_time_evolution_wrapper, qc_qmc

np.set_printoptions(precision=4, edgeitems=10, linewidth=150, suppress=True)


@pytest.fixture
def qmc_data():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.41729459]], requires_grad=False)
    mol = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g")
    trial = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    prop = chemistry_preparation(mol, geometry, trial)
    dev = qml.device("lightning.qubit", wires=4)
    Ehf = hartree_fock_energy(trial, prop)
    return (trial, prop, dev, Ehf)


def V_T() -> None:
    """Define V_T through UCCSD circuit."""
    qml.RX(np.pi / 2.0, wires=0)
    for i in range(1, 4):
        qml.Hadamard(wires=i)

    for i in range(3):
        qml.CNOT(wires=[i, i + 1])

    qml.RZ(0.12, wires=3)
    for i in range(3)[::-1]:
        qml.CNOT(wires=[i, i + 1])

    qml.RX(-np.pi / 2.0, wires=0)
    for i in range(1, 4):
        qml.Hadamard(wires=i)


def test_properties(qmc_data):
    trial, prop, dev, Ehf = qmc_data
    assert np.allclose(prop.h1e, np.array([[-1.2473, -0.0], [-0.0, -0.4813]]), atol=1e-4)
    assert np.allclose(
        prop.eri,
        np.array(
            [
                [[[0.6728, 0.0], [0.0, 0.1818]], [[0.0, 0.1818], [0.662, 0.0]]],
                [[[0.0, 0.662], [0.1818, 0.0]], [[0.1818, 0.0], [0.0, 0.6958]]],
            ],
        ),
        atol=1e-4,
    )
    assert np.allclose(
        prop.v_0,
        np.array(
            [
                [-0.3289 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, -0.3289 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.4039 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.4039 + 0.0j],
            ],
        ),
        atol=1e-4,
    )


def test_qc_qmc(qmc_data):
    trial, prop, dev, Ehf = qmc_data
    num_steps = 5
    num_walkers = 15
    qe_step_size = 2

    with patch("multiprocessing.pool.Pool.map") as evolution_mock:
        energy_data_mock = [1.0 for _ in range(num_steps)]
        quantum_data_mock = [0.0 if i % qe_step_size == 0 else 1.0 for i in range(num_steps)]
        evolution_mock.return_value = [
            (energy_data_mock, energy_data_mock, quantum_data_mock, quantum_data_mock),
        ] * num_walkers

        # Start QC-QMC computation
        quantum_energies, energies = qc_qmc(
            num_walkers=num_walkers,
            num_steps=num_steps,
            dtau=1,
            quantum_evaluations_every_n_steps=qe_step_size,
            trial=trial,
            prop=prop,
            V_T=V_T,
            dev=dev,
            max_pool=2,
        )
    assert len(energies) == num_steps
    assert len(quantum_energies) == num_steps // qe_step_size


def test_q_full_imag_time_evolution(qmc_data):
    trial, prop, dev, Ehf = qmc_data
    num_steps = 4
    qe_step_size = 2
    num_walkers = 2
    dtau = 1

    walkers = [trial] * num_walkers
    weights = [1.0] * num_walkers
    inputs = [
        (num_steps, qe_step_size, dtau, trial, prop, Ehf, walker, weight, V_T, dev)
        for walker, weight in zip(walkers, weights)
    ]

    results = [q_full_imag_time_evolution_wrapper(input_arg) for input_arg in inputs]
    assert len(results) == num_walkers
    assert len(results[0][0]) == num_steps


def test_classical_qmc(qmc_data):
    trial, prop, dev, Ehf = qmc_data
    num_steps = 4
    num_walkers = 15

    # Start QMC computation
    local_energies, energies = classical_qmc(
        num_walkers=num_walkers,
        num_steps=num_steps,
        dtau=1,
        trial=trial,
        prop=prop,
        max_pool=2,
    )
    assert len(energies) == num_steps
    assert len(local_energies) == num_walkers
    assert len(local_energies[0]) == num_steps


def test_full_imag_time_evolution(qmc_data):
    trial, prop, dev, Ehf = qmc_data

    num_steps = 4
    num_walkers = 2
    dtau = 1

    walkers = [trial] * num_walkers
    weights = [1.0] * num_walkers

    inputs = [
        (num_steps, dtau, trial, prop, Ehf, walker, weight)
        for walker, weight in zip(walkers, weights)
    ]

    energy_list, weights = [full_imag_time_evolution_wrapper(input_arg) for input_arg in inputs]
    assert len(energy_list) == num_walkers
    assert len(weights) == num_walkers
