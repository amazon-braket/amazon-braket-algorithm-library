import pytest

import pennylane as qml
from pennylane import numpy as np

from braket.experimental.algorithms.qc_qmc.classical_afqmc import (
    chemistry_preparation,
    hartree_fock_energy,
    classical_afqmc,
    full_imag_time_evolution_wrapper,
)
from braket.experimental.algorithms.qc_qmc.qc_qmc import qc_qmc, q_full_imag_time_evolution_wrapper

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
    num_steps = 4
    qe_step_size = 2

    # Start QC-QMC computation
    quantum_energies, energies = qc_qmc(
        num_walkers=15,
        num_steps=num_steps,
        dtau=1,
        quantum_evaluations_every_n_steps=qe_step_size,
        trial=trial,
        prop=prop,
        max_pool=2,
        dev=dev,
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
        (num_steps, qe_step_size, dtau, trial, prop, Ehf, walker, weight, dev)
        for walker, weight in zip(walkers, weights)
    ]

    results = [q_full_imag_time_evolution_wrapper(input_arg) for input_arg in inputs]
    assert len(results) == num_walkers
    assert len(results[0][0]) == num_steps


def test_classical_afqmc(qmc_data):
    trial, prop, dev, Ehf = qmc_data
    num_steps = 4
    num_walkers = 15

    # Start QMC computation
    local_energies, energies = classical_afqmc(
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
