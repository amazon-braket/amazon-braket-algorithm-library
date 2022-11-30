import pennylane as qml
from pennylane import numpy as np

from braket.experimental.algorithms.qc_qmc.classical_afqmc import chemistry_preparation
from braket.experimental.algorithms.qc_qmc.qc_qmc import qc_qmc

np.set_printoptions(precision=4, edgeitems=10, linewidth=150, suppress=True)


def test_properties():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.41729459]], requires_grad=False)
    mol = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g")
    trial = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    prop = chemistry_preparation(mol, geometry, trial)
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


def test_qc_qmc():
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.41729459]], requires_grad=False)
    mol = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g")
    trial = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    prop = chemistry_preparation(mol, geometry, trial)
    dev = qml.device("lightning.qubit", wires=4)

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
