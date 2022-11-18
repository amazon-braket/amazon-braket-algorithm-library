import numpy as np
import pennylane as qml
from pyscf import fci, gto

from braket.experimental.algorithms.afqmc.classical_afqmc import chemistry_preparation
from braket.experimental.algorithms.afqmc.qc_qmc import quantum_afqmc

np.set_printoptions(precision=4, edgeitems=10, linewidth=150, suppress=True)


def test_quantum_afqmc():

    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 0.75", basis="sto-3g")
    hf = mol.RHF()
    hf.kernel()
    myci = fci.FCI(hf)
    myci.kernel()
    trial = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    prop = chemistry_preparation(mol, hf, trial)
    dev = qml.device("lightning.qubit", wires=4)

    num_steps = 4
    qe_step_size = 2

    # Start QC-QFQMC computation
    quantum_energies, energies = quantum_afqmc(
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
