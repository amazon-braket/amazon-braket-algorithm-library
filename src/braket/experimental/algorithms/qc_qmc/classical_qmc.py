import copy
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import pennylane as qml
from openfermion.circuits.low_rank import low_rank_two_body_decomposition
from pennylane import numpy as np
from scipy.linalg import det, expm, qr


@dataclass
class ChemicalProperties:
    h1e: np.ndarray  # one-body term
    eri: np.ndarray  # two-body term
    nuclear_repulsion: float  # nuclear repulsion energy
    v_0: np.ndarray  # one-body term stored as np.ndarray with mean-field subtraction
    h_chem: np.ndarray  # one-body term stored as np.ndarray, without mean-field subtraction
    v_gamma: List[np.ndarray]  # 1j * l_gamma
    l_gamma: List[np.ndarray]  # Cholesky vector decomposed from two-body terms
    mf_shift: np.ndarray  # mean-field shift
    lambda_l: List[np.ndarray]  # eigenvalues of Cholesky vectors
    u_l: List[np.ndarray]  # eigenvectors of Cholesky vectors


def classical_qmc(
    num_walkers: int,
    num_steps: int,
    dtau: float,
    trial: np.ndarray,
    prop: ChemicalProperties,
    max_pool: int = 8,
) -> Tuple[float, float]:
    """Classical Auxiliary-Field Quantum Monte Carlo.

    Args:
        num_walkers (int): Number of walkers.
        num_steps (int): Number of (imaginary) time steps
        dtau (float): Increment of each time step
        trial (ndarray): Trial wavefunction.
        prop (ChemicalProperties): Chemical properties.
        max_pool (int): Max workers. Defaults to 8.

    Returns:
        Tuple[float, float]: Energies
    """
    e_hf = hartree_fock_energy(trial, prop)

    walkers = [trial] * num_walkers
    weights = [1.0] * num_walkers

    inputs = [
        (num_steps, dtau, trial, prop, e_hf, walker, weight)
        for walker, weight in zip(walkers, weights)
    ]

    # parallelize with multiprocessing
    with mp.Pool(max_pool) as pool:
        results = list(pool.map(full_imag_time_evolution_wrapper, inputs))

    local_energies, weights = map(np.array, zip(*results))
    energies = np.real(np.average(local_energies, weights=weights, axis=0))
    return local_energies, energies


def hartree_fock_energy(trial: np.ndarray, prop: ChemicalProperties) -> float:
    """Compute Hatree Fock energy.

    Args:
        trial (ndarray): Trial wavefunction.
        prop (ChemicalProperties): Chemical properties.

    Returns:
        float: Energy
    """
    trial_up = trial[::2, ::2]
    trial_down = trial[1::2, 1::2]
    # compute  one particle Green's function
    green_funcs = [greens_pq(trial_up, trial_up), greens_pq(trial_down, trial_down)]
    e_hf = local_energy(prop.h1e, prop.eri, green_funcs, prop.nuclear_repulsion)
    return e_hf


def full_imag_time_evolution_wrapper(args: Tuple) -> Callable:
    return full_imag_time_evolution(*args)


def full_imag_time_evolution(
    num_steps: int,
    dtau: float,
    trial: np.ndarray,
    prop: ChemicalProperties,
    e_shift: float,
    walker: np.ndarray,
    weight: float,
) -> Tuple[List[float], float]:
    """Imaginary time evolution of a single walker.

    Args:
        num_steps (int): number of time steps
        dtau (float): imaginary time step size
        trial (ndarray): trial state as np.ndarray, e.g., for h2 HartreeFock state, it is
            np.array([[1,0], [0,1], [0,0], [0,0]])
        prop (ChemicalProperties): Chemical properties.
        e_shift (float): Reference energy, i.e. Hartree-Fock energy
        walker (ndarray): normalized walker state as np.ndarray, others are the same as trial
        weight (float): weight for sampling.

    Returns:
        Tuple[List[float], float]: energy_list, weights
    """
    # random seed for multiprocessing
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

    energy_list, weights = [], []
    for _ in range(num_steps):
        e_loc, walker, weight = imag_time_propogator(dtau, trial, walker, weight, prop, e_shift)
        energy_list.append(e_loc)
        weights.append(weight)
    return energy_list, weights


def imag_time_propogator(
    dtau: float,
    trial: np.ndarray,
    walker: np.ndarray,
    weight: float,
    prop: ChemicalProperties,
    e_shift: float,
) -> Tuple[float, np.ndarray, float]:
    """Propagate a walker by one time step.

    Args:
        dtau (float): imaginary time step size
        trial (ndarray): trial state as np.ndarray, e.g., for h2 HartreeFock state, it is
            np.array([[1,0], [0,1], [0,0], [0,0]])
        walker (ndarray): normalized walker state as np.ndarray, others are the same as trial
        weight (float): weight for sampling.
        prop (ChemicalProperties): Chemical properties.
        e_shift (float): Reference energy, i.e. Hartree-Fock energy

    Returns:
        Tuple[float, ndarray, float]: e_loc, new_walker, new_weight
    """
    # First compute the bias force using the expectation value of L operators
    num_fields = len(prop.v_gamma)

    # compute the overlap integral
    ovlp = np.linalg.det(trial.transpose().conj() @ walker)

    trial_up = trial[::2, ::2]
    trial_down = trial[1::2, 1::2]
    walker_up = walker[::2, ::2]
    walker_down = walker[1::2, 1::2]
    green_funcs = [greens_pq(trial_up, walker_up), greens_pq(trial_down, walker_down)]
    e_loc = local_energy(prop.h1e, prop.eri, green_funcs, prop.nuclear_repulsion)

    # sampling the auxiliary fields
    x = np.random.normal(0.0, 1.0, size=num_fields)

    # update the walker
    new_walker = propagate_walker(
        x, prop.v_0, prop.v_gamma, prop.mf_shift, dtau, trial, walker, green_funcs
    )

    # Define the Id operator and find new weight
    new_ovlp = np.linalg.det(trial.transpose().conj() @ new_walker)
    arg = np.angle(new_ovlp / ovlp)

    new_weight = weight * np.exp(-dtau * (np.real(e_loc) - e_shift)) * np.max([0.0, np.cos(arg)])

    return e_loc, new_walker, new_weight


def local_energy(h1e: np.ndarray, eri: np.ndarray, green_funcs: np.ndarray, enuc: float) -> float:
    r"""Calculate local energy for generic two-body Hamiltonian using the full (spatial)
    form for the two-electron integrals.

    Args:
        h1e (ndarray): one-body term.
        eri (ndarray): two-body term.
        green_funcs (ndarray): Walker's "green's function".
        enuc (float): Nuclear repulsion energy.

    Returns:
        float: kinetic, potential energies and nuclear repulsion energy.
    """
    e1 = np.einsum("ij,ij->", h1e, green_funcs[0]) + np.einsum("ij,ij->", h1e, green_funcs[1])

    euu = 0.5 * (
        np.einsum("ijkl,il,jk->", eri, green_funcs[0], green_funcs[0])
        - np.einsum("ijkl,ik,jl->", eri, green_funcs[0], green_funcs[0])
    )
    edd = 0.5 * (
        np.einsum("ijkl,il,jk->", eri, green_funcs[1], green_funcs[1])
        - np.einsum("ijkl,ik,jl->", eri, green_funcs[1], green_funcs[1])
    )
    eud = 0.5 * np.einsum("ijkl,il,jk->", eri, green_funcs[0], green_funcs[1])
    edu = 0.5 * np.einsum("ijkl,il,jk->", eri, green_funcs[1], green_funcs[0])
    e2 = euu + edd + eud + edu

    return e1 + e2 + enuc


def reortho(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Reorthogonalise a MxN matrix A. Performs a QR decomposition of A. Note that for consistency
    elsewhere we want to preserve detR > 0 which is not guaranteed. We thus factor the signs of the
    diagonal of R into Q.

    Args:
        A (ndarray): MxN matrix.

    Returns:
        Tuple[ndarray, float]: (Q, detR)
        Q (ndarray): Orthogonal matrix. A = QR.
        detR (float): Determinant of upper triangular matrix (R) from QR decomposition.
    """
    (Q, R) = qr(A, mode="economic")
    signs = np.diag(np.sign(np.diag(R)))
    Q = Q.dot(signs)
    detR = det(signs.dot(R))
    return (Q, detR)


def greens_pq(psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """This function computes the one-body Green's function.

    Args:
        psi (ndarray): wavefunction
        phi (ndarray): wavefunction

    Returns:
        ndarray: one-body Green's function
    """
    overlap_inverse = np.linalg.inv(psi.transpose() @ phi)
    green_funcs = phi @ overlap_inverse @ psi.transpose()
    return green_funcs


def chemistry_preparation(
    mol: qml.qchem.molecule.Molecule, geometry: np.ndarray, trial: np.ndarray
) -> ChemicalProperties:
    """Return the one- and two-electron integrals from Pennylane.

    Args:
        mol (Molecule): Pennylane molecular structure.
        geometry (ndarray): Atomic coordiantes for the molecule.
        trial (ndarray): Trial wavefunction.

    Returns:
        ChemicalProperties: chemical properties
        v_0: one-body term stored as np.ndarray, with mean-field subtraction
        h_chem: one-body term stored as np.ndarray, without mean-field subtraction
        v_gamma: 1.j*l_gamma
        l_gamma: Cholesky vector decomposed from two-body terms
        mf_shift: mean-field shift
        nuclear_repulsion: nuclear repulsion constant
    """

    # h1e = qml.qchem.core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(geometry)
    h2e = qml.qchem.repulsion_tensor(mol.basis_set)()
    nuclear_repulsion = qml.qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)()[0]
    # Get the one and two electron integral in the Hatree Fock basis
    h1e = qml.qchem.electron_integrals(mol)()[1]
    # For the modified physics notation adapted to quantum computing convention.
    for _ in range(4):
        h2e = np.tensordot(h2e, mol.mo_coefficients, axes=1).transpose(3, 0, 1, 2)
    eri = h2e.transpose(0, 2, 3, 1)

    lamb, g, one_body_correction, residue = low_rank_two_body_decomposition(eri, spin_basis=False)
    v_0 = np.kron(h1e, np.eye(2)) + 0.5 * one_body_correction
    h_chem = copy.deepcopy(v_0)
    num_spin_orbitals = int(h_chem.shape[0])

    l_gamma = [np.sqrt(i) * j for i, j in zip(lamb, g)]
    v_gamma = [1.0j * np.sqrt(i) * j for i, j in zip(lamb, g)]

    trial_up = trial[::2, ::2]
    trial_down = trial[1::2, 1::2]
    green_funcs = [greens_pq(trial_up, trial_up), greens_pq(trial_down, trial_down)]

    # compute mean-field shift as an imaginary value
    mf_shift = np.array([])
    for i in v_gamma:
        value = np.einsum("ij,ij->", i[::2, ::2], green_funcs[0])
        value += np.einsum("ij,ij->", i[1::2, 1::2], green_funcs[1])
        mf_shift = np.append(mf_shift, value)

    # Note that we neglect the prime symbol for simplicity.
    for s, v in zip(mf_shift, v_gamma):
        v_0 -= s * v

    lambda_l = []
    u_l = []
    for i in l_gamma:
        if np.count_nonzero(np.round(i - np.diag(np.diagonal(i)), 7)) != 0:
            eigval, eigvec = np.linalg.eigh(i)
            lambda_l.append(eigval)
            u_l.append(eigvec)
        else:
            lambda_l.append(np.diagonal(i))
            u_l.append(np.eye(num_spin_orbitals))

    return ChemicalProperties(
        h1e, eri, nuclear_repulsion, v_0, h_chem, v_gamma, l_gamma, mf_shift, lambda_l, u_l
    )


def propagate_walker(
    x: np.ndarray,
    v_0: List[np.ndarray],
    v_gamma: List[np.ndarray],
    mf_shift: np.ndarray,
    dtau: float,
    trial: np.ndarray,
    walker: np.ndarray,
    green_funcs: List[np.ndarray],
) -> np.ndarray:
    r"""Update the walker forward in imaginary time.

    Args:
        x (ndarray): auxiliary fields
        v_0 (List[ndarray]): modified one-body term from reordering the two-body
            operator + mean-field subtraction.
        v_gamma (List[ndarray]): Cholesky vectors stored in list (L, num_spin_orbitals,
            num_spin_orbitals), without mf_shift.
        mf_shift (ndarray): mean-field shift \Bar{v}_{\gamma} stored in np.array format
        dtau (float): imaginary time step size
        trial (ndarray): trial state as np.ndarray, e.g., for h2 HartreeFock state,
            it is np.array([[1,0], [0,1], [0,0], [0,0]])
        walker (ndarray): walker state as np.ndarray, others are the same as trial
        green_funcs (List[ndarray]): one-body Green's function

    Returns:
        ndarray: new walker for next time step
    """
    num_spin_orbitals, num_electrons = trial.shape
    num_fields = len(v_gamma)

    v_expectation = np.array([])
    for i in v_gamma:
        value = np.einsum("ij,ij->", i[::2, ::2], green_funcs[0])
        value += np.einsum("ij,ij->", i[1::2, 1::2], green_funcs[1])
        v_expectation = np.append(v_expectation, value)

    xbar = -np.sqrt(dtau) * (v_expectation - mf_shift)
    # Sampling the auxiliary fields
    xshifted = x - xbar

    # Define the B operator B(x - \bar{x})
    exp_v0 = expm(-dtau / 2 * v_0)

    V = np.zeros((num_spin_orbitals, num_spin_orbitals), dtype=np.complex128)
    for i in range(num_fields):
        V += np.sqrt(dtau) * xshifted[i] * v_gamma[i]
    exp_V = expm(V)

    # Note that v_gamma doesn't include the mf_shift, there is an additional term coming from
    # -(x - xbar)*mf_shift, this term is also a complex value.

    B = exp_v0 @ exp_V @ exp_v0

    # Find the new walker state
    new_walker, _ = reortho(B @ walker)

    return new_walker
