import numpy as np

# from braket.circuits import Circuit, FreeParameter, Observable, circuit
# from braket.devices import Device


def get_covariance_matrix(features: np.ndarray, bias: bool = False) -> np.ndarray:
    """Calculates covariance matrix

    Args:
        features (np.ndarray): A set of features
        bias (bool): Normalization of features

    Returns:
        np.ndarray: Calculated covariance matrix
    """

    return np.cov(features, bias=bias)


def get_density_matrix(cov: np.ndarray) -> np.ndarray:
    """Calculates density matrix

    Args:
        cov (np.ndarray): Calculated covariance matrix

    Returns:
        np.ndarray: Calculated density matrix
    """

    return cov / np.trace(cov)


def purify_density_matrix(den: np.ndarray) -> np.ndarray:
    """Purify density matrix into a pure state

    Args:
        den (np.ndarray): Density matrix

    Returns:
        np.ndarray: Density matrix in pure state
    """

    eig_val, eig_vec = np.linalg.eig(den)
    pure_den = np.sqrt(eig_val) * eig_vec

    return pure_den


# def qpca(pure_den: np.ndarray) -> Circuit:
#     """Return a 5-qubit circuit implementing QPCA

#     Args:
#         pure_den (np.ndarray): Pure density matrix

#     Returns:
#         Circuit: Circuit implementation of 5-qubit QPCA
#     """

#     circ = Circuit()
