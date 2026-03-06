from math import sin

from braket.circuits import Circuit


def rabi_probability(theta: float) -> float:
    """Return excited-state probability for a single-qubit Rx rotation.

    Args:
        theta (float): Rotation angle.

    Returns:
        float: Probability of measuring |1>.
    """
    return sin(theta / 2) ** 2


def rabi_circuit(theta: float) -> Circuit:
    """Generate a single-qubit Rabi oscillation circuit.

    Args:
        theta (float): Rotation angle.

    Returns:
        Circuit: Circuit implementing Rx(theta) on qubit 0.
    """
    return Circuit().rx(0, theta)
