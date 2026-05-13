import math
from math import sin

from braket.circuits import Circuit, ResultType
from braket.devices import LocalSimulator


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


def rabi_simulated_dynamics(
    theta: float,
    *,
    gamma_t1: float = 0.0,
    gamma_t2: float = 0.0,
    delta: float = 0.0,
    dtheta: float | None = None,
) -> Circuit:
    """Generate a single-qubit circuit for noisy or detuned Rabi dynamics.

    Args:
        theta (float): Total resonant drive rotation angle.
        gamma_t1 (float): Amplitude damping strength per unit rotation angle.
        gamma_t2 (float): Phase damping strength per unit rotation angle.
        delta (float): Detuning strength relative to the resonant drive.
        dtheta (float | None): Step size for stepwise evolution. If None,
            noise and detuning are applied once after the full rotation.

    Returns:
        Circuit: Circuit implementing the requested Rabi dynamics.
    """
    circ = Circuit()

    if dtheta is None:
        circ.rx(0, theta)

        if delta != 0.0:
            circ.rz(0, delta * theta)
        if gamma_t1 != 0.0:
            circ.amplitude_damping(0, gamma_t1)
        if gamma_t2 != 0.0:
            circ.phase_damping(0, gamma_t2)

    else:
        n_steps = max(1, math.ceil(theta / dtheta))
        dtheta_eff = theta / n_steps
        dphi = delta * dtheta_eff
        gamma_t1_step = gamma_t1 * dtheta_eff
        gamma_t2_step = gamma_t2 * dtheta_eff

        for _ in range(n_steps):
            circ.rx(0, dtheta_eff)

            if delta != 0.0:
                circ.rz(0, dphi)
            if gamma_t1 != 0.0:
                circ.amplitude_damping(0, gamma_t1_step)
            if gamma_t2 != 0.0:
                circ.phase_damping(0, gamma_t2_step)

    circ.add_result_type(ResultType.Probability(target=[0]))
    return circ


def excited_state_probability(circ: Circuit, device: LocalSimulator) -> float:
    """Run a probability-result circuit and return the excited-state probability.

    Args:
        circ (Circuit): Circuit with a probability result type.
        device (LocalSimulator): Simulator used to run the circuit.

    Returns:
        float: Probability of measuring |1>.
    """
    task = device.run(circ, shots=0)
    probs = task.result().result_types[0].value
    return float(probs[1])
