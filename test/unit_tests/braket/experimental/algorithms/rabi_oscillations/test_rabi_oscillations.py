import math

from braket.devices import LocalSimulator
from braket.experimental.algorithms.rabi_oscillations import (
    excited_state_probability,
    rabi_circuit,
    rabi_probability,
    rabi_simulated_dynamics,
)


def test_rabi_probability_edge_cases():
    assert rabi_probability(0.0) == 0.0
    assert math.isclose(rabi_probability(math.pi), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rabi_probability(2 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_rabi_circuit_has_single_rx_instruction():
    circ = rabi_circuit(0.123)
    assert circ.qubit_count == 1
    assert len(circ.instructions) == 1

    instr = circ.instructions[0]
    assert instr.operator.name == "Rx"
    assert list(instr.target) == [0]


def test_rabi_simulated_dynamics_defaults_to_single_rx_with_probability_result():
    circ = rabi_simulated_dynamics(0.123)

    assert circ.qubit_count == 1
    assert len(circ.instructions) == 1
    assert circ.instructions[0].operator.name == "Rx"
    assert len(circ.result_types) == 1


def test_rabi_simulated_dynamics_adds_single_channel_noise():
    circ = rabi_simulated_dynamics(0.123, gamma_t1=0.1, gamma_t2=0.2, delta=0.3)

    operator_names = [instruction.operator.name for instruction in circ.instructions]

    assert operator_names == ["Rx", "Rz", "AmplitudeDamping", "PhaseDamping"]
    assert len(circ.result_types) == 1


def test_rabi_simulated_dynamics_adds_stepwise_noise_and_detuning():
    circ = rabi_simulated_dynamics(
        math.pi,
        gamma_t1=0.1,
        gamma_t2=0.2,
        delta=0.3,
        dtheta=0.5,
    )

    n_steps = math.ceil(math.pi / 0.5)
    operator_names = [instruction.operator.name for instruction in circ.instructions]

    assert len(circ.instructions) == 4 * n_steps
    assert operator_names[:4] == ["Rx", "Rz", "AmplitudeDamping", "PhaseDamping"]
    assert len(circ.result_types) == 1


def test_excited_state_probability_returns_probability_of_one():
    device = LocalSimulator()
    circ = rabi_simulated_dynamics(math.pi)

    assert math.isclose(
        excited_state_probability(circ, device),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

def test_rabi_simulated_dynamics_stepwise_only_detuning():
    circ = rabi_simulated_dynamics(
        math.pi,
        delta=0.3,
        dtheta=0.5,
    )

    operator_names = [instruction.operator.name for instruction in circ.instructions]

    assert "Rz" in operator_names
    assert "AmplitudeDamping" not in operator_names
    assert "PhaseDamping" not in operator_names


def test_rabi_simulated_dynamics_stepwise_only_t1():
    circ = rabi_simulated_dynamics(
        math.pi,
        gamma_t1=0.1,
        dtheta=0.5,
    )

    operator_names = [instruction.operator.name for instruction in circ.instructions]

    assert "AmplitudeDamping" in operator_names
    assert "Rz" not in operator_names
    assert "PhaseDamping" not in operator_names


def test_rabi_simulated_dynamics_stepwise_only_t2():
    circ = rabi_simulated_dynamics(
        math.pi,
        gamma_t2=0.1,
        dtheta=0.5,
    )

    operator_names = [instruction.operator.name for instruction in circ.instructions]

    assert "PhaseDamping" in operator_names
    assert "Rz" not in operator_names
    assert "AmplitudeDamping" not in operator_names
