import numpy as np
import pytest

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.mps_encoding import mps_circuit_from_statevector


def test_compile_single_qubit_state() -> None:
    # General random state
    state = np.random.randn(2) + 1j * np.random.randn(2)
    state /= np.linalg.norm(state)

    circuit = mps_circuit_from_statevector(state, max_num_layers=1)
    result = LocalSimulator("braket_sv").run(circuit.state_vector(), shots=0).result()
    np.testing.assert_allclose(result.result_types[0].value, state, atol=1e-6)


@pytest.mark.parametrize("num_qubits", [5, 8, 10, 11])
def test_compile_area_law_states(num_qubits: int) -> None:
    # Given random_superposition can generate volume-law entangled states,
    # we manually construct an exclusively area-law entangled state here
    state = np.random.rand(2**num_qubits) + 1j * np.random.rand(2**num_qubits)
    state /= np.linalg.norm(state)

    circuit = mps_circuit_from_statevector(state, max_num_layers=6)
    result = LocalSimulator("braket_sv").run(circuit.state_vector(), shots=0).result()
    fidelity = np.vdot(result.result_types[0].value, state)
    assert np.abs(fidelity) > 0.85


def test_compile_trivial_state_with_mps_pass() -> None:
    # Define a circuit that produces a trivial state
    # aka a product state
    trivial_circuit = Circuit()

    trivial_circuit.ry(angle=np.pi / 2, target=9)
    trivial_circuit.rx(angle=np.pi, target=9)
    trivial_circuit.rz(angle=np.pi / 4, target=9)
    for i in reversed(range(9)):
        trivial_circuit.cnot(i + 1, i)
        trivial_circuit.rz(angle=-np.pi / (2 ** (9 - i)), target=i)
        trivial_circuit.cnot(i + 1, i)
        trivial_circuit.rz(angle=np.pi / (2 ** (9 - i)), target=i)
        trivial_circuit.ry(angle=np.pi / 2, target=i)
        trivial_circuit.rx(angle=np.pi, target=i)
        trivial_circuit.rz(angle=np.pi / 4, target=i)
    for i in reversed(range(9)):
        trivial_circuit.rz(angle=np.pi / (2 ** (9 - i)), target=i)
    for i in reversed(range(9)):
        trivial_circuit.cnot(i + 1, i)
    for i in reversed(range(9)):
        trivial_circuit.cnot(i, i + 1)

    simulator = LocalSimulator("braket_sv")
    state = simulator.run(trivial_circuit.state_vector(), shots=0).result().result_types[0].value
    circuit = mps_circuit_from_statevector(state, max_num_layers=1)
    result = (
        LocalSimulator("braket_sv")
        .run(circuit.state_vector(), shots=0)
        .result()
        .result_types[0]
        .value
    )
    np.testing.assert_allclose(result, state, atol=1e-6)
