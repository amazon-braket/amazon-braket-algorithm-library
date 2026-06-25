import numpy as np
import pytest

from braket.devices import LocalSimulator
from braket.experimental.algorithms.sweeping import (
    generate_staircase_ansatz,
    sweep_state_approximation,
)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def compile_with_state_sweep_pass(num_qubits: int) -> None:
    state = np.random.uniform(-1, 1, 2**num_qubits) + 1j * np.random.uniform(-1, 1, 2**num_qubits)
    state /= np.linalg.norm(state)

    compiled_circuit = sweep_state_approximation(
        target_state=state,
        unitary_layers=generate_staircase_ansatz(
            num_qubits=num_qubits, num_layers=int((num_qubits**2) / 2)
        ),
        num_sweeps=100 * num_qubits,
        log=False,
    )

    result = LocalSimulator("braket_sv").run(compiled_circuit.state_vector(), shots=0).result()
    fidelity = np.abs(np.vdot(state, result.values[0]))

    assert fidelity > 0.99
