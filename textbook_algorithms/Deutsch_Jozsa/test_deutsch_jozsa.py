import pytest
from braket.devices import LocalSimulator
from deutsch_jozsa import deutsch_josza_gate, dj_algorithm, marginalize_measurements


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_deutsch_jozsa(num_qubits):
    shots = 100
    oracle_gate = deutsch_josza_gate("balanced", num_qubits)
    print(oracle_gate)
    dj_circuit = dj_algorithm(oracle_gate, num_qubits)
    print(dj_circuit)

    local_simulator = LocalSimulator()
    task = local_simulator.run(dj_circuit, shots=shots)
    result = task.result()
    counts = marginalize_measurements(result.measurement_counts)
    print(counts)

    assert counts["1" * num_qubits] == shots
