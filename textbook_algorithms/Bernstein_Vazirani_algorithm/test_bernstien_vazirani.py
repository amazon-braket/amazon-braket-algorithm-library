import pytest
from bernstein_vazirani import bernstein_vazirani_circuit, marginalize_measurements
from braket.devices import LocalSimulator


@pytest.mark.parametrize("hidden_string, shots", [("100", 500), ("10011", 100), ("11", 50)])
def test_bv_circuit(hidden_string: str, shots: int):
    local_simulator = LocalSimulator()
    bv_circuit = bernstein_vazirani_circuit(hidden_string)
    task = local_simulator.run(bv_circuit, shots=shots)
    result = task.result()
    counts = marginalize_measurements(result.measurement_counts)
    print(counts)
    assert counts[hidden_string] == shots
