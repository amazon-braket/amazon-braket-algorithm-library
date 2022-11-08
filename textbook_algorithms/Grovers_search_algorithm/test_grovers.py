from braket.devices import LocalSimulator
from grovers import grover, oracles


def test_grover():
    oracles_circuits = oracles()
    circuit = grover("000", oracles_circuits)

    local_simulator = LocalSimulator()
    task = local_simulator.run(circuit, shots=1000)
    probs = task.result().measurement_probabilities

    assert max(probs, key=probs.get) == "000"
