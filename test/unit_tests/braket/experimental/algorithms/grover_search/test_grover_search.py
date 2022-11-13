from braket.devices import LocalSimulator
from braket.experimental.algorithms.grover_search.grover_search import (
    grover_search,
    get_oracles,
)


def test_grover_search():
    oracles_circuits = oracles()
    circuit = grover("000", oracles_circuits)

    local_simulator = LocalSimulator()
    task = local_simulator.run(circuit, shots=1000)
    probs = task.result().measurement_probabilities

    assert max(probs, key=probs.get) == "000"
