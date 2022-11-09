import math

from bells_inequality import get_bell_results, submit_bell_tasks
from braket.devices import LocalSimulator


def test_bell():
    local_simulator = LocalSimulator()
    local_tasks = submit_bell_tasks(local_simulator, shots=0)
    results, pAB, pAC, pBC = get_bell_results(local_tasks)
    assert math.isclose(pAB, -0.5)
    assert math.isclose(pBC, -0.5)
    assert math.isclose(pAC, 0.5)
    assert len(results) == 3
