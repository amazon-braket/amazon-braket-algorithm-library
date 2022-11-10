import pytest
from braket.devices import LocalSimulator
from .simons_algorithm import submit_simons_tasks, process_simons_results

local_simulator = LocalSimulator()

@pytest.mark.parametrize("secret", "10110")
def test_simons_algorithm(secret):   
    task = submit_simons_tasks(secret_s=secret, device=local_simulator)
    revelead_secret = process_simons_results(task)
    assert secret == revelead_secret