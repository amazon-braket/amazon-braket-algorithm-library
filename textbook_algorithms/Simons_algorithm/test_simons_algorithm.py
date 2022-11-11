import pytest
from braket.devices import LocalSimulator
from .simons_algorithm import simons_oracle, submit_simons_task, process_simons_results

local_simulator = LocalSimulator()


@pytest.mark.parametrize("secret", "10110")
def test_simons_algorithm(secret):
    oracle = simons_oracle(secret)
    task = submit_simons_task(oracle=oracle, device=local_simulator)
    revelead_secret = process_simons_results(task)
    assert secret == revelead_secret
