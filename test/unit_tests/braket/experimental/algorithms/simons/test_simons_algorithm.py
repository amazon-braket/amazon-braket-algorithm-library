# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import pytest
from braket.devices import LocalSimulator

from braket.experimental.algorithms.simons import (
    get_simons_algorithm_results,
    run_simons_algorithm,
    simons_oracle,
)

local_simulator = LocalSimulator()


@pytest.mark.parametrize("secret", ["00000", "10110"])
def test_simons_algorithm(secret):
    oracle = simons_oracle(secret)
    task = run_simons_algorithm(oracle=oracle, device=local_simulator)
    processed_results = get_simons_algorithm_results(task)
    revelead_secret = processed_results["secret_string"]
    assert secret == revelead_secret


@pytest.mark.xfail(raises=RuntimeError)
def test_low_shot_number():
    secret_5_qubit = "10110"
    oracle = simons_oracle(secret_5_qubit)
    task = run_simons_algorithm(oracle=oracle, device=local_simulator, shots=4)
    get_simons_algorithm_results(task)


@pytest.mark.xfail(raises=ValueError)
def test_bad_string():
    bad_string = "a0110"
    simons_oracle(bad_string)
