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

from .simons_algorithm import run_simons_algorithm, simons_oracle

local_simulator = LocalSimulator()


@pytest.mark.parametrize("secret", "10110")
def test_simons_algorithm(secret):
    oracle = simons_oracle(secret)
    result = run_simons_algorithm(oracle=oracle, device=local_simulator)
    revelead_secret = result["secret_string"]
    assert secret == revelead_secret


def test_simons_algorithm():
    secret = "10110"
    oracle = simons_oracle(secret)
    with pytest.raises(RuntimeError):
        run_simons_algorithm(oracle=oracle, device=local_simulator, shots=4)
