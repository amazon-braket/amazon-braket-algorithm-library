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

# from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.shors_algorithm.shors_algorithm import (  # noqa: F401,E501
    get_factors_from_results,
    run_shors_algorithm,
    shors_algorithm,
)

local_simulator = LocalSimulator()


def test_invalid_a_N():
    with pytest.raises(ValueError):
        shors_algorithm(1, 1)
    with pytest.raises(ValueError):
        shors_algorithm(10, 3)
    with pytest.raises(ValueError):
        shors_algorithm(17, 30)
    with pytest.raises(ValueError):
        shors_algorithm(45, 30)
    with pytest.raises(ValueError):
        shors_algorithm(15, 10)
    with pytest.raises(ValueError):
        shors_algorithm(15, 1)


def test_shors_algorithm():
    N = 15
    a = 2
    shor = shors_algorithm(N, a)
    local_simulator = LocalSimulator()
    output = run_shors_algorithm(shor, local_simulator)
    get_factors_from_results(output, N, a)


def test_all_valid_a():
    local_simulator = LocalSimulator()
    N = 15
    for a in [2, 7, 8, 11, 13]:
        shor = shors_algorithm(N, a)
        output = run_shors_algorithm(shor, local_simulator)
        get_factors_from_results(output, N, a)


def test_no_counts():
    with pytest.raises(TypeError):
        output = {"measurement_counts": False}
        N = 15
        a = 7
        get_factors_from_results(output, N, a)
