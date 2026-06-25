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

from braket.experimental.algorithms.shors.shors import (
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
    integer_N = 15
    integer_a = 2
    shor = shors_algorithm(integer_N, integer_a)
    local_simulator = LocalSimulator()
    output = run_shors_algorithm(shor, local_simulator)
    aggregate_results = get_factors_from_results(output, integer_N, integer_a, False)
    assert aggregate_results["guessed_factors"] == {3, 5}


def test_shors_algorithm_for_33():
    integer_N = 33
    integer_a = 10
    shor = shors_algorithm(integer_N, integer_a)
    local_simulator = LocalSimulator()
    output = run_shors_algorithm(shor, local_simulator)
    aggregate_results = get_factors_from_results(output, integer_N, integer_a, False)
    assert aggregate_results["guessed_factors"] == {3, 11}


def test_all_valid_a():
    local_simulator = LocalSimulator()
    integer_N = 15
    for integer_a in [2, 7, 8, 11, 13]:
        shor = shors_algorithm(integer_N, integer_a)
        output = run_shors_algorithm(shor, local_simulator)
        aggregate_results = get_factors_from_results(output, integer_N, integer_a, True)
        assert aggregate_results["guessed_factors"] == {3, 5}


def test_all_valid_a_for_33():
    local_simulator = LocalSimulator()
    integer_N = 33
    for integer_a in [10, 23]:
        shor = shors_algorithm(integer_N, integer_a)
        output = run_shors_algorithm(shor, local_simulator)
        aggregate_results = get_factors_from_results(output, integer_N, integer_a, True)
        assert aggregate_results["guessed_factors"] == {3, 11}


def test_no_counts():
    with pytest.raises(TypeError):
        output = {"measurement_counts": False}
        integer_N = 15
        integer_a = 7
        get_factors_from_results(output, integer_N, integer_a, True)
