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

import numpy as np
import pytest
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.bernstein_vazirani.bernstein_vazirani import (
    bernstein_vazirani_circuit,
    get_bernstein_vazirani_results,
    plot_bitstrings,
    run_bernstein_vazirani,
)


def test_get_bernstein_vazirani():
    bv_circuit = bernstein_vazirani_circuit("0")
    expected = Circuit().h(1).z(1).h(0).i(0).h(0).probability(0)
    print(bv_circuit)
    print()
    print(expected)
    assert bv_circuit == expected


@pytest.mark.parametrize("hidden_string, shots", [("100", 0), ("11", 10), ("10111", 50)])
def test_get_bernstein_vazirani_results(hidden_string: str, shots: int):
    local_simulator = LocalSimulator()
    bv_circuit = bernstein_vazirani_circuit(hidden_string)
    task = run_bernstein_vazirani(bv_circuit, local_simulator, shots=shots)
    bv_result = get_bernstein_vazirani_results(task)
    assert np.isclose(bv_result[hidden_string], 1.0)


def test_plot_bitstrings():
    plot_bitstrings({"0": 1})
