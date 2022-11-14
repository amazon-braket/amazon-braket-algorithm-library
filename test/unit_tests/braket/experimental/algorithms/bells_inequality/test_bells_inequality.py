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


import math

from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.bells_inequality import (
    bell_singlet,
    bell_singlet_rotated,
    get_bell_inequality_results,
    run_bell_inequality_tasks,
)


def test_singlet():
    circ = bell_singlet(0, 1)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1)
    assert circ == expected


def test_singlet_rotated_zero():
    circ = bell_singlet_rotated(0, 1, 0, 0)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1).probability()
    assert circ == expected


def test_singlet_rotated():
    circ = bell_singlet_rotated(0, 1, 0.5, 0.25)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1).rx(0, 0.5).rx(1, 0.25).probability()
    assert circ == expected


def test_bell_inequality_shots_0():
    local_simulator = LocalSimulator()
    local_tasks = run_bell_inequality_tasks(local_simulator, shots=0)
    assert len(local_tasks) == 3

    results, pAB, pAC, pBC = get_bell_inequality_results(local_tasks)
    assert math.isclose(pAB, -0.5)
    assert math.isclose(pBC, -0.5)
    assert math.isclose(pAC, 0.5)
    assert len(results) == 3


def test_bell_inequality():
    local_simulator = LocalSimulator()
    local_tasks = run_bell_inequality_tasks(local_simulator, shots=10)
    assert len(local_tasks) == 3
    results, pAB, pAC, pBC = get_bell_inequality_results(local_tasks)
    assert len(results) == 3
