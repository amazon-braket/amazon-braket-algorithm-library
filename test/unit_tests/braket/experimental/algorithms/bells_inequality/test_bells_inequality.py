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

from braket.experimental.algorithms.bells_inequality import bell_singlet  # noqa:F401
from braket.experimental.algorithms.bells_inequality import (
    bell_singlet_rotated_basis,
    create_bell_inequality_circuits,
    get_bell_inequality_results,
    run_bell_inequality,
)


def test_singlet():
    circ = Circuit().bell_singlet(0, 1)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1)
    assert circ == expected


def test_singlet_rotated_zero():
    circ = bell_singlet_rotated_basis(0, 1, 0, 0)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1).probability()
    assert circ == expected


def test_singlet_rotated():
    circ = bell_singlet_rotated_basis(0, 1, 0.5, 0.25)
    expected = Circuit().x(0).x(1).h(0).cnot(0, 1).rx(0, 0.5).rx(1, 0.25).probability()
    assert circ == expected


def test_bell_inequality_shots_0():
    circs = create_bell_inequality_circuits(0, 1)
    assert len(circs) == 3
    tasks = run_bell_inequality(circs, LocalSimulator(), shots=0)
    results, pAB, pAC, pBC = get_bell_inequality_results(tasks)
    assert math.isclose(pAB, -0.5)
    assert math.isclose(pBC, -0.5)
    assert math.isclose(pAC, 0.5)
    assert len(results) == 3


def test_bell_inequality():
    circs = create_bell_inequality_circuits(0, 1)
    assert len(circs) == 3
    tasks = run_bell_inequality(circs, LocalSimulator(), shots=10)
    results, pAB, pAC, pBC = get_bell_inequality_results(tasks)
    assert len(tasks) == 3
    results, pAB, pAC, pBC = get_bell_inequality_results(tasks)
    assert len(results) == 3
