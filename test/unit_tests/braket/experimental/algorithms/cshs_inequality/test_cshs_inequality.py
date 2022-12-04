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

import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.cshs_inequality import (
    bell_singlet,
    bell_singlet_rotated_basis,
    get_cshs_results,
    submit_cshs_tasks,
)


def test_bell_singlet():
    circ = bell_singlet(0, 1)
    expected = Circuit().h(0).cnot(0, 1)
    assert circ == expected


def test_bell_singlet_rotated_basis():
    circ = bell_singlet_rotated_basis(0, 1, 0.5, 0.25)
    expected = Circuit().h(0).cnot(0, 1).ry(0, 0.5).ry(1, 0.25)
    assert circ == expected


def test_bell_inequality():
    tasks = submit_cshs_tasks(LocalSimulator())
    assert len(tasks) == 4
    cshs_value, cshs_inequality_lhs, results, E_ab, E_ab_, E_a_b, E_a_b_ = get_cshs_results(tasks)
    assert len(results) == 4
