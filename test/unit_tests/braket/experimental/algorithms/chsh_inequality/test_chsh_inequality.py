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
from braket.devices import LocalSimulator

from braket.experimental.algorithms.chsh_inequality import (
    create_chsh_inequality_circuits,
    get_chsh_results,
    run_chsh_inequality,
)


def test_chsh_reduces_to_bell():
    circuits = create_chsh_inequality_circuits(0, 1, a1=(np.pi / 3), a2=0, b1=0, b2=(2 * np.pi / 3))
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(local_tasks)
    assert np.isclose(E_a1b1, -0.5)
    assert np.isclose(E_a1b2, -0.5)
    assert np.isclose(E_a2b1, -1)
    assert np.isclose(E_a2b2, 0.5)
    assert np.isclose(chsh_value, -2.5)
    assert len(results) == 4


def test_chsh_reduces_to_bell_not_verbose():
    circuits = create_chsh_inequality_circuits(0, 1, a1=(np.pi / 3), a2=0, b1=0, b2=(2 * np.pi / 3))
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(
        local_tasks, verbose=False
    )
    assert np.isclose(E_a1b1, -0.5)
    assert np.isclose(E_a1b2, -0.5)
    assert np.isclose(E_a2b1, -1)
    assert np.isclose(E_a2b2, 0.5)
    assert np.isclose(chsh_value, -2.5)
    assert len(results) == 4


def test_chsh_no_violation():
    circuits = create_chsh_inequality_circuits(0, 1, a1=0, a2=0, b1=0, b2=0)
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(local_tasks)
    assert np.isclose(E_a1b1, -1)
    assert np.isclose(E_a1b2, -1)
    assert np.isclose(E_a2b1, -1)
    assert np.isclose(E_a2b2, -1)
    assert np.isclose(chsh_value, -2)
    assert len(results) == 4


def test_max_chsh_violation():
    circuits = create_chsh_inequality_circuits(
        0, 1, a1=(np.pi / 2), a2=0, b1=(np.pi / 4), b2=(3 * np.pi / 4)
    )
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, E_a1b1, E_a1b2, E_a2b1, E_a2b2 = get_chsh_results(local_tasks)
    assert np.isclose(E_a1b1, -np.sqrt(2) / 2)
    assert np.isclose(E_a1b2, -np.sqrt(2) / 2)
    assert np.isclose(E_a2b1, -np.sqrt(2) / 2)
    assert np.isclose(E_a2b2, np.sqrt(2) / 2)
    assert np.isclose(chsh_value, -2 * np.sqrt(2))
    assert len(results) == 4
