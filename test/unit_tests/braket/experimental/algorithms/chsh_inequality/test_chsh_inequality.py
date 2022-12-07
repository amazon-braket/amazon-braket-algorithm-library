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
    circuits = create_chsh_inequality_circuits(0, 1, 0, np.pi / 3, 2 * np.pi / 3, np.pi / 3)
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, pAB, pAC, pDB, pDC = get_chsh_results(local_tasks)
    assert np.isclose(pAB, -0.5)
    assert np.isclose(pAC, 0.5)
    assert np.isclose(pDB, -1)
    assert np.isclose(pDC, -0.5)
    assert np.isclose(chsh_value, 2.5)
    assert len(results) == 4


def test_max_chsh_violation():
    circuits = create_chsh_inequality_circuits(0, 1, 0, np.pi / 4, 3 * np.pi / 4, np.pi / 2)
    local_tasks = run_chsh_inequality(circuits, LocalSimulator(), shots=0)
    chsh_value, results, pAB, pAC, pDB, pDC = get_chsh_results(local_tasks)
    assert np.isclose(pAB, -np.sqrt(2) / 2)
    assert np.isclose(pAC, np.sqrt(2) / 2)
    assert np.isclose(pDB, -np.sqrt(2) / 2)
    assert np.isclose(pDC, -np.sqrt(2) / 2)
    assert np.isclose(chsh_value, 2 * np.sqrt(2))
    assert len(results) == 4
