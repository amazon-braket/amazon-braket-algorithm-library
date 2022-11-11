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

from braket.devices import LocalSimulator

from braket.experimental.algorithms.bells_inequality import get_bell_results, submit_bell_tasks


def test_bell():
    local_simulator = LocalSimulator()
    local_tasks = submit_bell_tasks(local_simulator, shots=0)
    results, pAB, pAC, pBC = get_bell_results(local_tasks)
    assert math.isclose(pAB, -0.5)
    assert math.isclose(pBC, -0.5)
    assert math.isclose(pAC, 0.5)
    assert len(results) == 3
