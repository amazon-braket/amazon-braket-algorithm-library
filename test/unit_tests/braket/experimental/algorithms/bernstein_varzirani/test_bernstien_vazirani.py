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

from braket.experimental.algorithms.bernstein_vazirani.bernstein_vazirani import (
    bernstein_vazirani_circuit,
    marginalize_measurements,
)


@pytest.mark.parametrize("hidden_string, shots", [("100", 500), ("10011", 100), ("11", 50)])
def test_bv_circuit(hidden_string: str, shots: int):
    local_simulator = LocalSimulator()
    bv_circuit = bernstein_vazirani_circuit(hidden_string)
    task = local_simulator.run(bv_circuit, shots=shots)
    result = task.result()
    counts = marginalize_measurements(result.measurement_counts)
    print(counts)
    assert counts[hidden_string] == shots
