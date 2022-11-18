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

# Quantum Fourier Transform: Amazon Braket Algorithm Library

import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_fourier_transform import (
    quantum_fourier_transform as qft,
)


def test_qft():
    result = qft.run_quantum_fourier_transform(
        n_qubits=2, n_shots=100, state_prep_circ=Circuit().h(0).h(1), device=LocalSimulator()
    )
    assert np.allclose(result.values[0], [1.0, 0.0, 0.0, 0.0])


def test_inverse_qft():
    result = qft.run_quantum_fourier_transform(
        n_qubits=2,
        n_shots=100,
        state_prep_circ=Circuit().h(0).h(1),
        device=LocalSimulator(),
        inverse=True,
    )
    assert np.allclose(result.values[0], [1.0, 0.0, 0.0, 0.0])
