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

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.quantum_phase_estimation import quantum_phase_estimation as qpe


# controlled unitary apply function
def controlled_unitary_apply_cnot_func(qpe_circ, control_qubit, query_qubits):
    qpe_circ.cnot(control_qubit, query_qubits)


# CNOT controlled unitary with 2 precision qubits, and H gate query prep
def test_cnot_qpe_run_2_precision_qubits():
    # prep
    precision_qubits = [0, 1]
    query_qubits = [2]

    # create circuit and apply query preparation
    qpe_circ = Circuit().h(query_qubits)

    # apply qpe
    qpe_circ = qpe.quantum_phase_estimation_circuit(
        qpe_circ, precision_qubits, query_qubits, controlled_unitary_apply_cnot_func
    )

    assert len(qpe_circ.instructions) == 10
    assert qpe_circ.depth == 8
    # assert qpe_circ.instructions[3].ascii_symbols == ("C", "X")

    # run QPE
    task = qpe.run_quantum_phase_estimation(qpe_circ, LocalSimulator())

    agg_result = qpe.get_quantum_phase_estimation_results(
        task, precision_qubits, query_qubits, verbose=True
    )

    # validate excepted QPE output
    assert agg_result["measurement_counts"]["001"] > 300
    assert agg_result["measurement_counts"]["000"] > 300
    assert len(agg_result["measurement_counts"]) == 2
    assert agg_result["phases_decimal"] == [0.0]
    assert set(agg_result["eigenvalue_estimates"]) == {1.0 + 0.0j}


# 0 shots results in no measurement counts but valid phase estimates
def test_0_shots():
    # prep
    precision_qubits = [0, 1]
    query_qubits = [2]

    # create circuit and apply query preparation
    qpe_circ = Circuit().h(query_qubits)

    # apply qpe
    qpe_circ = qpe.quantum_phase_estimation_circuit(
        qpe_circ, precision_qubits, query_qubits, controlled_unitary_apply_cnot_func
    )

    # run QPE
    task = qpe.run_quantum_phase_estimation(qpe_circ, LocalSimulator(), shots=0)

    agg_result = qpe.get_quantum_phase_estimation_results(
        task, precision_qubits, query_qubits, verbose=True
    )

    assert agg_result["measurement_counts"] is None
    assert np.isclose(agg_result["phases_decimal"][0], 0.5)


# CNOT controlled unitary with 3 precision qubits, and H gate query prep
def test_cnot_qpe_run_3_precision_qubits():
    # prep
    precision_qubits = [0, 1, 2]
    query_qubits = [3]

    # create circuit and apply query preparation
    qpe_circ = Circuit().h(query_qubits)

    # apply qpe
    qpe_circ = qpe.quantum_phase_estimation_circuit(
        qpe_circ, precision_qubits, query_qubits, controlled_unitary_apply_cnot_func
    )

    print("Circuit: ", qpe_circ)
    # run QPE
    task = qpe.run_quantum_phase_estimation(qpe_circ, LocalSimulator())

    agg_result = qpe.get_quantum_phase_estimation_results(
        task, precision_qubits, query_qubits, verbose=False
    )

    # validate excepted QPE output
    assert agg_result["measurement_counts"]["0001"] > 300
    assert agg_result["measurement_counts"]["0000"] > 300
    assert len(agg_result["measurement_counts"]) == 2
    assert agg_result["phases_decimal"] == [0.0]
    assert set(agg_result["eigenvalue_estimates"]) == {1.0 + 0.0j}


# CNOT controlled unitary with 2 precision qubits, and HX gate query prep
def test_cnot_qpe_run_HX_eigenstate():
    # prep
    precision_qubits = [0, 1]
    query_qubits = [2]

    # create circuit and apply query preparation
    qpe_circ = Circuit().x(query_qubits).h(query_qubits)

    # apply qpe
    qpe_circ = qpe.quantum_phase_estimation_circuit(
        qpe_circ, precision_qubits, query_qubits, controlled_unitary_apply_cnot_func
    )

    print("Circuit: ", qpe_circ)
    # run QPE
    task = qpe.run_quantum_phase_estimation(qpe_circ, LocalSimulator())

    agg_result = qpe.get_quantum_phase_estimation_results(
        task, precision_qubits, query_qubits, verbose=True
    )

    # validate excepted QPE output
    assert agg_result["measurement_counts"]["101"] > 300
    assert agg_result["measurement_counts"]["100"] > 300
    assert len(agg_result["measurement_counts"]) == 2
    assert agg_result["phases_decimal"] == [0.5]
    assert set(agg_result["eigenvalue_estimates"]) == {-1.0 + 0.0j}


# CNOT controlled unitary with 2 precision qubits, and X gate query prep
def test_cnot_qpe_run_X_eigenstate():
    # prep
    precision_qubits = [0, 1]
    query_qubits = [2]

    # create circuit and apply query preparation
    qpe_circ = Circuit().x(query_qubits)

    # apply qpe
    qpe_circ = qpe.quantum_phase_estimation_circuit(
        qpe_circ, precision_qubits, query_qubits, controlled_unitary_apply_cnot_func
    )

    print("Circuit: ", qpe_circ)
    # run QPE
    task = qpe.run_quantum_phase_estimation(qpe_circ, LocalSimulator())

    agg_result = qpe.get_quantum_phase_estimation_results(
        task, precision_qubits, query_qubits, verbose=True
    )

    # validate excepted QPE output
    assert agg_result["measurement_counts"]["000"] > 100
    assert agg_result["measurement_counts"]["100"] > 100
    assert agg_result["measurement_counts"]["101"] > 100
    assert agg_result["measurement_counts"]["001"] > 100
    assert len(agg_result["measurement_counts"]) == 4
    assert set(agg_result["phases_decimal"]) == {0.0, 0.5}
    assert set(agg_result["eigenvalue_estimates"]) == {1.0 + 0.0j, -1.0 + 0.0j}


# inverse QFT circuit validation with 2 input qubits
def test_inverse_qft():
    # prep
    qubits = [0, 1]

    # run inverse qft
    qft_circ = qpe.inverse_qft(qubits)

    # validate circuit output
    assert len(qft_circ.instructions) == 4
    assert qft_circ.instructions[2].operator.ascii_symbols == ("C", "PHASE(-1.57)")


# The qubit label 0 must be a valid precision control, just like every other label.
def test_qpe_phase_estimation_is_independent_of_qubit_labels():
    def controlled_z(qpe_circ, control_qubit, query_qubits):
        qpe_circ.cz(control_qubit, query_qubits[0])

    for precision_qubit, query_qubit in [(0, 1), (1, 0)]:
        qpe_circ = Circuit().x(query_qubit)
        qpe_circ.add_circuit(
            qpe.quantum_phase_estimation([precision_qubit], [query_qubit], controlled_z)
        )
        qpe_circ.probability(target=[precision_qubit])
        probabilities = LocalSimulator().run(qpe_circ, shots=0).result().values[0]
        np.testing.assert_allclose(probabilities, [0, 1], atol=1e-12)


def test_qpe_controlled_unitary_powers_include_qubit_zero():
    controls = []

    def record_controlled_x(qpe_circ, control_qubit, query_qubits):
        controls.append(control_qubit)
        qpe_circ.cnot(control_qubit, query_qubits[0])

    qpe.quantum_phase_estimation([0, 1, 3], [2], record_controlled_x)
    assert controls == [3] + [1] * 2 + [0] * 4
