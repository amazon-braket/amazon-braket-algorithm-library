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

from braket.circuits import Circuit
from braket.devices import LocalSimulator

from braket.experimental.algorithms.quantum_partition_function import (
    quantum_partition_function as qpf,
)


# CNOT controlled unitary with 2 precision qubits, and H gate query prep
def test_qpf_qft_run_2_qubits():
    # prep
    pi = 3.14
    qubits = [0, 1]
    q0 = qubits[0]
    q1 = qubits[1]

    # Prepare state for quantum fourier circuit
    qpf_circ = Circuit().h(range(2)).rz(q0, -pi).rz(q1, pi)

    # apply qpf
    # qpf_qft.quantum_partition_function(
    #     'qft', qubits
    # )
    qpf_circ.add(qpf.quantum_partition_function("qft", qubits))

    assert len(qpf_circ.instructions) == 8
    assert qpf_circ.depth == 6
    assert qpf_circ.instructions[3].ascii_symbols == ("Rz(3.14)",)

    # run qpf
    potts_model = {}
    potts_model["iccc-check"] = True
    potts_model["qft-func"] = {}
    potts_model["qft-func"]["circuit"] = qpf_circ
    potts_model["qft-func"]["param"] = {}
    potts_model["qft-func"]["param"]["shots"] = 1000
    potts_model["qft-func"]["param"]["device"] = LocalSimulator()

    step = "qft"

    result_dict = qpf.run_quantum_partition_function(potts_model, step)

    # print(
    #     f"test_cnot_qpf_run_2_precision_qubits Results: \
    #   {qpf.get_quantum_partition_function_results(result)}"
    # )

    # validate excepted qpf output
    counts_result = result_dict["qft-func"]["task"].result().measurement_counts

    assert counts_result["11"] > 300
    assert counts_result["01"] > 300
    assert len(counts_result) == 2
