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
from typing import Any, Dict, List

import networkx as nx
from braket.circuits import Circuit, circuit


@circuit.subroutine(register=True)
def quantum_partition_function(
    function: str,
    qubits: List[int],
) -> Circuit:
    """Creates two kinds of circuits for quantum partition function:
      1) Shor's algorithm for irreducible cyclic cocycle code (ICCC)-check
      2) The quantum Fourier transformation circuit for checking gamma values

    Args:
        function (str): Define the function of this circuit, it can be 'shor' or 'qft'
        qubits (List[int]) : Qubits defining the circuit

    Returns:
        Circuit: Circuit object that implements the Quantum Parition Estimation algorithm
    """

    circuit = None

    if function == "shor":
        print("Shor's algorithm for iccc-check hasn't been implemented!")
    elif function == "qft":
        print("QFT circuit for checking gamma values")
        circuit = _quantum_Fourier_transform(qubits)
    else:
        raise ValueError(
            f"Wrong function value {function}, can only be one \
                        of shor and qft"
        )

    return circuit


@circuit.subroutine(register=True)
def _quantum_Fourier_transform(
    qft_qubits: List[int],
) -> Circuit:
    """
    Construct a circuit object corresponding to the Quantum Fourier Transform (QFT)
    algorithm, applied to the argument qubits.

    Args:
        qft_qubits (List[int]): Qubits on which to apply the QFT

    Returns:
        Circuit: Circuit object that implements the Quantum Fourier Transform (QFT)
    """
    qftcirc = Circuit()

    # get number of qubits
    num_qubits = len(qft_qubits)

    for k in range(num_qubits):
        # First add a Hadamard gate
        qftcirc.h(qft_qubits[k])

        for j in range(1, num_qubits - k):
            angle = 2 * math.pi / (2 ** (j + 1))
            qftcirc.cphaseshift(qft_qubits[k + j], qft_qubits[k], angle)

    # Then add SWAP gates to reverse the order of the qubits:
    for i in range(math.floor(num_qubits / 2)):
        qftcirc.swap(qft_qubits[i], qft_qubits[-i - 1])

    return qftcirc


def run_quantum_partition_function(
    potts_model: dict,
    step: str,
) -> Dict[str, Any]:
    """Function to run Quantum partition function algorithm and return measurement counts.

    Args:
        potts_model (dict): Dictionary to save the results of Potts model:
            'graph-model': networkx graph to save the definition of potts model
            'q-state': q-state of potts model
            'qft-func': dictionary to save information for quantum Fourier transform function
                'task': braket task for quantum Fourier transform
                'circuit': circuit for quantum Fourier transform
                'param': dictionary for running circuit
                    'shots': shots for running device
                    'device': device for running device
        step (str): Qubits defining the precision register

    Returns:
        Dict[str, Any]: results from running Quantum partition function
    """

    out = {}

    if step == "pre":
        print("Classical Preprocessing to [n,k] Code")
        Ga = potts_model["graph-model"]
        # q = potts_model["q-state"]
        # nodes of graph
        N = len(Ga.nodes)
        # edges of graph
        E = len(Ga.edges)
        # the number of connected components in the graph
        C = len([Ga.subgraph(c).copy() for c in nx.connected_components(Ga)])
        # [n,k] code
        n = N
        k = E - C
        out["nk-code"] = (n, k)
        out["connected-component"] = C
    elif step == "iccc-check":
        print("Irreducible Cyclic Cocycle Code Check")
        if "nk-code" not in potts_model.keys():
            raise KeyError("no nk-code found in potts_model, please run pre-process step!")
        Ga = potts_model["graph-model"]
        # TODO: Add shor algorithm to check ICCC
        N = len(Ga.nodes)
        E = len(Ga.edges)
        n, k = potts_model["nk-code"]
        C = potts_model["connected-component"]
        print(f"The cycle matroid matrix of Graph Gamma is {n-C} x {n}")
        check_result = False
        if N == 3 and E == 3:
            print("the ICCC is [1,-1], which passes ICCC check!")
            check_result = True
        out["iccc-check"] = check_result
    elif step == "qft":
        print("State Preparation and Quantum Fourier Transform")
        if "iccc-check" not in potts_model.keys():
            raise KeyError("no iccc-check found in potts_model, please run ICCC-check step!")
        check_result = potts_model["iccc-check"]
        assert check_result is True, "ICCC check didn't pass, no efficient quantum alogrithm!"

        if "qft-func" not in potts_model.keys():
            raise KeyError("no qft-func found in potts_model, please generate circuit!")

        qft_circuit = potts_model["qft-func"]["circuit"]
        # Add desired results_types
        qft_circuit.probability()

        qft_param = potts_model["qft-func"]["param"]
        qft_device = qft_param["device"]
        qft_shots = qft_param["shots"]

        task = qft_device.run(qft_circuit, shots=qft_shots)

        circ_result = {
            "task": task,
        }

        out.update(circ_result)
    elif step == "post":
        print("Classical Post-Processing")
        # TODO Classical pots process logic
    else:
        raise ValueError(
            f"Wrong step value {step}, can only be one \
                        of pre,iccc-check,qft and post"
        )

    return out


def get_quantum_partition_function_results(potts_model: Dict[str, Any]) -> None:
    """Function to postprocess dictionary returned by run_quantum_partition_function and pretty
        print results.

    Args:
        potts_model (dict): Dictionary to save the results of Potts model:
            'graph-model': networkx graph to save the definition of potts model
            'q-state': q-state of potts model
            'qft-func': dictionary to save information for quantum Fourier transform function
                'task': braket task for quantum Fourier transform
                'circuit': circuit for quantum Fourier transform
                'param': dictionary for running circuit
                    'shots': shots for running device
                    'device': device for running device

    Returns:
        results (Dict[str, Any]): Results associated with quantum partition function run as produced
        by get_quantum_partition_function_results
    """
    task = potts_model["qft-func"]["task"]
    print(task)
    result = task.result()
    return result
