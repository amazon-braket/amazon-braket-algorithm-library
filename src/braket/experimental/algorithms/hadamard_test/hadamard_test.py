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

from braket.circuits import Circuit, circuit, Instruction, Qubit


@circuit.subroutine(register=True)
def hadamard_test_circuit(ancilla_qubit: Qubit, controlled_unitary: Circuit, component: str = 'real') -> Circuit:
    """Implements the Hadamard test circuit for estimating real or imaginary parts
    of the expected value of a unitary operator.
    
    Args:
        ancilla_qubit (Qubit): The ancilla qubit used as control
        controlled_unitary (Circuit): The unitary operation to be controlled
        component (str): Either 'real' or 'imaginary' to determine which component to estimate
    
    Returns:
        Circuit: The complete Hadamard test circuit
    """
    if component not in ['real', 'imaginary']:
        raise ValueError("Component must be either 'real' or 'imaginary'")

    circ = Circuit()

    circ.h(ancilla_qubit)
    if component == 'imaginary':
        circ.s(ancilla_qubit).adjoint()
    
    # Add control qubit to the unitary circuit
    for inst in controlled_unitary.instructions:
        targets = [q + 1 for q in inst.target]
        controlled_inst = Instruction(
            operator=inst.operator,
            target=targets,
            control=ancilla_qubit
        )
        circ.add_instruction(controlled_inst)
    
    circ.h(ancilla_qubit)
    
    return circ
