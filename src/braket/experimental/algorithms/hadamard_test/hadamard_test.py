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

from braket.circuits import Circuit, circuit, Instruction


@circuit.subroutine(register=True)
def hadamard_test(controlled_unitary: Circuit, phase: str = 'real') -> Circuit:
    """Implements the Hadamard test circuit for estimating real or imaginary parts
    of the expected value of a unitary operator.
    
    Args:
        controlled_unitary (Circuit): The unitary operation to be controlled
        phase (str): Either 'real' or 'imaginary' to determine which component to estimate
    
    Returns:
        Circuit: The complete Hadamard test circuit
    """
    if phase not in ['real', 'imaginary']:
        raise ValueError("Phase must be either 'real' or 'imaginary'")

    circ = Circuit()
    
    circ.h(0)
    if phase == 'imaginary':
        circ.s(0).adjoint()
        
    # Add control qubit to the unitary circuit
    for inst in controlled_unitary.instructions:
        targets = [q + 1 for q in inst.target]
        controlled_inst = Instruction(
            operator=inst.operator,
            target=targets,
            control=0
        )
        circ.add_instruction(controlled_inst)
    
    circ.h(0)
    
    return circ
