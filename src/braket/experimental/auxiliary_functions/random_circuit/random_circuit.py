import inspect
import math
import random
from typing import List, Optional

from braket.circuits import Circuit, Gate, Instruction
from braket.circuits.gates import CNot, H, S, T


def random_circuit(
    num_qubits: int,
    num_gates: int,
    gate_set: Optional[List[Gate]] = None,
    seed: Optional[int] = None,
) -> Circuit:
    """
    Generates a random quantum circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_gates (int): Number of instructions (gates) in the circuit.
        gate_set (Optional[List[Gate]]): List of basis gates for the random circuit
            (default is None).
        seed (Optional[int]): Random seed for reproducibility (default is None).

    Returns:
        Circuit: random quantum circuit.
    """
    # Set the seed if provided
    if seed is not None:
        random.seed(seed)

    # Default gate_set (Clifford + T) if gate_set is None
    if not gate_set:
        gate_set = [CNot, S, T, H]

    instructions = []
    for _ in range(num_gates):
        gate = random.choice(gate_set)
        gate_qubits = gate.fixed_qubit_count()

        # Select random qubits for the gate
        qubits = random.sample(range(num_qubits), gate_qubits)

        # Get the constructor's signature to determine required parameters
        init_signature = inspect.signature(gate.__init__)

        # Calculate the number of parameters (excluding 'self')
        num_params = len(init_signature.parameters) - 1

        # Generate random parameters for the gate in the range [0, 2*pi]
        params = [random.uniform(0, 2 * math.pi) for _ in range(num_params)]

        # Create the gate instance
        g = gate(*params)

        # Add the gate as an instruction
        instructions.append(Instruction(g, qubits))

    # Create a circuit with the list of instructions
    circuit = Circuit().add(instructions)
    return circuit
