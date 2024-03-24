import inspect
import math
import random

from braket.circuits import Circuit, Instruction, gates
from braket.devices import Device
from braket.tasks import QuantumTask


def filter_gate_set(max_operands: int):
    """
    Filters and returns Braket gate classes that require a maximum number of qubits.

    Args:
        max_operands (int): Maximum number of qubits (operands) the gate acts on.

    Returns:
        list: A list of gate classes constrained by the maximum number of operands.
    """
    # Use list comprehension to select gate classes that meet the criteria
    # Check if it's a class and if it has the 'fixed_qubit_count' method
    # Check if qubit_count is an integer and if it's lower than or equal to min_qubits
    selected_classes = [
        getattr(gates, cls_name)
        for cls_name in dir(gates)
        if isinstance((cls := getattr(gates, cls_name)), type)
        and hasattr(cls, "fixed_qubit_count")
        and isinstance((qubit_count := cls.fixed_qubit_count()), int)
        and qubit_count <= max_operands
    ]
    return selected_classes


def random_circuit(num_qubits: int, num_gates: int, max_operands: int, seed=None) -> Circuit:
    """
    Generates a random quantum circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_gates (int): Number of instructions (gates) in the circuit.
        max_operands (int): Maximum number of qubits for each gate.
        seed (Optional[int]): Random seed for reproducibility (default is None).

    Returns:
        Circuit: random quantum circuit.
    """
    # Set the seed if provided
    if seed is not None:
        random.seed(seed)
    # Get filtered gate set based on the maximum number of operands (qubits)
    filtered_gate_set = filter_gate_set(max_operands)
    instructions = []
    for _ in range(num_gates):
        # Choose a random gate from the filtered set
        gate_class = random.choice(filtered_gate_set)
        gate_qubits = gate_class.fixed_qubit_count()

        # Select random qubits for the gate
        qubits = random.sample(range(num_qubits), gate_qubits)

        # Get the constructor's signature to determine required parameters
        init_signature = inspect.signature(gate_class.__init__)

        # Calculate the number of parameters (excluding 'self')
        num_params = len(init_signature.parameters) - 1

        # Generate random parameters for the gate in the range [0, 2*pi]
        params = [random.uniform(0, 2 * math.pi) for _ in range(num_params)]

        # Create the gate instance
        gate = gate_class(*params)

        # Add the gate as an instruction
        instructions.append(Instruction(gate, qubits))

    # Create a circuit with the list of instructions
    circuit = Circuit().add(instructions)
    return circuit


def run_random_circuit(
    circuit: Circuit,
    device: Device,
    shots: int = 1000,
) -> QuantumTask:
    """Function to run random circuit and return measurement counts.

    Args:
        circuit (Circuit): Quantum Phase Estimation circuit
        device (Device): Braket device backend
        shots (int) : Number of measurement shots (default is 1000).

    Returns:
        QuantumTask: Task from running Quantum Phase Estimation
    """

    task = device.run(circuit, shots=shots)

    return task
