import inspect
import random

from braket.circuits import Circuit, Instruction, gates


def get_filtered_gates(max_qubits: int):
    """
    Filters and returns quantum gate classes from the Braket gates module that require a minimum number of qubits.

    Args:
        max_qubits (int): Maximum number of qubits required by the gate.

    Returns:
        list: A list of gate classes that require at least min_qubits.
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
        and qubit_count <= max_qubits
    ]
    return selected_classes


def random_circuit(
    num_qubits: int, num_instructions: int, max_qubits_gate: int, seed=None
) -> Circuit:
    """
    Generates a random quantum circuit using the Braket framework.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_instructions (int): Number of instructions (gates) in the circuit.
        max_qubits_gate (int): Maximum number of qubits for each gate.
        seed (Optional[int]): Random seed for reproducibility (default is None).

    Returns:
        Circuit: A random quantum circuit generated using specified parameters.
    """
    # Set the seed if provided
    if seed is not None:
        random.seed(seed)
    # Get filtered gate set based on the minimum number of qubits
    filtered_gate_set = get_filtered_gates(max_qubits_gate)
    instructions = []
    print(filtered_gate_set)
    for _ in range(num_instructions):
        # Choose a random gate from the filtered set
        gate_class = random.choice(filtered_gate_set)
        gate_qubits = gate_class.fixed_qubit_count()

        # Select random qubits for the gate
        qubits = random.sample(range(num_qubits), gate_qubits)

        # Get the constructor's signature to determine required parameters
        init_signature = inspect.signature(gate_class.__init__)

        # Calculate the number of parameters (excluding 'self')
        num_params = len(init_signature.parameters) - 1

        # Generate random parameters for the gate
        params = [random.uniform(0, 1) for _ in range(num_params)]

        # Create the gate instance
        gate = gate_class(*params)

        # Add the gate as an instruction
        instructions.append(Instruction(gate, qubits))

    # Create a circuit with the list of instructions
    return Circuit().add(instructions)
