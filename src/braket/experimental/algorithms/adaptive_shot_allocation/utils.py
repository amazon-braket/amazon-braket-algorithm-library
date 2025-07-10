
import numpy as np

from braket.circuits import Circuit, Observable
from typing import List, Union
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

from braket.experimental.algorithms.adaptive_shot_allocation.adaptive_allocator import (
                                                                AdaptiveShotAllocator, 
                                                                MeasurementData)

"""
Utilities for creating and manipulating quantum circuits.
"""

_localSim = LocalSimulator()

def create_random_state(num_qubits: int = 4) -> Circuit:
    """
    Generate a quantum circuit with random rotations and entanglement.

    Args:
        num_qubits (int): Number of qubits in the circuit

    Returns:
        Circuit: Quantum circuit with random state preparation
    """
    circ = Circuit()

    # First layer of rotations
    for i in range(num_qubits):
        circ.ry(i, np.pi * np.random.rand())

    # Entangling layer
    for i in range(num_qubits - 1):
        circ.cnot(control=i, target=i+1)

    # Second layer of rotations
    for i in range(num_qubits):
        circ.ry(i, np.pi * np.random.rand())

    return circ


def create_bell_state() -> Circuit:
    """
    Generate a Bell state on the first two qubits.

    Returns:
        Circuit: Quantum circuit preparing a Bell state
    """
    return Circuit().h(0).cnot(control=0, target=1)


def observable_from_string(s: str) -> Observable:
    """
    Convert Pauli string to Braket observable.

    Args:
        s (str): Pauli string representation (e.g., "IXYZ")

    Returns:
        Observable: Corresponding Braket observable
    """
    gates = {"I": Observable.I, "X": Observable.X,
             "Y": Observable.Y, "Z": Observable.Z}
    return Observable.TensorProduct([gates[i[1]](i[0]) for i in enumerate(s)])


def get_exact_expectation(circuit: Circuit, paulis: List[str], coeffs: List[float]) -> float:
    """
    Calculate exact expectation value for a Hamiltonian.

    Args:
        circuit (Circuit): Quantum circuit to measure
        paulis (List[str]): List of Pauli string operators
        coeffs (List[float]): Corresponding coefficients

    Returns:
        float: Exact expectation value
    """
    device = _localSim
    e_exact = 0.0
    for c, p in zip(coeffs, paulis):
        expect_circ = circuit.copy()
        expect_circ.expectation(observable_from_string(p))
        result = device.run(expect_circ, shots=0).result()
        e_exact += c * result.values[0]
    return e_exact

"""
Utilities for allocating measurement shots across different measurement groups.
"""

def get_uniform_shots(num_groups: int, total_shots: int) -> List[int]:
    """
    Generate uniform shot allocation across measurement groups.

    Args:
        num_groups (int): Number of measurement groups
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    shots = [total_shots // num_groups] * num_groups
    remainder = total_shots % num_groups
    for i in range(remainder):
        shots[i] += 1
    return shots


def get_random_shots(num_groups: int, total_shots: int) -> List[int]:
    """
    Generate random shot allocation across measurement groups.

    Args:
        num_groups (int): Number of measurement groups
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    weights = np.random.rand(num_groups)
    shots = np.floor(weights * total_shots / sum(weights)).astype(int)
    remainder = total_shots - sum(shots)
    for i in range(remainder):
        shots[i] += 1
    return shots.tolist()


def get_weighted_shots(cliq: List[List[int]], coeffs: List[float], total_shots: int) -> List[int]:
    """
    Generate weighted shot allocation based on coefficient magnitudes.

    Args:
        cliq (List[List[int]]): List of measurement groups (cliques)
        coeffs (List[float]): Coefficients for each term
        total_shots (int): Total number of shots to allocate

    Returns:
        List[int]: Number of shots allocated to each group
    """
    weights = np.array([sum(np.abs(np.array(coeffs)[c])) for c in cliq])
    shots = np.floor(weights * total_shots / sum(weights)).astype(int)
    remainder = total_shots - sum(shots)
    for i in range(remainder):
        shots[i] += 1
    return shots.tolist()


"""
Utilities for handling quantum measurements and running experiments.
"""

def run_fixed_allocation(
    device: Union[LocalSimulator, AwsDevice],
    circuit: Circuit,
    estimator: AdaptiveShotAllocator,
    shot_allocation: List[int]
) -> MeasurementData:
    """
    Run experiment with a specific shot allocation.

    Args:
        device: Quantum device to run the experiment on
        circuit: Quantum circuit to measure
        estimator: AdaptiveShotAllocator instance containing Pauli terms
        shot_allocation: Number of shots to use for each measurement group

    Returns:
        MeasurementData: Measurement outcomes for each term pair
    """
    
    # Step 1. Submit all tasks.
    tasks = {}
    for c_idx, c in enumerate(estimator.cliq):
        if not shot_allocation[c_idx]:
            continue

        measurement_circ = circuit.copy()
        for p in c:
            measurement_circ.sample(
                observable_from_string(estimator.paulis[p]))

        tasks[c_idx] = device.run(
            measurement_circ, shots=shot_allocation[c_idx])
        
    # Step 2. Post-process results.
    measurements = [[{(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0}
                for _ in range(len(estimator.paulis))]
                for _ in range(len(estimator.paulis))]
        
    
    while(tasks):
        task_to_process = None
        
        for c_idx in tasks:
            # Check task status
            state = tasks[c_idx].state()
            assert state in ["CREATED", "QUEUED", "RUNNING", "COMPLETED"], \
                f"Encountered quantum task failure (status: {state})."

            if state == "COMPLETED":
                task_to_process = c_idx
                break
        
        if task_to_process is None:
            continue
            
        # Task is ready for post-processing
        result = tasks[task_to_process].result()
        c = estimator.cliq[task_to_process]
        for i_idx, i in enumerate(c):
            for j_idx, j in enumerate(c):
                for s in range(len(result.values[i_idx])):
                    measurements[i][j][(result.values[i_idx][s],
                                        result.values[j_idx][s])] += 1
        # Remove task from the queue
        tasks.pop(task_to_process)


    return measurements


def run_adaptive_allocation(
    device: Union[LocalSimulator, AwsDevice],
    circuit: Circuit,
    estimator: AdaptiveShotAllocator,
    shots_per_round: int,
    num_rounds: int,
    verbose: bool = False
) -> MeasurementData:
    """
    Run adaptive shot allocation process.

    Args:
        device: Quantum device to run the experiment on
        circuit: Quantum circuit to measure
        estimator: AdaptiveShotAllocator instance
        shots_per_round: Number of shots to allocate in each round
        num_rounds: Number of adaptation rounds
        verbose: Whether to print progress information

    Returns:
        MeasurementData: Final measurement outcomes
    """
    if verbose:
        print(
            f"Running {num_rounds} rounds with {shots_per_round} shots each:")
    for i in range(num_rounds):
        if verbose:
            print(f"Round {i+1}/{num_rounds}...")
        shots_to_run = estimator.incremental_shot_allocation(shots_per_round)
        new_measurements = run_fixed_allocation(
            device, circuit, estimator, shots_to_run)
        estimator.update_measurements(new_measurements)

    return estimator.measurements
