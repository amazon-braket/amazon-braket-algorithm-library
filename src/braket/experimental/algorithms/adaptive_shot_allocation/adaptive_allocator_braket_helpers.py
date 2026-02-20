"""
Helper functions to handle quantum measurements and adaptive shot allocation
experiments on Amazon Braket.
"""

from typing import List, Union

from braket.aws import AwsDevice
from braket.circuits import Circuit, Observable
from braket.devices import LocalSimulator
from braket.experimental.algorithms.adaptive_shot_allocation.adaptive_allocator import (
    AdaptiveShotAllocator,
    MeasurementData,
)


def observable_from_string(pauli_string: str) -> Observable:
    """
    Convert Pauli string to Braket observable.

    Args:
        s (str): Pauli string representation (e.g., "IXYZ")

    Returns:
        Observable: Corresponding Braket observable
    """
    gates = {"I": Observable.I, "X": Observable.X, "Y": Observable.Y, "Z": Observable.Z}
    return Observable.TensorProduct([gates[i[1]](i[0]) for i in enumerate(pauli_string)])


def run_fixed_allocation(
    device: Union[LocalSimulator, AwsDevice],
    circuit: Circuit,
    estimator: AdaptiveShotAllocator,
    shot_allocation: List[int],
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
            measurement_circ.sample(observable_from_string(estimator.paulis[p]))

        tasks[c_idx] = device.run(measurement_circ, shots=shot_allocation[c_idx])

    # Step 2. Post-process results.
    measurements = [
        [{(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0} for _ in range(len(estimator.paulis))]
        for _ in range(len(estimator.paulis))
    ]

    while tasks:
        task_to_process = None

        for c_idx in tasks:
            # Check task status
            state = tasks[c_idx].state()
            assert state in ["CREATED", "QUEUED", "RUNNING", "COMPLETED"], (
                f"Encountered quantum task failure (status: {state})."
            )

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
                    measurements[i][j][(result.values[i_idx][s], result.values[j_idx][s])] += 1
        # Remove task from the queue
        tasks.pop(task_to_process)

    return measurements


def run_adaptive_allocation(
    device: Union[LocalSimulator, AwsDevice],
    circuit: Circuit,
    estimator: AdaptiveShotAllocator,
    shots_per_round: int,
    num_rounds: int,
    verbose: bool = False,
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
        print(f"Running {num_rounds} rounds with {shots_per_round} shots each:")
    for i in range(num_rounds):
        if verbose:
            print(f"Round {i + 1}/{num_rounds}...")
        shots_to_run = estimator.incremental_shot_allocation(shots_per_round)
        new_measurements = run_fixed_allocation(device, circuit, estimator, shots_to_run)
        estimator.update_measurements(new_measurements)

    return estimator.measurements
