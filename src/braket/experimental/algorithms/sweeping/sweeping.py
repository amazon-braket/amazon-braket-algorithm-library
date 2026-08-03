import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray
from tqdm import tqdm

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.experimental.algorithms.sweeping.typing import UNITARY_LAYER


class StateSweepApproximator:
    """Initialize the StateSweepApproximator class.

    This algorithm performs sweeps over unitary layers representing
    an ansatz to be trained to match the target state to improve
    its approximation. Due to the difficulty in converging to good
    fidelity with pure classical ML approaches, and the time it takes
    to train them, sweeping is a much better alternative, running faster
    and giving smooth improvement.

    This approach is inspired by Rudolph et al. [1] which was used to do
    the same task under Oall approach.

    [1] https://www.nature.com/articles/s41467-023-43908-6
    """

    @staticmethod
    def get_tensor_network_from_unitary_layers(
        num_qubits: int, unitary_layers: list[UNITARY_LAYER]
    ) -> qtn.TensorNetwork:
        """Create `qtn.TensorNetwork` from unitary layers.

        Args:
            num_qubits (int): The number of qubits used by target state.
            unitary_layers (list[UNITARY_LAYER]): The unitary layers.

        Returns:
            tensor_network (qtn.TensorNetwork): The tensor network.
        """
        circuit = qtn.Circuit(N=num_qubits)
        gate_tracker: list[str] = []

        for i, layer in enumerate(unitary_layers):
            for j, (qubits, unitary) in enumerate(layer):
                circuit.apply_gate_raw(
                    unitary.reshape(2 * len(qubits) * (2,)),
                    where=qubits,
                    contract=False,
                )
                gate_tracker.append(f"{i}_{j}")

        tensor_network = qtn.TensorNetwork(circuit.psi)

        # We do not want to include the qubits, so we will
        # explicitly control the iteration index
        gate_index = 0

        for gate in tensor_network:
            # We only update the gates, not the qubits
            if "PSI0" in gate.tags:
                continue

            # Remove existing tags from the gate
            gate.drop_tags(tags=gate.tags)

            # Marshal the gate with the gate tracker
            # This is needed to ensure the gates are properly tagged
            # for updating the unitary layers
            gate.add_tag(gate_tracker[gate_index])
            gate_index += 1

        return tensor_network

    @staticmethod
    def sweep_unitary_layers(
        target_mps: qtn.MatrixProductState,
        tensor_network: qtn.TensorNetwork,
        unitary_layers: list[UNITARY_LAYER],
    ) -> list[UNITARY_LAYER]:
        """Sweep the unitary layers to improve the fidelity between
        tensor network created by the unitary layers and the target
        state.

        Args:
            target_mps (qtn.MatrixProductState): The target state represented as a MPS.
            tensor_network (qtn.TensorNetwork): The tensor network created by the unitary layers.
            unitary_layers (list[UNITARY_LAYER]): The unitary layers.

        Returns:
            unitary_layers (list[UNITARY_LAYER]): The unitary layers which have been updated inplace.
        """
        target_mps_adjoint = target_mps.conj()

        current_tn = qtn.MatrixProductState.from_dense(
            tensor_network.to_dense(tensor_network.outer_inds())
        )

        for gate in reversed(tensor_network.tensors):
            num_qubits = int(len(gate.inds) / 2)

            if "PSI0" in gate.tags:
                continue

            left_inds = gate.inds[:num_qubits]
            right_inds = gate.inds[num_qubits:]

            # To avoid unnecessary contraction, we "move"
            # through the tensor network by applying the adjoint
            # of the tensor to the tensor network, and applying
            # the updated tensor to MPS
            current_tn = current_tn @ gate.conj()
            environment_tensor: qtn.TensorNetwork = target_mps_adjoint @ current_tn

            u, _, vh = np.linalg.svd(environment_tensor.to_dense((left_inds), (right_inds)))
            u_new = np.dot(u, vh)

            # To avoid unnecessary contraction, we "move"
            # through the tensor network by applying the adjoint
            # of the tensor to the tensor network, and applying
            # the updated tensor to MPS
            new_tensor = qtn.Tensor(
                u_new.reshape(2 * num_qubits * (2,)).conj(),
                inds=gate.inds,
                tags=gate.tags,
            )

            target_mps_adjoint = target_mps_adjoint @ new_tensor

            gate_tag = list(gate.tags)[0]
            layer_index, block_index = gate_tag.split("_")

            unitary_layers[int(layer_index)][int(block_index)] = (
                unitary_layers[int(layer_index)][int(block_index)][0],
                u_new.conj(),
            )

        return unitary_layers

    @staticmethod
    def circuit_from_unitary_layers(
        num_qubits: int, unitary_layers: list[UNITARY_LAYER]
    ) -> Circuit:
        """Create a `Circuit` instance from
        the unitary layers.

        Args:
            num_qubits (int): The number of qubits used by target state.
            unitary_layers (list[UNITARY_LAYER]): The unitary layers.

        Returns:
            circuit (Circuit): The qiskit circuit.
        """
        circuit = Circuit()

        for layer in unitary_layers:
            for qubits, unitary_matrix in layer:
                circuit.unitary(
                    matrix=unitary_matrix,
                    targets=qubits,
                )

        return circuit

    def __call__(
        self,
        target_state: NDArray[np.complex128],
        unitary_layers: list[UNITARY_LAYER],
        num_sweeps: int,
        log: bool = False,
    ) -> Circuit:
        """Approximate a state via sweeping.

        Args:
            target_state (NDArray[np.complex128]): The state we want to approximate.
            unitary_layers (list[UNITARY_LAYER]): The initial unitary layers.
            num_sweeps (int): The number of times to sweep the unitary layers.
            log (bool): Whether to print logs of fidelity or not.

        Returns:
            circuit (Circuit): The circuit.
        """
        num_qubits = int(np.log2(target_state.shape[0]))
        target_mps = qtn.MatrixProductState.from_dense(target_state)

        for i in tqdm(range(num_sweeps)):
            circuit_tensor_network = self.get_tensor_network_from_unitary_layers(
                num_qubits, unitary_layers
            )

            if log and i % 20 == 0:
                fidelity = (
                    np.abs(
                        np.vdot(
                            target_state,
                            circuit_tensor_network.to_dense(
                                circuit_tensor_network.outer_inds()
                            ).reshape(target_state.shape),
                        )
                    )
                    ** 2
                )
                print(f"Fidelity: {fidelity}")

            unitary_layers = self.sweep_unitary_layers(
                target_mps, circuit_tensor_network, unitary_layers
            )

        circuit = self.circuit_from_unitary_layers(num_qubits, unitary_layers)

        if log:
            result = LocalSimulator("braket_sv").run(circuit.state_vector(), shots=0).result()
            fidelity = (
                np.abs(
                    np.vdot(
                        result.values[0],
                        target_state,
                    )
                )
                ** 2
            )
            print(f"Final Fidelity: {fidelity}")

        return circuit


def sweep_state_approximation(
    target_state: NDArray[np.complex128],
    unitary_layers: list[UNITARY_LAYER],
    num_sweeps: int,
    log: bool = False,
) -> Circuit:
    """Approximate a state via sweeping.

    Args:
        target_state (NDArray[np.complex128]): The state we want to approximate.
        unitary_layers (list[UNITARY_LAYER]): The initial unitary layers.
        num_sweeps (int): The number of times to sweep the unitary layers.
        log (bool): Whether to print logs of fidelity or not.

    Returns:
        circuit (Circuit): The circuit.
    """
    approximator = StateSweepApproximator()
    return approximator(target_state, unitary_layers, num_sweeps, log)
