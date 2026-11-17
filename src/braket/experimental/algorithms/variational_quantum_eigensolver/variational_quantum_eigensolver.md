The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm that approximates the ground state energy of a Hamiltonian. A parameterized quantum circuit (the ansatz) prepares a trial state, the quantum device estimates its energy by measuring the Pauli terms of the Hamiltonian, and a classical optimizer updates the circuit parameters to minimize that energy. By the variational principle the result is an upper bound on the true ground state energy. VQE is one of the leading candidates for quantum chemistry and materials simulation on near-term devices. This example computes the ground state energy of the hydrogen molecule and its dissociation curve.

<!--
[metadata-name]: Variational Quantum Eigensolver
[metadata-tags]: Textbook
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/variational_quantum_eigensolver
-->
