The Quantum Counting algorithm, introduced by Brassard, Høyer, and Tapp (1998), solves a generalization of the quantum search problem. Instead of merely detecting whether a marked item exists in an unstructured database of N elements, quantum counting determines the number of marked items M. The algorithm combines Grover's search operator with Quantum Phase Estimation (QPE): it prepares the Grover operator G = D·O (oracle followed by diffusion), whose eigenvalues encode θ such that M = N·sin²(θ/2), then uses QPE to estimate θ with t precision qubits. The algorithm achieves a quadratic speedup over classical counting, requiring only O(√N) oracle queries instead of the classical Θ(N).

<!--
[metadata-name]: Quantum Counting
[metadata-tags]: Textbook
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/quantum_counting
-->
