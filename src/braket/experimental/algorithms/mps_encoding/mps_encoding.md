Matrix Product State (MPS) encoding tackles the critical task of state preparation in $O(N)$ circuit depth, an exponential reduction in circuit depth compared to
exact encoding algorithms like Shende, Isometry, and Mottonen. The current implementation uses an analytical decomposition based on work by
Ran et al. to create a staircase ansatz comprised of 1 and 2 qubit unitary gates. Main consideration with the decomposition is that it is prone to
approximation limitations imposed by MPS IR, which limits the adequate performance of the synthesis to area-law entangled states only.

<!--
[metadata-name]: MPS Encoding
[metadata-tags]: Textbook
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/mps_encoding
-->
