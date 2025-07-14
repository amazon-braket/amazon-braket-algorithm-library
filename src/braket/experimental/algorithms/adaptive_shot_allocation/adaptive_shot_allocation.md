Properly allocating shots to different terms of a Hamiltonian when calculating expectation values can drasticly improve the accuracy of the estimator. A generally adopted strategy is to allocate shots proportionally to the magnitude of a term's coefficient. However, in certain situations that strategy can severely underperform. The code here implements an adaptive shot-allocation algorithm from (see "Adaptive Estimation of Quantum Observables" by Shlosberg et al, DOI: 10.22331/q-2023-01-26-906), and the accompanying notebooks illustrate the benefits of using it.

<!--
[metadata-name]: Adaptive Shot Allocation
[metadata-tags]: Advanced
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/adaptive_shot_allocation
-->
