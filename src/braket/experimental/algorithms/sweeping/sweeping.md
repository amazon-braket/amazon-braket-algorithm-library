Tensor networks are a natural way to interact with quantum circuits and develop algorithms for them. A major application of tensor networks is in performing circuit optimization,
either via MPS/MPO IRs, or via sweeping. We focus on using tensor network sweeping as an algorithm to optimize arbitrary ansatzes to approximate a target statevector or unitary
in a smooth, consistently improving manner. The approach is a much better alternative for such cases compared to gradient-based and ML approaches which are prone to local minimas
and barren plateaus. For a better understanding of the technique see [Rudolph2022](https://arxiv.org/abs/2209.00595).

<!--
[metadata-name]: Sweeping
[metadata-tags]: Textbook
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/sweeping
-->