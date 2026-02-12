The Harrow-Hassidim-Lloyd (HHL) algorithm is a quantum algorithm for solving systems of linear equations of the form Ax = b. Given an N×N Hermitian matrix A and a unit vector b, the algorithm produces a quantum state |x⟩ whose amplitudes encode the solution vector x = A⁻¹b. The HHL algorithm is one of the fundamental quantum algorithms expected to provide an exponential speedup over classical methods: for sparse, well-conditioned matrices, HHL runs in O(log(N) κ²) time versus O(Nκ) classically, where κ is the condition number of A. Applications include machine learning, computational finance, solving differential equations, and quantum chemistry.

<!-- 
[metadata-name]: HHL Algorithm
[metadata-tags]: Advanced
[metadata-url]: https://github.com/amazon-braket/amazon-braket-algorithm-library/blob/main/src/braket/experimental/algorithms/hhl
-->
