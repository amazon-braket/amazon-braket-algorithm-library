Simon's algorithm solves the problem of given a function f:{0,1}^n→{0,1}^n that maps bit strings to bit strings. We’re also given the promise that the function f either maps each unique input to a unique output, or maps two distinct inputs to one unique output, without knowing which. This means that f is either one-to-one or two-to-one, and that we are given the promise there exists an unknown string s such that, for all input strings x, f(x)=f(x⊕s). When s is non-zero, the function is two-to-one as it maps exactly two inputs to every unique output. When s is the zero string, the function is one-to-one.

<!--
[metadata-name]: Simon's Algorithm
[metadata-tags]: Textbook
[metadata-url]: https://github.com/aws-samples/amazon-braket-algorithm-library/tree/main/src/braket/experimental/algorithms/simons
-->
