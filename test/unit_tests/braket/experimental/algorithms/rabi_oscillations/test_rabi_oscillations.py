import math

from braket.experimental.algorithms.rabi_oscillations import rabi_circuit, rabi_probability


def test_rabi_probability_edge_cases():
    assert rabi_probability(0.0) == 0.0
    assert math.isclose(rabi_probability(math.pi), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(rabi_probability(2 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_rabi_circuit_has_single_rx_instruction():
    circ = rabi_circuit(0.123)
    assert circ.qubit_count == 1
    # one Rx instruction on qubit 0
    assert len(circ.instructions) == 1
    instr = circ.instructions[0]
    assert instr.operator.name == "Rx"
    assert list(instr.target) == [0]
