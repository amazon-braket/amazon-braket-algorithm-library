import unittest
from unittest.mock import patch

import networkx as nx
import pytest

from braket.experimental.algorithms.adaptive_shot_allocation.adaptive_allocator import (
    AdaptiveShotAllocator,
    commute,
    gen_commute,
    qwc_commute,
    term_variance_estimate,
    terms_covariance_estimate,
)

# Fixtures for common test setups


@pytest.fixture
def simple_allocator():
    """Create a simple allocator with two Pauli terms"""
    return AdaptiveShotAllocator(["IX", "ZY"], [0.5, -0.3])


@pytest.fixture
def commuting_allocator():
    """Create an allocator with commuting Pauli terms"""
    return AdaptiveShotAllocator(["IX", "ZI", "II"], [0.5, 0.3, 0.1])


@pytest.fixture
def mock_measurements_2terms():
    """Create mock measurement data for 2 terms"""
    return [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 4},
        ],
        [
            {(1, 1): 3, (1, -1): 1, (-1, 1): 2, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},
        ],
    ]


# 1. Tests for Helper Functions


def test_commute_function():
    # Test qubit-wise commutation (qwc=True)
    assert commute("IXYZ", "IXYZ", qwc=True) == True  # Same operators commute
    assert commute("IXYZ", "IXYY", qwc=True) == False  # Different on one qubit
    # I commutes with anything
    assert commute("IXYZ", "IIIZ", qwc=True) == True
    # I commutes with anything
    assert commute("III", "XYZ", qwc=True) == True

    # Test general commutation (qwc=False)
    assert commute("IXYZ", "IXYZ", qwc=False) == True  # Same operators commute
    # Odd number of differences
    assert commute("IXYZ", "IXYY", qwc=False) == False
    # Even number of differences
    assert commute("IXYZ", "IXYI", qwc=False) == True
    # Even number of differences
    assert commute("XY", "YX", qwc=False) == True

    with pytest.raises(ValueError):
        commute("XY", "YXY", qwc=False)


def test_partial_commute_functions():
    # Test qwc_commute (qubit-wise commutation)
    assert qwc_commute("IXYZ", "IXYZ") == True
    assert qwc_commute("IXYZ", "IXYY") == False

    # Test gen_commute (general commutation)
    assert gen_commute("IXYZ", "IXYZ") == True
    assert gen_commute("XY", "YX") == True
    assert gen_commute("XYZ", "YZX") == False  # Odd number of differences


def test_term_variance_estimate():
    # Test with no measurements (prior only)
    # With no measurements, the formula gives 4*(1*1)/(2*3) = 4/6 = 2/3
    assert abs(term_variance_estimate(0) - 2 / 3) < 1e-10

    # Test with mock measurements
    mock_measurements = [[{(1, 1): 10, (1, -1): 0, (-1, 1): 0, (-1, -1): 5}]]
    expected_variance = 4 * ((10 + 1) * (5 + 1)) / ((10 + 5 + 2) * (10 + 5 + 3))
    assert abs(term_variance_estimate(0, mock_measurements) - expected_variance) < 1e-10


def test_terms_covariance_estimate():
    # Test with no measurements (prior only)
    # With no measurements, the formula should give a value close to 0
    assert abs(terms_covariance_estimate(0, 1)) < 1e-10

    # Test with mock measurements
    mock_measurements = [
        [{(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5}],
        [{(1, 1): 7, (1, -1): 0, (-1, 1): 0, (-1, -1): 3}],
    ]
    # Add cross-measurements
    mock_measurements[0].append({(1, 1): 4, (1, -1): 1, (-1, 1): 2, (-1, -1): 3})
    mock_measurements[1].insert(0, {(1, 1): 4, (1, -1): 2, (-1, 1): 1, (-1, -1): 3})

    # Calculate covariance
    result = terms_covariance_estimate(0, 1, mock_measurements)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0  # Covariance should be in this range


# 2. Tests for AdaptiveShotAllocator Class


def test_allocator_initialization():
    # Test valid initialization
    paulis = ["IX", "ZY"]
    coeffs = [0.5, -0.3]
    allocator = AdaptiveShotAllocator(paulis, coeffs)

    assert allocator.num_terms == 2
    assert allocator.paulis == paulis
    assert allocator.coeffs == coeffs
    assert isinstance(allocator.graph, nx.Graph)
    assert len(allocator.cliq) > 0  # Should have at least one clique

    # Test initialization with invalid inputs
    with pytest.raises(ValueError):
        # Mismatched lengths
        AdaptiveShotAllocator(["IX", "ZY", "XY"], [0.5, -0.3])

    with pytest.raises(ValueError):
        # Invalid Pauli string
        AdaptiveShotAllocator(["IX", "ZA"], [0.5, -0.3])


def test_reset_method(simple_allocator):
    # Modify some internal state
    simple_allocator.shots = [10, 20]

    # Reset and check state
    simple_allocator.reset()
    assert simple_allocator.shots is None
    assert all(
        all(outcome == 0 for outcome in simple_allocator.measurements[i][j].values())
        for i in range(simple_allocator.num_terms)
        for j in range(simple_allocator.num_terms)
    )


def test_generate_graph():
    allocator = AdaptiveShotAllocator(["IX", "ZY", "ZI"], [0.5, -0.3, 0.2])

    # Check graph properties
    assert allocator.graph.number_of_nodes() == 3

    # IX and ZI should commute (edge exists)
    assert allocator.graph.has_edge(0, 2)

    # IX and ZY should not commute (no edge)
    assert not allocator.graph.has_edge(0, 1)

    # Test with custom commutation function
    custom_graph = allocator._generate_graph(commute=gen_commute)
    assert isinstance(custom_graph, nx.Graph)


def test_partition_graph():
    # Create a simple graph with known clique structure
    allocator = AdaptiveShotAllocator(["II", "IX", "IZ", "ZI"], [1.0, 1.0, 1.0, 1.0])

    # II, IX, ZI should form one clique (all commute with each other)
    # IZ should be in a separate clique

    # Check that cliques are formed correctly
    cliques_found = False
    for clique in allocator.cliq:
        if set(clique) == {0, 1, 3}:  # II, IX, ZI
            cliques_found = True
            break

    assert cliques_found, "Expected clique not found"


def test_incremental_shot_allocation(simple_allocator):
    # Test allocation with no prior shots
    allocation = simple_allocator.incremental_shot_allocation(10)
    assert len(allocation) == len(simple_allocator.cliq)
    assert sum(allocation) == 10

    # Test with invalid input
    with pytest.raises(ValueError):
        simple_allocator.incremental_shot_allocation(-5)


def test_error_estimate(simple_allocator):
    # Set up some mock shots and measurements
    simple_allocator.shots = [10, 15]

    # Mock measurements to update graph weights
    mock_measurements = [[{(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4}] * 2] * 2
    simple_allocator.update_measurements(mock_measurements)

    # Calculate error estimate
    error = simple_allocator.error_estimate()
    assert isinstance(error, float)
    assert error > 0  # Error should be positive


def test_update_measurements(simple_allocator, mock_measurements_2terms):
    # Update measurements
    simple_allocator.update_measurements(mock_measurements_2terms)

    # Check that shots were updated correctly
    assert simple_allocator.shots is not None
    # Total shots from mock_measurements
    assert sum(simple_allocator.shots) == 20

    # Check that measurements were updated
    assert simple_allocator.measurements[0][0][(1, 1)] == 5
    assert simple_allocator.measurements[1][1][(1, 1)] == 6


def test_expectation_from_measurements():
    allocator = AdaptiveShotAllocator(["IX", "ZY"], [0.5, -0.3])

    # Create mock measurements with known expectation values
    # For IX: (8-2)/10 = 0.6
    # For ZY: no data -> 0.0
    mock_measurements = [
        [
            {(1, 1): 8, (1, -1): 0, (-1, 1): 0, (-1, -1): 2},
            {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0},
        ],
        [
            {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0},
            {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0},
        ],
    ]

    # Expected: 0.5 * 0.6 + (-0.3) * (0.0) = 0.3 + 0.0 = 0.0
    expected = 0.5 * 0.6 + (-0.3) * (0.0)
    result = allocator.expectation_from_measurements(mock_measurements)
    assert abs(result - expected) < 1e-10


def test_validate_measurements():
    # These commute, should be in same clique
    allocator = AdaptiveShotAllocator(["IX", "ZI"], [0.5, 0.3])

    # Valid measurements
    valid_measurements = [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 4},
        ],
        [
            {(1, 1): 3, (1, -1): 1, (-1, 1): 2, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},
        ],
    ]

    assert allocator._validate_measurements(valid_measurements) == True

    # Invalid measurements (wrong size)
    invalid_measurements = [[{(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5}]]

    with pytest.raises(AssertionError):
        allocator._validate_measurements(invalid_measurements)

    # Invalid measurements (inconsistent counts)
    invalid_measurements = [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 3},  # Sum is 9, not 10
        ],
        [
            {(1, 1): 3, (1, -1): 1, (-1, 1): 2, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},
        ],
    ]

    with pytest.raises(AssertionError):
        allocator._validate_measurements(invalid_measurements)

    # Invalid measurements (not symmetric)
    invalid_measurements = [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 4},
        ],
        [
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},
        ],
    ]

    with pytest.raises(AssertionError):
        allocator._validate_measurements(invalid_measurements)

    # Invalid measurements (negative)
    invalid_measurements = [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},
            {(1, 1): 3, (1, -1): 4, (-1, 1): -1, (-1, -1): 4},
        ],
        [
            {(1, 1): 3, (1, -1): -1, (-1, 1): 4, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},
        ],
    ]

    with pytest.raises(AssertionError):
        allocator._validate_measurements(invalid_measurements)


def test_shots_from_measurements():
    allocator = AdaptiveShotAllocator(["IX", "IZ"], [0.5, 0.3])

    # Create mock measurements with known shot counts
    mock_measurements = [
        [
            {(1, 1): 5, (1, -1): 0, (-1, 1): 0, (-1, -1): 5},  # 10 shots
            {(1, 1): 3, (1, -1): 2, (-1, 1): 1, (-1, -1): 4},
        ],
        [
            {(1, 1): 3, (1, -1): 1, (-1, 1): 2, (-1, -1): 4},
            {(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4},  # 10 shots
        ],
    ]

    shots = allocator.shots_from_measurements(mock_measurements)
    assert len(shots) == len(allocator.cliq)
    assert shots[0] == 10  # First clique should have 10 shots


class VisualizationTest(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_graph(*args):
        paulis = ["XX", "IZ", "ZI", "YY", "XI"]
        coeffs = [0.5, 0.3, -0.2, 1.0, 2.0]
        allocator = AdaptiveShotAllocator(paulis, coeffs)
        allocator.visualize_graph()
        allocator.visualize_graph(show_cliques=False)


def test_full_workflow(*args):
    """Test the full workflow of the AdaptiveShotAllocator"""
    # Initialize with simple Pauli terms
    paulis = ["XX", "XY", "ZI"]
    coeffs = [0.5, 0.3, -0.2]
    allocator = AdaptiveShotAllocator(paulis, coeffs)

    # Check initial state
    assert allocator.shots is None

    # Allocate some shots
    allocation = allocator.incremental_shot_allocation(30)
    assert sum(allocation) == 30

    # Create mock measurements based on allocation
    mock_measurements = []
    for i in range(len(paulis)):
        row = []
        for j in range(len(paulis)):
            if i == j:
                # Diagonal elements
                row.append({(1, 1): 6, (1, -1): 0, (-1, 1): 0, (-1, -1): 4})
            else:
                # Off-diagonal elements
                row.append({(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0})
        mock_measurements.append(row)

    # Update with measurements
    allocator.update_measurements(mock_measurements)

    # Check that shots were updated
    assert allocator.shots is not None

    # Calculate expectation value
    expectation = allocator.expectation_from_measurements()
    assert isinstance(expectation, float)

    # Calculate error estimate
    error = allocator.error_estimate()
    assert isinstance(error, float)
    assert error > 0

    # Allocate more shots
    more_allocation = allocator.incremental_shot_allocation(10)
    assert sum(more_allocation) == 10
