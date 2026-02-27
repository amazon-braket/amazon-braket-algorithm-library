import random
import warnings
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import approximation

# Type alias for measurement data structure (see AdaptiveShotAllocator.measurements)
MeasurementData = List[List[Dict[Tuple[int, int], int]]]

"""
Helper functions for Pauli operator commutation and Bayesian statistics calculations.
These functions support the AdaptiveShotAllocator class.
"""


def commute(a: str, b: str, qwc: bool = True) -> bool:
    """
    Check if two Pauli strings commute.

    Args:
        a (str): First Pauli string (e.g., "IXYZ")
        b (str): Second Pauli string
        qwc (bool): If True, use qubit-wise commutation rules.
                   If False, use general commutation rules.

    Returns:
        bool: True if the Pauli strings commute, False otherwise

    Raises:
        ValueError: If the Pauli strigns are of different length.
    """
    if len(a) != len(b):
        raise ValueError("Pauli strings must be of the same length.")
    count = 0
    for i in zip(a, b):
        count += (i[0] != "I") and (i[1] != "I") and (i[0] != i[1])
    return (count == 0) if qwc else (count % 2 == 0)


# Partial functions for specific commutation checks
qwc_commute = partial(commute, qwc=True)  # Qubit-wise commutation
gen_commute = partial(commute, qwc=False)  # General commutation

# BAYESIAN STATISTICS (closed formulas from Appendix B of arXiv:2110.15339v6)


def term_variance_estimate(
    term_idx: int, measurements: Union[MeasurementData, None] = None
) -> float:
    """
    Estimate variance for a single Pauli term.
    See Eq 14 in Appendix B of "Adaptive Estimation of Quantum Observables (arXiv:2110.15339v6)

    Args:
        term_idx (int): Index of the Pauli term
        measurements (List[List[Dict]], optional): Measurement outcomes

    Returns:
        float: Estimated variance for the term
    """
    x0, x1 = 0, 0
    if measurements:
        x0 = measurements[term_idx][term_idx][(1, 1)]
        x1 = measurements[term_idx][term_idx][(-1, -1)]
    return 4 * ((x0 + 1) * (x1 + 1)) / ((x0 + x1 + 2) * (x0 + x1 + 3))


def terms_covariance_estimate(
    i: int, j: int, measurements: Union[MeasurementData, None] = None
) -> float:
    """
    Estimate covariance between two Pauli terms.
    See Eq 25-6 in Appendix B of "Adaptive Estimation of Quantum Observables (arXiv:2110.15339v6)

    Args:
        i (int): Index of first Pauli term
        j (int): Index of second Pauli term
        measurements (List[List[Dict]], optional): Measurement outcomes

    Returns:
        float: Estimated covariance between terms
    """
    x0, x1 = 0, 0
    y0, y1 = 0, 0
    xy00, xy01, xy10, xy11 = 0, 0, 0, 0

    if measurements:
        x0, x1 = measurements[i][i][(1, 1)], measurements[i][i][(-1, -1)]
        y0, y1 = measurements[j][j][(1, 1)], measurements[j][j][(-1, -1)]
        xy00 = measurements[i][j][(1, 1)]
        xy01 = measurements[i][j][(1, -1)]
        xy10 = measurements[i][j][(-1, 1)]
        xy11 = measurements[i][j][(-1, -1)]

    # Calculate prior probabilities
    p00 = 4 * ((x0 + 1) * (y0 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2))
    p01 = 4 * ((x0 + 1) * (y1 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2))
    p10 = 4 * ((x1 + 1) * (y0 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2))
    p11 = 4 * ((x1 + 1) * (y1 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2))

    # Return Bayesian covariance estimate
    return (
        4
        * ((xy00 + p00) * (xy11 + p11) - (xy01 + p01) * (xy10 + p10))
        / ((xy00 + xy01 + xy10 + xy11 + 4) * (xy00 + xy01 + xy10 + xy11 + 5))
    )


class AdaptiveShotAllocator:
    """
    A class for adaptive measurement allocation in expectation value calculations.

    This class manages a Hamiltonian encoded as a graph of commuting Pauli operators,
    and estimates measurement covariances in clique measurements, to reduce the variance
    of an expectation value calculation. It uses a graph-based approach where:
    - Nodes represent Pauli terms
    - Edges connect commuting terms
    - Terms are grouped into cliques for simultaneous measurement
    - Shot allocation is optimized based on (estimation of) clique error contributions

    Attributes:
        num_terms (int): Number of Pauli terms in the Hamiltonian
        paulis (List[str]): List of Pauli string representations
        coeffs (List[float]): Coefficients for each Pauli term
        graph (nx.Graph): Graph representing commuting relationships
        cliq (List[List[int]]): List of cliques for measurement grouping
        measurements (MeasurementData): Measurement outcomes for term pairs
            "measurements[i][j][key]" is the nuber of times paulis[i] and paulis[j]
            have been measured together and produced the tuple "key" (one of the
            four possible outcomes (1,1), (1,-1), (-1, 1) and (-1,-1)).
            Note that measurements[i][j][key] should be non-zero only if i and j are in
            the same clique. In practice, measurements[i][j] is never referenced if i and j
            are not members of the same clique.
        shots (Union[List[int], None]): Number of shots allocated to each clique
    """

    num_terms: int
    paulis: List[str]
    coeffs: List[float]
    graph: nx.Graph
    cliq: List[List[int]]
    measurements: MeasurementData
    shots: Union[List[int], None]

    def __init__(self, paulis: List[str], coeffs: List[float]) -> None:
        """
        Initialize the AdaptiveShotAllocator with Pauli terms and their coefficients.

        Args:
            paulis (List[str]): List of Pauli string representations (e.g., "IXYZ")
            coeffs (List[float]): Corresponding coefficients for each Pauli term

        Raises:
            ValueError: If number of Paulis doesn't match number of coefficients
            ValueError: If any Pauli string contains invalid characters
        """
        self.num_terms = len(coeffs)
        if len(paulis) != self.num_terms:
            raise ValueError("Number of Paulis must match coefficients")

        # Validate Pauli strings
        valid_chars = set("IXYZ")
        for pauli in paulis:
            if not set(pauli).issubset(valid_chars):
                raise ValueError(f"Invalid Pauli string: {pauli}. Must only contain I, X, Y, or Z")
        self.paulis = paulis
        self.coeffs = coeffs
        self.graph = self._generate_graph()
        self._partition_graph()
        self.reset()

    def reset(self):
        """
        Reset the measurement data and shot allocation.
        This would be used e.g. when changing the state on which the Hamiltonian
        expectation value is calculated.

        This method:
        - Reinitializes the measurement counts matrix
        - Clears shot allocation history
        - Updates graph weights to initial values
        """
        # Initialize measurement counts for all term pairs
        self.measurements = [
            [{(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0} for _ in range(self.num_terms)]
            for _ in range(self.num_terms)
        ]
        self.shots = None  # Clear shot allocation history
        self._update_graph_weights()  # Reset graph weights

    def _generate_graph(self, commute: Callable[[str, str], bool] = qwc_commute):
        """
        Generate a graph representing commuting relationships between Pauli terms.

        Args:
            commute (Callable[[str, str], bool]): Function to test if two Pauli terms commute.
                Defaults to qubit-wise commutation check.

        Returns:
            networkx.Graph: Graph where nodes are Pauli terms and edges connect commuting terms

        Note:
            Each node is labeled with its Pauli string representation.
            Edges are added between terms that commute according to the provided function.
        """
        if commute != qwc_commute:
            warnings.warn(
                "Braket only supports simultaneous measurement of qubit-wise commuting operators."
            )

        # Create graph and add nodes with Pauli string labels
        self.graph = nx.Graph()
        for p in range(self.num_terms):
            self.graph.add_node(p, label=self.paulis[p])

        # Add edges between commuting terms
        for i in range(self.num_terms):
            # we only check upper triangle as this is not a directed graph
            for j in range(i, self.num_terms):
                if commute(self.paulis[i], self.paulis[j]):
                    self.graph.add_edge(i, j)

        return self.graph

    def _partition_graph(self):
        """
        Partition the graph into cliques of commuting Pauli terms.

        Uses the clique removal algorithm to find a partition of the graph into cliques.
        This grouping allows for simultaneous measurement of commuting terms.

        The cliques are stored in self.cliq as sorted lists of node indices.
        """
        _, cliq = approximation.clique_removal(self.graph)
        self.cliq = [sorted(i) for i in cliq]  # Sort cliques for consistency

    def visualize_graph(
        self, node_size: int = 1230, font_size: int = 10, show_cliques: bool = True
    ) -> None:
        """
        Visualize the graph with colored edges based on clique membership.

        Args:
            node_size (int): Size of nodes in the visualization
            font_size (int): Size of font for node labels
            show_cliques (bool): If True, color edges by clique membership.
                               If False, show all edges in the graph.
        """

        # Generate random colors for each clique
        cliq_colors = [
            "#" + "".join([hex(random.randint(0, 255))[2:].zfill(2) for _ in range(3)])
            for _ in self.cliq
        ]

        # Build edge list and colors
        if show_cliques:
            el = []
            ec = []
            for e in self.graph.edges:
                for i, c in enumerate(self.cliq):
                    if (e[0] in c) and (e[1] in c):
                        el.append(e)
                        ec.append(cliq_colors[i])
                        break
        else:
            el = list(self.graph.edges)
            # Use gray for all edges when showing full graph
            ec = ["gray" for _ in el]

        # Create layout
        pos = nx.circular_layout(self.graph)

        # Draw the graph
        plt.figure(figsize=(10, 10))
        nx.draw(
            self.graph,
            pos=pos,
            with_labels=False,
            node_color="white",
            node_size=node_size,
            edgelist=el,
            edge_color=ec,
        )

        # Add labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=nx.get_node_attributes(self.graph, "label"),
            font_size=font_size,
            font_color="black",
            font_family="Times",
        )

        # Set edge colors
        ax = plt.gca()
        ax.collections[0].set_edgecolor("black")

        plt.show()

    def _update_graph_weights(self) -> None:
        """
        Update edge weights in the graph based on measurement statistics.

        Edge weights are computed as the product of:
        1. The coefficients of the connected terms
        2. Either:
           - The variance estimate (for self-edges)
           - The covariance estimate (for edges between different terms)
        """
        measurements = self.measurements

        for e in self.graph.edges:
            i, j = e[0], e[1]
            # Base weight from coefficients
            weight = self.coeffs[i] * self.coeffs[j]
            # Multiply by variance/covariance estimate
            weight *= (
                term_variance_estimate(i, measurements)
                if i == j
                else terms_covariance_estimate(i, j, measurements)
            )
            self.graph[i][j]["weight"] = weight

    def incremental_shot_allocation(self, num_shots: int) -> List[int]:
        """
        Propose allocation of measurement shots to cliques using greedy minimization of
        the estimated error.

        The allocation strategy minimizes the estimated total variance by:
        1. Calculating the contribution from each clique based on current shots
        2. Allocating shots one at a time to the clique with highest contribution
        3. Updating the estimates after each shot allocation

        This greedy approach ensures that each new shot is allocated to the clique
        that will provide the largest reduction in the total error estimate.

        Args:
            num_shots (int): Total number of new shots to propose allocating across cliques

        Returns:
            List[int]: Proposed number of new shots for each clique, where the index
                      corresponds to the clique index and the value is the number of
                      new shots proposed for that clique.

        Raises:
            ValueError: If num_shots is not a positive integer
            RuntimeError: If graph weights have not been properly initialized
        """
        if not isinstance(num_shots, int) or num_shots <= 0:
            raise ValueError("num_shots must be a positive integer")
        # Initialize array for proposed new shot allocation
        proposed_allocation = [0 for c in self.cliq]

        # Get current shots (or use 0 if None)
        current_shots = self.shots if self.shots else [0 for c in self.cliq]

        # Calculate weighted covariance matrix for error estimates
        weighted_covariance = nx.adjacency_matrix(self.graph, weight="weight").toarray()

        # Initialize error estimates for each clique using current shots
        # Error = sqrt(sum of covariances) / number of shots
        clique_error = [
            np.sqrt(weighted_covariance[c, c].sum()) / (current_shots[e] or not current_shots[e])
            for e, c in enumerate(self.cliq)
        ]

        # Allocate shots one at a time
        for _ in range(num_shots):
            # Find clique with highest error contribution
            cliq_id = max(range(len(self.cliq)), key=lambda x: clique_error[x])

            # Update proposed allocation for chosen clique
            proposed_allocation[cliq_id] += 1

            # Update error estimate for the chosen clique
            # New error = Old error * (n/(n+1)) where n+1 is the updated shot count.
            total_shots = current_shots[cliq_id] + proposed_allocation[cliq_id]
            clique_error[cliq_id] *= ((total_shots - 1) or not (total_shots - 1)) / (total_shots)

        return proposed_allocation

    def error_estimate(self) -> float:
        """
        Calculate the estimated standard error of the energy expectation value.

        The error is computed as sqrt(sum(variance/shots)) where:
        - variance comes from the weighted covariance matrix (product of coefficients
          and estimated variances/covariances)
        - shots is the number of measurements for each clique

        Returns:
            float: Estimated standard error based on current shot allocation
                  and measurement statistics.

        Raises:
            RuntimeError: If no shots have been allocated yet
            ValueError: If graph weights have not been properly initialized
        """
        # Get weighted covariance matrix
        weighted_covariance = nx.adjacency_matrix(self.graph, weight="weight").toarray()

        # Sum variance contributions from each clique
        variance_estimate = sum(
            [
                (weighted_covariance[c, c].sum()) / (self.shots[e] or not self.shots[e])
                for e, c in enumerate(self.cliq)
            ]
        )

        return np.sqrt(variance_estimate)

    def expectation_from_measurements(
        self, measurements: Union[MeasurementData, None] = None
    ) -> float:
        """
        Calculate the energy expectation value from measurement results
        for the different Pauli string observables.

        For each Pauli term, computes <P> = (N++ - N--)/N_total where:
        - N++ is the count of +1 measurements
        - N-- is the count of -1 measurements
        - N_total is the total number of measurements

        Args:
            measurements (Optional[List[List[Dict]]]): Measurement outcomes to use.
                If None, uses the instance's measurements.

        Returns:
            float: Estimated expectation value of the Hamiltonian

        Raises:
            AssertionError: If invalid measurement combinations are found
        """
        if not measurements:
            measurements = self.measurements

        expectation = 0.0
        for i in range(len(self.coeffs)):
            # Verify no invalid measurements
            assert measurements[i][i][(1, -1)] == 0, "Invalid measurement detected: (+1,-1)"
            assert measurements[i][i][(-1, 1)] == 0, "Invalid measurement detected: (-1,+1)"

            # Calculate expectation for this term
            term_shots = measurements[i][i][(1, 1)] + measurements[i][i][(-1, -1)]
            if term_shots:
                term_expect = (
                    measurements[i][i][(1, 1)] - measurements[i][i][(-1, -1)]
                ) / term_shots
                expectation += self.coeffs[i] * term_expect

        return expectation

    def _validate_measurements(self, measurements: MeasurementData) -> bool:
        """
        Validate measurement data for consistency and correctness.

        Performs several checks:
        1. Correct number of measurement records
        2. Consistency of measurements within cliques
        3. Absence of invalid measurement combinations

        Args:
            measurements (List[List[Dict]]): Measurement data to validate

        Returns:
            bool: True if all validations pass

        Raises:
            AssertionError: If any validation check fails
        """
        assert len(measurements) == self.num_terms, "Wrong number of measurement records"

        for c in self.cliq:
            # Get total shots for this clique
            m_cliq = measurements[c[0]][c[0]][(1, 1)] + measurements[c[0]][c[0]][(-1, -1)]

            # Check consistency within clique
            for i in c:
                for j in c:
                    # All measurements should be positive
                    for v in measurements[i][j].values():
                        assert v >= 0, "Measurement counts should not be negative"

                    # Measurements should be symmetric
                    assert measurements[i][j][(1, 1)] == measurements[j][i][(1, 1)], (
                        "Measurement should be symmetric"
                    )
                    assert measurements[i][j][(-1, -1)] == measurements[j][i][(-1, -1)], (
                        "Measurement should be symmetric"
                    )
                    assert measurements[i][j][(1, -1)] == measurements[j][i][(-1, 1)], (
                        "Measurement should be symmetric"
                    )
                    assert measurements[i][j][(-1, 1)] == measurements[j][i][(1, -1)], (
                        "Measurement should be symmetric"
                    )

                    # All pairs in clique should have same total measurements
                    assert m_cliq == sum(measurements[i][j].values()), (
                        f"The number of times {i} and {j} were measured together should be "
                        "equal to the number of measurements of their clique."
                    )

                    # Diagonal elements should not have invalid combinations
                    if i == j:
                        assert measurements[i][i][(1, -1)] == measurements[i][i][(-1, 1)] == 0, (
                            "A measurement of a single term can only contribute to the "
                            "(1,1) or (-1,-1) counts."
                        )
        return True

    def shots_from_measurements(self, measurements: MeasurementData) -> List[int]:
        """
        Extract the number of shots allocated to each clique from measurement data.

        Args:
            measurements (List[List[Dict]]): Measurement data to analyze

        Returns:
            List[int]: Number of shots for each clique

        Note:
            Validates measurements before extracting shot counts to ensure data consistency.
        """
        self._validate_measurements(measurements)
        return [sum(measurements[c[0]][c[0]].values()) for c in self.cliq]

    def update_measurements(self, new_measurements: MeasurementData) -> None:
        """
        Update the allocator with a new set of measurements.

        Args:
            new_measurements (List[List[Dict]]): New measurement outcomes to incorporate.
                Must follow the same structure as self.measurements:
                - List[List[Dict]] where each Dict contains measurement outcomes
                - Dict keys are tuples (±1, ±1) representing measurement results
                - Dict values are counts of those results
        """
        # Validate new measurements
        self._validate_measurements(new_measurements)

        # Update measurement counts
        for i in range(self.num_terms):
            for j in range(self.num_terms):
                for outcome, count in new_measurements[i][j].items():
                    self.measurements[i][j][outcome] += count

        # Update shots based on new measurements
        new_shots = self.shots_from_measurements(new_measurements)
        if not self.shots:
            self.shots = [0 for _ in self.cliq]
        for i, shots in enumerate(new_shots):
            self.shots[i] += shots

        # Update graph weights with new measurements
        self._update_graph_weights()
