"""
Plate Layout System for Strain Organization with Edge Alternation

This module provides a complete system for organizing biological strains across
multi-well plates with technical replicates, optimizing for edge effect mitigation
through alternating inner/edge positioning across replicates.

Key Components:
- Base set creation with evenly-distributed negative controls
- Replicate generation with randomization and edge alternation
- Comprehensive visualization suite for validation

Dependencies:
- numpy: Array operations and numerical computations
- matplotlib: Static plotting and heatmaps
- seaborn: Enhanced statistical visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import copy


@dataclass
class PlateConfig:
    """
    Configuration parameters for plate layout generation.

    Attributes:
        n_strains: Total number of unique biological strains (used if strain_names not provided)
        n_rows: Number of rows per plate (e.g., 8 for 96-well)
        n_columns: Number of columns per plate (e.g., 12 for 96-well)
        n_negatives: Number of empty/negative control wells per plate
        n_replicates: Number of technical replicates
        random_seed: Random seed for reproducible randomization (default: None for random)
        strain_names: Optional list of strain names/IDs (overrides n_strains if provided)
    """
    n_rows: int
    n_columns: int
    n_negatives: int
    n_replicates: int
    n_strains: Optional[int] = None
    random_seed: Optional[int] = None
    strain_names: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.strain_names is not None:
            if not isinstance(self.strain_names, list):
                raise TypeError("strain_names must be a list of strings")
            if len(self.strain_names) == 0:
                raise ValueError("strain_names list cannot be empty")
            if len(self.strain_names) != len(set(self.strain_names)):
                raise ValueError("strain_names must contain unique values (duplicates found)")
            # Override n_strains with length of strain_names
            self.n_strains = len(self.strain_names)
        elif self.n_strains is None:
            raise ValueError("Must provide either n_strains or strain_names")
        elif self.n_strains <= 0:
            raise ValueError("n_strains must be positive")


class PlateLayoutGenerator:
    """
    Generates optimized plate layouts with strain assignments and negative controls.

    This class handles the creation of a base plate set and generation of multiple
    replicates with randomization while maximizing edge alternation between
    consecutive replicates.
    """

    def __init__(self, config: PlateConfig):
        """
        Initialize generator with configuration parameters.

        Args:
            config: PlateConfig object with layout specifications
        """
        self.config = config
        self.usable_wells = (config.n_rows * config.n_columns) - config.n_negatives
        self.n_plates = int(np.ceil(config.n_strains / self.usable_wells))
        self.base_set = None
        self.replicates = []
        self.metadata = {
            'alternation_scores': [],
            'negative_positions': {},
            'random_seed': config.random_seed,
            'strain_names': config.strain_names
        }

        # Set global random seed if specified
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            self._base_seed = config.random_seed
        else:
            self._base_seed = None

    def _is_adjacent_to_negative(self, position: Tuple[int, int],
                                 negative_positions: List[Tuple[int, int]]) -> bool:
        """
        Check if a position is adjacent (including diagonally) to any negative position.

        Args:
            position: (row, col) tuple to check
            negative_positions: List of existing negative positions

        Returns:
            True if position is adjacent to any negative, False otherwise
        """
        row, col = position
        for neg_row, neg_col in negative_positions:
            # Check all 8 neighbors (cardinal + diagonal)
            if abs(row - neg_row) <= 1 and abs(col - neg_col) <= 1:
                if (row, col) != (neg_row, neg_col):  # Not the same position
                    return True
        return False

    def _is_on_outer_edge(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is on the outermost rows or columns of the plate.

        Args:
            position: (row, col) tuple to check

        Returns:
            True if position is on outer edge, False otherwise
        """
        row, col = position
        return (row == 0 or row == self.config.n_rows - 1 or
                col == 0 or col == self.config.n_columns - 1)

    def _generate_negative_positions(self, plate_id: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Generate evenly-distributed negative control positions with constraints.

        Constraints applied:
        1. Negatives must NOT be on outermost rows or columns (edge wells)
        2. Negatives must NOT be adjacent to each other (including diagonally)
        3. Negatives should be evenly distributed across the plate interior

        Uses a grid-based distribution with constraint checking. If constraints
        cannot be satisfied, raises an error with recommendations.

        Args:
            plate_id: Plate identifier for reproducible randomization
            seed: Random seed for reproducibility (if None, uses instance seed)

        Returns:
            List of (row, col) tuples for negative positions

        Raises:
            ValueError: If negative count is too high for constraints
        """
        if seed is not None:
            np.random.seed(seed + plate_id)
        elif self._base_seed is not None:
            np.random.seed(self._base_seed + plate_id + 1000)

        # Calculate interior dimensions (excluding outer edge)
        interior_rows = max(0, self.config.n_rows - 2)
        interior_cols = max(0, self.config.n_columns - 2)
        interior_wells = interior_rows * interior_cols

        # Check if constraints can be satisfied
        # With adjacency constraint, theoretical max is ~interior_wells/9 (each negative blocks 3x3 area)
        max_negatives_theoretical = interior_wells // 9
        if self.config.n_negatives > max_negatives_theoretical:
            raise ValueError(
                f"Too many negatives ({self.config.n_negatives}) for plate size with constraints. "
                f"Maximum recommended: {max_negatives_theoretical} for {self.config.n_rows}x{self.config.n_columns} plate. "
                f"Interior wells available: {interior_wells}"
            )

        # Generate candidate positions from interior only
        interior_positions = [
            (r, c) for r in range(1, self.config.n_rows - 1)
            for c in range(1, self.config.n_columns - 1)
        ]

        if len(interior_positions) < self.config.n_negatives:
            raise ValueError(
                f"Plate too small: only {len(interior_positions)} interior positions "
                f"available for {self.config.n_negatives} negatives"
            )

        # Try systematic spacing first
        positions = []
        spacing = int(np.sqrt(interior_wells / self.config.n_negatives))

        if spacing >= 2:  # Spacing of 2+ helps avoid adjacency
            # Generate grid points in interior
            for i in range(self.config.n_negatives * 3):  # Try more candidates
                if len(positions) >= self.config.n_negatives:
                    break

                # Map index to interior coordinates
                row = 1 + (i * spacing) % interior_rows
                col = 1 + ((i * spacing) // interior_rows) % interior_cols

                candidate = (row, col)

                # Check constraints
                if (not self._is_on_outer_edge(candidate) and
                    not self._is_adjacent_to_negative(candidate, positions)):
                    positions.append(candidate)

        # Fill remaining with random selection from valid positions
        attempts = 0
        max_attempts = interior_wells * 10

        while len(positions) < self.config.n_negatives and attempts < max_attempts:
            candidate = interior_positions[np.random.randint(0, len(interior_positions))]

            # Check all constraints
            if (candidate not in positions and
                not self._is_on_outer_edge(candidate) and
                not self._is_adjacent_to_negative(candidate, positions)):
                positions.append(candidate)

            attempts += 1

        if len(positions) < self.config.n_negatives:
            raise ValueError(
                f"Could not place {self.config.n_negatives} negatives with constraints after "
                f"{max_attempts} attempts. Successfully placed: {len(positions)}. "
                f"Try reducing n_negatives or increasing plate size."
            )

        return positions[:self.config.n_negatives]

    def create_base_set(self) -> np.ndarray:
        """
        Create base plate set with sequential strain assignments.

        Strains are assigned in row-major order (left-to-right, top-to-bottom)
        across plates. If strain_names were provided, uses those names; otherwise
        generates default names like "S0000", "S0001", etc.

        Negative controls are distributed using the constrained evenly-spaced
        algorithm (no edge placement, no adjacency).

        Returns:
            Array of shape (n_plates, n_rows, n_columns) with strain IDs,
            "negative" markers, or None for unassigned wells
        """
        plates = np.full((self.n_plates, self.config.n_rows, self.config.n_columns),
                        None, dtype=object)

        # Use provided strain names or generate default names
        if self.config.strain_names is not None:
            strain_list = self.config.strain_names
        else:
            strain_list = [f"S{i:04d}" for i in range(self.config.n_strains)]

        strain_idx = 0

        for plate_id in range(self.n_plates):
            # Get negative positions for this plate
            neg_positions = self._generate_negative_positions(plate_id, seed=42)
            neg_set = set(neg_positions)

            # Assign strains in row-major order
            for row in range(self.config.n_rows):
                for col in range(self.config.n_columns):
                    if (row, col) in neg_set:
                        plates[plate_id, row, col] = "negative"
                    elif strain_idx < self.config.n_strains:
                        plates[plate_id, row, col] = strain_list[strain_idx]
                        strain_idx += 1

        self.base_set = plates
        return plates

    def _count_occupied_neighbors(self, plate: np.ndarray, row: int, col: int) -> int:
        """
        Count occupied cardinal neighbors (N, S, E, W) for a given well.

        Negative controls and None values are considered unoccupied.

        Args:
            plate: Single plate array
            row: Well row index
            col: Well column index

        Returns:
            Count of occupied neighbors (0-4)
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.config.n_rows and 0 <= nc < self.config.n_columns:
                cell = plate[nr, nc]
                if cell is not None and cell != "negative":
                    neighbors.append(True)

        return len(neighbors)

    def _is_inner_position(self, plate: np.ndarray, row: int, col: int) -> bool:
        """
        Determine if a position is 'inner' (4 occupied neighbors).

        Args:
            plate: Single plate array
            row: Well row index
            col: Well column index

        Returns:
            True if position has 4 occupied neighbors, False otherwise
        """
        return self._count_occupied_neighbors(plate, row, col) == 4

    def _get_strain_positions(self, plate: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Map strain IDs to their (row, col) positions on a plate.

        Args:
            plate: Single plate array

        Returns:
            Dictionary mapping strain_id -> (row, col)
        """
        positions = {}
        for row in range(self.config.n_rows):
            for col in range(self.config.n_columns):
                cell = plate[row, col]
                if cell is not None and cell != "negative":
                    positions[cell] = (row, col)
        return positions

    def _randomize_plate(self, plate: np.ndarray, previous_plate: Optional[np.ndarray] = None,
                        replicate_id: int = 0) -> np.ndarray:
        """
        Randomize strain positions within a plate while maximizing edge alternation.

        If a previous replicate is provided, attempts to place strains that were
        'inner' in edge positions and vice versa. Uses a greedy algorithm with
        fallback for conflicts.

        Args:
            plate: Plate to randomize
            previous_plate: Optional previous replicate for alternation optimization
            replicate_id: Replicate identifier for seed offset

        Returns:
            Randomized plate array
        """
        # Set seed for this specific randomization
        if self._base_seed is not None:
            np.random.seed(self._base_seed + replicate_id * 10000)

        new_plate = np.full_like(plate, None)

        # Get strain positions from original
        strain_positions = self._get_strain_positions(plate)
        strains = list(strain_positions.keys())

        # Generate new negative positions
        if self._base_seed is not None:
            plate_id_seed = self._base_seed + replicate_id * 10000 + 5000
        else:
            plate_id_seed = np.random.randint(0, 10000)

        neg_positions = self._generate_negative_positions(replicate_id, seed=plate_id_seed)
        neg_set = set(neg_positions)

        # Mark negatives
        for row, col in neg_positions:
            new_plate[row, col] = "negative"

        # Get available positions
        available = [(r, c) for r in range(self.config.n_rows)
                    for c in range(self.config.n_columns)
                    if (r, c) not in neg_set]

        if previous_plate is not None:
            # Categorize strains by previous inner/edge status
            prev_positions = self._get_strain_positions(previous_plate)
            inner_strains = [s for s in strains if s in prev_positions and
                           self._is_inner_position(previous_plate, *prev_positions[s])]
            edge_strains = [s for s in strains if s not in inner_strains and s in prev_positions]

            # Shuffle for randomization within categories
            np.random.shuffle(inner_strains)
            np.random.shuffle(edge_strains)
            np.random.shuffle(available)

            # Create temporary plate to check neighbor counts
            temp_plate = new_plate.copy()

            # Try to place inner strains in edge positions first
            assigned = set()
            for strain in inner_strains:
                for pos in available:
                    if pos not in assigned:
                        temp_plate[pos] = strain
                        if not self._is_inner_position(temp_plate, *pos):
                            new_plate[pos] = strain
                            assigned.add(pos)
                            break
                        temp_plate[pos] = None

            # Place edge strains in remaining positions (prefer inner)
            for strain in edge_strains:
                for pos in available:
                    if pos not in assigned:
                        new_plate[pos] = strain
                        assigned.add(pos)
                        break

            # Handle any remaining strains
            remaining_strains = [s for s in strains if s not in
                               [new_plate[p] for p in available if new_plate[p] is not None]]
            for strain in remaining_strains:
                for pos in available:
                    if pos not in assigned:
                        new_plate[pos] = strain
                        assigned.add(pos)
                        break
        else:
            # Simple random assignment
            np.random.shuffle(strains)
            for strain, pos in zip(strains, available):
                new_plate[pos] = strain

        return new_plate

    def get_strain_location(self, strain_id: str, replicate_id: int = 0) -> Optional[Tuple[int, int, int]]:
        """
        Find the location of a specific strain in a given replicate.

        Args:
            strain_id: Strain identifier to search for
            replicate_id: Replicate index (default: 0 for base set)

        Returns:
            Tuple of (plate_id, row, col) if found, None otherwise
        """
        if replicate_id >= len(self.replicates):
            return None

        for plate_id in range(self.n_plates):
            plate = self.replicates[replicate_id][plate_id]
            for row in range(self.config.n_rows):
                for col in range(self.config.n_columns):
                    if plate[row, col] == strain_id:
                        return (plate_id, row, col)
        return None

    def get_all_strain_locations(self, strain_id: str) -> Dict[int, Tuple[int, int, int]]:
        """
        Get locations of a strain across all replicates.

        Args:
            strain_id: Strain identifier to search for

        Returns:
            Dictionary mapping replicate_id -> (plate_id, row, col)
        """
        locations = {}
        for rep_id in range(len(self.replicates)):
            loc = self.get_strain_location(strain_id, rep_id)
            if loc is not None:
                locations[rep_id] = loc
        return locations
        """
        Calculate edge alternation compliance between two replicates.
        
        Measures what fraction of strains that were inner in plate1 are edge in
        plate2, and vice versa.
        
        Args:
            plate1: First replicate plate
            plate2: Second replicate plate
            
        Returns:
            Alternation score between 0.0 and 1.0
        """
        pos1 = self._get_strain_positions(plate1)
        pos2 = self._get_strain_positions(plate2)

        common_strains = set(pos1.keys()) & set(pos2.keys())
        if not common_strains:
            return 0.0

        alternations = 0
        for strain in common_strains:
            r1, c1 = pos1[strain]
            r2, c2 = pos2[strain]

            is_inner1 = self._is_inner_position(plate1, r1, c1)
            is_inner2 = self._is_inner_position(plate2, r2, c2)

            # Alternation achieved if inner->edge or edge->inner
            if is_inner1 != is_inner2:
                alternations += 1

        return alternations / len(common_strains)

    def generate_replicates(self) -> Dict:
        """
        Generate all replicates with randomization and edge alternation optimization.

        First replicate uses the base set. Subsequent replicates randomize positions
        while attempting to maximize edge alternation from the previous replicate.

        Returns:
            Dictionary containing:
                - 'replicates': List of plate arrays (one per replicate)
                - 'metadata': Alternation scores and negative positions
        """
        if self.base_set is None:
            self.create_base_set()

        self.replicates = [self.base_set.copy()]
        self.metadata['negative_positions'][0] = {}

        # Record negative positions for replicate 0
        for plate_id in range(self.n_plates):
            neg_pos = []
            for row in range(self.config.n_rows):
                for col in range(self.config.n_columns):
                    if self.base_set[plate_id, row, col] == "negative":
                        neg_pos.append((row, col))
            self.metadata['negative_positions'][0][plate_id] = neg_pos

        # Generate additional replicates
        for rep_id in range(1, self.config.n_replicates):
            new_replicate = np.full_like(self.base_set, None)
            previous_replicate = self.replicates[-1]

            self.metadata['negative_positions'][rep_id] = {}

            # Randomize each plate
            for plate_id in range(self.n_plates):
                new_replicate[plate_id] = self._randomize_plate(
                    self.base_set[plate_id],
                    previous_replicate[plate_id],
                    replicate_id=rep_id
                )

                # Record negative positions
                neg_pos = []
                for row in range(self.config.n_rows):
                    for col in range(self.config.n_columns):
                        if new_replicate[plate_id, row, col] == "negative":
                            neg_pos.append((row, col))
                self.metadata['negative_positions'][rep_id][plate_id] = neg_pos

            self.replicates.append(new_replicate)

            # Calculate alternation score
            scores = []
            for plate_id in range(self.n_plates):
                score = self._calculate_alternation_score(
                    previous_replicate[plate_id],
                    new_replicate[plate_id]
                )
                scores.append(score)
            self.metadata['alternation_scores'].append(np.mean(scores))

        return {
            'replicates': self.replicates,
            'metadata': self.metadata
        }


class PlateVisualizer:
    """
    Visualization suite for plate layout validation and analysis.

    Provides multiple visualization types including heatmaps, edge alternation
    comparisons, replicate overviews, and statistical summaries.
    """

    def __init__(self, generator: PlateLayoutGenerator):
        """
        Initialize visualizer with a PlateLayoutGenerator instance.

        Args:
            generator: PlateLayoutGenerator with generated replicates
        """
        self.generator = generator
        self.config = generator.config

    def visualize_plate_layout(self, replicate_id: int, plate_id: int,
                               figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Display single plate with color-coded strain IDs and marked negatives.

        Args:
            replicate_id: Replicate index (0-based)
            plate_id: Plate index (0-based)
            figsize: Figure dimensions in inches

        Returns:
            Matplotlib figure and axes objects
        """
        plate = self.generator.replicates[replicate_id][plate_id]

        # Create numeric array for plotting
        plot_data = np.zeros((self.config.n_rows, self.config.n_columns))
        annot_data = np.full((self.config.n_rows, self.config.n_columns), "", dtype=object)

        strain_to_num = {}
        current_num = 1

        for row in range(self.config.n_rows):
            for col in range(self.config.n_columns):
                cell = plate[row, col]
                if cell == "negative":
                    plot_data[row, col] = -1
                    annot_data[row, col] = "NEG"
                elif cell is None:
                    plot_data[row, col] = 0
                    annot_data[row, col] = ""
                else:
                    if cell not in strain_to_num:
                        strain_to_num[cell] = current_num
                        current_num += 1
                    plot_data[row, col] = strain_to_num[cell]
                    annot_data[row, col] = cell

        fig, ax = plt.subplots(figsize=figsize)

        # Create custom colormap
        cmap = sns.color_palette("tab20", as_cmap=True)

        sns.heatmap(plot_data, annot=annot_data, fmt='', cmap=cmap,
                   linewidths=1, linecolor='black', ax=ax,
                   cbar_kws={'label': 'Strain ID'}, vmin=-1)

        # Labels
        row_labels = [chr(65 + i) for i in range(self.config.n_rows)]
        col_labels = [str(i + 1) for i in range(self.config.n_columns)]
        ax.set_yticklabels(row_labels, rotation=0)
        ax.set_xticklabels(col_labels, rotation=0)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(f'Replicate {replicate_id + 1} - Plate {plate_id + 1}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig, ax

    def visualize_edge_alternation(self, replicate_i: int, replicate_j: int,
                                   plate_id: int, figsize: Tuple[int, int] = (16, 7)) -> plt.Figure:
        """
        Compare edge positions between two replicates side-by-side.

        Highlights inner positions (red) in replicate_i and should-be-edge
        positions in replicate_j to validate alternation.

        Args:
            replicate_i: First replicate index
            replicate_j: Second replicate index
            plate_id: Plate index
            figsize: Figure dimensions

        Returns:
            Matplotlib figure object
        """
        plate_i = self.generator.replicates[replicate_i][plate_id]
        plate_j = self.generator.replicates[replicate_j][plate_id]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Calculate alternation score
        score = self.generator._calculate_alternation_score(plate_i, plate_j)

        for ax, plate, rep_id, title_suffix in [(ax1, plate_i, replicate_i, "Inner Highlighted"),
                                                  (ax2, plate_j, replicate_j, "Edge Highlighted")]:
            # Create base heatmap
            plot_data = np.zeros((self.config.n_rows, self.config.n_columns))
            annot_data = np.full((self.config.n_rows, self.config.n_columns), "", dtype=object)

            for row in range(self.config.n_rows):
                for col in range(self.config.n_columns):
                    cell = plate[row, col]
                    if cell == "negative":
                        plot_data[row, col] = -1
                        annot_data[row, col] = "N"
                    elif cell is not None:
                        annot_data[row, col] = cell[-4:]  # Last 4 chars

                        # Check if inner/edge
                        if ax == ax1:  # First replicate - highlight inner
                            if self.generator._is_inner_position(plate, row, col):
                                plot_data[row, col] = 2  # Inner
                            else:
                                plot_data[row, col] = 1  # Edge
                        else:  # Second replicate - highlight edge
                            if not self.generator._is_inner_position(plate, row, col):
                                plot_data[row, col] = 2  # Edge
                            else:
                                plot_data[row, col] = 1  # Inner

            cmap = sns.color_palette(["#f0f0f0", "#90EE90", "#FF6B6B"])
            sns.heatmap(plot_data, annot=annot_data, fmt='', cmap=cmap,
                       linewidths=1, linecolor='black', ax=ax,
                       cbar=False, vmin=-1, vmax=2)

            row_labels = [chr(65 + i) for i in range(self.config.n_rows)]
            col_labels = [str(i + 1) for i in range(self.config.n_columns)]
            ax.set_yticklabels(row_labels, rotation=0)
            ax.set_xticklabels(col_labels, rotation=0)
            ax.set_title(f'Rep {rep_id + 1} - {title_suffix}')

        fig.suptitle(f'Edge Alternation Comparison - Plate {plate_id + 1}\n'
                    f'Alternation Score: {score:.2%}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_all_replicates(self, strain_id: str,
                                figsize: Tuple[int, int] = (16, 4)) -> plt.Figure:
        """
        Show position changes for a single strain across all replicates.

        Args:
            strain_id: Strain identifier (e.g., "S0001")
            figsize: Figure dimensions per replicate

        Returns:
            Matplotlib figure object
        """
        n_reps = len(self.generator.replicates)
        fig, axes = plt.subplots(1, n_reps, figsize=(figsize[0], figsize[1]))

        if n_reps == 1:
            axes = [axes]

        for rep_id, ax in enumerate(axes):
            # Find strain in this replicate
            found = False
            for plate_id in range(self.generator.n_plates):
                plate = self.generator.replicates[rep_id][plate_id]

                # Create visualization
                plot_data = np.zeros((self.config.n_rows, self.config.n_columns))

                for row in range(self.config.n_rows):
                    for col in range(self.config.n_columns):
                        cell = plate[row, col]
                        if cell == strain_id:
                            is_inner = self.generator._is_inner_position(plate, row, col)
                            plot_data[row, col] = 3 if is_inner else 2
                            found = True
                        elif cell == "negative":
                            plot_data[row, col] = -1
                        elif cell is not None:
                            plot_data[row, col] = 1

                if found:
                    cmap = sns.color_palette(["white", "#E8E8E8", "#90EE90", "#FF6B6B"])
                    sns.heatmap(plot_data, cmap=cmap, linewidths=0.5,
                               linecolor='gray', ax=ax, cbar=False, vmin=-1, vmax=3)

                    row_labels = [chr(65 + i) for i in range(self.config.n_rows)]
                    col_labels = [str(i + 1) for i in range(self.config.n_columns)]
                    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
                    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)
                    ax.set_title(f'Rep {rep_id + 1}')
                    break

        fig.suptitle(f'Strain {strain_id} Position Across Replicates\n'
                    f'Red=Inner | Green=Edge', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_negative_distribution(self, figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Show negative control frequency heatmap across all replicates.

        Validates that negatives appear in different positions across replicates.

        Args:
            figsize: Figure dimensions

        Returns:
            Matplotlib figure object
        """
        n_plates = self.generator.n_plates
        n_cols_plot = min(3, n_plates)
        n_rows_plot = int(np.ceil(n_plates / n_cols_plot))

        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=figsize)
        if n_plates == 1:
            axes = np.array([[axes]])
        elif n_rows_plot == 1:
            axes = axes.reshape(1, -1)

        for plate_id in range(n_plates):
            row_idx = plate_id // n_cols_plot
            col_idx = plate_id % n_cols_plot
            ax = axes[row_idx, col_idx]

            # Count negative frequency
            frequency = np.zeros((self.config.n_rows, self.config.n_columns))

            for rep_id in range(len(self.generator.replicates)):
                plate = self.generator.replicates[rep_id][plate_id]
                for row in range(self.config.n_rows):
                    for col in range(self.config.n_columns):
                        if plate[row, col] == "negative":
                            frequency[row, col] += 1

            sns.heatmap(frequency, annot=True, fmt='.0f', cmap='YlOrRd',
                       linewidths=1, linecolor='black', ax=ax,
                       cbar_kws={'label': 'Frequency'})

            row_labels = [chr(65 + i) for i in range(self.config.n_rows)]
            col_labels = [str(i + 1) for i in range(self.config.n_columns)]
            ax.set_yticklabels(row_labels, rotation=0)
            ax.set_xticklabels(col_labels, rotation=0)
            ax.set_title(f'Plate {plate_id + 1}')

        # Hide unused subplots
        for idx in range(n_plates, n_rows_plot * n_cols_plot):
            row_idx = idx // n_cols_plot
            col_idx = idx % n_cols_plot
            axes[row_idx, col_idx].axis('off')

        fig.suptitle('Negative Control Position Frequency Across Replicates',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_summary(self, figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Generate comprehensive statistical dashboard with multiple panels.

        Includes:
        - Alternation scores across replicate pairs
        - Inner vs edge position counts
        - Neighbor count distribution
        - Strain distribution across plates

        Args:
            figsize: Figure dimensions

        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Alternation scores
        ax1 = fig.add_subplot(gs[0, 0])
        if self.generator.metadata['alternation_scores']:
            x = range(1, len(self.generator.metadata['alternation_scores']) + 1)
            ax1.plot(x, self.generator.metadata['alternation_scores'],
                    marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Replicate Pair (i to i+1)', fontsize=11)
            ax1.set_ylabel('Alternation Score', fontsize=11)
            ax1.set_title('Edge Alternation Performance', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])

        # Panel 2: Inner vs Edge counts
        ax2 = fig.add_subplot(gs[0, 1])
        inner_counts = []
        edge_counts = []

        for rep_id, replicate in enumerate(self.generator.replicates):
            inner = 0
            edge = 0
            for plate_id in range(self.generator.n_plates):
                plate = replicate[plate_id]
                for row in range(self.config.n_rows):
                    for col in range(self.config.n_columns):
                        cell = plate[row, col]
                        if cell is not None and cell != "negative":
                            if self.generator._is_inner_position(plate, row, col):
                                inner += 1
                            else:
                                edge += 1
            inner_counts.append(inner)
            edge_counts.append(edge)

        x = np.arange(len(self.generator.replicates))
        width = 0.35
        ax2.bar(x - width/2, inner_counts, width, label='Inner', color='#FF6B6B')
        ax2.bar(x + width/2, edge_counts, width, label='Edge', color='#90EE90')
        ax2.set_xlabel('Replicate', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Inner vs Edge Position Distribution', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'R{i+1}' for i in x])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Neighbor count distribution
        ax3 = fig.add_subplot(gs[1, 0])
        all_neighbor_counts = []

        for replicate in self.generator.replicates:
            for plate_id in range(self.generator.n_plates):
                plate = replicate[plate_id]
                for row in range(self.config.n_rows):
                    for col in range(self.config.n_columns):
                        cell = plate[row, col]
                        if cell is not None and cell != "negative":
                            count = self.generator._count_occupied_neighbors(plate, row, col)
                            all_neighbor_counts.append(count)

        ax3.hist(all_neighbor_counts, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                edgecolor='black', color='skyblue')
        ax3.set_xlabel('Number of Occupied Neighbors', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Neighbor Count Distribution', fontweight='bold')
        ax3.set_xticks([0, 1, 2, 3, 4])
        ax3.grid(True, alpha=0.3, axis='y')

        # Panel 4: Strain distribution across plates
        ax4 = fig.add_subplot(gs[1, 1])
        strains_per_plate = []

        for plate_id in range(self.generator.n_plates):
            count = 0
            plate = self.generator.base_set[plate_id]
            for row in range(self.config.n_rows):
                for col in range(self.config.n_columns):
                    cell = plate[row, col]
                    if cell is not None and cell != "negative":
                        count += 1
            strains_per_plate.append(count)

        ax4.bar(range(self.generator.n_plates), strains_per_plate,
               color='coral', edgecolor='black')
        ax4.set_xlabel('Plate ID', fontsize=11)
        ax4.set_ylabel('Number of Strains', fontsize=11)
        ax4.set_title('Strain Distribution Across Plates', fontweight='bold')
        ax4.set_xticks(range(self.generator.n_plates))
        ax4.set_xticklabels([f'P{i+1}' for i in range(self.generator.n_plates)])
        ax4.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Plate Layout System - Statistical Summary Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)

        return fig


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("PLATE LAYOUT GENERATION SYSTEM")
    print("=" * 60)
    print("\nDemonstration with two configuration modes:\n")

    # ===== MODE 1: Using n_strains (generates default strain IDs) =====
    print("-" * 60)
    print("MODE 1: Using n_strains parameter")
    print("-" * 60)

    config1 = PlateConfig(
        n_strains=180,
        n_rows=8,
        n_columns=12,
        n_negatives=6,
        n_replicates=3,
        random_seed=42
    )

    print(f"  Strains: {config1.n_strains} (auto-generated IDs)")
    print(f"  Plate format: {config1.n_rows} x {config1.n_columns}")
    print(f"  Negatives per plate: {config1.n_negatives}")
    print(f"  Replicates: {config1.n_replicates}")

    generator1 = PlateLayoutGenerator(config1)
    base_set1 = generator1.create_base_set()
    print(f"\n  ✓ Generated {generator1.n_plates} plates with strain IDs: S0000, S0001, ..., S{config1.n_strains-1:04d}")

    # ===== MODE 2: Using custom strain names =====
    print("\n" + "-" * 60)
    print("MODE 2: Using custom strain_names list")
    print("-" * 60)

    # Example: Create custom strain names
    custom_strains = [
        "WT-Control",
        "Mutant-A1", "Mutant-A2", "Mutant-A3",
        "Mutant-B1", "Mutant-B2", "Mutant-B3",
        "KO-Gene1", "KO-Gene2", "KO-Gene3",
        "OE-Protein1", "OE-Protein2",
    ]
    custom_strains += [f"Strain_{i:03d}" for i in range(13, 50)]  # Add more strains

    config2 = PlateConfig(
        strain_names=custom_strains,  # Use custom names instead of n_strains
        n_rows=8,
        n_columns=12,
        n_negatives=4,
        n_replicates=2,
        random_seed=123
    )

    print(f"  Strains: {config2.n_strains} (from custom list)")
    print(f"  First few strains: {custom_strains[:5]}")
    print(f"  Plate format: {config2.n_rows} x {config2.n_columns}")
    print(f"  Negatives per plate: {config2.n_negatives}")
    print(f"  Replicates: {config2.n_replicates}")

    generator2 = PlateLayoutGenerator(config2)
    base_set2 = generator2.create_base_set()
    results2 = generator2.generate_replicates()

    print(f"\n  ✓ Generated {generator2.n_plates} plates with custom strain names")
    print(f"  ✓ Generated {len(results2['replicates'])} replicates")

    # Demonstrate strain lookup
    print("\n  Strain Location Lookup Examples:")
    test_strain = custom_strains[0]
    locations = generator2.get_all_strain_locations(test_strain)
    for rep_id, (plate_id, row, col) in locations.items():
        row_label = chr(65 + row)
        col_label = col + 1
        print(f"    {test_strain} in Rep {rep_id+1}: Plate {plate_id+1}, Well {row_label}{col_label}")

    # ===== Continue with MODE 1 for full demonstration =====
    print("\n" + "=" * 60)
    print("Continuing full demonstration with MODE 1...")
    print("=" * 60)

    config = config1
    generator = generator1
    base_set = base_set1

    print("\n" + "-" * 60)
    print("STEP 2: Generating replicates with edge alternation...")
    print("-" * 60)

    results = generator.generate_replicates()

    print(f"✓ Generated {len(results['replicates'])} replicates")
    if results['metadata']['alternation_scores']:
        avg_score = np.mean(results['metadata']['alternation_scores'])
        print(f"✓ Average alternation score: {avg_score:.2%}")

    # Create visualizer
    print("\n" + "-" * 60)
    print("STEP 3: Creating visualizations...")
    print("-" * 60)

    viz = PlateVisualizer(generator)

    # Example visualizations
    print("\nGenerating example visualizations:")
    print("  1. Plate layout for Replicate 1, Plate 1")
    fig1, _ = viz.visualize_plate_layout(0, 0)
    plt.savefig('plate_layout_rep1_plate1.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: plate_layout_rep1_plate1.png")

    if config.n_replicates > 1:
        print("  2. Edge alternation comparison (Rep 1 vs Rep 2)")
        fig2 = viz.visualize_edge_alternation(0, 1, 0)
        plt.savefig('edge_alternation_comparison.png', dpi=150, bbox_inches='tight')
        print("     ✓ Saved: edge_alternation_comparison.png")

    print("  3. Negative distribution heatmap")
    fig3 = viz.visualize_negative_distribution()
    plt.savefig('negative_distribution.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: negative_distribution.png")

    print("  4. Statistical summary dashboard")
    fig4 = viz.visualize_summary()
    plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: summary_dashboard.png")

    # Example: Track specific strain
    print("  5. Strain S0042 across all replicates")
    fig5 = viz.visualize_all_replicates('S0042')
    plt.savefig('strain_S0042_replicates.png', dpi=150, bbox_inches='tight')
    print("     ✓ Saved: strain_S0042_replicates.png")

    # Verification checklist
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKLIST")
    print("=" * 60)

    # Check negative uniqueness
    all_unique = True
    for plate_id in range(generator.n_plates):
        positions_per_rep = []
        for rep_id in range(config.n_replicates):
            pos_set = set(results['metadata']['negative_positions'][rep_id][plate_id])
            positions_per_rep.append(pos_set)

        for i in range(len(positions_per_rep) - 1):
            if positions_per_rep[i] == positions_per_rep[i + 1]:
                all_unique = False
                break

    print(f"✓ Negative uniqueness: {'PASS' if all_unique else 'FAIL'}")

    # Check strain consistency
    strain_consistent = True
    for rep_id in range(config.n_replicates):
        for plate_id in range(generator.n_plates):
            strains_base = set()
            strains_rep = set()

            for row in range(config.n_rows):
                for col in range(config.n_columns):
                    cell_base = base_set[plate_id, row, col]
                    cell_rep = results['replicates'][rep_id][plate_id, row, col]

                    if cell_base is not None and cell_base != "negative":
                        strains_base.add(cell_base)
                    if cell_rep is not None and cell_rep != "negative":
                        strains_rep.add(cell_rep)

            if strains_base != strains_rep:
                strain_consistent = False
                break

    print(f"✓ Strain consistency: {'PASS' if strain_consistent else 'FAIL'}")

    # Check edge alternation
    if results['metadata']['alternation_scores']:
        avg_score = np.mean(results['metadata']['alternation_scores'])
        alternation_pass = avg_score > 0.3  # At least 30% alternation
        print(f"✓ Edge alternation (avg {avg_score:.2%}): {'PASS' if alternation_pass else 'FAIL'}")

    # Check randomization
    if config.n_replicates > 1:
        identical = True
        for plate_id in range(generator.n_plates):
            if not np.array_equal(results['replicates'][0][plate_id],
                                 results['replicates'][1][plate_id]):
                identical = False
                break
        randomization_pass = not identical
        print(f"✓ Randomization: {'PASS' if randomization_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print("\nAll visualizations saved. Review the images to validate the layout.")
    print(f"\nRandom seed used: {config.random_seed if config.random_seed is not None else 'None (random)'}")
    print("  - Use the same seed to reproduce identical layouts")
    print("  - Set random_seed=None for different layouts each run")
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("\n1. With numeric strain count (auto-generated IDs):")
    print("   config = PlateConfig(n_strains=100, n_rows=8, n_columns=12,")
    print("                        n_negatives=6, n_replicates=3)")
    print("\n2. With custom strain names:")
    print("   my_strains = ['WT-Control', 'Mutant-A', 'Mutant-B', ...]")
    print("   config = PlateConfig(strain_names=my_strains, n_rows=8,")
    print("                        n_columns=12, n_negatives=6, n_replicates=3)")
    print("\n3. Basic workflow:")
    print("   generator = PlateLayoutGenerator(config)")
    print("   generator.create_base_set()")
    print("   results = generator.generate_replicates()")
    print("\n4. Find strain locations:")
    print("   location = generator.get_strain_location('WT-Control', replicate_id=0)")
    print("   all_locations = generator.get_all_strain_locations('WT-Control')")
    print("\n5. Visualize:")
    print("   viz = PlateVisualizer(generator)")
    print("   viz.visualize_plate_layout(replicate_id=0, plate_id=0)")
    print("   viz.visualize_summary()")
    print("\nFor custom analysis, access:")
    print("  - generator.replicates[rep_id][plate_id] for plate arrays")
    print("  - generator.metadata for scores, positions, and strain names")
    print("  - generator.metadata['strain_names'] for the strain list used")

    plt.show()
