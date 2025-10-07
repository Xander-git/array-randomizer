"""Automated Strain-to-Well Assignment System (single-module)

Implements:
- Base set creation with negative placement (unique rows/cols, 8-connectivity separation)
- Replicate generation with per-plate negative uniqueness across replicates and 8-connectivity
- Per-plate strain randomization with edge/inner alternation optimization
- DataFrame export (long, wide)
- Visualization: single plate, negative uniqueness verification
- Validation suite per specification

Public API:
- generate_layout
- export_layout
- export_metadata
- plot_plate
- plot_negative_uniqueness
- validate_layout

Notes:
- Replicate indices are 0-based throughout the API.
- Row/column indices are 0-based; row labels A.., column labels 1..n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Rectangle


# =====================
# Visualization config
# =====================

VISUALIZATION_CONFIG: Dict[str, object] = {
    "figsize": (12, 8),
    "dpi": 100,
    "strain_name_truncate": 10,
    "colormap": "tab20",
    "negative_color": "#000000",
    "inner_border_color": "#FF0000",
    "edge_border_color": "#00FF00",
    "font_size": 8,
    "save_format": "png",
}


# =============
# Data classes
# =============

@dataclass(frozen=True)
class GridSize:
    n_rows: int
    n_cols: int


# ======================
# Internal util helpers
# ======================

def _set_random_seed(seed: Optional[int]) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def _make_generic_strains(n_strains: int) -> List[str]:
    return [f"strain_{i+1:03d}" for i in range(n_strains)]


def _plate_ids(n: int) -> List[str]:
    return [f"plate_{i}" for i in range(n)]


def _rc_to_labels(r: int, c: int) -> Tuple[str, str, str]:
    row_label = chr(ord("A") + r)
    col_label = str(c + 1)
    well_id = f"{row_label}{col_label}"
    return row_label, col_label, well_id


def _neighbors_cardinal(r: int, c: int, n_rows: int, n_cols: int) -> List[Tuple[int, int]]:
    neighbors: List[Tuple[int, int]] = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r < n_rows - 1:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c < n_cols - 1:
        neighbors.append((r, c + 1))
    return neighbors


def _neighbors_all8(r: int, c: int, n_rows: int, n_cols: int) -> List[Tuple[int, int]]:
    res: List[Tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                res.append((rr, cc))
    return res


def _is_8_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 or (
        abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 0
    ) or (abs(a[0] - b[0]) == 0 and abs(a[1] - b[1]) == 1)


def _count_neighbors(plate: np.ndarray, r: int, c: int) -> int:
    n_rows, n_cols = plate.shape
    count = 0
    for rr, cc in _neighbors_cardinal(r, c, n_rows, n_cols):
        val = plate[rr, cc]
        if val is not None and val != "negative":
            count += 1
    return count


def _is_inner(plate: np.ndarray, r: int, c: int) -> bool:
    return _count_neighbors(plate, r, c) == 4


def _classify_positions(
    n_rows: int,
    n_cols: int,
    negative_positions: set[Tuple[int, int]],
    empty_positions: set[Tuple[int, int]],
) -> Dict[Tuple[int, int], str]:
    """Classify non-negative, non-empty positions as 'inner' or 'edge'.

    The classification assumes all non-negative and non-empty positions are occupied
    (i.e., neighbors count uses this occupancy assumption).
    """
    occupied = np.ones((n_rows, n_cols), dtype=bool)
    for r, c in negative_positions:
        occupied[r, c] = False
    for r, c in empty_positions:
        occupied[r, c] = False

    position_type: Dict[Tuple[int, int], str] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in negative_positions or (r, c) in empty_positions:
                continue
            # Count cardinal occupied neighbors
            neighbor_count = 0
            for rr, cc in _neighbors_cardinal(r, c, n_rows, n_cols):
                if occupied[rr, cc]:
                    neighbor_count += 1
            position_type[(r, c)] = "inner" if neighbor_count == 4 else "edge"
    return position_type


def _allowed_rows_cols_for_negatives(
    n_rows: int,
    n_cols: int,
    num_strains_on_plate: int,
) -> Tuple[List[int], List[int]]:
    """Compute allowed rows/cols for negatives given constraints:
    - Never first/last row or column
    - Confine within boundary of strains on the plate (row-major boundary)

    For boundary: if strains span ≥1 full row (i.e., num_strains_on_plate ≥ n_cols),
    the column boundary is the full width; otherwise it ends at the last strain column.
    """
    if num_strains_on_plate <= 0:
        return [], []

    last_index = num_strains_on_plate - 1
    last_row_end = last_index // n_cols
    last_col_end = last_index % n_cols

    # Rows: exclude first (0) and last (n_rows-1), and restrict to <= last_row_end
    if last_row_end >= 1:
        max_allowed_row = min(n_rows - 2, last_row_end)
        allowed_rows = list(range(1, max_allowed_row + 1))
    else:
        allowed_rows = []  # boundary confines to row 0, but first row is disallowed

    # Cols: exclude first (0) and last (n_cols-1)
    if last_row_end >= 1:
        # at least one full row → full width interior allowed
        allowed_cols = list(range(1, max(1, n_cols - 1)))
        # range(1, n_cols-1); protect against n_cols<2 via max
        allowed_cols = [c for c in allowed_cols if c <= n_cols - 2]
    else:
        # only partial first row → bound by last_col_end
        if last_col_end >= 1:
            max_allowed_col = min(n_cols - 2, last_col_end)
            allowed_cols = list(range(1, max_allowed_col + 1))
        else:
            allowed_cols = []

    return allowed_rows, allowed_cols


def _sample_negative_positions(
    n_rows: int,
    n_cols: int,
    k: int,
    prior_sets: Sequence[set[Tuple[int, int]]],
    allowed_rows: Optional[Sequence[int]] = None,
    allowed_cols: Optional[Sequence[int]] = None,
    max_tries: int = 20000,
) -> set[Tuple[int, int]]:
    """Sample k negative positions with constraints:
    - unique rows and unique columns within the set
    - no 8-connectivity adjacency within the set
    - no reuse of exact positions from any prior set
    - no 8-connectivity adjacency to any position in any prior set
    """
    if k == 0:
        return set()

    rows_all = list(allowed_rows) if allowed_rows is not None else list(range(n_rows))
    cols_all = list(allowed_cols) if allowed_cols is not None else list(range(n_cols))

    # Precompute union of prior positions for quick checks
    prior_union: set[Tuple[int, int]] = set()
    for s in prior_sets:
        prior_union.update(s)

    prior_neighbors: set[Tuple[int, int]] = set()
    for r, c in prior_union:
        prior_neighbors.update(_neighbors_all8(r, c, n_rows, n_cols))
        prior_neighbors.add((r, c))

    tries = 0
    while tries < max_tries:
        tries += 1
        if k > min(n_rows, n_cols):
            # Impossible by rule; caller should pre-validate
            break
        if k > len(rows_all) or k > len(cols_all):
            break
        chosen_rows = set(random.sample(rows_all, k))
        chosen_cols = set(random.sample(cols_all, k))
        # Pair rows and cols via a random permutation
        rows_list = list(chosen_rows)
        cols_list = list(chosen_cols)
        random.shuffle(rows_list)
        random.shuffle(cols_list)
        positions = [(rows_list[i], cols_list[i]) for i in range(k)]

        # Unique rows & cols guaranteed by construction; now check adjacency
        valid = True
        pos_set = set(positions)

        # No reuse and no adjacency to prior
        for pos in pos_set:
            if pos in prior_union:
                valid = False
                break
            if pos in prior_neighbors:
                valid = False
                break

        if not valid:
            continue

        # Check within-set 8-connectivity adjacency
        for i in range(k):
            if not valid:
                break
            for j in range(i + 1, k):
                if _is_8_adjacent(positions[i], positions[j]):
                    valid = False
                    break

        if valid:
            return pos_set

    raise ValueError(
        "Unable to sample valid negative positions under 8-connectivity and uniqueness constraints. "
        "Consider reducing n_negatives or n_replicates."
    )


def _chunk_strains_by_plate(
    strains: List[str],
    n_rows: int,
    n_cols: int,
    n_negatives: int,
) -> List[List[str]]:
    """Split strain list into per-plate chunks based on usable capacity."""
    capacity = n_rows * n_cols - n_negatives
    if capacity <= 0:
        return []
    return [strains[i : i + capacity] for i in range(0, len(strains), capacity)]


def _row_major_positions(n_rows: int, n_cols: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _assign_strains_with_alternation(
    strains: List[str],
    preferred_type_by_strain: Dict[str, str],
    positions_by_type: Dict[str, List[Tuple[int, int]]],
) -> Dict[Tuple[int, int], str]:
    """Assign strains to available positions, maximizing alternation preference greedily.

    - Try to place strains into their preferred type first (random order)
    - If type capacity runs out, place into the other type
    - Any remaining unfilled positions (if any) remain empty
    """
    assignments: Dict[Tuple[int, int], str] = {}
    edge_positions = list(positions_by_type.get("edge", []))
    inner_positions = list(positions_by_type.get("inner", []))
    random.shuffle(edge_positions)
    random.shuffle(inner_positions)

    # Split strains by preferred type
    pref_edge: List[str] = []
    pref_inner: List[str] = []
    for s in strains:
        pref = preferred_type_by_strain.get(s, "edge")
        if pref == "inner":
            pref_inner.append(s)
        else:
            pref_edge.append(s)
    random.shuffle(pref_edge)
    random.shuffle(pref_inner)

    # Place pref-edge
    while pref_edge and edge_positions:
        pos = edge_positions.pop()
        s = pref_edge.pop()
        assignments[pos] = s

    # Place pref-inner
    while pref_inner and inner_positions:
        pos = inner_positions.pop()
        s = pref_inner.pop()
        assignments[pos] = s

    # Remaining strains: place anywhere available (first alternate, then remaining)
    remaining_positions = edge_positions + inner_positions
    random.shuffle(remaining_positions)
    remaining_strains = pref_edge + pref_inner
    for pos, s in zip(remaining_positions, remaining_strains):
        assignments[pos] = s

    return assignments


# ====================
# Core implementation
# ====================

def generate_layout(
    strain_names: Optional[List[str]] = None,
    n_strains: Optional[int] = None,
    n_rows: int = 8,
    n_columns: int = 12,
    n_negatives: int = 0,
    n_replicates: int = 1,
    random_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Generate complete layout with replicates.

    Returns a dict containing configuration, per-replicate plates, mappings, alternation scores
    and negative placement history.
    """
    if strain_names is None and n_strains is None:
        raise ValueError("Provide either strain_names or n_strains.")
    if strain_names is not None and n_strains is not None and len(strain_names) != n_strains:
        raise ValueError("If both provided, len(strain_names) must equal n_strains.")

    _set_random_seed(random_seed)

    if strain_names is None:
        assert n_strains is not None
        strains = _make_generic_strains(n_strains)
    else:
        strains = list(strain_names)

    n_strains_val = len(strains)
    if n_negatives > min(n_rows, n_columns):
        raise ValueError(f"Maximum negatives: {min(n_rows, n_columns)}")

    total_wells = n_rows * n_columns
    if n_strains_val > total_wells - n_negatives:
        # Per-plate capacity check per spec; multi-plate handled below, this prevents zero-capacity plates
        pass

    if n_negatives * n_replicates > n_rows * n_columns:
        raise ValueError("Cannot place unique negatives across all replicates")

    usable_wells = n_rows * n_columns - n_negatives
    if usable_wells <= 0:
        raise ValueError("No usable wells after accounting for negatives.")

    n_plates = math.ceil(n_strains_val / usable_wells)
    plate_ids = _plate_ids(n_plates)

    # Split strains into per-plate chunks (row-major fill)
    strains_by_plate = _chunk_strains_by_plate(strains, n_rows, n_columns, n_negatives)
    if len(strains_by_plate) != n_plates:
        # Should not happen, but guard anyway
        n_plates = len(strains_by_plate)
        plate_ids = _plate_ids(n_plates)

    # Build base replicate (replicate 0)
    base_plates: Dict[str, np.ndarray] = {pid: np.full((n_rows, n_columns), None, dtype=object) for pid in plate_ids}
    strain_plate_mapping: Dict[str, str] = {}

    # Negative history tracking per replicate and plate
    negative_history: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    used_negative_positions_by_plate: Dict[str, set[Tuple[int, int]]] = {pid: set() for pid in plate_ids}

    replicate_records: List[Dict[str, Dict[str, np.ndarray]]] = []

    # Base negative placement per plate
    rep0_negatives_per_plate: Dict[str, set[Tuple[int, int]]] = {}
    for pid, plate_strains in zip(plate_ids, strains_by_plate):
        prior_sets: List[set[Tuple[int, int]]] = []
        # Constrain negatives to interior (no first/last row/col) and to strains' boundary
        allowed_rows, allowed_cols = _allowed_rows_cols_for_negatives(
            n_rows, n_columns, len(plate_strains)
        )
        neg_set = _sample_negative_positions(
            n_rows,
            n_columns,
            n_negatives,
            prior_sets,
            allowed_rows=allowed_rows,
            allowed_cols=allowed_cols,
        )
        rep0_negatives_per_plate[pid] = neg_set
        used_negative_positions_by_plate[pid].update(neg_set)

        # Determine empty positions (if strains fewer than non-negative wells)
        non_negative_positions = [pos for pos in _row_major_positions(n_rows, n_columns) if pos not in neg_set]
        capacity = len(non_negative_positions)
        num_strains_here = len(plate_strains)
        if num_strains_here > capacity:
            raise ValueError("Insufficient wells for all strains on base plate")

        empty_positions: set[Tuple[int, int]] = set(non_negative_positions[num_strains_here:])

        # Classify positions for later reference (not strictly needed for base fill)
        # Fill plate row-major excluding negatives; leave empties as None
        fill_positions = non_negative_positions[:num_strains_here]
        for s, (r, c) in zip(plate_strains, fill_positions):
            base_plates[pid][r, c] = s
            strain_plate_mapping[s] = pid
        for r, c in neg_set:
            base_plates[pid][r, c] = "negative"

        # Any remaining are already None

    replicate_records.append({"plates": base_plates})
    negative_history[f"replicate_0"] = {pid: sorted(list(rep0_negatives_per_plate[pid])) for pid in plate_ids}

    # Subsequent replicates
    for rep_idx in range(1, n_replicates):
        rep_plates: Dict[str, np.ndarray] = {pid: np.full((n_rows, n_columns), None, dtype=object) for pid in plate_ids}
        prev_rep = replicate_records[-1]

        for pid, plate_strains in zip(plate_ids, strains_by_plate):
            # Sample negatives with uniqueness across replicates and 8-connectivity away from priors
            prior_sets = [used_negative_positions_by_plate[pid]]
            allowed_rows, allowed_cols = _allowed_rows_cols_for_negatives(
                n_rows, n_columns, len(plate_strains)
            )
            neg_set = _sample_negative_positions(
                n_rows,
                n_columns,
                n_negatives,
                prior_sets,
                allowed_rows=allowed_rows,
                allowed_cols=allowed_cols,
            )
            used_negative_positions_by_plate[pid].update(neg_set)

            # Determine non-negative positions for this replicate
            non_negative_positions = [pos for pos in _row_major_positions(n_rows, n_columns) if pos not in neg_set]
            num_strains_here = len(plate_strains)
            if num_strains_here > len(non_negative_positions):
                raise ValueError("Insufficient wells for strains after negative sampling")

            # Choose empty positions (stable policy: trailing positions in row-major order)
            empty_positions: set[Tuple[int, int]] = set(non_negative_positions[num_strains_here:])

            # Compute position type classification for current replicate
            pos_type_now = _classify_positions(n_rows, n_columns, neg_set, empty_positions)

            # For alternation preference, determine previous type of each strain
            # Build a temporary occupancy plate for prev replicate to compute prev types reliably
            prev_plate_arr = prev_rep["plates"][pid]
            # Classification of previous replicate must consider previous empties too. Infer empties as None positions.
            prev_neg_set: set[Tuple[int, int]] = set(
                [(r, c) for r in range(n_rows) for c in range(n_columns) if prev_plate_arr[r, c] == "negative"]
            )
            prev_empty_set: set[Tuple[int, int]] = set(
                [(r, c) for r in range(n_rows) for c in range(n_columns) if prev_plate_arr[r, c] is None]
            )
            prev_pos_type = _classify_positions(n_rows, n_columns, prev_neg_set, prev_empty_set)

            preferred_type_by_strain: Dict[str, str] = {}
            for s in plate_strains:
                # Locate strain in prev plate
                loc = np.argwhere(prev_plate_arr == s)
                if loc.size == 0:
                    # If not found (should not happen), default preference to edge
                    preferred_type_by_strain[s] = "edge"
                    continue
                r_prev, c_prev = int(loc[0][0]), int(loc[0][1])
                prev_type = prev_pos_type.get((r_prev, c_prev), "edge")
                preferred_type_by_strain[s] = "edge" if prev_type == "inner" else "inner"

            # Build available positions by type (exclude empties)
            positions_by_type: Dict[str, List[Tuple[int, int]]] = {"inner": [], "edge": []}
            for pos, t in pos_type_now.items():
                positions_by_type[t].append(pos)

            assignments = _assign_strains_with_alternation(plate_strains, preferred_type_by_strain, positions_by_type)

            # Materialize plate
            for pos, s in assignments.items():
                r, c = pos
                rep_plates[pid][r, c] = s
            for r, c in neg_set:
                rep_plates[pid][r, c] = "negative"
            # Leave empties as None

        replicate_records.append({"plates": rep_plates})
        negative_history[f"replicate_{rep_idx}"] = {
            pid: sorted(
                list(
                    set((r, c) for r in range(n_rows) for c in range(n_columns) if rep_plates[pid][r, c] == "negative")
                )
            )
            for pid in plate_ids
        }

    # Alternation scores between consecutive replicates
    alternation_scores: List[float] = []
    for rep_idx in range(1, n_replicates):
        score = _calculate_alternation_score_between(
            replicate_records[rep_idx - 1], replicate_records[rep_idx], plate_ids
        )
        alternation_scores.append(score)

    data: Dict[str, object] = {
        "config": {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "n_negatives": n_negatives,
            "n_replicates": n_replicates,
            "random_seed": random_seed,
            "usable_wells": usable_wells,
            "n_strains": n_strains_val,
            "n_plates": n_plates,
        },
        "replicates": replicate_records,
        "strain_plate_mapping": strain_plate_mapping,
        "negative_history": negative_history,
        "alternation_scores": alternation_scores,
        "random_seed": random_seed,
    }
    return data


def _calculate_alternation_score_between(
    rep_i: Dict[str, Dict[str, np.ndarray]],
    rep_j: Dict[str, Dict[str, np.ndarray]],
    plate_ids: List[str],
) -> float:
    # Gather all strains and check their type edge/inner across plates
    total = 0
    flipped = 0
    for pid in plate_ids:
        plate_i = rep_i["plates"][pid]
        plate_j = rep_j["plates"][pid]

        # Compute classification maps based on negatives and empties of each replicate
        n_rows, n_cols = plate_i.shape
        neg_i = {(r, c) for r in range(n_rows) for c in range(n_cols) if plate_i[r, c] == "negative"}
        emp_i = {(r, c) for r in range(n_rows) for c in range(n_cols) if plate_i[r, c] is None}
        pos_type_i = _classify_positions(n_rows, n_cols, neg_i, emp_i)

        neg_j = {(r, c) for r in range(n_rows) for c in range(n_cols) if plate_j[r, c] == "negative"}
        emp_j = {(r, c) for r in range(n_rows) for c in range(n_cols) if plate_j[r, c] is None}
        pos_type_j = _classify_positions(n_rows, n_cols, neg_j, emp_j)

        # For each strain occurring on this plate
        strains_i = {plate_i[r, c] for r in range(n_rows) for c in range(n_cols) if plate_i[r, c] not in (None, "negative")}
        for s in strains_i:
            loc_i = np.argwhere(plate_i == s)
            loc_j = np.argwhere(plate_j == s)
            if loc_i.size == 0 or loc_j.size == 0:
                continue
            ri, ci = int(loc_i[0][0]), int(loc_i[0][1])
            rj, cj = int(loc_j[0][0]), int(loc_j[0][1])
            ti = pos_type_i.get((ri, ci), "edge")
            tj = pos_type_j.get((rj, cj), "edge")
            total += 1
            if ti != tj:
                flipped += 1

    return flipped / total if total > 0 else 0.0


# ===============
# DataFrame export
# ===============

def export_layout(
    data: Dict[str, object],
    format: str = "long",
    replicate_id: Optional[int] = None,
    plate_id: Optional[str] = None,
) -> pd.DataFrame:
    """Export layout to DataFrame.

    - format='long': one row per well with metadata
    - format='wide': pivot with columns A1.. for each (replicate_id, plate_id)
    """
    n_rows = int(data["config"]["n_rows"])  # type: ignore[index]
    n_cols = int(data["config"]["n_columns"])  # type: ignore[index]
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]

    rep_indices = list(range(len(replicates))) if replicate_id is None else [replicate_id]
    plate_ids = list(replicates[0]["plates"].keys())  # type: ignore[index]
    if plate_id is not None:
        plate_ids = [plate_id]

    records: List[Dict[str, object]] = []
    for rep_idx in rep_indices:
        plates = replicates[rep_idx]["plates"]
        for pid in plate_ids:
            arr = plates[pid]
            # For neighbor counts and position types, use the array directly
            for r in range(n_rows):
                for c in range(n_cols):
                    row_label, col_label, well_id = _rc_to_labels(r, c)
                    content = arr[r, c]
                    neighbor_count = _count_neighbors(arr, r, c)
                    if content == "negative":
                        position_type = "negative"
                        is_inner = False
                    elif content is None:
                        position_type = "empty"
                        is_inner = False
                    else:
                        is_inner = _is_inner(arr, r, c)
                        position_type = "inner" if is_inner else "edge"

                    records.append(
                        {
                            "replicate_id": rep_idx,
                            "plate_id": pid,
                            "row_index": r,
                            "col_index": c,
                            "row_label": row_label,
                            "col_label": col_label,
                            "well_id": well_id,
                            "content": content,
                            "is_inner": is_inner,
                            "neighbor_count": neighbor_count,
                            "position_type": position_type,
                        }
                    )

    df = pd.DataFrame.from_records(records)
    if format == "long":
        return df

    if format == "wide":
        # Pivot to (replicate_id, plate_id) index with well_id columns
        pivot = df.pivot_table(
            index=["replicate_id", "plate_id"],
            columns="well_id",
            values="content",
            aggfunc="first",
        )
        # Ensure full well_id ordering
        well_ids = [
            _rc_to_labels(r, c)[2] for r in range(n_rows) for c in range(n_cols)
        ]
        pivot = pivot.reindex(columns=well_ids)
        pivot = pivot.reset_index()
        return pivot

    raise ValueError("format must be 'long' or 'wide'")


def export_metadata(data: Dict[str, object]) -> Dict[str, object]:
    return {
        "random_seed": data.get("random_seed"),
        "n_strains": data["config"]["n_strains"],  # type: ignore[index]
        "n_plates": data["config"]["n_plates"],  # type: ignore[index]
        "alternation_scores": data.get("alternation_scores", []),
        "strain_plate_mapping": data.get("strain_plate_mapping", {}),
        "negative_position_history": data.get("negative_history", {}),
    }


# ===============
# Visualizations
# ===============

def _color_map_for_strains(strains: Sequence[str]) -> Dict[str, str]:
    cmap = plt.get_cmap(str(VISUALIZATION_CONFIG.get("colormap", "tab20")))
    n = max(1, len(strains))
    colors = [to_hex(cmap(i % cmap.N)) for i in range(n)]
    return {s: colors[i % len(colors)] for i, s in enumerate(strains)}


def plot_plate(
    data: Dict[str, object],
    replicate_id: int,
    plate_id: str,
    annotate: bool = True,
    highlight_inner: bool = True,
    highlight_negatives: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    n_rows = int(data["config"]["n_rows"])  # type: ignore[index]
    n_cols = int(data["config"]["n_columns"])  # type: ignore[index]
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]

    arr = replicates[replicate_id]["plates"][plate_id]

    # Collect strains for coloring
    strains = sorted({
        arr[r, c] for r in range(n_rows) for c in range(n_cols) if arr[r, c] not in (None, "negative")
    })
    color_map = _color_map_for_strains(strains)
    neg_color = str(VISUALIZATION_CONFIG["negative_color"])
    inner_border_color = str(VISUALIZATION_CONFIG["inner_border_color"])
    edge_border_color = str(VISUALIZATION_CONFIG["edge_border_color"])

    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"], dpi=VISUALIZATION_CONFIG["dpi"])  # type: ignore[arg-type]
    ax.set_aspect("equal")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0.5, n_cols + 0.5))
    ax.set_yticks(np.arange(0.5, n_rows + 0.5))
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)])
    ax.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    ax.grid(True, which="both", color="#cccccc", linewidth=0.5)
    ax.set_title(f"Replicate {replicate_id} - {plate_id}")

    # Draw cells
    for r in range(n_rows):
        for c in range(n_cols):
            val = arr[r, c]
            x0, y0 = c, r
            if val == "negative":
                facecolor = neg_color
            elif val is None:
                facecolor = "#ffffff"
            else:
                facecolor = color_map.get(str(val), "#cccccc")

            rect = Rectangle((x0, y0), 1, 1, facecolor=facecolor, edgecolor="#333333")
            ax.add_patch(rect)

            if annotate and val not in (None, "negative"):
                text = str(val)
                max_len = int(VISUALIZATION_CONFIG["strain_name_truncate"])  # type: ignore[arg-type]
                if len(text) > max_len:
                    text = text[:max_len] + "…"
                ax.text(x0 + 0.5, y0 + 0.55, text, ha="center", va="center", fontsize=VISUALIZATION_CONFIG["font_size"])  # type: ignore[arg-type]

            if highlight_inner or highlight_negatives:
                if val == "negative" and highlight_negatives:
                    rect.set_linewidth(1.5)
                    rect.set_edgecolor("#000000")
                elif val not in (None, "negative") and highlight_inner:
                    # Determine inner vs edge based on current occupancy
                    if _is_inner(arr, r, c):
                        rect.set_linewidth(2.0)
                        rect.set_edgecolor(inner_border_color)
                    else:
                        rect.set_linewidth(2.0)
                        rect.set_edgecolor(edge_border_color)

    # Legend
    handles = []
    labels = []
    # Optional: limit legend size for many strains
    max_legend_items = 20
    for i, s in enumerate(strains[:max_legend_items]):
        patch = Rectangle((0, 0), 1, 1, facecolor=color_map[s])
        handles.append(patch)
        labels.append(s)
    if len(strains) > max_legend_items:
        # Add overflow indicator
        patch = Rectangle((0, 0), 1, 1, facecolor="#cccccc")
        handles.append(patch)
        labels.append("…")
    handles.append(Rectangle((0, 0), 1, 1, facecolor=neg_color))
    labels.append("negative")
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    return fig, ax


def plot_negative_uniqueness(data: Dict[str, object]) -> plt.Figure:
    n_rows = int(data["config"]["n_rows"])  # type: ignore[index]
    n_cols = int(data["config"]["n_columns"])  # type: ignore[index]
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]
    plate_ids: List[str] = list(replicates[0]["plates"].keys())  # type: ignore[index]

    # Compute counts per plate across replicates
    counts_by_plate: Dict[str, np.ndarray] = {}
    rows_used: Dict[int, List[int]] = {}
    cols_used: Dict[int, List[int]] = {}
    for ridx, rep in enumerate(replicates):
        rows_used[ridx] = []
        cols_used[ridx] = []
        for pid in plate_ids:
            arr = rep["plates"][pid]
            if pid not in counts_by_plate:
                counts_by_plate[pid] = np.zeros((n_rows, n_cols), dtype=int)
            neg_positions = [(r, c) for r in range(n_rows) for c in range(n_cols) if arr[r, c] == "negative"]
            for r, c in neg_positions:
                counts_by_plate[pid][r, c] += 1
                rows_used[ridx].append(r)
                cols_used[ridx].append(c)

    # Aggregate for first plate for visualization simplicity
    first_plate = plate_ids[0]
    freq_heatmap = counts_by_plate[first_plate]

    # Panel B matrices: show row/column usage per replicate
    max_rep = len(replicates)
    row_usage = np.zeros((max_rep, n_rows), dtype=int)
    col_usage = np.zeros((max_rep, n_cols), dtype=int)
    for ridx in range(max_rep):
        for r in rows_used[ridx]:
            row_usage[ridx, r] += 1
        for c in cols_used[ridx]:
            col_usage[ridx, c] += 1

    # Panel C: scatter all negatives for first plate
    scatter_points_by_rep: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(max_rep)}
    for ridx, rep in enumerate(replicates):
        arr = rep["plates"][first_plate]
        for r in range(n_rows):
            for c in range(n_cols):
                if arr[r, c] == "negative":
                    scatter_points_by_rep[ridx].append((r, c))

    fig = plt.figure(figsize=VISUALIZATION_CONFIG["figsize"], dpi=VISUALIZATION_CONFIG["dpi"])  # type: ignore[arg-type]
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.2])

    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(freq_heatmap, cmap="Blues", interpolation="nearest")
    axA.set_title(f"Panel A: Negative usage (counts) - {first_plate}")
    axA.set_xticks(range(n_cols))
    axA.set_yticks(range(n_rows))
    axA.set_xticklabels([str(i + 1) for i in range(n_cols)])
    axA.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)

    axB1 = fig.add_subplot(gs[0, 1])
    axB1.imshow(row_usage, cmap="Purples", aspect="auto")
    axB1.set_title("Panel B (rows): Row usage per replicate")
    axB1.set_xlabel("Row index")
    axB1.set_ylabel("Replicate")

    axB2 = fig.add_subplot(gs[1, 1])
    axB2.imshow(col_usage, cmap="Greens", aspect="auto")
    axB2.set_title("Panel B (cols): Column usage per replicate")
    axB2.set_xlabel("Column index")
    axB2.set_ylabel("Replicate")

    axC = fig.add_subplot(gs[1, 0])
    colors_cycle = plt.get_cmap("tab10")
    for ridx, pts in scatter_points_by_rep.items():
        if not pts:
            continue
        ys = [p[0] + 0.5 for p in pts]
        xs = [p[1] + 0.5 for p in pts]
        axC.scatter(xs, ys, color=colors_cycle(ridx % 10), label=f"rep {ridx}", s=40, edgecolors="k")
    axC.set_xlim(0, n_cols)
    axC.set_ylim(0, n_rows)
    axC.invert_yaxis()
    axC.set_xticks(range(n_cols))
    axC.set_yticks(range(n_rows))
    axC.set_xticklabels([str(i + 1) for i in range(n_cols)])
    axC.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    axC.grid(True, which="both", color="#cccccc", linewidth=0.5)
    axC.set_title(f"Panel C: Spatial distribution - {first_plate}")
    axC.legend(loc="upper right")

    plt.tight_layout()
    return fig


# ======================
# Advanced viz (stubs)
# ======================

def plot_alternation_pairs(
    data: Dict[str, object],
    replicate_i: int,
    replicate_j: int,
    plate_id: Optional[str] = None,
) -> plt.Figure:
    """Stub: Alternation comparison between two replicates.

    Displays a summary figure with overall alternation score between the two replicates.
    """
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]
    plate_ids: List[str] = list(replicates[0]["plates"].keys())  # type: ignore[index]
    if plate_id is not None:
        plate_ids = [plate_id]
    score = _calculate_alternation_score_between(replicates[replicate_i], replicates[replicate_j], plate_ids)

    fig = plt.figure(figsize=VISUALIZATION_CONFIG["figsize"], dpi=VISUALIZATION_CONFIG["dpi"])  # type: ignore[arg-type]
    ax = fig.add_subplot(111)
    ax.axis("off")
    txt = (
        f"Alternation comparison (rep {replicate_i} → rep {replicate_j})\n"
        f"Plates: {', '.join(plate_ids)}\n"
        f"Alternation score: {score:.3f}\n\n"
        "(Detailed side-by-side visualization is deferred in this stub.)"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left")
    fig.tight_layout()
    return fig


def plot_strain_journey(data: Dict[str, object], strain_name: str) -> plt.Figure:
    """Stub: Strain trajectory across replicates.

    Shows a text summary of the strain's positions across replicates and plate consistency.
    """
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]
    n_rows = int(data["config"]["n_rows"])  # type: ignore[index]
    n_cols = int(data["config"]["n_columns"])  # type: ignore[index]
    plate_ids: List[str] = list(replicates[0]["plates"].keys())  # type: ignore[index]

    lines: List[str] = []
    plates_seen: set[str] = set()
    for ridx, rep in enumerate(replicates):
        found = False
        for pid in plate_ids:
            arr = rep["plates"][pid]
            loc = np.argwhere(arr == strain_name)
            if loc.size:
                r, c = int(loc[0][0]), int(loc[0][1])
                row_label, col_label, well_id = _rc_to_labels(r, c)
                lines.append(f"Rep {ridx} | {pid} | {well_id} ({row_label},{col_label})")
                plates_seen.add(pid)
                found = True
                break
        if not found:
            lines.append(f"Rep {ridx} | NOT FOUND")

    plate_consistent = len(plates_seen) == 1
    fig = plt.figure(figsize=VISUALIZATION_CONFIG["figsize"], dpi=VISUALIZATION_CONFIG["dpi"])  # type: ignore[arg-type]
    ax = fig.add_subplot(111)
    ax.axis("off")
    txt = (
        f"Strain journey: {strain_name}\n"
        + "\n".join(lines)
        + f"\n\nPlate consistent: {'Yes' if plate_consistent else 'No'}\n"
        "(Visual grid panels deferred in this stub.)"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left")
    fig.tight_layout()
    return fig


def plot_verification_dashboard(data: Dict[str, object]) -> plt.Figure:
    """Stub: Compact dashboard summarizing validation metrics and alternation trend."""
    validations = validate_layout(data)
    scores: List[float] = data.get("alternation_scores", [])  # type: ignore[assignment]

    fig, axes = plt.subplots(2, 3, figsize=VISUALIZATION_CONFIG["figsize"], dpi=VISUALIZATION_CONFIG["dpi"])  # type: ignore[arg-type]
    for ax in axes.flat:
        ax.axis("off")

    # Panel 1: Alternation score trend (textual)
    axes[0, 0].set_title("Alternation Score Trend")
    axes[0, 0].text(0.02, 0.98, "Scores: " + ", ".join(f"{s:.2f}" for s in scores), va="top", ha="left")

    # Panel 2: Position Type Distribution (textual placeholder)
    axes[0, 1].set_title("Position Type Distribution")
    axes[0, 1].text(0.02, 0.98, "(Distribution plot deferred)", va="top", ha="left")

    # Panel 3: Neighbor Count Histogram (textual placeholder)
    axes[0, 2].set_title("Neighbor Count Histogram")
    axes[0, 2].text(0.02, 0.98, "(Histogram deferred)", va="top", ha="left")

    # Panel 4: Strain-Plate Assignment (textual)
    axes[1, 0].set_title("Strain-Plate Assignment")
    spm = data.get("strain_plate_mapping", {})
    axes[1, 0].text(0.02, 0.98, f"Plates: {len(set(spm.values()))}", va="top", ha="left")

    # Panel 5: Negative Spacing Matrix (textual)
    axes[1, 1].set_title("Negative Spacing Matrix")
    axes[1, 1].text(0.02, 0.98, "(Heatmap deferred)", va="top", ha="left")

    # Panel 6: Randomization Entropy (textual)
    axes[1, 2].set_title("Randomization Entropy")
    axes[1, 2].text(0.02, 0.98, "(Entropy metric deferred)", va="top", ha="left")

    # Footer: validations
    fig.suptitle("Verification Dashboard (stub)")
    ok_lines = "\n".join(f"{k}: {'✓' if v else '✗'}" for k, v in validations.items())
    fig.text(0.01, 0.01, ok_lines, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# ============
# Validation
# ============

def validate_layout(data: Dict[str, object]) -> Dict[str, bool]:
    n_rows = int(data["config"]["n_rows"])  # type: ignore[index]
    n_cols = int(data["config"]["n_columns"])  # type: ignore[index]
    n_reps = int(data["config"]["n_replicates"])  # type: ignore[index]
    n_plates = int(data["config"]["n_plates"])  # type: ignore[index]
    n_strains_val = int(data["config"]["n_strains"])  # type: ignore[index]
    replicates: List[Dict[str, Dict[str, np.ndarray]]] = data["replicates"]  # type: ignore[assignment]
    plate_ids: List[str] = list(replicates[0]["plates"].keys())  # type: ignore[index]

    # 1) negative_uniqueness: no position reused across replicates PER PLATE
    negative_uniqueness_ok = True
    # also ensure 8-connectivity independence across replicates
    cross_rep_spacing_ok = True
    for pid in plate_ids:
        seen: set[Tuple[int, int]] = set()
        prior: set[Tuple[int, int]] = set()
        for rep in replicates:
            arr = rep["plates"][pid]
            negs = {(r, c) for r in range(n_rows) for c in range(n_cols) if arr[r, c] == "negative"}
            # uniqueness
            if any(p in seen for p in negs):
                negative_uniqueness_ok = False
            seen.update(negs)
            # spacing vs prior
            for p in negs:
                if p in prior:
                    cross_rep_spacing_ok = False
                    break
                for q in _neighbors_all8(p[0], p[1], n_rows, n_cols):
                    if q in prior:
                        cross_rep_spacing_ok = False
                        break
            prior.update(negs)

    # 2) negative_spacing: within replicate, no adjacent negatives
    negative_spacing_ok = True
    for rep in replicates:
        for pid in plate_ids:
            arr = rep["plates"][pid]
            negs = [(r, c) for r in range(n_rows) for c in range(n_cols) if arr[r, c] == "negative"]
            for i in range(len(negs)):
                for j in range(i + 1, len(negs)):
                    if _is_8_adjacent(negs[i], negs[j]):
                        negative_spacing_ok = False
                        break

    # 3) negative_row_col_unique: within replicate, unique rows and unique cols
    negative_row_col_unique_ok = True
    for rep in replicates:
        for pid in plate_ids:
            arr = rep["plates"][pid]
            negs = [(r, c) for r in range(n_rows) for c in range(n_cols) if arr[r, c] == "negative"]
            rows = {r for r, _ in negs}
            cols = {c for _, c in negs}
            if len(rows) != len(negs) or len(cols) != len(negs):
                negative_row_col_unique_ok = False

    # 4) strain_plate_consistency: same strain on same plate across replicates
    strain_plate_consistency_ok = True
    strain_plate_mapping: Dict[str, str] = data.get("strain_plate_mapping", {})  # type: ignore[assignment]
    for rep in replicates:
        for pid in plate_ids:
            arr = rep["plates"][pid]
            for r in range(n_rows):
                for c in range(n_cols):
                    val = arr[r, c]
                    if val not in (None, "negative"):
                        if strain_plate_mapping.get(str(val)) != pid:
                            strain_plate_consistency_ok = False

    # 5) strain_name_integrity: all input names present in outputs (per replicate)
    strain_name_integrity_ok = True
    input_strains: set[str] = set(strain_plate_mapping.keys())
    for rep in replicates:
        observed: set[str] = set()
        for pid in plate_ids:
            arr = rep["plates"][pid]
            for r in range(n_rows):
                for c in range(n_cols):
                    val = arr[r, c]
                    if val not in (None, "negative"):
                        observed.add(str(val))
        if observed != input_strains:
            strain_name_integrity_ok = False

    # 6) alternation_feasibility: score > 0.70 for valid configurations
    alternation_scores: List[float] = data.get("alternation_scores", [])  # type: ignore[assignment]
    alternation_feasibility_ok = True
    if alternation_scores:
        alternation_feasibility_ok = min(alternation_scores) > 0.70

    # 7) array_completeness: number of non-negative, non-None equals n_strains per replicate
    array_completeness_ok = True
    for rep in replicates:
        count = 0
        for pid in plate_ids:
            arr = rep["plates"][pid]
            count += sum(
                1
                for r in range(n_rows)
                for c in range(n_cols)
                if arr[r, c] not in (None, "negative")
            )
        if count != n_strains_val:
            array_completeness_ok = False

    # 8) export_consistency: DataFrame row count matches layout
    df_all = export_layout(data, format="long")
    expected_rows = n_reps * n_plates * n_rows * n_cols
    export_consistency_ok = len(df_all) == expected_rows

    return {
        "negative_uniqueness": negative_uniqueness_ok,
        "negative_spacing": negative_spacing_ok,
        "negative_row_col_unique": negative_row_col_unique_ok,
        "strain_plate_consistency": strain_plate_consistency_ok,
        "strain_name_integrity": strain_name_integrity_ok,
        "alternation_feasibility": alternation_feasibility_ok,
        "array_completeness": array_completeness_ok,
        "export_consistency": export_consistency_ok,
    }


# =====================
# Module __all__ export
# =====================

__all__ = [
    "generate_layout",
    "export_layout",
    "export_metadata",
    "plot_plate",
    "plot_negative_uniqueness",
    "plot_alternation_pairs",
    "plot_strain_journey",
    "plot_verification_dashboard",
    "validate_layout",
    "VISUALIZATION_CONFIG",
]


