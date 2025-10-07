<!-- cf1c43c4-5b58-4f05-9f08-87da141cd6ad 8ace9241-27e7-452d-b09b-d2a41bb81735 -->
# Automated Strain-to-Well Assignment (single-module)

## Scope decisions

- Single file: `array_randomizer.py` in project root
- Cross-replicate negative spacing: 8-connectivity (no edge or corner adjacency) per plate

## File layout

- `array_randomizer.py`
  - Public API: `generate_layout`, `export_layout`, `export_metadata`, `plot_plate`, `plot_negative_uniqueness`, `validate_layout`
  - Constants: `VISUALIZATION_CONFIG`

## Core data structures

- Config dict: rows, cols, negatives, replicates, seed
- Data dict (return of `generate_layout`):
  - `config`: input params and derived values
  - `replicates`: list[{'plates': dict[str, np.ndarray]}]
  - `strain_plate_mapping`: dict[str, str]
  - `negative_history`: dict[str, dict[str, list[tuple[int,int]]]]
  - `alternation_scores`: list[float]
  - `random_seed`: int|None

## Key algorithms

- Base set creation
  - Compute plates: `ceil(n_strains / ((rows*cols) - n_negatives))`
  - Assign strains row-major within each plate; keep `strain_plate_mapping`
  - Place negatives per plate with constraints:
    - unique rows, unique cols
    - 8-connectivity separation within replicate (no touching edges/corners)
  - Fill remaining wells with strains; negatives are labeled "negative"

- Replicate generation (2..n)
  - For each plate, sample a new negative set satisfying:
    - same within-replicate constraints as base
    - per-plate global uniqueness across replicates (no position reuse)
    - per-plate 8-connectivity away from all prior replicate negatives
  - Strain shuffle per-plate among non-negative wells only (no cross-plate moves)
  - Alternation optimization vs. previous replicate (per-plate):
    - Pre-compute `position_type` of every non-negative well (inner/edge) assuming all non-negative are filled
    - For each strain, prefer positions that flip its prior type (inner→edge, edge→inner)
    - Greedy assignment: place those with unique feasible spots first, then random fill

- Alternation scoring
  - `calculate_alternation_score(rep_i, rep_j)`: fraction of strains that flipped edge/inner between consecutive replicates (adjacent pairs)

## Public API (essential signatures)

```python
def generate_layout(
    strain_names: list[str] | None,
    n_strains: int | None,
    n_rows: int,
    n_columns: int,
    n_negatives: int,
    n_replicates: int,
    random_seed: int | None = None,
) -> dict: ...

def export_layout(
    data: dict,
    format: str = 'long',
    replicate_id: int | None = None,
    plate_id: str | None = None,
) -> 'pd.DataFrame': ...

def export_metadata(data: dict) -> dict: ...

def plot_plate(
    data: dict,
    replicate_id: int,
    plate_id: str,
    annotate: bool = True,
    highlight_inner: bool = True,
    highlight_negatives: bool = True,
): ...

def plot_negative_uniqueness(data: dict): ...

def validate_layout(data: dict) -> dict[str, bool]: ...
```

## Helper functions (internal)

- Seed control: `_set_random_seed(seed)`
- Strain generation: `_make_generic_strains(n_strains)`
- Plate math: `_plate_ids(n)`, `_rc_to_labels(r,c)`, `_neighbors_cardinal(r,c,rows,cols)`
- Occupancy utils: `_is_inner(occ_map, r, c)`, `_count_neighbors(occ_map, r, c)`
- Negative placement:
  - `_sample_negative_positions(rows, cols, k, prior_sets: list[set[tuple]], max_tries=10000)`
    - Row/col uniqueness via sampled row set + sampled col set + random permutation
    - Reject if any pair 8-adjacent; also reject if any 8-adjacent to `prior_sets`
- Alternation:
  - `_classify_positions(rows, cols, negative_set) -> dict[(r,c), 'inner'|'edge']`
  - `_assign_strains_with_alternation(strains, preferred_map, positions)`

## Exports

- Long format columns per spec: replicate_id, plate_id, row/col indices, labels, well_id, content, `is_inner`, `neighbor_count`, `position_type`
- Wide format: pivot by `(replicate_id, plate_id)` with columns A1..H12

## Validation & errors

- Hard checks:
  - `n_negatives <= min(rows, cols)`
  - Capacity per plate ≥ strains per plate
  - Per-plate feasibility: `n_negatives * n_replicates <= rows*cols` (for uniqueness); warn if 8-connectivity likely infeasible after many attempts
- `validate_layout` booleans per spec (8 items)
- Warnings on alternation < 0.7 (not fatal)

## Visualization

- Configurable via `VISUALIZATION_CONFIG`
- `plot_plate`: annotated heatmap with colors per strain, black negatives; red border for inner, green for edge
- `plot_negative_uniqueness`: frequency heatmap (A), row/col usage matrices (B), scatter by replicate (C)
- Defer advanced plots (`plot_alternation_pairs`, `plot_strain_journey`, `plot_verification_dashboard`) to Phase 4

## Usage examples

- Build base + replicates: `data = generate_layout(strain_names=[...], n_rows=8, n_columns=12, n_negatives=8, n_replicates=3, random_seed=42)`
- Export: `df_long = export_layout(data)`, `df_wide = export_layout(data, format='wide')`
- Plots: `plot_plate(data, 0, 'plate_0')`, `plot_negative_uniqueness(data)`

### To-dos

- [ ] Create single-module `array_randomizer.py` with public API and config
- [ ] Implement parameter normalization and constraint validation
- [ ] Implement base set creation and negative placement per plate
- [ ] Add replicate generation with 8-connectivity uniqueness across replicates
- [ ] Implement edge/inner classification and greedy alternation assignment
- [ ] Implement long and wide DataFrame exports
- [ ] Implement `validate_layout` with 8 checks
- [ ] Implement `plot_plate` and `plot_negative_uniqueness`
- [ ] Implement `export_metadata` with scores and histories
- [ ] Stub advanced plots for Phase 4