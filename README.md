## Array Randomizer

Automated strain-to-well assignment for multiwell plates with strict negative placement and replicate-aware randomization.

- Tracks strain names end-to-end
- Places negatives with spatial rules (unique rows/cols, 8-connectivity separation)
- Never places negatives on the outer border (first/last row or column)
- Confines negatives to the strain boundary on partially filled plates
- Randomizes per replicate without cross-plate moves; optimizes edge/inner alternation
- Exports long/wide DataFrames; includes validation and visualizations

### Installation

Requires Python >= 3.10. Dependencies are defined in `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Quickstart

```python
from array_randomizer import (
    generate_layout, export_layout, export_metadata,
    plot_plate, plot_negative_uniqueness, validate_layout,
)

strain_names = [f"S{i:03d}" for i in range(1, 161)]  # 160 strains → 2 plates on 96-well

data = generate_layout(
    strain_names=strain_names,
    n_rows=8,
    n_columns=12,
    n_negatives=8,
    n_replicates=3,
    random_seed=42,
)

df_long = export_layout(data, format='long')
df_wide = export_layout(data, format='wide')

meta = export_metadata(data)
checks = validate_layout(data)

fig1, ax1 = plot_plate(data, replicate_id=0, plate_id='plate_0')
fig2 = plot_negative_uniqueness(data)
```

See `example.ipynb` for a full walkthrough.

### Inputs and Parameters

Primary input:
- `strain_names: list[str]`

Alternative input:
- `n_strains: int` (auto-generates IDs `strain_001`, ...)

Required parameters:
- `n_rows: int`, `n_columns: int`
- `n_negatives: int` (≤ `min(n_rows, n_columns)`)
- `n_replicates: int`
- `random_seed: int | None`

Derived:
- `usable_wells = (n_rows * n_columns) - n_negatives`
- `n_plates = ceil(n_strains / usable_wells)`

### Negative Placement Rules

Within each replicate and plate:
- Unique rows and unique columns
- No adjacency between negatives (8-connectivity)
- Never on first/last row or first/last column
- If a plate is partially filled, negatives stay within the strain boundary (row‑major extent)

Across replicates (per plate):
- Positions never reused
- New negatives are 8‑connectivity independent from all prior negatives

### Replicate Randomization and Alternation

- Replicate 1: base set; row‑major strain assignment; negatives placed per rules
- Replicates 2..N: new negatives per rules, shuffle strains within the same plate only
- Greedy heuristic favors flipping position type (edge ↔ inner) vs. previous replicate
- Alternation scores recorded in `data["alternation_scores"]`

### API

```python
data = generate_layout(
    strain_names: list[str] | None,
    n_strains: int | None,
    n_rows: int,
    n_columns: int,
    n_negatives: int,
    n_replicates: int,
    random_seed: int | None = None,
) -> dict

df = export_layout(data, format='long'|'wide', replicate_id=None, plate_id=None)

meta = export_metadata(data)

fig, ax = plot_plate(data, replicate_id, plate_id,
                     annotate=True, highlight_inner=True, highlight_negatives=True)

fig = plot_negative_uniqueness(data)

checks = validate_layout(data)  # dict[str, bool]
```

### Reproducibility

Set `random_seed` for deterministic results (both `numpy` and `random` seeded).

### Troubleshooting

- Maximum negatives: must be ≤ `min(n_rows, n_columns)`
- Insufficient wells: reduce `n_strains` or `n_negatives`
- Unique negatives across replicates impossible: reduce `n_negatives * n_replicates`
- Unable to sample negatives: constraints too tight with interior/boundary; reduce `n_negatives` or increase strains

### Visualization Settings

Adjust `VISUALIZATION_CONFIG` in `array_randomizer.py`:
- `figsize`, `dpi`, `strain_name_truncate`, `colormap`
- `negative_color`, `inner_border_color`, `edge_border_color`, `font_size`, `save_format`


