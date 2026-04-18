## Anti-jamming-unsync

This directory now has three relevant notebook paths for the two-satellite relative delay-Doppler experiment:

- `delay_doppler_simu_coarse.ipynb`: recommended main entry. This is the notebook that matches the LaTeX-style coarse search setup with `BW=200 MHz`, `N_fft=1024`, delay grid `-200:200`, Doppler grid `-250:25:250 kHz`, and `8421` total hypotheses.
- `delay_doppler_simu_precision.ipynb`: fine-search variant. It keeps the same delay grid but uses a narrower and finer relative-Doppler grid, intended when you already know the inter-satellite relative Doppler is small and want tighter local estimation.
- `doppler_simu.ipynb`: legacy baseline notebook kept for comparison. It is no longer the preferred entry for the current two-satellite relative delay-Doppler workflow.

Both `delay_doppler_simu_coarse.ipynb` and `delay_doppler_simu_precision.ipynb` can now:

- generate raw physical CIRs `a_tx, tau_tx`
- optionally build a TX precoder from `a_tx`
- save the effective precoded CIR `a_eff`
- let the final search cell choose between `a_tx` and `a_eff`

The shared channel export is still `doppler_a_tx_tau_tx.jsonl`.

## Files

- `starlink_tracker.ipynb`: Starlink visibility sampling, ranking, tracking, and visualization.
- `delay_doppler_simu_coarse.ipynb`: coarse relative delay-Doppler search notebook aligned with the writeup.
- `delay_doppler_simu_precision.ipynb`: fine relative-Doppler search notebook for local refinement.
- `doppler_simu.ipynb`: older baseline notebook retained for reference.
- `Joint_waterfilling.py`: helper functions for dominant-SVD precoding, joint water-filling, and mapping `a_tx -> a_eff`.
- `leo_utils.py`: ENU geometry helpers used by the notebooks.
- `vsat_dish_3gpp.py`: custom Sionna RT antenna-pattern registration. This requires `sionna`, `mitsuba`, `drjit`, and `scipy`.
- `minmax_solvers.py`: optimization helpers imported by the notebooks.
- `starlink_timeseries_5min_all.csv`: visibility samples with ENU positions and velocities.
- `starlink_timeseries_5min_top5.csv`: tracked top-5 subset used by the notebooks.
- `doppler_a_tx_tau_tx.jsonl`: Doppler-aware channel export produced by the CIR-generation cells.
- `result_plot/`: output figures and animation from `starlink_tracker.ipynb`.

## Setup

Recommended Python version: `3.10` or `3.11`.

Install dependencies:

```bash
cd /home/sj4025/my_project/Anti-jamming-unsync
python -m pip install -r requirements.txt
```

Notes:

- `starlink_tracker.ipynb` needs network access only when `RUN_SAMPLE=True`, because it downloads fresh Starlink TLE data from Celestrak. With the default `RUN_SAMPLE=False`, it reuses the local CSV.
- The delay-Doppler notebooks depend on the Sionna RT stack. On some machines you may prefer to install a hardware-specific TensorFlow build first, then install the rest of `requirements.txt`.
- Run the notebooks from this directory so that local imports and relative paths resolve consistently.

## Recommended Workflow

1. Run `starlink_tracker.ipynb` only if you want to regenerate the tracked satellite CSV.
2. Use `delay_doppler_simu_coarse.ipynb` as the default notebook for the current two-satellite relative delay-Doppler experiment.
3. Use `delay_doppler_simu_precision.ipynb` when the relative Doppler range is already known to be small and you want a finer local grid.
4. Use `doppler_simu.ipynb` only if you need to reproduce the older baseline behavior.

If you only want to rerun the final search stage, the last cell in both `delay_doppler_simu_coarse.ipynb` and `delay_doppler_simu_precision.ipynb` can reuse `doppler_a_tx_tau_tx.jsonl` without recomputing the CIR tensors.

## delay_doppler_simu_coarse.ipynb

### What it does

- Loads `starlink_timeseries_5min_top5.csv` and keeps the first two ranked satellites by default.
- Builds Doppler-aware Sionna CIRs for the ground TX to the selected satellites.
- Optionally computes a TX precoder from `a_tx`.
- Saves `a_tx`, `tau_tx`, optional `a_eff`, and precoder metadata into `doppler_a_tx_tau_tx.jsonl`.
- Runs the final two-signal relative delay-Doppler search.

### Why this is the main notebook

Its default search grid matches the coarse search in the writeup:

- `BW = 200 MHz`
- `N_fft = 1024`
- `T_fft = 5.12 us`
- delay grid: `d = -200, ..., 200`
- Doppler grid: `f_d = -250, -225, ..., 250 kHz`
- hypotheses: `401 x 21 = 8421`

This is the right notebook if your goal is:

- two satellites receiving the same ground TX
- different per-satellite delay and Doppler
- a coarse joint relative delay-Doppler acquisition stage before later array combining

### Precoder controls

In the CIR export cell:

- `precoder_mode = "off"` keeps only `a_tx`
- `precoder_mode = "dominant"` builds a dominant right-singular-vector precoder
- `precoder_mode = "joint_waterfilling"` builds a water-filling precoder using `N0` and `P0`

In the final search cell:

- `search_channel_key = "a_tx"` uses the raw physical channel
- `search_channel_key = "a_eff"` uses the precoded effective channel

For RX-side relative delay-Doppler detection, `a_tx` is still the safer default because it preserves the direct physical interpretation. `a_eff` is available when you explicitly want to include TX precoding in the experiment.

## delay_doppler_simu_precision.ipynb

### What it does

This notebook shares the same overall structure as `delay_doppler_simu_coarse.ipynb`, including:

- the same two-satellite relative delay search logic
- the same JSONL reuse path
- the same optional `a_eff` generation
- the same `search_channel_key = "a_tx" | "a_eff"` switch in the final search cell

### How it differs from coarse

- Delay grid: same as coarse, `-200:200` blocks.
- Doppler grid: much narrower and finer, currently `-20:1:20 kHz`.
- Sionna sampling: currently `1 MHz` instead of `2 MHz`.
- Intended use: local refinement when the relative Doppler between the two satellites is already expected to be small.

Use `precision` when:

- geometry says the inter-satellite relative Doppler is modest
- the coarse grid is too wide or too coarse
- you want a finer local estimate after the coarse acquisition stage

Do not treat it as the notebook that matches the LaTeX coarse-search setup. For that, use `delay_doppler_simu_coarse.ipynb`.

## doppler_simu.ipynb

This notebook is the older baseline version. Compared with `delay_doppler_simu_coarse.ipynb`, it is missing the newer workflow features:

- no `a_eff` save/load path
- no `search_channel_key` switch
- no `Joint_waterfilling.py` precoder integration
- less explicit separation between coarse-search and fine-search use cases

For the current project direction, it should be treated as legacy reference rather than the default working notebook.

## Practical Notes

- All three notebooks assume the tracked CSV already contains ENU velocity columns. `starlink_tracker.ipynb` writes those columns.
- The Sionna RT CIR generation is the expensive step. Reusing `doppler_a_tx_tau_tx.jsonl` is the fastest way to iterate on the search cell.
- If you later move from relative acquisition to joint post-alignment processing, you can use the detected relative delay-Doppler to align the two satellite links and then stack them into a larger effective RX array. That is downstream of the current detection stage; the notebooks here only cover the relative search part.
