## Anti-jamming-unsync

This directory contains two notebook-driven workflows:

- `starlink_tracker.ipynb`: samples visible Starlink satellites, computes ENU position and velocity time series, tracks the rolling top-`k` satellites, and exports static/animated sky plots.
- `doppler_simu.ipynb`: builds a ground-to-satellite propagation setup with Sionna RT, generates Doppler-aware channel responses for two moving satellite receivers, saves the channel tensors to JSONL, and runs a two-dimensional delay-Doppler search experiment.

The notebooks share data through CSV and JSONL files already stored in this folder.

## Files

- `starlink_tracker.ipynb`: Starlink visibility sampling, ranking, tracking, and visualization.
- `doppler_simu.ipynb`: channel generation and delay-Doppler search over two received signals.
- `leo_utils.py`: ENU geometry helpers used by `doppler_simu.ipynb`.
- `vsat_dish_3gpp.py`: custom Sionna RT antenna-pattern registration. This requires `sionna`, `mitsuba`, `drjit`, and `scipy`.
- `minmax_solvers.py`: optimization helpers imported by `doppler_simu.ipynb`.
- `starlink_timeseries_5min_all.csv`: visibility samples with ENU positions and velocities.
- `starlink_timeseries_5min_top5.csv`: tracked top-5 subset used by `doppler_simu.ipynb`.
- `doppler_a_tx_tau_tx.jsonl`: Doppler-aware channel export produced by `doppler_simu.ipynb`.
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
- `doppler_simu.ipynb` depends on the Sionna RT stack. On some machines you may prefer to install a hardware-specific TensorFlow build first, then install the rest of `requirements.txt`.
- The notebooks also depend on local modules in this folder, so run them with this directory on the Python path. Running the notebook from this folder is the safest choice.

## Recommended Workflow

1. Run `starlink_tracker.ipynb` if you want to regenerate visibility samples or refresh plots.
2. Confirm that `starlink_timeseries_5min_top5.csv` exists and contains ENU position and velocity columns.
3. Run `doppler_simu.ipynb` to generate Doppler-aware channels and the delay-Doppler search visualization.

If you only want to rerun the final search stage in `doppler_simu.ipynb`, the last cell can reuse `doppler_a_tx_tau_tx.jsonl` without recomputing the CIR tensors, as long as that file exists.

## starlink_tracker.ipynb

### What it does

- Builds a `TrackerConfig` with observer location, sampling duration, elevation mask, and output paths.
- Optionally downloads fresh Starlink TLEs from Celestrak.
- Samples visible satellites over time.
- Converts the sampled geometry into ENU position and velocity columns.
- Tracks a rolling top-`k` visible set by slant range.
- Saves a static sky plot (`png` and `pdf`) and an animated sky plot (`gif`).

### Key configuration

The main controls live in the `TrackerConfig` dataclass near the top of the notebook:

- `lat_deg`, `lon_deg`, `elev_m`: observer location.
- `elev_min_deg`: visibility mask.
- `mode`, `duration_sec`, `interval_sec`: time-series sampling controls.
- `top_k`: number of satellites to keep in the tracked set.
- `visibility_csv`: full sampled visibility table.
- `tracked_csv`: tracked top-`k` table, currently `starlink_timeseries_5min_top5.csv`.

The execution switch is:

```python
RUN_SAMPLE = False
```

Set `RUN_SAMPLE=True` only when you want to download fresh TLEs and rebuild the CSV from scratch.

### Main outputs

- `starlink_timeseries_5min_all.csv`
- `starlink_timeseries_5min_top5.csv`
- `result_plot/starlink_sky_topk.png`
- `result_plot/starlink_sky_topk.pdf`
- `result_plot/starlink_sky_topk.gif`

## doppler_simu.ipynb

### What it does

- Loads `starlink_timeseries_5min_top5.csv` and sorts each timestamp by visibility rank.
- Defines one ground transmitter at `(0, 0, 0)` and one additional ground point used elsewhere in the notebook.
- Uses `compute_cir(...)` with Sionna RT synthetic arrays to generate Doppler-aware channel tensors for the selected satellites.
- Stores channel amplitudes and delays into `doppler_a_tx_tau_tx.jsonl`.
- Runs a block-level two-dimensional delay-Doppler search for the first two satellite receivers.

### Current default behavior

In the CIR export cell, the notebook currently uses:

- `run_steps = 1`
- `topk = 2`
- `cir_sampling_frequency = 1.0`
- `doppler_num_time_steps = 14`
- `out_path = "doppler_a_tx_tau_tx.jsonl"`

That means the default path is a small, focused experiment over the first timestamp and the first two tracked satellites.

### Delay-Doppler search stage

The final cell uses the LaTeX-style search setup:

- `BW = 200 MHz`
- `N_fft = 1024`
- `T_fft = 5.12 us`
- delay hypotheses: `d = -200, ..., 200`
- Doppler hypotheses: `f_d = -250, -225, ..., 250 kHz`

It then:

- extracts effective scalar links for the two receivers,
- estimates continuous relative delay and relative Doppler from the generated geometry,
- synthesizes block-rate observations,
- evaluates the 2D search metric over all `8421` hypotheses,
- displays summary tables and a heatmap with delay and Doppler cuts.

### Main outputs

- `doppler_a_tx_tau_tx.jsonl`
- in-notebook tables summarizing search parameters and estimated peaks
- in-notebook plots for the 2D metric and the one-dimensional cuts

## Practical Notes

- `doppler_simu.ipynb` expects the tracked CSV to already contain ENU velocity columns. `starlink_tracker.ipynb` writes those columns.
- The Sionna RT portion is the heaviest part of the workflow. If you only want the search plots, reuse the saved JSONL rather than recomputing the channel tensors.
- Relative file paths are written assuming the notebook is run from this directory. Some cells include fallbacks, but staying inside `Anti-jamming-unsync/` avoids path confusion.
