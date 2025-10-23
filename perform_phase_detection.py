#!/usr/bin/env python3
# Perform a simple sliding-window phase detection experiment:
# 1) Read a reference waveform.
# 2) Select a target phase segment [BEG:TER) from it.
# 3) Slide that phase across the entire series and compute a distance per position.
# 4) Plot per-position figures and assemble them into an animated GIF.
# 5) Report the best match (smallest distance).

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from synthetic_tests_lib import (
    create_dir,      # small helper to mkdir -p
    read_waveforms,  # user-provided data loader
    plot_figure,     # user-provided plotting routine
    calc_distance,   # user-provided distance metric between two signals
)

# =========================== USER CONFIG (edit here) =========================== #
# File prefix passed to read_waveforms(); adjust to your dataset naming.
FILEPREFIX = "STN_syn"
# Target phase segment in the reference series: [BEG:TER) (0-based, TER is exclusive).
BEG = 100
TER = 120
# Add zero-mean Gaussian noise to series1, scaled by std(series1) * SNR.
SNR = 1.0
# Extra parameter forwarded to calc_distance(); tweak per your distance function.
DIST_WINDOW = 2
# Random generator seed to make noise reproducible.
SEED = 19680801
# Toggle tqdm progress bars. Set False for quiet runs or non-interactive logs.
SHOW_PROGRESS = True
# Final GIF will be persisted/copied to this path (outside the temp workspace).
PERSIST_GIF_PATH = Path("figures/phaseDetection.gif")
# ============================================================================== #


# --------------------------- Utilities --------------------------- #

def add_noise(series: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by the series' standard deviation and SNR."""
    if snr <= 0:
        # Return a copy to avoid mutating the caller's array.
        return series.copy()
    return series + snr * rng.normal(0.0, np.std(series), size=series.size)


def compute_ylim(arr: np.ndarray, margin: float = 0.10) -> float:
    """Return symmetric y-limit (abs max * (1 + margin))."""
    ymax = float(np.max(np.abs(arr)))
    return ymax * (1.0 + margin)


def generate_x(series1: np.ndarray, center_x: int, window: int) -> np.ndarray:
    """
    Build a zero array with a windowed copy from series1 at [begx:terx).
    Indices are clipped to array bounds; terx is exclusive as per NumPy slicing.
    """
    length = len(series1)
    begx = max(center_x - window // 2, 0)
    # terx is exclusive; subtracting 1 ensures we don't overshoot the slice end at the boundary
    terx = min(center_x + window // 2, length - 1)
    out = np.zeros(length, dtype=series1.dtype)
    out[begx:terx] = series1[begx:terx]
    return out


def generate_y(phase_to_detect: np.ndarray, center_x: int, window: int, length: int) -> np.ndarray:
    """
    Build a zero array with a windowed copy of phase_to_detect at [begy:tery).
    The segment is clipped both by the target array bounds and by phase_to_detect size.
    """
    begy = max(center_x - window // 2, 0)
    tery = min(center_x + window // 2, length - 1)
    out = np.zeros(length, dtype=phase_to_detect.dtype)
    seg_len = min(tery - begy, phase_to_detect.size)
    if seg_len > 0:
        out[begy : begy + seg_len] = phase_to_detect[:seg_len]
    return out


def natural_key(s: str) -> Tuple[str, int]:
    """Natural sort key for names like 'phaseDetect12' -> ('phaseDetect', 12)."""
    m = re.match(r"^(\D+)(\d+)$", s)
    if not m:
        return (s, 0)
    return m.group(1), int(m.group(2))


def make_gif_from_pngs(png_dir: Path, out_gif: Path, frame_duration_ms: int = 200) -> None:
    """
    Collect PNGs from `png_dir` (sorted by natural order) and save an animated GIF to `out_gif`.
    Expects files named like phaseDetect<index>.png.
    """
    names = sorted((p.stem for p in png_dir.glob("*.png")), key=natural_key)
    if not names:
        return
    frames: List[Image.Image] = [Image.open(png_dir / f"{name}.png") for name in names]
    frames[0].save(
        out_gif,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration_ms,
        loop=0,
    )


def fmt_time_value(t: Any) -> str:
    """
    Best-effort formatter for a time value from `orig_times`.
    Supports floats/ints, numpy.datetime64, pandas Timestamp, and objects with isoformat().
    """
    # pandas/NumPy datetimes
    if isinstance(t, (np.datetime64,)):
        return pd.to_datetime(t).isoformat()
    # pandas Timestamp or any object with isoformat()
    if hasattr(t, "isoformat"):
        try:
            return t.isoformat()
        except Exception:
            pass
    # Fallback for numeric scalars
    try:
        return f"{float(t):.6f}"
    except Exception:
        return str(t)


# --------------------------- Core Processing --------------------------- #

def phase_search_and_plot(
    fileprefix: str,
    beg: int,
    ter: int,
    snr: float,
    dist_window: int,
    outdir: Path,
    csv_path: Path,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> tuple[int, float, Any, Path]:
    """
    Run the sliding search, compute distances, plot frames, save CSV and GIF in `outdir`.
    Returns (best_index, best_distance, best_time_value, gif_path_within_outdir).
    """
    create_dir(str(outdir))

    # Read traces; read_waveforms returns (times, series1, series2, ...) per user lib.
    orig_times, orig_r, _, _ = read_waveforms(fileprefix=fileprefix)

    # series1 is the noisy copy to emulate detection in noisy observations.
    series1 = add_noise(orig_r.copy(), snr, rng)
    # series2 holds the clean reference we extract the phase from.
    series2 = orig_r.copy()

    # Define phase window [beg:ter); validates user-specified bounds.
    assert 0 <= beg < ter <= len(series2), "Invalid BEG/TER bounds for phase segment."
    phase_to_detect = series2[beg:ter]
    window = ter - beg

    # Pre-compute y-limits so all frames share consistent axes.
    ylim_x = compute_ylim(series1)
    ylim_y = compute_ylim(phase_to_detect)

    xvals: List[int] = []
    dist_vals: List[float] = []

    # Avoid edges to reduce partial-window artifacts near array boundaries.
    start = window + 2
    stop = len(series2) - window - 2
    iters = range(start, stop)

    # Compute distance at each center index by placing the windowed snippets.
    iterator = tqdm(iters, desc="Computing distances", unit="pos") if show_progress else iters
    for center_x in iterator:
        sx = generate_x(series1, center_x, window)
        sy = generate_y(phase_to_detect, center_x, window, len(series2))
        d = calc_distance(sx, sy, window=dist_window)
        xvals.append(center_x)
        dist_vals.append(float(d))

    # Consolidate results and locate the minimum (best match).
    phase_df = pd.DataFrame({"xvals": xvals, "dist_vals": dist_vals})
    min_idx = int(phase_df["dist_vals"].idxmin())
    min_dist_x = int(phase_df.loc[min_idx, "xvals"])
    min_dist = float(phase_df.loc[min_idx, "dist_vals"])

    # Save the per-position distances as a CSV for reproducibility/debugging.
    # This goes into the temp workspace and is not persisted long-term by design.
    phase_df.to_csv(csv_path, index=False)

    # Render a frame for each position; outfigprefix controls per-frame filenames.
    iterator = tqdm(zip(xvals, dist_vals), total=len(xvals), desc="Plotting frames", unit="pos") if show_progress else zip(xvals, dist_vals)
    for center_x, distance in iterator:
        sx = generate_x(series1, center_x, window)
        sy = generate_y(phase_to_detect, center_x, window, len(series2))
        plot_figure(
            orig_times, sx, sy, series1, phase_df, distance, center_x,
            outfigprefix=str(outdir / f"phaseDetect{center_x}"),
            ylimX=ylim_x, ylimY=ylim_y, vline=min_dist_x,
        )

    # Also render a special "best" frame without dashed styling.
    best_sx = generate_x(series1, min_dist_x, window)
    best_sy = generate_y(phase_to_detect, min_dist_x, window, len(series2))
    plot_figure(
        orig_times, best_sx, best_sy, series1, phase_df, min_dist, min_dist_x,
        outfigprefix=str(outdir / "phaseDetect_best"),
        ylimX=ylim_x, ylimY=ylim_y, vline=min_dist_x, dashed=False,
    )

    # Build an animated GIF from all per-position PNGs in the temp output directory.
    gif_path_tmp = outdir / "phaseDetection.gif"
    make_gif_from_pngs(outdir, gif_path_tmp, frame_duration_ms=200)

    # Best time value for the best match index, if times are indexable.
    try:
        best_time_value = orig_times[min_dist_x]
    except Exception:
        best_time_value = min_dist_x  # fallback to index if times are not index-like

    return min_dist_x, min_dist, best_time_value, gif_path_tmp


# --------------------------- Entry Point --------------------------- #

if __name__ == "__main__":
    # Initialize RNG with a fixed seed for reproducible noise injection.
    rng = np.random.default_rng(SEED)

    # Use a temporary directory for all intermediate outputs to keep the repo clean.
    with tempfile.TemporaryDirectory(prefix="phase_detect_") as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output_figures"
        csv_path = tmp_path / "phase_search.txt"

        # Run the end-to-end search and plotting into the temp workspace.
        best_idx, best_dist, best_time_val, gif_tmp = phase_search_and_plot(
            fileprefix=FILEPREFIX,
            beg=BEG,
            ter=TER,
            snr=SNR,
            dist_window=DIST_WINDOW,
            outdir=outdir,
            csv_path=csv_path,
            rng=rng,
            show_progress=SHOW_PROGRESS,
        )

        # Persist only the GIF to the project directory; rest of temp files are discarded.
        shutil.copyfile(gif_tmp, PERSIST_GIF_PATH)

    # Final concise output suitable for logs/CLI consumption.
    print(f"Saved GIF: {PERSIST_GIF_PATH}")
    print(f"Best match index: {best_idx}")
    print(f"Best match time: {fmt_time_value(best_time_val)}")
    print(f"Minimum distance: {best_dist:.6g}")
