#!/usr/bin/env python3
# Sliding-window phase detection:
# 1) Read a reference waveform.
# 2) Take a target phase segment [BEG:TER).
# 3) Slide the segment across the series and compute a distance per position.
# 4) Plot per-position figures and assemble an animated GIF.
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
    create_dir,
    read_waveforms,
    plot_figure,
    calc_distance,
)

# =========================== USER CONFIG (edit here) =========================== #
FILEPREFIX = "example_traces/STN_syn"   # Prefix passed to read_waveforms()
BEG = 100                               # Start of target phase (inclusive, 0-based)
TER = 120                               # End of target phase (exclusive)
SNR = 1.0                               # Noise scale: std(series) * SNR (0 disables noise)
DIST_WINDOW = 2                         # Extra parameter forwarded to calc_distance()
SEED = 19680801                         # RNG seed (reproducible noise)
SHOW_PROGRESS = True                    # Toggle tqdm progress bars
PERSIST_GIF_PATH = Path("figures/phaseDetection.gif")  # Final GIF destination
# ============================================================================== #


# --------------------------- Utilities --------------------------- #

def add_noise(series: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by std(series) * snr; return a new array."""
    if snr <= 0:
        return series.copy()
    return series + snr * rng.normal(0.0, np.std(series), size=series.size)


def compute_ylim(arr: np.ndarray, margin: float = 0.10) -> float:
    """Symmetric y-limit: max(|arr|) * (1 + margin)."""
    ymax = float(np.max(np.abs(arr)))
    return ymax * (1.0 + margin)


def generate_x(series1: np.ndarray, center_x: int, window: int) -> np.ndarray:
    """
    Zero array with a windowed copy from series1 at [begx:terx).
    Indices are clipped to bounds; terx is exclusive (NumPy slicing).
    """
    length = len(series1)
    begx = max(center_x - window // 2, 0)
    # terx is exclusive; using length-1 avoids overshooting near the right edge
    terx = min(center_x + window // 2, length - 1)
    out = np.zeros(length, dtype=series1.dtype)
    out[begx:terx] = series1[begx:terx]
    return out


def generate_y(phase_to_detect: np.ndarray, center_x: int, window: int, length: int) -> np.ndarray:
    """
    Zero array with a windowed copy of phase_to_detect at [begy:tery),
    clipped by both the target array and phase length.
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
    Collect PNGs from png_dir (natural sort) and save an animated GIF to out_gif.
    Expects files like phaseDetect<index>.png.
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
    Format a time value from orig_times.
    Supports floats/ints, numpy.datetime64, pandas Timestamp, and isoformat()-able objects.
    """
    if isinstance(t, (np.datetime64,)):
        return pd.to_datetime(t).isoformat()
    if hasattr(t, "isoformat"):
        try:
            return t.isoformat()
        except Exception:
            pass
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
    Run the sliding search, plot frames, save CSV/GIF in outdir.
    Returns (best_index, best_distance, best_time_value, gif_path).
    """
    create_dir(str(outdir))

    # Read traces (times, series1, series2, ...).
    orig_times, orig_r, _, _ = read_waveforms(fileprefix=fileprefix)

    # series1: noisy copy (observation); series2: clean reference.
    series1 = add_noise(orig_r.copy(), snr, rng)
    series2 = orig_r.copy()

    # Phase window [beg:ter).
    assert 0 <= beg < ter <= len(series2), "Invalid BEG/TER bounds for phase segment."
    phase_to_detect = series2[beg:ter]
    window = ter - beg

    # Consistent axes across frames.
    ylim_x = compute_ylim(series1)
    ylim_y = compute_ylim(phase_to_detect)

    xvals: List[int] = []
    dist_vals: List[float] = []

    # Skip edges to reduce partial-window artifacts.
    start = window + 2
    stop = len(series2) - window - 2
    iters = range(start, stop)

    # Distance per center index.
    iterator = tqdm(iters, desc="Computing distances", unit="pos") if show_progress else iters
    for center_x in iterator:
        sx = generate_x(series1, center_x, window)
        sy = generate_y(phase_to_detect, center_x, window, len(series2))
        d = calc_distance(sx, sy, window=dist_window)
        xvals.append(center_x)
        dist_vals.append(float(d))

    # Consolidate results and find the minimum.
    phase_df = pd.DataFrame({"xvals": xvals, "dist_vals": dist_vals})
    min_idx = int(phase_df["dist_vals"].idxmin())
    min_dist_x = int(phase_df.loc[min_idx, "xvals"])
    min_dist = float(phase_df.loc[min_idx, "dist_vals"])

    # Save distances (debug/repro).
    phase_df.to_csv(csv_path, index=False)

    # Render a frame per position.
    iterator = tqdm(zip(xvals, dist_vals), total=len(xvals), desc="Plotting frames", unit="pos") if show_progress else zip(xvals, dist_vals)
    for center_x, distance in iterator:
        sx = generate_x(series1, center_x, window)
        sy = generate_y(phase_to_detect, center_x, window, len(series2))
        plot_figure(
            orig_times, sx, sy, series1, phase_df, distance, center_x,
            outfigprefix=str(outdir / f"phaseDetect{center_x}"),
            ylimX=ylim_x, ylimY=ylim_y, vline=min_dist_x,
        )

    # Render best match (no dashed styling).
    best_sx = generate_x(series1, min_dist_x, window)
    best_sy = generate_y(phase_to_detect, min_dist_x, window, len(series2))
    plot_figure(
        orig_times, best_sx, best_sy, series1, phase_df, min_dist, min_dist_x,
        outfigprefix=str(outdir / "phaseDetect_best"),
        ylimX=ylim_x, ylimY=ylim_y, vline=min_dist_x, dashed=False,
    )

    # Build animated GIF from per-position PNGs.
    gif_path_tmp = outdir / "phaseDetection.gif"
    make_gif_from_pngs(outdir, gif_path_tmp, frame_duration_ms=200)

    # Best time value for the best index (fallback to index if not indexable).
    try:
        best_time_value = orig_times[min_dist_x]
    except Exception:
        best_time_value = min_dist_x

    return min_dist_x, min_dist, best_time_value, gif_path_tmp


# --------------------------- Entry Point --------------------------- #

if __name__ == "__main__":
    # Fixed seed for reproducible noise.
    rng = np.random.default_rng(SEED)

    # Use a temp workspace for intermediates.
    with tempfile.TemporaryDirectory(prefix="phase_detect_") as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output_figures"
        csv_path = tmp_path / "phase_search.txt"

        # Run the end-to-end search.
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

        # Persist only the GIF; temp files are discarded.
        shutil.copyfile(gif_tmp, PERSIST_GIF_PATH)

    # Concise CLI/log output.
    print(f"Saved GIF: {PERSIST_GIF_PATH}")
    print(f"Best match index: {best_idx}")
    print(f"Best match time: {fmt_time_value(best_time_val)}")
    print(f"Minimum distance: {best_dist:.6g}")
