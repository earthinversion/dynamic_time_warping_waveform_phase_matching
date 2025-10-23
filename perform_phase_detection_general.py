#!/usr/bin/env python3
# Sliding-window template matching (DTW-based) with optional GIF generation.
# Toggle MAKE_GIF to speed up runs by skipping all plotting and image I/O.

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from obspy import read, Trace, UTCDateTime
from tqdm import tqdm
from scipy import stats
from dtaidistance import dtw

# Plot style (used only when MAKE_GIF=True)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18})

# =========================== USER CONFIG (edit here) =========================== #
WAVEFORM_PATH: str = "example_traces/STN_syn.z"
TEMPLATE_PATH: str | None = None
BEG: int = 100
TER: int = 120

# Preprocessing
APPLY_DEMEAN: bool = True
APPLY_DETREND: bool = True
TAPER_FRACTION: float = 0.05

# DTW options
DTW_USE_PRUNING: bool = True
DTW_WINDOW_RATIO: float | None = 0.1   # None = unconstrained
DTW_PSI: int = 0

# Output / performance toggles
MAKE_GIF: bool = False                 # set False to skip plotting & GIF (fast mode)
# MAKE_GIF: bool = True                 # set False to skip plotting & GIF (fast mode)
SHOW_PROGRESS: bool = True
PERSIST_GIF_PATH: Path = Path("figures/template_match.gif")
SAVE_DIST_CSV: bool = True
FRAME_STRIDE: int | None = None        # None -> auto stride ≈80 frames (only used if MAKE_GIF)
# ============================================================================== #


# --------------------------- Utilities --------------------------- #

def create_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_ylim(arr: np.ndarray, pad: float = 0.1) -> Tuple[float, float]:
    a = float(np.max(np.abs(arr))) if arr.size else 1.0
    a *= (1.0 + pad)
    return (-a, a)


def natural_key(name: str) -> tuple[str, int]:
    import re
    m = re.match(r"^(\D+)(\d+)$", name)
    if not m:
        return (name, 0)
    return (m.group(1), int(m.group(2)))


def make_gif_from_pngs(png_dir: Path, out_gif: Path, frame_duration_ms: int = 100) -> None:
    names = sorted((p.stem for p in png_dir.glob("*.png")), key=natural_key)
    if not names:
        return
    frames: List[Image.Image] = [Image.open(png_dir / f"{n}.png").convert("RGB") for n in names]
    frames[0].save(
        out_gif,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration_ms,
        loop=0,
    )


def fmt_time_value(starttime: Optional[UTCDateTime], idx: int, dt: float) -> str:
    if starttime is None:
        return f"sample={idx}, t={idx*dt:.6f}s"
    t = starttime + idx * dt
    return f"{t.isoformat()} (sample={idx}, t={idx*dt:.6f}s)"


# --------------------------- IO & Preprocessing --------------------------- #

def read_trace(path: str) -> Trace:
    st = read(path)
    if len(st) == 0:
        raise ValueError(f"No data in {path}")
    return st[0]


def preprocess(tr: Trace) -> Trace:
    tr = tr.copy()
    if APPLY_DEMEAN:
        tr.detrend("demean")
    if APPLY_DETREND:
        tr.detrend("linear")
    if TAPER_FRACTION and TAPER_FRACTION > 0:
        tr.taper(TAPER_FRACTION, type="cosine")
    return tr


def ensure_same_sampling(wave: Trace, tmpl: Trace) -> tuple[Trace, Trace]:
    if not np.isclose(wave.stats.sampling_rate, tmpl.stats.sampling_rate):
        tmpl = tmpl.copy()
        tmpl.resample(wave.stats.sampling_rate, no_filter=True)
    return wave, tmpl


# --------------------------- DTW distance --------------------------- #

def _zscore_safe(x: np.ndarray) -> np.ndarray:
    """Z-score with safety for near-constant vectors."""
    z = stats.zscore(x, ddof=0)
    if np.isnan(z).any():
        return np.zeros_like(x, dtype=np.float64)
    return z.astype(np.float64, copy=False)


def calc_dtw_distance_segment(
    wave: np.ndarray, tmpl_z: np.ndarray, center: int,
    window_ratio: float | None, use_pruning: bool, psi: int
) -> float:
    """
    DTW distance between z-scored template (tmpl_z) and a z-scored waveform segment at `center`.
    """
    L = len(tmpl_z)
    half = L // 2
    beg = max(center - half, 0)
    end = min(beg + L, len(wave))
    beg = max(0, end - L)  # adjust near right boundary

    seg = wave[beg:end]
    if seg.size != L:
        seg = np.pad(seg, (0, L - seg.size), mode="constant", constant_values=0.0)

    s1 = _zscore_safe(seg)
    s2 = tmpl_z

    w = None
    if window_ratio is not None and window_ratio > 0:
        w = max(1, int(round(window_ratio * L)))

    return float(
        dtw.distance_fast(
            s1, s2, window=w, use_pruning=use_pruning, psi=psi
        )
    )


# --------------------------- Core Matching (plotting optional) --------------------------- #

def padded_template_over_wave(n: int, tmpl: np.ndarray, center: int) -> np.ndarray:
    """Pad template to waveform length and place it centered at `center` (clipped at boundaries)."""
    L = len(tmpl)
    half = L // 2
    beg = max(center - half, 0)
    end = min(beg + L, n)
    beg = max(0, end - L)
    y = np.zeros(n, dtype=tmpl.dtype)
    y[beg:end] = tmpl[: end - beg]
    return y


def run_dtw_search(
    wave: Trace, tmpl: Trace, outdir: Path, make_gif: bool
) -> tuple[int, float, Optional[Path]]:
    """
    Slide template over waveform and compute DTW distance per sample.
    If `make_gif` is True, also render frames and assemble a GIF.
    Returns (best_index, min_distance, gif_path_or_None).
    """
    if make_gif:
        create_dir(outdir)

    wave_data = wave.data.astype(np.float64, copy=False)
    tmpl_data = tmpl.data.astype(np.float64, copy=False)
    tmpl_z = _zscore_safe(tmpl_data)  # precompute once

    n = len(wave_data)
    L = len(tmpl_z)
    dt = wave.stats.delta

    # Search bounds
    start = L // 2 + 1
    stop = n - (L - L // 2) - 1
    if stop <= start:
        start, stop = 0, n

    xvals: List[int] = []
    dvals: List[float] = []

    iterator = tqdm(range(start, stop), desc="Computing DTW distances", unit="pos") if SHOW_PROGRESS else range(start, stop)
    for c in iterator:
        d = calc_dtw_distance_segment(
            wave=wave_data,
            tmpl_z=tmpl_z,
            center=c,
            window_ratio=DTW_WINDOW_RATIO,
            use_pruning=DTW_USE_PRUNING,
            psi=DTW_PSI,
        )
        xvals.append(c)
        dvals.append(d)

    darr = np.asarray(dvals)
    min_idx_rel = int(np.argmin(darr))
    min_idx = xvals[min_idx_rel]
    min_dist = float(darr[min_idx_rel])

    if SAVE_DIST_CSV:
        if make_gif:
            csv_out = outdir / "distances.csv"
        else:
            csv_out = Path("distances.csv")
        pd.DataFrame({"index": xvals, "distance_dtw": dvals}).to_csv(csv_out, index=False)

    gif_path: Optional[Path] = None
    if make_gif:
        # Precompute for plotting
        ylim_wave = compute_ylim(wave_data)
        ylim_tmp = compute_ylim(tmpl_data)
        t_wave = np.arange(n) * dt
        dmin, dmax = float(darr.min()), float(darr.max())
        pad = 0.05 * (dmax - dmin + 1e-12)
        ylim_dist = (dmin - pad, dmax + pad)

        stride = FRAME_STRIDE if FRAME_STRIDE and FRAME_STRIDE > 0 else max(1, len(xvals) // 80)

        for i, (c, d) in enumerate(zip(xvals, dvals)):
            if (i % stride != 0) and (c != min_idx):
                continue

            fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)

            # Row 1: Series 1
            axes[0].plot(t_wave, wave_data, lw=1)
            half = L // 2
            beg = max(c - half, 0)
            end = min(beg + L, n)
            axes[0].fill_between(t_wave[beg:end], wave_data[beg:end], 0, alpha=0.25, color="red")
            axes[0].axvline(c * dt, ls="--", lw=1)
            axes[0].set_ylim(*ylim_wave)
            axes[0].set_ylabel("Series 1")

            # Row 2: Series 2 (padded template at current center)
            padded_tmpl = padded_template_over_wave(n, tmpl_data, c)
            axes[1].plot(t_wave, padded_tmpl, lw=1, color="C1")
            axes[1].set_ylim(*ylim_tmp)
            axes[1].set_ylabel("Series 2")

            # Row 3: DTW dissimilarity
            xs = np.array(xvals[: i + 1])
            ys = np.array(dvals[: i + 1])
            axes[2].plot(xs * dt, ys, lw=1.5, color="C4")
            axes[2].scatter([c * dt], [d], color="tab:red", s=35, zorder=3, label=f"DTW: {d:.4g}")
            axes[2].axvline(min_idx * dt, color="tab:orange", ls="--", lw=2)
            if i == 0:
                axes[2].legend(loc="upper left", frameon=True)
            axes[2].set_ylim(*ylim_dist)
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Dissimilarity (DTW)")

            fig.align_ylabels()
            fig.suptitle(f"Center={c}, t={c*dt:.3f}s, DTW={d:.6g}", y=0.98, fontsize=12)
            fig.savefig(outdir / f"phaseDetect{c}.png", dpi=130)
            plt.close(fig)

        gif_path = outdir / "template_match.gif"
        make_gif_from_pngs(outdir, gif_path, frame_duration_ms=120)

    return min_idx, min_dist, gif_path


# --------------------------- Entry Point --------------------------- #

if __name__ == "__main__":
    # Load waveform
    wave_tr = read_trace(WAVEFORM_PATH)
    wave_tr = preprocess(wave_tr)

    # Load/build template
    if TEMPLATE_PATH is not None:
        tmpl_tr = read_trace(TEMPLATE_PATH)
        tmpl_tr = preprocess(tmpl_tr)
        wave_tr, tmpl_tr = ensure_same_sampling(wave_tr, tmpl_tr)
    else:
        n = wave_tr.stats.npts
        dt = wave_tr.stats.delta
        assert 0 <= BEG < TER <= n, "BEG/TER out of bounds for waveform length."
        t0 = wave_tr.stats.starttime + BEG * dt
        t1 = wave_tr.stats.starttime + (TER - 1) * dt  # inclusive end for ObsPy trim
        tmpl_tr = wave_tr.copy().trim(t0, t1, pad=True, fill_value=0.0)
        desired = TER - BEG
        if tmpl_tr.stats.npts != desired:
            data = tmpl_tr.data.astype(np.float64)
            if data.size > desired:
                data = data[:desired]
            else:
                data = np.pad(data, (0, desired - data.size), mode="constant", constant_values=0.0)
            tmpl_tr = Trace(data=data, header=wave_tr.stats)
            tmpl_tr.stats.starttime = t0
        tmpl_tr = preprocess(tmpl_tr)

    # Run search; plot only if MAKE_GIF=True
    if MAKE_GIF:
        with tempfile.TemporaryDirectory(prefix="phase_detect_") as tmpdir:
            tmp = Path(tmpdir)
            frames_dir = tmp / "output_figures"
            best_idx, best_dist, gif_tmp = run_dtw_search(wave_tr, tmpl_tr, frames_dir, make_gif=True)
            shutil.copyfile(gif_tmp, PERSIST_GIF_PATH)
    else:
        # Fast mode: no temp dirs, no plotting
        best_idx, best_dist, _ = run_dtw_search(wave_tr, tmpl_tr, outdir=Path("."), make_gif=False)

    # Report concise results
    dt = wave_tr.stats.delta
    st = getattr(wave_tr.stats, "starttime", None)
    if MAKE_GIF:
        print(f"Saved GIF: {PERSIST_GIF_PATH}")
    else:
        print("GIF generation disabled (MAKE_GIF=False)")
    print(f"Best match index: {best_idx}")
    print(f"Best match time: {fmt_time_value(st, best_idx, dt)}")
    print(f"Minimum distance (DTW): {best_dist:.6g}")
