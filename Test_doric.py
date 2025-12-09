#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick QC for Doric test recordings WITHOUT TTLs, single ROI.

For each .doric file in DIR_PATH:
    * load ROI* from CAM1_EXC1 (ref), CAM1_EXC2 (green), CAM2_EXC3 (red)
    * align all signals to the CAM1_EXC2 (green) timebase
    * correct green & red (scale_ref + baseline + ΔF/F + z-score)
    * cut first and last 10 frames
    * save a 3-panel PDF:
        - top: raw signals + scaled reference (green & red)
        - middle: baseline-corrected ΔF/F (bcsig) for green & red
        - bottom: z-scored corrected traces (green & red)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

import doric as dr
import h5py

# ---------------------------------------------------------------------
# CONFIG – YOUR FOLDER
# ---------------------------------------------------------------------
DIR_PATH = r"E:\DA_comparison\Data\Iakovos_setup"

# Optional: add your custom env path if you need it
# sys.path.append(r"/Users/iakovos/opt/anaconda3/envs/pyiak/fipIak0616122")

# Matplotlib defaults
plt.rcParams["figure.figsize"] = [15, 5]
plt.rcParams["font.size"] = 14
rcParams["pdf.fonttype"] = 42
rcParams["font.family"] = "Arial"


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def scale_ref(sig, ref):
    """
    Scale reference channel to best fit signal (1st-order poly).

    Handles NaNs and nearly-constant reference traces gracefully.
    """
    sig = np.asarray(sig, dtype=float)
    ref = np.asarray(ref, dtype=float)

    # mask out NaN / inf
    mask = np.isfinite(sig) & np.isfinite(ref)
    if mask.sum() < 2:
        # not enough valid points – just return original ref
        return ref.copy()

    ref_m = ref[mask]
    sig_m = sig[mask]

    # if ref is (almost) constant, avoid polyfit crash
    if np.allclose(ref_m, ref_m[0]):
        if ref_m[0] != 0:
            scale = np.nanmean(sig_m) / ref_m[0]
            return ref * scale
        else:
            # both ref ~ 0 -> nothing to scale, return zeros
            return np.zeros_like(ref)

    # standard linear fit
    p = np.polyfit(ref_m, sig_m, 1)
    ref_scaled = np.polyval(p, ref)
    return ref_scaled


def fit_baseline(trace, window_size, percentile):
    """
    Rolling-percentile baseline with mirrored padding.
    window_size in *frames*, percentile e.g. 25/50/70.
    """
    trace = np.asarray(trace, dtype=float)
    expansion = window_size // 2

    pad_left = trace[:expansion][::-1]
    pad_right = trace[-expansion:][::-1]
    padded = np.concatenate([pad_left, trace, pad_right])

    s = pd.Series(padded)
    bl = s.rolling(window=window_size, center=True).agg(
        lambda w: np.percentile(w, percentile)
    )
    bl = bl.to_numpy()[expansion:-expansion]
    return bl


def correct_traceD(sig, ref):
    """
    Dopamine/GFAP-style correction:

        - scale reference to signal
        - ΔF/F = (sig - sref) / sref
        - subtract slow baseline (rolling percentile)
        - return z-scored corrected trace + intermediates

    Returns:
        stbcsig, bcsig, csig, sref,
        stbsig, stbBREF, stbbsref, bsig, bsref
    """
    sig = np.asarray(sig, dtype=float)
    ref = np.asarray(ref, dtype=float)

    sref = scale_ref(sig, ref)  # scale reference
    csig = (sig - sref) / sref  # ΔF/F trace

    baseline = fit_baseline(csig, 300, 50)
    baselineS = fit_baseline(sig, 300, 50)
    baselineREF = fit_baseline(sref, 300, 50)

    bcsig = csig - baseline  # ΔF/F minus baseline
    bsig = sig - baselineS
    bsref = sref - baselineREF

    stbcsig = stats.zscore(bcsig, nan_policy="omit")
    stbsig = stats.zscore(bsig, nan_policy="omit")
    stbBREF = stats.zscore(baselineREF, nan_policy="omit")
    stbbsref = stats.zscore(bsref, nan_policy="omit")

    return stbcsig, bcsig, csig, sref, stbsig, stbBREF, stbbsref, bsig, bsref


def align_to_green_time(time_green, signals_times_dict):
    """
    Interpolate all signals to the Green timebase.

    Parameters
    ----------
    time_green : 1D array
        Target timebase (e.g. TimeInGC).
    signals_times_dict : dict
        key -> (time_vector, signal_vector)

    Returns
    -------
    aligned_signals : dict
        key -> interpolated signal on time_green.
    """
    aligned_signals = {}
    time_green = np.asarray(time_green, dtype=float)

    for name, (time_vec, signal_vec) in signals_times_dict.items():
        time_vec = np.asarray(time_vec, dtype=float)
        signal_vec = np.asarray(signal_vec, dtype=float)
        aligned_signals[name] = np.interp(time_green, time_vec, signal_vec)

    return aligned_signals


def import_trace_single_roi(trace_file):
    """
    Read Doric 3-channel file for a SINGLE ROI (first ROI found).

    Returns:
        TimeInRF, TimeInGC, TimeInGR,
        SignalRef, SignalGreen, SignalRed
    """
    with h5py.File(trace_file, "r") as f:
        if "DataAcquisition" not in f:
            raise KeyError("No 'DataAcquisition' group in file (config file?)")

        bfpd = f["DataAcquisition"]["BFPD"]
        if "ROISignals" in bfpd:
            root = "ROISignals"
        elif "ROIs" in bfpd:
            root = "ROIs"
        else:
            raise KeyError(
                f"No 'ROISignals' or 'ROIs' group found in {trace_file!r}"
            )

        series = bfpd[root]["Series0001"]
        cams = list(series.keys())

        cam1_exc1 = next(
            k for k in cams if k.lower().startswith("cam1") and "exc1" in k.lower()
        )
        cam1_exc2 = next(
            k for k in cams if k.lower().startswith("cam1") and "exc2" in k.lower()
        )
        cam2_exc3 = next(
            k for k in cams if k.lower().startswith("cam2") and "exc3" in k.lower()
        )

        # pick FIRST ROI that exists in all 3 camera groups
        rois_exc1 = [k for k in series[cam1_exc1].keys() if k.lower().startswith("roi")]
        rois_exc2 = [k for k in series[cam1_exc2].keys() if k.lower().startswith("roi")]
        rois_exc3 = [k for k in series[cam2_exc3].keys() if k.lower().startswith("roi")]

        common_rois = sorted(set(rois_exc1) & set(rois_exc2) & set(rois_exc3))
        if not common_rois:
            raise KeyError("No common ROI* found across all cameras.")
        roi_name = common_rois[0]  # e.g. 'ROI01'

    base = ["DataAcquisition", "BFPD", root, "Series0001"]

    # Time vectors
    TimeInRF, _ = dr.h5read(trace_file, base + [cam1_exc1, "Time"])
    TimeInGC, _ = dr.h5read(trace_file, base + [cam1_exc2, "Time"])
    TimeInGR, _ = dr.h5read(trace_file, base + [cam2_exc3, "Time"])

    # Signal for that ROI
    SignalRef, _ = dr.h5read(trace_file, base + [cam1_exc1, roi_name])
    SignalGreen, _ = dr.h5read(trace_file, base + [cam1_exc2, roi_name])
    SignalRed, _ = dr.h5read(trace_file, base + [cam2_exc3, roi_name])

    return TimeInRF, TimeInGC, TimeInGR, SignalRef, SignalGreen, SignalRed


# ---------------------------------------------------------------------
# MAIN PER-FILE PROCESSING
# ---------------------------------------------------------------------
def process_doric_file(doric_path, out_dir=None):
    """Process one .doric file and save a 3-panel PDF."""
    if out_dir is None:
        out_dir = os.path.dirname(doric_path)

    base = os.path.splitext(os.path.basename(doric_path))[0]
    print(f"\n=== Processing {base} ===")

    # skip obvious config files
    if base.lower().startswith("config_"):
        print("  -> Skipping config file.")
        return

    (
        TimeInRF,
        TimeInGC,
        TimeInGR,
        SignalRef,
        SignalGreen,
        SignalRed,
    ) = import_trace_single_roi(doric_path)

    # Align signals to green timebase (TimeInGC)
    aligned = align_to_green_time(
        TimeInGC,
        {
            "Ref": (TimeInRF, SignalRef),
            "Green": (TimeInGC, SignalGreen),
            "Red": (TimeInGR, SignalRed),
        },
    )

    t = np.asarray(TimeInGC, dtype=float)
    Ref = aligned["Ref"]
    Green = aligned["Green"]
    Red = aligned["Red"]

    # Correct green & red using the same reference
    GreenCorr = correct_traceD(Green, Ref)
    RedCorr = correct_traceD(Red, Ref)

    # Unpack what we need
    G_z, G_bcsig, G_csig, G_sref, *_ = GreenCorr
    R_z, R_bcsig, R_csig, R_sref, *_ = RedCorr

    # Make sure everything is the same length
    all_arrays = [t, Green, Red, Ref, G_sref, R_sref, G_z, R_z, G_bcsig, R_bcsig]
    min_len = min(len(a) for a in all_arrays)

    # Cut first and last 10 frames
    CUT = 10
    if min_len <= 2 * CUT:
        print("  -> Not enough frames after trimming, skipping.")
        return

    start = CUT
    end = min_len - CUT

    t = t[start:end]
    Green = Green[start:end]
    Red = Red[start:end]
    Ref = Ref[start:end]
    G_sref = G_sref[start:end]
    R_sref = R_sref[start:end]
    G_z = G_z[start:end]
    R_z = R_z[start:end]
    G_bcsig = G_bcsig[start:end]
    R_bcsig = R_bcsig[start:end]

    # -----------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(base, fontsize=14)

    # ---- Top: raw signals + scaled reference ----
    ax0 = axes[0]
    ax0.plot(t, Green, color="tab:green", label="Green raw")
    ax0.plot(t, G_sref, color="tab:green", linestyle="--", alpha=0.7, label="Ref→Green scaled")

    ax0.plot(t, Red, color="tab:red", label="Red raw")
    ax0.plot(t, R_sref, color="tab:red", linestyle="--", alpha=0.7, label="Ref→Red scaled")

    ax0.set_ylabel("Signal (a.u.)")
    ax0.set_title("Raw signals and scaled reference (single ROI)")
    ax0.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- Middle: baseline-corrected ΔF/F (bcsig) ----
    ax1 = axes[1]
    ax1.plot(t, G_bcsig, color="tab:green", label="Green ΔF/F (baseline-corrected)")
    ax1.plot(t, R_bcsig, color="tab:red", label="Red ΔF/F (baseline-corrected)")

    ax1.set_ylabel("ΔF/F")
    ax1.set_title("Baseline-corrected ΔF/F")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # ---- Bottom: z-scored ΔF/F ----
    ax2 = axes[2]
    ax2.plot(t, G_z, color="tab:green", label="Green (z)")
    ax2.plot(t, R_z, color="tab:red", label="Red (z)")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Z-score (ΔF/F)")
    ax2.set_title("Baseline-corrected, z-scored traces")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, f"{base}_quickQC_singleROI.pdf")
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
def main():
    # In Spyder, %runfile passes its own options; we just ignore them
    dir_path = DIR_PATH

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    print(f"Searching for .doric files in:\n  {dir_path}")

    files = sorted(
        f for f in os.listdir(dir_path) if f.lower().endswith(".doric")
    )
    if not files:
        print("No .doric files found.")
        return

    for fname in files:
        fpath = os.path.join(dir_path, fname)
        try:
            process_doric_file(fpath, out_dir=dir_path)
        except Exception as e:
            print(f"[ERROR] Failed on {fname}: {e}")


if __name__ == "__main__":
    main()
