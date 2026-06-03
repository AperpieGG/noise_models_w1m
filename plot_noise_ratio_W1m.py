#!/usr/bin/env python3
"""
Plot measured RMS divided by total model noise for each source.
"""
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from utils_W1m import plot_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot measured/model total-noise ratio from a noise-model JSON."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="noise_model_*.json file produced by noise_model_W1m.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF filename. Defaults beside the JSON file.",
    )
    return parser.parse_args()


def load_noise_model(json_file):
    with open(json_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def magnitude_array(data):
    if "magnitude_list" in data:
        return np.asarray(data["magnitude_list"], dtype=float)
    if data.get("magnitude_system") == "G" and "Gmag_list" in data:
        return np.asarray(data["Gmag_list"], dtype=float)
    return np.asarray(data["Tmag_list"], dtype=float)


def model_noise_at_star_magnitudes(data, magnitudes):
    synthetic_mag = np.asarray(data["synthetic_mag"], dtype=float)
    total_noise = np.asarray(data["RNS"], dtype=float)
    finite = np.isfinite(synthetic_mag) & np.isfinite(total_noise) & (total_noise > 0)
    synthetic_mag = synthetic_mag[finite]
    total_noise = total_noise[finite]
    if len(synthetic_mag) < 2:
        raise ValueError("Noise model does not contain enough finite model points.")

    order = np.argsort(synthetic_mag)
    return np.interp(magnitudes, synthetic_mag[order], total_noise[order],
                     left=np.nan, right=np.nan)


def output_name(json_file, output):
    if output:
        return output
    root, _ = os.path.splitext(os.path.basename(json_file))
    return os.path.join(os.path.dirname(json_file), f"{root}_ratio.pdf")


def plot_ratio(data):
    plot_images()
    magnitudes = magnitude_array(data)
    measured_rms = np.asarray(data["RMS_list"], dtype=float)
    colors = np.asarray(data.get("COLOR", np.full(len(magnitudes), np.nan)), dtype=float)
    model_rms = model_noise_at_star_magnitudes(data, magnitudes)
    ratio = measured_rms / model_rms

    valid = (
        np.isfinite(magnitudes)
        & np.isfinite(measured_rms)
        & np.isfinite(model_rms)
        & np.isfinite(ratio)
        & (measured_rms > 0)
        & (model_rms > 0)
    )
    if not np.any(valid):
        raise ValueError("No finite measured/model ratios could be plotted.")

    magnitude_system = data.get("magnitude_system", "TESS")
    xlabel = "Gaia G Magnitude" if magnitude_system == "G" else "TESS Magnitude"

    fig, ax = plt.subplots(figsize=(8.5, 6))
    if np.any(np.isfinite(colors[valid])):
        scatter = ax.scatter(
            magnitudes[valid],
            ratio[valid],
            c=colors[valid],
            cmap="coolwarm",
            vmin=0.5,
            vmax=1.5,
            s=18,
            alpha=0.8,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r"$\mathrm{G_{BP} - G_{RP}}$")
    else:
        ax.scatter(magnitudes[valid], ratio[valid], s=18, alpha=0.8)

    ax.axhline(1.0, color="black", linewidth=1.2, label="measured = model")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Measured RMS / total model noise")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.invert_xaxis()

    finite_ratio = ratio[valid]
    p_lo, p_hi = np.nanpercentile(finite_ratio, [1, 99])
    if np.isfinite(p_lo) and np.isfinite(p_hi) and p_lo > 0 and p_hi > p_lo:
        ax.set_ylim(max(p_lo / 1.5, 0.05), p_hi * 1.5)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    data = load_noise_model(args.json_file)
    fig = plot_ratio(data)
    output_path = output_name(args.json_file, args.output)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved noise-ratio plot to {output_path}")


if __name__ == "__main__":
    main()
