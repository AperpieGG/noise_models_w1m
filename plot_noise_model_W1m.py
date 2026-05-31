#!/usr/bin/env python
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import json
import numpy as np
from utils_W1m import plot_images


def load_rms_mags_data(filename):
    """
    Load RMS and magnitude data from JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def mask_outliers_by_model(magnitude_list, RMS_list, color_list, synthetic_mag, RNS, deviation_factor=2):
    """
    Mask stars that have an RMS significantly higher than the model.
    Args:
        magnitude_list (list): List of magnitudes.
        RMS_list (list): List of RMS values.
        color_list (list): List of color values.
        synthetic_mag (list): Synthetic magnitude values for the model.
        RNS (list): Model noise values (RMS as a function of magnitude).
        deviation_factor (float): Factor to define significant deviation from model.

    Returns:
        masked_indices (list): Indices of stars that deviate from the model.
    """
    # Interpolate model RMS values to match the measured magnitudes.
    model_rms_interp = np.interp(magnitude_list, synthetic_mag, RNS)
    masked_indices = [i for i, (rms, model_rms) in enumerate(zip(RMS_list, model_rms_interp))
                      if rms > model_rms * deviation_factor]

    return masked_indices


def plot_noise_model(data):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    RMS_list = np.asarray(data['RMS_list'], dtype=float)
    magnitude_list = np.asarray(data.get('magnitude_list', data.get('Tmag_list')), dtype=float)
    magnitude_system = data.get('magnitude_system', 'TESS')
    color_list = np.asarray(data['COLOR'], dtype=float)
    synthetic_mag = np.asarray(data['synthetic_mag'], dtype=float)
    RNS = np.asarray(data['RNS'], dtype=float)
    photon_shot_noise = np.asarray(data['photon_shot_noise'], dtype=float)
    read_noise = np.asarray(data['read_noise'], dtype=float)
    dc_noise = np.asarray(data['dc_noise'], dtype=float)
    sky_noise = np.asarray(data['sky_noise'], dtype=float)
    scintillation_noise = np.asarray(data.get('scintillation_noise', data.get('N')), dtype=float)
    print(f'The average scintillation noise is: {np.nanmean(scintillation_noise)}')

    # Filter out stars with missing color information
    valid = np.isfinite(magnitude_list) & np.isfinite(RMS_list) & np.isfinite(color_list) & (RMS_list > 0)
    total_mags = magnitude_list[valid]
    total_RMS = RMS_list[valid]
    total_colors = color_list[valid]

    # Verify sizes match
    if len(total_mags) != len(total_RMS) or len(total_mags) != len(total_colors):
        print(f'The length of total_mags is {len(total_mags)}')
        print(f'The length of total_RMS is {len(total_RMS)}')
        print(f'The length of total_colors is {len(total_colors)}')
        raise ValueError("Mismatch in sizes: total_mags, total_RMS, and total_colors should be the same length.")

    # Scatter plot with remaining stars
    scatter = ax.scatter(total_mags, total_RMS, c=total_colors, cmap='coolwarm', vmin=0.5, vmax=1.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\mathrm{G_{BP} - G_{RP}}$')

    # Plot various noise sources
    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * np.nanmean(scintillation_noise),
            color='orange', label='scintillation noise', linestyle='--')

    # Plot formatting
    ax.set_xlabel('Gaia G Magnitude' if magnitude_system == 'G' else 'TESS Magnitude')
    ax.set_ylabel('RMS (ppm)')
    ax.set_yscale('log')
    ax.set_xlim(7.5, 14)
    ax.set_ylim(1000, 100000)
    ax.invert_xaxis()
    plt.legend(loc='best')
    plt.tight_layout()
    return fig


def output_name(json_file, output):
    if output:
        return output
    root, _ = os.path.splitext(os.path.basename(json_file))
    return os.path.join(os.path.dirname(json_file), f"{root}.png")


def main(json_file, output):
    # Set plot parameters
    plot_images()
    # Load RMS and magnitude data from JSON file
    data = load_rms_mags_data(json_file)

    # Plot RMS vs magnitudes
    fig = plot_noise_model(data)
    output_path = output_name(json_file, output)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved noise model plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMS vs Magnitudes')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing RMS and magnitude data')
    parser.add_argument('--output', type=str, default=None, help='Output PNG filename')
    args = parser.parse_args()

    # Run main function
    main(args.json_file, args.output)
