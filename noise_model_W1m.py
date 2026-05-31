#!/usr/bin/env python3
"""
Create a W1m noise-model JSON from combined relative photometry.
"""

import argparse
import json
import os

import numpy as np
from astropy.table import Table

from utils_W1m import bin_time_flux_error, camera_config, noise_sources


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy scalar and array values."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a noise-model JSON from rel_phot_*.fits.")
    parser.add_argument("directory", nargs="?", default=".",
                        help="Directory containing rel_phot_*.fits and phot_*.fits files.")
    parser.add_argument("--rel-file", type=str, default=None, help="Combined rel_phot_*.fits file.")
    parser.add_argument("--phot-file", type=str, default=None, help="Matching phot_*.fits file.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename.")
    parser.add_argument("--bin-size", type=int, default=1, help="Number of points to bin for RMS calculations.")
    parser.add_argument("--cam", type=str, default="qhy600.json", help="Camera config name/path.")
    parser.add_argument("--aperture", type=int, default=None, help="Photometry aperture to use.")
    return parser.parse_args()


def find_first_file(prefix, directory="."):
    matches = sorted(name for name in os.listdir(directory) if name.startswith(prefix) and name.endswith(".fits"))
    if not matches:
        return None
    return os.path.join(directory, matches[0])


def default_rel_file(directory="."):
    rel_file = find_first_file("rel_phot_", directory)
    if rel_file is None:
        raise FileNotFoundError(f"No rel_phot_*.fits file found in {directory}.")
    return rel_file


def matching_phot_file(rel_file, directory="."):
    basename = os.path.basename(rel_file)
    if basename.startswith("rel_phot_"):
        candidate = os.path.join(directory, "phot_" + basename[len("rel_phot_"):])
        if os.path.exists(candidate):
            return candidate
    return find_first_file("phot_", directory)


def field_name_from_rel_file(rel_file):
    basename = os.path.basename(rel_file)
    root, _ = os.path.splitext(basename)
    if root.startswith("rel_phot_"):
        return root[len("rel_phot_"):]
    return root


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def masked_float_or_nan(value):
    try:
        if np.ma.is_masked(value):
            return np.nan
    except TypeError:
        pass
    return float(value)


def rms_for_star(star_rows, bin_size):
    order = np.argsort(star_rows["jd_bary"])
    star_rows = star_rows[order]
    time = np.asarray(star_rows["jd_bary"], dtype=float)
    flux = np.asarray(star_rows["relative_flux"], dtype=float)
    flux_err = np.asarray(star_rows["relative_flux_err"], dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux)
    if "relative_flux_err" in star_rows.colnames:
        valid &= np.isfinite(flux_err)
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]

    if len(flux) == 0:
        return np.nan
    if bin_size > 1:
        if len(flux) < bin_size:
            return np.nan
        _, flux, _ = bin_time_flux_error(time, flux, flux_err, bin_fact=bin_size)
    return float(np.nanstd(flux) * 1e6)


def phot_metadata(phot_table, tic_id, aperture):
    rows = phot_table[phot_table["tic_id"] == tic_id]
    if len(rows) == 0:
        return None

    flux_col = f"flux_{aperture}"
    sky_col = f"flux_w_sky_{aperture}"
    required = {"Tmag", "gaiabp", "gaiarp", "airmass", "zp", flux_col, sky_col}
    missing = sorted(required - set(rows.colnames))
    if missing:
        raise KeyError(f"Photometry table is missing required columns for aperture {aperture}: {missing}")

    sky = np.asarray(rows[sky_col], dtype=float) - np.asarray(rows[flux_col], dtype=float)
    color = masked_float_or_nan(rows["gaiabp"][0]) - masked_float_or_nan(rows["gaiarp"][0])
    mag_column = "MAG" if "MAG" in rows.colnames else "Tmag"
    return {
        "MAG": float(rows[mag_column][0]),
        "COLOR": color,
        "sky_median": float(np.nanmedian(sky)),
        "sky_mean": float(np.nanmean(sky)),
        "airmass": np.asarray(rows["airmass"], dtype=float),
        "zp": np.asarray(rows["zp"], dtype=float),
    }


def build_noise_model(rel_table, phot_table, aperture, bin_size, config):
    tic_ids = sorted(int(tic_id) for tic_id in np.unique(rel_table["tic_id"]))

    rms_list = []
    magnitudes = []
    colors = []
    output_tic_ids = []
    sky_values = []
    all_airmass = []
    all_zp = []

    for tic_id in tic_ids:
        star_rows = rel_table[rel_table["tic_id"] == tic_id]
        metadata = phot_metadata(phot_table, tic_id, aperture)
        if metadata is None:
            continue

        rms = rms_for_star(star_rows, bin_size)
        if not np.isfinite(rms):
            continue

        output_tic_ids.append(tic_id)
        rms_list.append(rms)
        magnitudes.append(metadata["MAG"])
        colors.append(metadata["COLOR"])
        sky_values.append(metadata["sky_mean"])
        all_airmass.extend(metadata["airmass"][np.isfinite(metadata["airmass"])])
        all_zp.extend(metadata["zp"][np.isfinite(metadata["zp"])])

    if not output_tic_ids:
        raise ValueError("No valid TIC IDs were available for the noise model.")
    if not all_zp:
        raise ValueError("No valid zero points found in the photometry file.")
    if not all_airmass:
        raise ValueError("No valid airmass values found in the photometry file.")

    synthetic_mag, psn, sn, rn, dcn, scint, rns = noise_sources(
        np.asarray(sky_values, dtype=float),
        bin_size,
        np.asarray(all_airmass, dtype=float),
        np.asarray(all_zp, dtype=float),
        aperture,
        config["read_noise"],
        config["dark_current"],
        config["exposure"],
        config["gain"],
        config=config,
    )

    result = {
        "TIC_IDs": output_tic_ids,
        "RMS_list": rms_list,
        "magnitude_list": magnitudes,
        "COLOR": colors,
        "synthetic_mag": synthetic_mag,
        "photon_shot_noise": psn,
        "sky_noise": sn,
        "read_noise": rn,
        "dc_noise": dcn,
        "scintillation_noise": scint,
        "RNS": rns,
        "sky_mean_per_star": sky_values,
        "mean_zp": float(np.mean(all_zp)),
        "mean_airmass": float(np.mean(all_airmass)),
    }
    magnitude_key = "Gmag_list" if config["catalog"] == "gaia_dr3" else "Tmag_list"
    result[magnitude_key] = magnitudes
    return result


def main():
    args = parse_args()
    config = camera_config(args.cam)
    aperture = args.aperture if args.aperture is not None else config["phot_aperture"]
    directory = os.path.abspath(args.directory)

    rel_file = args.rel_file or default_rel_file(directory)
    phot_file = args.phot_file or matching_phot_file(rel_file, directory)
    if phot_file is None:
        raise FileNotFoundError("No matching phot_*.fits file found. Pass --phot-file explicitly.")

    rel_table = Table.read(rel_file)
    phot_table = Table.read(phot_file)
    camera_name = config["name"]
    output = args.output or os.path.join(
        directory, f"noise_model_{field_name_from_rel_file(rel_file)}_{camera_name}_bin{args.bin_size}.json"
    )

    print(f"Relative photometry file: {rel_file}")
    print(f"Photometry file: {phot_file}")
    print(f"Camera: {camera_name}, aperture: {aperture}, bin size: {args.bin_size}")

    data = build_noise_model(rel_table, phot_table, aperture, args.bin_size, config)
    data["rel_file"] = rel_file
    data["phot_file"] = phot_file
    data["camera"] = camera_name
    data["camera_config"] = config["config_path"]
    data["catalog"] = config["catalog"]
    data["magnitude_system"] = "G" if config["catalog"] == "gaia_dr3" else "TESS"
    data["location"] = config["location"]
    data["scintillation"] = config["scintillation"]
    data["aperture"] = aperture
    data["bin_size"] = args.bin_size

    with open(output, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, cls=NumpyEncoder)

    print(f"Noise model results saved to {output}")


if __name__ == "__main__":
    main()
