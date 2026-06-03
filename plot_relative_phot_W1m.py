#!/usr/bin/env python3
"""
Plot a relative-light-curve FITS table for one source ID.
"""
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.time import Time

from utils_W1m import plot_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot relative photometry for one numeric source ID."
    )
    parser.add_argument(
        "source_id",
        type=int,
        help="Numeric source ID. For Gaia catalogs this is the Gaia source ID; for TIC catalogs this is the TIC ID.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory containing rel_phot_*.fits, or an explicit rel_phot FITS file.",
    )
    parser.add_argument(
        "--rel-file",
        type=str,
        default=None,
        help="Explicit rel_phot_*.fits file. Overrides the positional path.",
    )
    parser.add_argument(
        "--phot-file",
        type=str,
        default=None,
        help="Matching phot_*.fits file used to add magnitude to the plot title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot filename. Defaults beside the rel_phot file.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "png"),
        default="pdf",
        help="Output format when --output is not supplied.",
    )
    parser.add_argument(
        "--time",
        choices=("relative", "jd"),
        default="relative",
        help="Plot time as relative hours from the first point or as BJD.",
    )
    return parser.parse_args()


def find_rel_file(path):
    if os.path.isfile(path):
        return path

    matches = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.startswith("rel_phot_") and name.endswith(".fits")
    )
    if not matches:
        raise FileNotFoundError(f"No rel_phot_*.fits file found in {path}.")
    if len(matches) > 1:
        print(f"Found {len(matches)} rel_phot files; using {matches[0]}")
    return matches[0]


def output_name(rel_file, source_id, output, output_format):
    if output:
        return output
    root, _ = os.path.splitext(os.path.basename(rel_file))
    return os.path.join(os.path.dirname(rel_file), f"{root}_{source_id}.{output_format}")


def matching_phot_file(rel_file):
    directory = os.path.dirname(rel_file) or "."
    basename = os.path.basename(rel_file)
    if basename.startswith("rel_phot_"):
        candidate = os.path.join(directory, "phot_" + basename[len("rel_phot_"):])
        if os.path.exists(candidate):
            return candidate
    return None


def source_rows(table, source_id):
    if "tic_id" not in table.colnames:
        raise KeyError("Relative photometry table is missing the 'tic_id' source-ID column.")
    rows = table[table["tic_id"] == source_id]
    if len(rows) == 0:
        available = np.unique(table["tic_id"])
        preview = ", ".join(str(int(value)) for value in available[:10])
        raise ValueError(
            f"Source ID {source_id} was not found. "
            f"First available IDs: {preview}"
        )
    return rows[np.argsort(rows["jd_bary"])]


def source_magnitude(source_id, phot_file):
    if phot_file is None or not os.path.exists(phot_file):
        return None
    table = Table.read(phot_file)
    if "tic_id" not in table.colnames:
        return None
    rows = table[table["tic_id"] == source_id]
    if len(rows) == 0:
        return None

    mag_system = table.meta.get("MAGSYS", "TESS")
    if "MAG" in rows.colnames:
        mag_column = "MAG"
    else:
        mag_column = "Gmag" if mag_system == "G" and "Gmag" in rows.colnames else "Tmag"
    if mag_column not in rows.colnames:
        return None

    label = "G" if mag_system == "G" else "T"
    return label, float(rows[mag_column][0])


def relative_time_hours(jd_bary):
    return (jd_bary - jd_bary[0]) * 24.0


def plot_relative_lightcurve(rows, source_id, time_mode, magnitude=None):
    plot_images()
    time = np.asarray(rows["jd_bary"], dtype=float)
    flux = np.asarray(rows["relative_flux"], dtype=float)
    flux_err = np.asarray(rows["relative_flux_err"], dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    if len(time) == 0:
        raise ValueError(f"Source ID {source_id} has no finite relative photometry points.")

    if time_mode == "relative":
        x = relative_time_hours(time)
        xlabel = f"Hours from {Time(time[0], format='jd', scale='tdb').isot}"
    else:
        x = time
        xlabel = "BJD TDB"

    median = np.nanmedian(flux)
    rms_ppm = np.nanstd(flux / median) * 1e6 if median != 0 else np.nan
    aperture = rows["aperture"][0] if "aperture" in rows.colnames else "unknown"
    n_comps = rows["n_comps"][0] if "n_comps" in rows.colnames else "unknown"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, flux, yerr=flux_err, fmt=".", markersize=3,
                elinewidth=0.5, alpha=0.75)
    ax.axhline(median, color="tab:red", linewidth=1,
               label=f"median={median:.6f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative flux")
    mag_text = ""
    if magnitude is not None:
        mag_text = f" | {magnitude[0]}={magnitude[1]:.3f}"
    ax.set_title(
        f"Source {source_id}{mag_text} | aperture={aperture} pix | "
        f"n={len(time)} | RMS={rms_ppm:.0f} ppm | comps={n_comps}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    rel_file = args.rel_file or find_rel_file(args.path)
    phot_file = args.phot_file or matching_phot_file(rel_file)
    table = Table.read(rel_file)
    rows = source_rows(table, args.source_id)
    magnitude = source_magnitude(args.source_id, phot_file)
    fig = plot_relative_lightcurve(rows, args.source_id, args.time, magnitude)
    output = output_name(rel_file, args.source_id, args.output, args.format)
    fig.savefig(output, format=os.path.splitext(output)[1].lstrip(".") or args.format,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved relative photometry plot to {output}")


if __name__ == "__main__":
    main()
