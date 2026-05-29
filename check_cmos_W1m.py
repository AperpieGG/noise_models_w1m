#!/usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""

from donuts import Donuts
import argparse
from astropy.io import fits
from astropy.time import Time
import numpy as np
import os
import shutil
import logging
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils_W1m import (
    filter_science_filenames,
    group_filenames_by_object_prefix,
    is_astrometrically_solved, plot_images
)

plot_images()

# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level
log_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(log_dir, exist_ok=True)

# Create file handler
file_handler = logging.FileHandler(os.path.join(log_dir, 'donuts.log'))
file_handler.setLevel(logging.INFO)  # Set the level for the file handler

# Create stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set the level for the stream handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)


def filter_filenames(directory):
    """
    Filter filenames based on specific criteria.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        Filtered list of filenames.
    """
    return filter_science_filenames(directory)


def check_headers(directory, filenames):
    """
    Check headers of all files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    filenames : list of str
    """
    no_wcs = os.path.join(directory, 'no_wcs')
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for file in filenames:
        try:
            with fits.open(os.path.join(directory, file)) as hdulist:
                header = hdulist[0].header
                if not is_astrometrically_solved(header, require_zp=False):
                    logger.warning(f"{file} does not have CTYPE1 and/or CTYPE2 in the header. "
                                   f"Moving to 'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    shutil.move(os.path.join(directory, file), new_path)

        except Exception as e:
            logger.error(f"Error checking header for {file}: {e}")

    logger.info(f"Done checking headers, number of files without CTYPE1 and/or CTYPE2: {len(os.listdir(no_wcs))}")


def check_donuts(file_groups, filenames):
    """
    Check donuts for each image in the directory.

    Parameters
    ----------
    file_groups : list of str
        Directory containing the images.
    filenames : list of str
        List of filenames.
    """
    # Assuming Donuts class and measure_shift function are defined elsewhere
    shift_rows = []

    for file_group in file_groups:
        if len(file_group) < 2:
            continue

        # Using the first filename as the reference image
        all_sx = []
        all_sy = []

        reference_image = file_group[0]
        logger.info(f"Reference image: {reference_image}")

        # Assuming Donuts class and measure_shift function are defined elsewhere
        d = Donuts(reference_image)

        for i in file_group[1:]:
            shift = d.measure_shift(i)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            logger.info(f'{i} shift X: {sx} Y: {sy}')

            with fits.open(i) as hdulist:
                date_obs = hdulist[0].header.get('DATE-OBS')

            shift_rows.append({
                "filename": i,
                "date_obs": date_obs,
                "shift_x": sx,
                "shift_y": sy,
            })

            all_sx.append(sx)
            all_sy.append(sy)

            if abs(sx) >= 1 or abs(sy) >= 1:
                logger.warning(f'{i} image shift too big X: {sx} Y: {sy}')
                if not os.path.exists('failed_donuts'):
                    os.mkdir('failed_donuts')
                logger.info(f'Moving {i} to failed_donuts/')
                shutil.move(i, os.path.join('failed_donuts', os.path.basename(i)))

        # Compute scatter (std) after all shifts
        if all_sx and all_sy:
            std_x = np.std(all_sx)
            std_y = np.std(all_sy)
            logger.info(f"Scatter of X shifts (std): {std_x:.3f} pixels")
            logger.info(f"Scatter of Y shifts (std): {std_y:.3f} pixels")
        else:
            logger.info("No shifts measured; scatter cannot be computed.")

    plot_shift_rows(shift_rows)


def plot_shift_rows(shift_rows):
    """
    Save a plot of Donuts X/Y shifts versus observation time.
    """
    if not shift_rows:
        logger.info("No Donuts shifts available for plotting.")
        return

    plot_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "donuts_pixel_shifts.png")

    times = []
    labels = []
    for index, row in enumerate(shift_rows):
        if row["date_obs"]:
            try:
                times.append(Time(row["date_obs"], format="isot").datetime)
                labels.append(row["date_obs"])
                continue
            except ValueError:
                pass
        times.append(index)
        labels.append(row["filename"])

    shift_x = [row["shift_x"] for row in shift_rows]
    shift_y = [row["shift_y"] for row in shift_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(times, shift_x, marker="o", color="tab:blue", label="X shift")
    ax.plot(times, shift_y, marker="o", color="tab:red", label="Y shift")
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.axhline(-0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pixel shift")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()

    if all(isinstance(t, int) for t in times):
        ax.set_xticks(times)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Donuts pixel shift plot to {plot_path}")


def arg_parse():
    parser = argparse.ArgumentParser(description="Check WCS headers and Donuts image shifts")
    parser.add_argument("--camera", type=str, default="QHY600")
    return parser.parse_args()


def main():
    args = arg_parse()

    # set directory for working
    directory = os.getcwd()
    logger.info(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logger.info(f"Number of files: {len(filenames)}")

    # Iterate over each filename to get the prefix
    prefix_groups = group_filenames_by_object_prefix(filenames, args.camera)
    prefixes = set(prefix_groups)
    logger.info(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = list(prefix_groups.values())

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(prefix_filenames, filenames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
