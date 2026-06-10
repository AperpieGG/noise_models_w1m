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
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils_W1m import (
    camera_config,
    filter_science_filenames,
    group_filenames_by_object_prefix,
    is_astrometrically_solved,
    plot_frame_diagnostics,
    plot_images,
    update_frame_diagnostics,
)

plot_images()

# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)

DONUTS_WORKER = None
MAX_ALLOWED_PIXEL_SHIFT = 2.0


def log_path(filename):
    """
    Return the current pipeline log path without creating files.
    """
    log_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    return os.path.join(log_dir, filename)


def setup_logging(filename):
    """
    Configure logging once the script is actually running.
    """
    path = log_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == path
               for handler in logger.handlers):
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(getattr(handler, "_w1m_stream", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        stream_handler._w1m_stream = True
        logger.addHandler(stream_handler)

    return path


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


def header_status(task):
    """
    Return whether a FITS file has the required WCS header cards.
    """
    directory, file = task
    try:
        with fits.open(os.path.join(directory, file)) as hdulist:
            header = hdulist[0].header
            return file, is_astrometrically_solved(header, require_zp=False), None
    except Exception as exc:
        return file, False, str(exc)


def check_headers(directory, filenames, workers=1):
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

    tasks = [(directory, file) for file in filenames]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(header_status, tasks))
    else:
        results = [header_status(task) for task in tasks]

    for file, has_wcs, error in results:
        if error:
            logger.error(f"Error checking header for {file}: {error}")
            continue
        if not has_wcs:
            logger.warning(f"{file} does not have CTYPE1 and/or CTYPE2 in the header. "
                           f"Moving to 'no_wcs' directory.")
            new_path = os.path.join(no_wcs, file)
            shutil.move(os.path.join(directory, file), new_path)

    logger.info(f"Done checking headers, number of files without CTYPE1 and/or CTYPE2: {len(os.listdir(no_wcs))}")


def init_donuts_worker(reference_image):
    """
    Create one Donuts reference matcher per worker process.
    """
    global DONUTS_WORKER
    DONUTS_WORKER = Donuts(reference_image)


def measure_donuts_shift(filename):
    """
    Measure one image shift against the worker's reference image.
    """
    try:
        shift = DONUTS_WORKER.measure_shift(filename)
        sx = round(shift.x.value, 2)
        sy = round(shift.y.value, 2)
        with fits.open(filename) as hdulist:
            date_obs = hdulist[0].header.get('DATE-OBS')
        return {
            "filename": filename,
            "date_obs": date_obs,
            "shift_x": sx,
            "shift_y": sy,
            "error": None,
        }
    except Exception as exc:
        return {
            "filename": filename,
            "date_obs": None,
            "shift_x": np.nan,
            "shift_y": np.nan,
            "error": str(exc),
        }


def measure_group_shifts(reference_image, files_to_check, workers):
    """
    Measure all files in one prefix group against the same reference image.
    """
    if workers <= 1:
        init_donuts_worker(reference_image)
        return [measure_donuts_shift(filename) for filename in files_to_check]

    results = []
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_donuts_worker,
            initargs=(reference_image,),
        ) as executor:
            future_to_filename = {
                executor.submit(measure_donuts_shift, filename): filename
                for filename in files_to_check
            }
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    results.append(future.result())
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    results.append({
                        "filename": filename,
                        "date_obs": None,
                        "shift_x": np.nan,
                        "shift_y": np.nan,
                        "error": str(exc),
                    })
    except BrokenProcessPool:
        logger.error(
            "Donuts worker process crashed while using %d workers. "
            "Retrying this group with one worker.",
            workers,
        )
        init_donuts_worker(reference_image)
        results = [measure_donuts_shift(filename) for filename in files_to_check]

    results.sort(key=lambda row: files_to_check.index(row["filename"]))
    return results


def check_donuts(file_groups, workers=1):
    """
    Check donuts for each image in the directory.

    Parameters
    ----------
    file_groups : list of str
        Directory containing the images.
    workers : int
        Number of worker processes to use.
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
        with fits.open(reference_image) as hdulist:
            reference_date_obs = hdulist[0].header.get("DATE-OBS")
        reference_row = {
            "filename": reference_image,
            "date_obs": reference_date_obs,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "error": None,
        }
        shift_rows.append(reference_row)
        update_frame_diagnostics(
            reference_image,
            date_obs=reference_date_obs,
            shift_x=0.0,
            shift_y=0.0,
            shift_r=0.0,
        )

        group_shift_rows = measure_group_shifts(reference_image, file_group[1:], workers)

        for row in group_shift_rows:
            i = row["filename"]
            if row["error"]:
                logger.error(f"Error measuring Donuts shift for {i}: {row['error']}")
                continue

            sx = row["shift_x"]
            sy = row["shift_y"]
            logger.info(f'{i} shift X: {sx} Y: {sy}')

            shift_rows.append(row)
            update_frame_diagnostics(
                i,
                date_obs=row["date_obs"],
                shift_x=sx,
                shift_y=sy,
                shift_r=np.hypot(sx, sy),
            )

            all_sx.append(sx)
            all_sy.append(sy)

            if abs(sx) >= MAX_ALLOWED_PIXEL_SHIFT or abs(sy) >= MAX_ALLOWED_PIXEL_SHIFT:
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
    plot_frame_diagnostics()


def plot_shift_rows(shift_rows):
    """
    Save a plot of Donuts X/Y shifts versus observation time.
    """
    if not shift_rows:
        logger.info("No Donuts shifts available for plotting.")
        return

    plot_images()
    plot_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "donuts_pixel_shifts.pdf")

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
    ax.axhline(1, color="1", linestyle="--", linewidth=1)
    ax.axhline(-1, color="1", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pixel shift")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()

    if all(isinstance(t, int) for t in times):
        ax.set_xticks(times)
        ax.set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Donuts pixel shift plot to {plot_path}")


def arg_parse():
    parser = argparse.ArgumentParser(description="Check WCS headers and Donuts image shifts")
    parser.add_argument("--camera", type=str, default="QHY600")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes for header and Donuts checks.")
    return parser.parse_args()


def main():
    args = arg_parse()
    config = camera_config(args.camera)
    workers = max(1, args.workers if args.workers is not None else config["workers"])

    log_file_path = log_path('donuts.log')
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        print(f"Donuts/check log already exists and is not empty: {log_file_path}. Skipping check_cmos_W1m.py.")
        return
    setup_logging('donuts.log')

    # set directory for working
    directory = os.getcwd()
    logger.info(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logger.info(f"Number of files: {len(filenames)}")
    logger.info(f"Using {workers} check worker(s).")

    # Iterate over each filename to get the prefix
    prefix_groups = group_filenames_by_object_prefix(filenames, args.camera)
    prefixes = set(prefix_groups)
    logger.info(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = list(prefix_groups.values())

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames, workers=workers)

    filenames = filter_filenames(directory)
    prefix_groups = group_filenames_by_object_prefix(filenames, args.camera)
    prefix_filenames = list(prefix_groups.values())

    # Check donuts for each group
    check_donuts(prefix_filenames, workers=workers)

    logger.info("Done.")


if __name__ == "__main__":
    main()
