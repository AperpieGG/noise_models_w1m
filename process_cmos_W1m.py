#! /usr/bin/env python
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from calibration_images_W1m import load_calibration_masters, reduce_image
from utils_W1m import (camera_config, filter_science_filenames, get_location,
                       group_filenames_by_object_prefix,
                       wcs_phot, _detect_objects_sep, get_catalog,
                       extract_airmass_and_zp, get_light_travel_times)
import warnings
import logging
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning


# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

MAX_ALLOWED_PIXEL_SHIFT = 50
N_OBJECTS_LIMIT = 200
DEFOCUS = 0.0
AREA_MIN = 10
AREA_MAX = 200
SCALE_MIN = 4.5
SCALE_MAX = 5.5
DETECTION_SIGMA = 3
ZP_CLIP_SIGMA = 3

OK, TOO_FEW_OBJECTS, UNKNOWN = range(3)
WORKER_CONTEXT = {}


def setup_logging(filename='process.log'):
    """
    Configure logging once the script is actually running.
    """
    log_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path
               for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(getattr(handler, "_w1m_stream", False) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        stream_handler._w1m_stream = True
        logger.addHandler(stream_handler)

    return log_path


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
    return filter_science_filenames(directory, extra_excluded=("catalog_input",))


def catalog_column(catalog, *names):
    """
    Return the first matching catalog column from a list of accepted names.
    """
    available = {name.lower(): name for name in catalog.columns.names}
    for name in names:
        match = available.get(name.lower())
        if match is not None:
            return catalog[match]
    raise KeyError(
        f"None of the expected catalog columns {names} exist. "
        f"Available columns: {catalog.columns.names}"
    )


def arg_parse():
    parser = argparse.ArgumentParser(description="Run calibrated WCS photometry")
    parser.add_argument("--camera", type=str, default="QHY600")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of image photometry worker processes.")
    return parser.parse_args()


def init_worker(context):
    WORKER_CONTEXT.update(context)


def process_frame(filename):
    context = WORKER_CONTEXT
    directory = context["directory"]
    prefix = context["prefix"]
    aperture_radii = context["aperture_radii"]
    gain = context["gain"]
    ra_key = context["ra_key"]
    dec_key = context["dec_key"]
    masters = context["masters"]
    phot_cat = context["phot_cat"]
    site_location = context["site_location"]

    reduced_data, frame_hdr, _ = reduce_image(filename, *masters, site_location=site_location)
    frame_data = reduced_data

    airmass, zp = extract_airmass_and_zp(frame_hdr)

    frame_bg = sep.Background(frame_data)
    bg_rms = frame_bg.rms()
    frame_data_corr_no_bg = frame_data - frame_bg
    estimate_coord = SkyCoord(ra=frame_hdr[ra_key],
                              dec=frame_hdr[dec_key],
                              unit=(u.deg, u.deg))

    frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                        AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
    if len(frame_objects) < N_OBJECTS_LIMIT:
        return filename, None, (
            f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!"
        )

    phot_ra = catalog_column(phot_cat, "ra_deg_corr", "RA_CORR")
    phot_dec = catalog_column(phot_cat, "dec_deg_corr", "DEC_CORR")
    phot_tic = catalog_column(phot_cat, "tic_id", "TIC")
    phot_bp = catalog_column(phot_cat, "gaiabp", "BPmag")
    phot_rp = catalog_column(phot_cat, "gaiarp", "RPmag")

    phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_ra, phot_dec, 1)

    half_exptime = frame_hdr['EXPTIME'] / 2.
    time_isot = Time([frame_hdr['DATE-OBS'] for i in range(len(phot_x))],
                     format='isot', scale='utc', location=site_location)
    time_jd = Time(time_isot.jd, format='jd', scale='utc', location=site_location)
    time_jd = time_jd + half_exptime * u.second
    ltt_bary, ltt_helio = get_light_travel_times(phot_ra, phot_dec, time_jd)
    time_bary = time_jd.tdb + ltt_bary

    frame_ids = [filename for i in range(len(phot_x))]
    frame_preamble = Table([frame_ids, phot_cat['Tmag'], phot_tic,
                            phot_bp, phot_rp, time_jd.value, time_bary.value,
                            phot_x, phot_y,
                            [airmass] * len(phot_x), [zp] * len(phot_x)],
                           names=("frame_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                  "jd_bary", "x", "y", "airmass", "zp"))

    frame_phot = wcs_phot(frame_data, phot_x, phot_y, aperture_radii, frame_data_corr_no_bg, bg_rms,
                          gain=gain)
    frame_output = hstack([frame_preamble, frame_phot])
    if not isinstance(frame_output, Table):
        frame_output = Table(frame_output)

    message = (
        f"Finished photometry for {filename}; "
        f"coord keys {ra_key}/{dec_key}: {estimate_coord.to_string('decimal')}"
    )
    return filename, frame_output, message


def process_frames(prefix_filenames, context, workers):
    if workers <= 1:
        return [process_frame(filename) for filename in prefix_filenames]

    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(context,)) as executor:
            future_to_filename = {executor.submit(process_frame, filename): filename for filename in prefix_filenames}
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append((filename, None, f"Failed photometry for {filename}: {exc}"))
    except (OSError, PermissionError) as exc:
        logging.warning(f"Multiprocessing unavailable ({exc}); falling back to 1 worker.")
        return [process_frame(filename) for filename in prefix_filenames]
    results.sort(key=lambda item: prefix_filenames.index(item[0]))
    return results


def main():
    setup_logging()
    args = arg_parse()
    config = camera_config(args.camera)
    gain = config["gain"]
    aperture_radii = [config["phot_aperture"]]
    ra_key = config["ra_key"]
    dec_key = config["dec_key"]
    estimate_coord_radius = config["phot_estimate_radius_deg"] * u.deg
    workers = max(1, args.workers if args.workers is not None else config["workers"])
    site_location = get_location(config)

    # set directory for the current working directory
    directory = os.getcwd()
    logging.info(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logging.info(f"Number of files: {len(filenames)}")

    # Get prefixes for each set of images
    filenames_by_prefix = group_filenames_by_object_prefix(filenames, args.camera)
    prefixes = filenames_by_prefix.keys()
    logging.info(f"The prefixes are: {prefixes}")
    logging.info(f"Using {workers} image photometry worker(s).")

    logging.info("Loading calibration masters once.")
    masters = load_calibration_masters()

    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

        # Open the photometry file for the current prefix
        if os.path.exists(phot_output_filename):
            logging.info(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.")
            continue

        logging.info(f"Creating new photometry file for prefix {prefix}.")

        # Iterate over filenames with the current prefix
        prefix_filenames = filenames_by_prefix[prefix]
        phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
        logging.info(f"Found catalog with name {prefix}_catalog_input.fits")
        context = {
            "directory": directory,
            "prefix": prefix,
            "aperture_radii": aperture_radii,
            "gain": gain,
            "ra_key": ra_key,
            "dec_key": dec_key,
            "estimate_coord_radius": estimate_coord_radius,
            "masters": masters,
            "phot_cat": phot_cat,
            "site_location": site_location,
        }
        init_worker(context)
        frame_results = process_frames(prefix_filenames, context, workers)
        phot_tables = []
        for filename, frame_table, message in frame_results:
            logging.info(message)
            if frame_table is not None:
                phot_tables.append(frame_table)

        # Save the photometry for the current prefix
        if phot_tables:
            phot_table = vstack(phot_tables)
            phot_table.write(phot_output_filename, overwrite=True)
            logging.info(f"Saved photometry for prefix {prefix} to {phot_output_filename}")
        else:
            logging.info(f"No photometry data for prefix {prefix}.")

    logging.info("Done!")


if __name__ == "__main__":
    main()
