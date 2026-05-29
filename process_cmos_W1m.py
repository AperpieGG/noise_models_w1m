#! /usr/bin/env python
import argparse
import os

import numpy as np

from calibration_images_W1m import reduce_images
from utils_W1m import (filter_science_filenames, get_location,
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
log_dir = os.environ.get("PIPELINE_LOG_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(log_dir, exist_ok=True)

# Create file handler
file_handler = logging.FileHandler(os.path.join(log_dir, 'process.log'))
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
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

GAIN = 1.131
MAX_ALLOWED_PIXEL_SHIFT = 50
N_OBJECTS_LIMIT = 200
APERTURE_RADII = [5]
DEFOCUS = 0.0
AREA_MIN = 10
AREA_MAX = 200
SCALE_MIN = 4.5
SCALE_MAX = 5.5
DETECTION_SIGMA = 3
ZP_CLIP_SIGMA = 3

OK, TOO_FEW_OBJECTS, UNKNOWN = range(3)


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
    return parser.parse_args()


def main():
    args = arg_parse()

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

    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

        # Open the photometry file for the current prefix
        if os.path.exists(phot_output_filename):
            logging.info(f"Photometry file for prefix {prefix} already exists, skipping to the next prefix.")
            continue

        logging.info(f"Creating new photometry file for prefix {prefix}.")
        phot_table = None

        # Iterate over filenames with the current prefix
        prefix_filenames = filenames_by_prefix[prefix]
        for filename in prefix_filenames:
            logging.info(f"Processing filename {filename}......")
            # Calibrate image and get FITS file
            logging.info(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")
            reduced_data, reduced_header, _ = reduce_images([filename])
            logging.info(f"The average pixel value for {filename} is {reduced_data[0].mean()}")
            # Convert reduced_data to a dictionary with filenames as keys
            reduced_data_dict = {filename: (data, header) for data, header in zip(reduced_data, reduced_header)}

            # Access the reduced data and header corresponding to the filename
            frame_data, frame_hdr = reduced_data_dict[filename]
            logging.info(f"Extracting photometry for {filename}")

            # Extract airmass and zero point from the header
            airmass, zp = extract_airmass_and_zp(frame_hdr)

            wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
            wcs_header = {}
            for line in [frame_hdr[i:i + 80] for i in range(0, len(frame_hdr), 80)]:
                key = line[0:8].strip()
                if '=' in line and key not in wcs_ignore_cards:
                    card = fits.Card.fromstring(line)
                    wcs_header[card.keyword] = card.value

            frame_bg = sep.Background(frame_data)
            # calculate background rms
            bg_rms = frame_bg.rms()
            frame_data_corr_no_bg = frame_data - frame_bg
            estimate_coord = SkyCoord(ra=frame_hdr['MNTRA'],
                                      dec=frame_hdr['MNTDEC'],
                                      unit=(u.deg, u.deg))
            estimate_coord_radius = 3 * u.deg

            frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                                AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
            if len(frame_objects) < N_OBJECTS_LIMIT:
                logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!")
                continue

            # Load the photometry catalog
            phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
            logging.info(f"Found catalog with name {prefix}_catalog_input.fits")
            phot_ra = catalog_column(phot_cat, "ra_deg_corr", "RA_CORR")
            phot_dec = catalog_column(phot_cat, "dec_deg_corr", "DEC_CORR")
            phot_tic = catalog_column(phot_cat, "tic_id", "TIC")
            phot_bp = catalog_column(phot_cat, "gaiabp", "BPmag")
            phot_rp = catalog_column(phot_cat, "gaiarp", "RPmag")

            # Convert RA and DEC to pixel coordinates using the WCS information from the header
            phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_ra, phot_dec, 1)

            # Do time conversions - one time value per format per target
            half_exptime = frame_hdr['EXPTIME'] / 2.
            time_isot = Time([frame_hdr['DATE-OBS'] for i in range(len(phot_x))],
                             format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            # Correct to mid-exposure time
            time_jd = time_jd + half_exptime * u.second
            ra = phot_ra
            dec = phot_dec
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            frame_ids = [filename for i in range(len(phot_x))]
            logging.info(f"Found {len(frame_ids)} sources")

            frame_preamble = Table([frame_ids, phot_cat['Tmag'], phot_tic,
                                    phot_bp, phot_rp, time_jd.value, time_bary.value,
                                    phot_x, phot_y,
                                    [airmass] * len(phot_x), [zp] * len(phot_x)],
                                   names=("frame_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                          "jd_bary", "x", "y", "airmass", "zp"))

            # Extract photometry at locations
            frame_phot = wcs_phot(frame_data, phot_x, phot_y, APERTURE_RADII, frame_data_corr_no_bg, bg_rms,
                                  gain=GAIN)

            # Stack the photometry and preamble
            frame_output = hstack([frame_preamble, frame_phot])

            # Convert frame_output to a Table if it's not already
            if not isinstance(frame_output, Table):
                frame_output = Table(frame_output)

            # Append the current frame's photometry to the accumulated photometry
            if phot_table is None:
                phot_table = frame_output
            else:
                phot_table = vstack([phot_table, frame_output])

            logging.info(f"Finished photometry for {filename}")

        # Save the photometry for the current prefix
        if phot_table is not None:
            phot_table.write(phot_output_filename, overwrite=True)
            logging.info(f"Saved photometry for prefix {prefix} to {phot_output_filename}")
        else:
            logging.info(f"No photometry data for prefix {prefix}.")

    logging.info("Done!")


if __name__ == "__main__":
    main()
