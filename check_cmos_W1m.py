#!/usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""

from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import logging
import warnings

# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level

# Create file handler
file_handler = logging.FileHandler('donuts.log')
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
    filtered_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames):
    """
    Extract unique prefixes from a list of filenames.

    Parameters
    ----------
    filenames : list of str
        List of filenames.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the filenames.
    """
    prefixes = set()
    for filename in filenames:
        prefix = filename[:11]
        prefixes.add(prefix)
    return prefixes


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
                ctype1 = header.get('CTYPE1')
                ctype2 = header.get('CTYPE2')

                if ctype1 is None or ctype2 is None:
                    logger.warning(f"{file} does not have CTYPE1 and/or CTYPE2 in the header. "
                                   f"Moving to 'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(os.path.join(directory, file), new_path)

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
    for filename, file_group in zip(filenames, file_groups):
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
            shifts = np.array([abs(sx), abs(sy)])

            all_sx.append(sx)
            all_sy.append(sy)

            if abs(sx) >= 0.5 or abs(sy) >= 0.5:
                logger.warning(f'{i} image shift too big X: {sx} Y: {sy}')
                if not os.path.exists('failed_donuts'):
                    os.mkdir('failed_donuts')
                comm = f'mv {i} failed_donuts/'
                logger.info(comm)
                os.system(comm)
    # Compute scatter (std) after all shifts
    if all_sx and all_sy:
        std_x = np.std(all_sx)
        std_y = np.std(all_sy)
        logger.info(f"Scatter of X shifts (std): {std_x:.3f} pixels")
        logger.info(f"Scatter of Y shifts (std): {std_y:.3f} pixels")
    else:
        logger.info("No shifts measured; scatter cannot be computed.")


def main():
    # set directory for working
    directory = os.getcwd()
    logger.info(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logger.info(f"Number of files: {len(filenames)}")

    # Iterate over each filename to get the prefix
    prefixes = get_prefix(filenames)
    logger.info(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = [[filename for filename in filenames if filename.startswith(prefix)] for prefix in prefixes]

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(prefix_filenames, filenames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
