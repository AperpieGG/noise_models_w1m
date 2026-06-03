#!/usr/bin/env python3
"""
Run through many reference images, generate catalogs
and try solving them one by one
"""
import os
import argparse as ap
import subprocess
from pathlib import Path
from astropy.io import fits
from utils_W1m import (
    camera_config,
    filter_science_filenames,
    is_astrometrically_solved,
    object_prefix,
    plot_frame_diagnostics,
    pointing_from_header,
)


def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser("Solve AG references images for CASUTools")
    p.add_argument('--defocus',
                   help='manual override for defocus (mm)',
                   type=float,
                   default=0.0)
    p.add_argument('--force3rd',
                   help='force a 3rd order distortion polyfit',
                   action='store_true',
                   default=False)
    p.add_argument('--save_matched_cat',
                   help='output the matched catalog with basic photometry',
                   action='store_true',
                   default=False)
    p.add_argument('--camera',
                   help='camera type (ccd, cmos or IMX571)',
                   type=str,
                   default='IMX571')
    p.add_argument('--script-dir',
                   help='Directory containing the pipeline Python scripts',
                   type=Path,
                   default=Path(__file__).resolve().parent)
    return p.parse_args()


def run_solver(script_dir, cat_file, fits_file, scale_min, scale_max, config, args):
    """
    Run solve_ref_images_W1m.py for one FITS file.
    """
    cmd = [
        "python",
        str(script_dir / "solve_ref_images_W1m.py"),
        cat_file,
        fits_file,
        "--scale_min",
        str(scale_min),
        "--scale_max",
        str(scale_max),
        "--photometry-mag-limit",
        str(config["photometry_mag_limit"]),
        "--min-photometry-targets",
        str(config["min_photometry_targets"]),
        "--phot-aperture",
        str(config["phot_aperture"]),
    ]
    if args.save_matched_cat:
        cmd.append("--save_matched_cat")
    if args.defocus is not None:
        cmd.extend(["--defocus", f"{args.defocus:.2f}"])
    if args.force3rd:
        cmd.append("--force3rd")
    result = subprocess.run(cmd, check=False).returncode
    remove_astrometric_diagnostics(fits_file)
    return result


def remove_astrometric_diagnostics(fits_file):
    """
    Remove per-image astrometric diagnostic plots after solving.
    """
    image_base = os.path.splitext(fits_file)[0]
    for suffix in ("_quiver_plot.png", "_wcs_residuals.png"):
        diagnostic = f"{image_base}{suffix}"
        if os.path.exists(diagnostic):
            os.remove(diagnostic)
            print(f"Removed astrometric diagnostic: {diagnostic}")


if __name__ == "__main__":
    # Grab command line args
    args = arg_parse()
    config = camera_config(args.camera)
    script_dir = args.script_dir.resolve()

    # Set scale values based on camera type
    print(f'Using camera type: {args.camera}')
    scale_min = config["scale_min"]
    scale_max = config["scale_max"]
    mag_system = "G" if config["catalog"] == "gaia_dr3" else "TESS"

    # Get a list of all FITS images, exclude whatever has catalog name in
    all_fits = sorted([f for f in os.listdir(".") if f.endswith(".fits")])
    print(f"Found {len(all_fits)} FITS files.")
    all_fits = filter_science_filenames(".")
    print(f"Found {len(all_fits)} FITS files after excluding not suitable files.")

    if not all_fits:
        print("No FITS files found.")
        exit(1)

    # Select the first image as the reference
    ref_image = all_fits[0]
    with fits.open(ref_image) as ff:
        prefix = object_prefix(ff[0].header, args.camera)

    print(f"Using reference image: {ref_image} with prefix: {prefix}")
    cat_file = f"{prefix}_catalog.fits"

    # Get coordinates from the reference image header
    with fits.open(ref_image) as ff:
        ra, dec, epoch, box_size = pointing_from_header(ff[0].header, args.camera)

    # Create the catalog if it doesn't exist or the configured source changed.
    create_catalog = not os.path.exists(cat_file)
    if not create_catalog:
        with fits.open(cat_file) as hdulist:
            header = hdulist[1].header
            existing_catalog = header.get("CATALOG", "tic82").lower()
            existing_blend_separation = float(header.get("BLENDSEP", 100.0))
            existing_blend_delta = float(header.get("BLENDDEL", 1.0))
            existing_mag_limit = float(header.get("MAGLIMIT", 16.0))
        create_catalog = (
            existing_catalog != config["catalog"].lower()
            or existing_blend_separation != config["blend_separation_arcsec"]
            or existing_blend_delta != config["blend_delta"]
            or existing_mag_limit != config["photometry_mag_limit"]
        )
        if create_catalog:
            print("Catalog source or blend settings changed; rebuilding the field catalog.")

    if create_catalog:
        print(f'Creating catalog file: {cat_file}')
        cmd_args = [
            "python",
            str(script_dir / "make_ref_catalog_W1m.py"),
            ra,
            dec,
            box_size,
            box_size,
            epoch,
            cat_file,
            "--catalog",
            config["catalog"],
            "--blend-separation-arcsec",
            str(config["blend_separation_arcsec"]),
            "--blend-delta",
            str(config["blend_delta"]),
            "--magnitude-limit",
            str(config["photometry_mag_limit"]),
        ]
        subprocess.run(cmd_args, check=True)
        print("Catalog created for image {} with prefix: {}\n".format(ref_image, prefix))

    # Solve reference image with catalog file
    if os.path.exists(cat_file):
        print(f'Solving reference image: {ref_image}')
        result = run_solver(script_dir, cat_file, ref_image, scale_min, scale_max, config, args)

        if result != 0:  # Exit if reference solving or catalog validation fails
            print(f"Reference image processing failed for {ref_image}. Exiting the script.")
            exit(1)
        else:
            print(f"Successfully solved the reference image {ref_image}.\n")

        # Iterate and solve remaining FITS images
        for fits_file in all_fits:
            if fits_file == ref_image:
                continue

            with fits.open(fits_file) as hdulist:
                current_prefix = object_prefix(hdulist[0].header, args.camera)

                if current_prefix.startswith(prefix):
                    if "_cat" not in fits_file and fits_file != ref_image:
                        solved_mag_system = hdulist[0].header.get("MAGSYS", "TESS")
                        if is_astrometrically_solved(hdulist[0].header) and solved_mag_system == mag_system:
                            print(f"Image {fits_file} is already solved. Skipping..\n")
                            continue

                        print(f"Solving image {fits_file} for prefix: {prefix}\n")
                        result = run_solver(script_dir, cat_file, fits_file, scale_min, scale_max, config, args)

                        if result != 0:
                            print(f"Failed to solve the image {fits_file}. Skipping to the next image.\n")
                            continue  # Skip this image and move to the next
                        else:
                            print(f"Successfully solved the image {fits_file}.\n")

        plot_frame_diagnostics()
