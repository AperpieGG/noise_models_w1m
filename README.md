# noise_models_w1m

Photometry and noise-model pipeline for W1m FITS image sequences. The pipeline solves a reference image, checks image registration, extracts calibrated aperture photometry, builds relative light curves, and compares the measured RMS against an analytic noise model.

## Quick Start

Run the full pipeline from the repository root by passing a camera config and an observing-night directory:

```bash
./run.sh qhy600 20250809
```

Or run it from inside the observing-night directory:

```bash
../run.sh qhy600
```

The first argument can be a camera name, a JSON filename in `configs/`, or a path to a JSON config. The second argument is optional and names a data directory under the repository root.

To override the worker count in the camera JSON:

```bash
PIPELINE_WORKERS=4 ./run.sh qhy600 20250809
```

Pipeline logs are written to the data directory, for example:

```text
20250809/logs/
```

## Camera Configs

Camera settings live in `configs/*.json`. These files control header keywords, plate scale, detector noise parameters, site location, scintillation parameters, and default worker count.

Important fields:

```json
{
  "name": "qhy600",
  "scale_min": 0.1,
  "scale_max": 0.5,
  "ra_key": "MNTRAD",
  "dec_key": "MNTDECD",
  "box_size": 1,
  "catalog": "tic82",
  "prefix_chars": null,
  "phot_aperture": 5,
  "gain": 1.13,
  "dark_current": 1.6,
  "read_noise": 1.56,
  "exposure": 10.0,
  "phot_estimate_radius_deg": 3.0,
  "location": {
    "lat": -24.615662,
    "lon": -70.391809,
    "height": 2433
  },
  "scintillation": {
    "diameter": 0.2,
    "h": 2433,
    "H": 8000,
    "c_Y": 1.56
  },
  "workers": 1
}
```

`ra_key` and `dec_key` are the FITS header cards used for pointing. `prefix_chars` controls how `OBJECT` names are grouped into fields. `location` is used for barycentric time corrections. `scintillation` is used by the analytic noise model.

`catalog` selects the VizieR source catalog used when a field catalog is first
created. Supported values are `tic82` (the default) and `gaia_dr3`. Gaia DR3
does not provide TIC IDs or TESS magnitudes, so the pipeline uses its Gaia
source ID and `Gmag` in the corresponding compatibility columns. Gaia runs
measure and plot Gaia `G`-band zero points and magnitudes; TIC 8.2 runs use
TESS magnitudes. If this setting changes, the wrapper rebuilds an existing
field catalog on its next run.

## Full Pipeline

[run.sh](./run.sh) is the normal entry point. It resolves the camera config, reads the configured worker count, moves into the data directory, sets `PIPELINE_LOG_DIR`, and runs:

```bash
python -u simple_wrapper_W1m.py --camera "$CAMERA_CONFIG_PATH"
python -u check_cmos_W1m.py --camera "$CAMERA_CONFIG_PATH" --workers "$WORKERS"
python -u adding_header_W1m.py --camera "$CAMERA_CONFIG_PATH"
python -u process_cmos_W1m.py --camera "$CAMERA_CONFIG_PATH" --workers "$WORKERS"
python -u relative_process_W1m.py --cam "$CAMERA_CONFIG_PATH" --workers "$WORKERS"
```

Typical outputs in the data directory include:

```text
<OBJECT>_catalog.fits
<OBJECT>_catalog_input.fits
phot_<OBJECT>.fits
rel_phot_<OBJECT>.fits
rel_phot_<OBJECT>_summary.txt
logs/
```

## Script Reference

### `run.sh`

Top-level orchestration script. Use this for normal processing.

What it does:

- Resolves the requested camera config.
- Reads `workers` from the camera config unless `PIPELINE_WORKERS` is set.
- Sets the working directory and `PIPELINE_LOG_DIR`.
- Runs reference solving, image checks, header updates, photometry, and relative photometry.

Example:

```bash
./run.sh qhy600 20250809
```

### `simple_wrapper_W1m.py`

Creates or reuses the field reference catalog and solves reference/science images using the configured camera plate scale.

What it does:

- Finds science FITS files in the current directory.
- Uses the first science image as the initial reference.
- Gets field center information from the camera-specific header keys.
- Runs `make_ref_catalog_W1m.py` if the field catalog does not exist.
- Runs `solve_ref_images_W1m.py` for the reference image and remaining unsolved images.

Common options:

```bash
python simple_wrapper_W1m.py --camera configs/qhy600.json
python simple_wrapper_W1m.py --camera configs/qhy600.json --force3rd
python simple_wrapper_W1m.py --camera configs/qhy600.json --save_matched_cat
```

### `make_ref_catalog_W1m.py`

Queries Vizier and builds the field catalog used for astrometric solving and photometry target selection.

What it does:

- Queries a supported VizieR catalog, default `tic82` (`IV/39/tic82`).
- Applies proper-motion corrections for the requested epoch.
- Flags blended sources.
- Writes a FITS catalog.

Example:

```bash
python make_ref_catalog_W1m.py RA DEC 1 1 2024-01-22T00:00:00 NG0719+0956_catalog.fits
```

### `solve_ref_images_W1m.py`

Solves FITS images against an existing reference catalog.

What it does:

- Detects image sources with SEP.
- Matches detected image sources to the catalog.
- Fits the WCS/distortion solution.
- Writes the astrometric solution into the FITS header.
- Optionally saves matched-catalog diagnostics.

Example:

```bash
python solve_ref_images_W1m.py NG0719+0956_catalog.fits IMAGE.fits --scale_min 0.1 --scale_max 0.5
```

### `check_cmos_W1m.py`

Checks solved images before photometry.

What it does:

- Moves images without WCS cards into `no_wcs/`.
- Groups images by object prefix.
- Uses Donuts to measure the shift of each image against the first image in its group.
- Moves images with shifts of at least 1 pixel into `failed_donuts/`.
- Writes `logs/donuts.log` and `logs/donuts_pixel_shifts.png`.

This script supports workers. The first image in each group is still the reference; all other images are independently compared against that reference.

Example:

```bash
python check_cmos_W1m.py --camera configs/qhy600.json --workers 4
```

### `adding_header_W1m.py`

Adds missing convenience header values to science FITS files.

What it does:

- Adds `FILTER = NONE` if missing.
- Estimates and adds `AIRMASS` if missing.
- Computes and adds `BJD` and `HJD` if missing.
- Uses the camera config location for time corrections when `--camera` is supplied.

Example:

```bash
python adding_header_W1m.py --camera configs/qhy600.json
```

### `calibration_images_W1m.py`

Shared calibration helper module. It is normally imported by other scripts, not run directly.

What it does:

- Builds or loads `master_bias.fits`.
- Builds or loads `master_dark.fits`.
- Builds or loads `master_flat.fits`.
- Applies bias, dark, and flat corrections through `reduce_image`.

### `process_cmos_W1m.py`

Extracts calibrated aperture photometry for each solved science image.

What it does:

- Loads calibration masters once.
- Reduces each science image.
- Measures airmass and zero point from the FITS header.
- Uses WCS to place catalog sources on the image.
- Runs aperture photometry with the configured aperture and gain.
- Writes `phot_<OBJECT>.fits`.

This script supports workers across images.

Example:

```bash
python process_cmos_W1m.py --camera configs/qhy600.json --workers 4
```

### `relative_process_W1m.py`

Builds relative light curves from `phot_*.fits`.

What it does:

- Selects comparison-star ensembles for each target.
- Searches a grid of magnitude, color, and crop limits.
- Chooses the comparison set with the lowest binned RMS.
- Writes `rel_phot_<OBJECT>.fits` and `rel_phot_<OBJECT>_summary.txt`.

This script supports workers across target TIC IDs.

Example:

```bash
python relative_process_W1m.py --cam configs/qhy600.json --workers 4
```

### `noise_model_W1m.py`

Builds a JSON noise model from relative photometry and calibrated photometry.

What it does:

- Reads `rel_phot_*.fits` and the matching `phot_*.fits`.
- Accepts the observing-night directory as a positional argument.
- Computes measured RMS per source ID.
- Computes photon, sky, read, dark-current, scintillation, and total noise terms.
- Uses detector, site, and scintillation values from the camera config.
- Uses Gaia `G` magnitudes for `gaia_dr3` catalogs and TESS magnitudes for `tic82`.
- Writes `noise_model_<OBJECT>_<CAMERA>_bin<N>.json`.

Example:

```bash
python noise_model_W1m.py 20250809 --cam configs/qhy600.json --bin-size 1
```

### `plot_noise_model_W1m.py`

Plots a noise-model JSON.

What it does:

- Reads the JSON generated by `noise_model_W1m.py`.
- Plots measured RMS against Gaia `G` or TESS magnitude, according to the JSON metadata.
- Overplots the analytic noise components.
- Writes a PNG beside the input JSON unless `--output` is passed.

Example:

```bash
python plot_noise_model_W1m.py 20250809/noise_model_NG0719+0956_qhy600_bin1.json
```

### `utils_W1m.py`

Shared utility module. It is imported by the pipeline scripts.

Main responsibilities:

- Camera config loading and config-path resolution.
- FITS filename filtering and object-prefix grouping.
- Observatory location handling.
- Barycentric/heliocentric light-travel-time calculations.
- SEP/WCS photometry helpers.
- Noise-source calculations, including scintillation noise.

## Parallel Processing

Parallelism is configured with the `workers` field in the camera JSON or by setting `PIPELINE_WORKERS`.

The bash script only passes the worker count through. The actual parallel work happens inside Python:

- `check_cmos_W1m.py`: WCS header checks and Donuts shift measurements.
- `process_cmos_W1m.py`: image reduction and aperture photometry.
- `relative_process_W1m.py`: relative-photometry optimisation per target.

## Notes

- Run the pipeline from a data directory, or pass the data directory as the second argument to `run.sh`.
- Empty top-level `logs/` files are not required; useful pipeline logs should be inside the data directory.
- Existing output files are often reused or skipped. Remove old products if you want to force a clean rerun.
