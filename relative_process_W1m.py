#!/usr/bin/env python3
"""
Build relative light curves from W1m photometry tables.

The script searches a small grid of comparison-star selection limits, builds a
weighted comparison ensemble for each trial, and keeps the trial with the
lowest unbinned RMS. The aperture is fixed; it is not optimized here.
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.table import Table, vstack

from utils_W1m import (
    bin_by_time_interval,
    camera_config,
    get_phot_files,
    read_phot_file,
    remove_outliers,
)

WORKER_CONTEXT = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Create weighted relative light curves.")
    parser.add_argument("--tic_id", type=int, help="Target TIC ID. If omitted, process all TIC IDs.")
    parser.add_argument("--cam", type=str, default="CMOS")
    parser.add_argument("--aperture", type=int, default=None, help="Fixed aperture radius to use.")
    parser.add_argument("--phot-file", type=str, default=None, help="Input phot_*.fits file.")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-file", type=str, default=None, help="Combined output relative-photometry FITS file.")
    parser.add_argument("--summary-file", type=str, default=None, help="Combined text summary file.")
    parser.add_argument("--bin-minutes", type=float, default=30.0)
    parser.add_argument("--min-comps", type=int, default=3)
    parser.add_argument("--max-comps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of target TIC optimization worker processes.")
    parser.add_argument("--dmb", type=float, nargs="*", default=[0.0, 0.2, 0.5])
    parser.add_argument("--dmf", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--crop", type=str, nargs="*", default=["none", "400", "800", "1200", "2000"])
    parser.add_argument("--color", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    return parser.parse_args()


def fixed_aperture(args):
    if args.aperture is not None:
        return args.aperture
    return camera_config(args.cam)["phot_aperture"]


def load_photometry_table(phot_file=None):
    if phot_file is None:
        phot_files = get_phot_files(".")
        if not phot_files:
            raise FileNotFoundError("No phot_*.fits file found in the current directory.")
        phot_file = phot_files[0]
    table = Table(read_phot_file(phot_file))
    return phot_file, table


def parse_crop_values(values):
    crop_values = []
    for value in values:
        if str(value).lower() in {"none", "null", "0"}:
            crop_values.append(None)
        else:
            crop_values.append(float(value))
    return crop_values


def robust_rms(flux):
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(flux)
    if np.sum(finite) < 3:
        return np.inf
    flux = flux[finite]
    median = np.nanmedian(flux)
    if not np.isfinite(median) or median == 0:
        return np.inf
    norm = flux / median
    mad = np.nanmedian(np.abs(norm - np.nanmedian(norm)))
    if mad == 0 or not np.isfinite(mad):
        return np.nanstd(norm)
    keep = np.abs(norm - np.nanmedian(norm)) < 5.0 * 1.4826 * mad
    if np.sum(keep) < 3:
        return np.inf
    return np.nanstd(norm[keep])


def detrend_airmass(flux, airmass):
    flux = np.asarray(flux, dtype=float)
    airmass = np.asarray(airmass, dtype=float)
    valid = np.isfinite(flux) & np.isfinite(airmass) & (flux > 0)
    if np.sum(valid) < 3:
        return flux / np.nanmedian(flux)
    coeff = np.polyfit(airmass[valid], flux[valid], 1)
    trend = np.polyval(coeff, airmass)
    trend[trend == 0] = np.nan
    corrected = flux / trend
    return corrected / np.nanmedian(corrected)


def grouped_star_arrays(table, aperture):
    flux_col = f"flux_{aperture}"
    fluxerr_col = f"fluxerr_{aperture}"
    sky_col = f"flux_w_sky_{aperture}"
    required = {"tic_id", "jd_bary", "Tmag", "gaiabp", "gaiarp", "x", "y", "airmass", flux_col, fluxerr_col, sky_col}
    missing = sorted(required - set(table.colnames))
    if missing:
        raise KeyError(f"Photometry table is missing required columns for aperture {aperture}: {missing}")

    stars = {}
    for tic_id in np.unique(table["tic_id"]):
        rows = table[table["tic_id"] == tic_id]
        order = np.argsort(rows["jd_bary"])
        rows = rows[order]
        flux = np.asarray(rows[flux_col], dtype=float)
        color_value = rows["gaiabp"][0] - rows["gaiarp"][0]
        try:
            if np.ma.is_masked(color_value):
                color = np.nan
            else:
                color = float(color_value)
        except (TypeError, ValueError):
            color = np.nan
        mag_column = "MAG" if "MAG" in rows.colnames else "Tmag"
        stars[int(tic_id)] = {
            "tic_id": int(tic_id),
            "time": np.asarray(rows["jd_bary"], dtype=float),
            "flux": flux,
            "fluxerr": np.asarray(rows[fluxerr_col], dtype=float),
            "sky": np.asarray(rows[sky_col], dtype=float) - flux,
            "tmag": float(rows["Tmag"][0]),
            "mag": float(rows[mag_column][0]),
            "color": color,
            "x": float(np.nanmedian(rows["x"])),
            "y": float(np.nanmedian(rows["y"])),
            "airmass": np.asarray(rows["airmass"], dtype=float),
            "rms": robust_rms(detrend_airmass(flux, rows["airmass"])),
            "n_points": len(rows),
        }
    return stars


def candidate_comps(stars, target_id, dmb, dmf, crop, color_lim, min_points):
    target = stars[target_id]
    candidates = []
    for tic_id, star in stars.items():
        if tic_id == target_id:
            continue
        if star["n_points"] != min_points:
            continue
        if star["mag"] <= 9.4:
            continue
        if not (target["mag"] - dmb <= star["mag"] <= target["mag"] + dmf):
            continue
        if not np.isfinite(target["color"]) or not np.isfinite(star["color"]):
            continue
        if abs(star["color"] - target["color"]) > color_lim:
            continue
        if crop is not None:
            half_size = crop / 2.0
            if abs(star["x"] - target["x"]) > half_size or abs(star["y"] - target["y"]) > half_size:
                continue
        if not np.isfinite(star["rms"]) or star["rms"] <= 0:
            continue
        candidates.append(star)
    return candidates


def weighted_reference_curve(comps, max_comps):
    comps = sorted(comps, key=lambda star: star["rms"])[:max_comps]
    comp_fluxes = np.vstack([star["flux"] / np.nanmedian(star["flux"]) for star in comps])
    raw_weights = np.array([1.0 / star["rms"] ** 2 for star in comps], dtype=float)
    weights = raw_weights / np.sum(raw_weights)
    reference = np.sum(weights[:, None] * comp_fluxes, axis=0)
    return reference, weights, comps


def relative_lightcurve_for_trial(stars, target_id, params, args, aperture):
    target = stars[target_id]
    comps = candidate_comps(
        stars,
        target_id,
        params["dmb"],
        params["dmf"],
        params["crop"],
        params["color_lim"],
        target["n_points"],
    )
    if len(comps) < args.min_comps:
        return None

    reference, weights, used_comps = weighted_reference_curve(comps, args.max_comps)
    target_norm = target["flux"] / np.nanmedian(target["flux"])
    rel_flux = target_norm / reference
    rel_flux /= np.nanmedian(rel_flux)

    rel_err = target["fluxerr"] / np.nanmedian(target["flux"])
    time, rel_flux, rel_err, _, _ = remove_outliers(target["time"], rel_flux, rel_err)
    if len(time) < 3:
        return None

    _, binned_flux, _ = bin_by_time_interval(time, rel_flux, rel_err, args.bin_minutes)
    if len(binned_flux) < 2:
        binned_rms_ppm = np.inf
    else:
        binned_rms_ppm = float(np.nanstd(binned_flux) * 1e6)

    unbinned_rms_ppm = float(np.nanstd(rel_flux) * 1e6)
    score = unbinned_rms_ppm
    if len(used_comps) < 5:
        score += (5 - len(used_comps)) * 100.0

    return {
        "tic_id": target_id,
        "aperture": aperture,
        "params": params,
        "time": time,
        "relative_flux": rel_flux,
        "relative_flux_err": rel_err,
        "binned_rms_ppm": binned_rms_ppm,
        "unbinned_rms_ppm": unbinned_rms_ppm,
        "score": score,
        "comp_ids": [star["tic_id"] for star in used_comps],
        "comp_weights": weights,
        "comp_rms": [star["rms"] for star in used_comps],
    }


def parameter_grid(args):
    crops = parse_crop_values(args.crop)
    for dmb, dmf, crop, color_lim in itertools.product(args.dmb, args.dmf, crops, args.color):
        yield {
            "dmb": float(dmb),
            "dmf": float(dmf),
            "crop": crop,
            "color_lim": float(color_lim),
        }


def optimise_target(stars, target_id, args, aperture):
    if target_id not in stars:
        raise ValueError(f"TIC ID {target_id} not found in photometry table.")

    best = None
    for params in parameter_grid(args):
        result = relative_lightcurve_for_trial(stars, target_id, params, args, aperture)
        if result is None:
            continue
        if best is None or result["score"] < best["score"]:
            best = result
    return best


def init_worker(context):
    WORKER_CONTEXT.update(context)


def optimise_target_worker(target_id):
    return target_id, optimise_target(
        WORKER_CONTEXT["stars"],
        int(target_id),
        WORKER_CONTEXT["args"],
        WORKER_CONTEXT["aperture"],
    )


def optimise_targets(target_ids, stars, args, aperture):
    workers = max(1, args.workers if args.workers is not None else camera_config(args.cam)["workers"])
    if workers <= 1 or len(target_ids) <= 1:
        return optimise_targets_sequential(target_ids, stars, args, aperture)

    context = {"stars": stars, "args": args, "aperture": aperture}
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(context,)) as executor:
            future_to_target = {
                executor.submit(optimise_target_worker, target_id): target_id for target_id in target_ids
            }
            for future in as_completed(future_to_target):
                target_id = future_to_target[future]
                try:
                    _, result = future.result()
                    results.append((target_id, result, None))
                except Exception as exc:
                    results.append((target_id, None, f"TIC {target_id}: failed with error: {exc}"))
    except (OSError, PermissionError) as exc:
        print(f"Multiprocessing unavailable ({exc}); falling back to 1 worker.")
        return optimise_targets_sequential(target_ids, stars, args, aperture)
    results.sort(key=lambda item: target_ids.index(item[0]))
    return results


def optimise_targets_sequential(target_ids, stars, args, aperture):
    results = []
    for target_id in target_ids:
        try:
            results.append(optimise_target_worker_local(target_id, stars, args, aperture))
        except Exception as exc:
            results.append((target_id, None, f"TIC {target_id}: failed with error: {exc}"))
    return results


def optimise_target_worker_local(target_id, stars, args, aperture):
    result = optimise_target(stars, int(target_id), args, aperture)
    return target_id, result, None


def output_paths(phot_file, args):
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(phot_file)
    if basename.startswith("phot_"):
        output_basename = "rel_" + basename
    else:
        output_basename = "rel_phot_" + basename

    root, _ = os.path.splitext(output_basename)
    lightcurve_path = args.output_file or os.path.join(args.output_dir, output_basename)
    summary_path = args.summary_file or os.path.join(args.output_dir, f"{root}_summary.txt")
    return lightcurve_path, summary_path


def result_to_table(result):
    n_rows = len(result["time"])
    crop = -1.0 if result["params"]["crop"] is None else result["params"]["crop"]
    return Table(
        [
            np.full(n_rows, result["tic_id"], dtype=np.int64),
            result["time"],
            result["relative_flux"],
            result["relative_flux_err"],
            np.full(n_rows, result["aperture"], dtype=np.int16),
            np.full(n_rows, result["binned_rms_ppm"], dtype=float),
            np.full(n_rows, result["unbinned_rms_ppm"], dtype=float),
            np.full(n_rows, len(result["comp_ids"]), dtype=np.int16),
            np.full(n_rows, result["params"]["dmb"], dtype=float),
            np.full(n_rows, result["params"]["dmf"], dtype=float),
            np.full(n_rows, crop, dtype=float),
            np.full(n_rows, result["params"]["color_lim"], dtype=float),
        ],
        names=(
            "tic_id",
            "jd_bary",
            "relative_flux",
            "relative_flux_err",
            "aperture",
            "binned_rms_ppm",
            "unbinned_rms_ppm",
            "n_comps",
            "dmb",
            "dmf",
            "crop",
            "color_lim",
        ),
    )


def write_summary(handle, result):
    tic_id = result["tic_id"]
    handle.write(f"TIC ID: {tic_id}\n")
    handle.write(f"Aperture: {result['aperture']}\n")
    handle.write(f"Binned RMS ppm: {result['binned_rms_ppm']:.3f}\n")
    handle.write(f"Unbinned RMS ppm: {result['unbinned_rms_ppm']:.3f}\n")
    handle.write(f"Score (unbinned RMS ppm plus penalties): {result['score']:.3f}\n")
    handle.write(f"Parameters: {result['params']}\n")
    handle.write("Comparison stars:\n")
    for comp_id, weight, rms in zip(result["comp_ids"], result["comp_weights"], result["comp_rms"]):
        handle.write(f"  {comp_id} weight={weight:.6f} rms={rms:.8f}\n")
    handle.write("\n")


def write_results(results, phot_file, args):
    lightcurve_path, summary_path = output_paths(phot_file, args)
    if not results:
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("No valid relative light curves were produced.\n")
        return None, summary_path

    tables = [result_to_table(result) for result in results]
    output = vstack(tables)
    output.meta["PHOTFILE"] = os.path.basename(phot_file)
    output.meta["NSTARS"] = len(results)
    config = camera_config(args.cam)
    output.meta["CATALOG"] = config["catalog"]
    output.meta["MAGSYS"] = "G" if config["catalog"] == "gaia_dr3" else "TESS"
    output.write(lightcurve_path, overwrite=True)

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Input photometry file: {phot_file}\n")
        handle.write(f"Relative photometry file: {lightcurve_path}\n")
        handle.write(f"Number of successful stars: {len(results)}\n\n")
        for result in results:
            write_summary(handle, result)

    return lightcurve_path, summary_path


def main():
    args = parse_args()
    aperture = fixed_aperture(args)
    phot_file, table = load_photometry_table(args.phot_file)
    stars = grouped_star_arrays(table, aperture)

    target_ids = [args.tic_id] if args.tic_id is not None else sorted(stars)
    print(f"Loaded {phot_file} with {len(stars)} stars. Using aperture {aperture}.")
    workers = max(1, args.workers if args.workers is not None else camera_config(args.cam)["workers"])
    print(f"Using {workers} target optimization worker(s).")

    results = []
    for target_id, result, error in optimise_targets(target_ids, stars, args, aperture):
        if error:
            print(error)
            continue

        if result is None:
            print(f"TIC {target_id}: no valid comparison ensemble found.")
            continue

        results.append(result)
        print(
            f"TIC {target_id}: binned RMS={result['binned_rms_ppm']:.2f} ppm, "
            f"unbinned RMS={result['unbinned_rms_ppm']:.2f} ppm, "
            f"n_comps={len(result['comp_ids'])}"
        )

    lightcurve_path, summary_path = write_results(results, phot_file, args)
    if lightcurve_path is None:
        print(f"No valid relative light curves written. Summary={summary_path}")
    else:
        print(f"Wrote combined relative photometry: {lightcurve_path}")
        print(f"Wrote combined summary: {summary_path}")


if __name__ == "__main__":
    main()
