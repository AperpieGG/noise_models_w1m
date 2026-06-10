import glob
import os
from dataclasses import dataclass

from astropy.io import fits
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils_W1m import get_location, get_light_travel_times

_MATCHED_CALIBRATION_CACHE = {}


@dataclass(frozen=True)
class CalibrationMaster:
    data: np.ndarray
    header: fits.Header
    kind: str


def _calibration_block_rows(image_shape, n_files, target_mb=None):
    """
    Choose a row-block size that keeps the temporary median stack modest.
    """
    if target_mb is None:
        target_mb = int(os.environ.get("W1M_CALIBRATION_STACK_MB", "256"))
    height, width = image_shape
    target_bytes = target_mb * 1024 * 1024
    bytes_per_row = width * n_files * np.dtype(np.float32).itemsize
    if bytes_per_row <= 0:
        return height
    return max(1, min(height, target_bytes // bytes_per_row))


def _image_shape_from_header(header):
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def _scaled_hdu_block(hdu, row_slice):
    """
    Read a FITS image row block and apply simple BSCALE/BZERO scaling.
    """
    header = hdu[0].header
    raw_data = hdu[0].data[row_slice, :]
    data = np.asarray(raw_data, dtype=np.float32)
    blank = header.get("BLANK")
    if blank is not None:
        data = data.copy()
        data[raw_data == blank] = np.nan
    bscale = float(header.get("BSCALE", 1.0))
    bzero = float(header.get("BZERO", 0.0))
    if bscale != 1.0:
        data *= bscale
    if bzero != 0.0:
        data += bzero
    return data


def _open_raw_image(filename):
    return fits.open(filename, memmap=True, do_not_scale_image_data=True)


def _chunked_median_master(files, label, calibrate_block=None):
    """
    Build a calibration master without storing the full frame stack in RAM.
    """
    with _open_raw_image(files[0]) as hdul:
        image_shape = _image_shape_from_header(hdul[0].header)
        header = hdul[0].header.copy()

    block_rows = _calibration_block_rows(image_shape, len(files))
    print(
        f"Creating master {label} from {len(files)} files in "
        f"{block_rows}-row blocks."
    )
    master = np.empty(image_shape, dtype=np.float32)

    hdulists = [_open_raw_image(filename) for filename in files]
    try:
        for row_start in range(0, image_shape[0], block_rows):
            row_stop = min(row_start + block_rows, image_shape[0])
            row_slice = slice(row_start, row_stop)
            stack = np.empty(
                (len(files), row_stop - row_start, image_shape[1]),
                dtype=np.float32,
            )

            for index, hdul in enumerate(hdulists):
                data = _scaled_hdu_block(hdul, row_slice)
                if calibrate_block is not None:
                    data = calibrate_block(data, hdul[0].header, row_slice, index)
                stack[index] = data

            master[row_slice, :] = np.median(stack, axis=0)
    finally:
        for hdul in hdulists:
            hdul.close()

    return master, header


def get_binning_from_header(header):
    """
    Return detector binning as (x, y), or None when the header does not say.
    """
    if header is None:
        return None

    if 'CAM-BIN' in header:
        value = str(header['CAM-BIN']).lower().replace(' ', '')
        if 'x' in value:
            x_bin, y_bin = value.split('x', maxsplit=1)
            return int(x_bin), int(y_bin)
        cam_bin = int(value)
        return cam_bin, cam_bin

    for x_key, y_key in (
        ('HBIN_SZ', 'VBIN_SZ'),
        ('XBINNING', 'YBINNING'),
        ('CCDXBIN', 'CCDYBIN'),
    ):
        if x_key in header and y_key in header:
            return int(header[x_key]), int(header[y_key])

    return None


def _master_parts(master, kind):
    if isinstance(master, CalibrationMaster):
        return master.data, master.header, master.kind
    return master, fits.Header(), kind


def _integer_rebin_factor(source_shape, target_shape, expected_factor=None):
    source_y, source_x = source_shape
    target_y, target_x = target_shape
    if expected_factor is not None:
        factor_x, factor_y = expected_factor
        trim_x = source_x - target_x * factor_x
        trim_y = source_y - target_y * factor_y
        if 0 <= trim_x < factor_x and 0 <= trim_y < factor_y:
            return expected_factor, (trim_x, trim_y)
        return None
    if source_x % target_x or source_y % target_y:
        return None
    return (source_x // target_x, source_y // target_y), (0, 0)


def _rebin_blocks(data, factor_x, factor_y, operation):
    target_y = data.shape[0] // factor_y
    target_x = data.shape[1] // factor_x
    blocks = data.reshape(target_y, factor_y, target_x, factor_x)
    if operation == 'sum':
        return blocks.sum(axis=(1, 3))
    return blocks.mean(axis=(1, 3))


def match_calibration_to_image(master, target_shape, target_header, kind,
                               rebin_mode='mean'):
    """
    Match a calibration master to an image, rebinned only by integer factors.
    """
    if master is None:
        return None
    if rebin_mode not in ('mean', 'sum'):
        raise ValueError(
            f"Unsupported calibration_rebin_mode '{rebin_mode}'. Use 'mean' or 'sum'."
        )

    data, calibration_header, master_kind = _master_parts(master, kind)
    if data.shape == target_shape:
        return data
    cache_key = (id(data), target_shape, master_kind, rebin_mode)
    if cache_key in _MATCHED_CALIBRATION_CACHE:
        return _MATCHED_CALIBRATION_CACHE[cache_key]

    calibration_binning = get_binning_from_header(calibration_header)
    target_binning = get_binning_from_header(target_header)
    factor = None
    method = 'array shapes'

    if calibration_binning is not None and target_binning is not None:
        cal_x, cal_y = calibration_binning
        target_x, target_y = target_binning
        if target_x % cal_x or target_y % cal_y:
            raise ValueError(
                f"Cannot rebin {master_kind} from header binning {cal_x}x{cal_y} "
                f"to science binning {target_x}x{target_y}: the factors are not integers."
            )
        factor = target_x // cal_x, target_y // cal_y
        method = 'FITS binning headers'

    expected_factor = factor
    if expected_factor is None and calibration_binning is None and target_binning is not None:
        # A headerless master is commonly an unbinned full-frame calibration.
        expected_factor = target_binning
        method = 'science FITS binning header and array shapes'
    shape_match = _integer_rebin_factor(data.shape, target_shape, expected_factor)
    if shape_match is None and expected_factor is not None:
        shape_match = _integer_rebin_factor(data.shape, target_shape)
    if shape_match is None:
        shape_factor = None
        trim_x = trim_y = 0
    else:
        shape_factor, (trim_x, trim_y) = shape_match
    if shape_factor is None:
        raise ValueError(
            f"Cannot match {master_kind} shape {data.shape} to science shape "
            f"{target_shape}: dimensions are not related by integer binning."
        )
    if factor is None:
        factor = shape_factor
    elif factor != shape_factor:
        raise ValueError(
            f"{master_kind} headers imply a {factor[0]}x{factor[1]} rebin, but "
            f"the array shapes imply {shape_factor[0]}x{shape_factor[1]}."
        )

    factor_x, factor_y = factor
    if factor_x < 1 or factor_y < 1 or (factor_x == 1 and factor_y == 1):
        raise ValueError(
            f"Cannot match {master_kind} shape {data.shape} to science shape "
            f"{target_shape} by down-binning."
        )

    operation = 'mean' if master_kind == 'flat' else rebin_mode
    if trim_x or trim_y:
        print(
            f"Trimming {trim_y} incomplete high-edge row(s) and {trim_x} "
            f"incomplete high-edge column(s) from master {master_kind} before rebinning."
        )
        data = data[:data.shape[0] - trim_y if trim_y else None,
                    :data.shape[1] - trim_x if trim_x else None]
    print(
        f"Rebinning master {master_kind} from shape {data.shape} to {target_shape} "
        f"using {factor_x}x{factor_y} block {operation} inferred from {method}."
    )
    matched_data = _rebin_blocks(data, factor_x, factor_y, operation)
    _MATCHED_CALIBRATION_CACHE[cache_key] = matched_data
    return matched_data


def bias():
    master_bias_path = os.path.join('.', 'master_bias.fits')
    if os.path.exists(master_bias_path):
        data, header = fits.getdata(master_bias_path, header=True)
        return CalibrationMaster(data, header, 'bias')
    else:
        files = [f for f in glob.glob(os.path.join('.', 'bias*.fits'))][:21]
        if not files:
            print('No bias files found. Skipping bias correction.')
            return None
        print('Creating master bias')
        master_bias, header = _chunked_median_master(files, 'bias')
        fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)
        print(f'Master bias saved to: {os.path.join(os.getcwd(), "master_bias.fits")}')
        return CalibrationMaster(master_bias, header, 'bias')


def dark(master_bias, calibration_rebin_mode='mean'):
    master_dark_path = os.path.join('.', 'master_dark.fits')
    if os.path.exists(master_dark_path):
        data, header = fits.getdata(master_dark_path, header=True)
        return CalibrationMaster(data, header, 'dark')
    else:
        files = [f for f in glob.glob(os.path.join('.', 'dark*.fits'))][:21]
        if not files:
            print('No dark files found. Skipping dark correction.')
            return None
        print('Creating master dark')
        with _open_raw_image(files[0]) as hdul:
            dark_shape = _image_shape_from_header(hdul[0].header)

        def calibrate_dark_block(dark_data, header, row_slice, _index):
            if master_bias is not None:
                matched_bias = match_calibration_to_image(
                    master_bias, dark_shape, header, 'bias',
                    calibration_rebin_mode
                )[row_slice, :]
                dark_data = dark_data - matched_bias
            return dark_data

        master_dark, header = _chunked_median_master(
            files, 'dark', calibrate_dark_block
        )
        fits.PrimaryHDU(master_dark, header=header).writeto(master_dark_path, overwrite=True)
        print(f'Master dark saved to: {os.path.join(os.getcwd(), "master_dark.fits")}')
        return CalibrationMaster(master_dark, header, 'dark')


def flat(master_bias, master_dark, dark_exposure=10, calibration_rebin_mode='mean'):
    master_flat_path = os.path.join('.', 'master_flat.fits')
    if os.path.exists(master_flat_path):
        data, header = fits.getdata(master_flat_path, header=True)
        return CalibrationMaster(data, header, 'flat')
    else:
        files = glob.glob(os.path.join('.', 'evening*.fits')) or glob.glob(os.path.join('.', 'morning*.fits'))
        if not files:
            print('No flat field files found. Skipping flat correction.')
            return None
        print(f'Found {len(files)} flat files. Creating master flat.')
        files = files[:21]
        with _open_raw_image(files[0]) as hdul:
            flat_shape = _image_shape_from_header(hdul[0].header)

        flat_norms = []
        for f in files:
            with _open_raw_image(f) as hdul:
                header = hdul[0].header
                total = 0.0
                count = 0
                block_rows = _calibration_block_rows(flat_shape, 1)
                for row_start in range(0, flat_shape[0], block_rows):
                    row_stop = min(row_start + block_rows, flat_shape[0])
                    row_slice = slice(row_start, row_stop)
                    data = _scaled_hdu_block(hdul, row_slice)
                    if master_bias is not None:
                        data -= match_calibration_to_image(
                            master_bias, flat_shape, header, 'bias',
                            calibration_rebin_mode
                        )[row_slice, :]
                    if master_dark is not None:
                        data -= match_calibration_to_image(
                            master_dark, flat_shape, header, 'dark',
                            calibration_rebin_mode
                        )[row_slice, :] * header['EXPTIME'] / dark_exposure
                    finite = np.isfinite(data)
                    total += np.nansum(data)
                    count += np.count_nonzero(finite)
                flat_norms.append(total / count)

        def calibrate_flat_block(data, header, row_slice, index):
            if master_bias is not None:
                matched_bias = match_calibration_to_image(
                    master_bias, flat_shape, header, 'bias',
                    calibration_rebin_mode
                )[row_slice, :]
                data = data - matched_bias
            if master_dark is not None:
                matched_dark = match_calibration_to_image(
                    master_dark, flat_shape, header, 'dark',
                    calibration_rebin_mode
                )[row_slice, :]
                data = data - matched_dark * header['EXPTIME'] / dark_exposure
            return data / flat_norms[index]

        master_flat, header = _chunked_median_master(
            files, 'flat', calibrate_flat_block
        )
        hdu = fits.PrimaryHDU(master_flat, header=header)
        hdu.writeto(os.path.join('.', 'master_flat.fits'), overwrite=True)
        with fits.open(os.path.join('.', 'master_flat.fits'), mode='update') as hdul:
            hdul[0].header['FILTER'] = 'NONE'
        print(f'Master flat saved to: {os.path.join(os.getcwd(), "master_flat.fits")}')
        return CalibrationMaster(master_flat, header, 'flat')


def load_calibration_masters(calibration_rebin_mode='mean'):
    master_bias = bias()
    master_dark = dark(master_bias, calibration_rebin_mode)
    master_flat = flat(master_bias, master_dark,
                       calibration_rebin_mode=calibration_rebin_mode)
    return master_bias, master_dark, master_flat


def reduce_image(filename, master_bias=None, master_dark=None, master_flat=None,
                 site_location=None, calibration_rebin_mode='mean'):
    fd, hdr = fits.getdata(filename, header=True)
    fd = fd.astype(np.float64)
    data_exp = round(float(hdr['EXPTIME']), 2)
    half_exptime = data_exp / 2.
    if site_location is None:
        site_location = get_location()
    time_isot = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=site_location)
    time_jd = Time(time_isot.jd, format='jd', scale='utc', location=site_location) + half_exptime * u.second
    try:
        ra = hdr['TELRAD']
        dec = hdr['TELDECD']
    except KeyError:
        ra = hdr.get('MNTRAD', 0)
        dec = hdr.get('MNTDECD', 0)
    ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
    time_bary = time_jd.tdb + ltt_bary
    time_helio = time_jd.utc + ltt_helio

    if master_bias is not None:
        print(f'Subtracting master bias from {filename}')
        fd -= match_calibration_to_image(
            master_bias, fd.shape, hdr, 'bias', calibration_rebin_mode
        )
        print(f'After bias subtraction, mean pixel value for {filename}: {np.mean(fd)}')
    if master_dark is not None:
        fd -= match_calibration_to_image(
            master_dark, fd.shape, hdr, 'dark', calibration_rebin_mode
        ) * hdr['EXPTIME'] / 10
    if master_flat is not None:
        print(f'Dividing by master flat from {filename}')
        matched_flat = match_calibration_to_image(
            master_flat, fd.shape, hdr, 'flat', calibration_rebin_mode
        )
        bad_flat = (~np.isfinite(matched_flat)) | (matched_flat <= 0)
        if np.any(bad_flat):
            print(f'Masking {np.count_nonzero(bad_flat)} invalid master flat pixel(s)')
            matched_flat = matched_flat.copy()
            matched_flat[bad_flat] = np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            fd /= matched_flat

    return fd, hdr, os.path.basename(filename)


def reduce_images(prefix_filenames, masters=None, calibration_rebin_mode='mean'):
    if masters is None:
        masters = load_calibration_masters(calibration_rebin_mode)
    master_bias, master_dark, master_flat = masters

    reduced_data = []
    reduced_header_info = []
    filenames = []

    for filename in prefix_filenames:
        try:
            fd, hdr, basename = reduce_image(
                filename, master_bias, master_dark, master_flat,
                calibration_rebin_mode=calibration_rebin_mode
            )
            reduced_data.append(fd)
            reduced_header_info.append(hdr)
            filenames.append(basename)
        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

    return reduced_data, reduced_header_info, filenames
