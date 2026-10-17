"""Local display conversion for single-frame, uncompressed frontal chest radiographs."""
from io import BytesIO
import numpy as np
from PIL import Image
from .images import load_image, validate_image


def load_radiograph(data, dicom=False, max_side=512):
    if not dicom:
        return load_image(data, max_side)
    if not data or len(data) > 40 * 1024 * 1024:
        raise ValueError('Use a DICOM file smaller than 40 MB.')
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
    try:
        ds = pydicom.dcmread(BytesIO(data))
    except Exception as exc:
        raise ValueError('Could not read this DICOM file.') from exc
    if str(getattr(ds, 'Modality', '')) not in {'DX', 'CR'}:
        raise ValueError('Only chest radiograph DX/CR DICOM is supported, not CT or MRI.')
    view = str(getattr(ds, 'ViewPosition', '')).upper()
    if view and view not in {'AP', 'PA'}:
        raise ValueError('Only frontal AP/PA radiographs are supported.')
    body = str(getattr(ds, 'BodyPartExamined', '')).upper()
    if body and body not in {'CHEST', 'THORAX'}:
        raise ValueError('The DICOM body part is not chest/thorax.')
    if int(getattr(ds, 'NumberOfFrames', 1)) != 1 or int(getattr(ds, 'SamplesPerPixel', 1)) != 1:
        raise ValueError('Use a single-frame grayscale radiograph.')
    if str(getattr(ds, 'PhotometricInterpretation', '')) not in {'MONOCHROME1', 'MONOCHROME2'}:
        raise ValueError('Unsupported DICOM photometric interpretation.')
    if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns') or int(ds.Rows)*int(ds.Columns) > 16_000_000:
        raise ValueError('Missing dimensions or image larger than 16 megapixels.')
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        raise ValueError('Compressed DICOM is not supported. Export an uncompressed radiograph.')
    try:
        raw = ds.pixel_array
        image = apply_voi_lut(apply_modality_lut(raw, ds), ds).astype(np.float32)
    except Exception as exc:
        raise ValueError('DICOM pixels or display transformation could not be decoded.') from exc
    valid = np.isfinite(image)
    if hasattr(ds, 'PixelPaddingValue'):
        pad = float(ds.PixelPaddingValue)
        limit = float(getattr(ds, 'PixelPaddingRangeLimit', pad))
        valid &= ~((raw >= min(pad, limit)) & (raw <= max(pad, limit)))
    if not valid.any():
        raise ValueError('No valid pixels remain after excluding padding.')
    lo, hi = float(image[valid].min()), float(image[valid].max())
    if hi <= lo:
        raise ValueError('The DICOM image has no intensity variation.')
    image = np.clip((image-lo)/(hi-lo), 0, 1)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        image = 1-image
    image[~valid] = 0
    height, width = image.shape
    preview = Image.fromarray(image)
    preview.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    result = validate_image(np.asarray(preview))
    # Strict whitelist: no names, dates, accession numbers, UIDs or free-text descriptions.
    return result, {'format': 'DICOM', 'modality': str(ds.Modality), 'view': view or 'unspecified',
                    'original_size': [width, height], 'working_size': [result.shape[1], result.shape[0]],
                    'conversion': 'modality LUT -> VOI LUT/window -> valid-pixel min/max -> MONOCHROME1 inversion if needed',
                    'privacy': 'Header identifiers omitted. Burned-in text may remain in pixels.'}
