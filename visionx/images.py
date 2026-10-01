from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def validate_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("Use a grayscale image at least 8 pixels wide and high.")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image values must be finite and between zero and one.")
    return image


def load_image(data: bytes, max_side=512):
    """Decode a standard 8-bit image; reject unsupported medical/intensity formats."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a nonempty image smaller than 10 MB.")
    if not 32 <= max_side <= 1024:
        raise ValueError("Working resolution must be between 32 and 1024 pixels.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "TIFF"}:
                    raise ValueError("Use a PNG, JPEG, or single-frame TIFF image.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are unsupported; export a single frame.")
                if source.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    raise ValueError("Export an 8-bit preview first. Raw 16-bit, float, and DICOM data are unsupported.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16-megapixel limit.")
                if min(source.size) < 8:
                    raise ValueError("Image is too small; both dimensions must be at least 8 pixels.")
                source = ImageOps.exif_transpose(source)
                original_size = list(source.size)
                if source.mode in {"RGBA", "LA", "P"}:
                    rgba = source.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    source = Image.alpha_composite(background, rgba).convert("RGB")
                source = source.convert("L")
                source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                image = validate_image(np.asarray(source, dtype=np.float32) / 255)
                return image, {"original_size": original_size, "working_size": list(source.size),
                               "conversion": "8-bit grayscale; aspect ratio preserved"}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This file cannot be decoded as a supported image.") from exc


def png_bytes(image):
    image = validate_image(image)
    buffer = BytesIO()
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def resize_like(image, reference):
    """Align spatial resolution only. This does not register different scenes."""
    image, reference = validate_image(image), validate_image(reference)
    if image.shape == reference.shape:
        return image
    resized = Image.fromarray(image).resize(
        (reference.shape[1], reference.shape[0]), Image.Resampling.BICUBIC
    )
    return np.clip(np.asarray(resized, dtype=np.float32), 0, 1)


def demo_image(size=256):
    """An explicitly synthetic geometric target, not a scan or patient image."""
    y, x = np.mgrid[:size, :size] / size
    image = 0.12 + 0.18 * x
    image[(x - .32) ** 2 + (y - .36) ** 2 < .17 ** 2] = .8
    image[(x > .58) & (x < .85) & (y > .18) & (y < .58)] = .6
    stripes = (y > .72) & (y < .87) & (x > .1) & (x < .9)
    image[stripes] = .4 + .3 * np.sin(2 * np.pi * 16 * x[stripes])
    return image.astype(np.float32)
