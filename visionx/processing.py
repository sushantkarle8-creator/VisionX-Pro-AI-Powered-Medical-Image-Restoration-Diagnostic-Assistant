from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from skimage import exposure, restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .images import resize_like, validate_image

METHODS = {
    "identity": "Unprocessed baseline",
    "nlm": "Non-local means denoising",
    "unsharp": "Unsharp masking",
    "clahe": "Adaptive contrast (CLAHE)",
    "swinir_denoise": "SwinIR grayscale denoising",
    "swinir_sr": "SwinIR 2x super-resolution",
}


@dataclass(frozen=True)
class Degradation:
    kind: str = "noise"
    severity: float = 25 / 255
    seed: int = 42

    def __post_init__(self):
        if self.kind not in {"noise", "blur", "motion_blur", "low_contrast"}:
            raise ValueError("Unknown degradation.")
        if not np.isfinite(self.severity) or not 0 <= self.severity <= 1:
            raise ValueError("Severity must be between zero and one.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be an unsigned 32-bit integer.")


def degrade(image, config: Degradation):
    image = validate_image(image)
    strength = config.severity
    if strength == 0:
        return image.copy()
    if config.kind == "noise":
        result = image + np.random.default_rng(config.seed).normal(0, strength, image.shape)
    elif config.kind == "blur":
        result = gaussian_filter(image, sigma=3 * strength)
    elif config.kind == "motion_blur":
        size = 2 * max(1, round(7 * strength)) + 1
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1 / size
        result = convolve(image, kernel, mode="reflect")
    else:
        result = (image - .5) * (1 - strength) + .5
    return np.clip(result, 0, 1).astype(np.float32)


def restore(image, method, model_runner=None):
    image = validate_image(image)
    if method == "identity":
        result = image.copy()
    elif method == "nlm":
        if float(np.ptp(image)) < 1e-7:
            return image.copy()
        estimate = float(restoration.estimate_sigma(image, channel_axis=None))
        sigma = max(estimate, 1e-4) if np.isfinite(estimate) else 1e-4
        result = restoration.denoise_nl_means(
            image, h=.8 * sigma, sigma=sigma, patch_size=5,
            patch_distance=6, fast_mode=True, channel_axis=None, preserve_range=True,
        )
    elif method == "unsharp":
        result = image + .8 * (image - gaussian_filter(image, sigma=1))
    elif method == "clahe":
        result = exposure.equalize_adapthist(image, clip_limit=.01)
    elif method in {"swinir_denoise", "swinir_sr"}:
        if model_runner is None:
            raise ValueError("SwinIR requires optional dependencies and a downloaded checkpoint.")
        result = model_runner(image, method)
    else:
        raise ValueError("Unknown restoration method.")
    return validate_image(np.clip(result, 0, 1))


def quality(reference, candidate):
    reference, candidate = validate_image(reference), validate_image(candidate)
    candidate = resize_like(candidate, reference)
    mse = float(np.mean((reference.astype(np.float64) - candidate) ** 2))
    # Null plus exact_match avoids nonstandard Infinity in JSON/CSV.
    psnr = None if mse == 0 else float(peak_signal_noise_ratio(reference, candidate, data_range=1))
    return {"psnr_db": psnr, "ssim": float(structural_similarity(reference, candidate, data_range=1)),
            "mse": mse, "exact_match": mse == 0}


def run_experiment(image, methods, degradation=None, model_runner=None, metadata=None):
    image = validate_image(image)
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Select at least one valid restoration method.")
    reference = image if degradation is not None else None
    observed = degrade(image, degradation) if degradation is not None else image.copy()
    selected = list(dict.fromkeys(["identity", *methods]))
    rows, outputs = [], {}
    for method in selected:
        started = time.perf_counter()
        output = restore(observed, method, model_runner)
        elapsed = time.perf_counter() - started
        row = {"method": method, "label": METHODS[method], "seconds": round(elapsed, 4),
               "output_width": output.shape[1], "output_height": output.shape[0]}
        if reference is not None:
            row.update(quality(reference, output))
        rows.append(row)
        outputs[method] = output
    return {"mode": "experiment" if reference is not None else "restore",
            "degradation": asdict(degradation) if degradation else None,
            "metadata": metadata or {}, "input": observed, "reference": reference,
            "results": rows, "outputs": outputs}
