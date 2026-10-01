from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

UPSTREAM_COMMIT = "6545850fbf8df298df73d81f3e8cba638787c8bd"
CHECKPOINTS = {
    "swinir_denoise": "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    "swinir_sr": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",
}
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def verify_checkpoint(path):
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not sidecar.exists():
        raise ValueError("Checkpoint missing. Run python download_models.py first.")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise ValueError("Checkpoint checksum mismatch. Download the model again.")
    return record


class SwinIRRunner:
    def __init__(self, method, weights_dir, device):
        try:
            import torch
            from .vendor.network_swinir import SwinIR
        except ImportError as exc:
            raise ValueError("Install requirements-models.txt to enable SwinIR.") from exc
        self.torch = torch
        self.lock = threading.Lock()
        self.method = method
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Choose cpu, cuda, or auto.")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Select CPU.")
        path = Path(weights_dir) / CHECKPOINTS[method]
        self.record = verify_checkpoint(path)
        denoise = method == "swinir_denoise"
        self.scale = 1 if denoise else 2
        self.model = SwinIR(
            upscale=self.scale, in_chans=1 if denoise else 3,
            img_size=128 if denoise else 64, window_size=8, img_range=1.,
            depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
            mlp_ratio=2, upsampler="" if denoise else "nearest+conv",
            resi_connection="1conv",
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def __call__(self, image, method=None, tile=128, overlap=16):
        from .images import validate_image
        image = validate_image(image)
        if method is not None and method != self.method:
            raise ValueError("Runner and requested method do not match.")
        if tile < 8 or tile % 8 or not 0 <= overlap < tile:
            raise ValueError("Tile must be a multiple of 8; overlap must be smaller than the tile.")
        torch = self.torch
        height, width = image.shape
        canvas = np.zeros((height * self.scale, width * self.scale), np.float32)
        counts = np.zeros_like(canvas)
        def starts(length):
            end = max(0, length - tile)
            return sorted(set([*range(0, end + 1, tile - overlap), end]))
        with self.lock, torch.inference_mode():
            for top in starts(height):
                for left in starts(width):
                    patch = image[top:top + tile, left:left + tile]
                    tensor = torch.from_numpy(patch.copy())[None, None]
                    if self.method == "swinir_sr":
                        tensor = tensor.repeat(1, 3, 1, 1)
                    output = self.model(tensor.to(self.device)).float().cpu().numpy()[0].mean(axis=0)
                    y, x = top * self.scale, left * self.scale
                    ph, pw = output.shape
                    canvas[y:y+ph, x:x+pw] += output
                    counts[y:y+ph, x:x+pw] += 1
        return np.clip(canvas / counts, 0, 1)


@lru_cache(maxsize=4)
def load_runner(method, weights_dir=str(DEFAULT_WEIGHTS), device="cpu"):
    if method not in CHECKPOINTS:
        raise ValueError("Unknown SwinIR model.")
    return SwinIRRunner(method, weights_dir, device)
