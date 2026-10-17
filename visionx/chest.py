"""Frozen chest X-ray inference and exact class activation maps (GAP + linear head)."""
from functools import lru_cache
from pathlib import Path
import hashlib
import threading

import numpy as np
from skimage.transform import resize
from .images import validate_image

LABELS = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
          'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural Thickening',
          'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
          'Lung Opacity', 'Enlarged Cardiomediastinum']
ROOT = Path(__file__).resolve().parents[1]
WEIGHT_NAME = 'chest-densenet121.pt'
MODEL_ID = 'torchxrayvision/densenet121-res224-all'
SOURCE_COMMIT = '4f9c7753880a391273e051f3857bce5089dfade5'


def preprocess(image):
    """Match XRV: [-1024,1024], center square crop, skimage resize to 224."""
    image = validate_image(image)
    height, width = image.shape
    side = min(height, width)
    top, left = height // 2 - side // 2, width // 2 - side // 2
    crop = image[top:top+side, left:left+side]
    normalized = (crop * 2048.0 - 1024.0)[None]
    tensor = resize(normalized, (1, 224, 224), mode='constant', preserve_range=True).astype(np.float32)
    return tensor, {'left': left, 'top': top, 'side': side, 'source_shape': [height, width]}


def display_crop(image):
    tensor, _ = preprocess(image)
    return np.clip((tensor[0] + 1024) / 2048, 0, 1)


def activation_maps(features, classifier_weights):
    """Signed CAM spatial means equal logits minus bias; positive part is displayed."""
    return np.einsum('kc,bchw->bkhw', classifier_weights, features)


class ChestModel:
    def __init__(self, weights_dir, device='cpu'):
        import json
        import torch
        from torch import nn
        from torchvision.models import densenet121
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Use cpu or cuda.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable on this computer.')
        path = Path(weights_dir) / WEIGHT_NAME
        record_path = path.with_suffix('.json')
        if not path.exists() or not record_path.exists():
            raise ValueError('Chest model is not installed. Run python setup_chest.py.')
        self.record = json.loads(record_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.record['sha256']:
            raise ValueError('Chest checkpoint checksum mismatch. Run setup_chest.py again.')
        self.torch, self.device = torch, device
        self.lock = threading.Lock()
        # Avoid CPU oversubscription on interactive workloads.
        if device == 'cpu':
            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        self.model = densenet121(weights=None, num_classes=len(LABELS))
        self.model.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.model.to(device).eval().requires_grad_(False)

    def predict(self, images):
        if not images or len(images) > 16:
            raise ValueError('Provide between 1 and 16 images per batch.')
        torch = self.torch
        prepared = [preprocess(image) for image in images]
        batch = torch.from_numpy(np.stack([item[0] for item in prepared])).to(self.device)
        with self.lock, torch.inference_mode():
            features = torch.relu(self.model.features(batch))
            pooled = features.mean(dim=(-2, -1))
            logits = self.model.classifier(pooled)
            scores = torch.sigmoid(logits).cpu().numpy()
            cams = activation_maps(features.cpu().numpy(), self.model.classifier.weight.cpu().numpy())
        return {'scores': scores, 'cams': cams, 'crops': [item[1] for item in prepared]}


@lru_cache(maxsize=2)
def load_chest(weights_dir=str(ROOT / 'weights'), device='cpu'):
    return ChestModel(weights_dir, device)


def overlay(image, cam, scale=None, opacity=.5):
    """CAM overlay on the actual 224x224 model crop, not an anatomical segmentation."""
    gray = display_crop(image)
    positive = np.maximum(np.asarray(cam), 0)
    scale = max(float(positive.max()) if scale is None else float(scale), 1e-8)
    heat = np.clip(resize(positive / scale, gray.shape, preserve_range=True), 0, 1)
    # Indigo -> amber -> coral; transparent where contribution is zero.
    red = np.clip(heat * 2.2, 0, 1)
    green = np.clip(1.8 * heat - .35, 0, .85)
    blue = np.clip(.65 - heat, 0, .65)
    color = np.stack([red, green, blue], axis=-1)
    alpha = heat[..., None] * opacity
    return np.rint((gray[..., None] * (1-alpha) + color * alpha) * 255).astype(np.uint8)
