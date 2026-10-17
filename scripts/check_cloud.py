"""Cloud provisioning and real CPU inference smoke check (public example only)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visionx.bootstrap import ensure_chest_weights
from visionx.chest import load_chest
from visionx.medical import load_radiograph

path = ensure_chest_weights(ROOT / 'weights')
assert ensure_chest_weights(ROOT / 'weights') == path
image, _ = load_radiograph((ROOT / 'assets/00000001_000.png').read_bytes(), max_side=256)
result = load_chest(device='cpu').predict([image])
assert result['scores'].shape == (1, 18)
print('Verified provisioning, cached reuse, and 18 real CPU model outputs.')
