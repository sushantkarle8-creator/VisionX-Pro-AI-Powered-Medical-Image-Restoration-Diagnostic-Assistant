"""Download and convert one hash-pinned official XRV checkpoint. No uploaded pickle loads."""
import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path
import types
import urllib.request
import warnings

from visionx.chest import ROOT, WEIGHT_NAME, MODEL_ID, SOURCE_COMMIT

SOURCE_NAME = 'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
SOURCE_URL = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/' + SOURCE_NAME
SOURCE_SHA256 = '56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899'


def convert(source, destination):
    import torch
    from torch import nn
    from torchvision.models import densenet121
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Upstream checksum differs from the reviewed release. Refusing to deserialize.')
    class LegacyModule(nn.Module):
        pass
    allowed = {'torchxrayvision.models.'+name: LegacyModule
               for name in ['DenseNet', '_DenseBlock', '_DenseLayer', '_Transition']}
    for cls in [nn.Sequential, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Linear]:
        allowed[cls.__module__+'.'+cls.__name__] = cls
    allowed.update({'__builtin__.set': set, 'builtins.set': set,
                    'collections.OrderedDict': collections.OrderedDict,
                    'torch._utils._rebuild_tensor_v2': torch._utils._rebuild_tensor_v2,
                    'torch._utils._rebuild_tensor': torch._utils._rebuild_tensor,
                    'torch._utils._rebuild_parameter': torch._utils._rebuild_parameter,
                    'torch.FloatStorage': torch.FloatStorage, 'torch.LongStorage': torch.LongStorage})
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            key = module + '.' + name
            if key not in allowed:
                raise ValueError('Unapproved legacy checkpoint global: ' + key)
            return allowed[key]
    module = types.SimpleNamespace(Unpickler=RestrictedUnpickler, load=pickle.load, __name__='restricted_xrv')
    # This older upstream format contains module persistent IDs unsupported by weights_only.
    # Conversion is restricted to the exact published bytes and an explicit class allowlist.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
        legacy = torch.load(source, map_location='cpu', weights_only=False, pickle_module=module)
    for child in legacy.modules():
        if not hasattr(child, '_non_persistent_buffers_set'):
            child._non_persistent_buffers_set = set()
    state = legacy.state_dict()
    check = densenet121(weights=None, num_classes=18)
    check.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    check.load_state_dict(state, strict=True)
    temporary = destination.with_suffix('.part')
    torch.save(state, temporary)
    temporary.replace(destination)
    record = {'model': MODEL_ID, 'url': SOURCE_URL, 'upstream_sha256': SOURCE_SHA256,
              'architecture_commit': SOURCE_COMMIT, 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'training': 'Upstream pretrained model; no VisionX training or local calibration',
              'score_transform': 'sigmoid(logits); upstream operating-point normalization is NOT applied'}
    destination.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def setup(directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    source = directory / SOURCE_NAME
    if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        temporary = source.with_suffix('.part')
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise ValueError('Downloaded checkpoint failed verification.')
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    record = convert(source, directory / WEIGHT_NAME)
    print('Chest model ready:', record['model'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'weights')
    setup(parser.parse_args().directory)
