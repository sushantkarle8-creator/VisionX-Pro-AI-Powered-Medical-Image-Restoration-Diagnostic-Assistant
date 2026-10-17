"""Verified, serialized model provisioning for ephemeral hosting."""
import hashlib
import json
from pathlib import Path
import threading

from .chest import WEIGHT_NAME

_setup_lock = threading.Lock()


def ensure_chest_weights(directory):
    """Reuse a verified conversion, otherwise run the hash-pinned setup process."""
    from setup_chest import SOURCE_SHA256, setup
    directory = Path(directory)
    checkpoint = directory / WEIGHT_NAME
    record_path = checkpoint.with_suffix('.json')

    def valid():
        try:
            record = json.loads(record_path.read_text(encoding='utf-8'))
            return (record['upstream_sha256'] == SOURCE_SHA256 and
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record['sha256'])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    with _setup_lock:
        if not valid():
            setup(directory)
        if not valid():
            raise ValueError('Prepared checkpoint failed verification.')
    return checkpoint
