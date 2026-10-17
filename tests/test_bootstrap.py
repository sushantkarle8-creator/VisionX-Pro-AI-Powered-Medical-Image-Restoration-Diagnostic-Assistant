import hashlib
import json

import pytest

import setup_chest
from visionx.bootstrap import ensure_chest_weights
from visionx.chest import WEIGHT_NAME


def test_provision_reuse_and_repair(tmp_path, monkeypatch):
    calls = []

    def provision(directory):
        calls.append(directory)
        path = directory / WEIGHT_NAME
        path.write_bytes(b'test-only-checkpoint')
        path.with_suffix('.json').write_text(json.dumps({
            'upstream_sha256': setup_chest.SOURCE_SHA256,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }))

    monkeypatch.setattr(setup_chest, 'setup', provision)
    path = ensure_chest_weights(tmp_path)
    ensure_chest_weights(tmp_path)
    assert len(calls) == 1
    path.write_bytes(b'corrupted')
    ensure_chest_weights(tmp_path)
    assert len(calls) == 2


def test_failed_setup_does_not_mark_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_chest, 'setup', lambda directory: None)
    with pytest.raises(ValueError, match='failed verification'):
        ensure_chest_weights(tmp_path)


def test_cloud_entrypoint_and_cpu_controls(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import visionx.bootstrap as bootstrap

    monkeypatch.setenv('VISIONX_CLOUD', '1')
    monkeypatch.setattr(bootstrap, 'ensure_chest_weights', lambda directory: True)
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'cloud/app.py'), default_timeout=30).run()
    assert not app.exception
    device = next(s for s in app.selectbox if s.label == 'Compute device')
    assert device.options == ['cpu']
    restoration = next(s for s in app.selectbox if s.label == 'Restoration method')
    assert not any('swinir' in option.lower() for option in restoration.options)
    assert any('cloud server' in c.value for c in app.caption)
