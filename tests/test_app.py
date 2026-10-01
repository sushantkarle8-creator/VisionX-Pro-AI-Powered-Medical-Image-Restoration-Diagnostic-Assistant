from pathlib import Path
from streamlit.testing.v1 import AppTest
APP = Path(__file__).resolve().parents[1] / "app.py"

def test_demo_and_configuration_invalidation():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Run restoration").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert app.session_state["run"]["mode"] == "experiment"
    next(r for r in app.radio if r.label == "Workflow").set_value("Restore an image").run()
    assert not app.exception
    assert "run" not in app.session_state
    next(b for b in app.button if b.label == "Run restoration").click().run()
    assert not app.exception
    assert all("ssim" not in row for row in app.session_state["run"]["results"])

def test_upload_requires_image():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    next(r for r in app.radio if r.label == "Image source").set_value("Upload image").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Run restoration").disabled
