from io import BytesIO
import json
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pytest
from visionx.images import demo_image, load_image
from visionx.processing import Degradation, degrade, quality, restore, run_experiment
from visionx.reporting import bundle, report_text

def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        load_image(b"not an image")

def test_aspect_ratio():
    buffer = BytesIO()
    Image.new("RGB", (400, 200)).save(buffer, "PNG")
    image, meta = load_image(buffer.getvalue(), max_side=128)
    assert image.shape == (64, 128)
    assert meta["original_size"] == [400, 200]

def test_16_bit_rejected():
    buffer = BytesIO()
    Image.fromarray(np.full((32, 32), 40000, dtype=np.uint16)).save(buffer, "PNG")
    with pytest.raises(ValueError, match="8-bit"):
        load_image(buffer.getvalue())

def test_multiframe_rejected():
    buffer = BytesIO()
    Image.new("L", (32,32)).save(buffer, "TIFF", save_all=True, append_images=[Image.new("L", (32,32))])
    with pytest.raises(ValueError, match="Multi-frame"):
        load_image(buffer.getvalue())

@pytest.mark.parametrize("kind", ["noise", "blur", "motion_blur", "low_contrast"])
def test_zero_degradation_is_identity(kind):
    image = demo_image(64)
    np.testing.assert_array_equal(degrade(image, Degradation(kind, 0)), image)

def test_noise_reproducibility():
    image = demo_image(64)
    a = degrade(image, Degradation("noise", .1, 17))
    np.testing.assert_array_equal(a, degrade(image, Degradation("noise", .1, 17)))
    assert not np.array_equal(a, degrade(image, Degradation("noise", .1, 18)))

@pytest.mark.parametrize("method", ["identity", "nlm", "unsharp", "clahe"])
def test_baselines(method):
    output = restore(degrade(demo_image(32), Degradation()), method)
    assert output.shape == (32, 32)
    assert np.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1

def test_metrics():
    image = demo_image(64)
    exact = quality(image, image)
    assert exact["exact_match"] and exact["psnr_db"] is None and exact["ssim"] == 1
    noisy = quality(image, degrade(image, Degradation()))
    assert not noisy["exact_match"] and noisy["ssim"] < 1 and noisy["psnr_db"] > 0

def test_no_reference_no_fake_metrics():
    run = run_experiment(demo_image(32), ["nlm"])
    assert run["reference"] is None
    assert all("ssim" not in row and "psnr_db" not in row for row in run["results"])
    assert "No clean reference" in report_text(run)

def test_nlm_improves_this_seeded_demo_only():
    run = run_experiment(demo_image(128), ["nlm"], Degradation("noise", .1, 42))
    baseline, denoised = run["results"]
    assert denoised["psnr_db"] > baseline["psnr_db"]
    assert denoised["ssim"] > baseline["ssim"]

def test_export_and_finite_json():
    run = run_experiment(demo_image(32), ["identity"], Degradation("noise", 0))
    with ZipFile(BytesIO(bundle(run))) as archive:
        assert {"metrics.json", "metrics.csv", "report.txt", "reference.png", "input.png", "results/identity.png"} <= set(archive.namelist())
        metadata = json.loads(archive.read("metrics.json"), parse_constant=lambda value: pytest.fail(value))
        assert metadata["results"][0]["exact_match"]
        assert metadata["results"][0]["psnr_db"] is None

def test_super_resolution_metric_alignment():
    image = demo_image(32)
    doubled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    assert np.isfinite(quality(image, doubled)["ssim"])

def test_missing_model_fails_explicitly():
    with pytest.raises(ValueError, match="SwinIR"):
        restore(demo_image(32), "swinir_denoise")


def test_constant_image_denoising_is_finite():
    image = np.full((32, 32), .5, dtype=np.float32)
    np.testing.assert_array_equal(restore(image, "nlm"), image)

