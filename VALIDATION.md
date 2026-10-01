# Validation of the rebuilt project

Validated locally on Windows with Python 3.12.14, NumPy 2.3.5, Pillow 11.3.0,
SciPy 1.16.3, scikit-image 0.25.2, Streamlit 1.50.0, PyTorch 2.7.1,
torchvision 0.22.1, and timm 1.0.22.

- 27 automated tests passed: processing, invalid-input, reference-free, export, Gemini mock, and app tests.
- CLI controlled-experiment and direct-restoration runs.
- Real CPU inference with both official checkpoints using strict state-dictionary loading.
- Overlapping-tile inference on a 39 x 39 input.
- Browser inspection of the running Streamlit demo.
- Gemini requests tested with mocked responses only; no live API key was used.

## Reproduce the real-model check

~~~text
python -m pip install -r requirements-models.txt
python download_models.py --model all
python scripts/smoke_models.py
~~~

## Actual smoke-test results

64 x 64 geometric target, additive Gaussian noise sigma 25/255, seed 42, CPU, four threads.
These values are from one synthetic smoke test and are **not benchmark or medical results**.

| Method | Output | PSNR at reference size | SSIM at reference size |
|---|---|---:|---:|
| Unprocessed noisy input | 64 x 64 | 20.395 dB | 0.5120 |
| SwinIR grayscale denoising | 64 x 64 | 38.350 dB | 0.9921 |
| SwinIR 2x super-resolution | 128 x 128 | 31.929 dB | 0.9576 |

The super-resolution output is downsampled for scoring; its larger dimensions do not establish
recovery of true high-resolution detail. See examples/model-smoke/metrics.json for full records.

The 256 x 256 classical-method demo also records methods that made the noisy input worse.
This is intentional: the app reports measurements instead of assuming every operation improves
an image. See examples/classical-demo/metrics.json and report.txt.

## Limits

No medical dataset, external clinical review, disease labels, model training, GPU benchmark,
or live Gemini validation was performed. The upstream architecture produces deprecation warnings
for an older timm import and torch.meshgrid signature; inference succeeded with the pinned versions.

