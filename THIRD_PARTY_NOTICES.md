# Third-party components

The architecture in visionx/vendor/network_swinir.py is copied without modification from
https://github.com/JingyunLiang/SwinIR at commit 6545850fbf8df298df73d81f3e8cba638787c8bd.
Original copyright comments are retained. The Apache-2.0 license is included in
visionx/vendor/LICENSE-SwinIR. SwinIR architecture and pretrained weights are upstream research.

The checkpoint downloader uses the upstream v0.0 release. This project performs no training.
See upstream terms for weights and dependencies.

Base version: Sushant Karle's VisionX Pro repository, commit
a1bb2ad05736c01ec0a134d37a61dfcd4540fad2. This revision restructures the original workflow
and replaces unsupported clinical claims with measured image-restoration experiments.

## Chest X-ray analysis

TorchXRayVision model and preprocessing reference: https://github.com/mlmed/torchxrayvision
at commit 4f9c7753880a391273e051f3857bce5089dfade5. Its main library is Apache-2.0 licensed.
VisionX adapts the documented preprocessing and integrates the official pretrained release using
torchvision's matching DenseNet-121 architecture. Weights are downloaded separately, not committed.
The legacy converter is restricted to a fixed official release hash. Runtime uses a plain state dict.

Torchvision: https://github.com/pytorch/vision (BSD-3-Clause).
CAM: Zhou et al., Learning Deep Features for Discriminative Localization, CVPR 2016,
https://arxiv.org/abs/1512.04150. Our CAM implementation uses the standard GAP/linear-head identity.

Demo images: NIH ChestX-ray14, distributed in the TorchXRayVision tests directory.
Original dataset: Wang et al., ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks
on Weakly-Supervised Classification and Localization of Common Thorax Diseases, CVPR 2017.
Exact sources and hashes are in assets/provenance.json. NIH public-domain data; attribution retained.
These are demonstration examples with possible training overlap, not held-out evaluation cases.

The current main app replaces free-form diagnostic text with traceable model outputs.
No clinical accuracy or authorship of upstream models is claimed.
