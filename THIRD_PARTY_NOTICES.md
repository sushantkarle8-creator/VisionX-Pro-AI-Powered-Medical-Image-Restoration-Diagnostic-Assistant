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
