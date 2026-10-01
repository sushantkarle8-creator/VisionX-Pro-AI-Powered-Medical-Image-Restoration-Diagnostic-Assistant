"""Optional integration check using real official weights, with no Gemini calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from visionx.images import demo_image
from visionx.models import DEFAULT_WEIGHTS, load_runner
from visionx.processing import Degradation, run_experiment
from visionx.reporting import bundle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=Path("runs/model-smoke.zip"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    methods = ["swinir_denoise", "swinir_sr"]
    runners = {method: load_runner(method, str(args.weights.resolve()), "cpu") for method in methods}
    metadata = {"source": "64x64 synthetic geometric target; smoke test only",
                "models": {method: dict(runner.record, device=runner.device) for method, runner in runners.items()}}
    run = run_experiment(demo_image(64), methods, Degradation("noise", 25/255, 42),
                         lambda image, method: runners[method](image), metadata)
    assert run["outputs"]["swinir_denoise"].shape == (64, 64)
    assert run["outputs"]["swinir_sr"].shape == (128, 128)
    for output in run["outputs"].values():
        assert np.isfinite(output).all()
    # Exercise overlap and non-window-multiple edges against the same model.
    tiled = runners["swinir_denoise"](demo_image(39), tile=32, overlap=8)
    assert tiled.shape == (39,39) and np.isfinite(tiled).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bundle(run))
    print(json.dumps(run["results"], indent=2))
    print("Both checkpoints and odd-sized overlapping tiles passed.")
    print("Saved:", args.output.resolve())

if __name__ == "__main__":
    main()

