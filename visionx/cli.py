import argparse
import hashlib
from pathlib import Path

from .images import demo_image, load_image, png_bytes
from .models import DEFAULT_WEIGHTS, load_runner
from .processing import Degradation, METHODS, run_experiment
from .reporting import bundle, report_text


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible VisionX restoration experiment.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--demo", action="store_true", help="Use a synthetic geometric target.")
    parser.add_argument("--mode", choices=["experiment", "restore"], default="experiment")
    parser.add_argument("--degradation", choices=["noise", "blur", "motion_blur", "low_contrast"], default="noise")
    parser.add_argument("--severity", type=float, default=25/255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=["nlm", "unsharp", "clahe"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=Path("runs/latest.zip"))
    args = parser.parse_args()
    try:
        if args.demo:
            image, metadata = load_image(png_bytes(demo_image()), args.max_side)
            metadata["source"] = "synthetic geometric demonstration; not medical data"
        else:
            raw = args.input.read_bytes()
            image, metadata = load_image(raw, args.max_side)
            metadata["input_sha256"] = hashlib.sha256(raw).hexdigest()
        runners = {method: load_runner(method, str(args.weights.resolve()), args.device)
                   for method in args.methods if method.startswith("swinir_")}
        metadata["models"] = {method: dict(runner.record, device=runner.device) for method, runner in runners.items()}
        config = Degradation(args.degradation, args.severity, args.seed) if args.mode == "experiment" else None
        run = run_experiment(image, args.methods, config,
                             lambda img, method: runners[method](img), metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(bundle(run))
        print(report_text(run))
        print("\nSaved: " + str(args.output.resolve()))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, "VisionX: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
