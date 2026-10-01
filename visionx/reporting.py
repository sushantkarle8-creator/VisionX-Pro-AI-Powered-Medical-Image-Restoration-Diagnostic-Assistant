import csv
from datetime import datetime, timezone
from io import BytesIO, StringIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

from .images import png_bytes


def manifest(run):
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": run["mode"], "degradation": run["degradation"],
            "metadata": run["metadata"], "results": run["results"],
            "evaluation": "Against the working-resolution reference; 2x results are downsampled with bicubic interpolation."
                          if run["reference"] is not None else "No clean reference; PSNR and SSIM are not computed.",
            "limitations": ["A single-image experiment is not a benchmark or clinical validation.",
                            "Higher PSNR/SSIM does not establish diagnostic accuracy.",
                            "No original-resolution super-resolution ground truth is provided.",
                            "Timings cover the restore call; model loading is excluded when preloaded."]}


def report_text(run):
    info = manifest(run)
    lines = ["VISIONX PRO - RESTORATION EXPERIMENT", "", "Mode: " + info["mode"],
             info["evaluation"], "", "Results:"]
    baseline = run["results"][0]
    for row in run["results"]:
        line = f"- {row['label']}: {row['output_width']}x{row['output_height']}; {row['seconds']:.4f} s"
        if "ssim" in row:
            psnr = "infinity (exact match)" if row["exact_match"] else f"{row['psnr_db']:.3f} dB"
            line += f"; PSNR {psnr}; SSIM {row['ssim']:.4f}; SSIM change {row['ssim'] - baseline['ssim']:+.4f}"
        lines.append(line)
    lines += ["", "Configuration:", json.dumps(info["degradation"], indent=2), "", "Limitations:"]
    lines += ["- " + item for item in info["limitations"]]
    lines += ["", "This report describes an image-processing experiment. It contains no diagnosis or treatment recommendation."]
    return "\n".join(lines)


def bundle(run):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("report.txt", report_text(run))
        archive.writestr("metrics.json", json.dumps(manifest(run), indent=2, allow_nan=False))
        csv_buffer = StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=list(run["results"][0]))
        writer.writeheader()
        writer.writerows(run["results"])
        archive.writestr("metrics.csv", csv_buffer.getvalue())
        archive.writestr("input.png", png_bytes(run["input"]))
        if run["reference"] is not None:
            archive.writestr("reference.png", png_bytes(run["reference"]))
        for method, image in run["outputs"].items():
            archive.writestr(f"results/{method}.png", png_bytes(image))
    return output.getvalue()
