import hashlib
import json
import os

import streamlit as st

from visionx.gemini import review_quality
from visionx.images import demo_image, load_image, png_bytes
from visionx.models import load_runner
from visionx.processing import Degradation, METHODS, run_experiment
from visionx.reporting import bundle, report_text

st.set_page_config(page_title="VisionX Pro", page_icon=":material/visibility:", layout="wide")
st.title("VisionX Pro")
st.write("Restore images. Compare methods. Measure what changed.")
st.caption("An image-restoration research prototype. Results and optional AI commentary are not diagnostic evidence.")

with st.sidebar:
    st.header("Set up a run")
    mode = st.radio("Workflow", ["Controlled experiment", "Restore an image"])
    source = st.radio("Image source", ["Synthetic demo", "Upload image"])
    upload = st.file_uploader("PNG, JPEG, or 8-bit TIFF", type=["png", "jpg", "jpeg", "tif", "tiff"]) if source == "Upload image" else None
    max_side = st.select_slider("Maximum working size", options=[128, 256, 512, 1024], value=256)
    methods = st.multiselect("Restoration methods", options=list(METHODS)[1:], default=["nlm"],
                            format_func=lambda key: METHODS[key])
    st.caption("The unprocessed baseline is always included. SwinIR requires the optional model setup.")
    device = st.selectbox("Compute device", ["cpu", "cuda", "auto"])
    config = None
    if mode == "Controlled experiment":
        kind = st.selectbox("Simulated degradation", ["noise", "blur", "motion_blur", "low_contrast"])
        severity = st.slider("Severity", 0., 1., 25/255, .01)
        seed = st.number_input("Random seed", min_value=0, max_value=2**32-1, value=42, step=1)
        config = Degradation(kind, severity, int(seed))
        st.caption("Noise: standard deviation = severity. Blur: sigma = 3 x severity. Motion: horizontal kernel. Contrast: scales by 1 - severity.")

raw = upload.getvalue() if upload else b""
fingerprint = hashlib.sha256(raw + json.dumps({
    "mode": mode, "source": source, "size": max_side, "methods": methods,
    "device": device, "config": config.__dict__ if config else None
}, sort_keys=True).encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state.pop("run", None)
    st.session_state.pop("review", None)
    st.session_state["fingerprint"] = fingerprint

if source == "Synthetic demo":
    st.info("The demo is a geometric test image, not an X-ray. No patient data or API key is needed.")
    preview, _ = load_image(png_bytes(demo_image()), max_side)
else:
    preview = None
    if upload:
        try:
            preview, _ = load_image(raw, max_side)
        except ValueError as exc:
            st.error(str(exc))
if preview is not None:
    st.image(png_bytes(preview), caption="Working image", width=320)

if st.button("Run restoration", type="primary", disabled=preview is None or not methods):
    try:
        with st.spinner("Processing images and measuring results"):
            if source == "Synthetic demo":
                # Apply the selected working-size policy to the demo as well.
                image, metadata = load_image(png_bytes(demo_image()), max_side)
                metadata["source"] = "synthetic geometric demonstration; not medical data"
            else:
                image, metadata = load_image(raw, max_side)
                metadata["input_sha256"] = hashlib.sha256(raw).hexdigest()
            runners = {method: load_runner(method, device=device) for method in methods if method.startswith("swinir_")}
            metadata["models"] = {method: dict(runner.record, device=runner.device) for method, runner in runners.items()}
            st.session_state["run"] = run_experiment(
                image, methods, config, lambda img, method: runners[method](img), metadata
            )
            st.session_state.pop("review", None)
    except Exception as exc:
        st.error("The run could not finish: " + str(exc))
        st.session_state.pop("run", None)

run = st.session_state.get("run")
if run:
    st.subheader("Results")
    if run["reference"] is None:
        st.info("There is no clean reference for this image. PSNR and SSIM are intentionally unavailable.")
    else:
        st.caption("Scores use the clean working-resolution reference. Super-resolution outputs are downsampled for comparison; these scores do not measure true high-resolution recovery.")
    display_rows = []
    for row in run["results"]:
        display = {"Method": row["label"], "Output size": f"{row['output_width']} x {row['output_height']}",
                   "Time (s)": round(row["seconds"], 3)}
        if "ssim" in row:
            display["PSNR (dB)"] = "infinity (exact match)" if row["exact_match"] else f"{row['psnr_db']:.3f}"
            display["SSIM"] = round(row["ssim"], 4)
            display["SSIM change"] = f"{row['ssim'] - run['results'][0]['ssim']:+.4f}"
        display_rows.append(display)
    st.dataframe(display_rows, hide_index=True, use_container_width=True)
    chosen = st.selectbox("Inspect a result", options=list(run["outputs"]),
                          index=1 if len(run["outputs"]) > 1 else 0, format_func=lambda key: METHODS[key])
    columns = st.columns(3 if run["reference"] is not None else 2)
    if run["reference"] is not None:
        columns[0].image(png_bytes(run["reference"]), caption="Clean reference", use_container_width=True)
    columns[-2].image(png_bytes(run["input"]), caption="Input to restoration", use_container_width=True)
    columns[-1].image(png_bytes(run["outputs"][chosen]), caption=METHODS[chosen], use_container_width=True)
    st.download_button("Download images, metrics and report", bundle(run), "visionx-results.zip", mime="application/zip")
    with st.expander("Read the measured report"):
        st.text(report_text(run))
    with st.expander("Optional Gemini image-quality review"):
        st.write("This sends the input and selected processed image to Google. Use only images you are authorized to share. This review is qualitative and can be wrong.")
        model_id = st.text_input("Available Gemini model ID", value=os.environ.get("GEMINI_MODEL", ""))
        key = st.text_input("Gemini API key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
        consent = st.checkbox("I authorize sending these two images to Google for this review.")
        review_id = hashlib.sha256((fingerprint + chosen + model_id).encode()).hexdigest()
        if st.button("Request image-quality review", disabled=not (consent and key and model_id)):
            try:
                with st.spinner("Requesting qualitative review"):
                    text = review_quality(run["input"], run["outputs"][chosen], key, model_id, consent=consent)
                    st.session_state["review"] = {"id": review_id, "text": text}
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        saved = st.session_state.get("review", {})
        if saved.get("id") == review_id:
            st.caption("Experimental AI-generated commentary; not a diagnosis.")
            st.write(saved["text"])
