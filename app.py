"""VisionX Pro: chest radiograph restoration and explainable finding analysis."""
import hashlib
import json
import numpy as np
import streamlit as st
from visionx.chest import LABELS, ROOT, WEIGHT_NAME, load_chest, overlay
from visionx.medical import load_radiograph
from visionx.images import resize_like
from visionx.models import load_runner
from visionx.processing import METHODS
from visionx.study import analyze, sensitivity, stress_experiment
from visionx.evidence import evidence_zip, pdf_report
from visionx.interface import setup_style, section, compare_view, findings_panel

st.set_page_config(page_title='VisionX Pro | Chest imaging',page_icon='◈',layout='wide',initial_sidebar_state='collapsed')
setup_style()
st.markdown('''<div class="vx-nav"><div class="vx-brand"><span class="vx-symbol">◈</span> VISIONX <span style="font-weight:400;color:#a2ad96">PRO</span></div><span class="vx-tag">MEDICAL IMAGING / RESEARCH WORKSPACE</span><span class="vx-badge">LOCAL INFERENCE</span></div><div class="vx-hero"><div><div class="vx-tag">CHEST RADIOGRAPH INTELLIGENCE</div><h1>A clearer image.<br><span>Evidence you can inspect.</span></h1></div><div class="vx-intro">Restore chest X-rays. Explore model findings. See exactly how processing changes the prediction.<div class="vx-steps"><span>01 / RESTORE</span><span>02 / ANALYZE</span><span>03 / VERIFY</span></div></div></div>''',unsafe_allow_html=True)

controls=st.columns([3,1.2,1.2])
with controls[1].popover('Study controls',use_container_width=True):
    st.header('Study configuration')
    source=st.radio('Study source',['Public sample 01','Public sample 02','Upload radiograph'])
    upload=st.file_uploader('Frontal chest X-ray',type=['png','jpg','jpeg','tif','tiff','dcm']) if source=='Upload radiograph' else None
    confirmed=st.checkbox('This is a de-identified frontal chest X-ray that I am authorized to use.') if source=='Upload radiograph' else True
    method=st.selectbox('Restoration method',['nlm','swinir_denoise','identity','clahe','unsharp'],format_func=lambda m:METHODS[m])
    size=st.select_slider('Working resolution',options=[224,256,384,512],value=256)
    device=st.selectbox('Compute device',['cpu','cuda'])
    simulated=st.checkbox('Controlled noise experiment',help='Use the source image as reference and add reproducible Gaussian noise. This is not scanner-calibrated noise.')
    noise=st.slider('Noise standard deviation',.0,.2,.05,.005) if simulated else 0.
    seed=st.number_input('Noise seed',min_value=0,max_value=2**32-1,value=42) if simulated else 42
    st.caption('Frontal chest X-rays only. No CT, MRI, pediatric validation, or COVID-specific classifier. Uploaded scans remain on this computer.')

raw=b''; image=None; metadata={}
try:
    if source.startswith('Public'):
        filename='00000001_000.png' if source.endswith('01') else '00027426_000.png'
        raw=(ROOT/'assets'/filename).read_bytes()
        image,metadata=load_radiograph(raw,max_side=size)
        metadata.update(source='NIH ChestX-ray14 public example '+filename,evaluation_scope='Demonstration only; may overlap upstream training data.')
    elif upload:
        raw=upload.getvalue()
        image,metadata=load_radiograph(raw,dicom=upload.name.lower().endswith('.dcm'),max_side=size)
        metadata['source']='User-supplied radiograph; filename omitted'
    if raw: metadata['source_file_sha256']=hashlib.sha256(raw).hexdigest()
except (ValueError,OSError) as exc:
    st.error(str(exc))

fingerprint=hashlib.sha256(raw+json.dumps([source,method,size,device,noise,int(seed),confirmed]).encode()).hexdigest()
if st.session_state.get('study_fingerprint')!=fingerprint:
    for key in ['study','stress','pdf','bundle']: st.session_state.pop(key,None)
    st.session_state['study_fingerprint']=fingerprint
ready=(ROOT/'weights'/WEIGHT_NAME).exists()
controls[0].caption(f'{source}  /  {METHODS[method]}  /  {size}px working resolution')
run_clicked=controls[2].button('Analyze study',type='primary',use_container_width=True,disabled=image is None or not confirmed)
if run_clicked:
    try:
        with st.spinner('Restoring the image and running paired chest analysis...'):
            classifier=load_chest(device=device)
            restorer=load_runner(method,device=device) if method.startswith('swinir') else None
            model_metadata=dict(metadata)
            if restorer: model_metadata['restoration_model']=restorer.record
            st.session_state['study']=analyze(image,method,classifier,restorer,model_metadata,noise,int(seed))
            st.session_state.pop('pdf',None);st.session_state.pop('bundle',None)
    except Exception as exc:
        st.session_state.pop('study',None)
        st.error('Analysis could not finish: '+str(exc))
study=st.session_state.get('study')
elapsed=f"{study['record']['classification_seconds']:.2f}s" if study else 'READY' if ready else 'SETUP'
shift=f"{study['record']['max_absolute_score_change']:.3f}" if study else '—'
st.markdown(f'''<div class="vx-stats"><div class="vx-stat"><small>CHEST FINDING OUTPUTS</small><b>18</b></div><div class="vx-stat"><small>PAIRED CLASSIFICATION</small><b>{elapsed}</b></div><div class="vx-stat"><small>MAXIMUM SCORE SHIFT</small><b>{shift}</b></div><div class="vx-stat"><small>IMAGE PROCESSING</small><b>On-device</b></div></div>''',unsafe_allow_html=True)
workspace,lab,model_card=st.tabs(['Imaging workspace','Restoration research','Model & evidence'])
with workspace:
    viewer,findings=st.columns([1.85,1],gap='large')
    with viewer:
        section('Image workspace','FRONTAL CHEST / GRAYSCALE')
        if study:
            view=st.radio('View',['Image comparison','Class activation','Difference map'],horizontal=True,label_visibility='collapsed')
            if view=='Image comparison': compare_view(study['input'],study['restored'])
            elif view=='Class activation':
                target=st.selectbox('Finding to localize',LABELS,index=int(np.argmax(study['inference']['scores'][0])))
                i=LABELS.index(target);cams=study['inference']['cams'][:2,i]
                scale=max(float(np.maximum(cams,0).max()),1e-8)
                compare_view(overlay(study['input'],cams[0],scale),overlay(study['restored'],cams[1],scale),'INPUT CAM','RESTORED CAM')
                st.caption('Coarse 7 × 7 class contributions on the model’s center crop, not lesion boundaries. Shared scale across both images.')
            else:
                diff=np.abs(study['input']-resize_like(study['restored'],study['input'])); ceiling=max(float(diff.max()),1e-8)
                heat=np.stack([diff/ceiling,np.sqrt(diff/ceiling)*.7,np.zeros_like(diff)],axis=-1)
                compare_view(study['input'],heat,'INPUT','ABSOLUTE CHANGE')
                st.caption(f'Absolute pixel change, normalized for display. Maximum {ceiling:.4f} on a 0–1 scale. Not a disease or hallucination map.')
        elif image is not None: compare_view(image,image,'PUBLIC SAMPLE' if source.startswith('Public') else 'INPUT','AWAITING ANALYSIS')
        else: st.info('Choose a sample or upload a frontal radiograph in the sidebar.')
    with findings:
        section('Finding responses','DENSENET-121')
        if study:
            findings_panel(study['rows']);record=study['record']
            st.markdown(f'<div class="vx-note">Largest response change: <b>{record["largest_change_finding"]}</b><br>Absolute shift {record["max_absolute_score_change"]:.4f}. A change is a reason to inspect both images, not evidence of improvement.</div>',unsafe_allow_html=True)
            with st.expander('All 18 finding scores'): st.dataframe(study['rows'],hide_index=True,use_container_width=True)
        else:
            st.markdown('''<div class="vx-panel"><div class="vx-tag">A PAIRED ANALYSIS</div><h3 style="font-size:25px;letter-spacing:-1px;margin:20px 0;">One scan.<br>Two model readings.</h3><p class="vx-muted">The same frozen model reads your original and restored image. Compare 18 finding responses and inspect where each class draws its evidence.</p><hr style="border-color:#343b30;"><p class="vx-muted">01 &nbsp; Restore with NLM or SwinIR<br><br>02 &nbsp; Score both images in one batch<br><br>03 &nbsp; Inspect activation and score shifts</p></div>''',unsafe_allow_html=True)
            if not ready: st.info('Install the chest model once: python setup_chest.py')
        st.caption('Research use only. Scores are not calibrated disease probabilities. No automated diagnosis or treatment recommendation.')
    if study:
        if 'reference_metrics' in study['record']:
            m=study['record']['reference_metrics'];d=study['record']['reference_score_drift']
            section('Controlled experiment','KNOWN SYNTHETIC CORRUPTION');c=st.columns(4)
            c[0].metric('Input SSIM',f"{m['input']['ssim']:.3f}");c[1].metric('Restored SSIM',f"{m['restored']['ssim']:.3f}")
            c[2].metric('Input score drift',f"{d['input']:.4f}");c[3].metric('Restored score drift',f"{d['restored']:.4f}")
            st.caption('SSIM compares with the supplied source. Drift compares with its model response. Neither establishes clinical accuracy.')
        section('Sensitivity check','SMALL INPUT CHANGES')
        st.write('Does this model give a stable response to small changes in brightness and noise?')
        if st.button('Run sensitivity check'):
            with st.spinner('Evaluating five input versions in one batch...'):
                study['sensitivity']=sensitivity(study['input'],load_chest(device=device))
                st.session_state.pop('pdf',None);st.session_state.pop('bundle',None)
        if study.get('sensitivity'):
            st.dataframe(sorted(study['sensitivity']['rows'],key=lambda r:-r['range']),hide_index=True,use_container_width=True)
            st.caption(study['sensitivity']['interpretation'])
        section('Export the evidence','LOCAL DOWNLOAD')
        if st.button('Prepare research report'):
            try:
                st.session_state['pdf']=pdf_report(study);st.session_state['bundle']=evidence_zip(study)
            except ImportError: st.error('Install requirements-clinical.txt to enable PDF export.')
        if 'pdf' in st.session_state:
            a,b=st.columns(2)
            a.download_button('Download PDF report',st.session_state['pdf'],'visionx-research-report.pdf','application/pdf',use_container_width=True)
            b.download_button('Download complete evidence',st.session_state['bundle'],'visionx-evidence.zip','application/zip',use_container_width=True)

with lab:
    section('Does a cleaner image preserve the model response?','PAIRED EXPERIMENT')
    st.write('Use the selected source as reference. Add three noise levels, repeat with three fixed seeds, and compare restoration methods on exactly the same inputs.')
    experiment_methods=st.multiselect('Methods to compare',['nlm','unsharp','clahe','swinir_denoise'],default=['nlm','unsharp'],format_func=lambda m:METHODS[m])
    stress_fingerprint=fingerprint+json.dumps(experiment_methods)
    if st.session_state.get('stress_fingerprint')!=stress_fingerprint:
        st.session_state.pop('stress',None)
        st.session_state['stress_fingerprint']=stress_fingerprint
    st.caption('Unprocessed baseline included. 9 conditions per method. SwinIR takes longer on CPU. Single-image stress test, not a clinical benchmark.')
    if st.button('Run paired experiment',disabled=image is None or not confirmed or not experiment_methods):
        try:
            progress=st.progress(0.)
            runner=load_runner('swinir_denoise',device=device) if 'swinir_denoise' in experiment_methods else None
            st.session_state['stress']=stress_experiment(image,load_chest(device=device),experiment_methods,model_runner=runner,progress=progress.progress)
            progress.empty()
        except Exception as exc: st.error('Experiment could not finish: '+str(exc))
    if st.session_state.get('stress'):
        import pandas as pd
        exp=st.session_state['stress'];frame=pd.DataFrame(exp['summary']);a,b=st.columns(2)
        colors=['#dfdbaa','#89b3a3','#b5a1ca','#cd946c','#7596b7'][:frame.method.nunique()]
        a.write('Structural similarity ↑');a.line_chart(frame.pivot(index='noise_sigma',columns='method',values='mean_ssim'),color=colors)
        b.write('Drift from reference response ↓');b.line_chart(frame.pivot(index='noise_sigma',columns='method',values='mean_score_drift'),color=colors)
        st.dataframe(frame,hide_index=True,use_container_width=True);st.caption(exp['scope'])
        st.download_button('Download experiment JSON',json.dumps(exp,indent=2,allow_nan=False),'visionx-paired-experiment.json','application/json')

with model_card:
    section('Every output has a source.','MODEL CARD / V3')
    st.markdown('''**Finding model:** TorchXRayVision DenseNet-121, `densenet121-res224-all`. Upstream weights trained on multiple chest X-ray datasets. VisionX integrates this pretrained model; it does not claim to have trained it.

**Input:** single frontal chest radiograph, grayscale, center-cropped to square, resized to 224 × 224, normalized to [-1024, 1024]. Cropping can omit peripheral anatomy. DX/CR DICOM previews apply modality/display transforms; identifying header tags are excluded from exports.

**Outputs:** 18 independent sigmoid scores. No local calibration, clinical thresholds, diagnostic accuracy, COVID-specific prediction, or pediatric performance has been established.

**Localization:** class activation mapping (CAM) from final features and linear classifier weights. Native maps are 7 × 7, so highlights are coarse and may reflect spurious associations.

**Restoration:** fast non-local means, or an upstream SwinIR transformer trained for generic grayscale denoising. Medical-domain validation is still required. The original remains available.

**Privacy:** processing is local; no cloud image upload. Uploads stay in session memory. Exported pixels may still contain burned-in identifiers.

**Demo data:** two public NIH ChestX-ray14 images distributed with TorchXRayVision. They may overlap training data and cannot establish generalization.

**Research next step:** patient-disjoint external validation with AUROC/AUPRC, calibration and subgroup analysis before/after restoration, plus expert review of detail preservation.

[TorchXRayVision](https://mlmed.org/torchxrayvision/) · [CAM](https://arxiv.org/abs/1512.04150) · [SwinIR](https://github.com/JingyunLiang/SwinIR) · [Reconstruction instability](https://pmc.ncbi.nlm.nih.gov/articles/PMC7720232/)''')
    if study:
        with st.expander('Exact run configuration and model hashes'): st.json(study['record'])
st.markdown('<div class="vx-footer"><span>VISIONX PRO / CHEST IMAGING RESEARCH</span><span>PRETRAINED MODELS · MEASURED OUTPUTS · LOCAL PROCESSING</span></div>',unsafe_allow_html=True)

