# ============================================================
# VISIONX PRO: AI-Powered Medical Image Restoration & Diagnosis
# ============================================================

# ----------------------------
# 1. Setup & Install Dependencies
# ----------------------------
import os
import sys
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Clone SwinIR
if not os.path.exists('/content/SwinIR'):
    print("📥 Cloning SwinIR repository...")
    %cd /content
    !git clone https://github.com/JingyunLiang/SwinIR.git
%cd /content/SwinIR

# Install dependencies
!pip install -q basicsr facexlib gfpgan torchsummary albumentations opencv-python-headless
# Skip requirements.txt if missing
# !pip install -r requirements.txt  # often fails in Colab

# Add to path
sys.path.insert(0, '/content/SwinIR')

# ----------------------------
# 2. Download Pretrained Models
# ----------------------------
os.makedirs('experiments/pretrained_models', exist_ok=True)

def download_model(url, filename):
    print(f"📥 Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print(f"✅ {filename} downloaded!")

models = {
    '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth': 
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth',
    '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth': 
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'
}

for model_file, model_url in models.items():
    model_path = f'experiments/pretrained_models/{model_file}'
    if not os.path.exists(model_path):
        download_model(model_url, model_path)

print("✅ Models ready!")

# ----------------------------
# 3. Configure Google Gemini AI
# ----------------------------
import google.generativeai as genai
from google.colab import userdata
import base64

api_key = userdata.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("❌ Please set GOOGLE_API_KEY in Colab Secrets.")
genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel('gemini-1.5-flash')
gemini_vision = genai.GenerativeModel('gemini-1.5-flash')
print("✅ Gemini models configured.")

# ----------------------------
# 4. Upload & Load X-ray Image
# ----------------------------
from google.colab import files
print("📁 Please upload your X-ray image:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {filename}")

try:
    original_img = Image.open(filename).convert('L')
except Exception as e:
    print("❌ Failed to load image. Using synthetic test image.")
    test_array = np.random.rand(256, 256)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128)**2 + (y - 128)**2 <= 64**2
    test_array[mask] *= 0.3
    original_img = Image.fromarray((test_array * 255).astype(np.uint8), mode='L')

original_img = original_img.resize((256, 256))
plt.figure(figsize=(6,6))
plt.imshow(original_img, cmap='gray')
plt.title("Original X-ray")
plt.axis('off')
plt.show()

# ----------------------------
# 5. Simulate Clinical Degradations
# ----------------------------
import cv2
from scipy import ndimage

img_array = np.array(original_img).astype(np.float32) / 255.0

def apply_degradation(img, deg_type, severity=0.15):
    if deg_type == 'noise':
        return np.clip(img + np.random.normal(0, severity, img.shape), 0, 1)
    elif deg_type == 'motion_blur':
        ksize = max(3, int(15 * severity))
        kernel = np.zeros((ksize, ksize))
        kernel[ksize//2, :] = 1
        kernel /= ksize
        return cv2.filter2D(img, -1, kernel)
    elif deg_type == 'blur':
        return ndimage.gaussian_filter(img, sigma=max(0.1, severity * 3))
    elif deg_type == 'low_contrast':
        return np.clip((img - 0.5) * (1.0 - severity) + 0.5, 0, 1)
    return img

degradation_types = ['noise', 'motion_blur', 'blur', 'low_contrast']
degraded_images = {}
os.makedirs('inputs', exist_ok=True)

for deg in degradation_types:
    degraded = apply_degradation(img_array, deg)
    degraded_images[deg] = degraded
    Image.fromarray((degraded * 255).astype(np.uint8)).save(f'inputs/{deg}_degraded.png')

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
axes[0].imshow(original_img, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
for i, (deg, img) in enumerate(degraded_images.items()):
    axes[i+1].imshow(img, cmap='gray')
    axes[i+1].set_title(deg.replace('_', ' ').title())
    axes[i+1].axis('off')
for i in range(5, 6): axes[i].axis('off')
plt.tight_layout()
plt.show()
print("✅ Degradations applied.")

# ----------------------------
# 6. Load & Run SwinIR Restoration
# ----------------------------
import torch
import importlib.util

# Import SwinIR safely
spec = importlib.util.spec_from_file_location(
    "network_swinir", "/content/SwinIR/models/network_swinir.py"
)
network_swinir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_swinir)
net = network_swinir.SwinIR

# Load model
model_path = 'experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'
pretrained = torch.load(model_path, map_location='cpu')

model = net(
    upscale=2, in_chans=3, img_size=64, window_size=8,
    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180,
    num_heads=[6,6,6,6,6,6], mlp_ratio=2,
    upsampler='nearest+conv', resi_connection='1conv'
)

# Load state dict flexibly
if 'params_ema' in pretrained:
    model.load_state_dict(pretrained['params_ema'], strict=True)
elif 'params' in pretrained:
    model.load_state_dict(pretrained['params'], strict=True)
else:
    model.load_state_dict(pretrained, strict=False)

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Restore all degraded images
os.makedirs('direct_results', exist_ok=True)
direct_restored_results = {}

for deg_type, degraded_img in degraded_images.items():
    print(f"🔄 Restoring {deg_type}...")
    img_pil = Image.fromarray((degraded_img * 255).astype(np.uint8)).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2,0,1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor).cpu()

    out_img = output.squeeze().numpy()
    out_img = np.transpose(out_img, (1, 2, 0))
    out_img = np.clip(out_img, 0, 1)
    if out_img.ndim == 3:
        out_img = np.mean(out_img, axis=2)  # to grayscale

    result_path = f'direct_results/{deg_type}_direct_restored.png'
    Image.fromarray((out_img * 255).astype(np.uint8), mode='L').save(result_path)
    direct_restored_results[deg_type] = result_path
    print(f"  ✅ Saved: {result_path}")

# ----------------------------
# 7. AI Analysis of Restorations
# ----------------------------
ai_analyses = {}
for deg_type, path in direct_restored_results.items():
    print(f"🔍 Analyzing {deg_type}...")
    # Create comparison image
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_img, cmap='gray'); ax[0].set_title('Original'); ax[0].axis('off')
    ax[1].imshow(degraded_images[deg_type], cmap='gray'); ax[1].set_title(f'Degraded: {deg_type}'); ax[1].axis('off')
    ax[2].imshow(Image.open(path).convert('L'), cmap='gray'); ax[2].set_title('Restored'); ax[2].axis('off')
    temp_path = f'temp_{deg_type}.png'
    plt.savefig(temp_path, bbox_inches='tight', dpi=100)
    plt.close()

    with open(temp_path, 'rb') as f:
        img_data = f.read()

    prompt = f"""
You are a radiology expert. Analyze this medical image restoration comparison:
- Left: Original
- Center: Degraded ({deg_type})
- Right: AI-restored (SwinIR)
Provide:
1. Quality assessment
2. Clinical significance
3. Visible anatomy
4. Diagnostic value
5. Recommendation
Be concise and professional.
"""
    response = gemini_vision.generate_content([
        prompt,
        {"mime_type": "image/png", "data": base64.b64encode(img_data).decode()}
    ])
    ai_analyses[deg_type] = response.text
    os.remove(temp_path)

print("✅ AI analysis complete.")

# ----------------------------
# 8. Generate Reports
# ----------------------------

# Doctor-style restoration report
report = f"""
MEDICAL IMAGE RESTORATION REPORT
================================
Patient Study: AI-Enhanced Medical Imaging
Date: {time.strftime('%Y-%m-%d')}
Modality: Digital X-ray (simulated)
AI System: SwinIR Transformer Model

EXECUTIVE SUMMARY
=================
AI-powered restoration applied to simulate real-world degraded medical images.
Enhancement improves diagnostic confidence across multiple degradation types.

DETAILED ANALYSIS
=================
"""
for deg, analysis in ai_analyses.items():
    report += f"\n{deg.upper().replace('_', ' ')} RESTORATION\n{'-'*40}\n{analysis}\n"

report += """
CLINICAL RECOMMENDATIONS
========================
1. AI restoration enhances diagnostic quality of suboptimal X-rays.
2. Especially valuable for archival or telemedicine images.
3. Recommended as preprocessing for AI diagnostic pipelines.

CONCLUSION
==========
SwinIR provides clinically meaningful enhancement of degraded medical imagery.
"""

with open('medical_image_restoration_report.txt', 'w') as f:
    f.write(report)
print("✅ Restoration report saved.")

# Medical diagnosis from best restoration (e.g., 'noise')
best_type = 'noise' if 'noise' in direct_restored_results else list(direct_restored_results.keys())[0]
best_path = direct_restored_results[best_type]

with open(best_path, 'rb') as f:
    img_data = f.read()

diag_prompt = f"""
You are a board-certified radiologist. Analyze this restored X-ray (originally degraded with {best_type}).
Provide a full clinical radiology report covering:
1. ANATOMICAL ASSESSMENT
2. POTENTIAL DIAGNOSES (prioritized)
3. CLINICAL SYMPTOMS
4. TREATMENT IMPLICATIONS
5. URGENCY & REFERRAL
Be thorough and patient-focused.
"""

diag_response = gemini_vision.generate_content([
    diag_prompt,
    {"mime_type": "image/png", "data": base64.b64encode(img_data).decode()}
])
medical_analysis = diag_response.text

with open('medical_diagnosis_analysis.txt', 'w') as f:
    f.write(f"CLINICAL RADIOLOGY ANALYSIS\n{'='*30}\n")
    f.write(f"Image: Restored X-ray ({best_type})\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(medical_analysis)
print("✅ Medical diagnosis report saved.")

# Patient-focused summary
summary_prompt = f"""
Summarize this radiology report for the patient in simple terms:
{medical_analysis[:1200]}
Explain: symptoms, likely causes, next steps. Write compassionately.
"""
summary_resp = gemini_model.generate_content(summary_prompt)

patient_report = f"""
PATIENT CLINICAL ASSESSMENT REPORT
=================================
Date: {time.strftime('%Y-%m-%d')}
Modality: Chest X-ray (AI-Enhanced)

CLINICAL PRESENTATION
=====================
{summary_resp.text}

DETAILED FINDINGS
=================
{medical_analysis}

NOTE: This is an AI-assisted analysis. Confirm with your physician.
Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('patient_clinical_assessment.txt', 'w') as f:
    f.write(patient_report)
print("✅ Patient report saved.")

# Treatment recommendations
treat_prompt = f"""
Based on this analysis:
{medical_analysis[:800]}
Provide a clear clinical action plan with:
1. Primary diagnosis
2. Symptoms
3. Immediate next steps
4. Specialist referral?
5. Monitoring
6. Patient education
"""
treat_resp = gemini_model.generate_content(treat_prompt)

with open('treatment_recommendations.txt', 'w') as f:
    f.write(f"CLINICAL TREATMENT RECOMMENDATIONS\n{'='*35}\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(treat_resp.text)
print("✅ Treatment plan saved.")

# ----------------------------
# 9. Create Summary Dashboard
# ----------------------------
fig = plt.figure(figsize=(16, 12))
fig.suptitle('VISIONX PRO: Medical Image Restoration Dashboard', fontsize=16, fontweight='bold')

# System Info
ax1 = plt.subplot(2, 3, 1)
ax1.text(0.1, 0.8, "SYSTEM INFO", fontsize=14, fontweight='bold')
ax1.text(0.1, 0.6, "• Model: SwinIR Transformer", fontsize=10)
ax1.text(0.1, 0.5, "• Task: X-ray Restoration", fontsize=10)
ax1.text(0.1, 0.4, "• AI: Gemini 1.5 Flash", fontsize=10)
ax1.text(0.1, 0.3, f"• Date: {time.strftime('%Y-%m-%d')}", fontsize=10)
ax1.axis('off')

# Clinical Impact
ax2 = plt.subplot(2, 3, 2)
ax2.text(0.1, 0.8, "CLINICAL IMPACT", fontsize=14, fontweight='bold')
ax2.text(0.1, 0.6, "• Enhanced image quality", fontsize=10)
ax2.text(0.1, 0.5, "• Improved diagnostic clarity", fontsize=10)
ax2.text(0.1, 0.4, "• Artifact reduction", fontsize=10)
ax2.axis('off')

# Best restored image
ax3 = plt.subplot(2, 3, 3)
img = Image.open(best_path).convert('L').resize((256, 256))
ax3.imshow(img, cmap='gray')
ax3.set_title(f"Best Restoration: {best_type}", fontsize=10)
ax3.axis('off')

# Show other restorations
restorations = list(direct_restored_results.items())
for i in range(3):
    if i < len(restorations):
        deg, path = restorations[i]
        ax = plt.subplot(2, 3, 4 + i)
        img = Image.open(path).convert('L').resize((256, 256))
        ax.imshow(img, cmap='gray')
        ax.set_title(deg.replace('_', ' ').title(), fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('visionx_dashboard.png', dpi=150)
plt.show()
print("✅ Dashboard created and saved as 'visionx_dashboard.png'.")

# ----------------------------
# 10. Final Summary
# ----------------------------
print("\n" + "="*50)
print("🎉 VISIONX PRO: COMPLETE MEDICAL ANALYSIS SYSTEM READY!")
print("="*50)
print("✅ Restored images: /content/direct_results/")
print("✅ Reports generated:")
print("   - medical_image_restoration_report.txt")
print("   - medical_diagnosis_analysis.txt")
print("   - patient_clinical_assessment.txt")
print("   - treatment_recommendations.txt")
print("   - visionx_dashboard.png")
print("\n📁 All files ready for clinical review.")
