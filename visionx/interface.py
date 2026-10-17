"""Presentation helpers. All imagery is rendered locally from the current study."""
import base64
from html import escape
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

CSS = '''<style>
.stApp{background:#0b0d0c;color:#eeeede} [data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{max-width:1460px;padding:2.2rem 3rem 4rem}
h1,h2,h3,p,label{font-family:Arial,Helvetica,sans-serif} h2{letter-spacing:-.04em;font-weight:500!important}
.vx-nav{display:flex;align-items:center;justify-content:space-between;padding:0 0 25px;border-bottom:1px solid #30342f}
.vx-brand{font-size:22px;font-weight:750;letter-spacing:-.8px;display:flex;align-items:center;gap:12px;color:#f1f0df}
.vx-symbol{color:#dfdbaa;font-size:30px;line-height:1}.vx-tag{font:10px 'Courier New',monospace;letter-spacing:1.7px;color:#a4aa9d}
.vx-badge{border:1px solid #474c3c;border-radius:4px;padding:7px 10px;color:#d8d9b5;font:10px monospace;letter-spacing:1px}
.vx-hero{position:relative;display:grid;grid-template-columns:1.5fr 1fr;gap:35px;padding:49px 0 38px;overflow:hidden}
.vx-hero::after{content:'';position:absolute;right:0;top:10px;width:220px;height:180px;background:radial-gradient(ellipse,#8e8d4d1c,transparent 70%);pointer-events:none}
.vx-hero h1{font-size:clamp(34px,4.7vw,67px);font-weight:500;letter-spacing:-3.5px;line-height:1.02;padding:11px 0 0;margin:0}.vx-hero h1 span{color:#a5aa9e}
.vx-intro{max-width:350px;align-self:end;color:#afb5aa;font-size:14px;line-height:1.75;padding:0 0 5px}
.vx-steps{display:flex;gap:18px;color:#e0ddba;font:10px monospace;letter-spacing:1px;margin-top:18px}
.vx-stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid #33372e;border-radius:6px;margin:8px 0 27px;background:#111410}
.vx-stat{padding:17px 21px;border-right:1px solid #33372e}.vx-stat:last-child{border-right:0}
.vx-stat b{display:block;font-size:23px;font-weight:500;letter-spacing:-.8px;color:#e6e3bc;margin-top:8px}.vx-stat small{font:9px monospace;letter-spacing:1.2px;color:#99a18f}
.vx-section{display:flex;align-items:center;justify-content:space-between;margin:20px 0 13px}.vx-section h3{font-size:17px;font-weight:500;letter-spacing:-.5px;margin:0;padding:0}.vx-section span{font:10px monospace;color:#939d8a}
.vx-panel{border:1px solid #343b30;padding:20px;background:#11150f;border-radius:6px}.vx-finding{padding:12px 0;border-bottom:1px solid #2d3329}.vx-finding:last-child{border-bottom:0}
.vx-finding-head{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#e5e7dd}.vx-finding-head em{font:11px monospace;color:#babf9e;font-style:normal}
.vx-track{height:4px;background:#272e21;margin-top:10px;position:relative;border-radius:2px}.vx-fill{height:4px;background:#dcd8a9;border-radius:2px}.vx-tick{position:absolute;top:-3px;width:2px;height:10px;background:#8bb2a4}
.vx-muted{font-size:11px;line-height:1.6;color:#9ba492}.vx-note{border-left:2px solid #d3cc9b;padding:12px 16px;margin:16px 0;color:#aeb6a5;font-size:12px;line-height:1.7;background:#151911}
.vx-footer{padding-top:22px;margin-top:35px;border-top:1px solid #343b30;font:10px monospace;color:#939d8a;display:flex;justify-content:space-between;gap:15px}
[data-testid="stSidebar"]{background:#11150f;border-right:1px solid #343b30}[data-baseweb="tab-list"]{gap:30px;border-bottom:1px solid #343b30}[data-baseweb="tab"]{font-size:13px}
.stButton>button[kind="primary"]{background:#dedbb0;color:#15180f;border:0;font-weight:600;border-radius:4px}.stButton>button{border-radius:4px;border-color:#424938}
[data-testid="stMetric"]{border:1px solid #343b30;border-radius:5px;padding:15px}[data-testid="stMetricValue"]{font-size:25px!important}
@media(max-width:800px){[data-testid="stMainBlockContainer"]{padding:1.5rem 1rem}.vx-hero{grid-template-columns:1fr;gap:20px;padding:30px 0}.vx-hero h1{letter-spacing:-2px}.vx-intro{max-width:none}.vx-stats{grid-template-columns:repeat(2,1fr)}.vx-stat:nth-child(2){border-right:0}.vx-nav .vx-tag{display:none}}
[data-testid="stToolbar"]{display:none}
@media(max-width:750px){[data-testid="stHorizontalBlock"]:has(iframe){flex-direction:column}[data-testid="stHorizontalBlock"]:has(iframe)>[data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important}.vx-stat{padding:12px 15px}.vx-stat b{font-size:20px}.vx-hero{padding:25px 0}}
</style>'''

def setup_style():
    st.markdown(CSS,unsafe_allow_html=True)

def image_url(image):
    array=np.asarray(image)
    if array.dtype!=np.uint8: array=np.rint(np.clip(array,0,1)*255).astype(np.uint8)
    output=BytesIO(); Image.fromarray(array).save(output,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()

def section(title,label=''):
    st.markdown(f'<div class="vx-section"><h3>{escape(title)}</h3><span>{escape(label)}</span></div>',unsafe_allow_html=True)

def compare_view(left,right,left_label='INPUT',right_label='RESTORED'):
    html='''<!DOCTYPE html><html><head><style>
    *{box-sizing:border-box}body{margin:0;background:#080a08;color:#e4e4cc;font-family:Arial,sans-serif}
    .viewer{height:475px;position:relative;overflow:hidden;border:1px solid #39402e;border-radius:6px;background:radial-gradient(ellipse,#161c16,#050705)}
    .scan{position:absolute;width:100%;height:100%;object-fit:contain;pointer-events:none}#upper{clip-path:inset(0 50% 0 0)}#divider{position:absolute;top:0;bottom:0;left:50%;width:1px;background:#dedbb0;pointer-events:none}
    #divider:after{content:'↔';position:absolute;top:50%;left:-17px;border:1px solid #dedbb0;background:#171d13;border-radius:50%;width:33px;height:33px;display:grid;place-items:center;color:#dedbb0}
    .badge{position:absolute;top:17px;padding:7px 9px;font:10px monospace;letter-spacing:1.5px;background:#0c100ddb;border:1px solid #46503a;border-radius:3px;pointer-events:none}.left{left:15px}.right{right:15px}
    input{position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize}.foot{display:flex;justify-content:space-between;font:10px monospace;letter-spacing:1px;color:#99a58d;padding:13px 3px}
    @media(max-width:500px){.viewer{height:350px}}
    </style></head><body><div class="viewer"><img class="scan" src="RIGHT" alt="Right comparison image"><img id="upper" class="scan" src="LEFT" alt="Left comparison image"><div id="divider"></div><span class="badge left">LEFT_LABEL</span><span class="badge right">RIGHT_LABEL</span><input type="range" min="0" max="100" value="50" aria-label="Image comparison position" oninput="document.getElementById('upper').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';document.getElementById('divider').style.left=this.value+'%';"></div><div class="foot"><span>DRAG TO COMPARE</span><span>LOCAL IMAGE WORKSPACE</span></div></body></html>'''
    html=html.replace('RIGHT_LABEL',escape(right_label)).replace('LEFT_LABEL',escape(left_label)).replace('src="RIGHT"','src="'+image_url(right)+'"').replace('src="LEFT"','src="'+image_url(left)+'"')
    components.html(html,height=515)

def findings_panel(rows):
    body='<div class="vx-panel"><div class="vx-tag">HIGHEST INPUT RESPONSES</div>'
    for row in sorted(rows,key=lambda r:-r['input_score'])[:6]:
        body+=f'''<div class="vx-finding"><div class="vx-finding-head"><span>{escape(row['finding'])}</span><em>{row['input_score']:.3f} → {row['restored_score']:.3f}</em></div><div class="vx-track"><div class="vx-fill" style="width:{row['input_score']*100:.2f}%"></div><div class="vx-tick" style="left:{row['restored_score']*100:.2f}%"></div></div></div>'''
    body+='<p class="vx-muted">Ivory: input · Green marker: restored<br>Scale 0–1. Independent model scores, not disease probabilities. Ranking is not a diagnosis.</p></div>'
    st.markdown(body,unsafe_allow_html=True)
