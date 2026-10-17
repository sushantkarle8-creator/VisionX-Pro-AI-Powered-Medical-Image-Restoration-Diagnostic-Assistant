"""Streamlit Community Cloud entry point; run with Python 3.12."""
import os
from pathlib import Path
import runpy
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['VISIONX_CLOUD'] = '1'

st.set_page_config(page_title='VisionX Pro | Chest imaging', page_icon='◈', layout='wide')


@st.cache_resource(show_spinner=False)
def initialize_model():
    from visionx.bootstrap import ensure_chest_weights
    ensure_chest_weights(ROOT / 'weights')
    return True


try:
    with st.spinner('Preparing the verified chest model. First startup may take a few minutes…'):
        initialize_model()
except Exception:
    st.error('The chest model could not be prepared. Retry, or check the deployment logs. No predictions were generated.')
    import logging
    logging.exception('Chest model initialization failed')
    if st.button('Retry model setup'):
        st.rerun()
    st.stop()

runpy.run_path(str(ROOT / 'app.py'), run_name='__main__')
