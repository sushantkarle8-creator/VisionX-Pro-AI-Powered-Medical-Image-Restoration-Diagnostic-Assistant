"""Run real local model inference and save reproducible smoke-test evidence."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from visionx.chest import load_chest,ROOT
from visionx.images import load_image
from visionx.models import load_runner
from visionx.study import analyze,sensitivity,stress_experiment
from visionx.evidence import serializable,pdf_report

output=ROOT/'examples'/'chest-analysis';output.mkdir(parents=True,exist_ok=True)
model=load_chest()
image,_=load_image((ROOT/'assets'/'00000001_000.png').read_bytes(),256)
model.predict([image]) # Warm up separately from the recorded run.
study=analyze(image,'nlm',model,metadata={'source':'Public NIH example; not held-out validation'},noise=.05)
study['sensitivity']=sensitivity(study['input'],model)
(output/'evidence.json').write_text(json.dumps(serializable(study),indent=2,allow_nan=False),encoding='utf-8')
(output/'report.pdf').write_bytes(pdf_report(study))
experiment=stress_experiment(image,model)
(output/'paired-experiment.json').write_text(json.dumps(experiment,indent=2,allow_nan=False),encoding='utf-8')
runner=load_runner('swinir_denoise')
deep=analyze(image,'swinir_denoise',model,runner,noise=.05)
(output/'swinir-evidence.json').write_text(json.dumps(serializable(deep),indent=2,allow_nan=False),encoding='utf-8')
assert len(study['rows'])==18 and study['inference']['cams'].shape==(3,18,7,7)
assert np.isfinite(deep['inference']['scores']).all()
print(json.dumps({'classification_seconds':study['record']['classification_seconds'],
                  'nlm_seconds':study['record']['restoration_seconds'],
                  'swinir_seconds':deep['record']['restoration_seconds'],
                  'nlm_reference_metrics':study['record']['reference_metrics'],
                  'nlm_score_drift':study['record']['reference_score_drift'],
                  'swinir_score_drift':deep['record']['reference_score_drift'],
                  'stress_rows':len(experiment['records'])},indent=2))
