from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from visionx.chest import preprocess, activation_maps, overlay, LABELS
from visionx.study import analyze, sensitivity, stress_experiment
from visionx.medical import load_radiograph
from visionx.evidence import evidence_zip, pdf_report

class FakeClassifier:
    record={'model':'test-double'}
    device='cpu'
    def predict(self,images):
        values=np.array([np.mean(x) for x in images],dtype=np.float32)
        scores=np.repeat(values[:,None],18,axis=1)
        return {'scores':scores,'cams':np.ones((len(images),18,7,7),np.float32),
                'crops':[preprocess(x)[1] for x in images]}

def target():
    return np.tile(np.linspace(.1,.9,40,dtype=np.float32),(32,1))

def test_preprocess_matches_crop_and_normalization():
    x=np.zeros((224,300),np.float32);x[:,38:262]=1
    transformed,crop=preprocess(x)
    assert crop['left']==38 and crop['side']==224
    assert transformed.shape==(1,224,224)
    np.testing.assert_allclose(transformed,1024)

def test_cam_identity_matches_linear_head():
    rng=np.random.default_rng(4)
    features=rng.random((2,8,7,7));weights=rng.normal(size=(18,8))
    cams=activation_maps(features,weights)
    np.testing.assert_allclose(cams.mean(axis=(-2,-1)),features.mean(axis=(-2,-1))@weights.T,atol=1e-12)

def test_zero_cam_overlay_is_grayscale():
    image=overlay(target(),np.zeros((7,7)))
    np.testing.assert_array_equal(image[:,:,0],image[:,:,1])
    assert image.shape==(224,224,3)

def test_direct_image_has_no_fake_reference_scores():
    result=analyze(target(),'identity',FakeClassifier())
    assert result['reference'] is None
    assert 'reference_metrics' not in result['record']
    assert result['record']['max_absolute_score_change']==0

def test_paired_experiment_and_seed_reproducibility():
    a=analyze(target(),'identity',FakeClassifier(),noise=.05,seed=7)
    b=analyze(target(),'identity',FakeClassifier(),noise=.05,seed=7)
    np.testing.assert_array_equal(a['input'],b['input'])
    assert a['record']['reference_score_drift']['input']==a['record']['reference_score_drift']['restored']
    experiment=stress_experiment(target(),FakeClassifier(),['identity','nlm'],levels=(.05,),seeds=(1,2))
    assert len(experiment['records'])==4 and len(experiment['summary'])==2
    assert experiment['summary'][0]['repeats']==2

def test_sensitivity_ranges_include_baseline():
    result=sensitivity(target(),FakeClassifier())
    assert len(result['variants'])==5
    assert all(r['minimum']<=r['baseline']<=r['maximum'] for r in result['rows'])

def make_dicom(**changes):
    from pydicom.dataset import FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,generate_uid
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=generate_uid();meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality='DX';ds.ViewPosition='PA';ds.BodyPartExamined='CHEST';ds.PatientName='PRIVATE^NAME'
    ds.Rows=32;ds.Columns=32;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
    ds.PixelData=np.arange(1024,dtype='<u2').reshape(32,32).tobytes()
    for key,value in changes.items(): setattr(ds,key,value)
    output=BytesIO();ds.save_as(output,enforce_file_format=True);return output.getvalue()

def test_dicom_inversion_and_no_identifiers():
    normal,metadata=load_radiograph(make_dicom(),dicom=True)
    inverse,_=load_radiograph(make_dicom(PhotometricInterpretation='MONOCHROME1'),dicom=True)
    np.testing.assert_allclose(normal+inverse,1,atol=1e-6)
    assert 'PRIVATE' not in str(metadata) and 'PatientName' not in metadata

@pytest.mark.parametrize('change',[{'Modality':'CT'},{'ViewPosition':'LAT'},{'NumberOfFrames':2},{'BodyPartExamined':'HAND'}])
def test_dicom_outside_scope_rejected(change):
    with pytest.raises(ValueError): load_radiograph(make_dicom(**change),dicom=True)

def test_evidence_exports_measured_scores():
    from zipfile import ZipFile
    import json
    result=analyze(target(),'nlm',FakeClassifier(),noise=.05)
    assert pdf_report(result).startswith(b'%PDF')
    with ZipFile(BytesIO(evidence_zip(result))) as z:
        data=json.loads(z.read('evidence.json'))
        assert len(data['findings'])==18
        assert 'reference.png' in z.namelist()

def test_workspace_analysis_and_stale_result_invalidation(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import visionx.chest as chest
    monkeypatch.setattr(chest,'load_chest',lambda **kw:FakeClassifier())
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'),default_timeout=30).run()
    assert not app.exception
    next(b for b in app.button if b.label=='Analyze study').click().run()
    assert not app.exception
    assert len(app.session_state['study']['rows'])==18
    next(r for r in app.radio if r.label=='Study source').set_value('Upload radiograph').run()
    assert 'study' not in app.session_state
    assert next(b for b in app.button if b.label=='Analyze study').disabled
