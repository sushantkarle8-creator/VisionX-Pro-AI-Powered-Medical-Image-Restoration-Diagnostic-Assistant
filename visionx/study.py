"""Paired restoration/classification and reproducible single-study stress experiments."""
from datetime import datetime, timezone
import hashlib
import time
import numpy as np
from .chest import LABELS, MODEL_ID
from .images import validate_image, resize_like
from .processing import restore, degrade, Degradation, quality, METHODS


def analyze(image, method, classifier, model_runner=None, metadata=None, noise=0., seed=42):
    image = validate_image(image)
    if method not in METHODS:
        raise ValueError('Unknown restoration method.')
    config = Degradation('noise', noise, seed)
    observed = degrade(image, config) if noise else image.copy()
    start = time.perf_counter()
    restored = restore(observed, method, model_runner)
    restore_s = time.perf_counter() - start
    start = time.perf_counter()
    images = [observed, restored] + ([image] if noise else [])
    inference = classifier.predict(images)
    inference_s = time.perf_counter() - start
    delta = inference['scores'][1] - inference['scores'][0]
    rows = [{'finding': label, 'input_score': float(inference['scores'][0, i]),
             'restored_score': float(inference['scores'][1, i]), 'change': float(delta[i])}
            for i, label in enumerate(LABELS)]
    result = {'input': observed, 'restored': restored, 'reference': image if noise else None,
              'inference': inference, 'rows': rows,
              'record': {'schema_version': '3.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
                         'input_sha256': hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                         'hash_scope': 'float32 working-resolution grayscale pixels before optional synthetic noise',
                         'method': method, 'model': MODEL_ID, 'model_provenance': classifier.record,
                         'metadata': metadata or {}, 'synthetic_noise_sigma': noise, 'seed': seed,
                         'restoration_seconds': restore_s, 'classification_seconds': inference_s,
                         'timing_scope': 'warm-capable calls; model load excluded; includes CPU output transfer',
                         'device': classifier.device,
                         'mean_absolute_score_change': float(np.abs(delta).mean()),
                         'max_absolute_score_change': float(np.abs(delta).max()),
                         'largest_change_finding': LABELS[int(np.argmax(np.abs(delta)))],
                         'model_crops': inference['crops'],
                         'score_definition': 'Independent sigmoid model outputs, not calibrated disease probabilities',
                         'limitations': ['Research prototype; not a clinical diagnosis or triage system.',
                                         'No local training, clinical validation, or decision threshold calibration.',
                                         'CAMs are coarse class-contribution maps, not lesion segmentations.',
                                         'Public demo cases are not held-out test data; no accuracy is inferred.']}}
    if noise:
        result['record']['reference_metrics'] = {'input': quality(image, observed), 'restored': quality(image, restored)}
        clean_scores = inference['scores'][2]
        result['record']['reference_score_drift'] = {
            'input': float(np.abs(inference['scores'][0]-clean_scores).mean()),
            'restored': float(np.abs(inference['scores'][1]-clean_scores).mean())}
    return result


def sensitivity(image, classifier):
    """Fixed small perturbations; response range is NOT a calibrated uncertainty interval."""
    image = validate_image(image)
    variants = [image]
    names = ['Unchanged', 'Brightness -3%', 'Brightness +3%', 'Noise seed 17', 'Noise seed 29']
    variants += [np.clip(image*.97, 0, 1), np.clip(image*1.03, 0, 1)]
    variants += [degrade(image, Degradation('noise', .01, seed)) for seed in [17, 29]]
    pred = classifier.predict(variants)['scores']
    return {'variants': names, 'scores': pred.tolist(),
            'rows': [{'finding': label, 'baseline': float(pred[0, i]),
                      'minimum': float(pred[:, i].min()), 'maximum': float(pred[:, i].max()),
                      'range': float(np.ptp(pred[:, i]))} for i, label in enumerate(LABELS)],
            'interpretation': 'Sensitivity to four specified perturbations, not clinical confidence, robustness certification, or a population confidence interval.'}


def stress_experiment(reference, classifier, methods=('nlm', 'unsharp'), levels=(.025, .05, .1), seeds=(11, 29, 47), model_runner=None, progress=None):
    """Paired single-image noise experiment; same corrupted input for every method."""
    reference = validate_image(reference)
    if not methods or any(m not in METHODS for m in methods) or len(methods) > 4:
        raise ValueError('Choose one to four supported methods.')
    if not levels or not seeds or len(levels)*len(seeds) > 30:
        raise ValueError('Choose between 1 and 30 corruption conditions.')
    selected = list(dict.fromkeys(['identity', *methods]))
    clean_scores = classifier.predict([reference])['scores'][0]
    records = []
    total = len(levels)*len(seeds)
    for condition, (sigma, seed) in enumerate((s, t) for s in levels for t in seeds):
        observed = degrade(reference, Degradation('noise', sigma, seed))
        results, times = [], []
        for method in selected:
            started = time.perf_counter()
            results.append(restore(observed, method, model_runner))
            times.append(time.perf_counter()-started)
        scores = classifier.predict(results)['scores']
        for i, method in enumerate(selected):
            metric = quality(reference, results[i])
            records.append({'noise_sigma': sigma, 'seed': seed, 'method': method, **metric,
                            'mean_score_drift': float(np.abs(scores[i]-clean_scores).mean()),
                            'max_score_drift': float(np.abs(scores[i]-clean_scores).max()),
                            'restoration_seconds': times[i]})
        if progress:
            progress((condition+1)/total)
    summary = []
    for sigma in levels:
        for method in selected:
            group = [r for r in records if r['noise_sigma']==sigma and r['method']==method]
            drift = [r['mean_score_drift'] for r in group]
            summary.append({'noise_sigma': sigma, 'method': method, 'repeats': len(group),
                            'mean_ssim': float(np.mean([r['ssim'] for r in group])),
                            'mean_score_drift': float(np.mean(drift)),
                            'min_score_drift': float(min(drift)), 'max_score_drift': float(max(drift)),
                            'mean_restoration_seconds': float(np.mean([r['restoration_seconds'] for r in group]))})
    return {'records': records, 'summary': summary, 'levels': list(levels), 'seeds': list(seeds),
            'model': MODEL_ID, 'source_pixel_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
            'scope': 'One image, repeated synthetic Gaussian noise. Reference is the supplied image, not a guaranteed pristine scan. Drift is consistency with original model output, not diagnostic accuracy. Seed range is not a patient-level confidence interval.'}
