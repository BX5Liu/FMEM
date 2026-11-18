import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('FMEM_DATA_ROOT', '.'))
CODE = Path(__file__).resolve().parent
BASE = CODE.parent
OUT = Path(os.environ['FMEM_EDAIC_OUTPUT'])
MANIFEST = Path(os.environ['FMEM_EDAIC_MANIFEST'])
SR = 16000


def inputs():

    with MANIFEST.open() as handle:
        rows = [dict(subject_id=r['subject_id'],audio_path=r['audio_path'],text_path=r['text_path'],
                     split=r['official_split']) for r in csv.DictReader(handle)]
    if len(rows) != 275 or len({r['subject_id'] for r in rows}) != 275:
        raise ValueError('Expected the existing 275 distinct inputs')
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_wave(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    if rate != SR:
        divisor = gcd(rate, SR)
        wave = resample_poly(wave, SR // divisor, rate // divisor, axis=0).astype('float32')
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite input waveform')
    return wave


def select_clips(turns, duration, groups=3, per_group=2, max_seconds=30., min_seconds=3., allow_empty=False):

    selected = []
    for group in range(groups):
        left, right = duration * group / groups, duration * (group + 1) / groups
        candidates = []
        for turn in turns:
            start, end = max(left, turn[0]), min(right, turn[1])
            if end - start >= min_seconds:
                candidates.append((start, end))
        if not candidates:
            if allow_empty:
                selected.append([])
                continue
            raise ValueError('No eligible participant interval in an audio group')

        chosen = []
        for target in np.linspace(left, right, per_group):
            options = []
            for start, end in candidates:
                length = min(max_seconds, end - start)
                clip_start = max(start, min(float(target) - length / 2, end - length))
                interval = (float(clip_start), float(clip_start + length))
                if any(abs(interval[0] - old[0]) < .1 and abs(interval[1] - old[1]) < .1 for old in chosen):
                    continue
                options.append((abs((interval[0] + interval[1]) / 2 - target), -length, interval))
            if options:
                chosen.append(min(options)[2])
        if not chosen:
            raise ValueError('Empty clip selection')
        selected.append(chosen)
    return selected


def settings():
    return json.loads((OUT / 'settings.json').read_text())


def require_gpu():
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inference through the L40 Slurm scripts')

