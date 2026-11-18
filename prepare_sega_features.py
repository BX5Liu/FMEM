"""Index the released SEGA tensors by EATD participant."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ablation import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--audio-npz', required=True)
    parser.add_argument('--text-npz', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_dataset(args.audio_npz, args.text_npz, args.data_root,
                        threshold=53, strict_expected=True)
    labels = dict(zip(map(str, data['ids']), data['y']))
    expected_hash = json.loads((Path(__file__).parent / 'upstream/sega_features.json').read_text())['feature_sha256']
    fields = {
        'text': 'text_simple_roberta_768.npy',
        'audio': 'audio_new_features_b_3_len_88.npz',
        'question': 'questions_b_3_len_768.npz',
        'gender': 'gender_b.npz',
        'emotion': 'emotion_probs_b_3_7.npz',
    }
    arrays = {field: [] for field in fields}
    ids = []
    for prefix, cohort, n in [('train', 't', 83), ('valid', 'v', 79)]:
        group = sorted([i for i in labels if i.startswith(cohort + '_')],
                       key=lambda i: int(i.split('_')[1]))
        if len(group) != n:
            raise ValueError('SEGA participant counts differ from EATD')
        with np.load(args.feature_dir / f'{prefix}_labels.npz') as z:
            np.testing.assert_array_equal(z['arr_0'].astype(int), [labels[i] for i in group])
        for field, suffix in fields.items():
            file = args.feature_dir / f'{prefix}_{suffix}'
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected_hash[file.name]:
                raise ValueError(f'Unexpected SEGA feature file: {file.name}')
            if file.suffix == '.npz':
                with np.load(file) as z:
                    x = z['arr_0']
            else:
                x = np.load(file)
            if len(x) != n:
                raise ValueError(f'Unexpected number of rows in {file.name}')
            arrays[field].append(x.astype(np.float32))
        ids.extend(group)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(args.output, subject_ids=np.asarray(ids),
                        labels=np.asarray([labels[i] for i in ids]),
                        **{key: np.concatenate(parts) for key, parts in arrays.items()})


if __name__ == '__main__':
    main()
