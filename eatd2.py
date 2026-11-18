"""Evaluate the EATD model and its source-selected threshold on EATD-Corpus 2.0."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ablation import load_dataset, metric_row, strict_profile
from fixed_final_transfer import fit_variant, score_variant, text_features_without_labels
from generalization import source_data


def main():
    ap = argparse.ArgumentParser()
    for name in ['source_audio', 'source_text', 'source_llm', 'source_vocal', 'source_root',
                 'source_vocal12', 'target_audio', 'target_text', 'target_llm', 'target_root',
                 'target_vocal12', 'outdir']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    source_ids, y, features = source_data(args)
    features[2] = strict_profile(args.source_vocal12, source_ids)
    fitted = fit_variant(source_ids, y, features, 'equal')
    threshold = fitted['thresholds']['source_oof_f1']
    np.savez_compressed(out / 'source_calibration.npz', subject_ids=source_ids,
                        scores=fitted['calibration'])
    (out / 'source_fit.json').write_text(json.dumps({
        'participants': len(y), 'threshold': threshold,
        'normalizers': fitted['normalizers'], 'fusion': 'equal'
    }, indent=2) + '\n')

    with np.load(args.target_vocal12, allow_pickle=False) as data:
        ids = data['subject_ids'].astype(str)
    target = [text_features_without_labels(args.target_text, ids),
              strict_profile(args.target_llm, ids),
              strict_profile(args.target_vocal12, ids)]
    scores = score_variant(fitted, np.column_stack([
        (model.decision(x) - mean) / sd
        for model, x, (mean, sd) in zip(fitted['models'], target, fitted['normalizers'])
    ]))
    data = load_dataset(args.target_audio, args.target_text, args.target_root,
                        threshold=53, expected_n=91, expected_neg=53, expected_pos=38,
                        strict_expected=True)
    np.testing.assert_array_equal(ids, np.asarray(data['ids']).astype(str))
    y_test = np.asarray(data['y'], dtype=int)
    pd.DataFrame(dict(subject_id=ids, label=y_test, score=scores, threshold=threshold,
                      model='FMEM', dataset='EATD_to_EATD2', calibration='pure_source',
                      operating_point='source_oof_f1')).to_csv(out / 'subject_predictions.csv', index=False)
    metrics = metric_row(y_test, scores, threshold=threshold)
    pd.DataFrame([dict(model='FMEM', **metrics)]).to_csv(out / 'summary.csv', index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
