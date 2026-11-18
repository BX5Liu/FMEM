import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from ablation import (best_f1_threshold, fit_score, BranchSpec, metric_row,
                      load_fold_manifest, strict_profile)
from edaic.common import MANIFEST, OUT as TIMESTAMP_OUT, inputs
from generalization import source_data
from round2_evaluate import score_branch


RULES = ['quantile_f1', 'exact_f1', 'exact_bacc']


def threshold_grid(y, scores):
    scores = np.asarray(scores, float)
    unique = np.unique(scores)
    grid = np.r_[np.nextafter(unique[0], -np.inf),
                 unique[:-1] + np.diff(unique) / 2,
                 np.nextafter(unique[-1], np.inf)]
    selected = scores[:, None] >= grid
    tp = (selected & (y[:, None] == 1)).sum(axis=0)
    fp = selected.sum(axis=0) - tp
    npos = int(y.sum())
    nneg = len(y) - npos
    return grid, 2 * tp / (npos + tp + fp), (tp / npos + 1 - fp / nneg) / 2


def exact_choice(grid, objective, secondary):
    indices = np.flatnonzero(np.isclose(objective, objective.max(), atol=1e-12, rtol=0))
    return float(max((float(secondary[i]), float(grid[i])) for i in indices)[1])


def choose(y, scores):
    grid, f1, bacc = threshold_grid(y, scores)
    return dict(quantile_f1=best_f1_threshold(y, scores),
                exact_f1=exact_choice(grid, f1, bacc),
                exact_bacc=exact_choice(grid, bacc, f1))


def metrics(y, score, threshold=None, pred=None):
    result = metric_row(y, score, threshold=threshold, pred=pred)
    if threshold is None:
        result.pop('threshold')
    return result


def mean_sd(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}


def summarize(reps):
    return {rule: {key: mean_sd([row[key] for row in rows])
                   for key in ['auc', 'f1', 'bacc', 'precision', 'recall', 'specificity', 'ppr']}
            for rule, rows in reps.items()}


def assert_row(row, expected):
    assert all(abs(row[key] - value) < 5.1e-5 for key, value in expected.items()), row


def compare(ref_y, ref_score, other_y, other_score):
    choices = choose(ref_y, ref_score)
    return {name: {'threshold': threshold, 'calibration': metrics(ref_y, ref_score, threshold),
                   'evaluation': metrics(other_y, other_score, threshold)}
            for name, threshold in choices.items()}


def examine_dev(y, scores):
    splitter = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=42)
    reps, thresholds = defaultdict(list), defaultdict(list)
    heldout_scores = {rule: np.zeros(len(y), int) for rule in RULES}
    for i, (fit, validation) in enumerate(splitter.split(scores, y)):
        for rule, threshold in choose(y[fit], scores[fit]).items():
            thresholds[rule].append(threshold)
            heldout_scores[rule][validation] = scores[validation] >= threshold
        if i % 4 == 3:
            for rule in RULES:
                reps[rule].append(metrics(y, scores, pred=heldout_scores[rule]))
    summary = summarize(reps)
    for rule in RULES:
        summary[rule]['threshold_q10_q50_q90'] = np.quantile(thresholds[rule], [.1, .5, .9]).tolist()
    return summary


def edaic(source):
    manifest = inputs()
    ids = np.asarray([row['subject_id'] for row in manifest])
    with MANIFEST.open() as handle:
        labels = {row['subject_id']: int(row['label']) for row in csv.DictReader(handle)}
    y = np.array([labels[i] for i in ids])
    split = np.array([row['split'].lower() for row in manifest])
    tr = np.flatnonzero(split == 'train')
    dev = np.flatnonzero(np.isin(split, ['dev', 'development', 'val']))
    te = np.flatnonzero(split == 'test')
    assert (len(tr), len(dev), len(te)) == (163, 56, 56)
    assert len(set(np.r_[tr, dev, te])) == len(ids)
    evaluation = np.r_[dev, te]
    text = source / 'round2_features'
    paths = [text/'embedding_clean_sentence3.npz', text/'observable_clean_sentence3.npz',
             TIMESTAMP_OUT/'features/participant3_vocal.npz']
    zs = []
    for index, path in enumerate(paths):
        x = strict_profile(path, ids)
        zs.append(fit_score(x[tr], y[tr], x[evaluation],
                            BranchSpec('mean', 0, .03 if index == 0 else .1), 42+100*index)[0])
    s = np.asarray(zs).mean(axis=0)
    ds, ts = np.split(s, [len(dev)])

    development_cv = examine_dev(y[dev], ds)
    out = compare(y[dev], ds, y[te], ts)
    assert_row(out['quantile_f1']['evaluation'], dict(auc=.8416, f1=.5667, bacc=.6667, threshold=-.3247425253))
    grid, f1, bacc = threshold_grid(y[dev], ds)
    tied = np.isclose(f1, f1.max(), rtol=0, atol=1e-12)
    selected_count = [int((ds >= t).sum()) for t in grid[tied]]
    return dict(n_calibration=len(dev), positives_calibration=int(y[dev].sum()),
                n_test=len(te), positives_test=int(y[te].sum()),
                best_f1_midpoints=grid[tied].tolist(), best_f1_positive_counts=selected_count,
                best_f1_bacc=bacc[tied].tolist(), development_cv=development_cv, rules=out)


def source_cv(ids, y, xs, folds):
    results = defaultdict(list)
    repeats = sorted({f['repeat'] for f in folds})
    assert len(repeats) == 5
    for repeat in repeats:
        preds = {rule: np.full(len(y), -1, int) for rule in RULES}
        scores = np.full(len(y), np.nan)
        for k, fold in enumerate(folds):
            if fold['repeat'] != repeat:
                continue
            tr, te = [np.asarray(fold[key], int) for key in ['train_idx', 'val_idx']]
            assert not set(tr).intersection(te)
            assert np.isnan(scores[te]).all()
            pairs = [score_branch(x, ids, y, tr, te, .03 if i == 0 else .1,
                                  i, k, 'eatd')[:2] for i, x in enumerate(xs)]
            cal, test = [np.stack(part).mean(axis=0) for part in zip(*pairs)]
            scores[te] = test
            for rule, threshold in choose(y[tr], cal).items():
                preds[rule][te] = test >= threshold
        assert np.isfinite(scores).all()
        for rule in RULES:
            assert (preds[rule] >= 0).all()
            results[rule].append(metrics(y, scores, pred=preds[rule]))
    out = summarize(results)
    assert_row({key: row['mean'] for key, row in out['quantile_f1'].items()},
               dict(auc=.7537, f1=.4430, bacc=.6809))
    return out


def eatd2(transfer, ids, y):
    with np.load(transfer/'source_calibration.npz') as z:
        assert np.array_equal(z['subject_ids'].astype(str), ids.astype(str))
        calibration = z['scores']
    with (transfer/'subject_predictions.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['dataset']=='EATD_to_EATD2'
                and r['model']=='FMEM' and r['calibration']=='pure_source'
                and r['operating_point']=='source_oof_f1']
    assert len(rows) == len({r['subject_id'] for r in rows}) == 91
    outcomes = np.array([int(r['label']) for r in rows])
    scores = np.array([float(r['score']) for r in rows])
    out = compare(y, calibration, outcomes, scores)
    assert_row(out['quantile_f1']['evaluation'], dict(auc=.6500, f1=.4478, bacc=.5653))
    assert abs(out['quantile_f1']['threshold'] - float(rows[0]['threshold'])) < 1e-10
    return dict(n_calibration=len(y), positives_calibration=int(y.sum()),
                n_test=len(outcomes), positives_test=int(outcomes.sum()), rules=out)


def main():
    parser = argparse.ArgumentParser()
    for name in ['source_audio', 'source_text', 'source_llm', 'source_vocal',
                 'source_root', 'source_vocal12', 'fold_json', 'source_results', 'eatd2_results', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = Path(args.source_results)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    ids, y, xs = source_data(args)
    xs[2] = strict_profile(args.source_vocal12, ids)
    folds = load_fold_manifest(args.fold_json, ids)['folds']
    results = {}
    for name, compute in [('E-DAIC', lambda: edaic(source)),
                          ('EATD-Corpus', lambda: source_cv(ids, y, xs, folds)),
                          ('EATD-Corpus 2.0', lambda: eatd2(Path(args.eatd2_results), ids, y))]:
        results[name] = compute()
        (dest/'audit.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        print(name + ' complete', flush=True)
    (dest/'COMPLETE').write_text('Threshold analysis complete\n')


if __name__ == '__main__':
    main()
