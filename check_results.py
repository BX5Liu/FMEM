"""Check archived or newly generated FMEM results against the paper."""
import argparse
import csv
from pathlib import Path


def check(path, name, keys, expected, operating_point=None):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row['model'] == name and (operating_point is None or row['operating_point'] == operating_point)
        ]
    if len(rows) != 1 or any(abs(float(rows[0][key]) - val) > .000051
                             for key, val in zip(keys, expected)):
        raise ValueError(f'{path}: expected {expected}, found {rows}')
    print(f'{name}: ' + ', '.join(f'{key}={float(rows[0][key]):.4f}' for key in keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='?', type=Path, help='experiment output directory')
    parser.add_argument('--bundled', action='store_true', help='check the archived results shipped with the code')
    args = parser.parse_args()
    if args.bundled:
        if args.results is not None:
            parser.error('pass either an output directory or --bundled')
        root = Path(__file__).resolve().parent / 'results'
        eatd = root / 'eatd_core.csv'
        eatd2 = root / 'eatd2_fmem.csv'
        edaic = root / 'edaic_paper_summary.csv'
    else:
        if args.results is None:
            parser.error('an output directory or --bundled is required')
        root = args.results
        eatd = root / 'eatd/core/summary.csv'
        eatd2 = root / 'eatd2/fmem/summary.csv'
        edaic = root / 'edaic/summary.csv'
    metrics = ['auc', 'f1', 'bacc']
    check(eatd, 'SLEF-D-12x12-unbounded', [k+'_mean' for k in metrics],
          [.7537, .4430, .6809], 'train_oof_f1')
    check(eatd2, 'FMEM', metrics, [.6500, .4478, .5653])
    for name, row in [
        ('Sentence thirds / three audio groups', [.8416, .5667, .6667]),
        ('Timestamp thirds / three audio groups', [.8552, .7059, .7888]),
        ('Feature concatenation + LR', [.8643, .6207, .7262]),
        ('Unstandardized logit averaging', [.8311, .7027, .7926]),
        ('Development AUC weighting', [.8718, .7143, .8130]),
    ]:
        check(edaic, name, metrics, row)


if __name__ == '__main__':
    main()
