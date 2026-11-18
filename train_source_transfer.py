import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

import Y_deep_architecture_baselines as compact
import train_published as author
from ablation import load_dataset, strict_profile, make_inner_splits, best_f1_threshold, metric_row
from fixed_final_transfer import text_features_without_labels


MODELS = ['eatd_gru_bilstm_style', 'lightweight_mult', 'misa_style', 'sega', 'caf']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_inputs(text, audio, model):
    inputs = {}
    for key, value in [('text', text), ('audio', audio)]:
        inputs[key] = (value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)).astype(np.float32)
    if model == 'sega':
        inputs['text'] = inputs['text'][:, :, None, :]
        inputs['audio'] = inputs['audio'][:, :, None, :]
        inputs.update(question=np.zeros_like(inputs['text']), gender=np.zeros(len(text), np.float32),
                      emotion=np.zeros((len(text), 3, 7), np.float32))
    return inputs


def compact_fit(name, text, audio, labels, train, valid, seed, device, epochs, select):
    reducers = [compact.TopicReducer(96, seed + offset).fit(x[train])
                for offset, x in enumerate([text, audio])]
    tt, at = [r.transform(x[train]) for r, x in zip(reducers, [text, audio])]
    tv, av = [r.transform(x[valid]) for r, x in zip(reducers, [text, audio])]
    if select:
        model, chosen, _ = compact.train_with_validation(name, tt, at, labels[train], tv, av, labels[valid],
            64, .3, 1e-3, 1e-4, 32, epochs, 12, seed, device)
    else:
        chosen = epochs
        model = compact.train_fixed_epochs(name, tt, at, labels[train], 64, .3, 1e-3, 1e-4,
                                            32, epochs, seed, device)
    reference = compact.predict(model, tt, at, device, 32)
    score = compact.zscore(reference, compact.predict(model, tv, av, device, 32))
    norm = (float(reference.mean()), max(float(reference.std()), 1e-8))
    return score, int(chosen), model, reducers, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    for key in ['source-audio', 'source-text', 'source-root', 'matched-source-audio', 'matched-target-audio',
                'target-audio', 'target-text', 'target-root', 'outdir']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '6')))
    if not torch.cuda.is_available():
        raise RuntimeError('Submit through GPU sbatch')
    device = torch.device('cuda')
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    definition = dict(arguments=vars(args), driver_sha256=digest(__file__), seeds=[42, 1337, 2025],
        protocol='EATD162_source_threefold_epoch_and_F1_selection_then_refit_external91',
        target_fit_n=0, feature_sha256={key:digest(getattr(args, key)) for key in
        ['source_text', 'source_audio', 'matched_source_audio', 'matched_target_audio', 'target_text']})
    if (root/'definition.json').exists():
        previous=json.loads((root/'definition.json').read_text())
        if (root/'COMPLETE').exists():
            current_check={k:v for k,v in definition.items() if k!='driver_sha256'}
            previous_check={k:v for k,v in previous.items() if k!='driver_sha256'}
            if current_check!=previous_check:
                raise ValueError('Completed run has another definition')
            print('Completed run retained with its original source provenance',flush=True)
            return
        if previous!=definition:
            raise ValueError('Definition changed')
    (root/'definition.json').write_text(json.dumps(definition, indent=2))
    source = load_dataset(args.source_audio, args.source_text, args.source_root, threshold=53, strict_expected=True)
    ids = np.asarray(source['ids']).astype(str)
    labels = np.asarray(source['y'], int)
    if (len(ids), int(labels.sum())) != (162, 30):
        raise ValueError('Source cohort changed')
    text = source['text'].astype(np.float32)
    audio = strict_profile(args.matched_source_audio, ids).astype(np.float32)
    options = SimpleNamespace(model=args.model, features='qwen', dataset='eatd')
    inputs = author_inputs(text, audio, args.model) if args.model in ['sega', 'caf'] else None
    epochs = 100 if args.model != 'caf' else 80
    seeds = [42, 1337, 2025]
    for seed in seeds:
        out = root/str(seed)
        out.mkdir(exist_ok=True)
        if (out/'scores_before_evaluation.npz').exists():
            continue
        oof = np.full(len(ids), np.nan)
        choices = []
        for fold, (train, valid) in enumerate(make_inner_splits(ids, labels, 3, seed)):
            fold_seed = seed + 97*fold
            if inputs is not None:
                score, chosen, model = author.fit(options, inputs, labels, train, valid, fold_seed,
                                                   device, epochs, True)
            else:
                score, chosen, model, _, _ = compact_fit(args.model, text, audio, labels, train, valid,
                                                         fold_seed, device, epochs, True)
            oof[valid] = score
            choices.append(int(chosen))
            del model
            torch.cuda.empty_cache()
            print(json.dumps(dict(model=args.model, seed=seed, source_fold=fold, selected_epoch=chosen)), flush=True)
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete source OOF coverage')
        threshold = float(best_f1_threshold(labels, oof))
        selected = max(1, int(np.median(choices)))
        train = np.arange(len(ids))
        if inputs is not None:
            _, _, model = author.fit(options, inputs, labels, train, train, seed+909, device, selected, False)
            reference = author.predict(model, options, author.preprocess(options, inputs, train, train), device)
            normalizer = (float(reference.mean()), max(float(reference.std()), 1e-8))
            reducers = None
        else:
            _, _, model, reducers, normalizer = compact_fit(args.model, text, audio, labels, train, train,
                                                            seed+909, device, selected, False)
        selection = dict(seed=seed, selected_epoch=selected, inner_epochs=choices, threshold=threshold,
                         normalizer=normalizer, train_ids=ids.tolist(), target_statistics_fit_n=0,
                         target_labels_used_for_selection=False, feature_sha256=definition['feature_sha256'])
        torch.save(dict(state_dict=model.state_dict(), selection=selection, model=args.model), out/'checkpoint_before_target.pt')
        joblib.dump(reducers if reducers is not None else inputs, out/'source_preprocessing.joblib')
        (out/'selection_before_target.json').write_text(json.dumps(selection, indent=2))
        np.savez_compressed(out/'source_oof.npz', subject_ids=ids, labels=labels, scores=oof)
        target_ids = np.load(args.matched_target_audio, allow_pickle=False)['subject_ids'].astype(str)
        target_text = text_features_without_labels(args.target_text, target_ids).astype(np.float32)
        target_audio = strict_profile(args.matched_target_audio, target_ids).astype(np.float32)
        if len(target_ids) != 91 or len(set(target_ids)) != 91 or set(ids) & set(target_ids):
            raise ValueError('Source/target identity check failed')
        if inputs is not None:
            tx = author_inputs(target_text, target_audio, args.model)
            combined = {k:np.concatenate([inputs[k], tx[k]]) for k in inputs}
            raw = author.predict(model, options, author.preprocess(options, combined, train,
                np.arange(len(ids), len(ids)+len(target_ids))), device)
        else:
            tt, at = [r.transform(x) for r, x in zip(reducers, [target_text, target_audio])]
            raw = compact.predict(model, tt, at, device, 32)
        score = (raw-normalizer[0])/normalizer[1]
        np.savez_compressed(out/'scores_before_evaluation.npz', subject_ids=target_ids, scores=score,
                            predictions=(score>=threshold).astype(int), threshold=threshold)
        del model
        torch.cuda.empty_cache()
    target = load_dataset(args.target_audio, args.target_text, args.target_root, threshold=53,
                          expected_n=91, expected_neg=53, expected_pos=38, strict_expected=True)
    target_ids = np.asarray(target['ids']).astype(str)
    truth = np.asarray(target['y'], int)
    records = []
    for seed in seeds:
        saved = np.load(root/str(seed)/'scores_before_evaluation.npz', allow_pickle=False)
        positions = {s:i for i,s in enumerate(saved['subject_ids'].astype(str))}
        order = np.asarray([positions[s] for s in target_ids])
        score, pred = saved['scores'][order], saved['predictions'][order]
        record=metric_row(truth, score, pred=pred)
        record.update(model=args.model, seed=seed, threshold=float(saved['threshold']))
        records.append(record)
        pd.DataFrame(dict(subject_id=target_ids, label=truth, score=score, pred=pred,
                          seed=seed)).to_csv(root/str(seed)/'subject_predictions.csv', index=False)
    metrics = pd.DataFrame(records)
    metrics.to_csv(root/'per_seed_metrics.csv', index=False)
    summary = dict(model=args.model, seeds=len(seeds), n_train=162, n_test=91)
    for key in ['auc','f1','bacc','recall','specificity']:
        summary[key+'_mean'] = float(metrics[key].mean())
        summary[key+'_std'] = float(metrics[key].std(ddof=1))
    pd.DataFrame([summary]).to_csv(root/'summary.csv', index=False)
    (root/'COMPLETE').write_text('ok\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
