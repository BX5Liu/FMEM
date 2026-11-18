import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import train_published as original
from ablation import strict_profile, best_f1_threshold, metric_row


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sega', 'caf', 'depmamba'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--text-npz', required=True)
    parser.add_argument('--audio-npz', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--max-epochs', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Submit training through GPU sbatch')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '6')))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    definition = dict(model=args.model, seed=args.seed, protocol='official_163_56_56',
                      selection='development_auc_checkpoint_then_development_f1_threshold',
                      input_sha256={name: sha256(path) for name, path in
                          [('manifest', args.manifest), ('text', args.text_npz), ('audio', args.audio_npz)]},
                      driver_sha256=sha256(__file__), feature_setting='shared_frozen_Qwen_text_audio_three_contexts')
    definition_path = out / 'definition.json'
    if definition_path.exists() and json.loads(definition_path.read_text()) != definition:
        raise ValueError('Run definition changed; choose another output directory')
    definition_path.write_text(json.dumps(definition, indent=2))
    if (out / 'COMPLETE').exists():
        print('Existing completed run matches its definition', flush=True)
        return
    manifest = pd.read_csv(args.manifest, dtype={'subject_id': str})
    ids = manifest.subject_id.to_numpy(str)
    if len(ids) != 275 or len(set(ids)) != 275:
        raise ValueError('Expected 275 unique participants')
    splits = manifest.official_split.to_numpy(str)
    train = np.flatnonzero(splits == 'train')
    dev = np.flatnonzero(splits == 'dev')
    test = np.flatnonzero(splits == 'test')
    if tuple(map(len, [train, dev, test])) != (163, 56, 56):
        raise ValueError('Official partition changed')
    if any(set(ids[a]) & set(ids[b]) for a, b in [(train, dev), (train, test), (dev, test)]):
        raise ValueError('Participant overlap')
    text = strict_profile(args.text_npz, ids).astype(np.float32)
    audio = strict_profile(args.audio_npz, ids).astype(np.float32)
    if text.shape[1] != 3 or audio.shape[1] != 3:
        raise ValueError('Expected three existing contexts')
    inputs = dict(text=text, audio=audio)
    for key, value in inputs.items():
        inputs[key] = value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)
    if args.model == 'sega':
        inputs['text'] = inputs['text'][:, :, None, :]
        inputs['audio'] = inputs['audio'][:, :, None, :]
        inputs.update(question=np.zeros_like(inputs['text']), gender=np.zeros(len(ids), np.float32),
                      emotion=np.zeros((len(ids), 3, 7), np.float32))
    options = SimpleNamespace(model=args.model, features='qwen', dataset='edaic_official')
    epochs = args.max_epochs or (100 if args.model == 'sega' else 120 if args.model == 'depmamba' else 80)
    labels = np.zeros(len(ids), dtype=int)
    labels[train] = manifest.label.to_numpy(int)[train]
    labels[dev] = manifest.label.to_numpy(int)[dev]
    started = time.time()
    if args.model == 'depmamba':
        from depmamba_adapter import fit_model
        dev_scores, selected_epoch, model = fit_model(options, inputs, labels, train, dev,
                                                     args.seed, torch.device('cuda'), epochs, out)
    else:
        dev_scores, selected_epoch, model = original.fit(options, inputs, labels, train, dev,
                                                        args.seed, torch.device('cuda'), epochs, True)
    threshold = float(best_f1_threshold(labels[dev], dev_scores))
    torch.save(dict(state_dict=model.state_dict(), definition=definition, selected_epoch=selected_epoch,
                    threshold=threshold), out / 'checkpoint_before_test.pt')
    (out / 'selection_before_test.json').write_text(json.dumps(dict(selected_epoch=selected_epoch,
        threshold=threshold, train_n=len(train), dev_n=len(dev), test_fitting_n=0), indent=2))
    train_inputs = original.preprocess(options, inputs, train, train)
    test_inputs = original.preprocess(options, inputs, train, test)
    train_scores = original.predict(model, options, train_inputs, torch.device('cuda'))
    scores = original.predict(model, options, test_inputs, torch.device('cuda'))
    scores = (scores - train_scores.mean()) / max(float(train_scores.std()), 1e-8)
    truth = manifest.label.to_numpy(int)[test]
    predictions = (scores >= threshold).astype(int)
    display = {'sega': 'SEGA-shared-Qwen-TA', 'caf': 'CAF-Mamba-shared-Qwen-TA',
               'depmamba': 'DepMamba-shared-Qwen-TA'}[args.model]
    pd.DataFrame(dict(subject_id=ids[test], label=truth, score=scores, pred=predictions,
                      model=display, seed=args.seed, repeat=args.seed, fold=0,
                      operating_point='development_f1', threshold=threshold)).to_csv(out / 'subject_predictions.csv', index=False)
    result = dict(model=display, seed=args.seed, operating_point='development_f1',
                  selected_epoch=selected_epoch, **metric_row(truth, scores, pred=predictions))
    result['threshold'] = threshold
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)
    (out / 'run_metadata.json').write_text(json.dumps(dict(definition=definition, elapsed_seconds=time.time() - started,
        original_model_module=str(Path(original.__file__).resolve()), input_shapes={k: list(v.shape) for k, v in inputs.items()},
        training_n=163, development_n=56, test_n=56, test_statistics_fit_n=0, test_labels_used_for_selection=False,
        reproduction_scope='Author fusion architecture transferred to shared frozen audio/text inputs; native input modalities and auxiliaries are not claimed'), indent=2))
    (out / 'COMPLETE').write_text('ok\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
