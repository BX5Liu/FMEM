import copy
import json
from pathlib import Path

import numpy as np
import torch
from dep_author.DepMamba import DepMamba
from ablation import safe_auc
import train_published as original


def make_model(inputs):
    return DepMamba(audio_input_size=inputs['audio'].shape[-1],
                    video_input_size=inputs['text'].shape[-1], mm_input_size=256,
                    mm_output_sizes=[256], dropout=.1, d_ffn=1024, num_layers=1,
                    activation='GELU', causal=False,
                    mamba_config=dict(d_state=16, expand=4, d_conv=4, bidirectional=True))


def fit_model(options, inputs, labels, train, dev, seed, device, epochs, out):
    original.seed_all(seed)
    model = make_model(inputs).to(device)
    train_x = original.preprocess(options, inputs, train, train)
    dev_x = original.preprocess(options, inputs, train, dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_auc = -np.inf
    best_state = None
    selected_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.
        for start in range(0, len(order), 16):
            ix = order[start:start + 16]
            x = torch.cat([torch.as_tensor(train_x[k][ix], device=device) for k in ['text', 'audio']], dim=-1)
            target = torch.as_tensor(labels[train[ix]], device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            score = model(x).reshape(-1)
            loss = loss_fn(score, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        dev_score = original.predict(model, options, dev_x, device)
        auc = float(safe_auc(labels[dev], dev_score))
        history.append(dict(epoch=epoch, train_loss=total / len(train), dev_auc=auc))
        if auc > best_auc + 1e-12:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
        if epoch % 10 == 0 or epoch == epochs:
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    reference = original.predict(model, options, train_x, device)
    score = original.predict(model, options, dev_x, device)
    score = (score - reference.mean()) / max(float(reference.std()), 1e-8)
    (Path(out) / 'training_history.json').write_text(json.dumps(history, indent=2))
    return score, selected_epoch, model
