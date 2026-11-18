import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from common import BASE, MANIFEST, OUT, ROOT, inputs, save_json


def main():
    if not (Path(os.environ.get('FMEM_EDAIC_CACHE', OUT))/'TIMESTAMP_QC_PASSED').exists():
        raise RuntimeError('Released-transcript timestamp checks incomplete')
    sys.path[:0]=[str(BASE),os.environ['FMEM_REFERENCE_DIR']]
    from version_validation import EvidenceEngine, choose_and_predict, config
    from ablation import strict_profile,fit_score,BranchSpec,best_f1_threshold,metric_row
    rows=inputs()
    ids=np.asarray([r['subject_id'] for r in rows])
    with MANIFEST.open() as handle:
        labels={r['subject_id']:int(r['label']) for r in csv.DictReader(handle)}
    y=np.asarray([labels[s] for s in ids])
    split=np.asarray([r['split'].lower() for r in rows])
    train=np.flatnonzero(split=='train');dev=np.flatnonzero(np.isin(split,['dev','development','val']))
    test=np.flatnonzero(split=='test')
    if tuple(map(len,[train,dev,test]))!=(163,56,56):
        raise ValueError('Training/development/test membership changed')
    if set(train)&set(dev) or set(train)&set(test) or set(dev)&set(test):
        raise ValueError('Participant leakage')
    base_results=Path(os.environ['FMEM_FEATURE_ROOT'])
    f=base_results/'round2_features'
    banks={
        'semantic3':strict_profile(f/'embedding_clean_sentence3.npz',ids),
        'observable3':strict_profile(f/'observable_clean_sentence3.npz',ids),
        'semantic1':strict_profile(f/'embedding_whole2048.npz',ids),
        'observable1':strict_profile(f/'observable_whole2048.npz',ids),
        'semantic5':strict_profile(f/'embedding_sentence5.npz',ids),
        'observable5':strict_profile(f/'observable_sentence5.npz',ids),
        'semantic_timeline3':strict_profile(f/'embedding_timeline3.npz',ids),
        'observable_timeline3':strict_profile(f/'observable_timeline3.npz',ids)}
    cache_dir=Path(os.environ.get('FMEM_EDAIC_CACHE', OUT))
    records=[json.loads((cache_dir/'feature_records'/(subject+'.json')).read_text()) for subject in ids]
    feature_dir=OUT/'features';feature_dir.mkdir(parents=True, exist_ok=True)
    coverage=[]
    for policy in ['participant3','participant1','participant5']:
        for modality in ['vocal','matched']:
            key=policy+'_'+modality
            banks[key]=np.asarray([r['features'][policy][modality] for r in records],dtype=np.float32)
            valid=np.asarray([r['features'][policy]['valid_groups'] for r in records],dtype=bool)
            if np.any(banks[key][~valid]!=0) or not valid.any(axis=1).all():
                raise ValueError('Invalid missing-group representation')
            np.savez_compressed(feature_dir/(key+'.npz'),subject_ids=ids,features=banks[key],profiles=banks[key],valid_groups=valid)
        for row,record in zip(rows,records):
            seconds=record['features'][policy]['clip_seconds']
            coverage.append(dict(subject_id=row['subject_id'],split=row['split'],policy=policy,
                                 clip_count=sum(map(len,seconds)),audio_seconds=sum(map(sum,seconds))))
    pd.DataFrame(coverage).to_csv(OUT/'clip_coverage.csv',index=False)
    cache={}
    def scores(key):
        if key not in cache:
            index=0 if key.startswith('semantic') else (1 if key.startswith('observable') else 2)
            C=.03 if index==0 else .1
            x=banks[key]
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite features')
            score,_=fit_score(x[train],y[train],x[np.r_[dev,test]],BranchSpec('mean',0,C),42+100*index)
            cache[key]=(score[:len(dev)],score[len(dev):])
        return cache[key]
    plans=[]
    def add(name,keys,scope):
        plans.append(dict(model=name,keys=keys,scope=scope))
    for name,modality in [('Matched audio embeddings','matched'),('Vocal-12','vocal')]:
        key='participant3_'+modality
        add('Participant audio / '+name,[key],'audio_feature_validity')
        add('Participant audio / FMEM with '+name,['semantic3','observable3',key],'audio_feature_validity')
    for omitted,keys in [('text semantics',['observable3','participant3_vocal']),
                         ('text evidence',['semantic3','participant3_vocal']),
                         ('audio evidence',['semantic3','observable3'])]:
        add('Participant audio / without '+omitted,keys,'component_ablation')
    for title,sem,obs,voice in [
        ('Whole text / one audio group','semantic1','observable1','participant1_vocal'),
        ('Sentence thirds / three audio groups','semantic3','observable3','participant3_vocal'),
        ('Timestamp thirds / three audio groups','semantic_timeline3','observable_timeline3','participant3_vocal'),
        ('Sentence fifths / five audio groups','semantic5','observable5','participant5_vocal')]:
        add(title,[sem,obs,voice],'input_grouping')
    results=[];predictions=[]
    for plan in plans:
        dev_score=np.mean([scores(key)[0] for key in plan['keys']],axis=0)
        test_score=np.mean([scores(key)[1] for key in plan['keys']],axis=0)
        threshold=float(best_f1_threshold(y[dev],dev_score))
        pred=(test_score>=threshold).astype(int)
        result=dict(model=plan['model'],scope=plan['scope'],**metric_row(y[test],test_score,pred=pred))
        result['threshold']=threshold
        results.append(result)
        predictions.extend(dict(subject_id=str(ids[i]),model=plan['model'],score=float(s),prediction=int(p),label=int(y[i]))
                           for i,s,p in zip(test,test_score,pred))
    zdev=np.column_stack([scores(key)[0] for key in ['semantic3','observable3','participant3_vocal']])
    ztest=np.column_stack([scores(key)[1] for key in ['semantic3','observable3','participant3_vocal']])
    from sklearn.metrics import roc_auc_score
    reliability=np.asarray([max(roc_auc_score(y[dev],zdev[:,j])-.5,.05) for j in range(3)])
    reliability/=reliability.sum()
    for name,ds,ts in [
        ('Sigmoid score averaging', (1/(1+np.exp(-np.clip(zdev,-700,700)))).mean(1),
                                    (1/(1+np.exp(-np.clip(ztest,-700,700)))).mean(1)),
        ('Development AUC weighting',zdev@reliability,ztest@reliability),
        ('Audio score clipping at three', (zdev[:,:2].sum(1)+np.clip(zdev[:,2],-3,3))/3,
                                          (ztest[:,:2].sum(1)+np.clip(ztest[:,2],-3,3))/3)]:
        threshold=float(best_f1_threshold(y[dev],ds))
        result=dict(model=name,scope='fusion_control',**metric_row(y[test],ts,pred=(ts>=threshold).astype(int)))
        result['threshold']=threshold
        results.append(result)
    def controls():
        xs = [banks[k] for k in ['semantic3', 'observable3', 'participant3_vocal']]
        engine = EvidenceEngine(ids, y, xs, 'edaic', names=['semantic3', 'observable3', 'audio3'])
        ds, ts, choice, _ = choose_and_predict(
            engine, config('Feature concatenation + LR', 'feature_lr'), train, test, 0, dev)
        raw = np.column_stack([
            fit_score(x[train], y[train], x[np.r_[dev,test]],
                      BranchSpec('mean', 0, .03 if index == 0 else .1),
                      42 + 100 * index, standardize=False)[0]
            for index, x in enumerate(xs)
        ]).mean(axis=1)
        return [('Feature concatenation + LR', ds, ts, choice),
                ('Unstandardized logit averaging', raw[:len(dev)], raw[len(dev):], {})]

    def evaluate(plans):
        metrics = []
        choices = []
        predictions = []
        for title, ds, ts, choice in plans:
            assert ds.shape == (56,) and ts.shape == (56,)
            assert np.isfinite(ds).all() and np.isfinite(ts).all()
            threshold = float(best_f1_threshold(y[dev], ds))
            pred = (ts >= threshold).astype(int)
            result = dict(model=title, scope='fusion_control', **metric_row(y[test], ts, pred=pred))
            result['threshold'] = threshold
            metrics.append(result)
            choices.append(dict(model=title, threshold=threshold, **choice))
            predictions.extend(dict(subject_id=str(ids[i]), model=title, score=float(score),
                                    prediction=int(p), label=int(y[i]))
                               for i, score, p in zip(test, ts, pred))
        return metrics, choices, predictions

    extra, choices, extra_predictions = evaluate(controls())
    results.extend(extra)
    predictions.extend(extra_predictions)
    pd.DataFrame(choices).to_csv(OUT/'fusion_choices.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'COMPLETE').write_text('E-DAIC speaker-selection, feature, component and grouping comparisons complete\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
