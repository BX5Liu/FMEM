from __future__ import annotations
import argparse,hashlib,json,joblib
from pathlib import Path
import numpy as np
import pandas as pd
from ablation import (load_dataset,strict_profile,BranchModel,BranchSpec,make_inner_splits,
    fit_score,zapply,best_f1_threshold,prior_threshold,metric_row)

def combine(z):return .8*z[:,0]+.2*z[:,1]+.1*np.tanh(z[:,2])

def source_data(a):
    d=load_dataset(a.source_audio,a.source_text,a.source_root,threshold=53,strict_expected=True)
    ids=np.asarray(d['ids']);y=np.asarray(d['y'],int)
    if (len(ids),int(y.sum()))!=(162,30):raise ValueError('EATD source cohort changed')
    return ids,y,[d['text'],strict_profile(a.source_llm,ids),strict_profile(a.source_vocal,ids)]

def fit_source(ids,y,xs):
    oof=np.zeros((len(y),3));models=[];normalizers=[]
    for b,x in enumerate(xs):
        seed=42+41*b;spec=BranchSpec('mean',0,.03 if b==0 else .1)
        for j,(tr,va) in enumerate(make_inner_splits(ids,y,3,seed)):
            oof[va,b]=fit_score(x[tr],y[tr],x[va],spec,seed+97*j)[0]
        model=BranchModel(spec,seed).fit(x,y);ref=model.decision(x)
        models.append(model);normalizers.append(dict(mean=float(ref.mean()),std=max(float(ref.std()),1e-8)))
    cal=combine(oof)
    if not np.isfinite(cal).all():raise ValueError('Incomplete source calibration')
    thresholds=dict(source_oof_f1=float(best_f1_threshold(y,cal)),source_prevalence=float(prior_threshold(cal,float(y.mean()))))
    return models,normalizers,thresholds,cal

def score_target(models,norm,xs):
    z=np.column_stack([(m.decision(x)-n['mean'])/n['std'] for m,n,x in zip(models,norm,xs)])
    if not np.isfinite(z).all():raise ValueError('Nonfinite target score')
    return combine(z),z

def targets(a):


    from edaic_ceef7_common import load_manifest
    m=load_manifest(a.edaic_manifest);ei=m.subject_id.to_numpy(str)
    ex=[strict_profile(p,ei) for p in [a.edaic_text,a.edaic_llm,a.edaic_vocal]]
    splits=m.official_split.to_numpy(str) if 'official_split' in m else np.where(m.split=='test','test','train_dev')
    edaic=dict(name='EATD_to_EDAIC',ids=ei,X=ex,y=m.label.to_numpy(int),splits=splits)
    d=load_dataset(a.target_audio,a.target_text,a.target_root,threshold=53,expected_n=91,expected_neg=53,expected_pos=38,strict_expected=True)
    ti=np.asarray(d['ids']);tx=[d['text'],strict_profile(a.target_llm,ti),strict_profile(a.target_vocal,ti)]
    eatd2=dict(name='EATD_to_EATD2',ids=ti,X=tx,y=np.asarray(d['y'],int),splits=np.full(len(ti),'test'))
    return [edaic,eatd2]

def evaluation(targets,all_scores,thresholds,out,model_name='SLEF-D-fixed-source162'):
    rows=[];metrics=[]
    for target in targets:
        score=all_scores[target['name']];y=target['y'];ids=target['ids'];split=target['splits']
        if len(set(map(str,ids)))!=len(ids) or len(score)!=len(ids):raise ValueError('Target ID/score alignment')
        scope={'test':np.arange(len(ids))} if target['name']=='EATD_to_EATD2' else {'all_275':np.arange(len(ids)),'official_test_56':np.flatnonzero(split=='test')}
        if target['name']=='EATD_to_EDAIC' and len(scope['official_test_56'])!=56:raise ValueError('Official test cohort changed')
        for op,th in thresholds.items():
            pred=(score>=th).astype(int)
            rows.extend(dict(dataset=target['name'],subject_id=str(s),label=int(y[i]),model=model_name,operating_point=op,score=float(score[i]),pred=int(pred[i]),threshold=float(th),original_split=str(split[i])) for i,s in enumerate(ids))
            for name,ix in scope.items():
                met=metric_row(y[ix],score[ix],pred=pred[ix])

                from sklearn.metrics import roc_auc_score
                rng=np.random.default_rng(42);pos=np.flatnonzero(y[ix]==1);neg=np.flatnonzero(y[ix]==0);draw=[]
                for _ in range(5000):
                    sample=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
                    draw.append(roc_auc_score(y[ix][sample],score[ix][sample]))
                ci=np.percentile(draw,[2.5,97.5])
                met['threshold']=float(th)
                metrics.append(dict(dataset=target['name'],scope=name,model=model_name,operating_point=op,n_test=len(ix),positive_n=int(y[ix].sum()),auc_ci_lo=float(ci[0]),auc_ci_hi=float(ci[1]),**met))
    pd.DataFrame(rows).to_csv(out/'subject_predictions.csv',index=False)
    d=pd.DataFrame(metrics);d.to_csv(out/'summary.csv',index=False);print(d.to_string(index=False),flush=True)
    (out/'COMPLETE').write_text('ok\n')
    return d

def parser():
    ap=argparse.ArgumentParser()
    for name in ['source_audio','source_text','source_llm','source_vocal','source_root',
                 'edaic_manifest','edaic_text','edaic_llm','edaic_vocal',
                 'target_audio','target_text','target_llm','target_vocal','target_root','outdir']:
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--evaluation_only',action='store_true')
    return ap

def main():
    a=parser().parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if a.evaluation_only:
        spec=json.loads((out/'fixed_method.json').read_text())
        ts=targets(a);scores={}
        for target in ts:
            z=np.load(out/(target['name']+'_scores.npz'),allow_pickle=False)
            np.testing.assert_array_equal(z['subject_ids'],target['ids']);scores[target['name']]=z['scores']
        evaluation(ts,scores,spec['threshold_policy'],out);return
    ids,y,xs=source_data(a)
    models,norm,thresholds,cal=fit_source(ids,y,xs)
    spec=dict(method='SLEF-D',source='EATD 162 subjects, 30 positives, SDS >=53',
        encoder='Qwen3-Embedding-8B',observable=12,vocal=8,C_semantic=.03,C_evidence=.1,
        fusion='0.8 z_semantic + 0.2 z_observable + 0.1 tanh(z_vocal)',pooling='L2 each context -> mean -> L2 -> source-fitted StandardScaler',
        target_EDAIC='Uniform raw ASR Text-column clean_sentence3, three audio thirds with two windows each',
        score_normalization='Source training decision mean/std only, reused unchanged on target',
        threshold_policy=thresholds,primary_operating_point='source_oof_f1',target_training_subjects=0,
        endpoints='EATD and EATD2 SDS>=53; E-DAIC released PHQ_Binary; cross-corpus instrument shift is retained',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arguments=vars(a))
    (out/'fixed_method.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding='utf8')
    joblib.dump(dict(models=models,normalizers=norm,thresholds=thresholds,method=spec),out/'source162_models.joblib')
    pd.DataFrame(dict(subject_id=ids,label=y,source_oof_score=cal)).to_csv(out/'source_oof_predictions.csv',index=False)
    ts=targets(a);scores={}
    for target in ts:
        score,z=score_target(models,norm,target['X']);scores[target['name']]=score

        np.savez_compressed(out/(target['name']+'_scores.npz'),subject_ids=target['ids'],scores=score,branch_scores=z)
    evaluation(ts,scores,thresholds,out)

if __name__=='__main__':main()
