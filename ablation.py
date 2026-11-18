from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

OLD = Path(os.environ.get('FMEM_REFERENCE_DIR',
    str(Path(__file__).resolve().parent / 'reference')))
sys.path.insert(0, str(OLD))
from cef_common import load_dataset
from ceef2_core import BranchModel, BranchSpec
from ceef7_utils import (best_f1_threshold, prior_threshold, load_fold_manifest,
    make_inner_splits, metric_row, safe_auc, summarize_repeats, holm_adjust)

TEXT_NAMES = ['low_mood','anhedonia','hopelessness','self_blame','fatigue',
    'anxiety','social_withdrawal','cognitive_difficulty','negative_future',
    'limited_elaboration','affective_mismatch','limited_support']
VOCAL_NAMES = ['slow_speech','long_pauses','low_energy','flat_pitch',
    'sad_tone','tense_tone','low_engagement','unstable_voice']
GROUPS = {'llm_affect':[0,1,5], 'llm_cognition':[2,3,7,8],
    'llm_energy':[4], 'llm_interpersonal':[6,11], 'llm_response':[9,10],
    'vocal_timing':[0,1], 'vocal_prosody':[2,3],
    'vocal_affect_engagement':[4,5,6], 'vocal_quality':[7]}
ALPHAS = np.round(np.linspace(0,1,11),2)

def strict_profile(path, ids):
    with np.load(path, allow_pickle=True) as z:
        key = 'subject_ids' if 'subject_ids' in z else 'ids'
        raw = [str(v).strip() for v in z[key]]
        if len(set(raw)) != len(raw):
            raise ValueError('Duplicate feature IDs: '+str(path))
        pos = {s:i for i,s in enumerate(raw)}
        missing = set(map(str,ids))-set(pos)
        if missing:
            raise ValueError('Missing exact IDs: '+str(sorted(missing)[:5]))
        value_key='profiles' if 'profiles' in z else 'features'
        arr = np.asarray(z[value_key],float)[[pos[str(s)] for s in ids]]
    if arr.ndim==2:
        arr=arr[:,None,:]
    if not np.isfinite(arr).all():
        raise ValueError('Nonfinite evidence: '+str(path))
    return arr

def zapply(ref,x):
    return (x-ref.mean())/max(float(ref.std()),1e-8)

def splits(ids,y,seed,kind):
    if kind=='eatd':
        return list(make_inner_splits(ids,y,3,seed))
    return list(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))

def rank_dimensions(x,y,c):

    model=BranchModel(BranchSpec('mean',0,c),42).fit(x,y)
    w=np.abs(model.clf.coef_[0])
    return np.argsort(-w,kind='stable')

def fit_score(x,y,xt,spec,seed,selector=None,standardize=True):
    chosen=None
    if selector is not None:
        chosen=rank_dimensions(x,y,spec.c)[:selector]
        x,xt=x[...,chosen],xt[...,chosen]
    m=BranchModel(spec,seed).fit(x,y)
    ref=m.decision(x)
    score=m.decision(xt)
    return (zapply(ref,score) if standardize else score), chosen

def oof(x,ids,y,spec,seed,kind,selector=None,standardize=True):
    score=np.full(len(y),np.nan)
    for j,(tr,va) in enumerate(splits(ids,y,seed,kind)):
        score[va],_=fit_score(x[tr],y[tr],x[va],spec,seed+97*j,
            selector,standardize)
    if not np.isfinite(score).all():
        raise RuntimeError('Inner OOF coverage')
    return score

def variants(X, suite):
    v={}
    for b in ('text','llm','vocal'):
        v[b]=(b, X[b], BranchSpec('mean',0,.03 if b=='text' else .1),None,True)
    if suite in ('weights','core'):
        return v
    if suite=='expanded':
        for b,pool in [('llm','expanded_llm'),('vocal','expanded_vocal')]:
            if pool not in X:
                continue
            x=X[pool];v[b+'_extended_all']=(b,x,BranchSpec('mean',0,.1),None,True)
            budgets=(4,8,12,16) if b=='llm' else (2,4,8,12)
            for budget in budgets:
                if budget<x.shape[-1]:
                    v[f'{b}_expanded_top{budget}']=(b,x,BranchSpec('mean',0,.1),budget,True)
        if not any(k.startswith(('llm_extended','vocal_extended')) for k in v):
            raise ValueError('expanded suite requires an expanded feature pool')
        return v
    for b in ('llm','vocal'):
        names=TEXT_NAMES if b=='llm' else VOCAL_NAMES
        for j,name in enumerate(names):
            keep=[i for i in range(X[b].shape[-1]) if i!=j]
            v[b+'_drop_'+name]=(b,X[b][...,keep],BranchSpec('mean',0,.1),None,True)
        budgets=(3,6,9,12) if b=='llm' else (2,4,6,8)
        for k in budgets:
            if k==X[b].shape[-1]:
                continue


            v[f'{b}_top{k}']=(b,X[b],BranchSpec('mean',0,.1),k,True)
            for r in range(3):
                keep=np.sort(np.random.default_rng(1300+k*17+r).choice(X[b].shape[-1],k,replace=False))
                v[f'{b}_random{k}_{r}']=(b,X[b][...,keep],BranchSpec('mean',0,.1),None,True)
        for g,drop in GROUPS.items():
            if not g.startswith(b+'_'):
                continue
            keep=[i for i in range(X[b].shape[-1]) if i not in drop]
            v['without_'+g]=(b,X[b][...,keep],BranchSpec('mean',0,.1),None,True)
    for b in ('text','llm','vocal'):
        for j,context in enumerate(('positive','neutral','negative')):
            v[f'{b}_only_context{j}_{context}']=(b,X[b][:,j:j+1],
                BranchSpec('mean',0,.03 if b=='text' else .1),None,True)
        v[b+'_raw_mean']=(b,np.mean(X[b],axis=1,keepdims=True),
            BranchSpec('mean',0,.03 if b=='text' else .1),None,True)
        v[b+'_no_score_z']=(b,X[b],BranchSpec('mean',0,.03 if b=='text' else .1),None,False)
    return v

def compose(d,suite,alpha=None):
    t,l,v=d['text'],d['llm'],d['vocal']
    anchor=.8*t+.2*l
    full=anchor+.1*np.tanh(v)
    out={'SLEF-D':full,'without_vocal':anchor,
         'without_observable':t+.1*np.tanh(v),
         'without_semantic':l+.1*np.tanh(v),'text_only':t,
         'observable_only':l,'vocal_only':v,
         'remove_observable_fixed_weights':.8*t+.1*np.tanh(v),
         'remove_semantic_fixed_weights':.2*l+.1*np.tanh(v)}
    if suite=='core':
        return out
    if suite=='weights':
        for a in ALPHAS:
            out[f'alpha_text_{a:.2f}']=a*t+(1-a)*l+.1*np.tanh(v)
            out[f'anchor_alpha_text_{a:.2f}']=a*t+(1-a)*l
        if alpha is not None:
            out['nested_alpha']=alpha*t+(1-alpha)*l+.1*np.tanh(v)
        return out
    for name in d:
        if name in ('text','llm','vocal'):
            continue
        if name.startswith('llm_') or name.startswith('without_llm_'):
            out[name]=.8*t+.2*d[name]+.1*np.tanh(v)
        elif name.startswith('vocal_') or name.startswith('without_vocal_'):
            out[name]=anchor+.1*np.tanh(d[name])
        else:
            out[name]=.8*d[name]+.2*l+.1*np.tanh(v)
    if suite=='expanded':
        if 'llm_extended_all' in d and 'vocal_extended_all' in d:
            out['both_extended']=.8*t+.2*d['llm_extended_all']+.1*np.tanh(d['vocal_extended_all'])
        return out
    for j in range(3):
        out[f'all_only_context{j}']=.8*d[f'text_only_context{j}_{("positive","neutral","negative")[j]}']+.2*d[f'llm_only_context{j}_{("positive","neutral","negative")[j]}']+.1*np.tanh(d[f'vocal_only_context{j}_{("positive","neutral","negative")[j]}'])
    for lam in (.025,.05,.20):
        out[f'lambda_{lam:.3f}']=anchor+lam*np.tanh(v)
    out['unbounded_residual']=anchor+.1*v
    out['equal_three_scores']=(t+l+v)/3
    out['no_score_standardization']=.8*d['text_no_score_z']+.2*d['llm_no_score_z']+.1*np.tanh(d['vocal_no_score_z'])
    return out

def select_alpha(inner,y):
    scores=np.asarray([safe_auc(y,a*inner['text']+(1-a)*inner['llm']+.1*np.tanh(inner['vocal'])) for a in ALPHAS])

    best=np.flatnonzero(scores>=scores.max()-1e-12)
    j=min(best,key=lambda j:(abs(ALPHAS[j]-.8),ALPHAS[j]))
    return float(ALPHAS[j])

def check_folds(folds,n):
    for r in sorted(set(int(f['repeat']) for f in folds)):
        counts=np.zeros(n,int)
        for f in folds:
            if int(f['repeat'])!=r:
                continue
            tr=np.asarray(f['train_idx'],int);va=np.asarray(f['val_idx'],int)
            if len(set(tr)&set(va)) or len(set(tr)|set(va))!=n:
                raise ValueError('Invalid/disjoint outer split')
            counts[va]+=1
        if not np.all(counts==1):
            raise ValueError('Incomplete/duplicate outer OOF coverage')

def bootstrap_auc_delta(y,candidate,reference,nboot,seed):

    p=np.flatnonzero(y==1);n=np.flatnonzero(y==0)
    mc=(candidate[p,None]>candidate[n]).astype(float)+.5*(candidate[p,None]==candidate[n])
    mr=(reference[p,None]>reference[n]).astype(float)+.5*(reference[p,None]==reference[n])
    diff=mc-mr
    rng=np.random.default_rng(seed);vals=[]
    for start in range(0,nboot,250):
        b=min(250,nboot-start)
        wp=rng.multinomial(len(p),np.full(len(p),1/len(p)),b)
        wn=rng.multinomial(len(n),np.full(len(n),1/len(n)),b)
        vals.extend(np.einsum('bi,ij,bj->b',wp,diff,wn)/(len(p)*len(n)))
    vals=np.asarray(vals)
    lo,hi=np.quantile(vals,[.025,.975]);delta=diff.mean()

    null=vals-delta
    pval=(1+np.sum(np.abs(null)>=abs(delta)))/(nboot+1)
    return dict(delta_auc=float(delta),ci_lo=float(lo),ci_hi=float(hi),p_boot=float(pval))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['eatd','edaic'],required=True)
    ap.add_argument('--suite',choices=['components','weights','expanded','core'],required=True)
    for key in ('text_npz','llm_npz','vocal_npz','outdir'):
        ap.add_argument('--'+key,required=True)
    for key in ('audio_npz','data_root','fold_json','manifest'):
        ap.add_argument('--'+key,default='')
    ap.add_argument('--expanded_llm',default='')
    ap.add_argument('--expanded_vocal',default='')
    ap.add_argument('--n_boot',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--eatd_seed_stride',type=int,choices=[41,53],default=41)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    if args.dataset=='eatd':
        data=load_dataset(args.audio_npz,args.text_npz,args.data_root,threshold=53,strict_expected=True)
        ids,y=data['ids'],data['y'];X={'text':data['text'],
            'llm':strict_profile(args.llm_npz,ids),'vocal':strict_profile(args.vocal_npz,ids)}
        foldobj=load_fold_manifest(args.fold_json,ids);folds=foldobj['folds']
        check_folds(folds,len(y))
    else:
        from edaic_ceef7_common import load_manifest,align_features
        m=load_manifest(args.manifest)
        ids,y,part,X=align_features(m,{'text':args.text_npz,'llm':args.llm_npz,'vocal':args.vocal_npz})
        tr=np.flatnonzero(part=='train');va=np.flatnonzero(part=='test')
        if len(tr)!=219 or len(va)!=56:
            raise ValueError('Expected official 219/56 development/test protocol')
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    if X['llm'].shape[-1]!=12 or X['vocal'].shape[-1]!=8:
        raise ValueError('Expected locked 12/8 indicator ordering')
    for key in ('expanded_llm','expanded_vocal'):
        if getattr(args,key):
            X[key]=strict_profile(getattr(args,key),ids)
    variants_=variants(X,args.suite)
    selections=[];subject_rows=[];fold_rows=[]
    if args.smoke:
        variants_={k:variants_[k] for k in ('text','llm','vocal')}
        args.suite='weights';folds=folds[:1]
    for k,f in enumerate(folds):
        tr=np.asarray(f['train_idx'],int);va=np.asarray(f['val_idx'],int)
        inner={};outer={}
        for name,(b,x,spec,selector,standardize) in variants_.items():
            bi=('text','llm','vocal').index(b)
            if args.dataset=='eatd':
                si=args.seed+1000*k+args.eatd_seed_stride*bi;so=args.seed+2000*k+args.eatd_seed_stride*bi
                inner[name]=oof(x[tr],ids[tr],y[tr],spec,si,args.dataset,selector,standardize)
            else:

                bi={'text':0,'llm':1,'vocal':2}[b];so=args.seed+100*bi
                arr=[]
                for r in range(5):
                    cv=StratifiedKFold(3,shuffle=True,random_state=so+1009*r)
                    pred=np.full(len(tr),np.nan)
                    for fj,(itr,iva) in enumerate(cv.split(np.zeros(len(tr)),y[tr])):
                        pred[iva],_=fit_score(x[tr][itr],y[tr][itr],x[tr][iva],spec,so+97*r+fj,selector,standardize)
                    arr.append(pred)
                inner[name]=np.mean(arr,axis=0)
            outer[name],chosen=fit_score(x[tr],y[tr],x[va],spec,so,selector,standardize)
            if chosen is not None:
                selections.append(dict(repeat=f['repeat'],fold=f['fold'],model=name,selected_indices=json.dumps(chosen.tolist())))
        alpha=select_alpha(inner,y[tr]) if args.suite=='weights' else None
        pin=compose(inner,args.suite,alpha);pout=compose(outer,args.suite,alpha)
        for name,score in pout.items():
            for op in (('train_oof_f1',) if args.dataset=='eatd' else ('train_prevalence','train_oof_f1')):
                threshold=best_f1_threshold(y[tr],pin[name]) if op=='train_oof_f1' else prior_threshold(pin[name],float(y[tr].mean()))
                pred=(score>=threshold).astype(int)
                fold_rows.append(dict(model=name,repeat=f['repeat'],fold=f['fold'],operating_point=op,alpha=alpha,**metric_row(y[va],score,pred=pred),threshold_value=threshold))
                subject_rows.extend(dict(subject_id=str(ids[j]),label=int(y[j]),repeat=f['repeat'],fold=f['fold'],model=name,operating_point=op,score=float(score[i]),pred=int(pred[i])) for i,j in enumerate(va))
        pd.DataFrame(subject_rows).to_csv(out/'subject_predictions.partial.csv',index=False)
        print(f'{args.dataset} {args.suite} fold {k+1}/{len(folds)} completed; alpha={alpha}',flush=True)
    scores=pd.DataFrame(subject_rows);scores.to_csv(out/'subject_predictions.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
    if selections:
        pd.DataFrame(selections).to_csv(out/'training_only_dimension_selections.csv',index=False)
    metrics=[]
    for (name,r,op),d in scores.groupby(['model','repeat','operating_point']):
        if not args.smoke and args.dataset=='eatd' and (len(d)!=162 or d.subject_id.nunique()!=162):
            raise RuntimeError('Subject prediction coverage failure')
        metrics.append(dict(model=name,repeat=r,operating_point=op,**metric_row(d.label.to_numpy(),d.score.to_numpy(),pred=d.pred.to_numpy())))
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(d).assign(operating_point=op) for op,d in metrics.groupby('operating_point')],ignore_index=True)
    summary.to_csv(out/'summary.csv',index=False)
    effects=[]
    for op,dd in scores.groupby('operating_point'):
        wide=dd.groupby(['subject_id','label','model']).score.mean().unstack('model')
        yy=wide.index.get_level_values('label').to_numpy(int)
        for name in sorted(set(wide.columns)-{'SLEF-D'}):
            eff=bootstrap_auc_delta(yy,wide['SLEF-D'].to_numpy(),wide[name].to_numpy(),args.n_boot,args.seed)
            effects.append(dict(candidate='SLEF-D',reference=name,operating_point=op,status='post_lock_exploratory_subject_ensemble',**eff))
    effect=pd.DataFrame(effects)
    for op,dd in effect.groupby('operating_point'):
        effect.loc[dd.index,'p_holm_within_suite']=holm_adjust(dd.p_boot)
    effect.to_csv(out/'paired_effects.csv',index=False)
    metadata=dict(arguments=vars(args),feature_shapes={n:list(x.shape) for n,x in X.items()},
        item_names={'llm':TEXT_NAMES,'vocal':VOCAL_NAMES},groups=GROUPS,
        alpha_grid=ALPHAS.tolist(),locked_alpha=.8,locked_lambda=.1,
        note='Post-lock exploratory sensitivity; no external-label selection. Smaller subsets alone cannot prove 12/8 are globally optimal.')
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    (out/'COMPLETE').write_text('ok\n')

if __name__=='__main__':
    main()
