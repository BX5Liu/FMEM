from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from ablation import (TEXT_NAMES,VOCAL_NAMES,strict_profile,load_dataset,
    load_fold_manifest,check_folds,make_inner_splits,rank_dimensions,fit_score,
    BranchSpec,best_f1_threshold,prior_threshold,metric_row,summarize_repeats,
    bootstrap_auc_delta,holm_adjust)

TEXT_POOL=TEXT_NAMES+['sleep_disturbance','irritability','rumination','indecision']
VOCAL_POOL=VOCAL_NAMES+['low_loudness_variation','irregular_rhythm',
    'imprecise_articulation','voice_tremor']

def random_sets(n,k,count,seed):
    if count>len(list(itertools.combinations(range(n),k)))-1:
        raise ValueError('Too many unique alternatives')
    rng=np.random.default_rng(seed);seen={tuple(range(k))};result=[]
    while len(result)<count:
        s=tuple(sorted(rng.choice(n,k,replace=False).tolist()))
        if s not in seen:seen.add(s);result.append(s)
    return result

def plans(count=30):
    p={'Original-12-8':('original','original'),
       'Nested-top12':('nested','original'),
       'Nested-top8':('original','nested'),
       'Nested-top12-top8':('nested','nested'),
       'Full-pool16-12':('all','all')}

    for old,new in [(4,12),(5,13),(8,14),(7,15)]:
        s=list(range(12));s[old]=new
        p['Text-swap-'+TEXT_POOL[old]+'-to-'+TEXT_POOL[new]]=(tuple(sorted(s)),'original')
    for old,new in [(2,8),(1,9),(0,10),(7,11)]:
        s=list(range(8));s[old]=new
        p['Vocal-swap-'+VOCAL_POOL[old]+'-to-'+VOCAL_POOL[new]]=('original',tuple(sorted(s)))
    for j,s in enumerate(random_sets(16,12,count,20261007)):
        p[f'Text-random12-{j:02d}']=(s,'original')
    for j,s in enumerate(random_sets(12,8,count,20261008)):
        p[f'Vocal-random8-{j:02d}']=('original',s)
    return p

def score_evidence(x,ids,y,tr,va,budget,selection,c,bi,k,dataset,records,names):
    seed=42+100*bi if dataset=='edaic' else 42+1000*k+41*bi
    outerseed=seed if dataset=='edaic' else 42+2000*k+41*bi
    cvs=[]
    if dataset=='edaic':
        for r in range(5):
            cvs+=list(StratifiedKFold(3,shuffle=True,random_state=seed+1009*r).split(np.zeros(len(tr)),y[tr]))
    else:cvs=list(make_inner_splits(ids[tr],y[tr],3,seed))
    fixed=list(range(budget)) if selection=='original' else list(range(x.shape[-1])) if selection=='all' else None if selection=='nested' else list(selection)
    xx=x if fixed is None else x[...,fixed]
    selector=budget if selection=='nested' else None
    sums=np.zeros(len(tr));counts=np.zeros(len(tr))
    for j,(it,iv) in enumerate(cvs):
        fs=seed+97*(j//3)+j%3 if dataset=='edaic' else seed+97*j
        pred,chosen=fit_score(xx[tr][it],y[tr][it],xx[tr][iv],BranchSpec('mean',0,c),fs,selector)
        sums[iv]+=pred;counts[iv]+=1
        if chosen is not None:
            records.extend(dict(outer_index=k,branch=bi,stage='inner',inner_index=j,indicator=names[i]) for i in chosen)
    pred,chosen=fit_score(xx[tr],y[tr],xx[va],BranchSpec('mean',0,c),outerseed,selector)
    if chosen is not None:
        records.extend(dict(outer_index=k,branch=bi,stage='outer',inner_index=-1,indicator=names[i]) for i in chosen)
    if (counts==0).any():raise ValueError('OOF coverage failure')
    return sums/counts,pred

def train_diagnostics(x,names,tr,k):

    z=x[tr].mean(1);rows=[];pairs=[]
    corr=np.asarray(spearmanr(z,axis=0).statistic)
    for i,name in enumerate(names):
        rows.append(dict(outer_index=k,indicator=name,train_subjects=len(tr),
            mean_logodds=float(z[:,i].mean()),std_logodds=float(z[:,i].std()),
            iqr_logodds=float(np.diff(np.percentile(z[:,i],[25,75]))[0]),
            fraction_positive_logodds=float((z[:,i]>0).mean()),
            fraction_absolute_logodds_over5=float((np.abs(z[:,i])>5).mean()),
            distinct_subject_values=int(len(np.unique(z[:,i])))))
        for j in range(i):
            pairs.append(dict(outer_index=k,indicator_a=names[j],indicator_b=name,spearman=float(corr[i,j])))
    return rows,pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['eatd','edaic'],required=True)
    for n in ('text_npz','llm_npz','vocal_npz','expanded_llm','expanded_vocal','outdir'):ap.add_argument('--'+n,required=True)
    for n in ('audio_npz','data_root','fold_json','manifest'):ap.add_argument('--'+n,default='')
    ap.add_argument('--random_sets',type=int,default=30);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if a.dataset=='eatd':
        d=load_dataset(a.audio_npz,a.text_npz,a.data_root,threshold=53,strict_expected=True)
        ids=np.asarray(d['ids']);y=d['y'];T=d['text'];folds=load_fold_manifest(a.fold_json,ids)['folds'];check_folds(folds,len(y))
    else:
        from edaic_ceef7_common import load_manifest,align_features
        ids,y,part,aligned=align_features(load_manifest(a.manifest),{'text':a.text_npz})
        T=aligned['text'];tr=np.flatnonzero(part=='train');va=np.flatnonzero(part=='test')
        if (len(tr),len(va))!=(219,56):raise ValueError('Official subject split changed')
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    L=strict_profile(a.expanded_llm,ids);V=strict_profile(a.expanded_vocal,ids)
    np.testing.assert_array_equal(L[...,:12],strict_profile(a.llm_npz,ids))
    np.testing.assert_array_equal(V[...,:8],strict_profile(a.vocal_npz,ids))
    if (L.shape[-1],V.shape[-1])!=(16,12):raise ValueError('Unexpected candidate pool')
    design=plans(2 if a.smoke else a.random_sets)
    (out/'preregistered_compositions.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    if a.smoke:folds=folds[:1]
    from round2_evaluate import score_branch
    predictions=[];selections=[];diagnostics=[];correlations=[]
    for k,f in enumerate(folds):
        tr=np.asarray(f['train_idx']);va=np.asarray(f['val_idx'])
        tin,tout,_=score_branch(T,ids,y,tr,va,.03,0,k,a.dataset)
        caches={}
        for b,(x,names,budget,bi) in {'L':(L,TEXT_POOL,12,1),'V':(V,VOCAL_POOL,8,2)}.items():
            rows,pairs=train_diagnostics(x,names,tr,k)
            diagnostics.extend(dict(branch=b,**r) for r in rows);correlations.extend(dict(branch=b,**r) for r in pairs)
            for selection in dict.fromkeys(v[bi-1] for v in design.values()):
                caches[b,selection]=score_evidence(x,ids,y,tr,va,budget,selection,.1,bi,k,a.dataset,selections,names)
        for name,(ls,vs) in design.items():
            lin,lout=caches['L',ls];vin,vout=caches['V',vs]
            scores=[(name,.8*tin+.2*lin+.1*np.tanh(vin),.8*tout+.2*lout+.1*np.tanh(vout))]
            if vs=='original':scores.append((name+'__Observable-only',lin,lout))
            if ls=='original':scores.append((name+'__Vocal-only',vin,vout))
            for model,sin,sout in scores:
                op='train_prevalence' if a.dataset=='edaic' else 'train_oof_f1'
                threshold=prior_threshold(sin,float(y[tr].mean())) if a.dataset=='edaic' else best_f1_threshold(y[tr],sin)
                predictions.extend(dict(subject_id=str(ids[idx]),label=int(y[idx]),repeat=f['repeat'],fold=f['fold'],model=model,
                    operating_point=op,score=float(sout[i]),pred=int(sout[i]>=threshold),threshold=float(threshold)) for i,idx in enumerate(va))
        pd.DataFrame(predictions).to_csv(out/'subject_predictions.partial.csv',index=False)
        print(f'{a.dataset}: composition outer {k+1}/{len(folds)} complete',flush=True)
    df=pd.DataFrame(predictions);df.to_csv(out/'subject_predictions.csv',index=False)
    repeat=[]
    for (model,r,op),g in df.groupby(['model','repeat','operating_point']):
        repeat.append(dict(model=model,repeat=r,operating_point=op,**metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())))
    rep=pd.DataFrame(repeat);rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary=summarize_repeats(rep);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'training_only_selections.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'training_only_indicator_variation.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'training_only_redundancy.csv',index=False)
    selection=pd.DataFrame(selections)
    frequencies=[]
    for (bi,stage),g in selection.groupby(['branch','stage']):
        denom=len(g[['outer_index','inner_index']].drop_duplicates())
        for name in (TEXT_POOL if bi==1 else VOCAL_POOL):
            frequencies.append(dict(branch=bi,stage=stage,indicator=name,selected_fits=int((g.indicator==name).sum()),total_fits=denom,frequency=float((g.indicator==name).sum()/denom)))
    pd.DataFrame(frequencies).to_csv(out/'selection_stability.csv',index=False)
    effects=[];wide=df.groupby(['subject_id','label','model']).score.mean().unstack('model');yy=wide.index.get_level_values('label').to_numpy(int)
    for name in ['Nested-top12','Nested-top8','Nested-top12-top8','Full-pool16-12']:
        effects.append(dict(candidate=name,reference='Original-12-8',**bootstrap_auc_delta(yy,wide[name].to_numpy(),wide['Original-12-8'].to_numpy(),10000,42)))
    effects=pd.DataFrame(effects);effects['p_holm_four_planned']=holm_adjust(effects.p_boot);effects.to_csv(out/'paired_effects.csv',index=False)
    distribution=[];baseline=float(summary.set_index('model').loc['Original-12-8','auc_mean'])
    for prefix in ['Text-random12-','Vocal-random8-']:
        v=summary.loc[summary.model.str.startswith(prefix)&~summary.model.str.contains('__'),'auc_mean'].to_numpy()
        distribution.append(dict(family=prefix,count=len(v),original_auc=baseline,random_mean=float(v.mean()),random_min=float(v.min()),random_max=float(v.max()),
            original_descriptive_percentile=float(100*((v<baseline-1e-12).sum()+.5*(np.abs(v-baseline)<=1e-12).sum())/len(v))))
    pd.DataFrame(distribution).to_csv(out/'equal_budget_distribution.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a),candidate_text=TEXT_POOL,candidate_vocal=VOCAL_POOL,
        scope='Post-lock exploratory finite-pool composition controls; no global optimum or clinical indicator validity claim',
        random_note='30 unique fixed-size alternatives per branch; percentile is descriptive, not a hypothesis-test p-value. Evaluation outcomes never choose final indicators.',
        stability_note='Selection frequencies count correlated training fits, not independent experiment repeats.',
        diagnostics_note='Raw model-score variation and correlation on training subjects; not clinical prevalence or human evidence validation.',
        fusion_weights=[.8,.2,.1],split=folds),indent=2),encoding='utf-8')
    (out/'COMPLETE').write_text('ok\n');print(pd.DataFrame(distribution).to_string(index=False),flush=True)

if __name__=='__main__':main()
