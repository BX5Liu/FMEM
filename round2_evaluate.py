from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from ablation import (strict_profile,fit_score,zapply,bootstrap_auc_delta,
    BranchSpec,load_dataset,load_fold_manifest,check_folds,make_inner_splits,
    best_f1_threshold,prior_threshold,metric_row,summarize_repeats,safe_auc,holm_adjust)
from round2_extract import STRATEGIES,AUDIO_STRATEGIES

def pooled(x,mode='mean'):
    x=np.asarray(x,float)
    z=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    if mode=='concat': v=z.reshape(len(z),-1)
    elif mode=='meanstd': v=np.concatenate([z.mean(1),z.std(1)],axis=1)
    elif mode=='max': v=z.max(1)
    else: v=z.mean(1)
    v=v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)
    return v[:,None,:]

def score_branch(x,ids,y,tr,va,c,bi,k,dataset,tune=False,raw_nuisance=False):
    def fitted(xtrain,ytrain,xval,C,fitseed):
        if not raw_nuisance:
            return fit_score(xtrain,ytrain,xval,BranchSpec('mean',0,C),fitseed)[0]
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler=StandardScaler().fit(xtrain[:,0,:])
        model=LogisticRegression(C=C,solver='liblinear',class_weight='balanced',max_iter=5000,random_state=fitseed)
        model.fit(scaler.transform(xtrain[:,0,:]),ytrain)
        ref=model.decision_function(scaler.transform(xtrain[:,0,:]))
        return zapply(ref,model.decision_function(scaler.transform(xval[:,0,:])))
    seed=42+100*bi if dataset=='edaic' else 42+1000*k+41*bi
    outerseed=seed if dataset=='edaic' else 42+2000*k+41*bi
    if dataset!='edaic': cvs=list(make_inner_splits(ids[tr],y[tr],3,seed))
    else:
        cvs=[]
        for r in range(5):
            for fj,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+1009*r).split(np.zeros(len(tr)),y[tr])):
                cvs.append((it,iv))
    candidates=sorted(set([c,.01,.03,.1,.3])) if tune else [c]
    score_candidates=[]
    for C in candidates:
        sums=np.zeros(len(tr));counts=np.zeros(len(tr))
        for j,(it,iv) in enumerate(cvs):
            fitseed=seed+97*(j//3)+j%3 if dataset=='edaic' else seed+97*j
            pred=fitted(x[tr][it],y[tr][it],x[tr][iv],C,fitseed)
            sums[iv]+=pred;counts[iv]+=1
        if (counts==0).any(): raise ValueError('OOF coverage')
        score_candidates.append(sums/counts)
    aucs=np.asarray([safe_auc(y[tr],s) for s in score_candidates])
    best=np.flatnonzero(aucs>=aucs.max()-1e-12)
    j=min(best,key=lambda j:(abs(np.log(candidates[j]/c)),candidates[j]))
    C=candidates[j];score=fitted(x[tr],y[tr],x[va],C,outerseed)
    return score_candidates[j],score,C

def choose_fusion(z,y):
    best=None
    for alpha in np.linspace(0,1,11):
        for lam in [0.,.05,.1,.2]:
            s=alpha*z['T']+(1-alpha)*z['L']+lam*np.tanh(z['V'])

            key=(safe_auc(y,s),-abs(alpha-.8),-abs(lam-.1),-lam)
            if best is None or key>best[0]: best=(key,float(alpha),float(lam))
    return best[1:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['eatd','edaic','eatd2','eatd_original'],required=True)
    ap.add_argument('--suite',choices=['cached','regrouped','semantic'],required=True)
    for n in ('text_npz','llm_npz','vocal_npz','outdir'): ap.add_argument('--'+n,required=True)
    for n in ('audio_npz','egemaps_npz','data_root','fold_json','manifest','feature_dir','audit_csv'): ap.add_argument('--'+n,default='')
    for n in ('target_audio','target_text','target_llm','target_vocal','target_root'): ap.add_argument('--'+n,default='')
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if a.dataset in ('eatd','eatd2','eatd_original'):
        data=load_dataset(a.audio_npz,a.text_npz,a.data_root,threshold=53,strict_expected=True)
        ids=np.asarray(data['ids']);y=data['y'];T=data['text'];A=data['audio_sem']
        folds=load_fold_manifest(a.fold_json,ids)['folds'];check_folds(folds,len(y))
        if a.dataset=='eatd_original':
            tr=np.flatnonzero(np.char.startswith(ids.astype(str),'t_'))
            va=np.flatnonzero(np.char.startswith(ids.astype(str),'v_'))
            if (len(tr),len(va),int(y[tr].sum()),int(y[va].sum()))!=(83,79,19,11):
                raise ValueError('Original EATD cohort split/labels changed')
            folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    else:
        from edaic_ceef7_common import load_manifest,align_features
        m=load_manifest(a.manifest)
        ids,y,part,aligned=align_features(m,{'text':a.text_npz,'audio':a.audio_npz})
        T=aligned['text'];A=aligned['audio']
        tr=np.flatnonzero(part=='train');va=np.flatnonzero(part=='test')
        assert len(tr)==219 and len(va)==56
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    L=strict_profile(a.llm_npz,ids);V=strict_profile(a.vocal_npz,ids)
    if a.dataset=='eatd2':
        target=load_dataset(a.target_audio,a.target_text,a.target_root,threshold=53,
                            expected_n=91,expected_neg=53,expected_pos=38,strict_expected=True)
        source_n=len(y);targetids=np.asarray(target['ids'])
        Lt=strict_profile(a.target_llm,targetids);Vt=strict_profile(a.target_vocal,targetids)
        T=np.concatenate([T,target['text']]);A=np.concatenate([A,target['audio_sem']])
        L=np.concatenate([L,Lt]);V=np.concatenate([V,Vt]);y=np.r_[y,target['y']]


        ids=np.r_[ids,['EATD2:'+str(s) for s in targetids]]
        folds=[dict(repeat=0,fold=0,train_idx=list(range(source_n)),val_idx=list(range(source_n,len(y))))]
    branches={'T':(T,.03,0),'L':(L,.1,1),'V':(V,.1,2),'A':(A,.1,3)}
    configs={'SLEF-D':('fixed',['T','L','V']), 'Semantic-only':('single',['T']),
             'Observable-only':('single',['L']),'Vocal-only':('single',['V']),
             'Qwen-audio-only':('single',['A'])}
    diagnostics=[]
    for b,(x,c,bi) in list(branches.items())[:3]:
        rng=np.random.default_rng(42)
        perm=np.asarray([rng.permutation(x.shape[1]) for _ in ids])
        shuffled=np.take_along_axis(x,perm[:,:,None],axis=1)
        delta=float(np.abs(pooled(x)-pooled(shuffled)).max())
        if delta>1e-10: raise ValueError('Mean pooling permutation invariance failed')
        diagnostics.append(dict(branch=b,max_representation_delta_after_context_permutation=delta))
    if a.suite=='cached':
        for mode in ['concat','meanstd','max']:
            keys=[]
            for b in ('T','L','V'):
                name=b+'_'+mode;x,c,bi=branches[b];branches[name]=(pooled(x,mode),c,bi);keys.append(name)
            configs['SLEFD_pool_'+mode]=('fixed',keys)
        for keys in [['T','L'],['T','L','V'],['T','A'],['L','V']]:
            name='concat_'+'_'.join(keys)
            x=np.concatenate([pooled(branches[b][0]) for b in keys],axis=-1)
            branches[name]=(x,.03,0);configs[name+'_fixedC']=('single',[name])
            branches[name+'_nestedC']=(x,.03,0);configs[name+'_nestedC']=('single',[name+'_nestedC'])
        tuned=[]
        for b in ('T','L','V'):
            branches[b+'_nestedC']=branches[b];tuned.append(b+'_nestedC')
        configs['SLEFD_nestedC_fixed_fusion']=('fixed',tuned)
        configs['SLEFD_nestedC_alpha_lambda']=('selected',tuned)
        configs['SLEFD_nested_alpha_lambda']=('selected',['T','L','V'])
        if a.egemaps_npz:
            with np.load(a.egemaps_npz,allow_pickle=True) as z:
                rawids=[str(x) for x in z['subject_ids']];index={s:i for i,s in enumerate(rawids)}
                arr=z['profiles'] if 'profiles' in z else z['features']
                e=np.asarray(arr,float)[[index[str(s)] for s in ids]]
            if e.ndim==2: e=e[:,None,:]
            branches['EG']=(e,.1,3);configs['eGeMAPS-only']=('single',['EG'])
        if a.dataset=='edaic' and a.audit_csv and Path(a.audit_csv).exists():
            d=pd.read_csv(a.audit_csv,dtype={'subject_id':str}).set_index('subject_id').loc[ids]
            modes=np.asarray([float(int(s)>=600) for s in ids])
            nuisance=np.column_stack([np.log1p(d.audio_seconds),np.log1p(d.clean_chars),np.log1p(d.raw_rows),modes])[:,None,:]
            branches['N']=(nuisance,.1,3);configs['duration_length_mode_only']=('single',['N'])
    else:
        featuredir=Path(a.feature_dir)
        for strategy in STRATEGIES:
            t='T_'+strategy;l='L_'+strategy
            branches[t]=(strict_profile(featuredir/('embedding_'+strategy+'.npz'),ids),.03,0)
            configs['Semantic_'+strategy]=('single',[t])
            if a.suite=='semantic': continue
            branches[l]=(strict_profile(featuredir/('observable_'+strategy+'.npz'),ids),.1,1)
            configs['Semantic_'+strategy]=('single',[t]);configs['Observable_'+strategy]=('single',[l])
            configs['Anchor_'+strategy]=('anchor',[t,l])

            configs['SLEFD_'+strategy+'_frozenV']=('fixed',[t,l,'V'])
        for strategy in (AUDIO_STRATEGIES if a.suite=='regrouped' else []):
            v='V_'+strategy;branches[v]=(strict_profile(featuredir/('vocal_'+strategy+'.npz'),ids),.1,2)
            configs['Vocal_'+strategy]=('single',[v])
            configs['SLEFD_frozenText_'+strategy]=('fixed',['T','L',v])
            configs['SLEFD_cleanSentence3_'+strategy]=('fixed',['T_clean_sentence3','L_clean_sentence3',v])
        if a.suite=='regrouped':
            configs['SLEFD_whole2048_audio6']=('fixed',['T_whole2048','L_whole2048','V_whole6'])
            configs['SLEFD_whole8192_audio6']=('fixed',['T_whole8192','L_whole8192','V_whole6'])
            configs['SLEFD_timeline3_audio3x2']=('fixed',['T_timeline3','L_timeline3','V_thirds2'])
            configs['SLEFD_sentence5_audio5x2']=('fixed',['T_sentence5','L_sentence5','V_fifths2'])
    if a.smoke:
        folds=folds[:1];configs={k:v for k,v in configs.items() if k in ('SLEF-D','Semantic-only','SLEFD_nested_alpha_lambda')}
    required={k for _,keys in configs.values() for k in keys}
    subjectrows=[];choices=[];foldmetrics=[]
    for k,f in enumerate(folds):
        tr=np.asarray(f['train_idx'],int);va=np.asarray(f['val_idx'],int);inner={};outer={}
        for b in sorted(required):
            x,c,bi=branches[b]
            inner[b],outer[b],chosen=score_branch(x,ids,y,tr,va,c,bi,k,a.dataset,b.endswith('_nestedC'),raw_nuisance=b=='N')
            if b.endswith('_nestedC'): choices.append(dict(repeat=f['repeat'],fold=f['fold'],model=b,C=chosen))
        for name,(mode,keys) in configs.items():
            if mode=='single': sin,sout=inner[keys[0]],outer[keys[0]];alpha=lam=None
            else:
                alpha,lam=.8,.1 if mode!='anchor' else 0.
                if mode=='selected':
                    alpha,lam=choose_fusion(dict(zip(['T','L','V'],[inner[b] for b in keys])),y[tr])
                    choices.append(dict(repeat=f['repeat'],fold=f['fold'],model=name,alpha=alpha,lambda_value=lam))
                sin=alpha*inner[keys[0]]+(1-alpha)*inner[keys[1]]
                sout=alpha*outer[keys[0]]+(1-alpha)*outer[keys[1]]
                if len(keys)>2: sin=sin+lam*np.tanh(inner[keys[2]]);sout=sout+lam*np.tanh(outer[keys[2]])
            for op in (['train_oof_f1'] if a.dataset in ('eatd','eatd_original') else ['train_prevalence','train_oof_f1']):
                thr=best_f1_threshold(y[tr],sin) if op=='train_oof_f1' else prior_threshold(sin,float(y[tr].mean()))
                pred=(sout>=thr).astype(int)
                met=metric_row(y[va],sout,pred=pred)
                foldmetrics.append(dict(model=name,repeat=f['repeat'],fold=f['fold'],operating_point=op,alpha=alpha,lambda_value=lam,threshold_value=thr,**met))
                subjectrows.extend(dict(subject_id=str(ids[j]),label=int(y[j]),model=name,repeat=f['repeat'],fold=f['fold'],operating_point=op,score=float(sout[i]),pred=int(pred[i]),threshold=float(thr)) for i,j in enumerate(va))
        pd.DataFrame(subjectrows).to_csv(out/'subject_predictions.partial.csv',index=False)
        print(f'{a.dataset} {a.suite} outer fold {k+1}/{len(folds)} complete',flush=True)
    scores=pd.DataFrame(subjectrows);scores.to_csv(out/'subject_predictions.csv',index=False)
    pd.DataFrame(foldmetrics).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'training_only_choices.csv',index=False)
    repeat=[]
    for (model,r,op),d in scores.groupby(['model','repeat','operating_point']):
        repeat.append(dict(model=model,repeat=r,operating_point=op,average_precision=average_precision_score(d.label,d.score),**metric_row(d.label.to_numpy(),d.score.to_numpy(),pred=d.pred.to_numpy())))
    repeat=pd.DataFrame(repeat);repeat.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(d).assign(operating_point=op) for op,d in repeat.groupby('operating_point')],ignore_index=True)
    apstats=repeat.groupby(['model','operating_point']).average_precision.agg(['mean','std']).reset_index().rename(columns={'mean':'average_precision_mean','std':'average_precision_std'})
    summary=summary.merge(apstats,on=['model','operating_point']);summary.to_csv(out/'summary.csv',index=False)
    effects=[]
    for op,d in scores.groupby('operating_point'):
        wide=d.groupby(['subject_id','label','model']).score.mean().unstack('model');yy=wide.index.get_level_values('label').to_numpy(int)
        for name in set(wide)-{'SLEF-D'}:
            effects.append(dict(candidate=name,reference='SLEF-D',operating_point=op,**bootstrap_auc_delta(yy,wide[name].to_numpy(),wide['SLEF-D'].to_numpy(),10000,42)))
    e=pd.DataFrame(effects)
    for op,d in e.groupby('operating_point'): e.loc[d.index,'p_holm_within_suite']=holm_adjust(d.p_boot)
    e.to_csv(out/'paired_effects.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'context_permutation_invariance.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a),configs=configs,feature_shapes={b:list(x.shape) for b,(x,_,_) in branches.items()},
        status='post-lock exploratory; fixed outer split; train-only choices',
        nestedC_threshold_note='C/fusion use training OOF for model selection and threshold; the untouched outer fold provides evaluation, OOF after selection is not an unbiased performance estimate',
        grouping_note='Mean pooling ignores order. Whole extraction and pooling pre-existing segments are different operations.'),indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True);(out/'COMPLETE').write_text('ok\n')

if __name__=='__main__': main()
