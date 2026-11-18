import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from ablation import (strict_profile,load_dataset,load_fold_manifest,check_folds,fit_score,
    BranchSpec,make_inner_splits,safe_auc,best_f1_threshold,prior_threshold,metric_row,
    summarize_repeats,bootstrap_auc_delta,holm_adjust,TEXT_NAMES,VOCAL_NAMES,GROUPS)
from tune_fusion import Engine,inner_splits
from round2_evaluate import score_branch,pooled
from indicator_justification import plans as composition_plans

class EvidenceEngine(Engine):
    def __init__(self,ids,y,xs,dataset,names=None,selectors=None,disabled=(),shared=None,standardize=True):
        super().__init__(ids,y,xs,dataset)
        self.names=names or ['T','L','V'];self.selectors=selectors or [None]*3
        self.disabled=set(disabled);self.shared={} if shared is None else shared;self.standardize=standardize
    def branches(self,tr,ev,seed):
        train=[];test=[]
        for b,x in enumerate(self.X):
            if b in self.disabled:
                train.append(np.zeros(len(tr)));test.append(np.zeros(len(ev)));continue
            key=('stack',self.names[b],self.selectors[b],b,self.standardize,tuple(tr),tuple(ev),seed,self.y[tr].tobytes())
            if key not in self.shared:
                C=.03 if b==0 else .1;sin=np.full(len(tr),np.nan)
                for j,(it,iv) in enumerate(inner_splits(self.ids[tr],self.y[tr],seed+100*b,self.dataset)):
                    sin[iv],chosen=fit_score(x[tr[it]],self.y[tr[it]],x[tr[iv]],BranchSpec('mean',0,C),seed+100*b+97*j,
                        selector=self.selectors[b],standardize=self.standardize)
                    if chosen is not None:self.shared.setdefault('_selections',[]).append(dict(branch=self.names[b],budget=self.selectors[b],stage='inner',seed=seed+100*b+97*j,selected=list(map(int,chosen)),training_n=len(it)))
                sout,chosen=fit_score(x[tr],self.y[tr],x[ev],BranchSpec('mean',0,C),seed+100*b,
                    selector=self.selectors[b],standardize=self.standardize)
                if chosen is not None:self.shared.setdefault('_selections',[]).append(dict(branch=self.names[b],budget=self.selectors[b],stage='outer',seed=seed+100*b,selected=list(map(int,chosen)),training_n=len(tr)))
                if not np.isfinite(sin).all():raise ValueError('Source branch OOF coverage')
                self.shared[key]=(sin,sout)
            a,bout=self.shared[key];train.append(a);test.append(bout)
        return np.column_stack(train),np.column_stack(test)
    def legacy_branches(self,tr,ev,k):
        sin=[];sout=[]
        for b,x in enumerate(self.X):
            if b in self.disabled:
                sin.append(np.zeros(len(tr)));sout.append(np.zeros(len(ev)));continue
            key=('legacy',self.names[b],b,tuple(tr),tuple(ev),k,self.y[tr].tobytes())
            if key not in self.shared:
                self.shared[key]=score_branch(x,self.ids,self.y,tr,ev,.03 if b==0 else .1,b,k,self.dataset)[:2]
            a,bout=self.shared[key];sin.append(a);sout.append(bout)
        return np.column_stack(sin),np.column_stack(sout)

def transform_fixed(z,kind):
    anchor=.8*z[:,0]+.2*z[:,1]
    if kind=='anchor':return anchor
    if kind=='unbounded':return anchor+.1*z[:,2]
    if kind=='probability':return .9/(1+np.exp(-np.clip(anchor,-700,700)))+.1/(1+np.exp(-np.clip(z[:,2],-700,700)))
    if kind=='equal':return z.mean(1)
    if kind.startswith('single'):return z[:,int(kind[-1])]
    return anchor+.1*np.tanh(z[:,2])

def config(name,kind='linear',**kw):return dict(name=name,kind=kind,**kw)

def design(suite,has_audio,has_direct,has_expanded,has_egemaps,has_matched=False):
    p=[config('SLEF-D-original','fixed'),config('SLEF-D-matched-fixed','matched_fixed'),config('SLEF-D-linear')]
    if suite=='core':
        p += [config('Text-Symptom-Anchor','anchor'),config('Linear-residual-control','unbounded'),
              config('Probability-fusion-control','probability'),config('Equal-score-fusion','equal')]
        for b,label in enumerate(['semantic','observable','vocal']):
            p += [config('Fixed-without-'+label,'fixed',disabled=[b]),config('Linear-without-'+label,disabled=[b]),
                  config('Single-'+label,'single'+str(b)),config('Linear-only-'+label,disabled=[i for i in range(3) if i!=b])]
        p += [config('Evidence-concat-LR','feature_lr')]
        if has_audio:p += [config('Generic-audio-only','single2',replace_v='A'),config('Anchor-plus-generic-audio','fixed',replace_v='A'),
                           config('Linear-replace-vocal-with-audio',replace_v='A'),config('Generic-TA-concat-LR','feature_lr',replace_v='A',disabled=[1])]
        if has_matched:p += [config('Matched-Qwen2-Audio-only','single2',replace_v='AM'),config('Anchor-plus-matched-audio','fixed',replace_v='AM'),
            config('Linear-replace-vocal-with-matched-audio',replace_v='AM'),config('Matched-TA-concat-LR','feature_lr',replace_v='AM',disabled=[1])]
        if has_direct:p += [config('Direct-audio-only','single2',replace_v='D'),config('Anchor-plus-direct-audio','fixed',replace_v='D'),config('Linear-replace-vocal-with-direct',replace_v='D')]
        if has_egemaps:p += [config('eGeMAPS-only','single2',replace_v='EG'),config('Linear-replace-vocal-with-eGeMAPS',replace_v='EG')]
        p += [config('Linear-random-observable',noise=[1]),config('Linear-random-vocal',noise=[2]),config('Linear-random-both',noise=[1,2])]
    elif suite=='indicators':
        for b,names in [(1,TEXT_NAMES),(2,VOCAL_NAMES)]:
            for j,name in enumerate(names):p.append(config('Linear-drop-'+name,drop=[b,[j]]))
        for group,drop in GROUPS.items():p.append(config('Linear-without-'+group,drop=[1 if group.startswith('llm') else 2,drop]))
        for b,budgets in [(1,[3,6,9]),(2,[2,4,6])]:
            for budget in budgets:
                p.append(config(('Observable' if b==1 else 'Vocal')+'-nested-top'+str(budget),selector=[b,budget]))
                for r in range(3):
                    keep=np.sort(np.random.default_rng(1300+budget*17+r).choice(12 if b==1 else 8,budget,replace=False)).tolist()
                    p.append(config(('Observable' if b==1 else 'Vocal')+'-random'+str(budget)+'-'+str(r),keep=[b,keep]))
        if has_expanded:
            p += [config('Linear-full-pool16-12',expanded='both'),config('Linear-expanded-text16',expanded='L'),config('Linear-expanded-vocal12',expanded='V'),
                  config('Linear-expanded-top12',expanded='L',selector=[1,12]),config('Linear-expanded-top8',expanded='V',selector=[2,8]),
                  config('Linear-expanded-top12-top8',expanded='both',selectors=[None,12,8])]
    elif suite=='compositions':
        if not has_expanded:raise ValueError('Expanded pool required for composition experiments')
        for name,(ls,vs) in composition_plans(30).items():
            p.append(config('Linear-composition-'+name,composition=[ls,vs]))
    elif suite=='robustness':
        p += [config('Linear-C-'+str(C),fixed_C=C) for C in [.01,.1,1.]]
        for j in range(3):p.append(config('Linear-segment'+str(j)+'-only',context=j))
        for mode in ['concat','meanstd','max']:p.append(config('Linear-pooling-'+mode,pooling=mode))
        p += [config('Linear-without-score-normalization',standardize=False)]
        p += [config('Linear-train-label-permutation-'+str(r),label_permutation=r) for r in range(10)]
    else:raise ValueError('Unknown suite')
    return p

def engine_for(plan,ids,y,banks,dataset,shared):
    xs=[banks[b] for b in ['T','L','V']];names=['T','L','V'];selectors=[None]*3
    if plan.get('replace_v'):xs[2]=banks[plan['replace_v']];names[2]=plan['replace_v']
    if plan.get('expanded'):
        for b,label in [(1,'L'),(2,'V')]:
            if plan['expanded'] in [label,'both']:xs[b]=banks[label+'X'];names[b]=label+'X'
    if 'composition' in plan:
        for b,choice in zip([1,2],plan['composition']):
            if choice=='original':continue
            label='L' if b==1 else 'V';xs[b]=banks[label+'X'];names[b]=label+'X'
            if choice=='nested':selectors[b]=12 if b==1 else 8
            elif choice!='all':xs[b]=xs[b][...,choice];names[b]+='-keep'+str(choice)
    if 'drop' in plan:
        b,drop=plan['drop'];keep=[i for i in range(xs[b].shape[-1]) if i not in drop]
        xs[b]=xs[b][...,keep];names[b]+='-keep'+str(keep)
    if 'keep' in plan:
        b,keep=plan['keep'];xs[b]=xs[b][...,keep];names[b]+='-keep'+str(keep)
    if 'selector' in plan:selectors[plan['selector'][0]]=plan['selector'][1]
    if 'selectors' in plan:selectors=plan['selectors']
    for b in plan.get('noise',[]):
        xs[b]=np.random.default_rng(20261007+b).normal(size=xs[b].shape);names[b]='noise-'+names[b]
    if 'context' in plan:
        xs=[x[:,plan['context']:plan['context']+1] for x in xs];names=[s+'-ctx'+str(plan['context']) for s in names]
    if 'pooling' in plan:
        xs=[pooled(x,plan['pooling']) for x in xs];names=[s+'-pool-'+plan['pooling'] for s in names]
    disabled=plan.get('disabled',[])
    if plan['kind']=='feature_lr':
        xs=[np.zeros_like(x) if b in disabled else x for b,x in enumerate(xs)]
    return EvidenceEngine(ids,y,xs,dataset,names,selectors,disabled,shared,plan.get('standardize',True))

def choose_and_predict(engine,plan,tr,ev,k,dev=None):
    seed=42+1000*k;kind=plan['kind'];family=kind in ['linear','feature_lr']
    if kind=='weight_grid':
        if dev is None:
            zi=np.full((len(tr),3),np.nan)
            for j,(it,iv) in enumerate(inner_splits(engine.ids[tr],engine.y[tr],seed,engine.dataset)):
                zi[iv]=engine.branches(tr[it],tr[iv],seed+97*j)[1]
            cy=engine.y[tr]
        else:zi=engine.branches(tr,dev,seed+909)[1];cy=engine.y[dev]
        zt=engine.branches(tr,ev,seed+909)[1]
        grid=[(alpha,lam) for alpha in [.5,.6,.7,.8,.9,1.] for lam in [.05,.1,.15,.2]]
        def mix(z,g):return g[0]*z[:,0]+(1-g[0])*z[:,1]+g[1]*np.tanh(z[:,2])
        aucs=[safe_auc(cy,mix(zi,g)) for g in grid];best=max(aucs);j=next(i for i,v in enumerate(aucs) if v>=best-1e-12)
        return mix(zi,grid[j]),mix(zt,grid[j]),dict(C=None,alpha=grid[j][0],lambda_audio=grid[j][1],inner_auc=aucs[j],selection='Training/development AUC grid; lambda>0'),None
    if kind=='matched_fixed':
        if dev is None:
            sin=np.full(len(tr),np.nan)
            for j,(it,iv) in enumerate(inner_splits(engine.ids[tr],engine.y[tr],seed,engine.dataset)):
                _,z=engine.branches(tr[it],tr[iv],seed+97*j);sin[iv]=transform_fixed(z,'fixed')
        else:
            _,z=engine.branches(tr,dev,seed+909);sin=transform_fixed(z,'fixed')
        _,z=engine.branches(tr,ev,seed+909)
        return sin,transform_fixed(z,'fixed'),dict(C=None),None
    if not family:
        if dev is None:
            zin,zout=engine.legacy_branches(tr,ev,k);return transform_fixed(zin,kind),transform_fixed(zout,kind),dict(C=None),None
        _,zd=engine.legacy_branches(tr,dev,k);_,zt=engine.legacy_branches(tr,ev,k)
        return transform_fixed(zd,kind),transform_fixed(zt,kind),dict(C=None),None
    options=[plan['fixed_C']] if 'fixed_C' in plan else [.01,.1,1.]
    cals=[];inner=list(inner_splits(engine.ids[tr],engine.y[tr],seed,engine.dataset)) if dev is None else None
    for C in options:
        c=dict(mode='stack_linear' if kind=='linear' else 'feature_lr',C=C)
        if dev is not None:cal=engine.predict(c,tr,dev,seed+909)[0]
        else:
            cal=np.full(len(tr),np.nan)
            for j,(it,iv) in enumerate(inner):cal[iv]=engine.predict(c,tr[it],tr[iv],seed+97*j)[0]
        cals.append(cal)
    cy=engine.y[dev] if dev is not None else engine.y[tr]
    aucs=[safe_auc(cy,s) for s in cals];best=max(aucs);j=next(i for i,s in enumerate(aucs) if s>=best-1e-12)
    C=options[j];c=dict(mode='stack_linear' if kind=='linear' else 'feature_lr',C=C)
    pred=engine.predict(c,tr,ev,seed+909)[0];weights=None
    if kind=='linear':
        zin,_=engine.branches(tr,ev,seed+909);ss=StandardScaler().fit(zin)
        model=LogisticRegression(C=C,solver='liblinear',class_weight='balanced',random_state=seed+909,max_iter=3000).fit(ss.transform(zin),engine.y[tr])
        weights=model.coef_[0].tolist()
        if plan['name']=='SLEF-D-linear' and abs(weights[2])<=1e-12:raise ValueError('Main model lost audio branch')
    return cals[j],pred,dict(C=C,inner_auc=aucs[j],selection='development AUC' if dev is not None else 'training OOF AUC'),weights

def load(a):
    if a.dataset.startswith('eatd'):
        d=load_dataset(a.audio_npz,a.text_npz,a.data_root,threshold=53,strict_expected=True)
        ids=np.asarray(d['ids']);y=np.asarray(d['y'],int);banks=dict(T=d['text'],A=d['audio_sem'])
        folds=load_fold_manifest(a.fold_json,ids)['folds'];check_folds(folds,len(ids));dataset='eatd'
        if a.dataset=='eatd_original':
            tr=np.flatnonzero(np.char.startswith(ids.astype(str),'t_'));ev=np.flatnonzero(np.char.startswith(ids.astype(str),'v_'))
            folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=ev.tolist())]
    else:
        from edaic_ceef7_common import load_manifest
        m=load_manifest(a.manifest);ids=m.subject_id.to_numpy(str);y=m.label.to_numpy(int)
        banks=dict(T=strict_profile(a.text_npz,ids),A=strict_profile(a.audio_npz,ids));dataset='edaic'
        splits=m.official_split.to_numpy(str)
        tr=np.flatnonzero(splits=='train') if a.dataset=='edaic_official' else np.flatnonzero(splits!='test')
        ev=np.flatnonzero(splits=='test');dev=np.flatnonzero(splits=='dev') if a.dataset=='edaic_official' else None
        if (len(tr),len(ev))!=((163,56) if dev is not None else (219,56)):raise ValueError('Official cohort changed')
        if dev is not None and len(dev)!=56:raise ValueError('Development cohort changed')
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=ev.tolist(),dev_idx=None if dev is None else dev.tolist())]
    banks.update(L=strict_profile(a.llm_npz,ids),V=strict_profile(a.vocal_npz,ids))
    for arg,key in [('direct_npz','D'),('expanded_llm','LX'),('expanded_vocal','VX'),('egemaps_npz','EG'),('matched_audio','AM')]:
        path=getattr(a,arg)
        if path:banks[key]=strict_profile(path,ids)
    if 'LX' in banks:
        np.testing.assert_array_equal(banks['LX'][...,:12],banks['L']);np.testing.assert_array_equal(banks['VX'][...,:8],banks['V'])
    if banks['L'].shape[-1]!=12 or banks['V'].shape[-1]!=8:raise ValueError('Primary evidence dimensionality changed')
    return ids,y,banks,folds,dataset

def save_results(rows,choices,out,a,folds,plan):
    reference='SLEF-D-linear' if a.method_family=='linear' else 'SLEF-D-weight-grid'
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False);rep=[]
    for (model,repeat,op),g in df.groupby(['model','repeat','operating_point']):
        if g.subject_id.duplicated().any():raise ValueError('Repeated evaluation person within repeat')
        met=metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())
        rep.append(dict(model=model,repeat=repeat,operating_point=op,average_precision=average_precision_score(g.label,g.score),**met))
    rep=pd.DataFrame(rep);rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(g).assign(operating_point=op) for op,g in rep.groupby('operating_point')],ignore_index=True)
    ap=rep.groupby(['model','operating_point']).average_precision.agg(['mean','std']).reset_index().rename(columns={'mean':'average_precision_mean','std':'average_precision_std'})
    summary=summary.merge(ap,on=['model','operating_point']);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'training_only_choices.csv',index=False)
    effects=[]
    for op,g in df.groupby('operating_point'):
        w=g.groupby(['subject_id','label','model']).score.mean().unstack('model');yy=w.index.get_level_values('label').to_numpy(int)
        for candidate in w.columns:
            if candidate==reference:continue
            effects.append(dict(candidate=reference,reference=candidate,operating_point=op,
                analysis='subject repeat-ensemble' if a.dataset=='eatd' else 'single test subjects',
                **bootstrap_auc_delta(yy,w[reference].to_numpy(),w[candidate].to_numpy(),a.n_boot,42)))
    effects=pd.DataFrame(effects)
    for op,g in effects.groupby('operating_point'):effects.loc[g.index,'p_holm_within_declared_suite']=holm_adjust(g.p_boot)
    effects.to_csv(out/'paired_effects.csv',index=False)
    if a.suite=='compositions':
        z=summary.loc[summary.operating_point==summary.operating_point.iloc[0]];base=float(z.loc[z.model==reference,'auc_mean'].iloc[0]);dist=[]
        for family in ['Text-random12-','Vocal-random8-']:
            prefix='Linear' if a.method_family=='linear' else 'Weight-grid'
            v=z.loc[z.model.str.startswith(prefix+'-composition-'+family),'auc_mean'].to_numpy()
            dist.append(dict(family=family,n=len(v),original_auc=base,random_mean=float(v.mean()),random_min=float(v.min()),random_max=float(v.max()),
                original_descriptive_percentile=float(100*((v<base-1e-12).sum()+.5*(np.abs(v-base)<=1e-12).sum())/len(v))))
        pd.DataFrame(dist).to_csv(out/'equal_budget_distribution.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a),plans=plan,folds=folds,
        scope='Version-specific original vs learned linear multimodal evaluation; no test-table maximum is promoted to the selected method',
        selection='Outer training OOF AUC; official163 uses development AUC. Threshold selected only on source OOF/development.',
        statistics='Repeat SD is split sensitivity; bootstrap uses one mean score per subject; random-set percentile is descriptive.',
        ablation='Remove a branch/indicator before fitting and refit the entire affected branch and meta head. Single-modality rows are controls only.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf8')
    (out/'COMPLETE').write_text('ok\n');print(summary.to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['eatd','eatd_original','edaic','edaic_official'],required=True)
    ap.add_argument('--method_family',choices=['linear','weight_grid'],default='linear')
    ap.add_argument('--suite',choices=['core','indicators','compositions','robustness'],required=True)
    for n in ['text_npz','llm_npz','vocal_npz','audio_npz','outdir']:ap.add_argument('--'+n,required=True)
    for n in ['manifest','data_root','fold_json','direct_npz','expanded_llm','expanded_vocal','egemaps_npz','matched_audio']:ap.add_argument('--'+n,default='')
    ap.add_argument('--n_boot',type=int,default=10000);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);ids,y,banks,folds,dataset=load(a)
    plan=design(a.suite,'A' in banks,'D' in banks,'LX' in banks,'EG' in banks,'AM' in banks)
    if a.method_family=='weight_grid':
        plan=[p for p in plan if 'fixed_C' not in p]
        for p in plan:
            if p['kind']=='linear':
                p['kind']='weight_grid';p['name']='SLEF-D-weight-grid' if p['name']=='SLEF-D-linear' else p['name'].replace('Linear','Weight-grid')
    if a.smoke:
        folds=folds[:1];plan=[p for p in plan if p['name'] in ['SLEF-D-original','SLEF-D-linear','Linear-without-vocal','Observable-nested-top6']]
    definition=dict(arguments=vars(a),plans=plan,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    definition=json.loads(json.dumps(definition))
    existing=out/'run_definition.json'
    if existing.exists() and json.loads(existing.read_text())!=definition:raise ValueError('Definition changed; use a new output directory')
    existing.write_text(json.dumps(definition,indent=2))
    rows=[];choices=[]
    for k,f in enumerate(folds):
        checkpoint=out/('outer'+str(k)+'_completed.json')
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text());rows.extend(saved['rows']);choices.extend(saved['choices']);continue
        tr=np.asarray(f['train_idx']);ev=np.asarray(f['val_idx']);dev=None if not f.get('dev_idx') else np.asarray(f['dev_idx'])
        if set(tr)&set(ev) or dev is not None and (set(tr)&set(dev) or set(dev)&set(ev)):raise ValueError('Subject split overlap')
        shared={};local=[];localchoices=[]
        for p in plan:
            training_y=y.copy()
            if 'label_permutation' in p:training_y[tr]=np.random.default_rng(9000+p['label_permutation']+100*k).permutation(y[tr])
            engine=engine_for(p,ids,training_y,banks,dataset,shared)
            sin,sout,choice,weights=choose_and_predict(engine,p,tr,ev,k,dev)


            cy=training_y[tr] if dev is None else y[dev]
            ops=['train_oof_f1'] if dataset=='eatd' else ['train_prevalence','train_oof_f1']
            if dev is not None:ops=['development_f1','development_prevalence']
            localchoices.append(dict(model=p['name'],repeat=f['repeat'],fold=f['fold'],**choice,meta_weights=None if weights is None else json.dumps(weights)))
            for op in ops:
                th=best_f1_threshold(cy,sin) if op in ['train_oof_f1','development_f1'] else prior_threshold(sin,float(cy.mean()))
                local.extend(dict(subject_id=str(ids[idx]),label=int(y[idx]),model=p['name'],repeat=f['repeat'],fold=f['fold'],operating_point=op,
                    score=float(sout[j]),pred=int(sout[j]>=th),threshold=float(th)) for j,idx in enumerate(ev))
            print(f'{a.dataset} {a.suite} outer {k+1}/{len(folds)} {p["name"]}',flush=True)
        (out/('outer'+str(k)+'_training_only_selection.json')).write_text(json.dumps(shared.get('_selections',[])))
        checkpoint.write_text(json.dumps(dict(rows=local,choices=localchoices)))
        rows.extend(local);choices.extend(localchoices);pd.DataFrame(rows).to_csv(out/'subject_predictions.partial.csv',index=False)
    save_results(rows,choices,out,a,folds,plan)

if __name__=='__main__':main()
