import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from version_validation import load,engine_for,choose_and_predict,config
from fixed_score_fusion import predict
from ablation import TEXT_NAMES,VOCAL_NAMES,GROUPS,best_f1_threshold,prior_threshold,metric_row,summarize_repeats,bootstrap_auc_delta,holm_adjust

VOCAL12_NAMES=VOCAL_NAMES+['low_loudness_variation','irregular_rhythm','imprecise_articulation','voice_tremor']

def design(suite):
    base=config('SLEF-D-12x12','balanced',expanded='V',fusion='equal')
    p=[base]
    if suite=='core':
        p += [config('SLEF-D-12x12-unbounded','balanced',expanded='V',fusion='equal'),config('SLEF-D-12x12-reliability','balanced',expanded='V',fusion='reliability'),
          config('SLEF-D-original','fixed'),config('Original-formula-Vocal12','fixed',expanded='V'),config('Probability-equal-Vocal12','balanced',expanded='V',fusion='probability'),
          config('Equal-without-score-normalization','balanced',expanded='V',fusion='equal',standardize=False),config('Evidence-concat-LR','feature_lr',expanded='V'),
          config('Balanced-original-Vocal8','balanced',fusion='equal'),config('Balanced-selected-Vocal8','balanced',expanded='V',selector=[2,8],fusion='equal')]
        for b,n in enumerate(['semantic','observable','vocal']):
            p += [config('Balanced-without-'+n,'balanced',expanded='V',disabled=[b],fusion='equal'),config('Single-'+n,'single'+str(b),expanded='V',disabled=[j for j in range(3) if j!=b])]
        p += [config('Balanced-matched-audio','balanced',replace_v='AM',fusion='equal'),config('Matched-audio-only','single2',replace_v='AM')]
    elif suite=='indicators':
        for b,names in [(1,TEXT_NAMES),(2,VOCAL12_NAMES)]:
            for j,n in enumerate(names):p.append(config('Drop-'+n,'balanced',expanded='V',drop=[b,[j]],fusion='equal'))
        groups=dict(GROUPS,vocal_prosody=[2,3,8,9],vocal_quality=[7,10,11])
        for group,drop in groups.items():p.append(config('Without-'+group,'balanced',expanded='V',drop=[1 if group.startswith('llm') else 2,drop],fusion='equal'))
        for b,budgets in [(1,[4,8]),(2,[4,8])]:
            for budget in budgets:p.append(config(('Observable' if b==1 else 'Vocal')+'-nested-top'+str(budget),'balanced',expanded='V',selector=[b,budget],fusion='equal'))
        p += [config('Expanded-text16-control','balanced',expanded='both',fusion='equal'),config('Expanded-text-selected12','balanced',expanded='both',selector=[1,12],fusion='equal')]
    elif suite=='compositions':
        for r in range(30):
            for b,pool,budget,name in [(1,16,12,'Text-random12'),(2,12,8,'Vocal-random8')]:
                keep=np.sort(np.random.default_rng(91000+1000*b+r).choice(pool,budget,replace=False)).tolist()
                p.append(config(name+'-'+str(r),'balanced',expanded='both' if b==1 else 'V',keep=[b,keep],fusion='equal'))
        p += [config('Vocal-original8','balanced',fusion='equal'),config('Vocal-nested8','balanced',expanded='V',selector=[2,8],fusion='equal')]
    elif suite=='robustness':
        p.append(config('Voice-unbounded','balanced',expanded='V',fusion='equal'))
        for mode in ['concat','meanstd','max']:p.append(config('Pooling-'+mode,'balanced',expanded='V',pooling=mode,fusion='equal'))
        for j in range(3):p.append(config('Context-'+str(j),'balanced',expanded='V',context=j,fusion='equal'))
        for b,n in [(1,'observable'),(2,'vocal')]:p.append(config('Random-'+n,'balanced',expanded='V',noise=[b],fusion='equal'))
        for j in range(10):p.append(config('Train-label-permutation-'+str(j),'balanced',expanded='V',label_permutation=j,fusion='equal'))
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['eatd','eatd_original','edaic','edaic_official'],required=True);ap.add_argument('--suite',choices=['core','indicators','compositions','robustness'],default='core')
    for n in ['text_npz','llm_npz','vocal_npz','audio_npz','expanded_vocal','matched_audio','outdir']:ap.add_argument('--'+n,required=True)
    for n in ['data_root','fold_json','manifest','expanded_llm']:ap.add_argument('--'+n,default='')
    a=ap.parse_args();a.direct_npz='';a.egemaps_npz='';ids,y,banks,folds,ds=load(a);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if banks['VX'].shape[-1]!=12:raise ValueError('Final vocal inventory changed')
    np.testing.assert_array_equal(banks['VX'][...,:8],banks['V']);plans=design(a.suite)
    definition=dict(arguments=vars(a),plans=plans,code_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('fixed_score_fusion.py').read_bytes()).hexdigest())
    old=out/'run_definition.json'
    if old.exists() and json.loads(old.read_text())!=json.loads(json.dumps(definition)):raise ValueError('Changed definition; use a new output directory')
    old.write_text(json.dumps(definition,indent=2));rows=[];choices=[]
    for k,f in enumerate(folds):
        cp=out/f'outer{k}_completed.json'
        if cp.exists():
            d=json.loads(cp.read_text());rows+=d['rows'];choices+=d['choices'];continue
        tr=np.asarray(f['train_idx']);ev=np.asarray(f['val_idx']);dev=None if not f.get('dev_idx') else np.asarray(f['dev_idx']);shared={};local=[];ch=[]
        assert not set(tr)&set(ev)
        for p in plans:
            yy=y.copy()
            if 'label_permutation' in p:yy[tr]=np.random.default_rng(120000+100*k+p['label_permutation']).permutation(yy[tr])
            e=engine_for(p,ids,yy,banks,ds,shared)
            if p['kind']=='balanced':sin,sout,choice,_=predict(e,p,tr,ev,k,dev)
            else:sin,sout,choice,_=choose_and_predict(e,p,tr,ev,k,dev)
            cy=yy[tr] if dev is None else y[dev];ops=['train_oof_f1'] if ds=='eatd' else ['train_prevalence','train_oof_f1']
            if dev is not None:ops=['development_f1','development_prevalence']
            for op in ops:
                th=best_f1_threshold(cy,sin) if op.endswith('f1') else prior_threshold(sin,float(cy.mean()))
                local.extend(dict(subject_id=str(ids[i]),label=int(y[i]),model=p['name'],repeat=f['repeat'],fold=f['fold'],operating_point=op,score=float(sout[j]),pred=int(sout[j]>=th)) for j,i in enumerate(ev))
                ch.append(dict(model=p['name'],repeat=f['repeat'],fold=f['fold'],operating_point=op,threshold=float(th),**choice))
        cp.write_text(json.dumps(dict(rows=local,choices=ch)));rows+=local;choices+=ch
        print(f'{a.dataset} {a.suite} outer {k+1}/{len(folds)} complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False);pd.DataFrame(choices).to_csv(out/'training_only_choices.csv',index=False)
    rep=pd.DataFrame([dict(model=m,repeat=r,operating_point=op,**metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())) for (m,r,op),g in df.groupby(['model','repeat','operating_point'])])
    rep.to_csv(out/'repeat_metrics.csv',index=False);summary=pd.concat([summarize_repeats(g).assign(operating_point=op) for op,g in rep.groupby('operating_point')]);summary.to_csv(out/'summary.csv',index=False)
    effects=[]
    for op,g in df.groupby('operating_point'):
        w=g.groupby(['subject_id','label','model']).score.mean().unstack('model');yy=w.index.get_level_values('label').to_numpy(int)
        for ref in w.columns:
            if ref=='SLEF-D-12x12':continue
            effects.append(dict(candidate='SLEF-D-12x12',reference=ref,operating_point=op,**bootstrap_auc_delta(yy,w['SLEF-D-12x12'].to_numpy(),w[ref].to_numpy(),10000,42)))
    eff=pd.DataFrame(effects)
    for op,g in eff.groupby('operating_point'):eff.loc[g.index,'p_holm']=holm_adjust(g.p_boot)
    eff.to_csv(out/'paired_effects.csv',index=False);(out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a),scope='Final fixed-feature/fusion validation. Evidence selectors and all branch scalers fit current training only. Removal refits remaining inputs; fixed balance renormalizes active branches. Test labels cannot select method or threshold.',feature_names=VOCAL12_NAMES,source_exploration='Previously examined source CV informed this compact candidate; additional target results never choose it.'),indent=2))
    (out/'COMPLETE').write_text('fixed method validation complete\n');print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
