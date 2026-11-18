from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from ablation import (strict_profile,load_dataset,load_fold_manifest,check_folds,
    make_inner_splits,fit_score,BranchSpec,safe_auc,metric_row,best_f1_threshold,
    prior_threshold,summarize_repeats,bootstrap_auc_delta,holm_adjust)
from round2_evaluate import score_branch,pooled

SEEDS=[42,1337,2025]

def inner_splits(ids,y,seed,dataset):
    if dataset.startswith('eatd'):
        return list(make_inner_splits(ids,y,3,seed))
    return list(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))

def load_inputs(a):
    if a.dataset.startswith('eatd'):
        d=load_dataset(a.audio_npz,a.text_npz,a.data_root,threshold=53,strict_expected=True)
        ids=np.asarray(d['ids']);y=d['y'];T=d['text']
        folds=load_fold_manifest(a.fold_json,ids)['folds'];check_folds(folds,len(y))
        if a.dataset=='eatd_original':
            tr=np.flatnonzero(np.char.startswith(ids.astype(str),'t_'));va=np.flatnonzero(np.char.startswith(ids.astype(str),'v_'))
            if (len(tr),len(va),int(y[tr].sum()),int(y[va].sum()))!=(83,79,19,11):raise ValueError('EATD split changed')
            folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    else:
        from edaic_ceef7_common import load_manifest,align_features
        m=load_manifest(a.manifest);ids,y,part,x=align_features(m,{'text':a.text_npz,'audio':a.audio_npz});T=x['text']
        tr=np.flatnonzero(part=='train');va=np.flatnonzero(part=='test')
        if (len(tr),len(va))!=(219,56):raise ValueError('E-DAIC split changed')
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    return ids,np.asarray(y,int),[T,strict_profile(a.llm_npz,ids),strict_profile(a.vocal_npz,ids)],folds

def candidates():
    c=[dict(family='Fixed-SLEFD',mode='fixed')]
    for C in [.01,.1,1.]:
        for mode in ['stack_linear','stack_interaction']:
            c.append(dict(family=mode,mode=mode,C=C))
    for wd in [.01,.1]:
        for epochs in [40,100]:c.append(dict(family='bounded_residual_gate',mode='gate',wd=wd,epochs=epochs))
    for C in [.01,.1,1.]:c.append(dict(family='projected_feature_LR',mode='feature_lr',C=C))
    for mode in ['projected_mlp','modality_attention','context_attention']:
        for hidden in [16,32]:
            for epochs in [30,80]:c.append(dict(family=mode,mode=mode,hidden=hidden,epochs=epochs,wd=.1))
    return c

def interaction(z):
    return np.column_stack([z,z[:,0]*z[:,1],z[:,0]*z[:,2],z[:,1]*z[:,2],
                            np.abs(z[:,0]-z[:,1]),np.abs(z[:,0]-z[:,2])])

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Linear(3,1);self.v=nn.Linear(3,1)
        nn.init.zeros_(self.a.weight);nn.init.constant_(self.a.bias,np.log(.6/.4))
        nn.init.zeros_(self.v.weight);nn.init.zeros_(self.v.bias)
    def forward(self,z):
        alpha=.5+.5*torch.sigmoid(self.a(z)).squeeze(-1)
        lam=.2*torch.sigmoid(self.v(z)).squeeze(-1)
        return alpha*z[:,0]+(1-alpha)*z[:,1]+lam*torch.tanh(z[:,2])

class FeatureHead(nn.Module):
    def __init__(self,dims,h,mode):
        super().__init__();self.mode=mode
        self.proj=nn.ModuleList([nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.4)) for d in dims])
        if mode in ['modality_attention','context_attention']:
            self.attn=nn.MultiheadAttention(h,4,dropout=.3,batch_first=True)
            self.query=nn.Parameter(torch.zeros(1,1,h));nn.init.normal_(self.query,std=.02)
        self.out=nn.Linear(h*3 if mode=='projected_mlp' else h,1)
    def forward(self,xs):
        if self.mode=='context_attention':

            tokens=torch.cat([p(x) for p,x in zip(self.proj,xs)],dim=1)
        else:
            tokens=torch.stack([p(x).mean(1) for p,x in zip(self.proj,xs)],dim=1)
        if self.mode=='projected_mlp':z=tokens.flatten(1)
        else:
            q=self.query.expand(tokens.shape[0],-1,-1);z=self.attn(q,tokens,tokens,need_weights=False)[0][:,0]
        return self.out(z).squeeze(-1)

class Engine:
    def __init__(self,ids,y,X,dataset):
        self.ids,self.y,self.X,self.dataset=ids,y,X,dataset;self.base_cache={};self.feature_cache={}
    def branches(self,tr,ev,seed):

        key=(tuple(tr),tuple(ev),seed)
        if key in self.base_cache:return self.base_cache[key]
        ztrain=np.zeros((len(tr),3));ztest=np.zeros((len(ev),3))
        for b,x in enumerate(self.X):
            C=.03 if b==0 else .1
            for j,(it,iv) in enumerate(inner_splits(self.ids[tr],self.y[tr],seed+100*b,self.dataset)):
                ztrain[iv,b]=fit_score(x[tr[it]],self.y[tr[it]],x[tr[iv]],BranchSpec('mean',0,C),seed+100*b+97*j)[0]
            ztest[:,b]=fit_score(x[tr],self.y[tr],x[ev],BranchSpec('mean',0,C),seed+100*b)[0]
        if not np.isfinite(ztrain).all() or not np.isfinite(ztest).all():raise ValueError('Cross-fit coverage')
        self.base_cache[key]=(ztrain,ztest);return ztrain,ztest
    def features(self,tr,ev,context=False):
        key=(tuple(tr),tuple(ev),context)
        if key in self.feature_cache:return self.feature_cache[key]
        arrays=[];tests=[]
        for b,x in enumerate(self.X):
            z=np.asarray(x,np.float32)
            z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8)
            if not context:z=pooled(x).astype(np.float32)
            dims=z.shape[-1]

            if b==0:
                p=PCA(n_components=min(32,len(tr)*z.shape[1]-1,dims),svd_solver='full')
                p.fit(z[tr].reshape(-1,dims));ztr=p.transform(z[tr].reshape(-1,dims)).reshape(len(tr),z.shape[1],-1)
                zev=p.transform(z[ev].reshape(-1,dims)).reshape(len(ev),z.shape[1],-1)
            else:ztr,zev=z[tr],z[ev]
            ss=StandardScaler().fit(ztr.reshape(-1,ztr.shape[-1]))
            arrays.append(ss.transform(ztr.reshape(-1,ztr.shape[-1])).reshape(ztr.shape).astype(np.float32))
            tests.append(ss.transform(zev.reshape(-1,zev.shape[-1])).reshape(zev.shape).astype(np.float32))
        self.feature_cache[key]=(arrays,tests);return arrays,tests
    def predict(self,c,tr,ev,seed):
        if set(tr)&set(ev):raise ValueError('Train/eval subject overlap')
        mode=c['mode'];yt=self.y[tr]
        if mode in ['stack_linear','stack_interaction','gate']:
            zin,zout=self.branches(tr,ev,seed)
            if mode.startswith('stack'):
                a,b=(interaction(zin),interaction(zout)) if mode=='stack_interaction' else (zin,zout)
                ss=StandardScaler().fit(a);model=LogisticRegression(C=c['C'],solver='liblinear',class_weight='balanced',random_state=seed,max_iter=3000).fit(ss.transform(a),yt)
                return model.decision_function(ss.transform(b)),{}
            xx=[zin[:,None,:]];xe=[zout[:,None,:]]
        else:
            xx,xe=self.features(tr,ev,mode=='context_attention')
            if mode=='feature_lr':
                a=np.concatenate([v.mean(1) for v in xx],1);b=np.concatenate([v.mean(1) for v in xe],1)
                model=LogisticRegression(C=c['C'],solver='liblinear',class_weight='balanced',random_state=seed,max_iter=3000).fit(a,yt)
                return model.decision_function(b),{}
        pred=[];diagnostics=[]
        device='cuda' if torch.cuda.is_available() else 'cpu'
        for r,s in enumerate(SEEDS):
            torch.manual_seed(seed+s);np.random.seed(seed+s)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+s)
            model=Gate() if mode=='gate' else FeatureHead([v.shape[-1] for v in xx],c['hidden'],mode)
            model=model.to(device);tx=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in xx];te=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in xe]
            ty=torch.tensor(yt,dtype=torch.float32,device=device)
            loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((yt==0).sum()/max(1,yt.sum())),device=device))
            opt=torch.optim.AdamW(model.parameters(),lr=.01 if mode=='gate' else .001,weight_decay=c['wd'])
            for epoch in range(c['epochs']):
                model.train();order=np.random.permutation(len(tr))
                for start in range(0,len(tr),32):
                    ix=order[start:start+32];opt.zero_grad(set_to_none=True)
                    logits=model(tx[0][ix,0]) if mode=='gate' else model([v[ix] for v in tx])
                    obj=loss(logits,ty[ix]);obj.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    if not torch.isfinite(obj):raise ValueError('Nonfinite training loss')
            model.eval()
            with torch.no_grad():
                z=model(te[0][:,0]) if mode=='gate' else model(te);pred.append(z.float().cpu().numpy())
                if mode=='gate':

                    a=.5+.5*torch.sigmoid(model.a(tx[0][:,0]));v=.2*torch.sigmoid(model.v(tx[0][:,0]))
                    diagnostics.append(dict(seed=s,train_alpha_mean=float(a.mean()),train_lambda_mean=float(v.mean())))
            del model,opt
        return np.mean(pred,axis=0),dict(gate_training_diagnostics=diagnostics)

def write_results(rows,choices,foldmetrics,a,out):
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'training_only_choices.csv',index=False)
    pd.DataFrame(foldmetrics).to_csv(out/'fold_metrics.csv',index=False)
    rep=[]
    for (name,r,op),g in df.groupby(['model','repeat','operating_point']):
        rep.append(dict(model=name,repeat=r,operating_point=op,average_precision=average_precision_score(g.label,g.score),**metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())))
    rep=pd.DataFrame(rep);rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(g).assign(operating_point=op) for op,g in rep.groupby('operating_point')]);summary.to_csv(out/'summary.csv',index=False)
    effect=[]
    for op,g in df.groupby('operating_point'):
        wide=g.groupby(['subject_id','label','model']).score.mean().unstack('model');y=wide.index.get_level_values('label').to_numpy(int)
        for name in sorted(set(wide)-{'Fixed-SLEFD'}):
            effect.append(dict(candidate=name,reference='Fixed-SLEFD',operating_point=op,**bootstrap_auc_delta(y,wide[name].to_numpy(),wide['Fixed-SLEFD'].to_numpy(),10000,42)))
    e=pd.DataFrame(effect)
    for op,g in e.groupby('operating_point'):e.loc[g.index,'p_holm_within_suite']=holm_adjust(g.p_boot)
    e.to_csv(out/'paired_effects.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a),candidates=candidates(),seed_ensemble=SEEDS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Post-lock exploratory nested optimization; prior tests already examined',selection='Inner-subject OOF AUC, deterministic candidate-order ties; family and whole-method selection happen inside each outer training fold',
        stacking='For EACH inner training set generate another 3-fold OOF branch matrix; fit the meta head on that matrix, never in-sample branch predictions',
        epochs='Fixed candidates 30/80 for feature heads, 40/100 for gate; no outer-label early stopping',
        interpretation='No guarantee of a global optimum. Context attention is attention over independent branch-context tokens, not invented temporal alignment.'),indent=2))
    (out/'COMPLETE').write_text('ok\n');print(summary.to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['eatd','eatd_original','edaic'],required=True)
    for n in ['text_npz','audio_npz','llm_npz','vocal_npz','outdir']:ap.add_argument('--'+n,required=True)
    for n in ['data_root','fold_json','manifest']:ap.add_argument('--'+n,default='')
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    definition=dict(arguments=vars(a),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidates=candidates(),seeds=SEEDS)
    locked=out/'run_definition.json'
    if locked.exists() and json.loads(locked.read_text())!=definition:raise ValueError('Run definition changed; use a new output directory')
    locked.write_text(json.dumps(definition,sort_keys=True,indent=2))
    torch.set_num_threads(6);ids,y,X,folds=load_inputs(a);engine=Engine(ids,y,X,a.dataset)
    grid=candidates();rows=[];choices=[];foldmetrics=[]
    if a.smoke:
        folds=folds[:1];grid=[grid[0],next(c for c in grid if c['mode']=='stack_linear'),next(c for c in grid if c['mode']=='gate'),next(c for c in grid if c['mode']=='context_attention')]
        grid=[dict(c,epochs=2) if 'epochs' in c else c for c in grid]
    for k,f in enumerate(folds):
        tr=np.asarray(f['train_idx'],int);ev=np.asarray(f['val_idx'],int);seed=42+1000*k
        inner=inner_splits(ids[tr],y[tr],seed,a.dataset);pool=[]

        zz=[];ze=[]
        for b,x in enumerate(X):
            z,v,_=score_branch(x,ids,y,tr,ev,.03 if b==0 else .1,b,k,a.dataset)
            zz.append(z);ze.append(v)
        base_in=.8*zz[0]+.2*zz[1]+.1*np.tanh(zz[2]);base_out=.8*ze[0]+.2*ze[1]+.1*np.tanh(ze[2])
        for ci,c in enumerate(grid):
            if c['mode']=='fixed':oof=base_in;final=base_out;diag={}
            else:
                oof=np.full(len(tr),np.nan)
                for j,(it,iv) in enumerate(inner):oof[iv]=engine.predict(c,tr[it],tr[iv],seed+97*j)[0]
                final,diag=engine.predict(c,tr,ev,seed+909)
            if not np.isfinite(oof).all() or not np.isfinite(final).all():raise ValueError('Prediction coverage')
            pool.append((c,oof,final,safe_auc(y[tr],oof),diag))
            print(f'{a.dataset} outer {k+1}/{len(folds)} candidate {ci+1}/{len(grid)} {c} inner_auc={pool[-1][3]:.5f}',flush=True)
        selected={}
        for family in dict.fromkeys(c['family'] for c in grid):
            ix=[i for i,p in enumerate(pool) if p[0]['family']==family]
            best=max(p[3] for p in [pool[i] for i in ix]);idx=next(i for i in ix if pool[i][3]>=best-1e-12);selected[family]=idx
        best=max(p[3] for p in pool);selected['SLEFD_train_selected']=next(i for i,p in enumerate(pool) if p[3]>=best-1e-12)
        for name,idx in selected.items():
            c,oof,final,auc,diag=pool[idx]
            choices.append(dict(model=name,repeat=f['repeat'],fold=f['fold'],candidate=json.dumps(c),inner_auc=auc,diagnostics=json.dumps(diag)))
            for op in (['train_oof_f1'] if a.dataset.startswith('eatd') else ['train_prevalence','train_oof_f1']):
                th=best_f1_threshold(y[tr],oof) if op=='train_oof_f1' else prior_threshold(oof,float(y[tr].mean()))
                pred=(final>=th).astype(int);foldmetrics.append(dict(model=name,repeat=f['repeat'],fold=f['fold'],operating_point=op,**metric_row(y[ev],final,pred=pred)))
                rows.extend(dict(subject_id=str(ids[v]),label=int(y[v]),model=name,repeat=f['repeat'],fold=f['fold'],operating_point=op,score=float(final[i]),pred=int(pred[i]),threshold=float(th)) for i,v in enumerate(ev))
        pd.DataFrame(rows).to_csv(out/'subject_predictions.partial.csv',index=False)
        pd.DataFrame(choices).to_csv(out/'training_only_choices.partial.csv',index=False)
        engine.base_cache.clear();engine.feature_cache.clear()
    write_results(rows,choices,foldmetrics,a,out)

if __name__=='__main__':main()
