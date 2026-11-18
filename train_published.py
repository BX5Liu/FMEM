from __future__ import annotations
import argparse,copy,json,os,random,itertools,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from ablation import (load_dataset,load_fold_manifest,check_folds,splits,safe_auc,
    best_f1_threshold,prior_threshold,metric_row,summarize_repeats)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_inputs(a):
    if a.dataset in ('eatd','eatd_original'):
        d=load_dataset(a.audio_npz,a.text_npz,a.data_root,threshold=53,strict_expected=True)
        ids,y=d['ids'],d['y'];folds=load_fold_manifest(a.fold_json,ids)['folds'];check_folds(folds,len(y))
        if a.dataset=='eatd_original':
            ids=np.asarray(ids).astype(str)
            tr=np.flatnonzero(np.char.startswith(ids,'t_'));va=np.flatnonzero(np.char.startswith(ids,'v_'))
            if (len(tr),len(va),int(y[tr].sum()),int(y[va].sum()))!=(83,79,19,11):
                raise ValueError('Original EATD cohort split/labels changed')
            folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
        if a.features=='native':
            z=np.load(a.author_npz,allow_pickle=False)
            ix={str(s):i for i,s in enumerate(z['subject_ids'])}
            order=np.asarray([ix[str(s)] for s in ids]);np.testing.assert_array_equal(z['labels'][order],y)
            X={k:z[k][order].astype(np.float32) for k in ('text','audio','question','gender','emotion')}
        else:
            X={'text':d['text'].astype(np.float32),'audio':d['audio_sem'].astype(np.float32)}
    else:
        if a.features=='native':raise ValueError('No native author E-DAIC feature release')
        from edaic_ceef7_common import load_manifest,align_features
        ids,y,part,X=align_features(load_manifest(a.manifest),{'text':a.text_npz,'audio':a.audio_npz})
        tr=np.flatnonzero(part=='train');va=np.flatnonzero(part=='test')
        if (len(tr),len(va))!=(219,56):raise ValueError('Official split changed')
        folds=[dict(repeat=0,fold=0,train_idx=tr.tolist(),val_idx=va.tolist())]
    if a.features=='qwen':
        for k in ('text','audio'):
            x=np.asarray(X[k],np.float32);X[k]=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
            if X[k].shape[1]!=3:raise ValueError('Expected original three contexts')
        if a.model=='sega':
            X['text']=X['text'][:,:,None,:];X['audio']=X['audio'][:,:,None,:]
            X.update(question=np.zeros_like(X['text']),gender=np.zeros(len(y),np.float32),emotion=np.zeros((len(y),3,7),np.float32))
    if any(not np.isfinite(v).all() for v in X.values()):raise ValueError('Nonfinite features')
    return np.asarray(ids),np.asarray(y,int),X,folds

def make_model(a,X):
    if a.model=='sega':


        for k,v in dict(PROXY='1',DIRECTED='1',READOUT='vsum_a',LGAT='2',AUG='none',
            AUDIO_BN='1',DROP_EMO='0' if a.features=='native' else '1').items():os.environ[k]=v
        from vendor.sega_graph import GraphFusion
        config=dict(text_embed_size=X['text'].shape[-1],audio_embed_size=X['audio'].shape[-1],
            text_hidden_dims=256,audio_hidden_dims=256,rnn_layers=1,dropout=.3,num_classes=2,bidirectional=True)
        return GraphFusion(config)
    from vendor.CafMambaBimodal import CafMambaBimodal
    return CafMambaBimodal(audio_input_size=X['audio'].shape[-1],video_input_size=X['text'].shape[-1],
        mm_input_size=256,mm_output_sizes=[256],dropout=.1,mamba_config=dict(d_state=16,expand=4,d_conv=4))

def preprocess(a,X,tr,indices):


    if a.features=='native':return {k:v[indices].copy() for k,v in X.items()}
    out={k:v[indices].copy() for k,v in X.items()}
    for k in ('text','audio'):
        scaler=StandardScaler().fit(X[k][tr].reshape(-1,X[k].shape[-1]))
        out[k]=scaler.transform(X[k][indices].reshape(-1,X[k].shape[-1])).reshape(out[k].shape).astype(np.float32)
    return out

def augmentation(X,y,enabled):
    if not enabled:return X,y

    row=[];perm=[]
    for i,label in enumerate(y):
        ps=list(itertools.permutations(range(3))) if label else [(0,1,2)]
        for p in ps:row.append(i);perm.append(p)
    row=np.asarray(row);perm=np.asarray(perm)
    out={}
    for k,v in X.items():
        out[k]=v[row] if k=='gender' else v[row[:,None],perm]
    return out,y[row]

def logits(model,a,inputs):
    if a.model=='sega':
        return model(*(inputs[k] for k in ('text','audio','question','gender','emotion')),process='test')[0]
    return model(torch.cat([inputs['text'],inputs['audio']],dim=-1)).reshape(-1)

@torch.no_grad()
def predict(model,a,X,device):
    model.eval();result=[]
    for start in range(0,len(X['text']),16):
        inp={k:torch.as_tensor(v[start:start+16],device=device) for k,v in X.items()}
        p=logits(model,a,inp)
        result.append((p[:,1]-p[:,0] if a.model=='sega' else p).detach().cpu().numpy())
    return np.concatenate(result)

def fit(a,X,y,tr,va,seed,device,epochs,select_epoch):
    seed_all(seed);model=make_model(a,X).to(device)
    train=preprocess(a,X,tr,tr);valid=preprocess(a,X,tr,va)
    train,yt=augmentation(train,y[tr],a.model=='sega' and a.dataset in ('eatd','eatd_original'))
    native_train=preprocess(a,X,tr,tr)
    opt=torch.optim.Adam(model.parameters(),lr=2e-5 if a.model=='sega' else 1e-4,
        weight_decay=1e-2 if a.model=='sega' else 0.)
    lossfn=nn.CrossEntropyLoss() if a.model=='sega' else nn.BCEWithLogitsLoss()
    best_auc=-np.inf;best_state=None;best_epoch=epochs
    bs=8 if a.model=='sega' else 16
    rng=np.random.default_rng(seed)
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(yt));total=0.
        for start in range(0,len(order),bs):
            ix=order[start:start+bs];inp={k:torch.as_tensor(v[ix],device=device) for k,v in train.items()}
            target=torch.as_tensor(yt[ix],device=device,dtype=torch.long if a.model=='sega' else torch.float32)
            opt.zero_grad(set_to_none=True);pred=logits(model,a,inp);loss=lossfn(pred,target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(ix)
        if select_epoch:
            auc=safe_auc(y[va],predict(model,a,valid,device))
            if auc>best_auc+1e-12:
                best_auc=auc;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        if epoch%20==0 or epoch==epochs:
            print(f'fit seed={seed} ep={epoch}/{epochs} loss={total/len(yt):.4f} inner_auc={best_auc if select_epoch else "not_used"}',flush=True)
    if select_epoch:model.load_state_dict(best_state)
    ref=predict(model,a,native_train,device);score=predict(model,a,valid,device)
    score=(score-ref.mean())/max(float(ref.std()),1e-8)
    return score,best_epoch,model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['sega','caf'],required=True)
    ap.add_argument('--features',choices=['native','qwen'],required=True)
    ap.add_argument('--dataset',choices=['eatd','edaic','eatd_original'],required=True)
    for k in ('audio_npz','text_npz','outdir'):ap.add_argument('--'+k,required=True)
    for k in ('data_root','fold_json','manifest','author_npz'):ap.add_argument('--'+k,default='')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--max_epochs',type=int,default=0)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.model=='caf' and a.features=='native':raise ValueError('CAF is an explicitly documented AT transfer')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','6')))
    if not torch.cuda.is_available():raise RuntimeError('Run training through GPU sbatch')
    device=torch.device('cuda');out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    ids,y,X,folds=load_inputs(a)
    name=('SEGA-author-native' if a.features=='native' else 'SEGA-Qwen-TA-noaux-noquestion' if a.model=='sega' else 'CAF-Mamba-Qwen-AT-transfer')
    epochs=a.max_epochs or (100 if a.model=='sega' else 80)
    if a.smoke:folds=folds[:1];epochs=1
    rows=[];diagnostics=[]
    for k,f in enumerate(folds):
        foldout=out/f'fold_{k:02d}';foldout.mkdir(exist_ok=True)
        saved=foldout/'predictions.csv'
        if saved.exists() and (foldout/'COMPLETE').exists():
            rows+=pd.read_csv(saved,dtype={'subject_id':str}).to_dict('records')
            diagnostics+=json.loads((foldout/'diagnostics.json').read_text());continue
        tr=np.asarray(f['train_idx']);va=np.asarray(f['val_idx'])
        inner_repeats=[];chosen=[]
        for ir in range(5 if a.dataset=='edaic' and not a.smoke else 1):
            inner=np.full(len(tr),np.nan)
            for j,(itr,iva) in enumerate(splits(ids[tr],y[tr],a.seed+1000*k+1009*ir,'eatd' if a.dataset=='eatd_original' else a.dataset)):
                score,ep,_=fit(a,X,y,tr[itr],tr[iva],a.seed+1000*k+97*j+97*ir,device,epochs,True)
                inner[iva]=score;chosen.append(ep)
                diagnostics.append(dict(repeat=f['repeat'],fold=f['fold'],inner_repeat=ir,inner_fold=j,best_epoch=ep,inner_auc=safe_auc(y[tr][iva],score)))
            inner_repeats.append(inner)
        inner=np.mean(inner_repeats,axis=0)
        if not np.isfinite(inner).all():raise RuntimeError('Inner OOF coverage failure')
        fixed_ep=int(np.median(chosen))
        score,_,model=fit(a,X,y,tr,va,a.seed+2000*k+909,device,fixed_ep,False)
        torch.save(dict(state_dict=model.state_dict(),model=name,seed=a.seed+2000*k+909,
            epochs=fixed_ep,train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist(),arguments=vars(a)),foldout/'checkpoint.pt')
        current=[]
        for op in (('train_oof_f1',) if a.dataset in ('eatd','eatd_original') else ('train_prevalence','train_oof_f1')):
            th=best_f1_threshold(y[tr],inner) if op=='train_oof_f1' else prior_threshold(inner,float(y[tr].mean()))
            for i,idx in enumerate(va):
                current.append(dict(subject_id=str(ids[idx]),label=int(y[idx]),repeat=f['repeat'],fold=f['fold'],
                    model=name,operating_point=op,score=float(score[i]),pred=int(score[i]>=th),threshold=th,training_selected_epoch=fixed_ep))
        pd.DataFrame(current).to_csv(saved,index=False);rows+=current
        (foldout/'diagnostics.json').write_text(json.dumps([d for d in diagnostics if d['repeat']==f['repeat'] and d['fold']==f['fold']],indent=2))
        (foldout/'COMPLETE').write_text('ok\n')
        pd.DataFrame(rows).to_csv(out/'subject_predictions.partial.csv',index=False)
        print(f'{name} outer fold {k+1}/{len(folds)} COMPLETE; training-selected epochs={fixed_ep}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False)
    mr=[]
    for (r,op),d in df.groupby(['repeat','operating_point']):
        if not a.smoke and a.dataset=='eatd' and (len(d)!=162 or d.subject_id.nunique()!=162):raise ValueError('Subject coverage failure')
        mr.append(dict(model=name,repeat=r,operating_point=op,**metric_row(d.label.to_numpy(),d.score.to_numpy(),pred=d.pred.to_numpy())))
    md=pd.DataFrame(mr);md.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(d).assign(operating_point=op) for op,d in md.groupby('operating_point')])
    summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'training_only_epoch_selection.csv',index=False)
    metadata=dict(arguments=vars(a),model=name,feature_shapes={k:list(v.shape) for k,v in X.items()},
        scope='Author-source native-feature reproduction with frozen common evaluation' if a.features=='native' else 'Author fusion architecture transferred to shared frozen Qwen audio/text inputs',
        deviations=['Training-only checkpoint and threshold selection','No label-dependent augmentation at evaluation',
            'Train subject context permutations apply consistently to all context fields',
            'CAF uses official PyTorch Mamba reference scan instead of compiled CUDA scan; AV to AT input transfer',
            'E-DAIC uses the existing temporal three-context cache, not 85 question categories'],
        source_manifest=json.loads(Path('models_provenance.json').read_text()))
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2));(out/'COMPLETE').write_text('ok\n')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
