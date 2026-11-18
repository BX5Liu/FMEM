from __future__ import annotations
import argparse,copy,json,hashlib,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from vendor.pttsd_author.pttsd import ProbabilisticLSTM
from vendor.pttsd_author.distributions import gaussian_nll
from vendor.pttsd_author.pttsd_config import ImprovedBayesianLSTMConfig
from ablation import metric_row,summarize_repeats,best_f1_threshold,prior_threshold

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def make_model():
    cfg=ImprovedBayesianLSTMConfig()
    return ProbabilisticLSTM(embed_dim=384,hidden_dim=cfg.hidden_dim,num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,num_outputs=1,dropout=cfg.dropout,
        fixed_uncertainty_budget=cfg.fixed_uncertainty_budget,distribution_type='gaussian',
        uncertainty_conditioning=cfg.uncertainty_conditioning,seq2seq=True)

def batches(X,indices,device):
    for start in range(0,len(indices),10):
        ix=np.asarray(indices[start:start+10]);lengths=np.asarray([len(X[i]) for i in ix]);longest=int(lengths.max())
        x=np.zeros((len(ix),longest,384),np.float32);mask=np.zeros((len(ix),longest),bool)
        for j,i in enumerate(ix):x[j,:lengths[j]]=X[i];mask[j,:lengths[j]]=True
        yield ix,torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device),lengths

@torch.no_grad()
def predict(model,X,indices,device):
    model.eval();result=[]
    for ix,x,mask,lengths in batches(X,indices,device):
        mean,std=model(x,attention_mask=mask)
        last=torch.as_tensor(lengths-1,device=device)
        value=mean[torch.arange(len(ix),device=device),last,0]
        result.extend(torch.expm1(value).cpu().numpy().tolist())
    result=np.asarray(result)
    if not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
    return result

def fit(X,phq,tr,va,seed,device,epochs=50,select=True):
    cfg=ImprovedBayesianLSTMConfig();seed_all(seed);model=make_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=cfg.min_lr)
    rng=np.random.default_rng(seed);best=np.inf;state=None;best_epoch=epochs;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();trainloss=[]
        for ix,x,mask,lengths in batches(X,rng.permutation(tr),device):
            yy=torch.as_tensor(np.log1p(phq[ix]),device=device,dtype=torch.float32)[:,None].expand_as(mask)
            opt.zero_grad(set_to_none=True);mean,std=model(x,attention_mask=mask)
            loss=gaussian_nll(mean.squeeze(-1),std.squeeze(-1),yy,mask,alpha=cfg.nll_alpha,beta=cfg.nll_beta,gamma=cfg.nll_gamma)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();trainloss.append(float(loss.detach()))
        validloss=[]
        if select:
            model.eval()
            with torch.no_grad():
                for ix,x,mask,lengths in batches(X,va,device):
                    yy=torch.as_tensor(np.log1p(phq[ix]),device=device,dtype=torch.float32)[:,None].expand_as(mask)
                    mean,std=model(x,attention_mask=mask)
                    validloss.append(float(gaussian_nll(mean.squeeze(-1),std.squeeze(-1),yy,mask,alpha=cfg.nll_alpha,beta=cfg.nll_beta,gamma=cfg.nll_gamma)))
            value=float(np.mean(validloss))
            if value<best:best=value;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
        scheduler.step();history.append(dict(epoch=epoch,train_nll=float(np.mean(trainloss)),validation_nll=float(np.mean(validloss)) if select else None))
        if epoch%10==0 or epoch==epochs:print(f'PTTSD seed={seed} epoch={epoch}/{epochs} training NLL={np.mean(trainloss):.4f}',flush=True)
        if select and stale>=15:break
    if select:model.load_state_dict(state)
    return model,best_epoch,history

def extract(a,m,out):
    from transformers import AutoModel,AutoTokenizer
    modelpath=Path(a.model_path_file).read_text().strip();tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    encoder=AutoModel.from_pretrained(modelpath,local_files_only=True).to('cuda').eval()
    maxlen=json.loads((Path(modelpath)/'sentence_bert_config.json').read_text())['max_seq_length']
    if int(maxlen)!=256:raise ValueError('Native MiniLM max sequence length changed')
    dest=out/'native_features';dest.mkdir(exist_ok=True);features=[];audit=[]
    for j,row in enumerate(m.itertuples(index=False)):
        path=dest/(str(row.subject_id)+'.npy')
        d=pd.read_csv(row.text_path);d.columns=d.columns.str.strip()
        if 'Confidence' in d:d=d[pd.to_numeric(d.Confidence,errors='coerce')>=.9]
        text=[' '.join(s.split()).strip() for s in d.Text if isinstance(s,str)]
        text=[s for s in text if len(s)>=3]
        if path.exists():z=np.load(path,allow_pickle=False)
        else:
            chunks=[]
            for start in range(0,len(text),64):
                inp=tokenizer(text[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
                with torch.no_grad():
                    h=encoder(**inp).last_hidden_state;mask=inp['attention_mask'].unsqueeze(-1).float()
                    z=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1e-9),p=2,dim=1)
                chunks.append(z.cpu().numpy())
            z=np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((1,384),np.float32)
            np.save(path,z,allow_pickle=False)
        if z.ndim!=2 or z.shape[-1]!=384 or not np.isfinite(z).all():raise ValueError('Invalid native feature')
        features.append(z);audit.append(dict(subject_id=row.subject_id,utterances=len(text),empty_sentinel=not bool(text)))
        if (j+1)%25==0:print(f'Native MiniLM {j+1}/{len(m)}',flush=True)
    objects=np.empty(len(features),object)
    for i,z in enumerate(features):objects[i]=z
    np.savez_compressed(out/'minilm_original_asr.npz',subject_ids=m.subject_id.to_numpy(str),features=objects)
    pd.DataFrame(audit).to_csv(out/'native_transcript_audit.csv',index=False)
    metadata=dict(model='sentence-transformers/all-MiniLM-L6-v2',download='ModelScope',path=modelpath,max_length=maxlen,pooling='attention-mask token mean then L2',transcript='Original E-DAIC ASR, Confidence >= .9, text >= 3 characters; no invented speaker labels',model_config_sha256=hashlib.sha256((Path(modelpath)/'config.json').read_bytes()).hexdigest())
    (out/'encoder_provenance.json').write_text(json.dumps(metadata,indent=2));(out/'EXTRACTION_COMPLETE').write_text('ok\n')

def verify():
    seed_all(42);torch.set_num_threads(2);model=make_model()
    x=torch.randn(2,7,384);mask=torch.tensor([[True]*7,[True]*4+[False]*3])
    mean,std=model(x,attention_mask=mask);loss=gaussian_nll(mean.squeeze(-1),std.squeeze(-1),torch.ones(2,7),mask,gamma=.5)
    loss.backward()
    assert mean.shape==std.shape==(2,7,1) and torch.isfinite(loss) and (std>0).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('Unmodified author Gaussian seq2seq model: masked forward/backward PASS',flush=True)

def audit_labels(m,out):


    for path,g in m.groupby('label_source'):
        raw=pd.read_csv(path);raw['source_key']=raw.Participant_ID.astype(int).astype(str)
        if raw.source_key.duplicated().any():raise ValueError('Duplicate released label IDs')
        raw=raw.set_index('source_key').loc[g.source_id.astype(int).astype(str)]
        np.testing.assert_array_equal(raw.PHQ_Binary.to_numpy(int),g.label.to_numpy(int))
        np.testing.assert_array_equal(raw.PHQ_Score.to_numpy(float),g.phq_score.to_numpy(float))
    phq=m.phq_score.to_numpy(float)
    if not np.isfinite(phq).all() or (phq<0).any() or (phq>24).any():raise ValueError('Invalid released PHQ totals')
    audit=m[['subject_id','official_split','label','phq_score']].copy()
    audit['total_ge10']=(phq>=10).astype(int);audit['endpoint_disagrees']=audit.label!=audit.total_ge10
    audit.to_csv(out/'released_label_endpoint_audit.csv',index=False)
    note=dict(disagreement_count=int(audit.endpoint_disagrees.sum()),disagreement_by_split=audit.groupby('official_split').endpoint_disagrees.sum().astype(int).to_dict(),
        classification_endpoint='Released PHQ_Binary, kept unchanged for comparability',regression_endpoint='Released PHQ_Score, kept unchanged',
        cutoff_note='Total >=10 and released PHQ_Binary are different endpoints in the present released files. Score-cutoff classification is a secondary operating point evaluated against the released binary endpoint, not proof of diagnosis.')
    (out/'label_endpoint_note.json').write_text(json.dumps(note,indent=2));print(json.dumps(note),flush=True)
    return phq

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['verify','extract','train'],required=True)
    ap.add_argument('--protocol',choices=['official','common219'],default='common219')
    ap.add_argument('--manifest',default='');ap.add_argument('--model_path_file',default='minilm_model_path.txt')
    ap.add_argument('--feature_dir',default='');ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='verify':verify();return
    if not torch.cuda.is_available():raise RuntimeError('Run native encoding/training through GPU sbatch')
    torch.set_num_threads(6);m=pd.read_csv(a.manifest,dtype={'subject_id':str})
    if m.subject_id.nunique()!=275 or len(m)!=275:raise ValueError('Manifest must contain 275 exact subject IDs')
    if a.stage=='extract':extract(a,m,out);return
    cache=np.load(Path(a.feature_dir)/'minilm_original_asr.npz',allow_pickle=True)
    np.testing.assert_array_equal(cache['subject_ids'],m.subject_id.to_numpy(str));X=cache['features'];y=m.label.to_numpy(int);phq=audit_labels(m,out)
    train=np.flatnonzero(m.official_split.to_numpy()=='train');dev=np.flatnonzero(m.official_split.to_numpy()=='dev');test=np.flatnonzero(m.official_split.to_numpy()=='test')
    if (len(train),len(dev),len(test))!=(163,56,56):raise ValueError('Official split changed')
    rows=[];histories=[];epochs=[]
    for r,seed in enumerate([42,1337,2025]):
        if a.protocol=='official':
            model,ep,h=fit(X,phq,train,dev,seed,'cuda');cal=predict(model,X,dev,'cuda');caly=y[dev]
        else:
            tr=np.r_[train,dev];cal=np.full(len(tr),np.nan);caly=y[tr];selected=[]
            for j,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(tr)),y[tr])):
                inner,ep,h=fit(X,phq,tr[it],tr[iv],seed+97*j,'cuda');selected.append(ep);cal[iv]=predict(inner,X,tr[iv],'cuda')
                histories.extend(dict(seed=seed,stage='inner',inner_fold=j,**v) for v in h)
            ep=int(np.median(selected));model,_,h=fit(X,phq,tr,test,seed+909,'cuda',epochs=ep,select=False)
        histories.extend(dict(seed=seed,stage='final',inner_fold=-1,**v) for v in h)
        if not np.isfinite(cal).all():raise ValueError('Incomplete calibration')
        score=predict(model,X,test,'cuda');epochs.append(dict(seed=seed,selected_epoch=ep))
        torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=ep,protocol=a.protocol),out/f'checkpoint_seed{seed}.pt')
        for op,th in [('score_cutoff10_secondary',10.),('development_f1' if a.protocol=='official' else 'train_oof_f1',best_f1_threshold(caly,cal)),('train_prevalence',prior_threshold(cal,float(caly.mean())))]:
            rows.extend(dict(subject_id=str(m.subject_id.iloc[idx]),label=int(y[idx]),phq_score=float(phq[idx]),model='PTTSD-native-MiniLM-originalASR-'+a.protocol,repeat=r,seed=seed,operating_point=op,score=float(score[i]),pred=int(score[i]>=th),threshold=float(th)) for i,idx in enumerate(test))
        pd.DataFrame(rows).to_csv(out/'subject_predictions.partial.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False);metrics=[]
    for (model,r,op),g in df.groupby(['model','repeat','operating_point']):
        error=g.score-g.phq_score
        metrics.append(dict(model=model,repeat=r,operating_point=op,mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),**metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())))
    rep=pd.DataFrame(metrics);rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(d).assign(operating_point=op) for op,d in rep.groupby('operating_point')]);summary.to_csv(out/'summary.csv',index=False)
    rep.groupby('operating_point')[['mae','rmse']].agg(['mean','std']).to_csv(out/'regression_summary.csv')
    pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False);pd.DataFrame(epochs).to_csv(out/'training_selected_epochs.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a), seeds=[42,1337,2025], text_encoder='MiniLM', transcript='released ASR', checkpoint_metric='validation NLL'),indent=2));(out/'COMPLETE').write_text('ok\n');print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
