import argparse,json,copy,hashlib,shutil,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from vendor.mtsp_author.multi_scale_transformer import MultiScaleTransformer
from vendor.mtsp_author.multi_scale_config import create_config_for_dataset
from vendor.mtsp_author.task_level_spl import TaskLevelSPL,SPLConfig
from vendor.mtsp_author.cluster_constraint_loss import ClusterConstraintLoss
from native_pttsd import audit_labels,seed_all
from ablation import metric_row,summarize_repeats,best_f1_threshold,prior_threshold

ITEMS=['PHQ_8NoInterest','PHQ_8Depressed','PHQ_8Sleep','PHQ_8Tired',
    'PHQ_8Appetite','PHQ_8Failure','PHQ_8Concentrating','PHQ_8Moving']

def make_model():
    c=create_config_for_dataset('edaic')
    return MultiScaleTransformer(input_dim=768,hidden_dim=c.hidden_dim,num_layers=c.num_layers,num_heads=4,
        dropout=c.dropout,num_subscales=8,subscale_ranges=[3]*8,prediction_mode='regression',use_normalization=c.use_normalization,
        per_subscale_normalization=c.per_subscale_normalization,normalization_min=c.normalization_min,normalization_max=c.normalization_max,
        subscale_min_list=c.subscale_min_list,subscale_max_list=c.subscale_max_list,output_head_type=c.output_head_type,
        use_task_graph=True,scale_type='PHQ-8',task_graph_embed_dim=c.task_graph_embed_dim,task_graph_hidden_dim=c.task_graph_hidden_dim,
        task_graph_num_layers=c.task_graph_num_layers,task_graph_num_heads=c.task_graph_num_heads,task_graph_dropout=c.task_graph_dropout,
        task_graph_omega_intra=c.task_graph_omega_intra,task_graph_omega_cross=c.task_graph_omega_cross,
        task_graph_learnable_weights=c.task_graph_learnable_weights,task_graph_fusion_type=c.task_graph_fusion_type)

def loss_modules(device):
    c=create_config_for_dataset('edaic')
    spl=TaskLevelSPL(SPLConfig(pace_function=c.spl_pace_function,initial_lambda=c.spl_initial_lambda,final_lambda=c.spl_final_lambda,
        lambda_growth=c.spl_lambda_growth,use_regularization=c.spl_use_regularization,min_task_ratio=c.spl_min_task_ratio),8).to(device)
    cluster=ClusterConstraintLoss(scale_type='PHQ-8',cluster1_weight=c.cluster1_weight,cluster2_weight=c.cluster2_weight,cluster3_weight=c.cluster3_weight,
        cluster4_weight=c.cluster4_weight,cluster5_weight=c.cluster5_weight).to(device)
    return spl,cluster

def training_loss(model,x,mask,items,spl,cluster):
    c=create_config_for_dataset('edaic');output=model(x,attention_mask=mask).clamp(0.,1.)
    per=(output-model.normalize_labels(items)).square()
    spl.update_task_difficulties(per.mean(0).detach())
    loss=spl.apply_task_weights(per).sum(1).mean()
    reg=spl.compute_regularization()
    if isinstance(reg,torch.Tensor):loss=loss+reg.to(x.device)
    return loss+c.cluster_constraint_weight*cluster(model.denormalize_predictions(output))

def batches(X,indices,device):
    for start in range(0,len(indices),50):
        ix=np.asarray(indices[start:start+50]);n=np.asarray([len(X[i]) for i in ix]);s=int(n.max())
        x=np.zeros((len(ix),s,768),np.float32);mask=np.zeros((len(ix),s),bool)
        for j,i in enumerate(ix):x[j,:n[j]]=X[i];mask[j,:n[j]]=True
        yield ix,torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device)

@torch.no_grad()
def predict(model,X,ix,device):
    model.eval();rows=[]
    for _,x,mask in batches(X,ix,device):rows.extend(model.denormalize_predictions(model(x,attention_mask=mask).clamp(0.,1.)).cpu().numpy().tolist())
    values=np.asarray(rows);assert np.isfinite(values).all()
    return values.sum(1)

def fit(X,items,tr,va,seed,device,epochs=80,select=True):
    if not np.isfinite(items[tr]).all() or (select and not np.isfinite(items[va]).all()):raise ValueError('Missing authenticated training item labels')
    c=create_config_for_dataset('edaic');seed_all(seed);model=make_model().to(device);spl,cluster=loss_modules(device)
    opt=torch.optim.Adam(model.parameters(),lr=c.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=c.min_lr)
    rng=np.random.default_rng(seed);best=np.inf;state=None;bestepoch=epochs;stale=0;history=[]
    for epoch in range(epochs):
        model.train();spl.update_lambda(epoch,80);losses=[]
        for ix,x,mask in batches(X,rng.permutation(tr),device):
            yy=torch.as_tensor(items[ix],device=device,dtype=torch.float32)
            opt.zero_grad(set_to_none=True);loss=training_loss(model,x,mask,yy,spl,cluster)
            if not torch.isfinite(loss):raise ValueError('Nonfinite full MTSP loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        value=None
        if select:
            value=float(np.abs(predict(model,X,va,device)-items[va].sum(1)).mean())
            if value<best:best=value;state=copy.deepcopy(model.state_dict());bestepoch=epoch+1;stale=0
            else:stale+=1
        scheduler.step();history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_total_mae=value))
        if (epoch+1)%10==0:print(f'MTSP seed={seed} epoch={epoch+1}/{epochs} loss={np.mean(losses):.4f}',flush=True)
        if select and stale>=15:break
    if select:model.load_state_dict(state)
    return model,bestepoch,history

def prepare_encoder(source,dest):

    from safetensors.torch import save_file
    from transformers import AutoModel,AutoTokenizer
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    weight=source/'pytorch_model.bin';digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    info=dest/'checkpoint_provenance.json'
    if info.exists() and json.loads(info.read_text())['source_sha256']!=digest:raise ValueError('MentalBERT source checkpoint changed')
    if not (dest/'model.safetensors').exists():
        state=torch.load(weight,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('Expected tensor-only pretrained state dict')
        save_file({k:v.contiguous().clone() for k,v in state.items()},str(dest/'model.safetensors'),metadata={'format':'pt'})
        for f in source.iterdir():
            if f.name in ['config.json','vocab.txt','special_tokens_map.json','tokenizer_config.json','tokenizer.json']:shutil.copy2(f,dest/f.name)
    info.write_text(json.dumps(dict(source=str(source),source_sha256=digest,conversion='torch weights_only=True, exact cloned tensors, safetensors for Transformers compatibility',identity_note='Existing MentalBERT checkpoint; native-family cache identity retained, upstream public release hash authentication recorded separately when available'),indent=2))
    t=AutoTokenizer.from_pretrained(dest,local_files_only=True);model=AutoModel.from_pretrained(dest,local_files_only=True)
    with torch.no_grad():h=model(**t('Native MentalBERT verification.',return_tensors='pt')).last_hidden_state
    assert h.shape[-1]==768 and torch.isfinite(h).all()
    print('MentalBERT tensor-only conversion and encoder load PASS',flush=True)

def extract(a,m,out):
    from transformers import AutoTokenizer,AutoModel
    t=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True);model=AutoModel.from_pretrained(a.model_path,local_files_only=True).to('cuda').eval()
    cache=out/'mentalbert_subject_features';cache.mkdir(exist_ok=True);features=[];audit=[]
    for j,row in enumerate(m.itertuples(index=False)):
        p=cache/(str(row.subject_id)+'.npy');d=pd.read_csv(row.text_path);d.columns=d.columns.str.strip()
        if 'Confidence' in d:d=d[pd.to_numeric(d.Confidence,errors='coerce')>=.9]
        text=[' '.join(s.split()).strip() for s in d.Text if isinstance(s,str)];text=[s for s in text if len(s)>=3]
        if p.exists():z=np.load(p,allow_pickle=False)
        else:
            blocks=[]
            for start in range(0,len(text),64):
                inp=t(text[start:start+64],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                with torch.no_grad():blocks.append(model(**inp).last_hidden_state[:,0,:].cpu().numpy())
            z=np.concatenate(blocks).astype(np.float32) if blocks else np.zeros((1,768),np.float32);np.save(p,z,allow_pickle=False)
        if not np.isfinite(z).all():raise ValueError('Invalid MentalBERT features')
        features.append(z);audit.append(dict(subject_id=row.subject_id,utterances=len(text),empty_sentinel=not bool(text)))
        if (j+1)%25==0:print(f'MentalBERT {j+1}/{len(m)}',flush=True)
    arr=np.empty(len(features),object)
    for i,z in enumerate(features):arr[i]=z
    np.savez_compressed(out/'mentalbert_original_asr.npz',subject_ids=m.subject_id.to_numpy(str),features=arr)
    pd.DataFrame(audit).to_csv(out/'native_transcript_audit.csv',index=False);(out/'MTSP_EXTRACTION_COMPLETE').write_text('ok\n')

def verify():
    torch.set_num_threads(2);seed_all(42);model=make_model();spl,cluster=loss_modules('cpu');spl.update_lambda(0,80)
    x=torch.randn(3,7,768);mask=torch.tensor([[True]*7,[True]*5+[False]*2,[True]*4+[False]*3]);yy=torch.tensor([[0,1,2,3,0,1,2,3]]*3,dtype=torch.float32)
    loss=training_loss(model,x,mask,yy,spl,cluster);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model(x,mask).shape==(3,8)
    print('Full author MTSP graph/SPL/cluster masked forward/backward PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['verify','prepare','extract','train'],required=True)
    for k in ('manifest','feature_dir','item_labels','model_path','model_source'):ap.add_argument('--'+k,default='')
    ap.add_argument('--protocol',choices=['common219','official'],default='common219');ap.add_argument('--outdir',required=True);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='verify':verify();return
    if a.stage=='prepare':prepare_encoder(a.model_source,a.model_path);return
    if not torch.cuda.is_available():raise ValueError('GPU sbatch required')
    torch.set_num_threads(6);m=pd.read_csv(a.manifest,dtype={'subject_id':str});assert len(m)==m.subject_id.nunique()==275
    if a.stage=='extract':extract(a,m,out);return
    z=np.load(Path(a.feature_dir)/'mentalbert_original_asr.npz',allow_pickle=True);np.testing.assert_array_equal(z['subject_ids'],m.subject_id.to_numpy(str));X=z['features']
    phq=audit_labels(m,out);y=m.label.to_numpy(int)
    original=pd.read_csv(a.item_labels).set_index('Participant_ID');assert original.index.is_unique
    items=np.full((len(m),8),np.nan)
    for i,r in enumerate(m.itertuples(index=False)):
        if r.official_split!='test':items[i]=original.loc[int(r.source_id),ITEMS].to_numpy(float)
    train=np.flatnonzero(m.official_split.to_numpy()=='train');dev=np.flatnonzero(m.official_split.to_numpy()=='dev');test=np.flatnonzero(m.official_split.to_numpy()=='test');tr=np.r_[train,dev]
    assert (len(train),len(dev),len(test))==(163,56,56) and np.isfinite(items[tr]).all()
    assert ((items[tr]>=0)&(items[tr]<=3)).all()
    labelnote=dict(item_source=str(a.item_labels),sha256=hashlib.sha256(Path(a.item_labels).read_bytes()).hexdigest(),authenticated_original_item_rows=219,
        original_item_sum_vs_released_total_disagreements=int((items[tr].sum(1)!=phq[tr]).sum()),alternative_275_item_file_used=False,test_item_labels_used=False)
    (out/'item_label_provenance.json').write_text(json.dumps(labelnote,indent=2));rows=[];histories=[];selected=[]
    for r,seed in enumerate([1021,42,1337]):
        if a.protocol=='official':model,ep,h=fit(X,items,train,dev,seed,'cuda');cal=predict(model,X,dev,'cuda');caly=y[dev]
        else:
            cal=np.full(len(tr),np.nan);caly=y[tr];eps=[]
            for j,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(tr)),y[tr])):
                inner,ep,h=fit(X,items,tr[it],tr[iv],seed+97*j,'cuda');cal[iv]=predict(inner,X,tr[iv],'cuda');eps.append(ep)
                histories.extend(dict(seed=seed,stage='inner',inner_fold=j,**v) for v in h)
            ep=int(np.median(eps));model,_,h=fit(X,items,tr,test,seed+909,'cuda',epochs=ep,select=False)
        histories.extend(dict(seed=seed,stage='final',inner_fold=-1,**v) for v in h);score=predict(model,X,test,'cuda');selected.append(dict(seed=seed,selected_epoch=ep))
        assert np.isfinite(cal).all()
        torch.save(dict(state_dict=model.state_dict(),seed=seed,epochs=ep,protocol=a.protocol,item_label_provenance=labelnote),out/f'checkpoint_seed{seed}.pt')
        for op,threshold in [('score_cutoff10_secondary',10.),('development_f1' if a.protocol=='official' else 'train_oof_f1',best_f1_threshold(caly,cal)),('train_prevalence',prior_threshold(cal,float(caly.mean())))]:
            rows.extend(dict(subject_id=str(m.subject_id.iloc[idx]),label=int(y[idx]),phq_score=float(phq[idx]),model='MTSP-full-native-MentalBERT-'+a.protocol,repeat=r,seed=seed,operating_point=op,score=float(score[i]),pred=int(score[i]>=threshold),threshold=float(threshold)) for i,idx in enumerate(test))
        pd.DataFrame(rows).to_csv(out/'subject_predictions.partial.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'subject_predictions.csv',index=False);metrics=[]
    for (name,r,op),g in df.groupby(['model','repeat','operating_point']):
        error=g.score-g.phq_score
        metrics.append(dict(model=name,repeat=r,operating_point=op,mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),**metric_row(g.label.to_numpy(),g.score.to_numpy(),pred=g.pred.to_numpy())))
    rep=pd.DataFrame(metrics);rep.to_csv(out/'repeat_metrics.csv',index=False)
    summary=pd.concat([summarize_repeats(d).assign(operating_point=op) for op,d in rep.groupby('operating_point')]);summary.to_csv(out/'summary.csv',index=False)
    rep.groupby('operating_point')[['mae','rmse']].agg(['mean','std']).to_csv(out/'regression_summary.csv')
    pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False);pd.DataFrame(selected).to_csv(out/'training_selected_epochs.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps(dict(arguments=vars(a), seeds=[1021,42,1337], item_labels=labelnote, text_encoder='MentalBERT', checkpoint_metric='total PHQ MAE'),indent=2));(out/'COMPLETE').write_text('ok\n');print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
