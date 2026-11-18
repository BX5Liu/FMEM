from pathlib import Path
import os,json
import numpy as np
import pandas as pd
from ablation import holm_adjust
root=Path(os.environ['OUT_SUPP']);source=root/'eatd/core';out=root/'final_paired_f1_v1';out.mkdir(exist_ok=True)
d=pd.read_csv(source/'subject_predictions.csv',dtype={'subject_id':str});d=d[d.operating_point=='train_oof_f1'];candidate='SLEF-D-12x12-unbounded'
refs=['SLEF-D-original','Balanced-without-vocal','Single-observable','Evidence-concat-LR','Balanced-original-Vocal8','Balanced-selected-Vocal8','Single-vocal','Matched-audio-only']
wide={}
for name in [candidate]+refs:
 g=d[d.model==name];wide[name]=g.pivot(index='subject_id',columns='repeat',values='pred').sort_index()
idx=wide[candidate].index;labels=d[d.model==candidate].groupby('subject_id').label.first().reindex(idx).to_numpy(int)
assert all(w.index.equals(idx) and not w.isna().any().any() for w in wide.values())
pos=np.flatnonzero(labels==1);neg=np.flatnonzero(labels==0);rng=np.random.default_rng(42);draws=[]
for start in range(0,10000,500):draws.append(np.concatenate([rng.choice(pos,(500,len(pos))),rng.choice(neg,(500,len(neg)))],axis=1))
def f1_sample(pred,indices):
 p=pred[indices];yy=labels[indices,None];tp=(p*yy).sum(axis=1);den=p.sum(axis=1)+yy.sum(axis=1)
 return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean(axis=1)
cp=wide[candidate].to_numpy(int);cpoint=f1_sample(cp,np.arange(len(labels))[None,:])[0];rows=[]
for ref in refs:
 rp=wide[ref].to_numpy(int);point=cpoint-f1_sample(rp,np.arange(len(labels))[None,:])[0]
 delta=np.concatenate([f1_sample(cp,ix)-f1_sample(rp,ix) for ix in draws]);pval=min(1.,2*min((np.sum(delta<=0)+1)/10001,(np.sum(delta>=0)+1)/10001))
 rows.append(dict(candidate=candidate,reference=ref,n_subjects=len(labels),n_repeats=cp.shape[1],delta_f1=point,ci_lo=float(np.quantile(delta,.025)),ci_hi=float(np.quantile(delta,.975)),p_boot=pval,scope='Paired label-stratified subject bootstrap, same draws across five repeats; exploratory source development'))
df=pd.DataFrame(rows);df['p_holm']=holm_adjust(df.p_boot);df.to_csv(out/'summary.csv',index=False);(out/'COMPLETE').write_text('paired F1 inference complete\n');print(df.to_string(index=False),flush=True)
