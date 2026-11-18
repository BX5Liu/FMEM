import numpy as np
from ablation import strict_profile,fit_score,BranchSpec,BranchModel,make_inner_splits,best_f1_threshold,prior_threshold,metric_row
from fixed_score_fusion import fuse

def text_features_without_labels(path,ids):
    from cef_common import TOPICS
    with np.load(path,allow_pickle=True) as z:
        if 'positive_ids' not in z:return strict_profile(path,ids)
        positions={str(s).strip():i for i,s in enumerate(z['positive_ids'])}
        if len(positions)!=len(z['positive_ids']) or set(ids)-set(positions):raise ValueError('Exact target text identity mismatch')
        indices=[positions[s] for s in ids]
        for topic in TOPICS:np.testing.assert_array_equal(z[topic+'_ids'].astype(str)[indices],ids)
        return np.stack([z[topic+'_features'][indices].astype(np.float32) for topic in TOPICS],axis=1)

def fit_variant(ids,y,xs,kind,disabled=()):
    zi=np.zeros((len(y),3));models=[];norm=[]
    for b,x in enumerate(xs):
        if b in disabled:models.append(None);norm.append(None);continue
        seed=42+41*b;branch=BranchSpec('mean',0,.03 if b==0 else .1)
        for j,(tr,va) in enumerate(make_inner_splits(ids,y,3,seed)):zi[va,b]=fit_score(x[tr],y[tr],x[va],branch,seed+97*j)[0]
        model=BranchModel(branch,seed).fit(x,y);ref=model.decision(x);models.append(model);norm.append((float(ref.mean()),max(float(ref.std()),1e-8)))
    choice={}
    calibration=fuse(zi,kind,disabled)
    thresholds={'source_oof_f1':float(best_f1_threshold(y,calibration)),'source_prevalence':float(prior_threshold(calibration,float(y.mean())))}
    return dict(models=models,normalizers=norm,thresholds=thresholds,calibration=calibration,kind=kind,disabled=list(disabled),choice=choice)

def score_variant(model,z):
    return fuse(z,model['kind'],model['disabled'])
