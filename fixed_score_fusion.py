import numpy as np
from ablation import safe_auc,fit_score,BranchSpec,make_inner_splits
from sklearn.model_selection import StratifiedKFold

def branches(engine,tr,ev,k):
    if all(s is None for s in engine.selectors) and engine.standardize:return engine.legacy_branches(tr,ev,k)
    train=[];test=[]
    for b,x in enumerate(engine.X):
        if b in engine.disabled:train.append(np.zeros(len(tr)));test.append(np.zeros(len(ev)));continue
        key=('paper-selectable',engine.names[b],engine.selectors[b],engine.standardize,b,tuple(tr),tuple(ev),k,engine.y[tr].tobytes())
        if key not in engine.shared:
            seed=42+100*b if engine.dataset=='edaic' else 42+1000*k+41*b
            so=seed if engine.dataset=='edaic' else 42+2000*k+41*b
            if engine.dataset=='edaic':
                cvs=[(it,iv) for r in range(5) for it,iv in StratifiedKFold(3,shuffle=True,random_state=seed+1009*r).split(np.zeros(len(tr)),engine.y[tr])]
            else:cvs=list(make_inner_splits(engine.ids[tr],engine.y[tr],3,seed))
            sums=np.zeros(len(tr));counts=np.zeros(len(tr));spec=BranchSpec('mean',0,.03 if b==0 else .1)
            for j,(it,iv) in enumerate(cvs):
                fs=seed+97*(j//3)+j%3 if engine.dataset=='edaic' else seed+97*j
                a,_=fit_score(x[tr[it]],engine.y[tr[it]],x[tr[iv]],spec,fs,engine.selectors[b],engine.standardize)
                sums[iv]+=a;counts[iv]+=1
            if np.any(counts==0):raise ValueError('OOF branch coverage')
            z,selected=fit_score(x[tr],engine.y[tr],x[ev],spec,so,engine.selectors[b],engine.standardize)
            if selected is not None:engine.shared.setdefault('_selections',[]).append(dict(branch=engine.names[b],selected=list(map(int,selected)),training_n=len(tr)))
            engine.shared[key]=(sums/counts,z)
        a,z=engine.shared[key];train.append(a);test.append(z)
    return np.column_stack(train),np.column_stack(test)

def fuse(z,kind='clip3',disabled=(),weights=None):
    active=[b for b in range(3) if b not in disabled]
    if not active:raise ValueError('No active modality')
    x=np.array(z,float,copy=True)
    if kind.startswith('clip'):x[:,2]=np.clip(x[:,2],-float(kind[4:]),float(kind[4:]))
    if kind=='probability':x=1/(1+np.exp(-np.clip(x,-700,700)))
    w=np.zeros(3)
    if weights is None:w[active]=1/len(active)
    else:
        w=np.asarray(weights,float).copy();w[list(disabled)]=0;w/=w.sum()
    return x@w

def predict(engine,plan,tr,ev,k,dev=None):
    if dev is None:zi,zt=branches(engine,tr,ev,k)
    else:
        _,zi=branches(engine,tr,dev,k);_,zt=branches(engine,tr,ev,k)
    weights=None
    if plan.get('fusion')=='reliability':

        yy=engine.y[tr] if dev is None else engine.y[dev]
        weights=np.asarray([max(safe_auc(yy,zi[:,b])-.5,.05) if b not in engine.disabled else 0. for b in range(3)])
        weights/=weights.sum()
    sin=fuse(zi,plan.get('fusion','clip3'),engine.disabled,weights)
    sout=fuse(zt,plan.get('fusion','clip3'),engine.disabled,weights)
    active=[b for b in range(3) if b not in engine.disabled]
    actual=np.zeros(3);actual[active]=1/len(active)
    return sin,sout,dict(fusion=plan.get('fusion','clip3'),weights=(actual if weights is None else weights).tolist(),clip=3 if plan.get('fusion','clip3')=='clip3' else None),None
