from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,os.environ['FMEM_REFERENCE_DIR'])
from edaic_ceef7_common import load_manifest,split_text_three

STRATEGIES=['clean_sentence3','whole2048','whole8192','timeline3','sentence5']
AUDIO_STRATEGIES=['thirds2','whole2','whole6','whole10','fifths2']

def transcript(path):
    d=pd.read_csv(path)
    if 'Text' not in d or not {'Start_Time','End_Time'}<=set(d):
        raise ValueError('Expected timestamped ASR Text column: '+str(path))
    d=d.sort_values(['Start_Time','End_Time'],kind='stable')
    d['Text']=d.Text.fillna('').astype(str)
    if not np.isfinite(d[['Start_Time','End_Time']].to_numpy(float)).all():
        raise ValueError('Nonfinite ASR timestamps')
    return d

def text_segments(d,strategy,duration):
    whole='\n'.join(d.Text)
    if strategy.startswith('whole'): return [whole]
    if strategy=='clean_sentence3': return split_text_three(whole)
    if strategy=='sentence5':
        import re
        units=[x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+',whole) if x.strip()]
        return [' '.join(np.asarray(units,dtype=object)[idx]) or 'No valid response.'
                for idx in np.array_split(np.arange(len(units)),5)]
    if strategy=='timeline3':


        bins=np.minimum(2,np.maximum(0,((d.Start_Time+d.End_Time)/2/duration*3).to_numpy(int)))
        return ['\n'.join(d.loc[bins==j,'Text']) or 'No valid response.' for j in range(3)]
    raise ValueError(strategy)

def response_with_budget(tok,text,budget):
    ids=tok.encode(text,add_special_tokens=False)
    if len(ids)<=budget: return text,len(ids),len(ids)


    n=budget//2
    return tok.decode(ids[:n]+ids[-(budget-n):],skip_special_tokens=True),len(ids),budget

def write_npz(out,name,ids,x,names=None):
    x=np.asarray(x,np.float32)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite extracted features')
    p=dict(subject_ids=np.asarray(ids),features=x,profiles=x,
           provenance=np.asarray('uniform_raw_ASR_Text_column_speaker_roles_unknown'))
    if names is not None: p['feature_names']=np.asarray(names)
    np.savez_compressed(out/(name+'.npz'),**p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--modality',choices=['audit','embedding','observable','vocal'],required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--model',default='')
    ap.add_argument('--outdir',required=True);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    m=load_manifest(a.manifest);m=m.iloc[:a.limit] if a.limit else m
    ids=m.subject_id.astype(str).tolist()
    definition=dict(version=1,modality=a.modality,model=a.model,
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ids=ids,text_strategies=STRATEGIES,audio_strategies=AUDIO_STRATEGIES,
        speaker_policy='ASR does not provide reliable speaker roles; no participant-only claim',
        clip_policy='head+tail response tokens; preserve all question/template suffix tokens')
    defpath=out/(a.modality+'_definition.json')
    if defpath.exists() and json.loads(defpath.read_text())!=definition:
        raise ValueError('Extraction definition changed; use a new output directory')
    defpath.write_text(json.dumps(definition,indent=2),encoding='utf-8')
    import soundfile as sf
    started=time.time();rows=[]
    if a.modality=='audit':
        from transformers import AutoTokenizer
        from E2_edaic_extract_llm_logodds import prompt,ITEMS
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
        audit_reference = os.environ.get('FMEM_EDAIC_AUDIT_REFERENCE_MANIFEST', '')
        reference_path = Path(audit_reference) if audit_reference else None
        old = (pd.read_csv(reference_path).set_index('subject_id')
               if reference_path is not None and reference_path.is_file() else None)
        from edaic_ceef7_common import read_text,participant_only_text
        for r in m.itertuples(index=False):
            d=transcript(r.text_path);info=sf.info(r.audio_path);duration=info.duration
            original_path=str(old.loc[int(r.subject_id),'text_path']) if old is not None and int(r.subject_id) in old.index else r.text_path
            legacy=participant_only_text(read_text(original_path))
            segs=split_text_three(legacy)
            counts=[len(tok.encode(prompt(tok,s,ITEMS[0]),add_special_tokens=False)) for s in segs]
            clean='\n'.join(d.Text);mid=(d.Start_Time+d.End_Time)/2


            cuts=np.array_split(np.arange(len(d)),3)
            boundaries=[float(d.End_Time.iloc[c[-1]])/duration if len(c) else None for c in cuts[:2]]
            rows.append(dict(subject_id=r.subject_id,split=r.split,old_text_format=Path(original_path).suffix,
                raw_rows=len(d),audio_seconds=duration,clean_chars=len(clean),legacy_chars=len(legacy),
                text_tokens=len(tok.encode(clean,add_special_tokens=False)),
                legacy_prompt_tokens_max=max(counts),legacy_truncated_segments=sum(c>1536 for c in counts),
                row_count_boundary1=boundaries[0],row_count_boundary2=boundaries[1],
                timestamps_outside_audio=int(((d.Start_Time<0)|(d.End_Time>duration+1)).sum()),
                empty_rows=int((d.Text.str.strip()=='').sum()),speaker_roles='unknown'))
        pd.DataFrame(rows).to_csv(out/'input_audit.csv',index=False)


        acoustic=[];acoustic_names=None
        for r in m.itertuples(index=False):
            p=Path(r.audio_path).parent/'features'/(str(r.subject_id)+'_OpenSMILE2.3.0_egemaps.csv')
            e=pd.read_csv(p,sep=';').drop(columns=['name','frameTime'])
            if e.shape[1]!=23 or not np.isfinite(e.to_numpy(float)).all():
                raise ValueError('Expected 23 finite supplied eGeMAPS LLDs')
            if acoustic_names is not None and acoustic_names!=list(e.columns):
                raise ValueError('Inconsistent acoustic columns')
            acoustic_names=list(e.columns)
            acoustic.append(np.r_[e.mean().to_numpy(),e.std(ddof=0).to_numpy()][None,:])
        write_npz(out,'egemaps23_meanstd46',ids,acoustic,
                  [s+'_mean' for s in acoustic_names]+[s+'_std' for s in acoustic_names])
        (out/'AUDIT_COMPLETE').write_text('ok\n');return
    cache=out/(a.modality+'_progress.jsonl');lookup={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                v=json.loads(line);lookup[(v['id'],v['strategy'])]=v
    if a.modality in ('embedding','observable'):
        from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
        tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
        cls=AutoModel if a.modality=='embedding' else AutoModelForCausalLM
        model=cls.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,
                 torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa').eval()
        if a.modality=='embedding':
            from E1_edaic_extract_text_last import embed
            names=None
        else:
            from E2_edaic_extract_llm_logodds import ITEMS,one_token_variants,score,prompt
            names=ITEMS
            from extract_expanded import TEXT_EXTRA_EN
            yes=one_token_variants(tok,['Yes',' Yes','yes',' yes','YES'])
            no=one_token_variants(tok,['No',' No','no',' no','NO'])
            overhead=max(len(tok.encode(prompt(tok,'',i),add_special_tokens=False)) for i in ITEMS)+24
        strategies=STRATEGIES
    else:
        import librosa
        from AK_audio_llm_evidence_v3 import Qwen2AudioScorer,ITEMS_EN,SR
        from extract_expanded import VOCAL_EXTRA_EN
        scorer=Qwen2AudioScorer(a.model,'bfloat16',30.,2);names=ITEMS_EN
        strategies=AUDIO_STRATEGIES
    all_x={s:[] for s in strategies};expanded=[];audit=[]
    for i,r in enumerate(m.itertuples(index=False)):
        duration=sf.info(r.audio_path).duration
        d=transcript(r.text_path) if a.modality!='vocal' else None
        wave=None
        for strategy in strategies:
            key=(r.subject_id,strategy)
            if key not in lookup:
                if a.modality=='vocal':
                    if wave is None: wave,_=librosa.load(r.audio_path,sr=SR,mono=True)
                    if strategy.startswith('whole'):
                        nwin=int(strategy[5:]);segs=[wave]
                    else:
                        nwin=2;segs=np.array_split(wave,3 if strategy=='thirds2' else 5)
                    scorer.n_windows=nwin;values=[];details=[]
                    items=names+VOCAL_EXTRA_EN if strategy=='thirds2' else names
                    for j,seg in enumerate(segs):
                        values.append([scorer.score_array(seg,'local.wav',f"Does the speaker's vocal delivery show {item}? Answer Yes or No.")[0] for item in items])
                        details.append(dict(segment=j,seconds=len(seg)/SR,n_windows=nwin))
                else:
                    segs=text_segments(d,strategy,duration);values=[];details=[]
                    maxlen=int(strategy[5:]) if strategy.startswith('whole') else (2048 if a.modality=='embedding' else 1536)
                    for j,seg in enumerate(segs):
                        budget=maxlen-(200 if a.modality=='embedding' else overhead)
                        clipped,norig,nkept=response_with_budget(tok,seg,budget)
                        if a.modality=='embedding': values.append(embed(model,tok,clipped,maxlen).tolist())
                        else:
                            items=names+TEXT_EXTRA_EN if strategy=='clean_sentence3' else names
                            for item in items:
                                if len(tok.encode(prompt(tok,clipped,item),add_special_tokens=False))>maxlen:
                                    raise ValueError('Protected evidence question would be truncated')
                            values.append([score(model,tok,clipped,item,yes,no,maxlen) for item in items])
                        details.append(dict(segment=j,tokens_original=norig,tokens_kept=nkept,max_len=maxlen))
                item=dict(id=r.subject_id,strategy=strategy,values=np.asarray(values).tolist(),details=details)
                with cache.open('a',encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                lookup[key]=item
            values=np.asarray(lookup[key]['values'])
            base_dim=None if names is None else len(names)
            if base_dim and values.shape[-1]>base_dim:
                expanded.append(values);values=values[...,:base_dim]
            all_x[strategy].append(values)
            audit.extend(dict(subject_id=r.subject_id,strategy=strategy,**v) for v in lookup[key]['details'])
        print(f'{a.modality} {i+1}/{len(m)} complete; seconds={time.time()-started:.1f}',flush=True)
    for strategy,arr in all_x.items(): write_npz(out,a.modality+'_'+strategy,ids,arr,names)
    if expanded:
        extras=TEXT_EXTRA_EN if a.modality=='observable' else VOCAL_EXTRA_EN
        strategy='clean_sentence3' if a.modality=='observable' else 'thirds2'
        write_npz(out,a.modality+'_'+strategy+'_expanded',ids,expanded,names+extras)
    pd.DataFrame(audit).to_csv(out/(a.modality+'_coverage.csv'),index=False)
    (out/(a.modality.upper()+'_COMPLETE')).write_text('ok\n')

if __name__=='__main__': main()
