import argparse,hashlib,json,os,re,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    ap=argparse.ArgumentParser()
    for n in ['source_vocal','target_vocal','model','outdir']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,os.environ['FMEM_REFERENCE_DIR'])
    from AK_audio_llm_evidence_v3 import Qwen2AudioScorer,audio_windows,SR
    from edaic_ceef7_common import load_manifest
    import librosa
    if not torch.cuda.is_available():raise RuntimeError('Submit through L40 sbatch')
    definition=dict(arguments=vars(a),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_config_sha256=hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
        supervised_fitting=False,adapter=None,window_seconds=30,
        extraction='Frozen Qwen2-Audio audio_tower mean of hidden time positions -> per-window L2 -> mean windows',
        comparison='Same checkpoint, waveform, context partition and deterministic windows as Vocal-8; no questions or clinical labels enter embeddings')
    f=out/'definition.json'
    if f.exists() and json.loads(f.read_text())!=definition:raise ValueError('Definition changed; use a new output folder')
    f.write_text(json.dumps(definition,indent=2))
    records=[]
    for cohort,path in [('eatd',a.source_vocal),('eatd2',a.target_vocal)]:

        z=np.load(path,allow_pickle=True);ids=z['subject_ids'].astype(str);paths=z['audio_paths'].astype(str)
        extraction=str(z['extraction']);nwin=int(re.search(r'n_windows=(\d+)',extraction).group(1))
        if a.model not in extraction:raise ValueError('The Vocal-8 cache used a different checkpoint')
        for subject,wavs in zip(ids,paths):records.append(dict(cohort=cohort,id=subject,paths=list(wavs),windows=nwin,thirds=False))
    if a.limit:records=records[:a.limit]
    if not a.limit and len(records)!=253:raise ValueError('Expected 162+91 participants')
    scorer=Qwen2AudioScorer(a.model,'bfloat16',30.,3);model=scorer.model;processor=scorer.processor
    model.eval();progress=out/'progress.jsonl';saved={}
    if progress.exists():
        for line in progress.read_text().splitlines():
            row=json.loads(line);saved[(row['cohort'],row['id'])]=row
    start=time.time()
    for n,row in enumerate(records):
        key=(row['cohort'],row['id'])
        if key not in saved:
            waves=[]
            for path in row['paths']:
                if not Path(path).is_file():raise ValueError('Missing source waveform')
                wave,_=librosa.load(path,sr=SR,mono=True)
                waves.extend(np.array_split(wave,3) if row['thirds'] else [wave])
            if len(waves)!=3:raise ValueError('Expected three contexts')
            context=[];counts=[]
            for wave in waves:
                vectors=[];windows=audio_windows(wave,30.,row['windows'])
                for window in windows:
                    inputs=processor.feature_extractor([window],sampling_rate=SR,padding='max_length',max_length=SR*30,return_tensors='pt')
                    feats=inputs.input_features
                    if feats.shape[-1]!=3000:raise ValueError('Expected 30-second audio feature frame budget')
                    with torch.no_grad():
                        h=model.audio_tower(feats.to(model.device,dtype=model.dtype)).last_hidden_state.float().mean(1)
                        vectors.append(torch.nn.functional.normalize(h,dim=-1).cpu().numpy()[0])
                context.append(np.mean(vectors,axis=0).tolist());counts.append(len(windows))
            item=dict(cohort=row['cohort'],id=row['id'],features=context,window_counts=counts)
            with progress.open('a',encoding='utf8') as h:h.write(json.dumps(item)+'\n')
            saved[key]=item
        print(f'Matched waveform embedding {n+1}/{len(records)} seconds={time.time()-start:.1f}',flush=True)
    for cohort in sorted(set(r['cohort'] for r in records)):
        subset=[r for r in records if r['cohort']==cohort];values=[saved[(r['cohort'],r['id'])] for r in subset]
        x=np.asarray([r['features'] for r in values],np.float32)
        if not np.isfinite(x).all() or float(x.std(axis=0).mean())<=1e-8:raise ValueError('Invalid or constant audio embeddings')
        np.savez_compressed(out/('matched_audio_'+cohort+'.npz'),subject_ids=np.asarray([r['id'] for r in values]),features=x,
            window_counts=np.asarray([r['window_counts'] for r in values]),provenance=np.asarray(json.dumps(definition)))
    if not a.limit:(out/'COMPLETE').write_text('all waveform-only matched embeddings complete\n')

if __name__=='__main__':main()
