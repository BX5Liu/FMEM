import argparse,json,hashlib,os,sys
from pathlib import Path
import numpy as np
from extract_expanded import VOCAL_EXTRA_ZH,VOCAL_EXTRA_EN
from cef_common import TOPICS
from ceef7_idsafe import resolve_folders,assert_no_signature_conflicts

def main():
    ap=argparse.ArgumentParser()
    for n in ['model','base_npz','source_vocal12','data_root','outdir']:ap.add_argument('--'+n,required=True)
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    with np.load(a.base_npz,allow_pickle=True) as z:ids=z['subject_ids'].astype(str);base=np.asarray(z['profiles'],float);names=z['feature_names'].astype(str).tolist()
    with np.load(a.source_vocal12,allow_pickle=True) as z:source_names=z['feature_names'].astype(str).tolist()
    if base.shape!=(91,3,8):raise ValueError('EATD2 original evidence cohort changed')
    if names!=source_names[:8]:raise ValueError('Source/target Vocal-8 inventories differ')
    extra_names=source_names[8:]
    if extra_names not in [VOCAL_EXTRA_ZH,VOCAL_EXTRA_EN]:raise ValueError('Unsupported source extra-item inventory')
    folders,diags=resolve_folders(ids,a.data_root);assert_no_signature_conflicts(diags,a.data_root)
    if set(ids)-set(folders):raise ValueError('Missing exact target audio folders')
    definition=dict(model=a.model,base_sha256=hashlib.sha256(Path(a.base_npz).read_bytes()).hexdigest(),source_definition_sha256=hashlib.sha256(Path(a.source_vocal12).read_bytes()).hexdigest(),extra_names=extra_names,windows=3,seconds=30.,label_access=False)
    dp=out/'cache_definition.json'
    if dp.exists() and json.loads(dp.read_text())!=definition:raise ValueError('Changed extraction definition')
    dp.write_text(json.dumps(definition,indent=2));cp=out/'progress.jsonl';cache={}
    if cp.exists():
        for line in cp.read_text().splitlines():
            item=json.loads(line);cache[(item['subject_id'],item['context'])]=item['scores']
    from AK_audio_llm_evidence_v3 import Qwen2AudioScorer,find_audio
    scorer=Qwen2AudioScorer(a.model,'bfloat16',30.,3);extra=np.full((91,3,4),np.nan);zh=extra_names==VOCAL_EXTRA_ZH
    for i,sid in enumerate(ids):
        for j,context in enumerate(TOPICS):
            key=(str(sid),context)
            if key not in cache:
                audio=find_audio(folders[sid],context)
                if audio is None:raise ValueError('Missing target waveform')
                prompts=[f'这段录音的发声方式是否表现出“{item}”？请只回答 Yes 或 No。' if zh else f"Does the speaker's vocal delivery show {item}? Answer Yes or No." for item in extra_names]
                values=[float(scorer.score(audio,p)[0]) for p in prompts]
                if not np.isfinite(values).all():raise ValueError('Nonfinite frozen extraction')
                cache[key]=values
                with cp.open('a') as h:h.write(json.dumps(dict(subject_id=str(sid),context=context,scores=values))+'\n')
            extra[i,j]=cache[key]
        print(f'target frozen Vocal12 {i+1}/91',flush=True)
    features=np.concatenate([base,extra],axis=-1);np.testing.assert_array_equal(features[...,:8],base)
    np.savez_compressed(out/'vocal12.npz',subject_ids=ids,profiles=features,feature_names=np.asarray(source_names),topics=np.asarray(TOPICS))
    (out/'COMPLETE').write_text('source-defined label-free target extraction complete\n')

if __name__=='__main__':main()
