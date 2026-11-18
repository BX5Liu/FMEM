from __future__ import annotations
import argparse,json,os,sys,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,os.environ['FMEM_REFERENCE_DIR'])
from ablation import strict_profile
from ceef7_idsafe import resolve_folders,assert_no_signature_conflicts
from cef_common import TOPICS

TEXT_EXTRA_ZH=['睡眠困难或睡眠过多','容易烦躁或易怒','反复纠结自己的负面经历','做决定困难或犹豫不决']
TEXT_EXTRA_EN=['sleep difficulty or excessive sleep','irritability or becoming easily annoyed',
    'repeated rumination about negative personal experiences','indecision or difficulty making decisions']
VOCAL_EXTRA_ZH=['声音响度变化很小','说话节奏不规则或忽快忽慢','咬字含糊或发音不清晰','声音带有明显的颤抖']
VOCAL_EXTRA_EN=['little variation in vocal loudness','irregular speech rhythm or abrupt rate changes',
    'imprecise articulation or unclear pronunciation','audible trembling or tremor in the voice']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--modality',choices=['text','audio'],required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--base_npz',required=True)
    ap.add_argument('--data_root',required=True);ap.add_argument('--outdir',required=True)
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    with np.load(a.base_npz,allow_pickle=True) as z:
        ids=[str(s) for s in z['subject_ids']]
        base=np.asarray(z['profiles'],float)
        names=[str(s) for s in z['feature_names']]
    assert base.shape==(162,3,12 if a.modality=='text' else 8)
    resolved,diagnostics=resolve_folders(ids,a.data_root)
    assert_no_signature_conflicts(diagnostics,a.data_root)
    if any(s not in resolved for s in ids):
        raise RuntimeError('Could not resolve all original subjects')
    cache=out/f'expanded_{a.modality}_progress.jsonl'
    definition=out/f'expanded_{a.modality}_cache_definition.json'
    spec=dict(modality=a.modality,model=a.model,base_sha256=hashlib.sha256(Path(a.base_npz).read_bytes()).hexdigest(),
        text_extra=TEXT_EXTRA_ZH,vocal_extra=VOCAL_EXTRA_ZH,version=1)
    if definition.exists() and json.loads(definition.read_text())!=spec:
        raise RuntimeError('Cached extraction belongs to another model or definition')
    if cache.exists() and not definition.exists():
        raise RuntimeError('Extraction cache has no provenance')
    definition.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    lookup={}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item=json.loads(line);lookup[(item['subject_id'],item['context'])]=item['scores']
    if a.modality=='text':
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from F2_llm_evidence_logodds import log_odds,single_token,read_text
        tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True,local_files_only=True).eval()
        yes,no=single_token(tok,'是'),single_token(tok,'否')
        items=TEXT_EXTRA_ZH
        def extract(folder,context):
            text=read_text(folder/f'{context}.txt')
            if not text:
                raise ValueError('Empty original transcript')
            return [log_odds(model,tok,text,item,yes,no,1536) for item in items]
    else:
        from AK_audio_llm_evidence_v3 import Qwen2AudioScorer,find_audio
        scorer=Qwen2AudioScorer(a.model,'bfloat16',30.,3)
        zh=any(any('\u4e00'<=c<='\u9fff' for c in s) for s in names)
        items=VOCAL_EXTRA_ZH if zh else VOCAL_EXTRA_EN
        def extract(folder,context):
            path=find_audio(folder,context)
            if path is None:
                raise ValueError('Missing original audio')
            return [scorer.score(path,(f'这段录音的发声方式是否表现出“{item}”？请只回答 Yes 或 No。' if zh else
                f"Does the speaker's vocal delivery show {item}? Answer Yes or No."))[0] for item in items]
    extra=np.full((len(ids),3,4),np.nan,dtype=np.float32)
    for i,sid in enumerate(ids):
        for j,context in enumerate(TOPICS):
            key=(sid,context)
            if key not in lookup:
                values=extract(resolved[sid],context)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite extraction')
                lookup[key]=values
                with cache.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(subject_id=sid,context=context,scores=values))+'\n')
            extra[i,j]=lookup[key]
        print(f'{a.modality}: {i+1}/{len(ids)} subjects complete',flush=True)
    target=out/('observable16.npz' if a.modality=='text' else 'vocal12.npz')
    np.savez_compressed(target,subject_ids=np.asarray(ids),profiles=np.concatenate([base,extra],axis=-1),
        feature_names=np.asarray(names+items),topics=np.asarray(TOPICS),
        extraction=np.asarray('Existing 12/8 frozen values + 4 post-lock pre-specified constructs.'))
    (out/f'expanded_{a.modality}_definition.json').write_text(json.dumps(dict(original_file=a.base_npz,
        model=a.model,extra_names=items,extra_standard_deviation=extra.reshape(-1,4).std(axis=0).tolist(),
        status='post_lock_exploratory_candidate_pool'),indent=2),encoding='utf-8')
    print('Saved',target,flush=True)

if __name__=='__main__':main()
