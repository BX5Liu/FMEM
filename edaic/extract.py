import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from common import BASE, OUT, SR, inputs, load_wave, require_gpu, save_json, settings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    args=parser.parse_args()
    require_gpu()
    if not (OUT/'TIMESTAMP_QC_PASSED').exists():
        raise RuntimeError('Released-transcript timestamp checks have not passed')
    cfg=settings()
    sys.path[:0]=[str(BASE),os.environ['FMEM_REFERENCE_DIR']]
    from AK_audio_llm_evidence_v3 import Qwen2AudioScorer,ITEMS_EN
    from extract_expanded import VOCAL_EXTRA_EN
    import torch
    scorer=Qwen2AudioScorer(cfg['audio_model'],'bfloat16',30.,1)
    embedding_dim=int(scorer.model.config.audio_config.d_model)
    items=ITEMS_EN+VOCAL_EXTRA_EN
    if len(items)!=12:
        raise ValueError('Vocal-12 inventory changed')
    rows=inputs()[args.shard::cfg['shard_count']]
    for index,row in enumerate(rows):
        path=OUT/'feature_records'/(row['subject_id']+'.json')
        if path.exists():
            continue
        wave=load_wave(row['audio_path']).mean(1)
        selections=json.loads((OUT/'clip_intervals'/(row['subject_id']+'.json')).read_text())
        all_features={}
        clip_cache={}
        for name,groups in selections.items():
            vocal=[]
            matched=[]
            durations=[]
            for group in groups:
                if not group:
                    vocal.append([0.]*len(items))
                    matched.append([0.]*embedding_dim)
                    durations.append([])
                    continue
                v=[];m=[];seconds=[]
                for start,end in group:
                    key=(int(round(start*SR)),int(round(end*SR)))
                    if key not in clip_cache:
                        audio=wave[key[0]:key[1]]
                        if not 3*SR-2 <= len(audio) <= 30*SR+2:
                            raise ValueError('Invalid audio clip duration')
                        values=[float(scorer.score_array(audio,'local.wav',
                            f"Does the speaker's vocal delivery show {item}? Answer Yes or No.")[0]) for item in items]
                        feat=scorer.processor.feature_extractor(
                            [audio],sampling_rate=SR,padding='max_length',max_length=SR*30,return_tensors='pt').input_features
                        if feat.shape[-1]!=3000:
                            raise ValueError('Audio model frame budget changed')
                        with torch.inference_mode():
                            vector=scorer.model.audio_tower(feat.to(scorer.model.device,dtype=scorer.model.dtype)).last_hidden_state.float().mean(1)
                            vector=torch.nn.functional.normalize(vector,dim=-1).cpu().numpy()[0]
                        clip_cache[key]=(values,vector.tolist(),len(audio)/SR)
                    values,vector,seconds_value=clip_cache[key]
                    v.append(values);m.append(vector);seconds.append(seconds_value)
                vocal.append(np.mean(v,axis=0).tolist())
                matched.append(np.mean(m,axis=0).tolist())
                durations.append(seconds)
            if not np.isfinite(np.asarray(vocal)).all() or not np.isfinite(np.asarray(matched)).all():
                raise ValueError('Nonfinite feature extraction')
            all_features[name]=dict(vocal=vocal,matched=matched,clip_seconds=durations,
                                   valid_groups=[bool(group) for group in groups])
        save_json(path,dict(subject_id=row['subject_id'],features=all_features,feature_names=items,
                            labels_accessed=False,speaker_selection='released_transcript_timestamp_intervals'))
        print(f'Audio extraction shard {args.shard}: {index+1}/{len(rows)}',flush=True)
    (OUT/f'FEATURE_SHARD_{args.shard}_COMPLETE').write_text('audio feature shard complete\n')


if __name__=='__main__':
    main()

