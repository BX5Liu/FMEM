import csv
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from common import BASE,CODE,OUT,SR,digest,inputs,save_json,select_clips

MAX_UTTERANCE_SECONDS=90.
REFERENCES={
    'provider_manual':'https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf',
    'timestamp_segmentation':'https://www.isca-archive.org/iberspeech_2024/mallolragolta24_iberspeech.pdf',
    'participant_side_metadata':'https://arxiv.org/html/2603.24651v1'}


def transcript_intervals(rows,duration):
    intervals=[];reasons=Counter()
    for row in rows:
        try:
            start,end=float(row['Start_Time']),float(row['End_Time'])
        except (ValueError,TypeError,KeyError):
            reasons['non_numeric_time']+=1;continue
        if not math.isfinite(start) or not math.isfinite(end):
            reasons['nonfinite_time']+=1;continue
        if end<=start:
            reasons['nonpositive_duration']+=1;continue
        if end-start>=MAX_UTTERANCE_SECONDS:
            reasons['longer_than_90_seconds']+=1;continue
        if start>=duration or end<=0:
            reasons['outside_audio']+=1;continue
        text=row.get('Text','').strip()
        if not text or text.lower() in ['nan','scrubbed_entry']:
            reasons['empty_or_scrubbed_text']+=1;continue
        if start<0 or end>duration:
            reasons['clamped_to_audio_bounds']+=1
        intervals.append([max(0.,start),min(duration,end)])
    intervals.sort()
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],end)
        else:
            merged.append([start,end])
    return merged,dict(reasons)


def main():
    import soundfile as sf
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=dict(audio_model=os.environ['AUDIO_LLM'],
        shard_count=4,clinical_labels_used_for_roles=False,
        speaker_policy='released participant-side transcript timestamp intervals',
        role_method='dataset annotation semantics; no question-prefix or forced two-speaker gate',
        max_utterance_seconds=MAX_UTTERANCE_SECONDS,min_clip_seconds=3.,max_clip_seconds=30.,
        missing_group_policy='explicit validity mask; pool available groups; no replacement speech or missing-person exclusion',
        references=REFERENCES,code_hashes={p.name:digest(p) for p in CODE.glob('*.py')})
    path=OUT/'settings.json'
    if path.exists() and json.loads(path.read_text())!=cfg:
        raise ValueError('Timestamp experiment definition changed; use another output directory')
    save_json(path,cfg)
    accepted=0;failures=[];totals=Counter();by_split=defaultdict(Counter);coverage=[]
    canonical_equal=0;source_rows=0
    for row in inputs():
        source=Path(row['text_path'])
        canonical=Path(row['audio_path']).with_name(row['subject_id']+'_Transcript.csv')
        if not canonical.is_file() or canonical.read_bytes()!=source.read_bytes():
            raise ValueError('Input transcript differs from dataset-directory source')
        canonical_equal+=1
        info=sf.info(row['audio_path']);duration=float(info.duration)
        with source.open(encoding='utf-8-sig') as handle:
            reader=csv.DictReader(handle)
            if set(reader.fieldnames or [])!={'Start_Time','End_Time','Text','Confidence'}:
                raise ValueError('Unexpected released-transcript schema')
            rows=list(reader)
        source_rows+=len(rows)
        turns,reasons=transcript_intervals(rows,duration)
        totals.update(reasons)
        by_split[row['split']]['subjects']+=1
        by_split[row['split']]['raw_audio_seconds']+=duration
        by_split[row['split']]['selected_interval_seconds']+=sum(end-start for start,end in turns)
        save_json(OUT/'released_intervals'/(row['subject_id']+'.json'),dict(
            transcript_sha256=digest(source),audio_bytes=Path(row['audio_path']).stat().st_size,
            channels=info.channels,duration=duration,retained_intervals=turns,rejected_rows=reasons,
            source='released participant-side ASR transcript',annotation_is_manual=False,
            unannotated_time='excluded, not automatically labelled as interviewer speech'))
        try:
            selections={name:select_clips(turns,duration,groups,per_group,allow_empty=True)
                        for name,groups,per_group in [('participant3',3,2),('participant1',1,6),('participant5',5,2)]}
            for name,groups in selections.items():
                if not any(groups):
                    raise ValueError('No eligible response audio for a participant')
                for group in groups:
                    for start,end in group:
                        if not any(start>=a-1e-8 and end<=b+1e-8 for a,b in turns):
                            raise ValueError('Selected clip crosses an unannotated interval')
                coverage.append(dict(split=row['split'],configuration=name,
                                     clips=sum(map(len,groups)),empty_groups=sum(not group for group in groups),
                                     seconds=sum(end-start for group in groups for start,end in group)))
            save_json(OUT/'clip_intervals'/(row['subject_id']+'.json'),selections)
            accepted+=1
        except ValueError as error:
            failures.append(dict(subject_id=row['subject_id'],reason=str(error)))
    coverage_summary={}
    for key in ['participant3','participant1','participant5']:
        values=[v for v in coverage if v['configuration']==key]
        coverage_summary[key]=dict(subjects=len(values),median_clips=float(np.median([v['clips'] for v in values])) if values else None,
            subjects_with_empty_groups=sum(v['empty_groups']>0 for v in values),empty_groups=sum(v['empty_groups'] for v in values),
            median_seconds=float(np.median([v['seconds'] for v in values])) if values else None)
    report=dict(n_expected=275,n_eligible=accepted,unresolved_count=len(failures),
        source_transcripts_identical=canonical_equal,source_rows=source_rows,row_checks=dict(totals),
        by_split={k:dict(v) for k,v in by_split.items()},coverage=coverage_summary,
        references=REFERENCES,
        speaker_roles_from='published semantics of the released E-DAIC response transcripts',
        clinical_labels_accessed=False,supervised_selection=False)
    save_json(OUT/'timestamp_audit_aggregate.json',report)
    save_json(OUT/'timestamp_review_required.json',failures)
    print(json.dumps(report,indent=2),flush=True)
    if failures:
        raise RuntimeError('Timestamp/coverage QC incomplete; no people may be silently omitted')
    (OUT/'TIMESTAMP_QC_PASSED').write_text('all 275 released-transcript timestamp and clip-coverage checks passed\n')


if __name__=='__main__':
    main()

