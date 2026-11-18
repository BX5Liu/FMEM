import json

import pandas as pd

from common import OUT, save_json


def main():
    if not (OUT/'COMPLETE').exists():
        raise RuntimeError('Primary speaker-selection evaluation incomplete')
    rows=[]
    for model in ['sega','caf','depmamba']:
        for seed in [42,1337,2025]:
            path=OUT/'baselines'/f'{model}_seed{seed}'
            if not (path/'COMPLETE').exists():
                raise RuntimeError('A baseline seed is incomplete')
            frame=pd.read_csv(path/'metrics.csv')
            rows.extend(frame.to_dict('records'))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'baseline_seed_metrics.csv',index=False)
    metrics=[key for key in ['auc','f1','bacc','acc','precision','recall','specificity'] if key in frame]
    summary=frame.groupby('model')[metrics].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/'baseline_summary.csv',index=False)
    coverage=pd.read_csv(OUT/'clip_coverage.csv')
    aggregate=coverage.groupby(['split','policy'])[['clip_count','audio_seconds']].agg(['min','median','max'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(OUT/'clip_coverage_aggregate.csv',index=False)
    (OUT/'ALL_EXPERIMENTS_COMPLETE').write_text('all speaker-selection controls and baseline seeds complete\n')
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    main()

