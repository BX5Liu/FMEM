import os,sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
from ablation import strict_profile

base=Path(os.environ.get('FMEM_REFERENCE_DIR',str(Path(__file__).resolve().parent/'reference')))
sys.path.insert(0,str(base))
import run_depression_architecture_baselines as original
original_load=original.load_dataset
matched=Path(os.environ['FMEM_MATCHED_AUDIO'])
def load(*args,**kwargs):
    data=original_load(*args,**kwargs)
    data['audio_sem']=strict_profile(str(matched),data['ids'])
    assert data['audio_sem'].shape[:2]==data['text'].shape[:2]
    return data
original.load_dataset=load
original.load_extra_branch=lambda filename,ids:strict_profile(filename,ids)
if __name__=='__main__':
    original.main()
    out=Path(sys.argv[sys.argv.index('--outdir')+1]);kind=sys.argv[sys.argv.index('--feature_set')+1]
    shutil.copy2(out/('architecture_'+kind+'_repeat_summary.csv'),out/'summary.csv')
    (out/'final_feature_provenance.json').write_text(json.dumps(dict(audio=str(matched),audio_sha256=hashlib.sha256(matched.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),note='Identical frozen waveform features and windows for generic and structured styles; structured adds Observable12/Vocal12; all PCA/scalers/epoch choices/thresholds fitted within training folds'),indent=2))
    (out/'COMPLETE').write_text('matched architecture feature verification complete\n')
