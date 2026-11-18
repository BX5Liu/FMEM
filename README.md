# Fine-Grained Multimodal Evidence Modeling

FMEM predicts binary depression-classification labels from text and audio responses. It combines general text semantics, Observable-12 text evidence, and Vocal-12 audio evidence by averaging standardized logistic-regression scores.

The repository contains feature extraction and evaluation scripts, adapted baselines, and threshold analyses for EATD-Corpus, EATD-Corpus 2.0, and E-DAIC.

## Setup

Use Python 3.10, Bash, Git, and CUDA for feature extraction and neural training. The requirements files list the versions used in the experiments. Install PyTorch and torchaudio with matching CUDA builds.

~~~sh
python -m pip install -r requirements-extraction.txt
python prepare_baselines.py sega depmamba mtsp pttsd
~~~

Set the dataset, model, and feature paths in a local shell configuration file. Pass its path through FMEM_CONFIG; config.sh loads it before each experiment.

~~~sh
export FMEM_DATA_ROOT=/path/to/datasets
export FMEM_PYTHON=/path/to/venv/bin/python
export FMEM_FEATURE_ROOT=/path/to/feature-cache
export FMEM_OUTPUT_ROOT=/path/to/results
export FMEM_EDAIC_MANIFEST=/path/to/edaic_manifest.csv
export FOLD_JSON=/path/to/eatd_folds.json
~~~

Use a separate output directory when experiment settings change. Dataset files, checkpoints, cached features, and participant-level predictions should remain outside the repository.

## Inputs

| Input | Path setting or location relative to FMEM_FEATURE_ROOT |
|---|---|
| Original EATD and EATD 2.0 participant folders | TRAIN_DATA_ROOT, TEST_DATA_ROOT |
| EATD and EATD 2.0 reference audio NPZs for cohort IDs and baseline audio | TRAIN_AUDIO, TEST_AUDIO |
| Semantic features for the two Chinese cohorts | BEST_TEXT, TGT_TEXT |
| Observable-12 features for the two Chinese cohorts | LLM_OBS, TGT_LLM |
| Vocal-8 caches used to build the twelve-indicator features | VOCAL_SRC, VOCAL_TGT |
| Original 5-by-3 EATD split manifest | FOLD_JSON |
| EATD expanded features | expanded_features/observable16.npz, expanded_features/vocal12.npz |
| EATD 2.0 Vocal-12 | final_target_vocal12/vocal12.npz |
| Chinese corpus matched audio embeddings | version4_matched_audio_features/matched_audio_eatd.npz and matched_audio_eatd2.npz |
| E-DAIC interview manifest | FMEM_EDAIC_MANIFEST |
| E-DAIC text banks | round2_features/embedding_*.npz and observable_*.npz for clean_sentence3, whole2048, timeline3, sentence5 |
| SEGA released EATD features | SEGA_FEATURES |
| E-DAIC PHQ-8 item labels for MTSP | PHQ_ITEMS |
| MTSP/PTTSD embeddings | round3_native_features/mentalbert_original_asr.npz and minilm_original_asr.npz |

The E-DAIC manifest contains subject_id, official_split (train, dev, test), label, text_path, and audio_path columns. Text paths point to the released timestamped transcript CSVs next to the corresponding audio files. The regression baselines also read phq_score, label_source, and source_id. The manifest builder is reference/build_edaic_official_eval_manifest.py. Run it with --help for the dataset inputs and use its edaic_official_three_split_manifest.csv output. The cohort contains 163 training, 56 development, and 56 test participants.

For cached E-DAIC extraction, FMEM_EDAIC_CACHE must point to a directory containing feature_records/ and TIMESTAMP_QC_PASSED. Unset FMEM_EDAIC_CACHE when generating a new audio cache. The code writes the stacked audio feature banks into FMEM_OUTPUT_ROOT/edaic/features/.

## Evaluation

Run the scripts from the repository root. Use --dry-run to inspect a Slurm command before submission. The supplied scripts request one L40 GPU and six CPU cores per task, with per-stage array limits.

~~~sh
bash submit.sh edaic-evaluate --dry-run
bash submit.sh eatd-evaluate
bash submit.sh eatd2-evaluate
bash submit.sh edaic-evaluate
~~~

| Experiment | Slurm stage | Results below FMEM_OUTPUT_ROOT |
|---|---|---|
| EATD main, feature comparison, ablations, fusion, inventory and aggregation | eatd-evaluate | eatd/core, eatd/indicators, eatd/compositions, eatd/robustness |
| EATD GRU/BiLSTM, modality attention, MulT, MISA | eatd-deep | eatd/deep |
| EATD ACMA and CANAMRF | eatd-architecture | eatd/architecture |
| EATD SEGA and CAF-Mamba | eatd-published | eatd/sega_native, eatd/sega_qwen, eatd/caf_qwen |
| EATD paired F1 bootstrap, after eatd-evaluate | eatd-bootstrap | final_paired_f1_v1 |
| EATD 2.0 FMEM | eatd2-evaluate | eatd2/fmem |
| EATD 2.0 GRU/BiLSTM, MulT, MISA, SEGA, CAF-Mamba | eatd2-baselines | eatd2 |
| E-DAIC main, feature comparison, ablations, fusion and grouping | edaic-evaluate | edaic/summary.csv |
| E-DAIC SEGA, CAF-Mamba and DepMamba, after edaic-evaluate | edaic-baselines | edaic/baselines |
| E-DAIC three-seed baseline summary, after edaic-baselines | edaic-collect | edaic/baseline_summary.csv |
| E-DAIC text baselines | mtsp, pttsd | edaic/mtsp, edaic/pttsd |
| FMEM threshold analysis, after edaic-evaluate and eatd2-evaluate | threshold-audit | threshold/audit.json |

Pass --dependency=afterok:JOBID to submit.sh to schedule a dependent stage. The threshold analysis compares quantile-F1, exact-F1, and exact-BACC selectors. The paper uses quantile-F1. EATD selects thresholds from inner-fold scores. E-DAIC selects on the development set. EATD 2.0 uses the full EATD training cohort and its source OOF threshold.

Check the archived tables without running the models:

~~~sh
python check_results.py --bundled
~~~

To check results from a new run, pass its output directory:

~~~sh
python check_results.py "$FMEM_OUTPUT_ROOT"
~~~

## Feature extraction

The pretrained model paths are QWEN_EMB, QWEN_LLM, AUDIO_LLM, MENTALBERT,
and MINILM_PATH_FILE. download_model.py accepts a ModelScope model ID and cache directory.

~~~sh
python download_model.py --model-id Qwen/Qwen3-Embedding-8B --cache-dir /path/to/models
~~~

For E-DAIC, run edaic-text for the text banks and edaic-prepare for eligible response intervals. Run edaic-extract after edaic-prepare. edaic-evaluate uses both the text banks and the completed audio extraction. The same main audio clips supply Vocal-12 and matched embeddings for the neural comparisons.

For EATD, reference/extract_text_pooling_variants_v3.py, reference/F2_llm_evidence_logodds.py, and reference/AK_audio_llm_evidence_v3.py generate the base semantic, Observable-12, and Vocal-8 banks from the cohort reference NPZ and the three response folders. Use --instruct for semantic extraction, --lang zh for the Chinese Vocal-8 bank, and --rubric observable for text evidence. Run --help for each extractor. Set BEST_TEXT and TGT_TEXT to the respective last-token banks. The optional E-DAIC input audit can compare against a previous manifest specified by FMEM_EDAIC_AUDIT_REFERENCE_MANIFEST; without it, the current manifest is used. Run expand-indicators, then eatd2-vocal to complete the Vocal-12 banks. Run eatd-matched-audio for the neural baseline inputs.

For the SEGA released-feature comparison, use prepare_sega_features.py to align the EATD/3-model/data/EATD-Features directory of the author checkout with the original EATD participants. Save its output at SEGA_FEATURES.

For MTSP, native_mtsp.py --stage prepare can convert the author encoder weights to the expected format. The MINILM_PATH_FILE file contains the local MiniLM checkpoint directory on one line. The mtsp and pttsd stages extract embeddings when their cache files are absent.

## Reference results

The results directory contains aggregate metrics. EATD reports mean and sample standard deviation across five repetitions. Neural baselines on E-DAIC and EATD-Corpus 2.0 report the mean and sample standard deviation across three training seeds. FMEM is evaluated using a single fitted model on each test set.

In the EATD archives, SLEF-D-12x12-unbounded identifies the FMEM configuration reported in the paper. SLEF-D-12x12 is the clipping control. The archived identifiers are preserved for comparison with the run outputs.

See THIRD_PARTY_NOTICES.md for upstream model sources and licensing information.
