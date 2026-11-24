# EATD-Corpus 2.0 and Fine-Grained Multimodal Evidence Modeling

This repository accompanies the paper **An Expanded Chinese Depression Dataset and Fine-Grained Multimodal Evidence Modeling for Depression Detection**.

EATD-Corpus 2.0 extends the original EATD-Corpus with audio recordings, corresponding text, and depression screening labels from 91 new participants. Fine-Grained Multimodal Evidence Modeling (FMEM) combines general text semantics, Observable-12 text evidence, and Vocal-12 audio evidence for binary depression detection by averaging standardized logistic regression scores.

This repository provides feature extraction and evaluation scripts, adapted baselines, and threshold analyses for EATD-Corpus, EATD-Corpus 2.0, and E-DAIC.

## Dataset Download

EATD-Corpus 2.0 is available as a password-protected archive through [GitHub Releases](../../releases). The dataset contains audio recordings, corresponding text, and depression screening labels from 91 participants, including 38 participants with positive screening results and 53 with negative screening results.

Download the dataset archive from the Releases page and extract it using the password provided separately to authorized users. The dataset is for academic research and must be used in accordance with the applicable data access and usage requirements.

## Setup

Use Python 3.10, Bash, Git, and CUDA for feature extraction and neural network training. The requirements files specify the package versions used in the experiments. Install PyTorch and torchaudio with matching CUDA builds.

```sh
python -m pip install -r requirements-extraction.txt
python prepare_baselines.py sega depmamba mtsp pttsd
```

Set the dataset, model, and feature paths in a local shell configuration file. Pass its path through `FMEM_CONFIG`; `config.sh` loads the file before each experiment.

```sh
export FMEM_DATA_ROOT=/path/to/datasets
export FMEM_PYTHON=/path/to/venv/bin/python
export FMEM_FEATURE_ROOT=/path/to/feature-cache
export FMEM_OUTPUT_ROOT=/path/to/results
export FMEM_EDAIC_MANIFEST=/path/to/edaic_manifest.csv
export FOLD_JSON=/path/to/eatd_folds.json
```

Use a separate output directory when experiment settings change. Keep downloaded dataset files, checkpoints, cached features, and participant-level predictions outside the local repository working directory.

## Inputs

| Input | Path setting or location relative to `FMEM_FEATURE_ROOT` |
|---|---|
| EATD-Corpus and EATD-Corpus 2.0 participant folders | `TRAIN_DATA_ROOT`, `TEST_DATA_ROOT` |
| EATD-Corpus and EATD-Corpus 2.0 reference audio NPZ files for participant IDs and baseline audio features | `TRAIN_AUDIO`, `TEST_AUDIO` |
| Text semantic features for EATD-Corpus and EATD-Corpus 2.0 | `BEST_TEXT`, `TGT_TEXT` |
| Observable-12 features for EATD-Corpus and EATD-Corpus 2.0 | `LLM_OBS`, `TGT_LLM` |
| Vocal-8 caches used to construct Vocal-12 | `VOCAL_SRC`, `VOCAL_TGT` |
| Original EATD-Corpus split manifest (five repetitions of three-fold cross-validation) | `FOLD_JSON` |
| EATD-Corpus expanded features | `expanded_features/observable16.npz`, `expanded_features/vocal12.npz` |
| EATD-Corpus 2.0 Vocal-12 features | `final_target_vocal12/vocal12.npz` |
| Matched audio embeddings for EATD-Corpus and EATD-Corpus 2.0 | `version4_matched_audio_features/matched_audio_eatd.npz`, `version4_matched_audio_features/matched_audio_eatd2.npz` |
| E-DAIC interview manifest | `FMEM_EDAIC_MANIFEST` |
| E-DAIC text features | `round2_features/embedding_*.npz` and `round2_features/observable_*.npz` for `clean_sentence3`, `whole2048`, `timeline3`, and `sentence5` |
| SEGA released EATD-Corpus features | `SEGA_FEATURES` |
| E-DAIC PHQ-8 item labels for MTSP | `PHQ_ITEMS` |
| MTSP and PTTSD embeddings | `round3_native_features/mentalbert_original_asr.npz`, `round3_native_features/minilm_original_asr.npz` |

The E-DAIC manifest contains `subject_id`, `official_split` (train, dev, test), `label`, `text_path`, and `audio_path` columns. Text paths point to the released timestamped transcript CSV files next to the corresponding audio files. The regression baselines also read `phq_score`, `label_source`, and `source_id`. Use `reference/build_edaic_official_eval_manifest.py` to generate the manifest. Run the script with `--help` to view the required inputs and use its `edaic_official_three_split_manifest.csv` output. The dataset contains 163 training, 56 development, and 56 test participants.

To reuse cached E-DAIC audio features, set `FMEM_EDAIC_CACHE` to a directory containing `feature_records/` and `TIMESTAMP_QC_PASSED`. Unset `FMEM_EDAIC_CACHE` when generating a new audio cache. The extracted audio features are saved in `FMEM_OUTPUT_ROOT/edaic/features/`.

## Evaluation

Run the scripts from the repository root. Use `--dry-run` to inspect a Slurm command before submission. The supplied scripts request one L40 GPU and six CPU cores per task, with array limits specified for each stage.

```sh
bash submit.sh edaic-evaluate --dry-run
bash submit.sh eatd-evaluate
bash submit.sh eatd2-evaluate
bash submit.sh edaic-evaluate
```

| Experiment | Slurm stage | Results relative to `FMEM_OUTPUT_ROOT` |
|---|---|---|
| EATD-Corpus main experiments, feature comparison, ablations, fusion, indicator inventory, and aggregation | `eatd-evaluate` | `eatd/core`, `eatd/indicators`, `eatd/compositions`, `eatd/robustness` |
| EATD-Corpus GRU/BiLSTM, modality attention, MulT, and MISA | `eatd-deep` | `eatd/deep` |
| EATD-Corpus ACMA and CANAMRF | `eatd-architecture` | `eatd/architecture` |
| EATD-Corpus SEGA and CAF-Mamba | `eatd-published` | `eatd/sega_native`, `eatd/sega_qwen`, `eatd/caf_qwen` |
| EATD-Corpus paired F1 bootstrap, after `eatd-evaluate` | `eatd-bootstrap` | `final_paired_f1_v1` |
| EATD-Corpus 2.0 FMEM | `eatd2-evaluate` | `eatd2/fmem` |
| EATD-Corpus 2.0 GRU/BiLSTM, MulT, MISA, SEGA, and CAF-Mamba | `eatd2-baselines` | `eatd2` |
| E-DAIC main experiments, feature comparison, ablations, fusion, and grouping | `edaic-evaluate` | `edaic/summary.csv` |
| E-DAIC SEGA, CAF-Mamba, and DepMamba, after `edaic-evaluate` | `edaic-baselines` | `edaic/baselines` |
| E-DAIC three-seed baseline summary, after `edaic-baselines` | `edaic-collect` | `edaic/baseline_summary.csv` |
| E-DAIC text baselines | `mtsp`, `pttsd` | `edaic/mtsp`, `edaic/pttsd` |
| FMEM threshold analysis, after `edaic-evaluate` and `eatd2-evaluate` | `threshold-audit` | `threshold/audit.json` |

Pass `--dependency=afterok:JOBID` to `submit.sh` to schedule a dependent stage. The threshold analysis compares quantile-F1, exact-F1, and exact-BACC threshold selection. The paper uses quantile-F1. For EATD-Corpus, thresholds are selected from inner-fold scores. For E-DAIC, thresholds are selected on the development set. For EATD-Corpus 2.0, the model is trained on the full EATD-Corpus cohort, and the threshold is selected from its out-of-fold scores.

Check the archived results without running the models:

```sh
python check_results.py --bundled
```

To check results from a new run, pass its output directory:

```sh
python check_results.py "$FMEM_OUTPUT_ROOT"
```

## Feature Extraction

The pretrained model paths are `QWEN_EMB`, `QWEN_LLM`, `AUDIO_LLM`, `MENTALBERT`, and `MINILM_PATH_FILE`. The script `download_model.py` accepts a ModelScope model ID and a cache directory.

```sh
python download_model.py --model-id Qwen/Qwen3-Embedding-8B --cache-dir /path/to/models
```

For E-DAIC, run `edaic-text` to extract text features and `edaic-prepare` to identify eligible response intervals. Run `edaic-extract` after `edaic-prepare` to extract audio features. The `edaic-evaluate` stage uses the extracted text and audio features. Vocal-12 and the matched audio embeddings used by the neural baselines are extracted from the same audio clips.

For EATD-Corpus and EATD-Corpus 2.0, use `reference/extract_text_pooling_variants_v3.py`, `reference/F2_llm_evidence_logodds.py`, and `reference/AK_audio_llm_evidence_v3.py` to extract text semantic features, Observable-12, and Vocal-8 from the corresponding cohort reference NPZ files and response folders. Use `--instruct` for text semantic feature extraction, `--rubric observable` for Observable-12, and `--lang zh` for Vocal-8. Run each script with `--help` to view its arguments. Set `BEST_TEXT` and `TGT_TEXT` to the corresponding last-token text embeddings. Run `expand-indicators`, then `eatd2-vocal`, to construct Vocal-12. Run `eatd-matched-audio` to extract audio embeddings for the neural baselines.

The optional E-DAIC input audit compares the current manifest against the reference manifest specified by `FMEM_EDAIC_AUDIT_REFERENCE_MANIFEST`. If this variable is unset, the current manifest is used as the reference.

For the SEGA released-feature comparison, use `prepare_sega_features.py` to align the `EATD/3-model/data/EATD-Features` directory from the original SEGA repository with the EATD-Corpus participants. Save the output at `SEGA_FEATURES`.

For MTSP, run `native_mtsp.py --stage prepare` to convert the original encoder weights to the required format. The file specified by `MINILM_PATH_FILE` contains the local MiniLM checkpoint directory on one line. The `mtsp` and `pttsd` stages extract embeddings if their cached features are absent.

## Reference Results

The `results/` directory contains aggregate evaluation metrics. EATD-Corpus results are reported as the mean and sample standard deviation across five repetitions. Neural baseline results on E-DAIC and EATD-Corpus 2.0 are reported as the mean and sample standard deviation across three training seeds. FMEM is evaluated using a single fitted model on each E-DAIC and EATD-Corpus 2.0 test set.

In the archived EATD-Corpus results, `SLEF-D-12x12-unbounded` identifies the FMEM configuration reported in the paper, and `SLEF-D-12x12` identifies the configuration with clipping. These original identifiers are retained to match the archived results and experiment outputs.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream model sources and licensing information.
