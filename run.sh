#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source config.sh
stage="${1:?Pass an experiment name}"
task="${SLURM_ARRAY_TASK_ID:-0}"
eatd=(--audio_npz "$TRAIN_AUDIO" --text_npz "$BEST_TEXT" --data_root "$TRAIN_DATA_ROOT" --fold_json "$FOLD_JSON")
source_args=(--source_audio "$TRAIN_AUDIO" --source_text "$BEST_TEXT" --source_llm "$LLM_OBS"
             --source_vocal "$VOCAL_SRC" --source_root "$TRAIN_DATA_ROOT"
             --source_vocal12 "$CACHE/expanded_features/vocal12.npz")

case "$stage" in
  edaic-text)
    modalities=(embedding observable)
    models=("$QWEN_EMB" "$QWEN_LLM")
    "$PYTHON" -B round2_extract.py --modality "${modalities[$task]}" --model "${models[$task]}" \
      --manifest "$FMEM_EDAIC_MANIFEST" --outdir "$CACHE/round2_features"
    ;;
  edaic-prepare) "$PYTHON" -B edaic/prepare_timestamps.py ;;
  edaic-extract) "$PYTHON" -B edaic/extract.py --shard "$task" ;;
  edaic-evaluate) "$PYTHON" -B edaic/evaluate.py ;;
  edaic-baselines)
    models=(sega caf depmamba)
    seeds=(42 1337 2025)
    model="${models[$((task / 3))]}"
    seed="${seeds[$((task % 3))]}"
    "$PYTHON" -B train_official_multimodal.py --model "$model" --seed "$seed" \
      --manifest "$FMEM_EDAIC_MANIFEST" --text-npz "$CACHE/round2_features/embedding_clean_sentence3.npz" \
      --audio-npz "$FMEM_EDAIC_OUTPUT/features/participant3_matched.npz" \
      --outdir "$FMEM_EDAIC_OUTPUT/baselines/${model}_seed${seed}"
    ;;
  edaic-collect) "$PYTHON" -B edaic/collect.py ;;
  mtsp|pttsd)
    native="$CACHE/round3_native_features"
    if [[ "$stage" == mtsp ]]; then
      if [[ ! -f "$native/mentalbert_original_asr.npz" ]]; then
        "$PYTHON" -B native_mtsp.py --stage extract --manifest "$FMEM_EDAIC_MANIFEST" \
          --model_path "$MENTALBERT" --outdir "$native"
      fi
      "$PYTHON" -B native_mtsp.py --stage train --protocol official --manifest "$FMEM_EDAIC_MANIFEST" \
        --item_labels "$PHQ_ITEMS" --feature_dir "$native" --outdir "$OUT_SUPP/edaic/mtsp"
    else
      if [[ ! -f "$native/minilm_original_asr.npz" ]]; then
        "$PYTHON" -B native_pttsd.py --stage extract --manifest "$FMEM_EDAIC_MANIFEST" \
          --model_path_file "$MINILM_PATH_FILE" --outdir "$native"
      fi
      "$PYTHON" -B native_pttsd.py --stage train --protocol official --manifest "$FMEM_EDAIC_MANIFEST" \
        --feature_dir "$native" --outdir "$OUT_SUPP/edaic/pttsd"
    fi
    ;;
  eatd-evaluate)
    suites=(core indicators compositions robustness)
    suite="${suites[$task]}"
    "$PYTHON" -B fixed_fusion_validation.py --dataset eatd --suite "$suite" "${eatd[@]}" \
      --llm_npz "$LLM_OBS" --vocal_npz "$VOCAL_SRC" \
      --expanded_llm "$CACHE/expanded_features/observable16.npz" \
      --expanded_vocal "$CACHE/expanded_features/vocal12.npz" \
      --matched_audio "$CACHE/version4_matched_audio_features/matched_audio_eatd.npz" \
      --outdir "$OUT_SUPP/eatd/$suite"
    ;;
  eatd-deep)
    "$PYTHON" -B reference/Y_deep_architecture_baselines.py "${eatd[@]}" --protocol honest \
      --architectures eatd_gru_bilstm_style,modal_attention,lightweight_mult,misa_style \
      --text_pca 96 --audio_pca 96 --hidden 64 --dropout 0.30 \
      --lr 0.001 --weight_decay 0.0001 --batch_size 32 --max_epochs 100 \
      --patience 12 --inner_folds 3 --seed 42 --outdir "$OUT_SUPP/eatd/deep"
    ;;
  eatd-architecture)
    export FMEM_MATCHED_AUDIO="$CACHE/version4_matched_audio_features/matched_audio_eatd.npz"
    "$PYTHON" -B final_architecture_features.py "${eatd[@]}" --feature_set generic \
      --observable_npz "$LLM_OBS" --vocal_npz "$CACHE/expanded_features/vocal12.npz" \
      --outdir "$OUT_SUPP/eatd/architecture"
    ;;
  eatd-published)
    models=(sega sega caf)
    features=(native qwen qwen)
    model="${models[$task]}"
    feature="${features[$task]}"
    "$PYTHON" -B train_published.py --dataset eatd --model "$model" --features "$feature" "${eatd[@]}" \
      --seed 42 --author_npz "$SEGA_FEATURES" --outdir "$OUT_SUPP/eatd/${model}_$feature"
    ;;
  eatd-bootstrap) "$PYTHON" -B final_paired_f1.py ;;
  eatd2-evaluate)
    "$PYTHON" -B eatd2.py "${source_args[@]}" --target_audio "$TEST_AUDIO" --target_text "$TGT_TEXT" \
      --target_llm "$TGT_LLM" --target_root "$TEST_DATA_ROOT" \
      --target_vocal12 "$CACHE/final_target_vocal12/vocal12.npz" --outdir "$OUT_SUPP/eatd2/fmem"
    ;;
  eatd2-baselines)
    models=(eatd_gru_bilstm_style lightweight_mult misa_style sega caf)
    model="${models[$task]}"
    "$PYTHON" -B train_source_transfer.py --model "$model" --source-audio "$TRAIN_AUDIO" \
      --source-text "$BEST_TEXT" --source-root "$TRAIN_DATA_ROOT" \
      --matched-source-audio "$CACHE/version4_matched_audio_features/matched_audio_eatd.npz" \
      --matched-target-audio "$CACHE/version4_matched_audio_features/matched_audio_eatd2.npz" \
      --target-audio "$TEST_AUDIO" --target-text "$TGT_TEXT" --target-root "$TEST_DATA_ROOT" \
      --outdir "$OUT_SUPP/eatd2/$model"
    ;;
  threshold-audit)
    "$PYTHON" -B threshold_audit.py "${source_args[@]}" --fold_json "$FOLD_JSON" \
      --source_results "$CACHE" --eatd2_results "$OUT_SUPP/eatd2/fmem" --out "$OUT_SUPP/threshold"
    ;;
  expand-indicators)
    if [[ "$task" == 0 ]]; then
      "$PYTHON" -B extract_expanded.py --modality text --model "$QWEN_LLM" --base_npz "$LLM_OBS" \
        --data_root "$TRAIN_DATA_ROOT" --outdir "$CACHE/expanded_features"
    else
      "$PYTHON" -B extract_expanded.py --modality audio --model "$AUDIO_LLM" --base_npz "$VOCAL_SRC" \
        --data_root "$TRAIN_DATA_ROOT" --outdir "$CACHE/expanded_features"
    fi
    ;;
  eatd2-vocal)
    "$PYTHON" -B extract_target_vocal12.py --model "$AUDIO_LLM" --base_npz "$VOCAL_TGT" \
      --source_vocal12 "$CACHE/expanded_features/vocal12.npz" \
      --data_root "$TEST_DATA_ROOT" --outdir "$CACHE/final_target_vocal12"
    ;;
  eatd-matched-audio)
    "$PYTHON" -B extract_matched_audio.py --source_vocal "$VOCAL_SRC" --target_vocal "$VOCAL_TGT" \
      --model "$AUDIO_LLM" --outdir "$CACHE/version4_matched_audio_features"
    ;;
  *) printf 'Unknown experiment: %s\n' "$stage" >&2; exit 2 ;;
esac
