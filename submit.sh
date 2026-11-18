#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
stage="${1:?Pass an experiment name}"
shift
time=1-12:00:00
mem=96G
array=()
case "$stage" in
  edaic-text) array=(--array=0-1%2); time=1-00:00:00 ;;
  edaic-prepare|edaic-evaluate|edaic-collect|eatd2-evaluate|eatd-bootstrap)
    time=1-00:00:00 ;;
  threshold-audit) time=00:12:00 ;;
  edaic-extract) array=(--array=0-3%2) ;;
  edaic-baselines) array=(--array=0-8%2) ;;
  mtsp|pttsd|eatd-architecture) ;;
  eatd-evaluate) array=(--array=0-3%1) ;;
  eatd-deep) time=48:00:00; mem=120G ;;
  eatd-published) array=(--array=0-2%3) ;;
  eatd2-baselines) array=(--array=0-4%4); time=1-00:00:00 ;;
  expand-indicators) array=(--array=0-1%2); time=1-00:00:00 ;;
  eatd2-vocal|eatd-matched-audio) time=1-00:00:00 ;;
  *) printf 'Unknown experiment: %s\n' "$stage" >&2; exit 2 ;;
esac
printf -v job 'bash %q %q' "$PWD/run.sh" "$stage"
log_dir="${FMEM_LOG_DIR:-$PWD/logs}"
command=(sbatch --parsable --partition=L40 --gres=gpu:l40:1 --cpus-per-task=6 --mem="$mem"
         --time="$time" --job-name="fmem-$stage" --chdir="$PWD"
         --output="$log_dir/$stage-%A_%a.out" --error="$log_dir/$stage-%A_%a.err" "${array[@]}")
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "${command[@]}" "$@" --wrap="$job"
  printf '\n'
else
  mkdir -p "$log_dir"
  "${command[@]}" "$@" --wrap="$job"
fi
