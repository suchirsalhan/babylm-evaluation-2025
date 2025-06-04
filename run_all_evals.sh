#!/bin/bash

models=(
  "babylm-seqlen/mamba-8192"
  "babylm-seqlen/mamba-8192-warmup"
  "babylm-seqlen/mamba-4096-warmup"
  "babylm-seqlen/mamba-2048-warmup"
  "babylm-seqlen/mamba-512-warmup"
  "babylm-seqlen/mamba-1024-warmup"
  "babylm-seqlen/mamba-64-warmup"
  "babylm-seqlen/mamba-128-warmup"
  "babylm-seqlen/mamba-256-warmup"
  "babylm-seqlen/opt-2048"
  "babylm-seqlen/mamba-2048"
  "babylm-seqlen/opt-256"
  "babylm-seqlen/opt-512"
  "babylm-seqlen/opt-1024"
  "babylm-seqlen/mamba-1024"
  "babylm-seqlen/mamba-512"
  "babylm-seqlen/mamba-256"
  "babylm-seqlen/mamba-128"
  "babylm-seqlen/mamba-64"
  "babylm-seqlen/opt-128"
  "babylm-seqlen/opt-64"
)

for model in "${models[@]}"
do
  echo "Running evaluation for $model"
  ./eval_zero_shot.sh "$model" causal
done
