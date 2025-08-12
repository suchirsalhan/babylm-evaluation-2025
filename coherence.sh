#!/bin/bash

models=(
  "suchirsalhan/coh1em02"
  "suchirsalhan/coh1em03"
)

tasks=(
  "causal"
  "mlm"
)

for model in "${models[@]}"
do
  for task in "${tasks[@]}"
  do
    echo "Running evaluation for $model on task $task"
    ./eval_zero_shot.sh "$model" "$task"
  done
done
