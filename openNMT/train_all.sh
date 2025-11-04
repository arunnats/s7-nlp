#!/bin/bash

DATA_DIR="output"
MODEL_DIR="models/summarizers"
CONFIG_TEMPLATE="transformer_config.yaml"

mkdir -p $MODEL_DIR

for i in {1..8}
do
  echo "#####################################################"
  echo "###   STARTING TRAINING FOR STRATEGY $i   ###"
  echo "#####################################################"

  # Create a temporary config for this strategy
  CONFIG_FILE="transformer_config_strategy_$i.yaml"

  # Copy the template and update the data paths
  sed "s|strategy_1|strategy_$i|g" $CONFIG_TEMPLATE > $CONFIG_FILE

  SAVE_MODEL="$MODEL_DIR/summarizer_strategy_$i"

  onmt_train \
    -config $CONFIG_FILE \
    -save_model $SAVE_MODEL \
    -world_size 1 \
    -gpu_ranks 0

  echo "--- Finished training for Strategy $i ---"
  rm $CONFIG_FILE
done

echo "✅ All 8 summarization models have been trained."