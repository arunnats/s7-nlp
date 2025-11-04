# Vocabulary

## Generate with size limits and explicit parameters
onmt_build_vocab \
  -config transformer_config.yaml \
  -n_sample -1 \
  -src_vocab_size 50000 \
  -tgt_vocab_size 50000 \
  -src_words_min_frequency 0 \
  -tgt_words_min_frequency 0
  -save_data onmt_data/processed_data

## Check the last 10 lines to ensure they're properly formatted
tail -20 onmt_data/shared_vocabulary.vocab

## Find lines that don't match the pattern "word<TAB>number"
awk -F'\t' 'NF!=2 || $2 !~ /^[0-9]+$/ {print NR": "$0}' onmt_data/shared_vocabulary.vocab

## Remove any blank lines and ensure proper format
grep -P '^\S+\t\d+$' onmt_data/shared_vocabulary.vocab > onmt_data/shared_vocabulary_clean.vocab

## Replace the old file
mv onmt_data/shared_vocabulary_clean.vocab onmt_data/shared_vocabulary.vocab

## Run train_alll
./train_all.sh

# Docker Control

## Run docker detached

docker run --gpus all -d \
  --name nlp_training \
  --user $(id -u):$(id -g) \
  -v /home/b220019cs:/workspace \
  gpu-workspace \
  bash -c "cd /workspace/nlp && ./train_all.sh > training.log 2>&1"

## Check if container is running
docker ps

## See the live logs (first few seconds)
docker logs -f nlp_training

# Interactive Shell

## Jump into the running container
docker exec -it nlp_training bash

## Now you're inside! Check status:
cd nlp
tail -100 training.log

## Check which models have been saved
ls -lh models/summarizers/

## Exit when done (container keeps running)
exit

## From the server (no Docker needed)
nvidia-smi

## Stop the training (if needed)
docker stop nlp_training

## Restart it
docker start nlp_training

## Remove the container (only after it's done)
docker rm nlp_training

## View real-time progress in the log file
docker exec nlp_training tail -f /workspace/nlp/training.log

# Enteilment

## Check Logs
docker logs entailment_training | tail -50

## run entailment
docker run --gpus all -d \
  --name entailment_training \
  --user $(id -u):$(id -g) \
  -v /home/b220019cs:/workspace \
  gpu-workspace \
  bash -c "cd /workspace/nlp && python3 train_entailment_models.py 2>&1 | tee entailment_training.log"
