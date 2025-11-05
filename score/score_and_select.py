#!/usr/bin/env python3
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys

STRATEGIES = 8
DATA_DIR = "output"
SUMMARIES_DIR = "test_summaries"
ENTAILMENT_DIR = "models/entailment"
OUTPUT_FILE = "final_predictions.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", file=sys.stderr, flush=True)

# Load test chunks
with open(f"{DATA_DIR}/strategy_1/test.src.txt", 'r', encoding='utf-8') as f:
    test_chunks = [line.strip() for line in f]

print(f"Loaded {len(test_chunks)} test documents", file=sys.stderr, flush=True)

# Load all summaries
all_summaries = []
for strategy in range(1, STRATEGIES + 1):
    with open(f"{SUMMARIES_DIR}/summaries_strategy_{strategy}.txt", 'r', encoding='utf-8') as f:
        all_summaries.append([line.strip() for line in f])
    print(f"Loaded strategy {strategy} summaries", file=sys.stderr, flush=True)

# Pre-load all models
models = {}
tokenizers = {}
for strategy in range(1, STRATEGIES + 1):
    model_dir = f"{ENTAILMENT_DIR}/strategy_{strategy}"
    tokenizers[strategy] = BertTokenizer.from_pretrained(model_dir)
    models[strategy] = BertForSequenceClassification.from_pretrained(model_dir)
    models[strategy].to(device)
    models[strategy].eval()
    print(f"Loaded model for strategy {strategy}", file=sys.stderr, flush=True)

final_summaries = []

for doc_idx in range(len(test_chunks)):
    chunk = test_chunks[doc_idx]
    best_score = -1
    best_summary = ""
    best_strategy = -1

    # Score each candidate
    for strategy in range(1, STRATEGIES + 1):
        summary = all_summaries[strategy - 1][doc_idx]

        if summary:
            tokenizer = tokenizers[strategy]
            model = models[strategy]

            inputs = tokenizer(
                chunk,
                summary,
                max_length=512,
                truncation=True,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0, 1].item()

            if score > best_score:
                best_score = score
                best_summary = summary
                best_strategy = strategy

    final_summaries.append(best_summary)

    if (doc_idx + 1) % 50 == 0:
        print(f"Processed {doc_idx + 1}/{len(test_chunks)} | Strategy {best_strategy} (score: {best_score:.4f})", file=sys.stderr, flush=True)

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for summary in final_summaries:
        f.write(summary + "\n")

print(f"✓ Final predictions saved to {OUTPUT_FILE}", file=sys.stderr, flush=True)