#!/usr/bin/env python3
"""Calculate ROUGE scores"""

from rouge_score import rouge_scorer
import numpy as np

def main():
    # Load predictions and references
    with open("final_predictions.txt", 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]

    with open("output/strategy_1/test.tgt.txt", 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate scores
    rouge1_f, rouge2_f, rougeL_f = [], [], []
    rouge1_p, rouge2_p, rougeL_p = [], [], []
    rouge1_r, rouge2_r, rougeL_r = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)

        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge1_p.append(scores['rouge1'].precision)
        rouge1_r.append(scores['rouge1'].recall)

        rouge2_f.append(scores['rouge2'].fmeasure)
        rouge2_p.append(scores['rouge2'].precision)
        rouge2_r.append(scores['rouge2'].recall)

        rougeL_f.append(scores['rougeL'].fmeasure)
        rougeL_p.append(scores['rougeL'].precision)
        rougeL_r.append(scores['rougeL'].recall)

    # Print results
    print("="*60)
    print("ROUGE Evaluation Results")
    print("="*60)
    print(f"\nROUGE-1:")
    print(f"  Precision: {np.mean(rouge1_p):.4f}")
    print(f"  Recall:    {np.mean(rouge1_r):.4f}")
    print(f"  F1:        {np.mean(rouge1_f):.4f}")

    print(f"\nROUGE-2:")
    print(f"  Precision: {np.mean(rouge2_p):.4f}")
    print(f"  Recall:    {np.mean(rouge2_r):.4f}")
    print(f"  F1:        {np.mean(rouge2_f):.4f}")

    print(f"\nROUGE-L:")
    print(f"  Precision: {np.mean(rougeL_p):.4f}")
    print(f"  Recall:    {np.mean(rougeL_r):.4f}")
    print(f"  F1:        {np.mean(rougeL_f):.4f}")

    print("="*60)

if __name__ == "__main__":
    main()