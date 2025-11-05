#!/usr/bin/env python3
"""
Evaluate baseline strategies individually to compare against ensemble
"""

from rouge_score import rouge_scorer
import numpy as np

def evaluate_strategy(strategy_num, predictions_file=None):
    """Evaluate one strategy"""

    # If no file provided, generate summaries from test output
    if predictions_file is None:
        # Just use first summary from each strategy (the direct output)
        try:
            with open(f"test_summaries/summaries_strategy_{strategy_num}.txt", 'r', encoding='utf-8') as f:
                predictions = [line.strip() for line in f]
        except:
            print(f"Error: Could not find test_summaries/summaries_strategy_{strategy_num}.txt")
            return None
    else:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = [line.strip() for line in f]

    # Load reference summaries
    with open("output/strategy_1/test.tgt.txt", 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    if len(predictions) != len(references):
        print(f"Warning: {len(predictions)} predictions vs {len(references)} references")

    # Calculate ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_f, rouge2_f, rougeL_f = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(rouge1_f),
        'rouge2': np.mean(rouge2_f),
        'rougeL': np.mean(rougeL_f),
    }

def main():
    print("="*60)
    print("Baseline Evaluation - Individual Strategies")
    print("="*60)

    # Evaluate each strategy individually
    results = {}
    for strategy in range(1, 9):
        scores = evaluate_strategy(strategy)
        if scores:
            results[strategy] = scores
            print(f"\nStrategy {strategy}:")
            print(f"  ROUGE-1: {scores['rouge1']:.4f}")
            print(f"  ROUGE-2: {scores['rouge2']:.4f}")
            print(f"  ROUGE-L: {scores['rougeL']:.4f}")

    # Ensemble (your final predictions)
    print("\n" + "="*60)
    print("Ensemble (Entailment Selection)")
    print("="*60)
    ensemble = evaluate_strategy(None, "final_predictions.txt")
    if ensemble:
        print(f"\n  ROUGE-1: {ensemble['rouge1']:.4f}")
        print(f"  ROUGE-2: {ensemble['rouge2']:.4f}")
        print(f"  ROUGE-L: {ensemble['rougeL']:.4f}")

    # Comparison table
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Method':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    print("-" * 56)

    for strategy in range(1, 9):
        if strategy in results:
            r = results[strategy]
            print(f"Strategy {strategy:<13} {r['rouge1']:<12.4f} {r['rouge2']:<12.4f} {r['rougeL']:<12.4f}")

    if ensemble:
        print(f"{'Ensemble':<20} {ensemble['rouge1']:<12.4f} {ensemble['rouge2']:<12.4f} {ensemble['rougeL']:<12.4f}")

        # Calculate improvement
        best_single = max(results[i]['rouge1'] for i in range(1, 9))
        improvement = ((ensemble['rouge1'] - best_single) / best_single) * 100

        print("\n" + "="*60)
        print(f"Improvement over best single strategy: {improvement:+.2f}%")
        print("="*60)

if __name__ == "__main__":
    main()