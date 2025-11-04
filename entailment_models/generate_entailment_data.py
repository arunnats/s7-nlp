#!/usr/bin/env python3
"""
Generate entailment training data for LegalSumm
Creates positive (fact) and negative (fake) chunk-summary pairs
"""

import json
import random
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Configuration
NUM_FAKE_PER_REAL = 10  # Paper uses 10 fake examples per real one
STRATEGIES = 8
DATA_DIR = Path("output")
OUTPUT_DIR = Path("entailment_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_train_data(strategy_num: int) -> List[Dict]:
    """Load training chunks and summaries for a strategy"""
    src_file = DATA_DIR / f"strategy_{strategy_num}" / "train.src.txt"
    tgt_file = DATA_DIR / f"strategy_{strategy_num}" / "train.tgt.txt"
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        chunks = [line.strip() for line in f_src]
        summaries = [line.strip() for line in f_tgt]
    
    return [{"chunk": c, "summary": s} for c, s in zip(chunks, summaries)]

def extract_category_from_summary(summary: str) -> str:
    """
    Extract category label from summary
    In legal summaries, categories are often in the first part
    We use the first 3 words as a proxy for category
    """
    words = summary.split()[:3]
    return " ".join(words).lower()

def group_by_category(data: List[Dict]) -> Dict[str, List[int]]:
    """Group document indices by their category"""
    categories = defaultdict(list)
    for idx, item in enumerate(data):
        category = extract_category_from_summary(item["summary"])
        categories[category].append(idx)
    return categories

def find_related_categories(target_category: str, all_categories: List[str]) -> List[str]:
    """
    Find categories related to target by checking bigram overlap
    Paper: categories that share word bigrams are considered related
    """
    target_bigrams = set(zip(target_category.split()[:-1], 
                             target_category.split()[1:]))
    related = []
    
    for cat in all_categories:
        if cat == target_category:
            continue
        cat_bigrams = set(zip(cat.split()[:-1], cat.split()[1:]))
        if target_bigrams & cat_bigrams:  # If any bigrams overlap
            related.append(cat)
    
    return related if related else all_categories[:10]  # Fallback to first 10

def generate_fake_examples(idx: int, data: List[Dict], 
                          category_groups: Dict[str, List[int]],
                          num_fakes: int = 10) -> List[Dict]:
    """
    Generate fake chunk-summary pairs using related category summaries
    """
    chunk = data[idx]["chunk"]
    category = extract_category_from_summary(data[idx]["summary"])
    
    # Get related categories
    related_cats = find_related_categories(category, list(category_groups.keys()))
    
    # Collect candidate fake summaries from related categories
    candidate_indices = []
    for cat in related_cats:
        candidate_indices.extend(category_groups[cat])
    
    # Remove the current document index
    candidate_indices = [i for i in candidate_indices if i != idx]
    
    # If not enough candidates, use all documents
    if len(candidate_indices) < num_fakes:
        candidate_indices = [i for i in range(len(data)) if i != idx]
    
    # Randomly select fake summaries
    fake_indices = random.sample(candidate_indices, min(num_fakes, len(candidate_indices)))
    
    fake_examples = []
    for fake_idx in fake_indices:
        fake_examples.append({
            "chunk": chunk,
            "summary": data[fake_idx]["summary"],
            "label": 0  # Fake
        })
    
    return fake_examples

def generate_entailment_data_for_strategy(strategy_num: int):
    """Generate entailment training data for one strategy"""
    print(f"\nProcessing Strategy {strategy_num}...")
    
    # Load training data
    data = load_train_data(strategy_num)
    print(f"  Loaded {len(data)} training examples")
    
    # Group by categories
    category_groups = group_by_category(data)
    print(f"  Found {len(category_groups)} categories")
    
    # Generate entailment pairs
    entailment_data = []
    
    for idx in range(len(data)):
        # Add real (fact) example
        entailment_data.append({
            "chunk": data[idx]["chunk"],
            "summary": data[idx]["summary"],
            "label": 1  # Fact
        })
        
        # Add fake examples
        fakes = generate_fake_examples(idx, data, category_groups, NUM_FAKE_PER_REAL)
        entailment_data.extend(fakes)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(data)} documents...")
    
    # Save to CSV
    output_file = OUTPUT_DIR / f"entailment_train_strategy_{strategy_num}.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk', 'summary', 'label'])
        writer.writeheader()
        writer.writerows(entailment_data)
    
    print(f"  Saved {len(entailment_data)} examples to {output_file}")
    print(f"  Ratio - Fact: {sum(1 for e in entailment_data if e['label'] == 1)}, "
          f"Fake: {sum(1 for e in entailment_data if e['label'] == 0)}")

def main():
    print("=" * 60)
    print("Generating Entailment Training Data for LegalSumm")
    print("=" * 60)
    
    random.seed(42)  # For reproducibility
    
    for strategy in range(1, STRATEGIES + 1):
        generate_entailment_data_for_strategy(strategy)
    
    print("\n" + "=" * 60)
    print("Done! Entailment data saved in:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()
