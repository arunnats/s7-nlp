#!/usr/bin/env python3
"""
Simple chunking utility for RulingBR-style documents.

This script reads legal documents in JSON or JSONL format, applies 8 different
chunking strategies as defined in the source paper (likely for abstractive
summarization research), and writes the output directly into an OpenNMT-py
compatible directory structure. This avoids manual data formatting and prepares
the data for training the summarization models.

Usage:
  python chunker.py --input train_data.jsonl --out data_for_opennmt --prefix train
  python chunker.py --input valid_data.jsonl --out data_for_opennmt --prefix valid --strategy 3
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any, TextIO
from multiprocessing import Pool, cpu_count

# --- Optional Dependency Imports ---
# Check for NLTK for advanced tokenization
try:
    import nltk
    _HAS_NLTK = True
    print("NLTK found, will use for tokenization.")
except ImportError:
    _HAS_NLTK = False
    print("NLTK not found, falling back to simple whitespace tokenization.")

# Check for Hugging Face Transformers for model-specific tokenization
try:
    import transformers
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# --- Tokenization Helpers ---

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizes text, preferring NLTK's word_tokenize, falling back to
    whitespace splitting if NLTK fails or is not installed.
    """
    if not text:
        return []

    if _HAS_NLTK:
        try:
            # Use NLTK's more sophisticated word tokenizer
            return nltk.word_tokenize(text)
        except Exception:
            # Fallback for unexpected NLTK errors
            return text.split()

    # Simple whitespace split if NLTK is unavailable
    return text.split()


def hf_tokenizer_factory(model_name: str):
    """
    Creates a function that uses a Hugging Face pre-trained tokenizer
    for tokenizing text, which is required for strategies that need
    model-specific token counts.
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("The 'transformers' library is not installed. Please install it to use Hugging Face tokenizers.")

    # Import inside the function to keep it contained to this context
    from transformers import AutoTokenizer
    # Load the specified pre-trained tokenizer (e.g., BERT, RoBERTa)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tokenize(text: str) -> List[str]:
        """The actual tokenization function returned by the factory."""
        if not text:
            return []
        # Encode the text, avoiding special tokens ([CLS], [SEP])
        enc = tok.encode(text, add_special_tokens=False)
        # Convert token IDs back to human-readable tokens/subwords
        return tok.convert_ids_to_tokens(enc)
    return _tokenize


def detokenize(tokens: List[str]) -> str:
    """
    Joins a list of tokens back into a single string, using a space.
    This is a simple detokenizer for the resulting chunks.
    """
    return " ".join(tokens)


# --- Chunking Logic ---

def take_prefix_tokens(text: str, n: int, tokenizer=None) -> str:
    """
    Takes the first 'n' tokens of the input text using the specified
    or default tokenizer, and returns them as a detokenized string.
    """
    tokenize_func = tokenizer if tokenizer else simple_tokenize
    tokens = tokenize_func(text)
    # Truncate the list of tokens to the first 'n'
    return detokenize(tokens[:n])


def concat_and_truncate(parts: List[str], max_tokens: int, tokenizer=None) -> str:
    """
    Concatenates content from a list of text parts, tokenizes them, and
    truncates the result to a maximum of 'max_tokens'.
    """
    all_tokens: List[str] = []
    tokenize_func = tokenizer if tokenizer else simple_tokenize

    for p in parts:
        if not p:
            continue
        # Add tokens from the current part to the main list
        all_tokens.extend(tokenize_func(p))
        # Stop processing parts if the maximum token limit is reached
        if len(all_tokens) >= max_tokens:
            break

    # Detokenize the concatenated and truncated list of tokens
    return detokenize(all_tokens[:max_tokens])


def chunk_document(doc: Dict[str, Any], tokenizer=None) -> List[str]:
    """
    Applies the 8 specific chunking strategies to a single legal document.
    The strategies are based on token limits and the order/combination
    of document parts (Report, Vote, Judgment/Acordao).
    """
    # Standardize access to document parts, accounting for possible key variations
    report = doc.get("relatorio", "") or doc.get("relatorio_text", "") or ""
    vote = doc.get("voto", "") or doc.get("voto_text", "") or ""
    judgment = doc.get("acordao", "") or doc.get("acordao_text", "") or ""

    # Convenience functions to apply the chosen tokenizer
    use_prefix = lambda text, n: take_prefix_tokens(text, n, tokenizer)
    use_concat = lambda parts, n: concat_and_truncate(parts, n, tokenizer)

    # --- 8 Chunking Strategies (Based on the RulingBR paper) ---

    # S1: Prefix of Report (300 tokens) + Prefix of Judgment (100 tokens)
    s1 = use_prefix(report, 300) + " " + use_prefix(judgment, 100)

    # S2: Prefix of Vote (300 tokens) + Prefix of Judgment (100 tokens)
    s2 = use_prefix(vote, 300) + " " + use_prefix(judgment, 100)

    # S3: Prefix of Report (150) + Prefix of Vote (150) + Prefix of Judgment (100)
    s3 = use_prefix(report, 150) + " " + use_prefix(vote, 150) + " " + use_prefix(judgment, 100)

    # S4: Prefix of Report only (400 tokens)
    s4 = use_prefix(report, 400)

    # S5: Prefix of Vote only (400 tokens)
    s5 = use_prefix(vote, 400)

    # S6: Concatenate (Report, Vote, Judgment), truncate to 400 tokens
    s6 = use_concat([report, vote, judgment], 400)

    # S7: Concatenate (Vote, Judgment, Report), truncate to 400 tokens
    s7 = use_concat([vote, judgment, report], 400)

    # S8: Concatenate (Judgment, Report, Vote), truncate to 400 tokens
    s8 = use_concat([judgment, report, vote], 400)

    # Return all 8 generated chunks, stripped of leading/trailing whitespace
    return [s1.strip(), s2.strip(), s3.strip(), s4.strip(), s5.strip(), s6.strip(), s7.strip(), s8.strip()]


# --- File I/O ---

def read_input(path: str) -> List[Dict[str, Any]]:
    """
    Reads the input file. Supports both a single JSON object containing
    a list of documents, or a JSONL (JSON Lines) file.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []

        lines = text.splitlines()
        # Check if it looks like a JSONL file (multiple lines, each starting with '{')
        if len(lines) > 1 and all(l.strip().startswith('{') for l in lines if l.strip()):
            return [json.loads(l) for l in lines if l.strip()]

        # Otherwise, treat it as a single JSON object (which may be a list)
        obj = json.loads(text)
        # Ensure the result is always a list of documents
        return obj if isinstance(obj, list) else [obj]


def save_openmt_data(docs: List[Dict[str, Any]], output_dir: str, prefix: str, strategy: int = 0):
    """
    Writes the source chunks and target summaries (ementa) to separate files
    for each specified chunking strategy, formatted for OpenNMT-py.

    - strategy=0 means process ALL 8 strategies.
    - strategy=N means process only strategy N.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Determine which strategies to process
    strategies_to_process = range(1, 9) if strategy == 0 else [strategy]

    # Dictionaries to hold file handles for source (.src.txt) and target (.tgt.txt)
    src_files: Dict[int, TextIO] = {}
    tgt_files: Dict[int, TextIO] = {}

    # Open files for all selected strategies
    for i in strategies_to_process:
        strat_dir = os.path.join(output_dir, f"strategy_{i}")
        os.makedirs(strat_dir, exist_ok=True) # Create subdirectory for each strategy
        src_path = os.path.join(strat_dir, f"{prefix}.src.txt")
        tgt_path = os.path.join(strat_dir, f"{prefix}.tgt.txt")
        src_files[i] = open(src_path, "w", encoding="utf-8")
        tgt_files[i] = open(tgt_path, "w", encoding="utf-8")

    # Write data document by document
    for doc in docs:
        # Get the target summary (ementa), cleaning up newlines
        summary = (doc.get("summary") or doc.get("ementa", "")).replace("\n", " ").strip()
        chunks = doc.get("chunks", []) # The chunks were added in the parallel step

        for i in strategies_to_process:
            # Get the chunk corresponding to the current strategy index (1-based)
            chunk_text = chunks[i - 1].strip() if len(chunks) >= i else ""

            # Write one line each to the source (chunk) and target (summary) files
            src_files[i].write(chunk_text + "\n")
            tgt_files[i].write(summary + "\n")

    # Close all open file handles
    for i in strategies_to_process:
        src_files[i].close()
        tgt_files[i].close()
        print(f"[LOG] Closed files for strategy {i}")

    print(f"✅ Wrote '{prefix}' data for strategies {list(strategies_to_process)} to '{output_dir}'.")


# --- Parallel Worker ---

def _process_doc(args):
    """
    Worker function for the multiprocessing Pool.
    It takes a document and the tokenizer model name, instantiates the tokenizer
    if a model name is provided, chunks the document, and returns the result.
    """
    doc, model_name = args
    
    tok = None
    if model_name and _HF_AVAILABLE:
        # NOTE: hf_tokenizer_factory will handle the actual loading
        # This function is defined at the top-level and is pickle-able
        tok = hf_tokenizer_factory(model_name)
        
    doc["chunks"] = chunk_document(doc, tokenizer=tok)
    return doc

# --- Main Entry ---

def main():
    """Main function to parse arguments, load data, run parallel chunking, and save results."""
    parser = argparse.ArgumentParser(description="Chunk RulingBR documents into OpenNMT-py format")
    parser.add_argument("--input", required=True, help="Path to the input JSON or JSONL file.")
    parser.add_argument("--out", required=True, help="Output directory for OpenNMT-py files.")
    parser.add_argument("--prefix", type=str, default="train", help="Prefix for output files (e.g., 'train', 'valid').")
    parser.add_argument("--strategy", type=int, choices=list(range(1, 9)) + [0], default=0, help="Chunking strategy (1-8) to use. 0 to use all.")
    parser.add_argument("--tokenizer-model", type=str, default="bert-base-multilingual-cased", help="Hugging Face model name for tokenization (if used).")
    parser.add_argument("--nrows", type=int, default=0, help="Number of documents to process. 0 means all.")
    args = parser.parse_args()

    print("[LOG] Starting chunking process...")
    docs = read_input(args.input)
    n = args.nrows or len(docs)
    print(f"[LOG] Loaded {len(docs)} documents. Processing first {n}...")

    # Display message if HF is used but unavailable
    if args.tokenizer_model and not _HF_AVAILABLE:
        print("WARNING: transformers not installed. Using basic tokenization.")
        
    # Set the model name to pass to workers
    model_to_use = args.tokenizer_model if _HF_AVAILABLE else None

    # --- Parallel Processing ---
    # Use all but one CPU core for parallel chunking
    num_cores = max(1, cpu_count() - 1)
    print(f"[LOG] Generating chunks using {num_cores} CPU cores...")

    # Prepare iterable for the pool worker: a list of (document, tokenizer) tuples
    pool_data = [(d, model_to_use) for d in docs[:n]]

    with Pool(processes=num_cores) as pool:
        processed_docs = []
        for idx, doc in enumerate(pool.imap(_process_doc, pool_data, chunksize=20)):
            if idx % 100 == 0 and idx > 0:
                print(f"[LOG] Processed {idx}/{n} documents...")
            processed_docs.append(doc)

    print("[LOG] Parallel chunk generation complete.")

    # --- Save Results ---
    save_openmt_data(processed_docs, args.out, args.prefix, args.strategy)
    print("[LOG] ✅ All processing completed successfully.")


if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()