#!/usr/bin/env python3
"""
Simple chunking utility for RulingBR-style documents.

This script reads legal documents in JSON or JSONL format, applies 8 different
chunking strategies as defined in the source paper, and writes the output
directly into an OpenNMT-py compatible directory structure. This avoids
manual data formatting and prepares the data for training the summarization models.

Usage:
  # Process all 8 strategies for the training set into the 'data_for_opennmt' directory
  python this_script.py --input train_data.jsonl --out data_for_opennmt --prefix train

  # Process only strategy 3 for the validation set
  python this_script.py --input valid_data.jsonl --out data_for_opennmt --prefix valid --strategy 3
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any, TextIO
from multiprocessing import Pool, cpu_count

# --- Optional Dependency Imports ---
# These try-except blocks check if optional libraries are installed without crashing
# the script if they are not.

try:
    # NLTK provides a more reliable tokenizer than splitting by whitespace.
    import nltk
    _HAS_NLTK = True
    print("NLTK found, will use for tokenization.")
except ImportError:
    _HAS_NLTK = False
    print("NLTK not found, falling back to simple whitespace tokenization.")

try:
    # Transformers library is needed for using Hugging Face tokenizers.
    import transformers
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# --- Tokenization Helper Functions ---

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizes a string into a list of words.
    - Prefers NLTK's `word_tokenize` if available.
    - Falls back to splitting the string by whitespace if NLTK is not installed.
    """
    if not text:
        return []
    if _HAS_NLTK:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            # Fallback in case NLTK has an issue (e.g., missing data)
            return text.split()
    return text.split()

def hf_tokenizer_factory(model_name: str):
    """
    Creates and returns a tokenizer function powered by the Hugging Face Transformers library.
    This is used for accurately counting tokens according to a specific model (like BERT).
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("The 'transformers' library is not installed. Please install it to use Hugging Face tokenizers.")
    
    from transformers import AutoTokenizer
    # Load the specified tokenizer model. 'use_fast=True' loads the faster Rust-based version.
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tokenize(text: str) -> List[str]:
        """The actual tokenizer function that will be returned."""
        if not text:
            return []
        # Encode the text to get token IDs, then convert them back to token strings.
        # This gives us the subword units (e.g., 'tokenization' -> ['token', '##ization']).
        enc = tok.encode(text, add_special_tokens=False)
        return tok.convert_ids_to_tokens(enc)

    return _tokenize

def detokenize(tokens: List[str]) -> str:
    """A simple helper to join a list of token strings back into a single string."""
    return " ".join(tokens)

# --- Chunking Logic Functions ---

def take_prefix_tokens(text: str, n: int, tokenizer=None) -> str:
    """Truncates text to the first `n` tokens."""
    # Use the provided tokenizer function if available, otherwise use the simple one.
    tokenize_func = tokenizer if tokenizer else simple_tokenize
    tokens = tokenize_func(text)
    return detokenize(tokens[:n])

def concat_and_truncate(parts: List[str], max_tokens: int, tokenizer=None) -> str:
    """Concatenates multiple text parts, then truncates the result to `max_tokens`."""
    all_tokens: List[str] = []
    tokenize_func = tokenizer if tokenizer else simple_tokenize
    for p in parts:
        if not p:
            continue
        all_tokens.extend(tokenize_func(p))
        # Stop adding parts if we've already reached the token limit.
        if len(all_tokens) >= max_tokens:
            break
    return detokenize(all_tokens[:max_tokens])

def chunk_document(doc: Dict[str, Any], tokenizer=None) -> List[str]:
    """
    Applies the 8 chunking strategies from the paper to a single document.
    Returns a list of 8 strings, where each string is a chunk.
    """
    # Safely get text from the document, with fallbacks for different key names.
    report = doc.get("relatorio", "") or doc.get("relatorio_text", "") or ""
    vote = doc.get("voto", "") or doc.get("voto_text", "") or ""
    judgment = doc.get("acordao", "") or doc.get("acordao_text", "") or ""

    # Define shortcuts for the two main chunking operations.
    use_prefix = lambda text, n: take_prefix_tokens(text, n, tokenizer)
    use_concat = lambda parts, n: concat_and_truncate(parts, n, tokenizer)

    print("[LOG] Generating 8 chunking strategies for current document...")

    # --- Define the 8 Strategies ---
    # Strategies 1-5: Truncate individual sections then combine them.
    s1 = use_prefix(report, 300) + " " + use_prefix(judgment, 100)
    s2 = use_prefix(vote, 300) + " " + use_prefix(judgment, 100)
    s3 = use_prefix(report, 150) + " " + use_prefix(vote, 150) + " " + use_prefix(judgment, 100)
    s4 = use_prefix(report, 400)
    s5 = use_prefix(vote, 400)
    
    # Strategies 6-8: Combine full sections first, then truncate the combined text.
    s6 = use_concat([report, vote, judgment], 400)
    s7 = use_concat([vote, judgment, report], 400)
    s8 = use_concat([judgment, report, vote], 400)

    print("[LOG] Finished generating 8 strategies for document.")

    # Return a list containing the 8 generated chunks.
    return [s1.strip(), s2.strip(), s3.strip(), s4.strip(), s5.strip(), s6.strip(), s7.strip(), s8.strip()]

# --- File I/O Functions ---

def read_input(path: str) -> List[Dict[str, Any]]:
    """
    Reads an input file that can be either a single JSON array or a JSONL file.
    Returns a list of documents (as dictionaries).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        lines = text.splitlines()
        # Heuristic to detect JSONL: more than one line, and each non-empty line starts with '{'.
        if len(lines) > 1 and all(l.strip().startswith('{') for l in lines if l.strip()):
            return [json.loads(l) for l in lines if l.strip()]
        # Otherwise, assume it's a standard JSON array or a single object.
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]

def save_openmt_data(docs: List[Dict[str, Any]], output_dir: str, prefix: str, strategy: int = 0):
    """
    Writes the chunked data into an OpenNMT-py compatible directory structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which strategies to process based on the --strategy argument.
    strategies_to_process = range(1, 9) if strategy == 0 else [strategy]

    # --- Efficiently open all necessary files at once ---
    src_files: Dict[int, TextIO] = {}
    tgt_files: Dict[int, TextIO] = {}
    
    for i in strategies_to_process:
        # Create a subdirectory for each strategy (e.g., 'output/strategy_1/').
        strat_dir = os.path.join(output_dir, f"strategy_{i}")
        os.makedirs(strat_dir, exist_ok=True)
        # Define the source and target file paths (e.g., 'train.src.txt').
        src_path = os.path.join(strat_dir, f"{prefix}.src.txt")
        tgt_path = os.path.join(strat_dir, f"{prefix}.tgt.txt")
        # Open the files and store the file handlers in dictionaries.
        src_files[i] = open(src_path, "w", encoding="utf-8")
        tgt_files[i] = open(tgt_path, "w", encoding="utf-8")

    # --- Process each document and write to the open files ---
    for doc in docs:
        # Get the summary, clean up newlines. Fallback to 'ementa' if 'summary' is missing.
        summary = (doc.get("summary") or doc.get("ementa", "")).replace("\n", " ").strip()
        chunks = doc.get("chunks", [])
        
        for i in strategies_to_process:
            # Get the correct chunk for the current strategy.
            chunk_text = chunks[i-1].strip() if len(chunks) >= i else ""
            # Write the chunk to the corresponding source file and the summary to the target file.
            src_files[i].write(chunk_text + "\n")
            tgt_files[i].write(summary + "\n")

    # --- Clean up by closing all opened files ---
    for i in strategies_to_process:
        src_files[i].close()
        tgt_files[i].close()
        print(f"[LOG] Closed files for strategy {i}")


    print(f"Success! Wrote '{prefix}' data for strategies {list(strategies_to_process)} to '{output_dir}'.")

# --- Main Execution Block ---

def main():
    """Parses command-line arguments and orchestrates the chunking and saving process."""
    # Setup the argument parser to define the command-line interface.
    parser = argparse.ArgumentParser(description="Chunk RulingBR documents into OpenNMT-py format")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file (e.g., rulingbr-v1.2.jsonl)")
    parser.add_argument("--out", required=True, help="Output directory for OpenNMT formatted files")
    parser.add_argument("--prefix", type=str, default="train", help="File prefix for the output files (e.g., train, valid, or test)")
    parser.add_argument("--strategy", type=int, choices=list(range(1, 9)), default=0, help="If set, only process this 1-based strategy index (default: process all)")
    parser.add_argument("--tokenizer-model", type=str, default="bert-base-multilingual-cased", help="Hugging Face tokenizer model for accurate token counting.")
    parser.add_argument("--nrows", type=int, default=0, help="Process only the first N rows for quick testing (0 = all)")
    args = parser.parse_args()

    # 1. Read all documents from the input file.
    print("[LOG] Starting chunking process...")
    docs = read_input(args.input)
    n = args.nrows or len(docs)
    print(f"[LOG] Loaded {len(docs)} documents. Processing first {n}.")
    
    # 2. Initialize the tokenizer if requested.
    hf_tok = None
    if args.tokenizer_model:
        if not _HF_AVAILABLE:
            print("WARNING: 'transformers' is not installed. Falling back to basic tokenization.")
        else:
            try:
                print(f"Loading tokenizer: {args.tokenizer_model}...")
                hf_tok = hf_tokenizer_factory(args.tokenizer_model)
                print("Tokenizer loaded.")
            except Exception as e:
                print(f"WARNING: Failed to load tokenizer '{args.tokenizer_model}': {e}\nFalling back to basic tokenization.")

    # 3. For each document, generate the 8 chunks and add them back to the document dictionary.
    print(f"[LOG] Generating chunks for {len(docs[:n])} documents using {cpu_count()} CPU cores...")

    def _process_doc(doc):
        """Wrapper for parallel chunking"""
        doc["chunks"] = chunk_document(doc, tokenizer=hf_tok)
        return doc

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        processed_docs = []
        for idx, doc in enumerate(pool.imap(_process_doc, docs[:n], chunksize=20)):
            if idx % 100 == 0 and idx > 0:
                print(f"[LOG] Chunked {idx} documents so far...")
            processed_docs.append(doc)

    docs = processed_docs
    print("[LOG] Parallel chunk generation complete.")

    # 4. Save the processed documents into the OpenNMT-py compatible directory  .
    print("[LOG] Saving chunked data to output directory...")
    save_openmt_data(docs[:n], args.out, args.prefix, args.strategy) 
    print("[LOG] All processing completed successfully.")

if __name__ == "__main__":
    # This ensures the main() function is called only when the script is executed directly.
    main()
