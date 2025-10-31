#!/usr/bin/env python3
"""Simple chunking utility for RulingBR-style documents.

Implements the 8 chunking strategies referenced in the project's
`implementation.md`. Intended to be run from the repository root and
used against `rulingbr/sample-5.json` or extracted `rulingbr-v1.2.jsonl`.

Usage:
  python tools/chunker.py --input rulingbr/sample-5.json --strategy 1 --out samples_ch1.jsonl

The script reads JSONL or a JSON array file and writes JSONL with added
`chunks` field containing an array of chunk texts (one per strategy)
for each document.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

try:
    # prefer nltk if available for basic tokenization
    import nltk
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

# Hugging Face tokenizer support (optional). We'll lazy-import when requested.
_HF_AVAILABLE = False
try:
    # do not import transformers/tokenizers at module import time; instead lazily load
    import transformers  # type: ignore
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into tokens. Use NLTK word_tokenize if available,
    otherwise fall back to whitespace split.
    """
    if not text:
        return []
    if _HAS_NLTK:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return text.split()
    return text.split()


def hf_tokenizer_factory(model_name: str):
    """Return a tokenizer callable that maps text -> list of tokens using HF tokenizer.

    The returned tokenizer produces tokens as the tokenizer's token strings (subwords).
    """
    if not _HF_AVAILABLE:
        raise RuntimeError("transformers is not installed; install it or use basic tokenization")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        # use tokenizer to return token strings (not ids) so we can detokenize approx with spaces
        enc = tok.encode(text, add_special_tokens=False)
        # convert ids back to tokens
        toks = tok.convert_ids_to_tokens(enc)
        return toks

    return _tokenize


def detokenize(tokens: List[str]) -> str:
    """Join tokens back into text."""
    return " ".join(tokens)


def take_prefix_tokens(text: str, n: int) -> str:
    tokens = simple_tokenize(text)
    return detokenize(tokens[:n])


def concat_and_truncate(parts: List[str], max_tokens: int) -> str:
    tokens: List[str] = []
    for p in parts:
        if not p:
            continue
        tokens.extend(simple_tokenize(p))
        if len(tokens) >= max_tokens:
            break
    return detokenize(tokens[:max_tokens])


def chunk_document(doc: Dict[str, Any], tokenizer=None) -> List[str]:
    """Return list of 8 chunks following the strategies in implementation.md.

    Expected fields in `doc`: 'relatorio' (report), 'voto' (vote), 'acordao' (judgment).
    Uses token counts approximations. If a field is missing, an empty string is used.
    """
    report = doc.get("relatorio", "") or doc.get("relatorio_text", "") or ""
    vote = doc.get("voto", "") or doc.get("voto_text", "") or ""
    judgment = doc.get("acordao", "") or doc.get("acordao_text", "") or ""

    # If an HF tokenizer callable is provided, use it for tokenization and truncation
    if tokenizer is not None:
        def tkn_prefix(text: str, n: int) -> str:
            toks = tokenizer(text)
            # tokens are subword units; join with space to produce readable output
            return detokenize(toks[:n])

        def tkn_concat(parts: List[str], max_tokens: int) -> str:
            toks: List[str] = []
            for p in parts:
                if not p:
                    continue
                toks.extend(tokenizer(p))
                if len(toks) >= max_tokens:
                    break
            return detokenize(toks[:max_tokens])

        use_prefix = tkn_prefix
        use_concat = tkn_concat
    else:
        use_prefix = lambda text, n: detokenize(simple_tokenize(text)[:n])
        use_concat = lambda parts, n: concat_and_truncate(parts, n)

    # Strategy definitions (tokens)
    # 1: 300 report + 100 judgment
    s1 = use_prefix(report, 300) + " " + use_prefix(judgment, 100)

    # 2: 300 vote + 100 judgment
    s2 = use_prefix(vote, 300) + " " + use_prefix(judgment, 100)

    # 3: 150 report + 150 vote + 100 judgment
    s3 = use_prefix(report, 150) + " " + use_prefix(vote, 150) + " " + use_prefix(judgment, 100)

    # 4: 400 report
    s4 = use_prefix(report, 400)

    # 5: 400 vote
    s5 = use_prefix(vote, 400)

    # 6: 400(report+vote+judgment) - concatenate then truncate to 400 tokens
    s6 = use_concat([report, vote, judgment], 400)

    # 7: 400(vote+judgment+report)
    s7 = use_concat([vote, judgment, report], 400)

    # 8: 400(judgment+report+vote)
    s8 = use_concat([judgment, report, vote], 400)

    return [s1.strip(), s2.strip(), s3.strip(), s4.strip(), s5.strip(), s6.strip(), s7.strip(), s8.strip()]


def read_input(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        # Try JSON lines (multiple JSON objects)
        lines = text.splitlines()
        if len(lines) > 1 and all(l.strip().startswith('{') for l in lines if l.strip()):
            docs = [json.loads(l) for l in lines if l.strip()]
            return docs
        # Otherwise, parse as JSON array or single object
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return [obj]


def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chunk RulingBR documents into predefined strategies")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file (rulingbr sample or jsonl)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--strategy", type=int, choices=list(range(1, 9)), default=0,
                        help="If set, only include this 1-based strategy index in 'chunks' array (default: include all)")
    parser.add_argument("--tokenizer-model", type=str, default="bert-base-multilingual-cased",
                        help="Hugging Face tokenizer model for token counting/truncation (default: bert-base-multilingual-cased). If transformers is not installed or model cannot be downloaded, falls back to basic tokenization.")
    parser.add_argument("--nrows", type=int, default=0, help="Process only first N rows (0 = all)")
    args = parser.parse_args()

    docs = read_input(args.input)
    out_docs = []
    n = args.nrows or len(docs)
    # Build optional HF tokenizer callable if requested
    hf_tok = None
    if args.tokenizer_model:
        if not _HF_AVAILABLE:
            # warn and fall back
            print("WARNING: transformers not available; falling back to basic tokenization. To enable HF tokenizer install requirements.txt")
            hf_tok = None
        else:
            try:
                hf_tok = hf_tokenizer_factory(args.tokenizer_model)
            except Exception as e:
                print(f"WARNING: failed to load tokenizer {args.tokenizer_model}: {e}\nFalling back to basic tokenization.")
                hf_tok = None

    for i, doc in enumerate(docs[:n]):
        chunks = chunk_document(doc, tokenizer=hf_tok)
        if args.strategy:
            idx = args.strategy - 1
            doc_out = dict(doc)
            doc_out["chunks"] = [chunks[idx]]
        else:
            doc_out = dict(doc)
            doc_out["chunks"] = chunks
        out_docs.append(doc_out)

    write_jsonl(out_docs, args.out)


if __name__ == "__main__":
    main()
