# Chunker utility

This small utility implements the 8 chunking strategies referenced in `implementation.md`.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

2. Run chunker on the sample file:

```bash
python tools/chunker.py --input rulingbr/sample-5.json --out sample_with_chunks.jsonl
```

3. To write only strategy 1 chunks:

```bash
python tools/chunker.py --input rulingbr/sample-5.json --out sample_ch1.jsonl --strategy 1
```

Output: a JSONL file where each line is the original document JSON plus a `chunks` array (8 items) or a single-item `chunks` array if `--strategy` is provided.
