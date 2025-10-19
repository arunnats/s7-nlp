# Step-by-Step Guide to Recreating LegalSumm

Here's a clear breakdown of how to recreate the **LegalSumm** approach for improving abstractive summarization of legal rulings using textual entailment:

1.  **Prepare the Dataset**

    - Obtain a large set of legal rulings with human-written summaries (e.g., RulingBR dataset).
    - Split each ruling into its main sections (e.g., summary, report, vote, judgment).
    - Use the summary as ground-truth and the other sections as source text.

2.  **Chunking Strategies**

    - Define multiple chunking strategies to create different "views" of each ruling. For example:
      - Strategy 1: 300 tokens from report + 100 from judgment
      - Strategy 2: 300 tokens from vote + 100 from judgment
      - Strategy 3: 150 from report + 150 from vote + 100 from judgment
      - ... (see Table 1 in the paper for all 8 strategies)
    - For each strategy, extract the corresponding chunk from each ruling, truncating to a max length (e.g., 400 tokens).

3.  **Train Summarization Models**

    - For each chunking strategy, train a separate Transformer-based summarization model (e.g., using `OpenNMT-py`).
    - Use encoder-decoder architecture with self-attention (6 layers each, 512 hidden size).
    - Train each model to generate a summary from its chunk.
    - Apply constraints during decoding (e.g., beam search, n-gram repetition avoidance, min/max summary length).

4.  **Generate Candidate Summaries**

    - For each ruling, use all chunking strategies to generate multiple candidate summaries (one per strategy).

5.  **Build Entailment Training Data**

    - For each chunk-summary pair:
      - Label the true chunk-summary pair as "fact".
      - Create "fake" pairs by combining the chunk with summaries from other, related rulings (using category similarity).
    - This creates a dataset for training the entailment module to distinguish fact from fake.

6.  **Train Entailment Module**

    - Fine-tune a BERT-based model (e.g., `bert-base-multilingual-cased`) for binary classification (fact vs. fake).
    - Input: `[CLS] chunk tokens [SEP] summary tokens [SEP]`
    - Output: Confidence score (0-1) that the summary is entailed by the chunk.
    - Train a separate entailment model for each chunking strategy.

7.  **Score and Select Final Summary**

    - For each ruling, score all candidate summaries using the corresponding entailment model.
    - Select the summary with the highest entailment score as the final output.

8.  **Evaluate Performance**
    - Use ROUGE metrics (ROUGE-N, ROUGE-L) to compare generated summaries to ground-truth.
    - Optionally, conduct human expert evaluation for coverage, coherence, faithfulness, and replaceability.

# Detailed Version

### **Phase 0: Environment Setup (The "Before You Start")**

Before you write any code, you'll need to set up your environment. This project is computationally expensive.

- **Get a GPU:** You will be training 8 Transformer models and 8 BERT models. You **will need** a powerful GPU. I recommend using Google Colab Pro, Kaggle, or a cloud VM with a GPU.
- **Install Core Libraries:**
  - `transformers`: (From Hugging Face) To download and fine-tune `bert-base-multilingual-cased` for the entailment module.
  - `OpenNMT-py`: The paper explicitly uses this for the Transformer summarization models.
  - `pandas`: To load and manipulate the dataset.
  - `rouge-score`: To run your final evaluation (Your Step 8).
  - `torch` or `tensorflow`: Your other libraries will depend on one of these.

---

### **Phase 1: Data Acquisition & Chunking (Your Steps 1 & 2)**

This is your first major coding task. You need to get the data and create the "views" for your models.

1.  **Get the Dataset:** Find and download the **RulingBR dataset**. It should contain the 10k rulings, each split into `summary`, `report`, `vote`, and `judgment` sections.
2.  **Write the Chunking Script:** This is the core of your Step 2. Create a Python script that:
    - Loads a single ruling.
    - Applies the 8 chunking strategies from **Table 1** of the paper to the `report`, `vote`, and `judgment` sections.
    - **The 8 Strategies from Table 1 are:**
      1.  300 tokens (report) + 100 tokens (judgment)
      2.  300 tokens (vote) + 100 tokens (judgment)
      3.  150 (report) + 150 (vote) + 100 (judgment)
      4.  400 (report)
      5.  400 (vote)
      6.  400 (report) + 400 (vote) + 400 (judgment)
      7.  400 (vote) + 400 (judgment) + 400 (report)
      8.  400 (judgment) + 400 (report) + 400 (vote)
    - **Note:** For strategies 6-8, the paper states they are concatenated and _then_ truncated to a max length of 400 tokens. The paper also notes a general 400-token limit for chunks.
3.  **Save the Training Data:** Your script's output should be formatted for training. For each of the 10k rulings, you'll create 8 "source" chunks and 1 "target" summary. You'll need to organize these into 8 separate datasets (one for each strategy). For example:
    - `strategy_1/train.source.txt` (all the chunks from strategy 1)
    - `strategy_1/train.target.txt` (the human-written summary, repeated for each chunk)
    - ...and so on for `strategy_2` through `strategy_8`.

---

### **Phase 2: Train Summarization Models (Your Step 3 & 4)**

**My advice: Start with _one_ model first.** Don't try to train all 8 at once.

1.  **Configure OpenNMT-py:** Use the data from `strategy_1`. Configure your `OpenNMT-py` model to match the paper: a Transformer with 6 encoder layers and 6 decoder layers, with an embedding/hidden dimension of 512.
2.  **Train Model 1:** Train your first summarization model on the `strategy_1` dataset. This will take time.
3.  **Generate Candidates (for one model):** Once trained, use this model to generate candidate summaries for your validation/test set. This completes your Step 4 _for a single strategy_.
4.  **Repeat:** Once you have a working pipeline for Strategy 1, repeat this process for the other 7 strategies. You will end up with 8 different trained summarization models.

---

### **Phase 3: Train Entailment Module (Your Step 5 & 6)**

This phase can be done in parallel with Phase 2. Again, start with _one_ model.

1.  **Build the "Fact/Fake" Dataset (Your Step 5):** This is your second major coding task.
    - **"Fact" Examples:** For each chunk in your `strategy_1` training set, pair it with its _correct_ human-written summary. Label this `1` (or "fact").
    - **"Fake" Examples:** This is the clever part. For each chunk, pair it with a human-written summary from a _different but related_ ruling. The paper suggests using "category similarity" to find a related case. Label this `0` (or "fake").
    - You will now have a new binary classification dataset: `(chunk, summary, label)`.
2.  **Format for BERT:** The input must be in the format: `[CLS] chunk tokens [SEP] summary tokens [SEP]`.
3.  **Train Entailment Model 1:**
    - Load `bert-base-multilingual-cased` using the `transformers` library (e.g., `AutoModelForSequenceClassification`).
    - Fine-tune this BERT model on your new "fact/fake" dataset for Strategy 1.
4.  **Repeat:** Just like in Phase 2, you must repeat this process for all 8 chunking strategies, resulting in 8 different trained entailment models.

---

### **Phase 4: Full Pipeline & Selection (Your Step 7)**

Now you combine everything. This is the final inference script (as seen in Figure 2 of the paper).

1.  **Write the Inference Script:** This script should take a _single new ruling_ as input.
2.  **Step 1:** Apply your 8 chunking strategies to get 8 chunks.
3.  **Step 2:** Feed each chunk to its corresponding summarization model (Chunk 1 -> Model 1, Chunk 2 -> Model 2, etc.). This gives you 8 candidate summaries.
4.  **Step 3:** Feed each `(chunk, candidate_summary)` pair to its corresponding entailment model (Pair 1 -> Entailment Model 1, etc.). This gives you 8 scores (a number between 0 and 1).
5.  **Step 4:** Find the highest score. The summary that produced that highest score is your final output.

---

### **Phase 5: Evaluation (Your Step 8)**

1.  **Run on Test Set:** Run your entire Phase 4 pipeline on every ruling in your test set.
2.  **Calculate ROUGE:** Use the `rouge-score` library to compare your final generated summaries against the ground-truth human-written summaries.
3.  **Analyze:** Check your ROUGE-N and ROUGE-L scores. If your implementation is correct, your scores should be an improvement over the baseline models mentioned in the paper.
