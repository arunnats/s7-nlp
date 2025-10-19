The paper introduces a method called **LegalSumm** designed to improve the automatic summarization of long and complex legal documents. The authors aim to solve two main challenges in this area: `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

- **Document Length:** Legal rulings are often too long to be processed by standard summarization models, which have a limited input size. Truncating the text can cause the loss of crucial information. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
- **Factual "Hallucination":** Abstractive summarization models can sometimes generate text that is not factually supported by the source document. This is a critical issue in the legal field, where precision and faithfulness to the original text are paramount. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

### The LegalSumm Method: What They Did

The core idea of **LegalSumm** is to break down the summarization task into a multi-step process that generates several candidate summaries and then selects the most factually consistent one. The process works as follows: `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

1.  **Create Different "Views" of the Document:** Instead of feeding the entire long document into one model, the system first splits the source text into smaller, more manageable "chunks." These chunks are created using several predefined strategies, such as combining text from different sections of the legal ruling (e.g., report, vote, and judgment sections). This ensures that different parts of the original document are captured. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
2.  **Generate Candidate Summaries:** Each chunk is fed into a separate Transformer-based summarization model. This results in multiple, independent candidate summaries, each one generated from a different "view" or part of the source text. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
3.  **Score and Select the Best Summary:** The system then uses a _textual entailment_ module to evaluate how faithful each candidate summary is to its corresponding chunk. The entailment model gives a score representing the likelihood that the summary's content can be inferred from the source chunk. The candidate summary with the highest score is chosen as the final output. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

This approach avoids factual "hallucination" by explicitly selecting the summary that is most strongly supported by the source text, rather than relying solely on the generative model's output. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

### Implementation Details

The authors provide specific details about the models, data, and training procedures used to build and evaluate **LegalSumm**.

**Dataset**

The experiments were conducted on the **RulingBR dataset**, which contains approximately 10,000 real court rulings from the Brazilian Supreme Court, written in Portuguese. Each ruling is divided into sections (summary, report, vote, judgment), which the model uses to create its chunks. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

**Model Architecture and Training**

The **LegalSumm** system is composed of two main components:

- **Summarization Module:**
  - This module uses standard **Transformer** models from the OpenNMT-py library. Each model has 6 encoder and 6 decoder layers with an embedding and hidden dimension of 512. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
  - A separate Transformer model is trained for each of the eight different chunking strategies, making each model a specialist for summarizing a specific type of text chunk. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
- **Entailment Module:**
  - This module uses **BERT** (_bert-base-multilingual-cased_), a powerful pre-trained language model, which is fine-tuned for the textual entailment task. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
  - To train this model, the authors created a dataset of "fact" and "fake" examples.
  - **Fact examples** consist of a text chunk and its correct, human-written summary.
  - **Fake examples** are created by pairing a text chunk with a summary from a _different but related_ legal case. This teaches the model to distinguish between a correct summary and one that is off-topic, even if the subject matter is similar. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
  - The model is trained as a binary classifier to predict whether a given chunk-summary pair is a "fact" or "fake," with the output score representing its confidence. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`

### Key Findings and Results

The paper evaluates **LegalSumm** by comparing it against other summarization models and through human evaluation.

- **Quantitative Results:** **LegalSumm** outperformed or matched strong baselines like **BART** and **BertSumAbs** on all ROUGE evaluation metrics, which measure the overlap between the generated summary and a reference summary. The most significant gains were in precision, indicating that **LegalSumm** generates more factually accurate summaries. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
- **Human Evaluation:** The authors asked eleven legal experts to rate the summaries generated by **LegalSumm** and a baseline model. The experts found **LegalSumm**'s summaries to be superior in all four categories evaluated:
  - Coverage of important topics
  - Coherence and flow
  - Faithfulness to the facts
  - Potential to replace the original human-written summary. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
- **Limitations:** The authors acknowledge that the system's performance is limited by **BERT**'s maximum input length of 512 tokens. They also note that while the summaries are of high quality, they are not yet reliable enough to fully replace human summarizers but could serve as excellent drafts to reduce their workload. `Improving-abstractive-summarization-of-legal-rulings-through-textual-entailment.pdf`
