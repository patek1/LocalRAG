# Data

This directory contains the input data required for the LocalRAG pipeline evaluation.

## Contents:

-   **`annotated_data/dev/`**: Contains the ClapNQ development set questions in JSONL format, separated into answerable and unanswerable files.
    -   `clapnq_dev_answerable.jsonl`: Questions that have a known gold answer within the corpus.
    -   `clapnq_dev_unanswerable.jsonl`: Questions designed to be unanswerable based on the provided corpus passages.
-   **`retrieval/dev/`**: Contains the retrieval ground truth (qrels) for the ClapNQ development set in TSV format. This maps question IDs to the known relevant passage IDs.
    -   `question_dev_answerable.tsv`: Ground truth for answerable questions.
    -   `question_dev_unanswerable.tsv`: Ground truth for unanswerable questions (by definition empty).

**Note:** The main passage corpus (`PrimeQA/clapnq_passages`) is downloaded dynamically from the Hugging Face Hub by the `src/data_loader.py` script and is cached elsewhere by the `datasets` library, not stored directly in this directory.