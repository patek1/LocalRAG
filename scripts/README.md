# Scripts

This directory contains utility scripts that support the LocalRAG project's functionality.

## Contents:

### Core Scripts

- **`create_index.py`**: The primary script for processing datasets and creating vector store indices.
  - Processes data from Hugging Face datasets
  - Creates standardized corpus and question files
  - Indexes corpus passages into ChromaDB for retrieval
  - **Usage**: 
    ```bash
    python scripts/create_index.py --dataset <DATASET_NAME> [--reindex] [--debug]
    ```
    - `--dataset`: Required. Specify the dataset to process (e.g., "ClapNQ", "TriviaQA", "HotpotQA")
    - `--reindex`: Optional. Force re-processing and re-indexing even if an index already exists
    - `--debug`: Optional. Process only a small subset of data for faster development/testing

## Workflow

The typical workflow with these scripts is:

1. Process and index each dataset you want to evaluate:
   ```bash
   python scripts/create_index.py --dataset ClapNQ
   python scripts/create_index.py --dataset TriviaQA
   python scripts/create_index.py --dataset HotpotQA
   ```

2. Then run the main script (from the project root) to evaluate models on the indexed datasets:
   ```bash
   python src/main.py --model <MODEL_TAG> --dataset <DATASET_NAME> [--limit N]
   ```

**Note**: The `create_index.py` script must be run first for each dataset before using that dataset in `main.py`. 