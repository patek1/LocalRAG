# LocalRAG

This project implements and evaluates a Retrieval-Augmented Generation (RAG) pipeline optimized for performance on Apple Silicon hardware, supporting multiple local language models and diverse datasets.

## Project Context

This repository contains the core components of a RAG system that:
1.  Establishes a functional RAG system using locally executable models (where feasible).  
2.  Evaluates its performance thoroughly across retrieval quality, generation quality, and latency metrics.  
3.  Delivers a fully reproducible foundation that others can extend with further performance‑improvement techniques.
4.  Supports evaluation of various language models and datasets to enable comprehensive benchmarking.

## Project Development

### Phase 1: Configuration and Data Loading
- Implemented configuration system with multi-dataset support using Pydantic models
- Created the foundation for data loading with HuggingFace datasets integration
- Established the project structure and module organization

### Phase 2: Data Processing and Indexing
- Implemented full ClapNQ dataset processing in `process_clapnq_for_rag` function
- Added data saving functionality for processed corpus and questions
- Created complete indexing pipeline using ChromaDB:
  - Initialized embedding model with MPS acceleration on Apple Silicon
  - Integrated vector store using ChromaDB for efficient retrieval
  - Implemented corpus indexing with batch processing
  - Added support for reindexing using the `--reindex` flag
- Created comprehensive test suite for:
  - Data processing functionality
  - Indexing pipeline
  - End-to-end script execution

### Phase 3: Retrieval, Generation and Evaluation
- Implemented retrieval system using the indexed corpus
- Integrated with Ollama for local generation
- Added evaluation metrics for retrieval, generation, and end-to-end performance
- Created a streamlined pipeline for processing questions through the RAG system

### Phase 4: Multi-Dataset Support
- Enhanced configuration system to handle multiple datasets
- Implemented directory structure for processed data and results
- Added random subset selection with reproducibility
- Updated metrics calculation to handle datasets without unanswerable questions

### Phase 5: TriviaQA and HotpotQA Integration
- Implemented data processing for TriviaQA and HotpotQA datasets
- Added dataset-specific field mappings and configurations
- Enhanced indexing script to handle different dataset structures
- Created comprehensive tests for new dataset processing

### Phase 6: Testing, Documentation, and GPU Optimization
- Performed comprehensive testing across all datasets and models
- Confirmed GPU utilization with `ollama_num_gpu=999` configuration
- Verified proper metrics collection and results organization
- Created detailed documentation of testing process and results
- Completed final project documentation and structure verification

## System Overview

The system uses LangChain to orchestrate a complete RAG pipeline with support for multiple models and datasets:

-   **Datasets**: Supports ClapNQ, TriviaQA, and HotpotQA datasets from Hugging Face.

-   **Embedding**: Uses `intfloat/e5-base-v2` for document and query embedding, optimized for Metal Performance Shaders (MPS) on Apple Silicon.

-   **Vector Storage**: ChromaDB for efficient local passage indexing and retrieval, with separate indices for each dataset.

-   **Generation**: Multiple models via Ollama, including:
    -   `gemma3:1b`, `gemma3:4b`, `gemma3:12b`
    -   `llama3.2:1b`, `llama3.2:3b`, `llama3.1:8b`
    -   `olmo2:7b`
    -   Any other models available through Ollama

-   **GPU Utilization**: Maximizes GPU usage with `ollama_num_gpu=999` setting to offload as many model layers as possible to the GPU.

-   **Evaluation**: Comprehensive metrics including:
    -   Retrieval: NDCG@10, Precision@10, Recall@10 (using `ranx`)
    -   Generation: ROUGE scores, Unanswerable Accuracy (using `evaluate`)
    -   End-to-End RAG Quality: Faithfulness, Answer Relevancy (using `RAGAs` with OpenAI's `gpt-4.1-mini-2025-04-14` API)
    -   Performance: Retrieval and Generation Latency (mean, median, P95, P99)

**Note on Evaluation Model:** While the goal is local execution, RAGAs evaluation using local models proved prohibitively resource-intensive on typical Apple Silicon hardware. Therefore, RAGAs evaluation currently utilizes the OpenAI API for faster and more reliable results. The core generation pipeline remains local.

## Prerequisites

### Software
-   Python 3.10 or higher
-   Git
-   Ollama ([ollama.com](https://ollama.com/)) installed and running
-   An OpenAI API key (optional, for RAGAs evaluation only)

### Hardware
-   Apple Silicon Mac (M1, M2, M3 series) with macOS 12.3+ recommended for MPS acceleration
-   Tested on M3 Max with 48GB RAM, but should run (potentially slower) on less powerful Apple Silicon machines
-   At least 20GB of free disk space (for models, dependencies, and vector stores)

### Data
-   All datasets are automatically downloaded from Hugging Face:
    -   ClapNQ: `PrimeQA/clapnq` and `PrimeQA/clapnq_passages`
    -   TriviaQA: `trivia_qa` (config: "rc")
    -   HotpotQA: `hotpot_qa` (config: "distractor")
-   Original ClapNQ qrels files are included in the repository under `data/retrieval/dev/`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/patek1/LocalRAG.git
    cd LocalRAG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *   For development and running:
        ```bash
        pip install -r requirements.txt
        ```
    *   For exact reproducibility (after a successful run): A frozen environment is provided in `requirements_frozen.txt`:
        ```bash
        pip install -r requirements_frozen.txt
        ```

4.  **Install and setup Ollama:**
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Pull the model(s) you want to evaluate:
        ```bash
        ollama pull gemma3:1b
        ollama pull gemma3:12b
        ollama pull llama3.1:8b
        # ... or any other Ollama models
        ```
    *   Ensure the Ollama application is running or start the server:
        ```bash
        ollama serve & # (Optional, runs in background)
        ```
    *   Verify Ollama is running and the models are present:
        ```bash
        ollama list
        ```

5.  **Set up OpenAI API key (optional, for RAGAs evaluation):**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your OpenAI API key:
        ```
        OPENAI_API_KEY="sk-your-api-key-here"
        ```
    *   **Note:** Using the OpenAI API incurs costs based on usage. If no API key is provided, the system will gracefully skip RAGAs evaluation.

6.  **Prepare Data:**
    *   Process and index each dataset you want to evaluate:
        ```bash
        python scripts/create_index.py --dataset ClapNQ
        python scripts/create_index.py --dataset TriviaQA
        python scripts/create_index.py --dataset HotpotQA
        ```
    *   This will:
        1. Download the datasets from Hugging Face
        2. Process them into standardized formats
        3. Create the necessary files in `data/processed_corpora/` and `data/processed_questions/`
        4. Build vector indices in `vectorstores/`

## Configuration

Key parameters are defined in `src/config.py`:
- `Settings`: Main configuration class with system-wide parameters
- `DatasetConfig`: Dataset-specific configurations (Hugging Face IDs, field mappings, etc.)
- `FieldMapping`: Maps between raw dataset fields and standardized pipeline fields

The most important settings are now accessible via command-line arguments to the scripts.

## Usage

### Two-Step Workflow

The LocalRAG pipeline now uses a two-step workflow:

1. **Index each dataset (once per dataset):**
   ```bash
   source .venv/bin/activate
   python scripts/create_index.py --dataset <DATASET_NAME>
   ```

2. **Run evaluations with different models:**
   ```bash
   source .venv/bin/activate
   python src/main.py --model <MODEL_TAG> --dataset <DATASET_NAME> [--limit N]
   ```

### Command-Line Options

#### `scripts/create_index.py`
- `--dataset <DATASET_NAME>`: (Required) Specifies the dataset to process and index (e.g., "ClapNQ", "TriviaQA", "HotpotQA")
- `--reindex`: (Optional) Forces re-processing and re-indexing of the dataset, even if an index already exists
- `--debug`: (Optional) Limits the dataset size for faster processing during development/testing

Example: Create or update the HotpotQA index:
```bash
python scripts/create_index.py --dataset HotpotQA --reindex
```

#### `src/main.py`
- `--model <MODEL_TAG>`: (Required) Specifies the Ollama model to use (e.g., "gemma3:1b", "gemma3:12b", "llama3.1:8b")
- `--dataset <DATASET_NAME>`: (Required) Specifies the dataset to use (must be indexed first)
- `--limit <N>`: (Optional) Process only N random questions from the dataset. Useful for quick testing and evaluation on a subset.

Example: Evaluate gemma3:12b on TriviaQA with 50 questions:
```bash
python src/main.py --model gemma3:12b --dataset TriviaQA --limit 50
```

**Notes:**
1. Ensure that you have previously indexed the dataset using `create_index.py` before running `main.py`
2. The same `--limit N` value with the same dataset will always use the same subset of questions for reproducibility
3. GPU utilization is automatically maximized with the `ollama_num_gpu=999` setting

## Expected Output

1.  **Console Logs:** Detailed progress information, latency numbers per question, final evaluation metrics, and any warnings/errors.

2.  **Vector Store:** Dataset-specific ChromaDB indices created in `vectorstores/<dataset_name>_corpus/`.

3.  **Results Files:** JSON files saved in the `results/<model_tag_safe>/<dataset_name>/` directory:
    -   `<N>_quality_metrics.json`: Contains retrieval (NDCG@10, P@10, R@10), generation (ROUGE, Unanswerable Accuracy), and RAGAs metrics for a run with N questions.
    -   `latency_metrics.json`: Contains mean, median, P95, P99 latency for retrieval and generation steps, along with the `source_subset_size` indicating the number of questions used to generate these stats.

## Performance Monitoring

Since this project targets local execution on Apple Silicon, manual performance monitoring is crucial:
-   Use tools like `asitop` (recommended), `mactop`, or macOS's built-in `Activity Monitor`.
-   Observe CPU usage, RAM consumption (especially during embedding/generation), and GPU utilization (MPS activity).
-   The `ollama_num_gpu=999` setting will attempt to offload as many model layers as possible to the GPU.

## Architecture

The project follows a modular structure:

-   `src/`: Core source code modules:
    -   `config.py`: Central configuration management with dataset configs
    -   `data_loader/`: Dataset loading and processing for all supported datasets
    -   `embedding.py`: Embedding model with MPS acceleration
    -   `vector_store.py`: ChromaDB vector store management
    -   `indexing.py`: Corpus indexing pipeline with batch processing
    -   `prompting.py`: Prompt formatting for the generator LLM
    -   `generation.py`: Ollama LLM initialization with GPU optimization
    -   `pipeline.py`: RAG chain orchestration
    -   `eval/`: Evaluation metrics modules
    -   `utils.py`: Common utilities
    -   `main.py`: Main execution script with CLI argument handling

-   `scripts/`: Utility scripts:
    -   `create_index.py`: Dataset processing and indexing script

-   `data/`: Data directory structure:
    -   `processed_corpora/`: Standardized corpus files (generated)
    -   `processed_questions/`: Standardized question files (generated)
    -   `subsets/`: Question ID subsets for reproducible evaluation (generated)
    -   Original ClapNQ files used by the ClapNQ processor



**Note:** Directories marked as "(generated)" are created during execution and are not included in the repository.

## Metrics

The evaluation pipeline calculates and reports:

1.  **Retrieval Quality** (via `ranx`):
    -   NDCG@10
    -   Precision@10
    -   Recall@10
2.  **Answer Quality** (via `evaluate` and custom logic):
    -   ROUGE-1, ROUGE-2, ROUGE-L (F-measure) for answerable questions.
    -   Unanswerable Accuracy: Percentage of unanswerable questions correctly identified as "unanswerable".
    -   Note: For datasets without explicit unanswerables (TriviaQA, HotpotQA), this reports "N/A".
3.  **RAG Quality** (via `RAGAs` using OpenAI, when API key is available):
    -   Faithfulness: How factually consistent the generated answer is with the retrieved context.
    -   Answer Relevancy: How relevant the generated answer is to the original question.
4.  **Performance**:
    -   Retrieval Latency (Mean, Median, P95, P99) in seconds.
    -   Generation Latency (Mean, Median, P95, P99) in seconds.

## Future Work

This project provides a robust framework for evaluating various RAG configurations. Future work could include:
- Implementing additional datasets beyond the current three
- Exploring advanced retrieval techniques
- Adding more sophisticated prompt engineering
- Implementing more comprehensive evaluation metrics
- Exploring fully local alternatives to OpenAI for RAGAs evaluation

## Contact

If you have questions or would like to discuss extending this project, feel free to contact me:
**Mischa Büchel**, [mischa@quantflow.tech](mischa@quantflow.tech)