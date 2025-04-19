# Baseline Local RAG Pipeline Evaluation on Apple Silicon

This project implements and evaluates a baseline Retrieval-Augmented Generation (RAG) pipeline, optimized for performance on Apple Silicon hardware, using the ClapNQ dataset.

## Project Context

This repository represents **Part 1** of a bachelor thesis project. The goal of this part is to:
1.  Establish a functional baseline RAG system using locally executable models (where feasible).
2.  Evaluate its performance comprehensively across retrieval quality, generation quality, and latency metrics.
3.  Provide a reproducible foundation for **Part 2**, which will explore and implement various techniques to improve the performance of this baseline system.

## System Overview

The system uses LangChain to orchestrate a complete RAG pipeline:

-   **Embedding**: Uses `intfloat/e5-base-v2` for document and query embedding, optimized for Metal Performance Shaders (MPS) on Apple Silicon.
-   **Vector Storage**: ChromaDB for efficient local passage indexing and retrieval.
-   **Generation**: Google's Gemma 3 12B model via Ollama for answer generation, running locally.
-   **Evaluation**: Comprehensive metrics including:
    -   Retrieval: NDCG@10, Precision@10, Recall@10 (using `ranx`)
    -   Generation: ROUGE scores, Unanswerable Accuracy (using `evaluate`)
    -   End-to-End RAG Quality: Faithfulness, Answer Relevancy (using `RAGAs` with OpenAI's `gpt-4.1-mini-2025-04-14` API)
    -   Performance: Retrieval and Generation Latency (mean, median, P95, P99)

**Note on Evaluation Model:** While the goal is local execution, RAGAs evaluation using local models (like `gemma3:27b`) proved prohibitively resource-intensive and slow on typical Apple Silicon hardware. Therefore, RAGAs evaluation currently utilizes the OpenAI API for faster and more reliable results. The core generation pipeline remains local.

## Prerequisites

### Software
-   Python 3.10 or higher
-   Git
-   Ollama ([ollama.com](https://ollama.com/)) installed and running.
-   An OpenAI API key (for RAGAs evaluation).

### Hardware
-   Apple Silicon Mac (M1, M2, M3 series) with macOS 12.3+ recommended for MPS acceleration.
-   Tested on M3 Max with 48GB RAM, but should run (potentially slower) on less powerful Apple Silicon machines.
-   At least 20GB of free disk space (for models, dependencies, and vector store).

### Data
-   Access to download the ClapNQ dataset (passages via Hugging Face datasets, questions/qrels provided in `data/`).
   -  HuggingFace Datasets: PrimeQA/clapnq_passages ([huggingface.co/datasets/PrimeQA/clapnq_passages](https://huggingface.co/datasets/PrimeQA/clapnq_passages/viewer/default/train?views%5B%5D=train))
   -  ClapNQ GitHub: PrimeQA/clapnq_passages ([github.com/primeqa/clapnq](https://github.com/primeqa/clapnq))

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
    *   For exact reproducibility (after a successful run): A frozen environment is provided in `requirements_frozen.txt`. You can use this for precise replication:
        ```bash
        pip install -r requirements_frozen.txt
        ```

4.  **Install and setup Ollama:**
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Pull the required model (this might take time and disk space):
        ```bash
        ollama pull gemma3:12b
        ```
    *   Ensure the Ollama application is running or start the server:
        ```bash
        ollama serve & # (Optional, runs in background)
        ```
    *   Verify Ollama is running and the model is present:
        ```bash
        ollama list
        ```

5.  **Set up OpenAI API key for evaluation:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your OpenAI API key:
        ```
        OPENAI_API_KEY="sk-your-api-key-here"
        ```
    *   **Note:** Using the OpenAI API incurs costs based on usage. Monitor your spending on the OpenAI platform.

6.  **Prepare Data:**
    *   The required `data/annotated_data/dev/` and `data/retrieval/dev/` files are included in the repository.
    *   The script will automatically download the `PrimeQA/clapnq_passages` dataset from Hugging Face upon first run if not already cached. Ensure you have an internet connection.

## Configuration

Key parameters like model names, file paths, retrieval settings (`k`), and output locations can be adjusted in `src/config.py`.

## Usage

### Running the Full Pipeline

1.  Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
2.  Run the main script:
    ```bash
    python src/main.py
    ```
    This will perform the following steps:
    -   Load data.
    -   Initialize models and vector store.
    -   Index passages (if the vector store is empty or `--reindex` is used).
    -   Run the RAG pipeline for each question in the dev set.
    -   Calculate all evaluation metrics.
    -   Print metrics to the console and save JSON results to the `results/` directory.

### Command-Line Options

-   `--reindex`: Force re-indexing of corpus passages, even if the vector store directory exists and contains data. Useful if the corpus or embedding model changes.
-   `--limit N`: Process only the first `N` questions from the development set. Useful for quick testing and debugging.

Example: Process 10 questions with a fresh index:
```bash
python src/main.py --limit 10 --reindex
```

## Expected Output

1.  **Console Logs:** Detailed progress information, latency numbers per question, final evaluation metrics, and any warnings/errors.
2.  **Vector Store:** Persistent ChromaDB index created in `vectorstores/intfloat_e5-base-v2/` (or as configured).
3.  **Results Files:** JSON files saved in the `results/` directory:
    -   `baseline_latency_metrics.json`: Contains mean, median, P95, P99 latency for retrieval and generation steps.
    -   `baseline_quality_metrics.json`: Contains retrieval (NDCG@10, P@10, R@10), generation (ROUGE, Unanswerable Accuracy), and RAGAs (faithfulness, answer_relevancy) scores.

## Performance Monitoring

Since this project targets local execution on Apple Silicon, manual performance monitoring is crucial:
-   Use tools like `asitop` (recommended), `mactop`, or macOS's built-in `Activity Monitor`.
-   Observe CPU usage, RAM consumption (especially during embedding/generation), and GPU utilization (MPS activity).
-   Note any thermal throttling or significant performance degradation during long runs.

## Architecture

The project follows a modular structure within the `src/` directory:

-   `config.py`: Central configuration management.
-   `data_loader.py`: Loads passages, questions, and ground truth.
-   `embedding.py`: Initializes the embedding model.
-   `vector_store.py`: Sets up and manages the ChromaDB vector store.
-   `indexing.py`: Handles the corpus indexing pipeline.
-   `prompting.py`: Formats prompts for the generator LLM.
-   `generation.py`: Initializes the generator LLM (Ollama).
-   `pipeline.py`: Orchestrates the core RAG chain (retrieval + generation).
-   `eval/`: Sub-package containing evaluation logic:
    -   `metrics_retrieval.py`: ranx-based retrieval metrics.
    -   `metrics_generation.py`: ROUGE and unanswerable accuracy.
    -   `metrics_ragas.py`: RAGAs setup and calculation (using OpenAI).
    -   `metrics_latency.py`: Latency statistics calculation and saving.
-   `utils.py`: Common utilities like logging setup.
-   `main.py`: Main script orchestrating the entire process.

## Metrics

The evaluation pipeline calculates and reports:

1.  **Retrieval Quality** (via `ranx`):
    -   NDCG@10
    -   Precision@10
    -   Recall@10
2.  **Answer Quality** (via `evaluate` and custom logic):
    -   ROUGE-1, ROUGE-2, ROUGE-L (F-measure) for answerable questions.
    -   Unanswerable Accuracy: Percentage of unanswerable questions correctly identified as "unanswerable".
3.  **RAG Quality** (via `RAGAs` using OpenAI):
    -   Faithfulness: How factually consistent the generated answer is with the retrieved context.
    -   Answer Relevancy: How relevant the generated answer is to the original question.
4.  **Performance**:
    -   Retrieval Latency (Mean, Median, P95, P99) in seconds.
    -   Generation Latency (Mean, Median, P95, P99) in seconds.

## Future Work (Thesis Part 2)

This baseline provides the foundation for future work, which will involve implementing and evaluating improvements based on techniques proposed in RAG research literature, focusing on the four RAG stages: 
   - Pre-Retrieval
   - Retrieval
   - Post-Retrieval
   - Generation.