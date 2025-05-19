# Source Code

This directory contains all the Python source code for the LocalRAG pipeline project.

## Structure:

The code is organized into modules based on functionality:

-   **`config.py`**: Central configuration settings including:
    -   `Settings`: Main configuration class with all pipeline parameters
    -   `DatasetConfig`: Configuration for each supported dataset (ClapNQ, TriviaQA, HotpotQA)
    -   `FieldMapping`: Mappings between raw dataset fields and standardized pipeline fields

-   **`data_loader/`**: Module for loading and processing datasets:
    -   Dataset-specific processors for ClapNQ, TriviaQA, and HotpotQA
    -   Standardization of corpus passages and questions across datasets
    -   See the module's README for more details on its components

-   **`embedding.py`**: Logic for initializing the sentence transformer embedding model with MPS acceleration.

-   **`vector_store.py`**: Logic for setting up and interacting with the ChromaDB vector store for each dataset.

-   **`indexing.py`**: The pipeline for indexing corpus passages into the vector store with batch processing.

-   **`prompting.py`**: Utilities for formatting prompts for the RAG model.

-   **`generation.py`**: Logic for initializing the local generator LLM through Ollama with:
    -   Model selection based on CLI arguments
    -   GPU layer utilization configuration

-   **`pipeline.py`**: Orchestrates the core RAG steps (retrieval and generation).

-   **`eval/`**: A sub-package containing modules for different evaluation metrics:
    -   `metrics_retrieval.py`: ranx-based retrieval metrics
    -   `metrics_generation.py`: ROUGE and unanswerable accuracy
    -   `metrics_ragas.py`: RAGAs setup and calculation (using OpenAI)
    -   `metrics_latency.py`: Latency statistics calculation and saving

-   **`utils.py`**: Common helper functions, such as logging setup.

-   **`main.py`**: The main entry point script that:
    -   Parses command-line arguments for model and dataset selection
    -   Loads processed data and vector stores
    -   Handles subset selection with reproducibility
    -   Executes the RAG pipeline and evaluation
    -   Organizes and saves results in the model/dataset directory structure