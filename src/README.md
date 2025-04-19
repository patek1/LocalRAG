# SOURCE

This directory contains all the Python source code for the LocalRAG pipeline project.

## Structure:

The code is organized into modules based on functionality:

-   **`config.py`**: Central configuration settings (paths, models, parameters).
-   **`data_loader.py`**: Functions for loading the corpus, questions, and ground truth data.
-   **`embedding.py`**: Logic for initializing the sentence transformer embedding model.
-   **`vector_store.py`**: Logic for setting up and interacting with the ChromaDB vector store.
-   **`indexing.py`**: The pipeline for indexing corpus passages into the vector store.
-   **`prompting.py`**: Utilities for formatting prompts for the RAG model.
-   **`generation.py`**: Logic for initializing the local generator LLM (via Ollama).
-   **`pipeline.py`**: Orchestrates the core RAG steps (retrieval and generation).
-   **`eval/`**: A sub-package containing modules for different evaluation metrics (retrieval, generation, RAGAs, latency).
-   **`utils.py`**: Common helper functions, such as logging setup.
-   **`main.py`**: The main entry point script to run the entire indexing, execution, and evaluation workflow.