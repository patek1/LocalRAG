# Data Loader

This module handles all dataset loading and processing functionality for the LocalRAG pipeline. It transforms raw datasets from Hugging Face into standardized formats for indexing and evaluation.

## Components:

- **`__init__.py`**: Exports key functions and handles module-level imports
- **`base.py`**: Defines base classes and interfaces for dataset processors
- **`utils.py`**: Shared utility functions for data processing

## Dataset Processors:

- **`clapnq_processor.py`**: Handles processing of the ClapNQ dataset
  - Transforms questions and passages into the standard format
  - Integrates with qrels data for retrieval evaluation

- **`triviaqa_processor.py`**: Handles processing of the TriviaQA dataset
  - Extracts questions, answers, and evidence documents
  - Maps to the standard format for the pipeline

- **`hotpotqa_processor.py`**: Handles processing of the HotpotQA dataset
  - Processes multi-hop questions and supporting facts
  - Converts to the standard pipeline format

## Usage Pattern:

Each processor implements dataset-specific processing while adhering to a common interface. The main entry point functions (`process_*_for_rag`) are called from `scripts/create_index.py` to prepare data for indexing.

All processors transform raw datasets into a standardized format:
- Corpus passages: `{"passage_id": "...", "text": "...", "title": "..."}`
- Questions: `{"question_id": "...", "question_text": "...", "gold_answer": "...", "is_answerable": true/false, "gold_passage_ids": ["id1", "id2", ...]}` 