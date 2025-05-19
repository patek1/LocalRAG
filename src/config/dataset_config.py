"""
Dataset configuration module for the Local RAG Pipeline.

This module defines the FieldMapping and DatasetConfig classes
that enable support for multiple datasets with different schemas.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class FieldMapping(BaseModel):
    """
    Maps raw dataset field names to the pipeline's internal expected names.
    
    This enables processing different datasets with varying schema structures
    through a common data pipeline.
    """
    question_id: str = "id"
    question_text: str = "question"
    answer_text: str = "answer"
    passage_id: str = "id"  # Field for passage ID mapping
    passage_content_fields: List[str] = Field(default_factory=lambda: ["text"])
    passage_title_field: Optional[List[str]] = None
    gold_passage_identifier_field_in_question_data: Optional[str] = None
    # Optional paths for ClapNQ qrels
    qrels_ans_path: Optional[str] = None
    qrels_unans_path: Optional[str] = None


class DatasetConfig(BaseModel):
    """
    Configuration for a specific dataset, including HF identifiers and field mappings.
    
    Contains methods to derive standardized paths for processed data and vector stores.
    """
    hf_id: str
    hf_config_name: Optional[str] = None
    hf_split_train: Optional[str] = "train"
    hf_split_eval: str = "validation"
    corpus_hf_id: Optional[str] = None
    corpus_hf_config_name: Optional[str] = None
    corpus_hf_split: Optional[str] = "train"
    field_mappings: FieldMapping
    has_explicit_unanswerables: bool = False
    vector_store_collection_name_suffix: str = "_collection"
    
    def get_processed_corpus_file_path(self, base_data_dir: Path, dataset_name_key: str) -> Path:
        """Get the path for the processed corpus file and ensure parent dirs exist."""
        path = base_data_dir / "processed_corpora" / f"{dataset_name_key.lower()}_corpus.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_processed_questions_file_path(self, base_data_dir: Path, dataset_name_key: str) -> Path:
        """Get the path for the processed questions file and ensure parent dirs exist."""
        path = base_data_dir / "processed_questions" / f"{dataset_name_key.lower()}_questions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_vector_store_dir(self, base_vector_store_dir: Path, dataset_name_key: str) -> Path:
        """Get the directory path for the dataset's vector store."""
        path = base_vector_store_dir / f"{dataset_name_key.lower()}_corpus"
        return path
    
    def get_subset_qids_file_path(self, base_data_dir: Path, dataset_name_key: str, limit: int) -> Path:
        """Get the file path for a subset of question IDs and ensure parent dirs exist."""
        path = base_data_dir / "subsets" / f"{dataset_name_key.lower()}_{limit}_qids.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_results_dir(self, base_results_dir: Path, ollama_model_tag: str, dataset_name_key: str) -> Path:
        """Get the directory path for results and ensure it exists."""
        # Replace colon with underscore for valid filename
        sanitized_model_tag = ollama_model_tag.replace(":", "_")
        path = base_results_dir / sanitized_model_tag / dataset_name_key.lower()
        path.mkdir(parents=True, exist_ok=True)
        return path 