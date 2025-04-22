"""
Configuration module for Local RAG Pipeline.

This module uses Pydantic to define and validate configuration settings
for the RAG pipeline components including data paths, model parameters,
vector store settings, and evaluation outputs.
"""

from pathlib import Path

from pydantic import BaseModel, field_validator


class Settings(BaseModel):
    """
    Configuration settings for the Local RAG Pipeline.
    
    Includes settings for data paths, embedding model, vector store,
    LLM parameters, and evaluation metrics output.
    """
    
    # Data Paths
    corpus_dataset_name: str = "PrimeQA/clapnq_passages"
    corpus_dataset_split: str = "train"
    dev_answerable_jsonl_path: str = "data/annotated_data/dev/clapnq_dev_answerable.jsonl"
    dev_unanswerable_jsonl_path: str = "data/annotated_data/dev/clapnq_dev_unanswerable.jsonl"
    retrieval_answerable_tsv_path: str = "data/retrieval/dev/question_dev_answerable.tsv"
    retrieval_unanswerable_tsv_path: str = "data/retrieval/dev/question_dev_unanswerable.tsv"
    
    # Vector Store
    vector_store_persist_dir: str = "./vectorstores/intfloat_e5-base-v2/"
    vector_store_collection_name: str = "clapnq_passages"
    
    # Models
    embedding_model_name: str = "intfloat/e5-base-v2"
    generator_llm_model: str = "gemma3:12b"  # Ollama model tag
    evaluator_gpt_model: str = "gpt-4.1-mini-2025-04-14"  # OpenAI model for RAGAs evaluation
    
    # Embedding Params
    embedding_device: str = "mps"  # Apple Silicon GPU
    embedding_normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    
    # Ollama Params
    ollama_temperature: float = 0.0  # Temperature = 0.0 for deterministic generation
    ollama_num_predict: int = 512  # Chunk size
    ollama_num_gpu: int = 1  # Number of GPU layers for Ollama
    
    # RAG Params
    retrieval_k: int = 10  # Increased from 3 to 10 to improve retrieval metrics
    
    # Output Paths
    results_dir: str = "./results/"
    latency_metrics_file: str = "baseline_latency_metrics.json"
    quality_metrics_file: str = "baseline_quality_metrics.json"
    
    @field_validator('results_dir', 'vector_store_persist_dir', mode='before')
    @classmethod
    def ensure_directory_exists(cls, directory_path: str) -> str:
        """Ensure that directory exists, creating it if necessary."""
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return directory_path
    
    def get_latency_metrics_path(self) -> str:
        """Get the full path to the latency metrics output file."""
        return str(Path(self.results_dir) / self.latency_metrics_file)
    
    def get_quality_metrics_path(self) -> str:
        """Get the full path to the quality metrics output file."""
        return str(Path(self.results_dir) / self.quality_metrics_file)


# Create a singleton instance for importing
settings = Settings() 