"""
Global settings module for the Local RAG Pipeline.

This module defines the Settings class which holds configuration for
all aspects of the pipeline, including paths, model parameters, and
dataset-specific configurations.
"""

from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel, Field, field_validator, root_validator

from .dataset_config import DatasetConfig, FieldMapping


class Settings(BaseModel):
    """
    Configuration settings for the Local RAG Pipeline.
    
    Includes settings for data paths, embedding model, vector store,
    LLM parameters, and evaluation metrics output.
    """
    
    # Data Directory Paths
    base_data_dir: str = "data"
    base_vector_store_dir: str = "vectorstores"
    
    # Dataset configurations
    dataset_configs: Dict[str, DatasetConfig] = Field(default_factory=dict)
    
    # Data Paths (To be moved to FieldMapping)
    retrieval_answerable_tsv_path: str = "data/retrieval/dev/question_dev_answerable.tsv"  # To be moved to FieldMapping
    retrieval_unanswerable_tsv_path: str = "data/retrieval/dev/question_dev_unanswerable.tsv"  # To be moved to FieldMapping
    
    # Models
    embedding_model_name: str = "intfloat/e5-base-v2"
    evaluator_gpt_model: str = "gpt-4.1-mini-2025-04-14"  # OpenAI model for RAGAs evaluation
    
    # Embedding Params
    embedding_device: str = "mps"  # Apple Silicon GPU
    embedding_normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    
    # Ollama Params
    ollama_temperature: float = 0.0  # Temperature = 0.0 for deterministic generation
    ollama_num_predict: int = 512  # Chunk size
    ollama_num_gpu: int = 999  # Number of GPU layers for Ollama (changed from 1 to 999)
    
    # RAG Params
    retrieval_k: int = 10  # Increased from 3 to 10 to improve retrieval metrics
    
    # Output Paths
    results_dir: str = "./results/"
    latency_metrics_file: str = "baseline_latency_metrics.json"  # Legacy, will be updated in Phase 4
    quality_metrics_file: str = "baseline_quality_metrics.json"  # Legacy, will be updated in Phase 4
    
    @field_validator('results_dir', 'base_data_dir', 'base_vector_store_dir', mode='before')
    @classmethod
    def ensure_directory_exists(cls, directory_path: str) -> str:
        """Ensure that directory exists, creating it if necessary."""
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return directory_path
    
    def get_active_dataset_config(self, dataset_name_key: str) -> DatasetConfig:
        """
        Get the DatasetConfig for the specified dataset name.
        
        Args:
            dataset_name_key: The key for the dataset configuration (e.g., "ClapNQ")
            
        Returns:
            The DatasetConfig for the specified dataset
            
        Raises:
            ValueError: If the dataset_name_key is not found in dataset_configs
        """
        if dataset_name_key not in self.dataset_configs:
            raise ValueError(
                f"Dataset '{dataset_name_key}' not found. Available datasets: {', '.join(self.dataset_configs.keys())}"
            )
        return self.dataset_configs[dataset_name_key]
    
    def get_latency_metrics_path(self) -> str:
        """Get the full path to the latency metrics output file (LEGACY)."""
        return str(Path(self.results_dir) / self.latency_metrics_file)
    
    def get_quality_metrics_path(self) -> str:
        """Get the full path to the quality metrics output file (LEGACY)."""
        return str(Path(self.results_dir) / self.quality_metrics_file)
    
    @root_validator(skip_on_failure=True)
    @classmethod
    def populate_dataset_configs(cls, values) -> Dict[str, Any]:
        """
        Populate the dataset_configs dictionary with default configurations for all supported datasets.
        
        This class method is a Pydantic root_validator that runs once during Settings initialization
        and ensures all dataset configs are properly set up.
        
        Args:
            values: Dictionary of field values being validated
            
        Returns:
            The updated values dictionary with dataset_configs populated
        """
        dataset_configs = values.get("dataset_configs", {})
        
        # ClapNQ configuration
        dataset_configs["ClapNQ"] = DatasetConfig(
            hf_id="PrimeQA/clapnq",
            hf_split_eval="validation",
            corpus_hf_id="PrimeQA/clapnq_passages",
            has_explicit_unanswerables=True,
            field_mappings=FieldMapping(
                question_id="id",
                question_text="input",
                answer_text="output[0].answer",  # Nested answer field in ClapNQ
                passage_id="id",  # Explicit mapping for passage_id
                passage_content_fields=["text"],
                passage_title_field=["title"],
                qrels_ans_path=values.get("retrieval_answerable_tsv_path"),
                qrels_unans_path=values.get("retrieval_unanswerable_tsv_path")
            )
        )
        
        # TriviaQA configuration - properly implemented based on dataset inspection
        dataset_configs["TriviaQA"] = DatasetConfig(
            hf_id="trivia_qa",
            hf_config_name="rc",  # Using the reading comprehension config which has entity pages and search results
            hf_split_train="train",
            hf_split_eval="validation",
            corpus_hf_id="trivia_qa",  # Using the same dataset for corpus extraction
            corpus_hf_config_name="rc",
            corpus_hf_split="train",
            has_explicit_unanswerables=False,  # TriviaQA questions are generally answerable
            vector_store_collection_name_suffix="_triviaqa_corpus_collection",
            field_mappings=FieldMapping(
                # Question fields
                question_id="question_id",
                question_text="question",
                answer_text="answer.value",  # Direct path to the answer value
                
                # Passage fields - will be processed specially in the adapter function
                # TriviaQA has two main sources of passages:
                # 1. entity_pages.wiki_context - Wikipedia articles relevant to the entity
                # 2. search_results.search_context - Web search results
                passage_content_fields=["entity_pages.wiki_context", "search_results.search_context"],
                passage_title_field=["entity_pages.title", "search_results.title"],
                
                # The processor will need to:
                # 1. Extract all unique passages from both sources
                # 2. Assign unique IDs to each passage (e.g., "triviaqa_wiki_{idx}" or "triviaqa_search_{idx}")
                # 3. For each question, identify which passages were originally associated with it
                #    These become the gold_passage_ids for retrieval evaluation
            )
        )
        
        # HotpotQA configuration based on dataset inspection
        dataset_configs["HotpotQA"] = DatasetConfig(
            hf_id="hotpot_qa",
            hf_config_name="distractor",  # 'distractor' config as specified in the spec
            hf_split_train="train",       # Used for building the global corpus of passages
            hf_split_eval="validation",   # Used for evaluation questions
            corpus_hf_id="hotpot_qa",     # Using the same dataset for corpus extraction
            corpus_hf_config_name="distractor",
            corpus_hf_split="train",      # Build corpus from train set contexts
            has_explicit_unanswerables=False,  # HotpotQA questions are generally answerable
            vector_store_collection_name_suffix="_hotpotqa_corpus_collection",
            field_mappings=FieldMapping(
                # Question fields - based on dataset inspection
                question_id="id",           # Question ID field is 'id'
                question_text="question",   # Question text field is 'question'
                answer_text="answer",       # Answer text field is 'answer'
                
                # Context structure is special in HotpotQA:
                # Each item has a 'context' dict with keys 'title' and 'sentences'
                # 'title' is a list of document titles
                # 'sentences' is a list of lists, where each inner list has sentences for a document
                # This requires special processing, and passage_content_fields doesn't directly apply
                # But we'll keep it for compatibility, and handle the special structure in the processor
                passage_content_fields=["context"],
                
                # Gold passage identification will use 'supporting_facts' which contains:
                # - 'title': list of document titles that contain supporting facts
                # - 'sent_id': list of sentence indices within those documents
                # This needs custom handling in the processor function, as it's a more complex structure
                gold_passage_identifier_field_in_question_data="supporting_facts"
            )
        )
        
        values["dataset_configs"] = dataset_configs
        return values


# Create a singleton instance for importing
settings = Settings() 