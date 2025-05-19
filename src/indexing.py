"""
Indexing functionality for the Local RAG Pipeline.

This module provides functions to index corpus passages into the vector store
by embedding them using the embedding model.
"""

import logging
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# Set up a logger for this module
logger = logging.getLogger(__name__)


def index_corpus(
    corpus_data: List[Dict],
    vector_store: Chroma,
    embedder: HuggingFaceEmbeddings,
    batch_size: int = 64
) -> None:
    """
    Index the corpus passages into the vector store.
    
    Args:
        corpus_data: List of passage dictionaries from load_corpus_passages
        vector_store: Initialized Chroma vector store
        embedder: Initialized HuggingFaceEmbeddings model
        batch_size: Number of documents to process in each batch
        
    Note:
        Each passage is prefixed with the passage prefix from settings
        before embedding. Original passage IDs and titles are stored in metadata.
        
        Reindexing is now handled externally by creating a fresh vector store.
    """
    total_passages = len(corpus_data)
    logger.info(f"Starting indexing of {total_passages} passages")
    
    # Prepare data for indexing
    texts_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    for p in corpus_data:
        # Create text with prefix for better embedding
        prefixed_text = f"{settings.passage_prefix}{p['text']}"
        
        # Create metadata dictionary
        metadata = {
            'title': p['title'],
            'original_id': p['id']
        }
        
        # Store in lists
        texts_to_add.append(prefixed_text)
        metadatas_to_add.append(metadata)
        ids_to_add.append(p['id'])
    
    # Add documents to vector store in batches
    total_batches = (total_passages + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, total_passages, batch_size):
        batch_end = min(i + batch_size, total_passages)
        batch_num = i // batch_size + 1
        
        batch_texts = texts_to_add[i:batch_end]
        batch_metadatas = metadatas_to_add[i:batch_end]
        batch_ids = ids_to_add[i:batch_end]
        
        try:
            logger.info(f"Indexing batch {batch_num}/{total_batches} (documents {i+1}-{batch_end})")
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            logger.error(f"Error indexing batch {batch_num}: {e}")
            logger.error(f"Batch range: {i+1}-{batch_end}")
            # Continue with next batch rather than failing completely
    
    logger.info(f"Completed indexing of {total_passages} passages into the vector store") 