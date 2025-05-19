"""
Vector store initialization for the Local RAG Pipeline.

This module provides functions to initialize and interact with ChromaDB
for persistent storage of vector embeddings.
"""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set up a logger for this module
logger = logging.getLogger(__name__)


def get_vector_store(embedding_model: HuggingFaceEmbeddings, 
                     persist_directory: Optional[str] = None,
                     collection_name: Optional[str] = None) -> Chroma:
    """
    Initialize the Chroma vector store with the specified embedding model.
    
    Args:
        embedding_model: The initialized embedding model to use for document embedding
        persist_directory: Directory path for where to persist the vector store.
                          Required parameter now that global setting is removed.
        collection_name: Name of the collection to use or create.
                        Required parameter now that global setting is removed.
    
    Returns:
        Chroma: Initialized vector store ready for document storage and retrieval
        
    Raises:
        ValueError: If persist_directory or collection_name parameters are not provided
        RuntimeError: If there's a critical error initializing the vector store
        
    Note:
        When persist_directory is provided, Chroma automatically persists data
        without requiring explicit persist() calls.
    """
    try:
        if not persist_directory:
            raise ValueError("persist_directory must be provided")
        
        if not collection_name:
            raise ValueError("collection_name must be provided")
        
        logger.info(
            f"Initializing Chroma vector store at '{persist_directory}' "
            f"with collection '{collection_name}'"
        )
        
        # Initialize the vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        
        logger.info("Successfully initialized Chroma vector store")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error initializing Chroma vector store: {e}")
        raise RuntimeError(f"Failed to initialize vector store: {e}") from e 