"""
Embedding model initialization for the Local RAG Pipeline.

This module provides functions to initialize HuggingFace embedding models
configured for Apple Silicon MPS acceleration.
"""

import logging
from typing import Literal, Union

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# Set up a logger for this module
logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model with MPS acceleration if available.
    
    Uses the embedding model specified in settings and attempts to configure for MPS
    if on Apple Silicon and GPU acceleration is available.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings model ready to use
        
    Raises:
        RuntimeError: If there's a critical error initializing the embedding model
    """
    try:
        # Check MPS availability
        device: Union[Literal["cpu"], Literal["cuda"], Literal["mps"]] = "cpu"
        
        # Try to use MPS if settings specify it and it's available
        if settings.embedding_device == "mps":
            if torch.backends.mps.is_available():
                logger.info(
                    "MPS (Metal Performance Shaders) is available. "
                    "Using Apple Silicon GPU acceleration for embeddings."
                )
                device = "mps"
            else:
                if torch.backends.mps.is_built():
                    logger.warning(
                        "MPS is built into PyTorch but not available. "
                        "You may be missing macOS 12.3+ or Apple Silicon. "
                        "Falling back to CPU for embeddings."
                    )
                else:
                    logger.warning(
                        "MPS is not built into this PyTorch installation. "
                        "Consider reinstalling PyTorch with Metal support. "
                        "Falling back to CPU for embeddings."
                    )
        
        # Initialize the embedding model
        logger.info(
            f"Initializing HuggingFaceEmbeddings model: {settings.embedding_model_name} on {device}"
        )
        
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": settings.embedding_normalize},
        )
        
        logger.info(f"Successfully initialized {settings.embedding_model_name} embedding model")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise RuntimeError(f"Failed to initialize embedding model: {e}") from e 