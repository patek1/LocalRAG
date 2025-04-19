"""
Generator LLM initialization for the Local RAG Pipeline.

This module provides functions to initialize and interact with the
Ollama LLM used for generating answers in the RAG pipeline.
"""

import logging
import requests

from langchain_ollama.llms import OllamaLLM

from src.config import settings

# Set up a logger for this module
logger = logging.getLogger(__name__)


def get_generator_llm() -> OllamaLLM:
    """
    Initialize the Ollama LLM for answer generation.
    
    Uses settings from config to initialize an OllamaLLM instance with
    the appropriate model and parameters.
    
    Returns:
        OllamaLLM: Initialized language model ready for generating answers
        
    Raises:
        RuntimeError: If the Ollama service is not running or the model is not available
    """
    try:
        # Check if Ollama service is running before initializing
        try:
            # Make a direct request to Ollama API to check if it's running
            ollama_url = "http://localhost:11434/api/tags"
            response = requests.get(ollama_url, timeout=5)
            
            if response.status_code != 200:
                logger.critical(f"Ollama service responded with status code: {response.status_code}")
                raise RuntimeError("Ollama service is not responding correctly")
            
            # Check if required model is available
            models = response.json().get("models", [])
            model_available = any(model.get("name") == settings.generator_llm_model for model in models)
            
            if not model_available:
                logger.critical(
                    f"Model '{settings.generator_llm_model}' not found in Ollama. "
                    f"Please run 'ollama pull {settings.generator_llm_model}' first."
                )
                raise RuntimeError(f"Model '{settings.generator_llm_model}' not available in Ollama")
                
            logger.info(f"Verified Ollama service is running and model '{settings.generator_llm_model}' is available")
                
        except requests.exceptions.RequestException as e:
            logger.critical(f"Failed to connect to Ollama service: {e}")
            raise RuntimeError(
                "Ollama service not running or not accessible. "
                "Please start Ollama with 'ollama serve'"
            ) from e
            
        # Initialize the Ollama LLM with settings from config
        logger.info(f"Initializing Ollama LLM with model: {settings.generator_llm_model}")
        generator_llm = OllamaLLM(
            model=settings.generator_llm_model,
            temperature=settings.ollama_temperature,
            num_predict=settings.ollama_num_predict,
            num_gpu=settings.ollama_num_gpu
        )
        
        logger.info("Generator LLM initialized successfully")
        return generator_llm
        
    except Exception as e:
        logger.critical(f"Failed to initialize Ollama LLM: {e}")
        raise RuntimeError(f"Failed to initialize generator LLM: {e}") from e 