"""
Pipeline functions for the LocalRAG system.

This module provides the core RAG pipeline functionality including document
retrieval and the full RAG chain (retrieve, prompt, generate).
"""

import logging
import time
from typing import Dict, List, Tuple, Any

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

from src.config import settings
from src.prompting import format_rag_prompt

# Set up a logger for this module
logger = logging.getLogger(__name__)


def retrieve_documents(
    question_text: str, 
    vector_store: Chroma, 
    embedder: HuggingFaceEmbeddings,
    k: int = settings.retrieval_k
) -> Tuple[List[Document], List[str], float]:
    """
    Retrieve relevant documents from the vector store for a given question.
    Explicitly embeds the query with the correct prefix.
    
    Args:
        question_text: The user question to retrieve passages for
        vector_store: Initialized Chroma vector store containing passages
        embedder: Initialized embedding model used for encoding the question
        k: Number of documents to retrieve (default: from settings.retrieval_k)
        
    Returns:
        Tuple containing:
        - List of Document objects with page_content and metadata
        - List of retrieved document IDs (strings)
        - Retrieval latency in seconds (float)
        
    Note:
        Returns empty lists and zero latency in case of retrieval errors
    """
    try:
        # Start timing
        start_time = time.monotonic()
        
        # Prepare query WITH prefix
        prefixed_query = f"{settings.query_prefix}{question_text}"
        
        # Explicitly embed the prefixed query
        query_embedding = embedder.embed_query(prefixed_query)
        
        # Use similarity_search_by_vector with the provided k value
        retrieved_docs = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
        
        # End timing
        end_time = time.monotonic()
        latency = end_time - start_time
        
        # Extract document IDs from metadata
        retrieved_ids = []
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.metadata.get('original_id', None)
            if doc_id is not None:
                retrieved_ids.append(str(doc_id))  # Ensure ID is string
            else:
                logger.warning(f"Retrieved Doc {i+1} has missing 'original_id' in metadata")
                
        return retrieved_docs, retrieved_ids, latency
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return [], [], 0.0


def run_rag_chain(
    question_text: str,
    vector_store: Chroma,
    embedder: HuggingFaceEmbeddings,
    generator_llm: OllamaLLM,
    k: int = settings.retrieval_k
) -> Dict[str, Any]:
    """
    Run the full RAG chain on a query: retrieve documents, format prompt, and generate answer.
    
    Args:
        question_text: The user question to answer
        vector_store: Initialized Chroma vector store containing passages
        embedder: Initialized embedding model used for retrieval
        generator_llm: Initialized Ollama LLM for answer generation
        k: Number of documents to retrieve (default: from settings.retrieval_k)
        
    Returns:
        Dictionary containing:
        - 'generated_answer': The LLM's response (str or None on error)
        - 'retrieved_contexts': List of passage text strings
        - 'retrieved_ids': List of passage ID strings
        - 'retrieval_latency': Time taken for retrieval in seconds (float)
        - 'generation_latency': Time taken for generation in seconds (float)
        - 'error': Error message string (only present if an error occurred)
    """
    result = {
        'generated_answer': None,
        'retrieved_contexts': [],
        'retrieved_ids': [],
        'retrieval_latency': 0.0,
        'generation_latency': 0.0
    }
    
    try:
        # Step 1: Retrieval - Get relevant documents from the vector store
        retrieved_docs, retrieved_ids, retrieval_latency = retrieve_documents(
            question_text=question_text,
            vector_store=vector_store,
            embedder=embedder,
            k=k
        )
        
        # Update the result dictionary with retrieval data
        result['retrieved_ids'] = retrieved_ids
        result['retrieval_latency'] = retrieval_latency
        result['retrieved_contexts'] = [doc.page_content for doc in retrieved_docs]
        
        # Step 2: Prompt Formatting - Format the RAG prompt
        formatted_prompt = format_rag_prompt(
            question=question_text,
            retrieved_docs=retrieved_docs
        )
        
        # Step 3: Generation - Generate answer using the LLM
        try:
            # Time the generation step
            generation_start_time = time.monotonic()
            
            # Invoke the LLM with the formatted prompt
            generated_answer = generator_llm.invoke(formatted_prompt)
            
            # Calculate generation latency
            generation_end_time = time.monotonic()
            generation_latency = generation_end_time - generation_start_time
            
            # Update result with generation data
            result['generated_answer'] = generated_answer
            result['generation_latency'] = generation_latency
            
        except Exception as e:
            # Handle generation errors
            logger.error(f"Answer generation failed: {e}")
            result['error'] = f"Generation error: {str(e)}"
            result['generation_latency'] = 0.0
    
    except Exception as e:
        # Handle any other errors in the RAG chain
        logger.error(f"RAG chain execution failed: {e}")
        result['error'] = f"RAG chain error: {str(e)}"
    
    return result 