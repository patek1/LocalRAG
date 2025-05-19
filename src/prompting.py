"""
Prompt formatting utilities for the Local RAG Pipeline.

This module provides functions for formatting prompts used in the
retrieval-augmented generation (RAG) chain.
"""

import logging
from typing import List

from langchain.docstore.document import Document

# Set up a logger for this module
logger = logging.getLogger(__name__)


def format_rag_prompt(question: str, retrieved_docs: List[Document]) -> str:
    """
    Format the RAG prompt for the generator LLM using retrieved documents
    and the original question.
    
    Args:
        question: The original user question
        retrieved_docs: List of Document objects containing the retrieved passages
        
    Returns:
        str: Formatted prompt string with context passages and question
        
    Example:
        >>> format_rag_prompt("What is RAG?", [doc1, doc2, doc3])
        "Context passages:
        ---
        Passage 1 (Title: RAG Introduction): Retrieved text about RAG...
        ---
        Passage 2 (Title: Another Title): More text...
        ---
        Passage 3 (Title: Last Title): Final text...
        ---
        Question: What is RAG?
        If the context passages do not contain enough information to answer the question, respond with the single word 'unanswerable'.
        Answer:"
    """
    # Start building the prompt with the header
    prompt_parts = ["Context passages:"]
    
    # If no documents were retrieved, note this in the prompt
    if not retrieved_docs:
        prompt_parts.append("No relevant passages found.")
    else:
        # Add each retrieved document as a passage
        for i, doc in enumerate(retrieved_docs):
            # Extract title from metadata (default to "Unknown" if not present)
            title = doc.metadata.get('title', 'Unknown')
            
            # Format the passage with title and separator
            passage = f"---\nPassage {i+1} (Title: {title}): {doc.page_content}"
            prompt_parts.append(passage)
    
    # Add final separator and the question
    prompt_parts.append("---")
    prompt_parts.append(f"Question: {question}")
    # Add explicit instruction for handling unanswerable questions
    prompt_parts.append("If the context passages do not contain enough information to answer the question, respond with the single word 'unanswerable'.")
    prompt_parts.append("Answer:")
    
    # Join all parts with newlines to create the final prompt
    formatted_prompt = "\n".join(prompt_parts)
    
    # Only log if debugging is enabled
    if logger.isEnabledFor(logging.DEBUG):
        # Log the formatted prompt at debug level with much shorter preview
        max_log_length = 200  # Reduced from 500
        log_prompt = formatted_prompt
        if len(log_prompt) > max_log_length:
            log_prompt = log_prompt[:max_log_length] + "... [truncated]"
        logger.debug(f"Generated RAG prompt: {log_prompt}")
    
    return formatted_prompt 