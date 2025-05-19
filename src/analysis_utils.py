"""
Utility functions for analyzing LLM outputs.

This module provides functions to help classify and analyze 
responses from language models, especially for unanswerable questions.
"""

import re


def classify_llm_output_for_unanswerable(raw_llm_output: str) -> str:
    """
    Classifies LLM output for unanswerable questions into categories.
    
    Args:
        raw_llm_output: The raw text output from the LLM
        
    Returns:
        A string category label from one of:
        - "EXACT_UNANSWERABLE": Exact match to "unanswerable"
        - "UNANSWERABLE_NEAR_MATCH": Matches "unanswerable" with punctuation
        - "UNANSWERABLE_EMBEDDED": Contains "unanswerable" within a longer response
        - "OTHER_REFUSAL": Contains other refusal phrases like "I don't know"
        - "ATTEMPTED_ANSWER": Appears to be attempting an answer
        - "UNKNOWN_EMPTY": Empty or whitespace-only response
        - "UNKNOWN_OTHER": Any other response type
    """
    # Handle None or empty inputs
    if not raw_llm_output:
        return "UNKNOWN_EMPTY"
        
    # Normalize the input for comparison
    normalized = raw_llm_output.strip().lower()
    
    # If normalized is empty after stripping, it was only whitespace
    if not normalized:
        return "UNKNOWN_EMPTY"
        
    # EXACT_UNANSWERABLE: Exact match with "unanswerable"
    if normalized == "unanswerable":
        return "EXACT_UNANSWERABLE"
        
    # UNANSWERABLE_NEAR_MATCH: Matches "unanswerable" with non-alphanumeric chars
    if re.match(r'^unanswerable\W*$', normalized):
        return "UNANSWERABLE_NEAR_MATCH"
        
    # UNANSWERABLE_EMBEDDED: Contains "unanswerable" somewhere in the text
    if "unanswerable" in normalized:
        return "UNANSWERABLE_EMBEDDED"
        
    # OTHER_REFUSAL: Contains common refusal phrases but not "unanswerable"
    refusal_phrases = [
        "i don't know", 
        "cannot determine", 
        "not enough information", 
        "no information",
        "insufficient information",
        "cannot answer",
        "unable to answer",
        "can't determine"
    ]
    
    for phrase in refusal_phrases:
        if phrase in normalized:
            return "OTHER_REFUSAL"
            
    # ATTEMPTED_ANSWER: Longer than 15 chars and not matching other categories
    if len(normalized) > 15:
        return "ATTEMPTED_ANSWER"
        
    # UNKNOWN_OTHER: Fallback for anything else
    return "UNKNOWN_OTHER" 