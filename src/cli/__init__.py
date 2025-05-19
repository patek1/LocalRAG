"""
Command Line Interface package for the Local RAG pipeline.

This package provides utilities for parsing and handling
command line arguments for the various scripts in the project.
"""

from .parser import parse_arguments

__all__ = ["parse_arguments"] 