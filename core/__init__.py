"""
Core Package Initialization Module.

This module exposes the primary domain orchestration layers for the 
Retrieval-Augmented Generation (RAG) system, ensuring streamlined 
imports across the application's presentation and distribution endpoints.
"""

from core.rag_system import RAGSystem
from core.chatbot import Chatbot

# Explicit definition of the public interface for the core package
__all__ = [
    "RAGSystem",
    "Chatbot"
]