import os
import glob
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader

class RAGSystem:
    def __init__(self, documents_dir: str = "documents"):
        self.documents_dir = documents_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        self.assigned_random_state = 2026

    def load_and_index_documents(self) -> int:
        search_path = os.path.join(self.documents_dir, "*.md")
        markdown_files = glob.glob(search_path)
        
        if not markdown_files:
            raise FileNotFoundError(f"No documents found in {self.documents_dir}")
            
        all_chunks: List[Document] = []
        for file_path in markdown_files:
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            chunks = self.text_splitter.split_documents(loaded_docs)
            all_chunks.extend(chunks)
                
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
        return len(all_chunks)

    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)
