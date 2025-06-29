"""
RAG System - Temporarily Disabled
This module is temporarily disabled due to heavy dependencies.
"""

class RAGSystem:
    def __init__(self):
        self.is_enabled = False
        print("RAG System is temporarily disabled")
    
    def add_document(self, text, metadata):
        print("RAG add_document: System disabled")
        return False
    
    def search(self, query, limit=5):
        print("RAG search: System disabled")
        return []
    
    def search_documents(self, query, n_results=5):
        print("RAG search_documents: System disabled")
        return []
    
    def search_by_type(self, doc_type, query="", n_results=5):
        print("RAG search_by_type: System disabled")
        return []
    
    def search_by_counterparty(self, counterparty, query="", n_results=5):
        print("RAG search_by_counterparty: System disabled")
        return []
    
    def get_all_documents(self):
        print("RAG get_all_documents: System disabled")
        return []
    
    def get_collection_stats(self):
        print("RAG get_collection_stats: System disabled")
        return {"error": "RAG system is disabled"}

# Global instance
rag_system = RAGSystem() 