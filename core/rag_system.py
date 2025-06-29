import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime

class RAGSystem:
    def __init__(self, collection_name="documents"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Инициализация RAG системы."""
        try:
            # Инициализируем модель для эмбеддингов
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Инициализируем ChromaDB
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Создаем или получаем коллекцию
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Подключен к существующей коллекции: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Коллекция документов для семантического поиска"}
                )
                print(f"✅ Создана новая коллекция: {self.collection_name}")
                
        except Exception as e:
            print(f"❌ Ошибка инициализации RAG системы: {e}")
            self.embedding_model = None
            self.client = None
            self.collection = None
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Создать эмбеддинг для текста."""
        if not self.embedding_model or not text:
            return None
        
        try:
            # Очищаем текст
            clean_text = self._clean_text(text)
            if not clean_text:
                return None
            
            # Создаем эмбеддинг
            embedding = self.embedding_model.encode(clean_text)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Ошибка создания эмбеддинга: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста для лучшего эмбеддинга."""
        if not text:
            return ""
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Убираем специальные символы, но оставляем русские и английские буквы, цифры
        text = re.sub(r'[^\w\sа-яёА-ЯЁ]', ' ', text)
        
        # Убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def add_document(self, doc_id: str, content: str, metadata: Dict) -> bool:
        """Добавить документ в векторную базу."""
        if not self.collection or not content:
            return False
        
        try:
            # Создаем эмбеддинг
            embedding = self.create_embedding(content)
            if not embedding:
                return False
            
            # Подготавливаем метаданные
            clean_metadata = self._prepare_metadata(metadata)
            
            # Добавляем в коллекцию
            self.collection.add(
                embeddings=[embedding],
                documents=[content[:1000]],  # Ограничиваем длину для ChromaDB
                metadatas=[clean_metadata],
                ids=[doc_id]
            )
            
            print(f"✅ Документ {doc_id} добавлен в векторную базу")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления документа в RAG: {e}")
            return False
    
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """Подготовка метаданных для ChromaDB."""
        clean_metadata = {}
        
        # Копируем только строковые значения
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = str(value)
            elif isinstance(value, list):
                clean_metadata[key] = ', '.join(map(str, value))
        
        return clean_metadata
    
    def search_documents(self, query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Семантический поиск документов."""
        if not self.collection or not query:
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                return []
            
            # Выполняем поиск
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Формируем результат
            documents = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc = {
                        'id': doc_id,
                        'content': results['documents'][0][i] if results['documents'] and results['documents'][0] else "",
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Ошибка поиска в RAG: {e}")
            return []
    
    def search_by_type(self, doc_type: str, query: str = "", n_results: int = 5) -> List[Dict]:
        """Поиск документов определенного типа."""
        # Комбинируем тип документа с запросом для семантического поиска
        search_query = f"{doc_type} {query}".strip()
        results = self.search_documents(search_query, n_results * 2)  # Берем больше результатов для фильтрации
        
        # Фильтруем по типу документа
        filtered_results = []
        for doc in results:
            metadata = doc.get('metadata', {})
            if metadata.get('type', '').lower() == doc_type.lower():
                filtered_results.append(doc)
                if len(filtered_results) >= n_results:
                    break
        
        return filtered_results
    
    def search_by_counterparty(self, counterparty: str, query: str = "", n_results: int = 5) -> List[Dict]:
        """Поиск документов по контрагенту."""
        # Комбинируем название контрагента с запросом для семантического поиска
        search_query = f"{counterparty} {query}".strip()
        results = self.search_documents(search_query, n_results * 2)  # Берем больше результатов для фильтрации
        
        # Фильтруем по контрагенту (частичное совпадение)
        filtered_results = []
        counterparty_lower = counterparty.lower()
        for doc in results:
            metadata = doc.get('metadata', {})
            counterparty_name = metadata.get('counterparty_name', '').lower()
            if counterparty_lower in counterparty_name or counterparty_name in counterparty_lower:
                filtered_results.append(doc)
                if len(filtered_results) >= n_results:
                    break
        
        return filtered_results
    
    def search_by_amount_range(self, min_amount: float = None, max_amount: float = None, query: str = "", n_results: int = 5) -> List[Dict]:
        """Поиск документов по диапазону сумм."""
        filters = {}
        if min_amount is not None:
            filters["amount_min"] = str(min_amount)
        if max_amount is not None:
            filters["amount_max"] = str(max_amount)
        
        return self.search_documents(query, n_results, filters)
    
    def update_document(self, doc_id: str, content: str, metadata: Dict) -> bool:
        """Обновить документ в векторной базе."""
        # Сначала удаляем старую версию
        self.delete_document(doc_id)
        # Добавляем новую версию
        return self.add_document(doc_id, content, metadata)
    
    def delete_document(self, doc_id: str) -> bool:
        """Удалить документ из векторной базы."""
        if not self.collection:
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            print(f"✅ Документ {doc_id} удален из векторной базы")
            return True
        except Exception as e:
            print(f"❌ Ошибка удаления документа из RAG: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Получить статистику коллекции."""
        if not self.collection:
            return {"error": "Коллекция не инициализирована"}
        
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "status": "active"
            }
        except Exception as e:
            return {"error": f"Ошибка получения статистики: {e}"}
    
    def clear_collection(self) -> bool:
        """Очистить всю коллекцию."""
        if not self.collection:
            return False
        
        try:
            self.collection.delete()
            print("✅ Коллекция очищена")
            return True
        except Exception as e:
            print(f"❌ Ошибка очистки коллекции: {e}")
            return False

# Глобальный экземпляр RAG системы
rag_system = RAGSystem() 