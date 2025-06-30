"""
AI-консьерж для работы с документами: OCR, теги, поиск, связи с задачами
"""
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Set
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re
from pathlib import Path
import shutil

class DocumentAssistant:
    def __init__(self):
        self.data = {
            'documents': [],  # список документов
            'tags': set(),  # множество всех тегов
            'task_links': {},  # связи с задачами
            'settings': {
                'ocr_enabled': True,
                'storage_path': 'documents_storage',
                'ocr_cache_path': 'ocr_cache'
            }
        }
        self.DATA_FILE = 'documents_data.json'
        self._load_data()
        self._ensure_directories()
    
    def _load_data(self):
        """Загрузка данных из файла"""
        if os.path.exists(self.DATA_FILE):
            with open(self.DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data['documents'] = data.get('documents', [])
                self.data['tags'] = set(data.get('tags', []))
                self.data['task_links'] = data.get('task_links', {})
                self.data['settings'].update(data.get('settings', {}))
    
    def _save_data(self):
        """Сохранение данных в файл"""
        data_to_save = {
            'documents': self.data['documents'],
            'tags': list(self.data['tags']),
            'task_links': self.data['task_links'],
            'settings': self.data['settings']
        }
        with open(self.DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    def _ensure_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(self.data['settings']['storage_path'], exist_ok=True)
        os.makedirs(self.data['settings']['ocr_cache_path'], exist_ok=True)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Извлечение текста из PDF"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Ошибка извлечения текста из PDF: {e}")
            return ""
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """Извлечение текста из изображения через OCR"""
        try:
            # Проверяем кэш OCR
            cache_path = Path(self.data['settings']['ocr_cache_path']) / f"{Path(file_path).stem}_ocr.txt"
            if cache_path.exists():
                return cache_path.read_text(encoding='utf-8')
            
            # Выполняем OCR
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='rus+eng')
            
            # Сохраняем в кэш
            cache_path.write_text(text, encoding='utf-8')
            return text
        except Exception as e:
            print(f"Ошибка OCR: {e}")
            return ""
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Определение типа документа по содержимому и имени"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        type_patterns = {
            'акт': r'акт|выполненных? работ',
            'счет': r'счет|счёт|invoice',
            'договор': r'договор|контракт|agreement',
            'накладная': r'накладная|товарная накладная',
            'спецификация': r'спецификация|specification',
            'письмо': r'письмо|уведомление|notice'
        }
        
        for doc_type, pattern in type_patterns.items():
            if re.search(pattern, text_lower) or re.search(pattern, filename_lower):
                return doc_type
        
        return 'другое'
    
    def _extract_metadata(self, text: str, filename: str) -> Dict:
        """Извлечение метаданных из документа"""
        text_lower = text.lower()
        
        # Поиск даты
        date_patterns = [
            r'от (\d{1,2}[./-]\d{1,2}[./-]\d{2,4})',
            r'«(\d{1,2})»\s+([а-яё]+)\s+(\d{4})',
            r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})'
        ]
        doc_date = None
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                # Преобразование даты в стандартный формат
                # TODO: добавить обработку различных форматов
                doc_date = match.group(1)
                break
        
        # Поиск номера
        number_match = re.search(r'(?:№|номер|n)\s*([a-zа-я0-9-/]+)', text_lower)
        doc_number = number_match.group(1) if number_match else None
        
        # Поиск суммы
        amount_match = re.search(r'сумма:?\s*(\d+(?:\s*\d+)*(?:[.,]\d+)?)\s*(?:руб|₽|rub)', text_lower)
        amount = amount_match.group(1).replace(' ', '') if amount_match else None
        
        # Поиск контрагента
        # TODO: улучшить поиск контрагента с использованием NER
        
        return {
            'date': doc_date,
            'number': doc_number,
            'amount': amount
        }
    
    def _check_signatures(self, file_path: str) -> bool:
        """Проверка наличия подписей в документе"""
        # TODO: реализовать проверку подписей через CV
        return True
    
    def add_document(
        self,
        file_path: str,
        tags: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Добавление нового документа"""
        # Копируем файл в хранилище
        filename = Path(file_path).name
        new_path = Path(self.data['settings']['storage_path']) / filename
        shutil.copy2(file_path, new_path)
        
        # Извлекаем текст
        text = ""
        if filename.lower().endswith('.pdf'):
            text = self._extract_text_from_pdf(new_path)
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            text = self._extract_text_from_image(new_path)
        
        # Определяем тип документа и метаданные
        doc_type = self._detect_document_type(text, filename)
        extracted_metadata = self._extract_metadata(text, filename)
        if metadata:
            extracted_metadata.update(metadata)
        
        # Проверяем подписи
        has_signatures = self._check_signatures(new_path)
        
        # Создаем запись о документе
        document = {
            'id': f"doc_{len(self.data['documents']) + 1}",
            'filename': filename,
            'path': str(new_path),
            'type': doc_type,
            'metadata': extracted_metadata,
            'tags': tags or [],
            'has_signatures': has_signatures,
            'text_content': text,
            'added_at': datetime.now().isoformat(),
            'task_id': task_id
        }
        
        # Обновляем данные
        self.data['documents'].append(document)
        if tags:
            self.data['tags'].update(tags)
        if task_id:
            if task_id not in self.data['task_links']:
                self.data['task_links'][task_id] = []
            self.data['task_links'][task_id].append(document['id'])
        
        self._save_data()
        return document
    
    def search_documents(
        self,
        query: str = None,
        doc_type: str = None,
        tags: List[str] = None,
        date_from: str = None,
        date_to: str = None,
        has_signatures: bool = None,
        task_id: str = None
    ) -> List[Dict]:
        """Поиск документов по различным критериям"""
        results = []
        
        for doc in self.data['documents']:
            # Фильтр по типу
            if doc_type and doc['type'] != doc_type:
                continue
            
            # Фильтр по тегам
            if tags and not all(tag in doc['tags'] for tag in tags):
                continue
            
            # Фильтр по подписям
            if has_signatures is not None and doc['has_signatures'] != has_signatures:
                continue
            
            # Фильтр по задаче
            if task_id and doc['task_id'] != task_id:
                continue
            
            # Фильтр по дате
            if doc['metadata'].get('date'):
                if date_from and doc['metadata']['date'] < date_from:
                    continue
                if date_to and doc['metadata']['date'] > date_to:
                    continue
            
            # Полнотекстовый поиск
            if query:
                query_lower = query.lower()
                searchable_text = (
                    f"{doc['filename']} {doc['type']} "
                    f"{doc['text_content']} {' '.join(doc['tags'])} "
                    f"{json.dumps(doc['metadata'], ensure_ascii=False)}"
                ).lower()
                
                if query_lower not in searchable_text:
                    continue
            
            results.append(doc)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Получение документа по ID"""
        for doc in self.data['documents']:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def update_document(self, doc_id: str, updates: Dict) -> Optional[Dict]:
        """Обновление информации о документе"""
        for i, doc in enumerate(self.data['documents']):
            if doc['id'] == doc_id:
                self.data['documents'][i].update(updates)
                self._save_data()
                return self.data['documents'][i]
        return None
    
    def get_documents_without_signatures(self) -> List[Dict]:
        """Получение списка документов без подписей"""
        return [doc for doc in self.data['documents'] if not doc['has_signatures']]
    
    def get_document_tasks(self, doc_id: str) -> List[str]:
        """Получение списка задач, связанных с документом"""
        tasks = []
        for task_id, doc_ids in self.data['task_links'].items():
            if doc_id in doc_ids:
                tasks.append(task_id)
        return tasks
    
    def link_document_to_task(self, doc_id: str, task_id: str) -> bool:
        """Связывание документа с задачей"""
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return False
        
        if task_id not in self.data['task_links']:
            self.data['task_links'][task_id] = []
        
        if doc_id not in self.data['task_links'][task_id]:
            self.data['task_links'][task_id].append(doc_id)
            self._save_data()
        
        return True

# Глобальный экземпляр для использования в других модулях
document_assistant = DocumentAssistant() 