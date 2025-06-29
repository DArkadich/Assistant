import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import img2pdf
from pdf2image import convert_from_path
import tempfile
from typing import List, Tuple, Optional
import re
from datetime import datetime

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def process_image(self, image_path: str) -> dict:
        """Обработать изображение документа."""
        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Не удалось загрузить изображение"}
            
            # Предобработка изображения
            processed_image = self._preprocess_image(image)
            
            # OCR распознавание
            text = self._extract_text(processed_image)
            
            # Анализ документа
            doc_info = self._analyze_document(text)
            
            return {
                "success": True,
                "text": text,
                "doc_info": doc_info,
                "image_path": image_path
            }
            
        except Exception as e:
            return {"error": f"Ошибка обработки изображения: {e}"}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для лучшего OCR."""
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Увеличиваем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Убираем шум
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Бинаризация
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _extract_text(self, image: np.ndarray) -> str:
        """Извлечь текст из изображения."""
        try:
            # Настройки OCR для русского языка
            config = '--oem 3 --psm 6 -l rus+eng'
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            print(f"Ошибка OCR: {e}")
            return ""
    
    def _analyze_document(self, text: str) -> dict:
        """Анализировать документ и извлечь ключевую информацию."""
        doc_info = {
            "type": None,
            "number": None,
            "date": None,
            "amount": None,
            "counterparty": None,
            "confidence": 0
        }
        
        if not text:
            return doc_info
        
        # Определяем тип документа
        doc_type = self._detect_document_type(text)
        doc_info["type"] = doc_type
        
        # Извлекаем номер документа
        number = self._extract_document_number(text, doc_type)
        doc_info["number"] = number
        
        # Извлекаем дату
        date = self._extract_date(text)
        doc_info["date"] = date
        
        # Извлекаем сумму
        amount = self._extract_amount(text)
        doc_info["amount"] = amount
        
        # Извлекаем контрагента
        counterparty = self._extract_counterparty(text)
        doc_info["counterparty"] = counterparty
        
        # Рассчитываем уверенность
        confidence = self._calculate_confidence(doc_info)
        doc_info["confidence"] = confidence
        
        return doc_info
    
    def _detect_document_type(self, text: str) -> str:
        """Определить тип документа."""
        text_lower = text.lower()
        
        # Ключевые слова для определения типа
        type_keywords = {
            "накладная": ["накладная", "товарная накладная", "тн", "накладная №"],
            "акт": ["акт", "акт выполненных работ", "акт приема-передачи"],
            "контракт": ["контракт", "договор", "соглашение"],
            "счёт": ["счет", "счёт", "счет-фактура", "счет №"],
            "упд": ["упд", "универсальный передаточный документ"],
            "гтд": ["гтд", "грузовая таможенная декларация", "таможенная декларация"]
        }
        
        for doc_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return doc_type
        
        return "неизвестно"
    
    def _extract_document_number(self, text: str, doc_type: str) -> str:
        """Извлечь номер документа."""
        # Паттерны для номеров документов
        patterns = [
            r"№\s*(\d+[-\w]*)",
            r"номер[:\s]*(\d+[-\w]*)",
            r"№\s*(\d+)",
            r"(\d{2,}[-\w]*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_date(self, text: str) -> str:
        """Извлечь дату из текста."""
        # Паттерны для дат
        date_patterns = [
            r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{1,2}\s+[а-яё]+\s+\d{4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_amount(self, text: str) -> str:
        """Извлечь сумму из текста."""
        # Паттерны для сумм
        amount_patterns = [
            r"(\d{1,3}(?:\s\d{3})*(?:\sруб|\s₽|руб|₽))",
            r"сумма[:\s]*(\d{1,3}(?:\s\d{3})*)",
            r"итого[:\s]*(\d{1,3}(?:\s\d{3})*)"
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_counterparty(self, text: str) -> str:
        """Извлечь название контрагента."""
        # Паттерны для организаций
        org_patterns = [
            r"([А-ЯЁ][а-яё]*(?:\s+[А-ЯЁ][а-яё]*)*\s+(?:ООО|ИП|АО|ЗАО))",
            r"(?:от|поставщик|продавец)[:\s]*([А-ЯЁ][а-яё]*(?:\s+[А-ЯЁ][а-яё]*)*)",
            r"(?:покупатель|заказчик)[:\s]*([А-ЯЁ][а-яё]*(?:\s+[А-ЯЁ][а-яё]*)*)"
        ]
        
        for pattern in org_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _calculate_confidence(self, doc_info: dict) -> int:
        """Рассчитать уверенность в распознавании."""
        confidence = 0
        
        if doc_info["type"] and doc_info["type"] != "неизвестно":
            confidence += 20
        if doc_info["number"]:
            confidence += 20
        if doc_info["date"]:
            confidence += 20
        if doc_info["amount"]:
            confidence += 20
        if doc_info["counterparty"]:
            confidence += 20
        
        return confidence
    
    def images_to_pdf(self, image_paths: List[str], output_path: str) -> bool:
        """Объединить несколько изображений в один PDF."""
        try:
            # Конвертируем изображения в PDF
            with open(output_path, "wb") as f:
                f.write(img2pdf.convert(image_paths))
            return True
        except Exception as e:
            print(f"Ошибка создания PDF: {e}")
            return False
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """Конвертировать PDF в изображения."""
        try:
            # Конвертируем PDF в изображения
            images = convert_from_path(pdf_path)
            image_paths = []
            
            for i, image in enumerate(images):
                temp_path = f"/tmp/page_{i}.jpg"
                image.save(temp_path, "JPEG")
                image_paths.append(temp_path)
            
            return image_paths
        except Exception as e:
            print(f"Ошибка конвертации PDF: {e}")
            return []
    
    def create_pdf_from_text(self, text: str, output_path: str, title: str = "Документ") -> bool:
        """Создать PDF из распознанного текста."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Заголовок
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 12))
            
            # Текст документа
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            return True
        except Exception as e:
            print(f"Ошибка создания PDF из текста: {e}")
            return False

# Глобальный экземпляр процессора изображений
image_processor = ImageProcessor() 