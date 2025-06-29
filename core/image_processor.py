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
        
        # Определяем и исправляем поворот
        corrected_image = self._fix_rotation(gray)
        
        # Увеличиваем размер изображения для лучшего распознавания
        height, width = corrected_image.shape
        scale_factor = 2.0
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        resized = cv2.resize(corrected_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Простая предобработка - только контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        
        return enhanced
    
    def _fix_rotation(self, image: np.ndarray) -> np.ndarray:
        """Определить и исправить поворот изображения."""
        try:
            # Используем Tesseract для определения ориентации
            osd = pytesseract.image_to_osd(image, config='--psm 0')
            
            # Извлекаем угол поворота
            angle = int(re.search('Rotate: (\d+)', osd).group(1))
            
            if angle != 0:
                # Поворачиваем изображение
                height, width = image.shape
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
            
            return image
            
        except Exception as e:
            print(f"Ошибка определения поворота: {e}")
            # Fallback: пробуем повернуть на 90, 180, 270 градусов
            return self._try_rotation_angles(image)
    
    def _try_rotation_angles(self, image: np.ndarray) -> np.ndarray:
        """Попробовать разные углы поворота и выбрать лучший."""
        angles = [90, 180, 270]
        best_image = image
        best_confidence = 0
        
        for angle in angles:
            height, width = image.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Оцениваем качество текста после поворота
            try:
                config = '--oem 3 --psm 6 -l rus+eng'
                text = pytesseract.image_to_string(rotated, config=config)
                confidence = self._estimate_text_quality(text)
                
                if confidence > best_confidence:
                    best_image = rotated
                    best_confidence = confidence
            except:
                continue
        
        return best_image
    
    def _extract_text(self, image: np.ndarray) -> str:
        """Извлечь текст из изображения с множественными стратегиями."""
        strategies = [
            # Стратегия 1: Простое изображение
            {'name': 'Простое изображение', 'config': '--oem 3 --psm 6 -l rus+eng', 'image': image},
            
            # Стратегия 2: Бинаризованное изображение
            {'name': 'Бинаризованное', 'config': '--oem 3 --psm 6 -l rus+eng', 'image': self._binarize_image(image)},
            
            # Стратегия 3: Разные PSM режимы
            {'name': 'PSM 8', 'config': '--oem 3 --psm 8 -l rus+eng', 'image': image},
            {'name': 'PSM 13', 'config': '--oem 3 --psm 13 -l rus+eng', 'image': image},
            
            # Стратегия 4: Только английский (иногда лучше для цифр)
            {'name': 'Только английский', 'config': '--oem 3 --psm 6 -l eng', 'image': image},
            
            # Стратегия 5: Только русский
            {'name': 'Только русский', 'config': '--oem 3 --psm 6 -l rus', 'image': image},
        ]
        
        best_text = ""
        best_confidence = 0
        best_strategy = "Не найдено"
        
        for strategy in strategies:
            try:
                text = pytesseract.image_to_string(strategy['image'], config=strategy['config'])
                confidence = self._estimate_text_quality(text)
                
                print(f"Стратегия '{strategy['name']}': уверенность {confidence:.2f}, текст: {text[:50]}...")
                
                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence
                    best_strategy = strategy['name']
                    
            except Exception as e:
                print(f"Ошибка стратегии '{strategy['name']}': {e}")
                continue
        
        print(f"Лучшая стратегия: '{best_strategy}' с уверенностью {best_confidence:.2f}")
        return best_text.strip()
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Создать бинаризованную версию изображения."""
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def _estimate_text_quality(self, text: str) -> float:
        """Оценить качество распознанного текста."""
        if not text:
            return 0
        
        # Подсчитываем русские буквы
        russian_chars = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 0
        
        # Процент русских букв
        russian_ratio = russian_chars / total_chars
        
        # Длина текста (предпочитаем более длинные тексты)
        length_score = min(len(text) / 100, 1.0)
        
        # Отсутствие бессмысленных символов
        nonsense_chars = sum(1 for c in text if c in '|[]{}()<>')
        nonsense_penalty = max(0, 1 - nonsense_chars / len(text))
        
        return russian_ratio * length_score * nonsense_penalty
    
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
        
        # Базовые баллы за найденную информацию
        if doc_info["type"] and doc_info["type"] != "неизвестно":
            confidence += 20
        if doc_info["number"]:
            confidence += 15
        if doc_info["date"]:
            confidence += 15
        if doc_info["amount"]:
            confidence += 20
        if doc_info["counterparty"]:
            confidence += 20
        
        # Дополнительные баллы за качество текста
        # (это будет рассчитано в _extract_text)
        
        return min(confidence, 100)
    
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
    
    def create_pdf_from_image(self, image_path: str, output_path: str) -> bool:
        """Создать PDF из изображения с исправленной ориентацией."""
        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Конвертируем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Определяем и исправляем поворот
            corrected_image = self._fix_rotation(gray)
            
            # Конвертируем обратно в BGR для сохранения
            corrected_bgr = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
            
            # Сохраняем исправленное изображение во временный файл
            temp_path = f"/tmp/corrected_{os.path.basename(image_path)}"
            cv2.imwrite(temp_path, corrected_bgr)
            
            # Создаем PDF из исправленного изображения
            success = self.images_to_pdf([temp_path], output_path)
            
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return success
            
        except Exception as e:
            print(f"Ошибка создания PDF из изображения: {e}")
            return False

# Глобальный экземпляр процессора изображений
image_processor = ImageProcessor() 