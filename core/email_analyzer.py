"""
Email Analyzer - Анализ переписки через IMAP
Модуль для чтения и анализа email через Gmail/Яндекс.Почту
"""

import imaplib
import email
import re
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from email.header import decode_header
from email.utils import parsedate_to_datetime
import openai
from dataclasses import dataclass
from enum import Enum

class EmailPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class EmailStatus(Enum):
    NEW = "new"
    READ = "read"
    REPLIED = "replied"
    FORWARDED = "forwarded"
    ARCHIVED = "archived"

@dataclass
class EmailMessage:
    id: str
    subject: str
    sender: str
    sender_email: str
    date: datetime
    content: str
    priority: EmailPriority
    status: EmailStatus
    labels: List[str]
    thread_id: str
    is_reply_needed: bool
    summary: str = ""

class EmailAnalyzer:
    def __init__(self):
        self.imap_connection = None
        self.email_config = self._load_config()
        self.openai_client = None
        self._initialize_openai()
    
    def _load_config(self) -> Dict:
        """Загрузка конфигурации email."""
        config_path = "email_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Создаем шаблон конфигурации
            config = {
                "gmail": {
                    "enabled": False,
                    "email": "",
                    "password": "",
                    "imap_server": "imap.gmail.com",
                    "imap_port": 993
                },
                "yandex": {
                    "enabled": False,
                    "email": "",
                    "password": "",
                    "imap_server": "imap.yandex.ru",
                    "imap_port": 993
                },
                "auto_summary": True,
                "priority_keywords": {
                    "urgent": ["срочно", "urgent", "немедленно", "asap"],
                    "high": ["важно", "important", "критично", "critical"],
                    "medium": ["обычно", "normal", "стандартно"],
                    "low": ["неважно", "low priority", "низкий приоритет"]
                },
                "reply_needed_keywords": [
                    "ответьте", "ответить", "reply", "жду ответ", "жду ответа",
                    "подтвердите", "подтверждение", "confirmation", "confirm"
                ]
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return config
    
    def _initialize_openai(self):
        """Инициализация OpenAI клиента."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI клиент инициализирован")
            else:
                print("⚠️ OpenAI API ключ не найден")
        except Exception as e:
            print(f"❌ Ошибка инициализации OpenAI: {e}")
    
    def connect_imap(self, provider: str = "gmail") -> bool:
        """Подключение к IMAP серверу."""
        try:
            config = self.email_config.get(provider, {})
            if not config.get("enabled", False):
                print(f"❌ {provider} не настроен")
                return False
            
            server = config["imap_server"]
            port = config["imap_port"]
            email_addr = config["email"]
            password = config["password"]
            
            # Подключаемся к серверу
            self.imap_connection = imaplib.IMAP4_SSL(server, port)
            self.imap_connection.login(email_addr, password)
            
            print(f"✅ Подключен к {provider} IMAP")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подключения к {provider}: {e}")
            return False
    
    def disconnect_imap(self):
        """Отключение от IMAP сервера."""
        if self.imap_connection:
            try:
                self.imap_connection.logout()
                self.imap_connection = None
                print("✅ Отключен от IMAP сервера")
            except Exception as e:
                print(f"❌ Ошибка отключения: {e}")
    
    def get_inbox_messages(self, limit: int = 50, days_back: int = 7) -> List[EmailMessage]:
        """Получение сообщений из входящих."""
        if not self.imap_connection:
            print("❌ Нет подключения к IMAP")
            return []
        
        try:
            # Выбираем папку INBOX
            self.imap_connection.select('INBOX')
            
            # Формируем дату для поиска
            date_since = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            
            # Ищем сообщения
            search_criteria = f'(SINCE "{date_since}")'
            _, message_numbers = self.imap_connection.search(None, search_criteria)
            
            messages = []
            email_list = message_numbers[0].split()
            
            # Ограничиваем количество сообщений
            if limit:
                email_list = email_list[-limit:]
            
            for num in email_list:
                try:
                    _, msg_data = self.imap_connection.fetch(num, '(RFC822)')
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    # Парсим сообщение
                    parsed_message = self._parse_email_message(email_message, num.decode())
                    if parsed_message:
                        messages.append(parsed_message)
                        
                except Exception as e:
                    print(f"❌ Ошибка парсинга сообщения {num}: {e}")
                    continue
            
            return messages
            
        except Exception as e:
            print(f"❌ Ошибка получения сообщений: {e}")
            return []
    
    def _parse_email_message(self, email_message, msg_id: str) -> Optional[EmailMessage]:
        """Парсинг email сообщения."""
        try:
            # Извлекаем заголовки
            subject = self._decode_header(email_message.get('Subject', ''))
            sender = self._decode_header(email_message.get('From', ''))
            date_str = email_message.get('Date', '')
            
            # Парсим дату
            try:
                date = parsedate_to_datetime(date_str)
            except:
                date = datetime.now()
            
            # Извлекаем email отправителя
            sender_email = self._extract_email(sender)
            
            # Извлекаем содержимое
            content = self._extract_content(email_message)
            
            # Определяем приоритет
            priority = self._determine_priority(subject, content)
            
            # Определяем статус
            status = self._determine_status(email_message)
            
            # Определяем метки
            labels = self._extract_labels(email_message)
            
            # Определяем thread_id
            thread_id = email_message.get('Message-ID', msg_id)
            
            # Определяем, нужен ли ответ
            is_reply_needed = self._needs_reply(subject, content)
            
            return EmailMessage(
                id=msg_id,
                subject=subject,
                sender=sender,
                sender_email=sender_email,
                date=date,
                content=content,
                priority=priority,
                status=status,
                labels=labels,
                thread_id=thread_id,
                is_reply_needed=is_reply_needed
            )
            
        except Exception as e:
            print(f"❌ Ошибка парсинга сообщения: {e}")
            return None
    
    def _decode_header(self, header: str) -> str:
        """Декодирование заголовка."""
        try:
            decoded_parts = decode_header(header)
            decoded_string = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += part
            return decoded_string
        except:
            return str(header)
    
    def _extract_email(self, sender: str) -> str:
        """Извлечение email адреса из строки отправителя."""
        email_pattern = r'<([^>]+)>'
        match = re.search(email_pattern, sender)
        if match:
            return match.group(1)
        
        # Если нет угловых скобок, ищем email в тексте
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, sender)
        if match:
            return match.group(0)
        
        return sender
    
    def _extract_content(self, email_message) -> str:
        """Извлечение содержимого email."""
        content = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        content += part.get_payload(decode=True).decode('latin-1', errors='ignore')
        else:
            try:
                content = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                content = email_message.get_payload(decode=True).decode('latin-1', errors='ignore')
        
        return content
    
    def _determine_priority(self, subject: str, content: str) -> EmailPriority:
        """Определение приоритета сообщения."""
        text = (subject + " " + content).lower()
        keywords = self.email_config.get("priority_keywords", {})
        
        for priority, words in keywords.items():
            for word in words:
                if word.lower() in text:
                    return EmailPriority(priority)
        
        return EmailPriority.MEDIUM
    
    def _determine_status(self, email_message) -> EmailStatus:
        """Определение статуса сообщения."""
        # Проверяем флаги сообщения
        flags = email_message.get('X-Flags', '')
        if '\\Seen' in flags:
            return EmailStatus.READ
        elif '\\Answered' in flags:
            return EmailStatus.REPLIED
        else:
            return EmailStatus.NEW
    
    def _extract_labels(self, email_message) -> List[str]:
        """Извлечение меток сообщения."""
        labels = []
        
        # Gmail метки
        x_gmail_labels = email_message.get('X-Gmail-Labels', '')
        if x_gmail_labels:
            labels.extend(x_gmail_labels.split(','))
        
        # Яндекс метки
        x_yandex_labels = email_message.get('X-Yandex-Labels', '')
        if x_yandex_labels:
            labels.extend(x_yandex_labels.split(','))
        
        return [label.strip() for label in labels if label.strip()]
    
    def _needs_reply(self, subject: str, content: str) -> bool:
        """Определение, нужен ли ответ на сообщение."""
        text = (subject + " " + content).lower()
        keywords = self.email_config.get("reply_needed_keywords", [])
        
        for keyword in keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def generate_summary(self, messages: List[EmailMessage]) -> Dict:
        """Генерация сводки по сообщениям."""
        if not self.openai_client:
            return {"error": "OpenAI не настроен"}
        
        try:
            # Группируем сообщения по категориям
            urgent_messages = [msg for msg in messages if msg.priority == EmailPriority.URGENT]
            high_priority = [msg for msg in messages if msg.priority == EmailPriority.HIGH]
            need_reply = [msg for msg in messages if msg.is_reply_needed]
            
            # Формируем текст для анализа
            summary_text = f"""
Анализ входящих сообщений за последние 7 дней:

Всего сообщений: {len(messages)}
Срочных: {len(urgent_messages)}
Важных: {len(high_priority)}
Требуют ответа: {len(need_reply)}

Детали по сообщениям:
"""
            
            for msg in messages[:10]:  # Первые 10 сообщений
                summary_text += f"""
- От: {msg.sender}
- Тема: {msg.subject}
- Приоритет: {msg.priority.value}
- Нужен ответ: {'Да' if msg.is_reply_needed else 'Нет'}
- Дата: {msg.date.strftime('%d.%m.%Y %H:%M')}
"""
            
            # Запрос к OpenAI для анализа
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты помощник по анализу email переписки. Создай краткую сводку по входящим сообщениям, выделив важное, срочное и требующее ответа."},
                    {"role": "user", "content": summary_text}
                ],
                max_tokens=500
            )
            
            summary = response.choices[0].message.content
            
            return {
                "total_messages": len(messages),
                "urgent_count": len(urgent_messages),
                "high_priority_count": len(high_priority),
                "need_reply_count": len(need_reply),
                "summary": summary,
                "urgent_messages": [{"subject": msg.subject, "sender": msg.sender} for msg in urgent_messages[:5]],
                "need_reply_messages": [{"subject": msg.subject, "sender": msg.sender} for msg in need_reply[:5]]
            }
            
        except Exception as e:
            return {"error": f"Ошибка генерации сводки: {e}"}
    
    def generate_reply_template(self, message: EmailMessage) -> str:
        """Генерация шаблона ответа."""
        if not self.openai_client:
            return "OpenAI не настроен для генерации шаблонов"
        
        try:
            prompt = f"""
Создай профессиональный шаблон ответа на следующее сообщение:

От: {message.sender}
Тема: {message.subject}
Содержание: {message.content[:500]}...

Шаблон должен быть:
1. Вежливым и профессиональным
2. Кратким и по делу
3. Содержать приветствие и подпись
4. Учитывать контекст сообщения
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты помощник по составлению профессиональных email ответов."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Ошибка генерации шаблона: {e}"
    
    def get_inbox_summary(self, provider: str = "gmail") -> Dict:
        """Получение сводки входящих."""
        if not self.connect_imap(provider):
            return {"error": f"Не удалось подключиться к {provider}"}
        
        try:
            messages = self.get_inbox_messages(limit=50, days_back=7)
            summary = self.generate_summary(messages)
            return summary
        finally:
            self.disconnect_imap()
    
    def get_messages_by_lens(self, lens: str, provider: str = "gmail") -> List[EmailMessage]:
        """Получение сообщений по линзам (категориям)."""
        if not self.connect_imap(provider):
            return []
        
        try:
            messages = self.get_inbox_messages(limit=100, days_back=30)
            
            if lens == "urgent":
                return [msg for msg in messages if msg.priority in [EmailPriority.URGENT, EmailPriority.HIGH]]
            elif lens == "need_reply":
                return [msg for msg in messages if msg.is_reply_needed]
            elif lens == "important":
                return [msg for msg in messages if msg.priority in [EmailPriority.URGENT, EmailPriority.HIGH] or msg.is_reply_needed]
            else:
                return messages
                
        finally:
            self.disconnect_imap()

# Глобальный экземпляр
email_analyzer = EmailAnalyzer() 