"""
Модуль контроля входящих сообщений
Мониторинг почты, Telegram, CRM, Google Docs
"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path
import re
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import openai
from core.task_manager import task_manager

class InboxMonitor:
    def __init__(self):
        self.data_file = Path("inbox_monitor.json")
        self.monitor_data = self._load_data()
        self.attention_keywords = [
            'срочно', 'urgent', 'важно', 'important', 'критично', 'critical',
            'дедлайн', 'deadline', 'срочный', 'urgent', 'немедленно', 'immediately',
            'проблема', 'problem', 'ошибка', 'error', 'сбой', 'failure',
            'контракт', 'contract', 'сделка', 'deal', 'оплата', 'payment',
            'встреча', 'meeting', 'звонок', 'call', 'презентация', 'presentation'
        ]
        self.oauth_settings_file = Path("email_oauth_settings.json")
        self.last_uid_file = Path("email_last_uid.txt")
        
    def _load_data(self) -> dict:
        """Загрузка данных мониторинга"""
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'messages': [],
            'reminders': [],
            'settings': {
                'response_timeout_days': 3,
                'attention_score_threshold': 7,
                'auto_remind': True,
                'channels': ['email', 'telegram', 'crm', 'google_docs']
            },
            'last_check': None
        }
    
    def _save_data(self):
        """Сохранение данных мониторинга"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.monitor_data, f, ensure_ascii=False, indent=2)
    
    def add_message(self, 
                   channel: str,
                   sender: str,
                   subject: str,
                   content: str,
                   timestamp: str,
                   message_id: str = None,
                   priority: str = 'normal',
                   requires_response: bool = True) -> str:
        """Добавление нового сообщения для мониторинга"""
        
        # Анализируем важность сообщения
        attention_score = self._calculate_attention_score(subject, content)
        needs_attention = attention_score >= self.monitor_data['settings']['attention_score_threshold']
        
        message = {
            'id': message_id or f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'channel': channel,
            'sender': sender,
            'subject': subject,
            'content': content[:500],  # Ограничиваем длину
            'timestamp': timestamp,
            'priority': priority,
            'requires_response': requires_response,
            'attention_score': attention_score,
            'needs_attention': needs_attention,
            'status': 'new',
            'response_sent': False,
            'response_date': None,
            'reminders_sent': 0,
            'last_reminder': None,
            'tags': self._extract_tags(subject, content)
        }
        
        self.monitor_data['messages'].append(message)
        self._save_data()
        
        # Создаём задачу через TaskManager, если требуется ответ
        if requires_response:
            self._create_task_from_message(message)
        
        return message['id']
    
    def _calculate_attention_score(self, subject: str, content: str) -> int:
        """Расчёт оценки важности сообщения"""
        score = 0
        text = f"{subject} {content}".lower()
        
        # Ключевые слова
        for keyword in self.attention_keywords:
            if keyword in text:
                score += 2
        
        # Срочность (срочно, urgent, немедленно)
        urgency_words = ['срочно', 'urgent', 'немедленно', 'immediately', 'asap']
        for word in urgency_words:
            if word in text:
                score += 5
        
        # Вопросы (требуют ответа)
        if '?' in text:
            score += 3
        
        # Упоминания денег/сумм
        if re.search(r'\d+[\s]*[₽$€]|\d+[\s]*(рубл|доллар|евро)', text):
            score += 4
        
        # Упоминания дат/сроков
        if re.search(r'\d{1,2}[./]\d{1,2}|\d{1,2}[-]\d{1,2}|\d{4}', text):
            score += 2
        
        # Личные обращения
        if re.search(r'ты|вы|вас|тебя|вам', text):
            score += 1
        
        return min(score, 10)  # Максимум 10 баллов
    
    def _extract_tags(self, subject: str, content: str) -> List[str]:
        """Извлечение тегов из сообщения"""
        tags = []
        text = f"{subject} {content}".lower()
        
        # Проекты
        project_patterns = ['проект', 'project', 'кп', 'cp', 'предложение', 'proposal']
        for pattern in project_patterns:
            if pattern in text:
                tags.append('project')
                break
        
        # Финансы
        finance_patterns = ['оплата', 'payment', 'счёт', 'invoice', 'контракт', 'contract']
        for pattern in finance_patterns:
            if pattern in text:
                tags.append('finance')
                break
        
        # Встречи
        meeting_patterns = ['встреча', 'meeting', 'звонок', 'call', 'конференция', 'conference']
        for pattern in meeting_patterns:
            if pattern in text:
                tags.append('meeting')
                break
        
        # Проблемы
        problem_patterns = ['проблема', 'problem', 'ошибка', 'error', 'сбой', 'failure']
        for pattern in problem_patterns:
            if pattern in text:
                tags.append('problem')
                break
        
        return tags
    
    def _create_task_from_message(self, message: Dict):
        """Создание задачи на основе сообщения через TaskManager."""
        due_date = self._calculate_due_date(message['timestamp'])
        
        task_manager.add_task(
            title=f"Ответить {message['sender']}: {message['subject']}",
            description=message['content'][:200],
            due_date=due_date,
            priority=3,  # Можно сделать маппинг из message['priority']
            tags=message.get('tags', []),
            source='inbox',
            source_id=message['id']
        )
    
    def _calculate_due_date(self, timestamp: str) -> str:
        """Расчёт дедлайна для ответа"""
        created_date = datetime.fromisoformat(timestamp)
        due_date = created_date + timedelta(days=self.monitor_data['settings']['response_timeout_days'])
        return due_date.isoformat()
    
    def mark_as_responded(self, message_id: str, response_date: str = None):
        """Отметка сообщения как отвеченного и обновление связанной задачи."""
        for message in self.monitor_data['messages']:
            if message['id'] == message_id:
                message['status'] = 'responded'
                message['response_sent'] = True
                message['response_date'] = response_date or datetime.now().isoformat()
                break
        
        # Обновляем связанную задачу в TaskManager
        tasks = task_manager.get_all_tasks()
        for task in tasks:
            if task.get('source_id') == message_id and task.get('source') == 'inbox':
                task_manager.update_task(task['id'], {'status': 'completed'})
                break
        
        self._save_data()
    
    def mark_as_ignored(self, message_id: str):
        """Отметка сообщения как проигнорированного"""
        for message in self.monitor_data['messages']:
            if message['id'] == message_id:
                message['status'] = 'ignored'
                break
        
        # Обновляем связанную задачу
        for task in self.monitor_data['tasks']:
            if task['message_id'] == message_id:
                task['status'] = 'ignored'
                break
        
        self._save_data()
    
    def get_messages_requiring_attention(self) -> List[Dict]:
        """Получение сообщений, требующих внимания"""
        now = datetime.now()
        attention_messages = []
        
        for message in self.monitor_data['messages']:
            if message['status'] != 'new':
                continue
            
            # Проверяем, требует ли внимания
            if message['needs_attention']:
                attention_messages.append(message)
                continue
            
            # Проверяем, не просрочен ли ответ
            if message['requires_response']:
                created_date = datetime.fromisoformat(message['timestamp'])
                days_since = (now - created_date).days
                
                if days_since >= self.monitor_data['settings']['response_timeout_days']:
                    attention_messages.append(message)
        
        return sorted(attention_messages, key=lambda x: x['attention_score'], reverse=True)
    
    def get_overdue_responses(self) -> List[Dict]:
        """Получение просроченных ответов (3+ дня)"""
        now = datetime.now()
        overdue_messages = []
        
        for message in self.monitor_data['messages']:
            if not message['requires_response'] or message['response_sent']:
                continue
            
            created_date = datetime.fromisoformat(message['timestamp'])
            days_since = (now - created_date).days
            
            if days_since >= self.monitor_data['settings']['response_timeout_days']:
                overdue_messages.append({
                    **message,
                    'days_overdue': days_since
                })
        
        return sorted(overdue_messages, key=lambda x: x['days_overdue'], reverse=True)
    
    def get_forgotten_messages(self) -> List[Dict]:
        """Получение забытых сообщений (7+ дней)"""
        now = datetime.now()
        forgotten_messages = []
        
        for message in self.monitor_data['messages']:
            if message['status'] != 'new':
                continue
            
            created_date = datetime.fromisoformat(message['timestamp'])
            days_since = (now - created_date).days
            
            if days_since >= 7:
                forgotten_messages.append({
                    **message,
                    'days_forgotten': days_since
                })
        
        return sorted(forgotten_messages, key=lambda x: x['days_forgotten'], reverse=True)
    
    def get_inbox_summary(self) -> Dict:
        """Получение сводки по входящим"""
        now = datetime.now()
        
        total_messages = len(self.monitor_data['messages'])
        new_messages = len([m for m in self.monitor_data['messages'] if m['status'] == 'new'])
        responded_messages = len([m for m in self.monitor_data['messages'] if m['response_sent']])
        ignored_messages = len([m for m in self.monitor_data['messages'] if m['status'] == 'ignored'])
        
        attention_messages = self.get_messages_requiring_attention()
        overdue_responses = self.get_overdue_responses()
        forgotten_messages = self.get_forgotten_messages()
        
        # Статистика по каналам
        channel_stats = {}
        for message in self.monitor_data['messages']:
            channel = message['channel']
            if channel not in channel_stats:
                channel_stats[channel] = {'total': 0, 'new': 0, 'responded': 0}
            channel_stats[channel]['total'] += 1
            if message['status'] == 'new':
                channel_stats[channel]['new'] += 1
            elif message['response_sent']:
                channel_stats[channel]['responded'] += 1
        
        return {
            'total_messages': total_messages,
            'new_messages': new_messages,
            'responded_messages': responded_messages,
            'ignored_messages': ignored_messages,
            'attention_messages_count': len(attention_messages),
            'overdue_responses_count': len(overdue_responses),
            'forgotten_messages_count': len(forgotten_messages),
            'channel_stats': channel_stats,
            'response_rate': (responded_messages / total_messages * 100) if total_messages > 0 else 0,
            'last_check': self.monitor_data.get('last_check')
        }
    
    def format_telegram_summary(self, summary: Dict) -> str:
        """Форматирование сводки для Telegram"""
        text = "📬 <b>Сводка по входящим сообщениям</b>\n\n"
        
        # Основная статистика
        text += f"📊 <b>Общая статистика:</b>\n"
        text += f"• Всего сообщений: {summary['total_messages']}\n"
        text += f"• Новых: {summary['new_messages']}\n"
        text += f"• Отвечено: {summary['responded_messages']}\n"
        text += f"• Процент ответов: {summary['response_rate']:.1f}%\n\n"
        
        # Проблемные области
        if summary['attention_messages_count'] > 0:
            text += f"⚠️ <b>Требуют внимания:</b> {summary['attention_messages_count']}\n"
        
        if summary['overdue_responses_count'] > 0:
            text += f"🚨 <b>Просроченные ответы:</b> {summary['overdue_responses_count']}\n"
        
        if summary['forgotten_messages_count'] > 0:
            text += f"😴 <b>Забытые сообщения:</b> {summary['forgotten_messages_count']}\n"
        
        if summary['attention_messages_count'] == 0 and summary['overdue_responses_count'] == 0:
            text += f"✅ Все в порядке!\n"
        
        text += "\n"
        
        # Статистика по каналам
        text += f"📱 <b>По каналам:</b>\n"
        for channel, stats in summary['channel_stats'].items():
            channel_emoji = {
                'email': '📧',
                'telegram': '💬',
                'crm': '📋',
                'google_docs': '📄'
            }.get(channel, '📱')
            
            text += f"{channel_emoji} {channel.title()}: {stats['new']}/{stats['total']} новых\n"
        
        return text
    
    def format_attention_report(self, messages: List[Dict]) -> str:
        """Форматирование отчёта по сообщениям, требующим внимания"""
        if not messages:
            return "✅ Нет сообщений, требующих внимания!"
        
        text = f"⚠️ <b>Сообщения, требующие внимания ({len(messages)}):</b>\n\n"
        
        for i, message in enumerate(messages[:10], 1):  # Показываем первые 10
            days_ago = (datetime.now() - datetime.fromisoformat(message['timestamp'])).days
            
            # Эмодзи для каналов
            channel_emoji = {
                'email': '📧',
                'telegram': '💬',
                'crm': '📋',
                'google_docs': '📄'
            }.get(message['channel'], '📱')
            
            # Эмодзи для приоритета
            priority_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'normal': '🟢'
            }.get(message['priority'], '🟢')
            
            text += f"{i}. {priority_emoji} {channel_emoji} <b>{message['sender']}</b>\n"
            text += f"   📝 {message['subject']}\n"
            text += f"   📅 {days_ago} дн. назад | 🎯 {message['attention_score']}/10\n"
            
            if message['tags']:
                text += f"   🏷️ {', '.join(message['tags'])}\n"
            
            text += "\n"
        
        if len(messages) > 10:
            text += f"... и ещё {len(messages) - 10} сообщений\n"
        
        return text
    
    def format_overdue_report(self, messages: List[Dict]) -> str:
        """Форматирование отчёта по просроченным ответам"""
        if not messages:
            return "✅ Нет просроченных ответов!"
        
        text = f"🚨 <b>Просроченные ответы ({len(messages)}):</b>\n\n"
        
        for i, message in enumerate(messages[:10], 1):  # Показываем первые 10
            channel_emoji = {
                'email': '📧',
                'telegram': '💬',
                'crm': '📋',
                'google_docs': '📄'
            }.get(message['channel'], '📱')
            
            text += f"{i}. {channel_emoji} <b>{message['sender']}</b>\n"
            text += f"   📝 {message['subject']}\n"
            text += f"   ⏰ {message['days_overdue']} дн. просрочено\n"
            text += f"   📅 {message['timestamp'][:10]}\n\n"
        
        if len(messages) > 10:
            text += f"... и ещё {len(messages) - 10} сообщений\n"
        
        return text
    
    def format_forgotten_report(self, messages: List[Dict]) -> str:
        """Форматирование отчёта по забытым сообщениям"""
        if not messages:
            return "✅ Нет забытых сообщений!"
        
        text = f"😴 <b>Забытые сообщения ({len(messages)}):</b>\n\n"
        
        for i, message in enumerate(messages[:10], 1):  # Показываем первые 10
            channel_emoji = {
                'email': '📧',
                'telegram': '💬',
                'crm': '📋',
                'google_docs': '📄'
            }.get(message['channel'], '📱')
            
            text += f"{i}. {channel_emoji} <b>{message['sender']}</b>\n"
            text += f"   📝 {message['subject']}\n"
            text += f"   😴 {message['days_forgotten']} дн. забыто\n"
            text += f"   📅 {message['timestamp'][:10]}\n\n"
        
        if len(messages) > 10:
            text += f"... и ещё {len(messages) - 10} сообщений\n"
        
        return text
    
    def get_reminder_suggestions(self) -> List[str]:
        """Получение предложений для напоминаний"""
        suggestions = []
        overdue_messages = self.get_overdue_responses()
        
        for message in overdue_messages[:5]:  # Топ-5 просроченных
            suggestion = f"📲 «Ты не ответил {message['sender']} по {message['subject']} — напомнить?»"
            suggestions.append(suggestion)
        
        return suggestions
    
    def should_send_reminder(self) -> bool:
        """Проверка, нужно ли отправлять напоминание"""
        if not self.monitor_data['settings']['auto_remind']:
            return False
        
        overdue_count = len(self.get_overdue_responses())
        attention_count = len(self.get_messages_requiring_attention())
        
        return overdue_count > 0 or attention_count > 3
    
    def mark_reminder_sent(self, message_id: str):
        """Отметка о том, что напоминание отправлено"""
        for message in self.monitor_data['messages']:
            if message['id'] == message_id:
                message['reminders_sent'] += 1
                message['last_reminder'] = datetime.now().isoformat()
                break
        
        self._save_data()
    
    def update_last_check(self):
        """Обновление времени последней проверки"""
        self.monitor_data['last_check'] = datetime.now().isoformat()
        self._save_data()
    
    def fetch_emails_via_oauth(self, provider: str = 'gmail', max_count: int = 20) -> int:
        """
        Загружает новые письма из почтового ящика через IMAP с OAuth2
        provider: gmail/yandex/outlook
        max_count: максимальное количество писем за раз
        Возвращает количество новых добавленных писем
        """
        # Загрузка OAuth токена и настроек
        if not self.oauth_settings_file.exists():
            print("OAuth settings file not found.")
            return 0
        with open(self.oauth_settings_file, 'r', encoding='utf-8') as f:
            oauth_settings = json.load(f)
        
        if provider not in oauth_settings:
            print(f"No OAuth settings for provider {provider}")
            return 0
        creds = oauth_settings[provider]
        
        # Подключение к IMAP
        if provider == 'gmail':
            imap_host = 'imap.gmail.com'
        elif provider == 'yandex':
            imap_host = 'imap.yandex.com'
        elif provider == 'outlook':
            imap_host = 'imap-mail.outlook.com'
        else:
            print(f"Unknown provider: {provider}")
            return 0
        
        # Используем XOAUTH2
        import imaplib
        import base64
        imap = imaplib.IMAP4_SSL(imap_host)
        auth_string = f"user={creds['email']}\1auth=Bearer {creds['access_token']}\1\1"
        imap.authenticate('XOAUTH2', lambda x: base64.b64encode(auth_string.encode()))
        imap.select('INBOX')
        
        # Получаем UID последнего обработанного письма
        last_uid = None
        if self.last_uid_file.exists():
            last_uid = self.last_uid_file.read_text().strip()
        
        # Ищем новые письма
        search_criteria = '(UNSEEN)'
        if last_uid:
            search_criteria = f'(UID {int(last_uid)+1}:*)'
        status, data = imap.uid('search', None, search_criteria)
        if status != 'OK':
            print("IMAP search failed")
            return 0
        uids = data[0].split()
        if not uids:
            return 0
        new_count = 0
        for uid in uids[-max_count:]:
            status, msg_data = imap.uid('fetch', uid, '(RFC822)')
            if status != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg.get('Subject'))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8', errors='ignore')
            sender = msg.get('From')
            date = msg.get('Date')
            timestamp = parsedate_to_datetime(date).isoformat() if date else datetime.now().isoformat()
            # Извлекаем текст письма
            content = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == 'text/plain' and part.get_content_disposition() is None:
                        charset = part.get_content_charset() or 'utf-8'
                        content = part.get_payload(decode=True).decode(charset, errors='ignore')
                        break
            else:
                charset = msg.get_content_charset() or 'utf-8'
                content = msg.get_payload(decode=True).decode(charset, errors='ignore')
            # Добавляем письмо в мониторинг
            self.add_message(
                channel='email',
                sender=sender,
                subject=subject,
                content=content,
                timestamp=timestamp,
                message_id=f"email_{uid.decode()}"
            )
            new_count += 1
            # Обновляем последний UID
            self.last_uid_file.write_text(uid.decode())
        imap.logout()
        return new_count
    
    def suggest_reply(self, message_id: str, system_prompt: str = None) -> str:
        """
        Генерирует шаблон ответа на письмо с помощью OpenAI
        message_id: ID письма
        system_prompt: дополнительная инструкция для стиля ответа
        Возвращает текст ответа
        """
        message = next((m for m in self.monitor_data['messages'] if m['id'] == message_id), None)
        if not message:
            return "Письмо не найдено."
        prompt = (
            f"Ты — вежливый бизнес-ассистент. Составь краткий и корректный ответ на это письмо на русском языке. "
            f"Если есть вопросы — задай их.\n\n"
            f"Тема: {message['subject']}\n"
            f"Текст письма: {message['content']}\n"
        )
        if system_prompt:
            prompt = system_prompt + "\n" + prompt
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=300,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Ошибка генерации ответа: {e}"

# Глобальный экземпляр
inbox_monitor = InboxMonitor() 