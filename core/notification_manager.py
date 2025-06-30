"""
Модуль для управления уведомлениями с учетом режима работы
"""
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
from core.work_mode import work_mode_manager

class NotificationManager:
    def __init__(self):
        self.notification_queue = []
        self.bot = None
    
    def set_bot(self, bot):
        """Установка экземпляра бота для отправки сообщений"""
        self.bot = bot
    
    async def send_notification(
        self,
        chat_id: int,
        text: str,
        priority: str = 'normal',
        project: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Отправка уведомления с учетом режима работы
        
        priority: critical, high, normal, low
        project: название проекта (опционально)
        """
        if not self.bot:
            print("[ERROR] Bot instance not set")
            return False
            
        # Проверяем режим работы
        if not work_mode_manager.should_notify(priority, project):
            # Добавляем в очередь для отправки позже
            self.notification_queue.append({
                'chat_id': chat_id,
                'text': text,
                'priority': priority,
                'project': project,
                'timestamp': datetime.now().isoformat(),
                'kwargs': kwargs
            })
            print(f"[DEBUG] Notification queued due to work mode: {text[:100]}...")
            return False
        
        try:
            # Добавляем метку о важности для критичных уведомлений
            if priority == 'critical':
                text = "❗️ КРИТИЧНО ❗️\n\n" + text
            elif priority == 'high':
                text = "⚠️ Важно!\n\n" + text
            
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                **kwargs
            )
            return True
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False
    
    async def send_queued_notifications(self, chat_id: int):
        """Отправка отложенных уведомлений"""
        if not self.notification_queue:
            return
        
        summary = "📬 Пропущенные уведомления:\n\n"
        by_project = {}
        
        for notif in self.notification_queue:
            if notif['chat_id'] != chat_id:
                continue
                
            project = notif['project'] or 'Без проекта'
            if project not in by_project:
                by_project[project] = []
            
            timestamp = datetime.fromisoformat(notif['timestamp'])
            time_str = timestamp.strftime("%H:%M")
            
            by_project[project].append(
                f"[{time_str}] {notif['text'][:100]}..."
            )
        
        for project, messages in by_project.items():
            summary += f"\n🔹 <b>{project}</b>\n"
            summary += "\n".join(messages) + "\n"
        
        if by_project:
            await self.bot.send_message(
                chat_id=chat_id,
                text=summary,
                parse_mode='HTML'
            )
            
            # Очищаем очередь для этого чата
            self.notification_queue = [
                n for n in self.notification_queue
                if n['chat_id'] != chat_id
            ]

# Глобальный экземпляр для использования в других модулях
notification_manager = NotificationManager() 