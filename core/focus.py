"""
Модуль для управления ежедневным фокусом пользователя
"""
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional

class FocusManager:
    def __init__(self):
        self.focus_data = {
            'daily_focus': {},  # date -> focus_text
            'focus_tasks': {},  # date -> list of tasks
            'settings': {
                'reminder_interval_minutes': 30,
                'last_reminder_time': None
            }
        }
        self.FOCUS_FILE = 'focus_data.json'
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных из файла"""
        if os.path.exists(self.FOCUS_FILE):
            with open(self.FOCUS_FILE, 'r', encoding='utf-8') as f:
                self.focus_data = json.load(f)
    
    def _save_data(self):
        """Сохранение данных в файл"""
        with open(self.FOCUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.focus_data, f, ensure_ascii=False, indent=2)
    
    def set_daily_focus(self, focus_text: str, date: Optional[str] = None) -> bool:
        """Установка фокуса на день"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self.focus_data['daily_focus'][date] = focus_text
        self._save_data()
        return True
    
    def get_daily_focus(self, date: Optional[str] = None) -> Optional[str]:
        """Получение фокуса на день"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.focus_data['daily_focus'].get(date)
    
    def is_task_in_focus(self, task_text: str, date: Optional[str] = None) -> bool:
        """Проверка, соответствует ли задача дневному фокусу"""
        focus = self.get_daily_focus(date)
        if not focus:
            return True  # Если фокус не установлен, все задачи в фокусе
        
        # Простая проверка на вхождение ключевых слов из фокуса в текст задачи
        focus_keywords = focus.lower().split()
        task_text_lower = task_text.lower()
        
        return any(keyword in task_text_lower for keyword in focus_keywords)
    
    def add_focus_task(self, task_text: str, date: Optional[str] = None) -> bool:
        """Добавление задачи в список фокусных задач"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if date not in self.focus_data['focus_tasks']:
            self.focus_data['focus_tasks'][date] = []
        
        self.focus_data['focus_tasks'][date].append({
            'task_text': task_text,
            'created_at': datetime.now().isoformat(),
            'completed': False
        })
        self._save_data()
        return True
    
    def get_focus_tasks(self, date: Optional[str] = None) -> List[Dict]:
        """Получение списка фокусных задач на день"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.focus_data['focus_tasks'].get(date, [])
    
    def mark_task_completed(self, task_text: str, date: Optional[str] = None) -> bool:
        """Отметка задачи как выполненной"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        tasks = self.focus_data['focus_tasks'].get(date, [])
        for task in tasks:
            if task['task_text'] == task_text:
                task['completed'] = True
                self._save_data()
                return True
        return False
    
    def should_remind_about_focus(self) -> bool:
        """Проверка необходимости напоминания о фокусе"""
        last_reminder = self.focus_data['settings']['last_reminder_time']
        if not last_reminder:
            return True
        
        last_reminder_time = datetime.fromisoformat(last_reminder)
        interval = timedelta(minutes=self.focus_data['settings']['reminder_interval_minutes'])
        
        return datetime.now() - last_reminder_time >= interval
    
    def update_last_reminder_time(self):
        """Обновление времени последнего напоминания"""
        self.focus_data['settings']['last_reminder_time'] = datetime.now().isoformat()
        self._save_data()

# Глобальный экземпляр для использования в других модулях
focus_manager = FocusManager() 