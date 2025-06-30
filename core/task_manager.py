"""
Центральный менеджер задач (TaskManager).

Единый модуль для создания, хранения, обновления и получения задач
из всех интегрированных систем (Planner, Inbox, Meetings, Finances и т.д.).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

class TaskManager:
    def __init__(self, data_file="tasks.json"):
        self.data_file = data_file
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[Dict]:
        """Загружает задачи из JSON-файла."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_tasks(self):
        """Сохраняет все задачи в JSON-файл."""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, ensure_ascii=False, indent=4)

    def _generate_id(self, title: str) -> str:
        """Генерирует уникальный ID для задачи."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Используем хэш для предотвращения длинных ID и обеспечения уникальности
        title_hash = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"task_{timestamp}_{title_hash}"

    def add_task(
        self,
        title: str,
        source: str,
        description: Optional[str] = None,
        due_date: Optional[str] = None,
        priority: int = 3,
        tags: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        assigned_to: Optional[str] = None
    ) -> Dict:
        """
        Добавляет новую задачу в систему.

        Предотвращает дублирование по source и source_id.
        """
        # Проверка на дубликат, если есть ID из источника
        if source_id:
            for task in self.tasks:
                if task.get('source') == source and task.get('source_id') == source_id:
                    # Задача уже существует, возвращаем ее
                    return task
        
        now = datetime.now().isoformat()
        
        new_task = {
            'id': self._generate_id(title),
            'title': title,
            'description': description,
            'due_date': due_date,
            'status': 'pending',  # 'pending', 'in_progress', 'completed', 'cancelled'
            'priority': priority, # 1 (высший) - 5 (низший)
            'progress': 0,
            'assigned_to': assigned_to,
            'source': source,
            'source_id': source_id,
            'tags': tags or [],
            'created_at': now,
            'updated_at': now,
        }
        
        self.tasks.append(new_task)
        self._save_tasks()
        return new_task

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Находит задачу по ее ID."""
        for task in self.tasks:
            if task['id'] == task_id:
                return task
        return None

    def update_task(self, task_id: str, updates: Dict) -> Optional[Dict]:
        """Обновляет задачу по ее ID."""
        for task in self.tasks:
            if task['id'] == task_id:
                # Не позволяем изменять системные поля напрямую
                updates.pop('id', None)
                updates.pop('created_at', None)
                updates.pop('source', None)
                
                task.update(updates)
                task['updated_at'] = datetime.now().isoformat()
                self._save_tasks()
                return task
        return None
    
    def get_all_tasks(self, status_filter: Optional[List[str]] = None) -> List[Dict]:
        """Возвращает все задачи, опционально фильтруя по статусу."""
        if status_filter:
            return [task for task in self.tasks if task['status'] in status_filter]
        return self.tasks

    def delete_task(self, task_id: str) -> bool:
        """Удаляет задачу по ID."""
        task_found = False
        initial_len = len(self.tasks)
        self.tasks = [task for task in self.tasks if task['id'] != task_id]
        
        if len(self.tasks) < initial_len:
            task_found = True
            self._save_tasks()
            
        return task_found

# Глобальный экземпляр для использования в других модулях
task_manager = TaskManager() 