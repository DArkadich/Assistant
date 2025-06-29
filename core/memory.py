import os
import json
from datetime import datetime
from typing import List, Dict, Optional

HISTORY_FILE = "memory/history.json"

class ChatMemory:
    def __init__(self, file_path: str = HISTORY_FILE):
        self.file_path = file_path
        self.history: List[Dict] = []
        self._load()
    
    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки истории: {e}")
                self.history = []
        else:
            self.history = []
    
    def _save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")
    
    def add_message(self, user_id: int, username: str, text: str, role: str = "user", timestamp: Optional[str] = None):
        """Добавить сообщение в историю."""
        if not timestamp:
            timestamp = datetime.now().isoformat()
        self.history.append({
            "user_id": user_id,
            "username": username,
            "text": text,
            "role": role,
            "timestamp": timestamp
        })
        self._save()
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск по истории сообщений по ключевым словам."""
        query_lower = query.lower()
        results = [msg for msg in self.history if query_lower in msg["text"].lower()]
        return results[-limit:]
    
    def get_discussions_with(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Найти обсуждения с упоминанием ключевого слова и участников."""
        keyword_lower = keyword.lower()
        results = [msg for msg in self.history if keyword_lower in msg["text"].lower()]
        # Собираем уникальных участников
        participants = set(msg["username"] for msg in results)
        return [{"username": u, "messages": [msg for msg in results if msg["username"] == u]} for u in participants]

# Глобальный экземпляр памяти
chat_memory = ChatMemory() 