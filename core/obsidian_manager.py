"""
Obsidian Manager — интеграция с Obsidian для хранения стратегий, решений и логов
Альтернатива Notion, которая работает локально и поддерживает Markdown
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import uuid
from pathlib import Path

class ObsidianManager:
    def __init__(self, vault_path: str = None):
        self.vault_path = vault_path or "./obsidian_vault"
        self.vault = Path(self.vault_path)
        self.vault_path = self.vault  # Store the Path object
        self.config_file = self.vault / "obsidian_config.json"
        
        self._create_vault_structure()
        self._load_config()
        print(f"✅ Obsidian Manager инициализирован: {self.vault_path}")
    
    def _create_vault_structure(self):
        folders = ["01-Стратегии", "02-Решения", "03-Логи", "04-Проекты", "05-Встречи", "06-Задачи", "07-Финансы", "08-Партнеры", "09-Клиенты", "10-Документы", "attachments"]
        for folder in folders:
            folder_path = self.vault / folder
            folder_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {'created_at': datetime.now().isoformat(), 'last_sync': None, 'note_count': 0, 'tags': [], 'projects': []}
            self._save_config()
    
    def _save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def create_strategy_note(self, title: str, content: str, tags: List[str] = None, category: str = "Общая") -> str:
        note_id = f"strategy_{uuid.uuid4().hex[:8]}"
        filename = f"{note_id}_{self._sanitize_filename(title)}.md"
        filepath = self.vault / "01-Стратегии" / filename
        
        note_content = f"""# {title}

**Категория:** {category}
**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**ID:** {note_id}

## Описание

{content}

## Теги
{', '.join(tags or [])}

## Связанные заметки

## История изменений
- {datetime.now().strftime('%Y-%m-%d %H:%M')} - Создана заметка
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        self.config['note_count'] += 1
        self._save_config()
        print(f"✅ Создана стратегия: {title} ({note_id})")
        return note_id
    
    def create_decision_note(self, title: str, problem: str, solution: str, reasoning: str = "", participants: List[str] = None, project: str = None) -> str:
        note_id = f"decision_{uuid.uuid4().hex[:8]}"
        filename = f"{note_id}_{self._sanitize_filename(title)}.md"
        filepath = self.vault / "02-Решения" / filename
        
        note_content = f"""# {title}

**Дата принятия:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**ID:** {note_id}
**Проект:** {project or 'Не указан'}

## Проблема

{problem}

## Решение

{solution}

## Обоснование

{reasoning}

## Участники обсуждения
{', '.join(participants or [])}

## Связанные заметки

## История изменений
- {datetime.now().strftime('%Y-%m-%d %H:%M')} - Принято решение
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        self.config['note_count'] += 1
        self._save_config()
        print(f"✅ Создано решение: {title} ({note_id})")
        return note_id
    
    def log_task_completion(self, task_text: str, completion_date: str = None, notes: str = "", project: str = None) -> str:
        note_id = f"task_log_{uuid.uuid4().hex[:8]}"
        completion_date = completion_date or datetime.now().strftime('%Y-%m-%d')
        filename = f"{completion_date}_{note_id}.md"
        filepath = self.vault / "03-Логи" / filename
        
        note_content = f"""# Лог выполнения задачи

**Дата:** {completion_date}
**ID:** {note_id}
**Проект:** {project or 'Не указан'}

## Задача

{task_text}

## Заметки о выполнении

{notes}

## Статус
✅ Выполнено

## Связанные заметки
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        self.config['note_count'] += 1
        self._save_config()
        print(f"✅ Записан лог задачи: {task_text[:50]}... ({note_id})")
        return note_id
    
    def log_conversation(self, participants: List[str], topic: str, key_points: List[str], decisions: List[str] = None, project: str = None) -> str:
        note_id = f"conv_log_{uuid.uuid4().hex[:8]}"
        filename = f"{datetime.now().strftime('%Y-%m-%d')}_{note_id}.md"
        filepath = self.vault / "03-Логи" / filename
        
        note_content = f"""# Лог разговора

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**ID:** {note_id}
**Проект:** {project or 'Не указан'}

## Участники

{', '.join(participants)}

## Тема

{topic}

## Ключевые моменты

{chr(10).join([f"- {point}" for point in key_points])}

## Принятые решения

{chr(10).join([f"- {decision}" for decision in decisions or []])}

## Связанные заметки
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(note_content)
        
        self.config['note_count'] += 1
        self._save_config()
        print(f"✅ Записан лог разговора: {topic[:50]}... ({note_id})")
        return note_id
    
    def search_notes(self, query: str, categories: List[str] = None, date_from: str = None, date_to: str = None) -> List[Dict]:
        results = []
        query_lower = query.lower()
        
        search_folders = []
        if categories:
            for category in categories:
                if category == "стратегии": search_folders.append("01-Стратегии")
                elif category == "решения": search_folders.append("02-Решения")
                elif category == "логи": search_folders.append("03-Логи")
                elif category == "проекты": search_folders.append("04-Проекты")
                elif category == "встречи": search_folders.append("05-Встречи")
                elif category == "задачи": search_folders.append("06-Задачи")
                elif category == "финансы": search_folders.append("07-Финансы")
                elif category == "партнеры": search_folders.append("08-Партнеры")
                elif category == "клиенты": search_folders.append("09-Клиенты")
                elif category == "документы": search_folders.append("10-Документы")
        else:
            search_folders = [f.name for f in self.vault.iterdir() if f.is_dir() and not f.name.startswith('.')]
        
        for folder_name in search_folders:
            folder_path = self.vault / folder_name
            if not folder_path.exists():
                continue
            
            for file_path in folder_path.glob("*.md"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if date_from or date_to:
                        file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                        if date_from and file_date < date_from: continue
                        if date_to and file_date > date_to: continue
                    
                    if query_lower in content.lower():
                        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
                        title = title_match.group(1) if title_match else file_path.stem
                        
                        id_match = re.search(r'\*\*ID:\*\* (.+)$', content, re.MULTILINE)
                        note_id = id_match.group(1) if id_match else file_path.stem
                        
                        date_match = re.search(r'\*\*Дата.*:\*\* (.+)$', content, re.MULTILINE)
                        created_date = date_match.group(1) if date_match else file_date
                        
                        results.append({
                            'id': note_id,
                            'title': title,
                            'category': folder_name,
                            'file_path': str(file_path),
                            'created_date': created_date,
                            'content_preview': content[:200] + "..." if len(content) > 200 else content
                        })
                        
                except Exception as e:
                    print(f"❌ Ошибка чтения файла {file_path}: {e}")
        
        results.sort(key=lambda x: x['created_date'], reverse=True)
        return results
    
    def get_statistics(self) -> Dict:
        stats = {'total_notes': 0, 'by_category': {}, 'recent_notes': [], 'projects': self.config['projects']}
        
        for folder_path in self.vault.iterdir():
            if folder_path.is_dir() and not folder_path.name.startswith('.'):
                category = folder_path.name
                note_count = len(list(folder_path.glob("*.md")))
                stats['by_category'][category] = note_count
                stats['total_notes'] += note_count
        
        all_files = []
        for folder_path in self.vault.iterdir():
            if folder_path.is_dir() and not folder_path.name.startswith('.'):
                for file_path in folder_path.glob("*.md"):
                    all_files.append((file_path, file_path.stat().st_mtime))
        
        all_files.sort(key=lambda x: x[1], reverse=True)
        for file_path, mtime in all_files[:10]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
                    title = title_match.group(1) if title_match else file_path.stem
                    
                    stats['recent_notes'].append({
                        'title': title,
                        'category': file_path.parent.name,
                        'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'),
                        'file_path': str(file_path)
                    })
            except Exception as e:
                print(f"❌ Ошибка чтения файла {file_path}: {e}")
        
        return stats
    
    def _sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r'[<>:"/\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized[:50]

# Создаем глобальный экземпляр
obsidian_manager = ObsidianManager()
