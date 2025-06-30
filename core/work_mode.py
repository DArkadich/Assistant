"""
Модуль для управления режимами работы (на объекте, в офисе и т.д.)
"""
from datetime import datetime
import json
import os
from typing import Dict, List, Optional

class WorkModeManager:
    def __init__(self):
        self.data = {
            'active_mode': None,  # текущий режим
            'project_filter': None,  # фильтр по проекту
            'start_time': None,  # время начала режима
            'settings': {
                'on_site': {
                    'notification_level': 'critical_only',
                    'auto_reply': 'Я сейчас на объекте. По срочным вопросам: +7...',
                    'project_specific': True,
                }
            }
        }
        self.MODES_FILE = 'work_modes.json'
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных из файла"""
        if os.path.exists(self.MODES_FILE):
            with open(self.MODES_FILE, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
    
    def _save_data(self):
        """Сохранение данных в файл"""
        with open(self.MODES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def activate_on_site_mode(self, project: Optional[str] = None) -> Dict:
        """Активация режима 'На объекте'"""
        self.data['active_mode'] = 'on_site'
        self.data['project_filter'] = project
        self.data['start_time'] = datetime.now().isoformat()
        self._save_data()
        return {
            'mode': 'on_site',
            'project': project,
            'start_time': self.data['start_time']
        }
    
    def deactivate_mode(self) -> bool:
        """Отключение активного режима"""
        if self.data['active_mode']:
            self.data['active_mode'] = None
            self.data['project_filter'] = None
            self.data['start_time'] = None
            self._save_data()
            return True
        return False
    
    def get_active_mode(self) -> Optional[Dict]:
        """Получение информации об активном режиме"""
        if not self.data['active_mode']:
            return None
        return {
            'mode': self.data['active_mode'],
            'project': self.data['project_filter'],
            'start_time': self.data['start_time']
        }
    
    def should_notify(self, priority: str, project: Optional[str] = None) -> bool:
        """Проверка необходимости уведомления в текущем режиме"""
        if not self.data['active_mode']:
            return True
            
        if self.data['active_mode'] == 'on_site':
            # В режиме "на объекте" пропускаем только критичные
            if priority != 'critical':
                return False
            
            # Если задан проект, проверяем соответствие
            if self.data['project_filter'] and project:
                return project == self.data['project_filter']
            
            return True
        
        return True
    
    def update_settings(self, mode: str, settings: Dict) -> bool:
        """Обновление настроек режима"""
        if mode in self.data['settings']:
            self.data['settings'][mode].update(settings)
            self._save_data()
            return True
        return False
    
    def get_settings(self, mode: str) -> Optional[Dict]:
        """Получение настроек режима"""
        return self.data['settings'].get(mode)

# Глобальный экземпляр для использования в других модулях
work_mode_manager = WorkModeManager() 