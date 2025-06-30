"""
Модуль командной работы: назначение задач, автонапоминания, отчётность
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

class TeamManager:
    def __init__(self, bot_token: str = None):
        self.bot_token = bot_token
        self.data_file = Path("team_data.json")
        self.team_data = self._load_data()
        self.owner_vacation_mode = False
        
    def _load_data(self) -> dict:
        """Загрузка данных команды"""
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'employees': {},
            'tasks': [],
            'reports': [],
            'notifications': []
        }
    
    def _save_data(self):
        """Сохранение данных команды"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.team_data, f, ensure_ascii=False, indent=2)
    
    def add_employee(self, name: str, telegram_id: str, role: str, 
                    chat_id: str = None) -> bool:
        """Добавление сотрудника"""
        self.team_data['employees'][telegram_id] = {
            'name': name,
            'role': role,
            'chat_id': chat_id,
            'active_tasks': [],
            'daily_reports': [],
            'created_at': datetime.now().isoformat()
        }
        self._save_data()
        return True
    
    def assign_task(self, employee_id: str, task: str, deadline: str, 
                   priority: str = 'medium', description: str = '') -> bool:
        """Назначение задачи сотруднику"""
        if employee_id not in self.team_data['employees']:
            return False
            
        task_id = f"task_{len(self.team_data['tasks']) + 1}"
        new_task = {
            'id': task_id,
            'employee_id': employee_id,
            'task': task,
            'deadline': deadline,
            'priority': priority,
            'description': description,
            'status': 'assigned',
            'created_at': datetime.now().isoformat(),
            'completed_at': None
        }
        
        self.team_data['tasks'].append(new_task)
        self.team_data['employees'][employee_id]['active_tasks'].append(task_id)
        self._save_data()
        return True
    
    def get_employee_tasks(self, employee_id: str) -> List[Dict]:
        """Получение задач сотрудника"""
        if employee_id not in self.team_data['employees']:
            return []
        
        employee_tasks = []
        for task in self.team_data['tasks']:
            if task['employee_id'] == employee_id and task['status'] != 'completed':
                employee_tasks.append(task)
        
        return employee_tasks
    
    def complete_task(self, task_id: str, employee_id: str, 
                     report: str = '') -> bool:
        """Завершение задачи с отчётом"""
        for task in self.team_data['tasks']:
            if task['id'] == task_id and task['employee_id'] == employee_id:
                task['status'] = 'completed'
                task['completed_at'] = datetime.now().isoformat()
                task['report'] = report
                
                # Удаляем из активных задач
                if task_id in self.team_data['employees'][employee_id]['active_tasks']:
                    self.team_data['employees'][employee_id]['active_tasks'].remove(task_id)
                
                self._save_data()
                return True
        return False
    
    def submit_daily_report(self, employee_id: str, report: str) -> bool:
        """Подача ежедневного отчёта"""
        if employee_id not in self.team_data['employees']:
            return False
            
        daily_report = {
            'employee_id': employee_id,
            'employee_name': self.team_data['employees'][employee_id]['name'],
            'report': report,
            'date': datetime.now().isoformat(),
            'tasks_completed': len([t for t in self.team_data['tasks'] 
                                  if t['employee_id'] == employee_id and 
                                  t['status'] == 'completed' and
                                  t['completed_at'] and
                                  datetime.fromisoformat(t['completed_at']).date() == datetime.now().date()])
        }
        
        self.team_data['reports'].append(daily_report)
        self.team_data['employees'][employee_id]['daily_reports'].append(daily_report)
        self._save_data()
        return True
    
    def get_team_status(self) -> Dict:
        """Получение статуса команды"""
        total_tasks = len(self.team_data['tasks'])
        completed_tasks = len([t for t in self.team_data['tasks'] if t['status'] == 'completed'])
        active_tasks = total_tasks - completed_tasks
        
        employee_stats = {}
        for emp_id, emp_data in self.team_data['employees'].items():
            emp_tasks = [t for t in self.team_data['tasks'] if t['employee_id'] == emp_id]
            emp_completed = len([t for t in emp_tasks if t['status'] == 'completed'])
            
            employee_stats[emp_data['name']] = {
                'role': emp_data['role'],
                'active_tasks': len(emp_data['active_tasks']),
                'completed_tasks': emp_completed,
                'total_tasks': len(emp_tasks)
            }
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'active_tasks': active_tasks,
            'completion_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'employees': employee_stats,
            'owner_vacation_mode': self.owner_vacation_mode
        }
    
    def enable_vacation_mode(self) -> bool:
        """Включение режима 'владелец в отпуске'"""
        self.owner_vacation_mode = True
        self._save_data()
        return True
    
    def disable_vacation_mode(self) -> bool:
        """Выключение режима 'владелец в отпуске'"""
        self.owner_vacation_mode = False
        self._save_data()
        return True
    
    async def send_reminder(self, employee_id: str, message: str) -> bool:
        """Отправка напоминания сотруднику"""
        if not self.bot_token or employee_id not in self.team_data['employees']:
            return False
            
        try:
            # Импортируем telegram только при необходимости
            import telegram
            bot = telegram.Bot(token=self.bot_token)
            chat_id = self.team_data['employees'][employee_id].get('chat_id')
            if chat_id:
                await bot.send_message(chat_id=chat_id, text=message)
                return True
        except Exception as e:
            print(f"Ошибка отправки напоминания: {e}")
        return False
    
    async def send_daily_reminders(self):
        """Отправка ежедневных напоминаний"""
        for emp_id, emp_data in self.team_data['employees'].items():
            active_tasks = self.get_employee_tasks(emp_id)
            if active_tasks:
                message = f"🔔 Доброе утро, {emp_data['name']}!\n\n"
                message += f"У вас {len(active_tasks)} активных задач:\n"
                
                for task in active_tasks[:3]:  # Показываем первые 3
                    deadline = datetime.fromisoformat(task['deadline'])
                    days_left = (deadline - datetime.now()).days
                    message += f"• {task['task']} (до {task['deadline']}, осталось {days_left} дн.)\n"
                
                if len(active_tasks) > 3:
                    message += f"... и ещё {len(active_tasks) - 3} задач\n"
                
                message += "\nНе забудьте отправить ежедневный отчёт командой: 'Отчёт: [ваш отчёт]'"
                
                await self.send_reminder(emp_id, message)
    
    def get_overdue_tasks(self) -> List[Dict]:
        """Получение просроченных задач"""
        overdue = []
        for task in self.team_data['tasks']:
            if task['status'] != 'completed':
                deadline = datetime.fromisoformat(task['deadline'])
                if deadline < datetime.now():
                    task['days_overdue'] = (datetime.now() - deadline).days
                    overdue.append(task)
        return overdue
    
    async def send_owner_vacation_notifications(self):
        """Отправка уведомлений в режиме 'владелец в отпуске'"""
        if not self.owner_vacation_mode:
            return
            
        # Проверяем просроченные задачи
        overdue_tasks = self.get_overdue_tasks()
        if overdue_tasks:
            message = "🚨 ВНИМАНИЕ! Просроченные задачи:\n\n"
            for task in overdue_tasks:
                emp_name = self.team_data['employees'][task['employee_id']]['name']
                message += f"• {emp_name}: {task['task']} (просрочено на {task['days_overdue']} дн.)\n"
            
            # Отправляем всем сотрудникам
            for emp_id in self.team_data['employees']:
                await self.send_reminder(emp_id, message)
        
        # Проверяем отсутствие отчётов
        today = datetime.now().date()
        for emp_id, emp_data in self.team_data['employees'].items():
            today_reports = [r for r in emp_data['daily_reports'] 
                           if datetime.fromisoformat(r['date']).date() == today]
            
            if not today_reports:
                await self.send_reminder(emp_id, 
                    f"⚠️ {emp_data['name']}, не забудьте отправить ежедневный отчёт!")

# Глобальный экземпляр
team_manager = TeamManager() 