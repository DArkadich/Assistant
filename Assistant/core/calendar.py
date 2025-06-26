# calendar.py — управление задачами и календарём
import json
import os
from datetime import datetime, timedelta

tasks = []
TASKS_FILE = 'tasks.json'

# --- Persistence ---
def save_tasks():
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

def load_tasks():
    global tasks
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            tasks.clear()
            tasks.extend(json.load(f))

load_tasks()

# --- Task logic ---
def add_task(task_text, date=None, time=None):
    """
    Добавить задачу.
    Пример: add_task('Встреча с Тигрой', date='2024-06-10', time='15:00')
    """
    task = {
        'id': len(tasks) + 1,
        'task_text': task_text,
        'date': date,
        'time': time,
        'done': False
    }
    tasks.append(task)
    save_tasks()
    return task

def get_daily_plan(date):
    """
    Получить задачи на день.
    Пример: get_daily_plan('2024-06-10')
    """
    return [t for t in tasks if t['date'] == date]

def get_week_plan(start_date=None):
    """
    Получить задачи на неделю (от start_date или от сегодня).
    Пример: get_week_plan('2024-06-10')
    """
    if not start_date:
        start = datetime.now().date()
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    week = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    return [t for t in tasks if t['date'] in week]

def delete_task(task_id):
    """
    Удалить задачу по id.
    Пример: delete_task(1)
    """
    global tasks
    tasks = [t for t in tasks if t['id'] != task_id]
    save_tasks()

def move_task(task_id, new_date):
    """
    Перенести задачу на другую дату.
    Пример: move_task(1, '2024-06-12')
    """
    for t in tasks:
        if t['id'] == task_id:
            t['date'] = new_date
            save_tasks()
            return t
    return None

def set_reminder(task_id, remind_at):
    """
    Установить напоминание (заглушка).
    Пример: set_reminder(1, '2024-06-10 14:50')
    """
    # TODO: Реализовать отправку напоминаний через Telegram
    pass

def mark_task_done(task_id):
    """
    Отметить задачу как выполненную.
    Пример: mark_task_done(1)
    """
    for t in tasks:
        if t['id'] == task_id:
            t['done'] = True
            save_tasks()
            return t
    return None

def get_tasks():
    """
    Получить все задачи.
    Пример: get_tasks()
    """
    return tasks 