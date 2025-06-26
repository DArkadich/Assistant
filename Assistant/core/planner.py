# planner.py — планирование задач и целей
import json
import os
from datetime import datetime, timedelta

goals = []
goal_tasks = {}  # ключ — goal_text, значение — список задач
GOALS_FILE = 'goals.json'

# --- Persistence ---
def save_goals():
    data = {
        'goals': goals,
        'goal_tasks': goal_tasks
    }
    with open(GOALS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_goals():
    global goals, goal_tasks
    if os.path.exists(GOALS_FILE):
        with open(GOALS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            goals = data.get('goals', [])
            goal_tasks = data.get('goal_tasks', {})

load_goals()

# --- Goal logic ---
def set_goal(goal_text, deadline=None):
    """
    Установить цель.
    Пример: set_goal('3 млн выручки', deadline='2024-09-01')
    """
    goal = {
        'goal_text': goal_text,
        'deadline': deadline,
        'progress': 0
    }
    goals.append(goal)
    goal_tasks[goal_text] = []
    save_goals()
    return goal

def get_goals():
    """
    Получить список целей.
    Пример: get_goals()
    """
    return goals

def get_goal_progress(goal_text):
    """
    Получить прогресс по цели.
    Пример: get_goal_progress('3 млн выручки')
    """
    for goal in goals:
        if goal['goal_text'] == goal_text:
            return {'goal_text': goal['goal_text'], 'progress': goal['progress'], 'deadline': goal['deadline']}
    return None

def update_goal_progress(goal_text, progress):
    """
    Обновить прогресс по цели (в процентах).
    Пример: update_goal_progress('3 млн выручки', 40)
    """
    for goal in goals:
        if goal['goal_text'] == goal_text:
            goal['progress'] = progress
            save_goals()
            return goal
    return None

def add_goal_task(goal_text, task_text, date=None):
    """
    Добавить промежуточную задачу к цели.
    Пример: add_goal_task('3 млн выручки', 'Позвонить 10 клиентам', date='2024-06-12')
    """
    if goal_text not in goal_tasks:
        goal_tasks[goal_text] = []
    task = {'task_text': task_text, 'date': date, 'done': False}
    goal_tasks[goal_text].append(task)
    save_goals()
    return task

def get_goal_tasks(goal_text):
    """
    Получить список задач по цели.
    Пример: get_goal_tasks('3 млн выручки')
    """
    return goal_tasks.get(goal_text, [])

def mark_goal_task_done(goal_text, task_text):
    """
    Отметить задачу по цели как выполненную.
    Пример: mark_goal_task_done('3 млн выручки', 'Позвонить 10 клиентам')
    """
    tasks = goal_tasks.get(goal_text, [])
    for task in tasks:
        if task['task_text'] == task_text:
            task['done'] = True
            save_goals()
            return task
    return None

def suggest_daily_tasks(goal_text, total_value, start_date, end_date, unit="единиц"):
    """
    Автоматически разбить цель на ежедневные задачи.
    Пример: suggest_daily_tasks('3 млн выручки', 3000000, '2024-06-01', '2024-06-30', unit='руб.')
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1
    if days <= 0:
        return []
    daily_value = total_value // days
    tasks = []
    for i in range(days):
        date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        task_text = f"Достичь {daily_value} {unit} по цели '{goal_text}'"
        task = add_goal_task(goal_text, task_text, date=date)
        tasks.append(task)
    save_goals()
    return tasks 