"""
Модуль "Антиразрыв": приоритизация, разгрузка, фокус
"""
from core.calendar import get_daily_plan, find_google_calendar_event_by_title_and_date, delete_google_calendar_event
from core.team_manager import team_manager
from core.payment_control import payment_control
from datetime import datetime

def get_tasks_for_prioritization() -> str:
    """Собирает просроченные и сегодняшние задачи с высоким приоритетом."""
    today = datetime.now().strftime("%Y-%m-%d")
    tasks = get_daily_plan(today)
    overdue_tasks = team_manager.get_overdue_tasks()
    
    priority_tasks = [t for t in tasks if not t.get('done')]
    
    result = "<b>Приоритеты на сегодня:</b>\n\n"
    
    if overdue_tasks:
        result += "<b>Просроченные задачи:</b>\n"
        for task in overdue_tasks:
            result += f"🔥 {task['description']} (ответственный: @{task['assignee_telegram_id']}, до {task['deadline']})\n"
        result += "\n"
        
    if priority_tasks:
        result += "<b>Задачи на сегодня:</b>\n"
        for task in priority_tasks:
            result += f"• {task['task_text']} {task.get('time', '')}\n"
    
    if not overdue_tasks and not priority_tasks:
        return "На сегодня нет срочных задач. Можно отдохнуть."
        
    return result

def get_todays_meetings() -> list:
    """Возвращает список сегодняшних встреч из Google Calendar."""
    today = datetime.now().strftime("%Y-%m-%d")
    events = get_daily_plan(today)
    return events

def cancel_meetings_by_ids(event_ids: list) -> dict:
    """Отменяет события в календаре по их ID."""
    cancelled_count = 0
    errors_count = 0
    for event_id in event_ids:
        if delete_google_calendar_event(event_id):
            cancelled_count += 1
        else:
            errors_count += 1
    return {'cancelled': cancelled_count, 'errors': errors_count}

def get_critical_summary() -> str:
    """Формирует сводку только по критичным точкам."""
    report = payment_control.get_control_report()
    overdue_tasks = team_manager.get_overdue_tasks()
    
    summary = "<b>Критическая сводка:</b>\n\n"
    
    has_critical_info = False
    
    if report['critical_alerts']:
        summary += "<b>Контроль платежей:</b>\n"
        for alert in report['critical_alerts']:
            summary += f"🔴 {alert}\n"
        summary += "\n"
        has_critical_info = True
        
    if overdue_tasks:
        summary += "<b>Просроченные задачи:</b>\n"
        for task in overdue_tasks:
            summary += f"🔥 {task['description']} (@{task['assignee_telegram_id']}, до {task['deadline']})\n"
        has_critical_info = True
        
    if not has_critical_info:
        return "Критичных проблем нет. Всё под контролем."
        
    return summary 