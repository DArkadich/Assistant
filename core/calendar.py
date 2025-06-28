# calendar.py — управление задачами и календарём
import json
import os
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
import dateparser

# Google Calendar API setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

tasks = []
TASKS_FILE = 'tasks.json'

MY_CALENDAR_ID = 'a36aec1de6b79b81968f7f03be7949680c7cd5391607b976bdbdb463021e868c@group.calendar.google.com'

# --- Google Calendar Authentication ---
def get_google_calendar_service():
    """Получить сервис Google Calendar API."""
    try:
        if not os.path.exists(CREDENTIALS_FILE):
            print(f"Файл {CREDENTIALS_FILE} не найден. Google Calendar интеграция отключена.")
            return None
        
        # Для Service Account используем ServiceAccountCredentials
        creds = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE, 
            scopes=SCOPES
        )
        
        service = build('calendar', 'v3', credentials=creds)
        return service
    except Exception as e:
        print(f"Ошибка при создании Google Calendar сервиса: {e}")
        return None

def add_to_google_calendar(task_text, date, time=None, description=""):
    """Добавить событие в Google Calendar."""
    service = get_google_calendar_service()
    if not service:
        return False
    
    try:
        # Формируем дату и время
        if time:
            start_time = f"{date}T{time}:00"
            end_time = f"{date}T{time}:30"  # 30 минут по умолчанию
        else:
            start_time = f"{date}T09:00:00"
            end_time = f"{date}T09:30:00"
        
        event = {
            'summary': task_text,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'Europe/Moscow',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Europe/Moscow',
            },
        }
        
        event = service.events().insert(calendarId=MY_CALENDAR_ID, body=event).execute()
        print(f"Событие добавлено в Google Calendar: {event.get('htmlLink')}")
        return True
    except Exception as e:
        print(f"Ошибка при добавлении в Google Calendar: {e}")
        return False

def get_google_calendar_events(date):
    """Получить события из Google Calendar на определённую дату."""
    service = get_google_calendar_service()
    if not service:
        return []
    
    try:
        # Время начала и конца дня
        start_time = f"{date}T00:00:00"
        end_time = f"{date}T23:59:59"
        
        events_result = service.events().list(
            calendarId=MY_CALENDAR_ID,
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        return events
    except Exception as e:
        print(f"Ошибка при получении событий из Google Calendar: {e}")
        return []

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
    
    # Добавляем в Google Calendar если указана дата
    if date:
        add_to_google_calendar(task_text, date, time, f"Задача из AI-ассистента")
    
    return task

def get_daily_plan(date):
    """
    Получить задачи на день.
    Пример: get_daily_plan('2024-06-10')
    """
    local_tasks = [t for t in tasks if t['date'] == date]
    
    # Добавляем события из Google Calendar
    google_events = get_google_calendar_events(date)
    for event in google_events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        if 'T' in start:  # Если есть время
            time = start.split('T')[1][:5]  # Берем только HH:MM
        else:
            time = None
        
        # Добавляем как локальную задачу, но с пометкой что это из Google Calendar
        task = {
            'id': f"gc_{event['id']}",
            'task_text': f"📅 {event['summary']}",
            'date': date,
            'time': time,
            'done': False,
            'from_google_calendar': True
        }
        local_tasks.append(task)
    
    # Сортируем по времени
    local_tasks.sort(key=lambda x: x['time'] or '23:59')
    
    return local_tasks

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

def find_google_calendar_event_by_title_and_date(title, date):
    """Найти событие по названию и дате. Возвращает event или None."""
    events = get_google_calendar_events(date)
    for event in events:
        if event.get('summary', '').strip().lower() == title.strip().lower():
            return event
    return None

def delete_google_calendar_event(event_id):
    """Удалить событие по event_id."""
    service = get_google_calendar_service()
    if not service:
        return False
    try:
        service.events().delete(calendarId=MY_CALENDAR_ID, eventId=event_id).execute()
        print(f"Событие {event_id} удалено из Google Calendar")
        return True
    except Exception as e:
        print(f"Ошибка при удалении события: {e}")
        return False

def update_google_calendar_event(event_id, new_title=None, new_time=None):
    """Обновить название и/или время события по event_id."""
    service = get_google_calendar_service()
    if not service:
        return False
    try:
        event = service.events().get(calendarId=MY_CALENDAR_ID, eventId=event_id).execute()
        updated = False
        if new_title:
            event['summary'] = new_title
            updated = True
        if new_time:
            # Меняем только время, дата остаётся прежней
            date = event['start']['dateTime'][:10]
            event['start']['dateTime'] = f"{date}T{new_time}:00"
            event['end']['dateTime'] = f"{date}T{new_time}:30"  # 30 минут по умолчанию
            updated = True
        if updated:
            service.events().update(calendarId=MY_CALENDAR_ID, eventId=event_id, body=event).execute()
            print(f"Событие {event_id} обновлено в Google Calendar")
            return True
        else:
            print("Нет изменений для обновления")
            return False
    except Exception as e:
        print(f"Ошибка при обновлении события: {e}")
        return False

def get_date_range_from_phrase(phrase):
    """Вернуть список дат (ГГГГ-ММ-ДД) по естественной фразе."""
    today = datetime.now().date()
    phrase = phrase.lower().strip()
    if 'прошл' in phrase and 'недел' in phrase:
        # Прошлая неделя (понедельник-воскресенье)
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
    elif 'следующ' in phrase and 'недел' in phrase:
        # Следующая неделя
        start = today + timedelta(days=(7 - today.weekday()))
        end = start + timedelta(days=6)
    elif 'сегодня' in phrase:
        start = end = today
    elif 'завтра' in phrase:
        start = end = today + timedelta(days=1)
    elif 'вчера' in phrase:
        start = end = today - timedelta(days=1)
    else:
        # Попробуем через dateparser
        dt = dateparser.parse(phrase, languages=['ru'])
        if dt:
            start = end = dt.date()
        else:
            return []
    # Список дат в формате ГГГГ-ММ-ДД
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end - start).days + 1)]
    return dates

def delete_all_google_calendar_events_in_range(date_list):
    """Удалить все события Google Calendar за список дат. Добавлен отладочный вывод."""
    service = get_google_calendar_service()
    if not service:
        return 0
    count = 0
    for date in date_list:
        events = get_google_calendar_events(date)
        print(f"[DEBUG] События на {date}: {len(events)}")
        for event in events:
            print(f"[DEBUG] Вижу событие: {event.get('summary')} (id={event.get('id')}) start={event.get('start')} end={event.get('end')}")
            try:
                service.events().delete(calendarId=MY_CALENDAR_ID, eventId=event['id']).execute()
                count += 1
            except Exception as e:
                print(f"Ошибка при удалении события {event['id']}: {e}")
    return count 