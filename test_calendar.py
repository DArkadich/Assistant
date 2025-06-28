#!/usr/bin/env python3
"""
Тестовый скрипт для проверки интеграции с Google Calendar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import calendar

def test_google_calendar():
    print("Тестирование интеграции с Google Calendar...")
    
    # Тест 1: Проверка авторизации
    print("\n1. Проверка авторизации...")
    service = calendar.get_google_calendar_service()
    if service:
        print("✅ Сервис Google Calendar создан успешно")
    else:
        print("❌ Ошибка создания сервиса Google Calendar")
        return False
    
    # Тест 2: Добавление тестового события
    print("\n2. Добавление тестового события...")
    from datetime import datetime, timedelta
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    success = calendar.add_to_google_calendar(
        "Тестовое событие от AI-ассистента",
        tomorrow,
        "15:00",
        "Это тестовое событие для проверки интеграции"
    )
    
    if success:
        print("✅ Событие добавлено в Google Calendar")
    else:
        print("❌ Ошибка добавления события")
        return False
    
    # Тест 3: Получение событий
    print("\n3. Получение событий...")
    events = calendar.get_google_calendar_events(tomorrow)
    if events:
        print(f"✅ Найдено {len(events)} событий на {tomorrow}")
        for event in events:
            print(f"   - {event['summary']}")
    else:
        print("⚠️  Событий не найдено (возможно, событие еще не синхронизировалось)")
    
    # Тест 4: Добавление задачи через add_task
    print("\n4. Добавление задачи через add_task...")
    task = calendar.add_task(
        "Тестовая задача с Google Calendar",
        date=tomorrow,
        time="16:00"
    )
    print(f"✅ Задача добавлена: {task['task_text']} на {task['date']} в {task['time']}")
    
    # Тест 5: Получение плана дня
    print("\n5. Получение плана дня...")
    daily_plan = calendar.get_daily_plan(tomorrow)
    if daily_plan:
        print(f"✅ Найдено {len(daily_plan)} задач/событий на {tomorrow}")
        for task in daily_plan:
            source = "Google Calendar" if task.get('from_google_calendar') else "Локальная задача"
            print(f"   - {task['task_text']} ({task['time']}) [{source}]")
    else:
        print("⚠️  Задач не найдено")
    
    print("\n🎉 Тестирование завершено!")
    return True

if __name__ == "__main__":
    test_google_calendar() 