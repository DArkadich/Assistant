"""
Модуль "Подготовка к встречам": сбор и синтез информации о контакте.
"""
from datetime import datetime, timedelta
from core.calendar import find_google_calendar_event_by_title_and_date
from core.amocrm import amocrm
from core.partners import partners_manager
from core.email_analyzer import email_analyzer
from core.memory import chat_memory
from core.team_manager import team_manager
from core.global_search import global_search

def prepare_for_meeting(person_name: str) -> dict:
    """Собирает всю информацию для подготовки к встрече."""
    report = {
        'person_name': person_name,
        'meeting': None,
        'contact_info': None,
        'partner_info': None,
        'last_emails': [],
        'last_messages': [],
        'related_tasks': [],
        'related_docs': [],
        'objective': "Цель не ясна. Сформулируй, чего ты хочешь добиться."
    }

    # 1. Поиск встречи в календаре (в ближайшие 7 дней)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    try:
        events = get_events_for_date_range(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        for event in events:
            if person_name.lower() in event.get('summary', '').lower():
                report['meeting'] = event
                break
    except Exception as e:
        print(f"Ошибка при поиске в календаре: {e}")

    # 2. Поиск в AmoCRM
    try:
        contacts = amocrm.get_contacts(query=person_name)
        if contacts:
            contact = contacts[0]
            report['contact_info'] = contact
            leads = amocrm.get_leads(contact_id=contact['id'])
            if leads:
                report['contact_info']['leads'] = leads
                # Пытаемся определить цель встречи
                open_leads = [l for l in leads if l['status_name'] not in ['Успешно реализовано', 'Закрыто и не реализовано']]
                if open_leads:
                    report['objective'] = f"Закрыть сделку: {open_leads[0]['name']} (статус: {open_leads[0]['status_name']}, бюджет: {open_leads[0]['price']})"
    except Exception as e:
        print(f"Ошибка при поиске в AmoCRM: {e}")
    
    # 3. Поиск в партнерской сети
    try:
        partner = partners_manager.find_partner_by_name(person_name)
        if partner:
            report['partner_info'] = partner
    except Exception as e:
        print(f"Ошибка при поиске в партнерах: {e}")

    # 4. Поиск почты
    contact_email = report.get('contact_info', {}).get('email')
    if contact_email:
        try:
            report['last_emails'] = email_analyzer.search_emails(query=f"from:{contact_email} OR to:{contact_email}", max_results=3)
        except Exception as e:
            print(f"Ошибка при поиске почты: {e}")

    # 5. Поиск сообщений в памяти
    try:
        report['last_messages'] = chat_memory.search(query=person_name, limit=5)
    except Exception as e:
        print(f"Ошибка при поиске в памяти: {e}")

    # 6. Поиск связанных задач
    try:
        all_tasks = team_manager.get_all_tasks()
        report['related_tasks'] = [t for t in all_tasks if person_name.lower() in t['description'].lower()]
        if not report['objective'].startswith("Закрыть сделку"):
             open_tasks = [t for t in report['related_tasks'] if not t['completed']]
             if open_tasks:
                 report['objective'] = f"Решить задачу: {open_tasks[0]['description']}"
    except Exception as e:
        print(f"Ошибка при поиске задач: {e}")

    # 7. Глобальный поиск по документам
    try:
        report['related_docs'] = global_search(person_name)
    except Exception as e:
        print(f"Ошибка глобального поиска: {e}")

    return report

def format_meeting_prep(data: dict) -> str:
    """Форматирует отчет для отправки в Telegram."""
    person = data['person_name']
    text = f"<b>Подготовка к встрече: {person}</b>\n\n"

    # Цель
    text += f"🎯 <b>Чего ты от него хочешь:</b>\n{data['objective']}\n\n"

    # Встреча
    if data['meeting']:
        meeting = data['meeting']
        start = meeting.get('start', {}).get('dateTime', meeting.get('start', {}).get('date', 'Нет данных'))
        start_dt = datetime.fromisoformat(start).strftime('%d %B, %H:%M') if 'T' in start else start
        text += f"🗓️ <b>Встреча:</b> {meeting.get('summary', 'Без темы')}\n"
        text += f"   <b>Когда:</b> {start_dt}\n"
        if meeting.get('location'):
            text += f"   <b>Где:</b> {meeting.get('location')}\n"
        text += "\n"

    # Контакт
    if data['contact_info']:
        contact = data['contact_info']
        text += f"👤 <b>Контакт в CRM:</b> {contact.get('name', '')}\n"
        if contact.get('company_name'):
             text += f"   <b>Компания:</b> {contact.get('company_name')}\n"
        if contact.get('position'):
             text += f"   <b>Должность:</b> {contact.get('position')}\n"
        if contact.get('phone'):
             text += f"   <b>Телефон:</b> {contact.get('phone')}\n"
        if contact.get('email'):
             text += f"   <b>Email:</b> {contact.get('email')}\n"
        if contact.get('leads'):
            text += "   <b>Сделки:</b>\n"
            for lead in contact['leads']:
                 status_emoji = "🟢" if lead['status_name'] == 'Успешно реализовано' else "🔴" if lead['status_name'] == 'Закрыто и не реализовано' else "🟡"
                 text += f"      {status_emoji} {lead['name']} ({lead['price']} руб.) - {lead['status_name']}\n"
        text += "\n"

    # Партнер
    if data['partner_info']:
        partner = data['partner_info']
        text += f"🤝 <b>Партнёр:</b>\n"
        text += f"   <b>Статус:</b> {partner.get('status')}\n"
        text += f"   <b>Канал:</b> {partner.get('channel')}\n"
        text += "\n"

    # Задачи
    if data['related_tasks']:
        text += f"📋 <b>Связанные задачи:</b>\n"
        for task in data['related_tasks']:
            status = "✅" if task.get('completed') else "⏳"
            text += f"   {status} {task.get('description')} (до {task.get('deadline')})\n"
        text += "\n"

    # Почта
    if data['last_emails']:
        text += "📧 <b>Последние письма:</b>\n"
        for email in data['last_emails']:
            text += f"   • От: {email.get('from', '')} | Тема: {email.get('subject', '')}\n"
        text += "\n"

    # Сообщения
    if data['last_messages']:
        text += "💬 <b>Последние сообщения:</b>\n"
        for msg in data['last_messages']:
            text += f"   • {msg.get('username', 'user')}: {msg.get('text', '')[:80]}...\n"
        text += "\n"

    # Документы
    if data['related_docs']:
        text += "📄 <b>Документы и заметки:</b>\n"
        for doc in data['related_docs']:
            text += f"   • {doc.get('type', 'Файл')}: {doc.get('title', 'Без названия')} (Источник: {doc.get('source', '')})\n"
        text += "\n"
        
    if len(text) < 200:
        text += "Информации по контакту не найдено. Возможно, стоит проверить написание имени."

    return text 