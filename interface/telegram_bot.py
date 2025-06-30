import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, CallbackQueryHandler
import openai
from core import calendar, finances, planner
import re
import json
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import dateparser
import asyncio
import threading
from core.calendar import *
from core.finances import *
from core.planner import *
from core.drive_manager import drive_manager
from core.image_processor import image_processor
from core.goals import goals_manager, GoalType, GoalPeriod
from core.memory import chat_memory
from core.speech_recognition import speech_recognizer
from core.email_analyzer import email_analyzer
from core.partners import partners_manager
from core.amocrm import amocrm
from core.obsidian_manager import obsidian_manager
from core.ai_critic import analyze_decision, format_critic_result
from email.message import EmailMessage
# from email.policy import EmailPriority, EmailStatus # ВРЕМЕННО ОТКЛЮЧЕНО, т.к. вызывает ImportError
import core.analytics
import core.team_manager
import core.payment_control
import core.inbox_monitor
import core.global_search
import core.digest
from core.ai_critic import analyze_decision, format_critic_result
import core.antistress
import core.meeting_prep
from core.focus import focus_manager
from core.work_mode import work_mode_manager
from core.notification_manager import notification_manager
from typing import Optional
from core.document_assistant import document_assistant
from core.speech_synthesizer import speech_synthesizer
from core.inbox_monitor import inbox_monitor
from core.deadline_monitor import deadline_monitor
from core.meeting_assistant import meeting_assistant
import tempfile

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
SMART_MODEL = os.getenv("OPENAI_SMART_MODEL", "gpt-4-1106-preview")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Управление состоянием тихого режима ---
STATE_FILE = 'bot_state.json'

def get_bot_state():
    """Загружает состояние бота из файла."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}

def save_bot_state(state):
    """Сохраняет состояние бота в файл."""
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

def set_quiet_mode(chat_id: int, enabled: bool):
    """Включает или выключает тихий режим для чата."""
    state = get_bot_state()
    if 'quiet_mode' not in state:
        state['quiet_mode'] = {}
    state['quiet_mode'][str(chat_id)] = enabled
    save_bot_state(state)

def is_quiet_mode_enabled(chat_id: int) -> bool:
    """Проверяет, включен ли тихий режим для чата."""
    state = get_bot_state()
    return state.get('quiet_mode', {}).get(str(chat_id), False)

# Примитивный роутер: если в сообщении есть ключевые слова — используем GPT-4.1
SMART_KEYWORDS = [
    "аналитика", "отчёт", "KPI", "генерируй", "письмо", "КП", "оффер", "сложный", "прогноз", "диаграмма", "выручка", "инвестор"
]

# --- GPT intent parsing for tasks ---
TASK_PROMPT = (
    "Ты — ассистент, который помогает вести задачи пользователя. "
    "На входе — фраза на русском языке. "
    "Верни JSON с полями: intent (add/view/delete/move/done/summary), "
    "task_text, date (ГГГГ-ММ-ДД если явно указана, иначе null), time (ЧЧ:ММ или null), new_date (если перенос), task_id (если есть). "
    "Если дата не указана явно, верни null. Не придумывай дату. "
    "Пример: {\"intent\": \"add\", \"task_text\": \"Встреча с Тигрой\", \"date\": \"2024-06-10\", \"time\": \"15:00\"}"
)

# --- GPT intent parsing for finances ---
FIN_PROMPT = (
    "Ты — ассистент, который ведёт финансовый учёт. "
    "На входе — фраза на русском языке. "
    "Верни JSON с полями: intent (income/expense/report/unclassified), "
    "amount (число), project (строка), description (строка), date (ГГГГ-ММ-ДД или null), period (строка или null), category (строка или null). "
    "Примеры: "
    "{\"intent\": \"income\", \"amount\": 400000, \"project\": \"ВБ\", \"description\": \"Поступили от ВБ\", \"date\": \"2024-06-10\"}"
    "{\"intent\": \"expense\", \"amount\": 15000, \"project\": \"Horien\", \"description\": \"Закупка коробок\", \"date\": \"2024-06-10\", \"category\": \"упаковка\"}"
    "{\"intent\": \"report\", \"period\": \"июнь\", \"project\": \"Horien\"}"
    "{\"intent\": \"unclassified\"}"
)

def choose_model(user_text: str) -> str:
    for word in SMART_KEYWORDS:
        if word.lower() in user_text.lower():
            return SMART_MODEL
    return DEFAULT_MODEL

async def ask_openai(user_text: str, system_prompt: str = None) -> str:
    model = choose_model(user_text)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

async def parse_task_intent(user_text: str):
    """Отправляет текст в GPT и возвращает dict с интентом и параметрами для задач."""
    gpt_response = await ask_openai(user_text, system_prompt=TASK_PROMPT)
    try:
        data = json.loads(gpt_response)
        return data
    except Exception:
        return None

async def parse_finance_intent(user_text: str):
    gpt_response = await ask_openai(user_text, system_prompt=FIN_PROMPT)
    try:
        data = json.loads(gpt_response)
        return data
    except Exception:
        return None

async def send_daily_summary(update: Update):
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Фокус дня
    focus = focus_manager.get_daily_focus()
    focus_text = f"<b>{focus}</b>" if focus else "Не задан"
    
    # Задачи
    tasks = calendar.get_daily_plan(today)
    if tasks:
        # Разделяем задачи на фокусные и нефокусные
        focus_tasks = []
        other_tasks = []
        for t in tasks:
            if focus_manager.is_task_in_focus(t['task_text']):
                focus_tasks.append(t)
            else:
                other_tasks.append(t)
        
        # Форматируем задачи
        if focus_tasks:
            tasks_text = "🎯 В фокусе:\n" + "\n".join([
                f"- {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}"
                for t in focus_tasks
            ])
            if other_tasks:
                tasks_text += "\n\n⚪️ Остальные задачи:\n" + "\n".join([
                    f"- {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}"
                    for t in other_tasks
                ])
        else:
            tasks_text = "\n".join([
                f"- {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}"
                for t in tasks
            ])
    else:
        tasks_text = "Нет задач."
    
    # Цели
    goals = planner.get_goals()
    if goals:
        goals_text = "\n".join([f"- {g['goal_text']} — {g['progress']}% (до {g['deadline']})" for g in goals])
    else:
        goals_text = "Нет целей."
    
    # Финансы (за сегодня)
    report = finances.get_report(period=today)
    finance_text = f"Доход: {report['income']}, Расход: {report['expense']}, Прибыль: {report['profit']}"
    # --- Чистый остаток общий ---
    total_balance = finances.get_total_balance()
    finance_text += f"\nЧистый остаток: {total_balance}"
    # --- Разметка по проектам ---
    project_reports = finances.get_report_by_project(period=today)
    if project_reports:
        finance_text += "\n\n<b>По проектам:</b>"
        for project, rep in project_reports.items():
            balance = finances.get_total_balance(project=project)
            finance_text += f"\n- {project}: Доход {rep['income']}, Расход {rep['expense']}, Прибыль {rep['profit']}, Остаток {balance}"
    
    # Итог
    summary = f"🎯 Фокус дня: {focus_text}\n\n🗓️ План на сегодня:\n{tasks_text}\n\n🎯 Цели:\n{goals_text}\n\n💰 Финансы за сегодня:\n{finance_text}"
    await update.message.reply_text(summary, parse_mode='HTML')

async def send_weekly_summary(update: Update):
    from core import calendar, planner, finances
    import pytz
    today = datetime.now(pytz.timezone('Europe/Moscow')).date()
    week_dates = [(today + timedelta(days=i)) for i in range(7)]
    week_strs = [d.strftime('%Y-%m-%d') for d in week_dates]
    week_days = [d.strftime('%A, %d %B').capitalize() for d in week_dates]

    # Категории для задач и целей
    CATEGORIES = [
        ("💸 Финансы", ["финансы", "оплата", "бюджет", "расход", "доход", "платёж", "счёт", "invoice", "pnl"]),
        ("👥 Команда", ["команда", "сотрудник", "встреча", "отчёт", "бриф", "митинг", "собрание", "hr"]),
        ("📈 Продажи", ["продажа", "кп", "оффер", "лид", "сделка", "crm", "amocrm", "продажи", "клиент"]),
        ("🤝 Клиенты", ["клиент", "обратная связь", "отправить", "договор", "контракт", "подписать", "документ"]),
    ]
    def categorize(text):
        for cat, keywords in CATEGORIES:
            for kw in keywords:
                if kw in text.lower():
                    return cat
        return "🔹 Другое"

    # Сбор задач по категориям
    tasks_by_cat = {cat: [] for cat, _ in CATEGORIES}
    tasks_by_cat["🔹 Другое"] = []
    for d in week_strs:
        for t in calendar.get_daily_plan(d):
            if t.get('done'): continue
            cat = categorize(t['task_text'])
            # Определяем статус и дедлайн
            deadline = d
            status = "в работе"
            if t.get('from_google_calendar'):
                status = "событие"
            # Просрочено?
            overdue = datetime.strptime(d, '%Y-%m-%d').date() < today
            soon = 0 <= (datetime.strptime(d, '%Y-%m-%d').date() - today).days <= 1
            reminder = " 🔔" if soon and not t.get('done') else ""
            overdue_mark = " ⏰ ПРОСРОЧЕНО" if overdue and not t.get('done') else ""
            tasks_by_cat[cat].append(f"[{d}] {t['task_text']} (статус: {status}){reminder}{overdue_mark}")

    # Сбор целей по категориям
    goals = planner.get_goals()
    goals_by_cat = {cat: [] for cat, _ in CATEGORIES}
    goals_by_cat["🔹 Другое"] = []
    for goal in goals:
        cat = categorize(goal['goal_text'])
        deadline = goal.get('deadline')
        progress = goal.get('progress', 0)
        days_left = (datetime.strptime(deadline, '%Y-%m-%d').date() - today).days if deadline else None
        soon = days_left is not None and 0 <= days_left <= 1
        overdue = days_left is not None and days_left < 0
        reminder = " 🔔" if soon else ""
        overdue_mark = " ⏰ ПРОСРОЧЕНО" if overdue else ""
        goals_by_cat[cat].append(f"{goal['goal_text']} — {progress}% (до {deadline or '—'}){reminder}{overdue_mark}")

    # Финансы за неделю
    week_income = finances.get_income_for_week()
    week_expense = finances.get_expense_for_week()
    finance_block = f"Доход: {week_income['total_amount']} руб.\nРасход: {week_expense['total_amount']} руб.\nПрибыль: {week_income['total_amount'] - week_expense['total_amount']} руб."

    # --- План по предстоящим платежам ---
    upcoming_payments = []
    for p in getattr(finances, 'payments', []):
        try:
            pay_date = datetime.strptime(p['date'], '%Y-%m-%d').date()
        except Exception:
            continue
        if today <= pay_date <= week_dates[-1]:
            upcoming_payments.append(p)
    upcoming_payments.sort(key=lambda p: p['date'])
    payments_block = ""
    if upcoming_payments:
        payments_block += "\n💳 <b>План по предстоящим платежам:</b>\n"
        for p in upcoming_payments:
            pay_date = p['date']
            overdue = datetime.strptime(pay_date, '%Y-%m-%d').date() < today
            soon = 0 <= (datetime.strptime(pay_date, '%Y-%m-%d').date() - today).days <= 1
            reminder = " 🔔" if soon else ""
            overdue_mark = " ⏰ ПРОСРОЧЕНО" if overdue else ""
            closed = finances.is_payment_closed(p)
            status = "✅ закрыт" if closed else "🕓 не закрыт"
            payments_block += (
                f"[{pay_date}] {p['amount']} руб. ({p['project']}) — {p['counterparty']}\n"
                f"   Назначение: {p['purpose']} | Статус: {status}{reminder}{overdue_mark}\n"
            )
    # Формируем итоговый текст
    summary = "🗓️ <b>План на неделю</b>\n"
    for cat, _ in CATEGORIES + [("🔹 Другое", [])]:
        if tasks_by_cat[cat] or goals_by_cat[cat]:
            summary += f"\n{cat}:\n"
            for t in tasks_by_cat[cat]:
                summary += f"- {t}\n"
            for g in goals_by_cat[cat]:
                summary += f"🎯 {g}\n"
    # Финансы
    summary += f"\n💰 <b>Финансы за неделю:</b>\n{finance_block}"
    # Платежи
    if payments_block:
        summary += payments_block
    await update.message.reply_text(summary, parse_mode='HTML')

# --- Для рассылки сводки ---
last_chat_id_file = 'last_chat_id.txt'
def save_last_chat_id(chat_id):
    with open(last_chat_id_file, 'w') as f:
        f.write(str(chat_id))
def load_last_chat_id():
    if os.path.exists(last_chat_id_file):
        with open(last_chat_id_file, 'r') as f:
            return int(f.read().strip())
    return None

async def send_daily_summary_to_chat(app, chat_id):
    if is_quiet_mode_enabled(chat_id):
        print(f"[Scheduler] Тихий режим для чата {chat_id}, ежедневная сводка пропущена.")
        return
    today = datetime.now().strftime("%Y-%m-%d")
    tasks = calendar.get_daily_plan(today)
    if tasks:
        tasks_text = "\n".join([f"- {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
    else:
        tasks_text = "Нет задач."
    goals = planner.get_goals()
    if goals:
        goals_text = "\n".join([f"- {g['goal_text']} — {g['progress']}% (до {g['deadline']})" for g in goals])
    else:
        goals_text = "Нет целей."
    report = finances.get_report(period=today)
    finance_text = f"Доход: {report['income']}, Расход: {report['expense']}, Прибыль: {report['profit']}"
    # --- Чистый остаток общий ---
    total_balance = finances.get_total_balance()
    finance_text += f"\nЧистый остаток: {total_balance}"
    # --- Разметка по проектам ---
    project_reports = finances.get_report_by_project(period=today)
    if project_reports:
        finance_text += "\n\n<b>По проектам:</b>"
        for project, rep in project_reports.items():
            balance = finances.get_total_balance(project=project)
            finance_text += f"\n- {project}: Доход {rep['income']}, Расход {rep['expense']}, Прибыль {rep['profit']}, Остаток {balance}"
    summary = f"🗓️ План на сегодня:\n{tasks_text}\n\n🎯 Цели:\n{goals_text}\n\n💰 Финансы за сегодня:\n{finance_text}"
    await app.bot.send_message(chat_id=chat_id, text=summary)

# --- Scheduler ---
def start_scheduler(app):
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Europe/Moscow'))
    
    def job():
        chat_id = load_last_chat_id()
        if chat_id:
            # Получаем текущий event loop или создаем новый
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Если нет активного loop, создаем новый
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Создаем и запускаем задачу
            try:
                if loop.is_running():
                    # Если loop уже запущен, создаем task
                    asyncio.create_task(send_daily_summary_to_chat(app, chat_id))
                else:
                    # Если loop не запущен, запускаем его
                    loop.run_until_complete(send_daily_summary_to_chat(app, chat_id))
            except Exception as e:
                print(f"Error in scheduler job: {e}")
    
    scheduler.add_job(job, 'cron', hour=8, minute=0)
    scheduler.start()

def validate_task_date(date_str):
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().date()
        if dt.date() < today:
            return None  # дата в прошлом, вероятно ошибка
        return date_str
    except Exception:
        return None

def extract_date_phrase(text):
    patterns = [
        r"завтра в \d{1,2}(:\d{2})?",
        r"послезавтра в \d{1,2}(:\d{2})?",
        r"завтра[\w\s:]*",
        r"послезавтра[\w\s:]*",
        r"сегодня[\w\s:]*",
        r"через [^ ]+",
        r"в \d{1,2}:\d{2}",
        r"в \w+",  # в пятницу
        r"на следующей неделе",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return text

def parse_natural_date(text):
    dt = dateparser.parse(text, languages=['ru'])
    if dt:
        return dt.strftime('%Y-%m-%d')
    return None

def smart_hour_from_phrase(phrase):
    # Если явно указано "утра", "ночи", "am" — 03:00
    if re.search(r"(утра|ночи|am)", phrase, re.IGNORECASE):
        return "03:00"
    # Если явно указано "дня", "вечера", "pm" — 15:00
    if re.search(r"(дня|вечера|pm)", phrase, re.IGNORECASE):
        return "15:00"
    # Если просто "в 3" — считаем 15:00 (рабочий день)
    hour_match = re.search(r"в (\d{1,2})(?!:)", phrase)
    if hour_match:
        hour = int(hour_match.group(1))
        if 1 <= hour <= 8:
            return f"{hour+12:02d}:00"  # 3 → 15:00, 8 → 20:00
        elif 9 <= hour <= 20:
            return f"{hour:02d}:00"
        else:
            return "15:00"
    return "09:00"

def parse_natural_date_and_time(text):
    phrase = extract_date_phrase(text)
    dt = dateparser.parse(phrase, languages=['ru'], settings={'PREFER_DATES_FROM': 'future'})
    if dt:
        date = dt.strftime('%Y-%m-%d')
        # Если время явно указано, используем его, иначе применяем smart_hour_from_phrase
        if re.search(r'\d{1,2}:\d{2}', phrase):
            time = dt.strftime('%H:%M')
        else:
            time = smart_hour_from_phrase(phrase)
        print(f"[DEBUG] dateparser: text='{text}', phrase='{phrase}', parsed_date='{date}', parsed_time='{time}'")
        return date, time
    print(f"[DEBUG] dateparser: text='{text}', phrase='{phrase}', parsed_date=None, parsed_time=None")
    return None, None

# --- Пуллинг Google Calendar ---
last_polled_events = {}

def start_calendar_polling(app):
    def poll():
        while True:
            now = datetime.now(pytz.timezone('Europe/Moscow'))
            if 9 <= now.hour < 20:
                interval = 300  # 5 минут
            else:
                interval = 3600  # 60 минут
            try:
                chat_id = load_last_chat_id()
                if chat_id:
                    check_calendar_changes_and_notify(app, chat_id)
            except Exception as e:
                print(f"[Calendar Polling] Error: {e}")
            finally:
                import time
                time.sleep(interval)
    t = threading.Thread(target=poll, daemon=True)
    t.start()

def check_calendar_changes_and_notify(app, chat_id):
    if is_quiet_mode_enabled(chat_id):
        print(f"[Calendar Polling] Тихий режим для чата {chat_id}, уведомления календаря пропущены.")
        return
    """Проверяет изменения в календаре и уведомляет пользователя."""
    from core.calendar import get_google_calendar_events
    global last_polled_events
    today = datetime.now().strftime('%Y-%m-%d')
    events = get_google_calendar_events(today)
    event_map = {e['id']: (e['summary'], e['start'].get('dateTime', e['start'].get('date'))) for e in events}
    # Сравниваем с предыдущим состоянием
    if last_polled_events:
        # Новые события
        new_events = [e for eid, e in event_map.items() if eid not in last_polled_events]
        # Изменённые события
        changed_events = [eid for eid in event_map if eid in last_polled_events and event_map[eid] != last_polled_events[eid]]
        # Удалённые события
        deleted_events = [eid for eid in last_polled_events if eid not in event_map]
        # Уведомления
        try:
            loop = asyncio.get_event_loop()
            if new_events:
                for summary, start in new_events:
                    asyncio.run_coroutine_threadsafe(
                        app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Новое событие: {summary} ({start})"),
                        loop
                    )
            if changed_events:
                for eid in changed_events:
                    summary, start = event_map[eid]
                    asyncio.run_coroutine_threadsafe(
                        app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Изменено событие: {summary} ({start})"),
                        loop
                    )
            if deleted_events:
                for eid in deleted_events:
                    summary, start = last_polled_events[eid]
                    asyncio.run_coroutine_threadsafe(
                        app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Удалено событие: {summary} ({start})"),
                        loop
                    )
        except RuntimeError:
            # Если нет event loop в текущем потоке, просто логируем
            print(f"[Calendar Polling] No event loop available for notifications")
    last_polled_events = event_map.copy()

# --- Расширение handle_message ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username or update.message.from_user.full_name
    # Сохраняем сообщение пользователя в память
    chat_memory.add_message(user_id=user_id, username=username, text=user_text, role="user")
    print(f"[DEBUG] Получено сообщение: {user_text}")
    save_last_chat_id(update.effective_chat.id)

    # Проверяем режим "На объекте"
    if await handle_on_site_mode(update, context):
        return
    
    # --- Проверка ожидания даты для финансов ---
    if context.user_data.get('awaiting_fin_date'):
        fin_intent = context.user_data.pop('awaiting_fin_date')
        import dateparser
        dt = dateparser.parse(user_text, languages=['ru'])
        if not dt:
            await update.message.reply_text("Не удалось распознать дату. Пожалуйста, укажи дату в формате ГГГГ-ММ-ДД или естественно (например, 'вчера', '25 июня').")
            context.user_data['awaiting_fin_date'] = fin_intent
            return
        fin_intent['date'] = dt.strftime('%Y-%m-%d')
        if fin_intent['intent'] == 'income':
            op = finances.add_income(
                fin_intent.get("amount"),
                fin_intent.get("project"),
                description=fin_intent.get("description"),
                date=fin_intent.get("date")
            )
            await update.message.reply_text(f"Доход добавлен: {op['amount']} ({op['project']}) — {op['description']} ({op['date']})")
        elif fin_intent['intent'] == 'expense':
            op = finances.add_expense(
                fin_intent.get("amount"),
                fin_intent.get("project"),
                description=fin_intent.get("description"),
                date=fin_intent.get("date"),
                category=fin_intent.get("category")
            )
            await update.message.reply_text(f"Расход добавлен: {op['amount']} ({op['project']}) — {op['description']} ({op['date']})")
        return

    # --- Проверка ожидания выбора платежа для документа ---
    if context.user_data.get('awaiting_payment_choice'):
        doc_data = context.user_data.pop('awaiting_payment_choice')
        try:
            payment_index = int(user_text.strip()) - 1
            all_payments = finances.payments
            if 0 <= payment_index < len(all_payments):
                payment = all_payments[payment_index]
                doc = finances.add_ved_document(
                    doc_type=doc_data['doc_type'],
                    number=doc_data['number'],
                    date=doc_data['date'],
                    payment_ids=[payment['id']]
                )
                await update.message.reply_text(f"Документ добавлен: {doc_data['doc_type']} №{doc_data['number']} от {doc_data['date']} для платежа {payment['counterparty']} ({payment['id']})")
            else:
                await update.message.reply_text("Неверный номер платежа. Попробуйте снова.")
                context.user_data['awaiting_payment_choice'] = doc_data
        except ValueError:
            await update.message.reply_text("Пожалуйста, введите число (номер платежа).")
            context.user_data['awaiting_payment_choice'] = doc_data
        return

    # --- Управление событиями Google Calendar ---
    # Удаление события
    m = re.match(r"удали событие ([^\n]+) (\d{4}-\d{2}-\d{2})", user_text, re.I)
    if m:
        title, date = m.group(1).strip(), m.group(2)
        event = calendar.find_google_calendar_event_by_title_and_date(title, date)
        if event:
            ok = calendar.delete_google_calendar_event(event['id'])
            if ok:
                await update.message.reply_text(f"Событие '{title}' на {date} удалено из Google Calendar.")
            else:
                await update.message.reply_text(f"Ошибка при удалении события '{title}' на {date}.")
        else:
            await update.message.reply_text(f"Событие '{title}' на {date} не найдено.")
        return
    # Переименование события
    m = re.match(r"переименуй событие ([^\n]+) (\d{4}-\d{2}-\d{2}) в ([^\n]+)", user_text, re.I)
    if m:
        old_title, date, new_title = m.group(1).strip(), m.group(2), m.group(3).strip()
        event = calendar.find_google_calendar_event_by_title_and_date(old_title, date)
        if event:
            ok = calendar.update_google_calendar_event(event['id'], new_title=new_title)
            if ok:
                await update.message.reply_text(f"Событие '{old_title}' на {date} переименовано в '{new_title}'.")
            else:
                await update.message.reply_text(f"Ошибка при переименовании события '{old_title}'.")
        else:
            await update.message.reply_text(f"Событие '{old_title}' на {date} не найдено.")
        return
    # Перенос события
    m = re.match(r"перенеси событие ([^\n]+) (\d{4}-\d{2}-\d{2}) на (\d{2}:\d{2})", user_text, re.I)
    if m:
        title, date, new_time = m.group(1).strip(), m.group(2), m.group(3)
        event = calendar.find_google_calendar_event_by_title_and_date(title, date)
        if event:
            ok = calendar.update_google_calendar_event(event['id'], new_time=new_time)
            if ok:
                await update.message.reply_text(f"Событие '{title}' на {date} перенесено на {new_time}.")
            else:
                await update.message.reply_text(f"Ошибка при переносе события '{title}'.")
        else:
            await update.message.reply_text(f"Событие '{title}' на {date} не найдено.")
        return
    # Массовое удаление событий/встреч/мероприятий/ивентов по диапазону
    m = re.match(r"удали все (события|встречи|мероприятия|ивенты|event[ыe]?|событие) ?(из календаря)? (за|на) ([^\n]+)", user_text, re.I)
    if m:
        phrase = m.group(4).strip()
        date_list = calendar.get_date_range_from_phrase(phrase)
        if not date_list:
            await update.message.reply_text(f"Не удалось определить диапазон дат по фразе: '{phrase}'")
            return
        count = calendar.delete_all_google_calendar_events_in_range(date_list)
        await update.message.reply_text(f"Удалено {count} событий из Google Calendar за: {', '.join(date_list)}")
        return
    # --- Сводка по естественным фразам ---
    if re.search(r"(что у меня|план на сегодня|утренняя сводка|дай сводку|сегодня)", user_text, re.I):
        await send_daily_summary(update)
        return
    if re.search(r"(план на неделю|неделя|недельная сводка)", user_text, re.I):
        await send_weekly_summary(update)
        return
    
    # --- Управление фокусом дня ---
    if await handle_focus_commands(update, context):
        return
    
    # --- Документооборот и документы ---
    # Добавление платежа
    if re.search(r"добавь платёж.*рубл", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду добавления платежа: {user_text}")
        m = re.match(r"добавь платёж (\d+) рублей? ([^\n]+) (входящий|исходящий) (в Россию|за границу) проект ([^\n]+) контрагент ([^\n]+) назначение ([^\n]+)", user_text, re.I)
        if m:
            amount = int(m.group(1))
            date_phrase = m.group(2).strip()
            direction = 'in' if m.group(3) == 'входящий' else 'out'
            country = 'RU' if m.group(4) == 'в Россию' else 'INT'
            project = m.group(5).strip()
            counterparty = m.group(6).strip()
            purpose = m.group(7).strip()
            
            # Парсим дату
            import dateparser
            dt = dateparser.parse(date_phrase, languages=['ru'])
            if not dt:
                await update.message.reply_text("Не удалось распознать дату. Укажи дату в формате ГГГГ-ММ-ДД или естественно (например, 'вчера').")
                return
            
            payment = finances.add_payment(
                amount=amount,
                date=dt.strftime('%Y-%m-%d'),
                direction=direction,
                country=country,
                project=project,
                counterparty=counterparty,
                purpose=purpose
            )
            await update.message.reply_text(f"Платёж добавлен: {amount} руб. ({direction}, {country}, {project}) — {counterparty} ({purpose})")
            return
        else:
            await update.message.reply_text("Формат: 'Добавь платёж <сумма> рублей <дата> <входящий/исходящий> <в Россию/за границу> проект <название> контрагент <название> назначение <описание>'")
            return
    
    # Добавление закупки
    if re.search(r"добавь закупку", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду добавления закупки: {user_text}")
        m = re.match(r"добавь закупку ([^\n]+) (\d+) рублей? ([^\n]+)", user_text, re.I)
        if m:
            name = m.group(1).strip()
            amount = int(m.group(2))
            date_phrase = m.group(3).strip()
            
            # Парсим дату
            import dateparser
            dt = dateparser.parse(date_phrase, languages=['ru'])
            if not dt:
                await update.message.reply_text("Не удалось распознать дату. Укажи дату в формате ГГГГ-ММ-ДД или естественно.")
                return
            
            purchase = finances.add_purchase(
                name=name,
                amount=amount,
                date=dt.strftime('%Y-%m-%d')
            )
            await update.message.reply_text(f"Закупка добавлена: {name} — {amount} руб. ({purchase['date']})")
            return
        else:
            await update.message.reply_text("Формат: 'Добавь закупку <название> <сумма> рублей <дата>'")
            return
    
    # Добавление документа
    if re.search(r"добавь документ.*(накладная|упд|гтд|счёт|контракт|акт)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду добавления документа: {user_text}")
        m = re.match(r"добавь документ (накладная|упд|гтд|счёт|контракт|акт) номер ([^\n]+) от ([^\n]+) для платежа ([a-f0-9-]+)", user_text, re.I)
        if m:
            doc_type = m.group(1)
            number = m.group(2).strip()
            date_phrase = m.group(3).strip()
            payment_id = m.group(4)
            
            # Парсим дату
            import dateparser
            dt = dateparser.parse(date_phrase, languages=['ru'])
            if not dt:
                await update.message.reply_text("Не удалось распознать дату. Укажи дату в формате ГГГГ-ММ-ДД или естественно.")
                return
            
            # Проверяем существование платежа
            payment = finances.find_payment_by_id(payment_id)
            if not payment:
                await update.message.reply_text(f"Платёж с ID {payment_id} не найден.")
                return
            
            doc = finances.add_document(
                doc_type=doc_type,
                number=number,
                date=dt.strftime('%Y-%m-%d'),
                payment_ids=[payment_id]
            )
            await update.message.reply_text(f"Документ добавлен: {doc_type} №{number} от {doc['date']} для платежа {payment_id}")
            return
        else:
            await update.message.reply_text("Формат: 'Добавь документ <тип> номер <номер> от <дата> для платежа <ID>'")
            return
    
    # Добавление документа через естественный язык (автопоиск платежа)
    if re.search(r"добавь (накладная|упд|гтд|счёт|контракт|акт).*для.*платежа", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду добавления документа (автопоиск): {user_text}")
        # Ищем тип документа
        doc_type_match = re.search(r"(накладная|упд|гтд|счёт|контракт|акт)", user_text, re.I)
        if not doc_type_match:
            await update.message.reply_text("Не удалось определить тип документа.")
            return
        doc_type = doc_type_match.group(1)
        
        # Ищем номер документа
        number_match = re.search(r"номер ([^\s]+)", user_text, re.I)
        if not number_match:
            await update.message.reply_text("Не удалось найти номер документа.")
            return
        number = number_match.group(1)
        
        # Ищем дату
        date_match = re.search(r"от ([^\s]+)", user_text, re.I)
        if not date_match:
            await update.message.reply_text("Не удалось найти дату документа.")
            return
        date_phrase = date_match.group(1)
        
        # Парсим дату
        import dateparser
        dt = dateparser.parse(date_phrase, languages=['ru'])
        if not dt:
            await update.message.reply_text("Не удалось распознать дату. Укажи дату в формате ГГГГ-ММ-ДД или естественно.")
            return
        
        # Ищем контрагента в тексте
        all_payments = finances.payments
        if not all_payments:
            await update.message.reply_text("Платежей нет. Сначала добавьте платёж.")
            return
        
        # Показываем список платежей для выбора
        text = f"Выберите платёж для документа {doc_type} №{number} от {dt.strftime('%Y-%m-%d')}:\n"
        for i, payment in enumerate(all_payments, 1):
            text += f"{i}. {payment['amount']} руб. — {payment['counterparty']} ({payment['date']}) [ID: {payment['id']}]\n"
        
        # Сохраняем данные документа в контекст
        context.user_data['awaiting_payment_choice'] = {
            'doc_type': doc_type,
            'number': number,
            'date': dt.strftime('%Y-%m-%d')
        }
        
        await update.message.reply_text(text + "\nОтветьте номером платежа (1, 2, 3...)")
        return
    
    # Просмотр незакрытых платежей
    if re.search(r"(покажи незакрытые платежи|незакрытые платежи|просроченные документы)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра незакрытых платежей: {user_text}")
        unclosed = finances.get_unclosed_payments()
        if not unclosed:
            await update.message.reply_text("Все платежи закрыты документами.")
        else:
            text = "Незакрытые платежи:\n"
            for payment in unclosed:
                required = finances.get_required_docs_for_payment(payment)
                docs = [finances.find_document_by_id(doc_id) for doc_id in payment['documents_ids']]
                doc_types = [d['type'] for d in docs if d]
                
                # Отладочная информация
                debug_info = f"\n[DEBUG] Платёж {payment['id']}:"
                debug_info += f"\n  Требуемые документы: {required}"
                debug_info += f"\n  Есть документы: {doc_types}"
                debug_info += f"\n  IDs документов: {payment['documents_ids']}"
                print(debug_info)
                
                missing = [req for req in required if req not in doc_types and req != 'накладная/упд' or (req == 'накладная/упд' and not any(t in doc_types for t in ['накладная', 'упд']))]
                
                text += f"\n💰 {payment['amount']} руб. ({payment['project']}) — {payment['counterparty']}\n"
                text += f"   Дата: {payment['date']}, Направление: {'входящий' if payment['direction'] == 'in' else 'исходящий'}\n"
                text += f"   ID: {payment['id']}\n"
                text += f"   Не хватает: {', '.join(missing) if missing else 'все документы есть'}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр всех платежей
    if re.search(r"(покажи все платежи|все платежи|список платежей)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра всех платежей: {user_text}")
        all_payments = finances.payments  # Получаем все платежи
        if not all_payments:
            await update.message.reply_text("Платежей пока нет.")
        else:
            text = "Все платежи:\n"
            for payment in all_payments:
                direction_text = 'входящий' if payment['direction'] == 'in' else 'исходящий'
                country_text = 'Россия' if payment['country'] == 'RU' else 'за границу'
                text += f"\n {payment['amount']} руб. ({payment['project']}) — {payment['counterparty']}\n"
                text += f"   Дата: {payment['date']}, {direction_text}, {country_text}\n"
                text += f"   ID: {payment['id']}\n"
                text += f"   Назначение: {payment['purpose']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Удаление платежа
    if re.search(r"(удали платёж|удалить платёж|удалить платеж|удали платеж).*([a-f0-9-]{36})", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду удаления платежа: {user_text}")
        payment_id_match = re.search(r"([a-f0-9-]{36})", user_text)
        if payment_id_match:
            payment_id = payment_id_match.group(1)
            payment = finances.find_payment_by_id(payment_id)
            if payment:
                # Показываем информацию о платеже перед удалением
                text = f"Удаляю платёж:\n"
                text += f"💰 {payment['amount']} руб. ({payment['project']}) — {payment['counterparty']}\n"
                text += f"Дата: {payment['date']}\n"
                text += f"Направление: {'входящий' if payment['direction'] == 'in' else 'исходящий'}\n"
                text += f"Назначение: {payment['purpose']}\n"
                
                if payment['documents_ids']:
                    text += f"\nСвязанные документы (будут удалены):\n"
                    for doc_id in payment['documents_ids']:
                        doc = finances.find_document_by_id(doc_id)
                        if doc:
                            text += f"  📄 {doc['type']} №{doc['number']} от {doc['date']}\n"
                
                # Удаляем платёж
                if finances.delete_payment(payment_id):
                    text += f"\n✅ Платёж успешно удалён вместе со всеми связанными документами."
                    await update.message.reply_text(text)
                else:
                    await update.message.reply_text("Ошибка при удалении платежа.")
            else:
                await update.message.reply_text(f"Платёж с ID {payment_id} не найден.")
        else:
            await update.message.reply_text("Формат: 'Удали платёж [ID_платежа]'")
        return
    
    # Просмотр всех закупок
    if re.search(r"(покажи все закупки|все закупки|список закупок)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра всех закупок: {user_text}")
        all_purchases = finances.purchases  # Получаем все закупки
        if not all_purchases:
            await update.message.reply_text("Закупок пока нет.")
        else:
            text = "Все закупки:\n"
            for purchase in all_purchases:
                text += f"\n📦 {purchase['name']} — {purchase['amount']} руб.\n"
                text += f"   Дата: {purchase['date']}\n"
                text += f"   ID: {purchase['id']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр всех документов
    if re.search(r"(покажи все документы|все документы|список документов)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра всех документов: {user_text}")
        all_documents = finances.documents  # Получаем все документы
        if not all_documents:
            await update.message.reply_text("Документов пока нет.")
        else:
            text = "Все документы:\n"
            for doc in all_documents:
                text += f"\n📄 {doc['type']} №{doc['number']} от {doc['date']}\n"
                text += f"   ID: {doc['id']}\n"
                if doc['payment_ids']:
                    payment = finances.find_payment_by_id(doc['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if doc.get('file_url'):
                    text += f"   📎 Файл: {doc['file_url']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр контрактов
    if re.search(r"(покажи контракты|контракты|список контрактов|все контракты)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра контрактов: {user_text}")
        contracts = [doc for doc in finances.documents if doc['type'] == 'контракт']
        if not contracts:
            await update.message.reply_text("Контрактов пока нет.")
        else:
            text = f"Контракты ({len(contracts)}):\n"
            for contract in contracts:
                text += f"\n📋 Контракт №{contract['number']} от {contract['date']}\n"
                text += f"   ID: {contract['id']}\n"
                if contract['payment_ids']:
                    payment = finances.find_payment_by_id(contract['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                        text += f"   Проект: {payment['project']}\n"
                        text += f"   Направление: {'входящий' if payment['direction'] == 'in' else 'исходящий'}\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if contract.get('file_url'):
                    text += f"   📎 Файл: {contract['file_url']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр накладных и УПД
    if re.search(r"(покажи накладные|накладные|упд|покажи упд|список накладных|список упд)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра накладных/УПД: {user_text}")
        invoices = [doc for doc in finances.documents if doc['type'] in ['накладная', 'упд']]
        if not invoices:
            await update.message.reply_text("Накладных и УПД пока нет.")
        else:
            text = f"Накладные и УПД ({len(invoices)}):\n"
            for invoice in invoices:
                text += f"\n📄 {invoice['type'].title()} №{invoice['number']} от {invoice['date']}\n"
                text += f"   ID: {invoice['id']}\n"
                if invoice['payment_ids']:
                    payment = finances.find_payment_by_id(invoice['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                        text += f"   Проект: {payment['project']}\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if invoice.get('file_url'):
                    text += f"   📎 Файл: {invoice['file_url']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр ГТД
    if re.search(r"(покажи гтд|гтд|список гтд|все гтд)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра ГТД: {user_text}")
        gtd_docs = [doc for doc in finances.documents if doc['type'] == 'гтд']
        if not gtd_docs:
            await update.message.reply_text("ГТД пока нет.")
        else:
            text = f"ГТД ({len(gtd_docs)}):\n"
            for gtd in gtd_docs:
                text += f"\n🌍 ГТД №{gtd['number']} от {gtd['date']}\n"
                text += f"   ID: {gtd['id']}\n"
                if gtd['payment_ids']:
                    payment = finances.find_payment_by_id(gtd['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                        text += f"   Проект: {payment['project']}\n"
                        text += f"   Страна: {'за границу' if payment['country'] == 'INT' else 'Россия'}\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if gtd.get('file_url'):
                    text += f"   📎 Файл: {gtd['file_url']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр счетов
    if re.search(r"(покажи счета|счета|список счетов|все счета)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра счетов: {user_text}")
        bills = [doc for doc in finances.documents if doc['type'] == 'счёт']
        if not bills:
            await update.message.reply_text("Счетов пока нет.")
        else:
            text = f"Счета ({len(bills)}):\n"
            for bill in bills:
                text += f"\n💰 Счёт №{bill['number']} от {bill['date']}\n"
                text += f"   ID: {bill['id']}\n"
                if bill['payment_ids']:
                    payment = finances.find_payment_by_id(bill['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                        text += f"   Проект: {payment['project']}\n"
                        text += f"   Направление: {'входящий' if payment['direction'] == 'in' else 'исходящий'}\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if bill.get('file_url'):
                    text += f"   📎 Файл: {bill['file_url']}\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр актов
    if re.search(r"(покажи акты|акты|список актов|все акты)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра актов: {user_text}")
        acts = [doc for doc in finances.documents if doc['type'] == 'акт']
        if not acts:
            await update.message.reply_text("Актов пока нет.")
        else:
            text = f"Акты ({len(acts)}):\n"
            for act in acts:
                text += f"\n📋 Акт №{act['number']} от {act['date']}\n"
                text += f"   ID: {act['id']}\n"
                text += f"   Тип: {act.get('entity_type', 'Не указан')}\n"
                text += f"   Создана: {datetime.fromtimestamp(act.get('created_at', 0)).strftime('%d.%m.%Y')}\n\n"
            
            await update.message.reply_text(text)
        return
    
    # Просмотр приходов за неделю
    if re.search(r"(покажи приходы за эту неделю|приходы за неделю|доходы за неделю|покажи доходы за неделю)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра приходов за неделю: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        week_data = finances.get_income_for_week(project=project)
        
        if not week_data['income_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Приходов за эту неделю{project_text} нет.")
        else:
            text = f"💰 Приходы за неделю ({week_data['week_start']} - {week_data['week_end']})"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for income in week_data['income_list']:
                text += f"📈 {income['amount']} руб. ({income['project']})\n"
                text += f"   {income['description']} — {income['date']}\n\n"
            
            text += f"💵 <b>Итого: {week_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр расходов за неделю
    if re.search(r"(покажи расходы за эту неделю|расходы за неделю|покажи траты за неделю|траты за неделю)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра расходов за неделю: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        week_data = finances.get_expense_for_week(project=project)
        
        if not week_data['expense_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Расходов за эту неделю{project_text} нет.")
        else:
            text = f"💸 Расходы за неделю ({week_data['week_start']} - {week_data['week_end']})"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for expense in week_data['expense_list']:
                category_text = f" [{expense.get('category', 'без категории')}]" if expense.get('category') else ""
                text += f"📉 {expense['amount']} руб. ({expense['project']}){category_text}\n"
                text += f"   {expense['description']} — {expense['date']}\n\n"
            
            text += f"💸 <b>Итого: {week_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр приходов за месяц
    if re.search(r"(покажи приходы за месяц|приходы за месяц|доходы за месяц|покажи доходы за месяц)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра приходов за месяц: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        month_data = finances.get_income_for_month(project=project)
        
        if not month_data['income_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Приходов за {month_data['period']}{project_text} нет.")
        else:
            text = f"💰 Приходы за {month_data['period']}"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for income in month_data['income_list']:
                text += f"📈 {income['amount']} руб. ({income['project']})\n"
                text += f"   {income['description']} — {income['date']}\n\n"
            
            text += f"💵 <b>Итого: {month_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр расходов за месяц
    if re.search(r"(покажи расходы за месяц|расходы за месяц|покажи траты за месяц|траты за месяц)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра расходов за месяц: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        month_data = finances.get_expense_for_month(project=project)
        
        if not month_data['expense_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Расходов за {month_data['period']}{project_text} нет.")
        else:
            text = f"💸 Расходы за {month_data['period']}"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for expense in month_data['expense_list']:
                category_text = f" [{expense.get('category', 'без категории')}]" if expense.get('category') else ""
                text += f"📉 {expense['amount']} руб. ({expense['project']}){category_text}\n"
                text += f"   {expense['description']} — {expense['date']}\n\n"
            
            text += f"💸 <b>Итого: {month_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр приходов за квартал
    if re.search(r"(покажи приходы за квартал|приходы за квартал|доходы за квартал|покажи доходы за квартал)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра приходов за квартал: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        quarter_data = finances.get_income_for_quarter(project=project)
        
        if not quarter_data['income_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Приходов за {quarter_data['period']}{project_text} нет.")
        else:
            text = f"💰 Приходы за {quarter_data['period']}"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for income in quarter_data['income_list']:
                text += f"📈 {income['amount']} руб. ({income['project']})\n"
                text += f"   {income['description']} — {income['date']}\n\n"
            
            text += f"💵 <b>Итого: {quarter_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр расходов за квартал
    if re.search(r"(покажи расходы за квартал|расходы за квартал|покажи траты за квартал|траты за квартал)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра расходов за квартал: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        quarter_data = finances.get_expense_for_quarter(project=project)
        
        if not quarter_data['expense_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Расходов за {quarter_data['period']}{project_text} нет.")
        else:
            text = f"💸 Расходы за {quarter_data['period']}"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for expense in quarter_data['expense_list']:
                category_text = f" [{expense.get('category', 'без категории')}]" if expense.get('category') else ""
                text += f"📉 {expense['amount']} руб. ({expense['project']}){category_text}\n"
                text += f"   {expense['description']} — {expense['date']}\n\n"
            
            text += f"💸 <b>Итого: {quarter_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр приходов за год
    if re.search(r"(покажи приходы за год|приходы за год|доходы за год|покажи доходы за год)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра приходов за год: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        year_data = finances.get_income_for_year(project=project)
        
        if not year_data['income_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Приходов за {year_data['period']} год{project_text} нет.")
        else:
            text = f"💰 Приходы за {year_data['period']} год"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for income in year_data['income_list']:
                text += f"📈 {income['amount']} руб. ({income['project']})\n"
                text += f"   {income['description']} — {income['date']}\n\n"
            
            text += f"💵 <b>Итого: {year_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Просмотр расходов за год
    if re.search(r"(покажи расходы за год|расходы за год|покажи траты за год|траты за год)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду просмотра расходов за год: {user_text}")
        
        # Проверяем, есть ли указание проекта
        project_match = re.search(r"проект[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        project = project_match.group(1).strip() if project_match else None
        
        year_data = finances.get_expense_for_year(project=project)
        
        if not year_data['expense_list']:
            project_text = f" по проекту '{project}'" if project else ""
            await update.message.reply_text(f"Расходов за {year_data['period']} год{project_text} нет.")
        else:
            text = f"💸 Расходы за {year_data['period']} год"
            if project:
                text += f" по проекту '{project}'"
            text += f":\n\n"
            
            for expense in year_data['expense_list']:
                category_text = f" [{expense.get('category', 'без категории')}]" if expense.get('category') else ""
                text += f"📉 {expense['amount']} руб. ({expense['project']}){category_text}\n"
                text += f"   {expense['description']} — {expense['date']}\n\n"
            
            text += f"💸 <b>Итого: {year_data['total_amount']} руб.</b>"
            
            await update.message.reply_text(text, parse_mode='HTML')
        return
    
    # Поиск документов по контрагенту
    if re.search(r"(найди документы.*контрагент|поиск.*контрагент|документы.*контрагент)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду поиска документов по контрагенту: {user_text}")
        
        # Извлекаем название контрагента
        counterparty_match = re.search(r"контрагент[а]?\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        if counterparty_match:
            counterparty_name = counterparty_match.group(1).strip()
            found_docs = finances.search_documents_by_counterparty(counterparty_name)
            
            if not found_docs:
                await update.message.reply_text(f"Документы контрагента '{counterparty_name}' не найдены.")
            else:
                text = f"📄 Документы контрагента '{counterparty_name}' ({len(found_docs)}):\n\n"
                for doc in found_docs:
                    text += f"📋 {doc['type'].title()} №{doc['number']} от {doc['date']}\n"
                    text += f"   ID: {doc['id']}\n"
                    if doc.get('amount'):
                        text += f"   Сумма: {doc['amount']} руб.\n"
                    if doc.get('description'):
                        text += f"   Описание: {doc['description']}\n"
                    if doc.get('file_url'):
                        text += f"   📎 Файл: {doc['file_url']}\n"
                    text += "\n"
                
                await update.message.reply_text(text)
        else:
            await update.message.reply_text("Формат: 'Найди документы контрагента [название]'")
        return
    
    # Поиск документов по ключевым словам
    if re.search(r"(найди документы.*про|поиск.*документы.*про|документы.*про)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду поиска документов по ключевым словам: {user_text}")
        
        # Извлекаем ключевые слова
        keywords_match = re.search(r"про\s+([а-яёa-z0-9\s]+)", user_text, re.I)
        if keywords_match:
            keywords = keywords_match.group(1).strip().split()
            found_docs = finances.search_documents_by_keywords(keywords)
            
            if not found_docs:
                await update.message.reply_text(f"Документы по ключевым словам '{', '.join(keywords)}' не найдены.")
            else:
                text = f"🔍 Документы по ключевым словам '{', '.join(keywords)}' ({len(found_docs)}):\n\n"
                for doc in found_docs:
                    text += f"📋 {doc['type'].title()} №{doc['number']} от {doc['date']}\n"
                    text += f"   ID: {doc['id']}\n"
                    if doc.get('counterparty_name'):
                        text += f"   Контрагент: {doc['counterparty_name']}\n"
                    if doc.get('amount'):
                        text += f"   Сумма: {doc['amount']} руб.\n"
                    if doc.get('description'):
                        text += f"   Описание: {doc['description']}\n"
                    text += "\n"
                
                await update.message.reply_text(text)
        else:
            await update.message.reply_text("Формат: 'Найди документы про [ключевые слова]'")
        return
    
    # Поиск документов по сумме
    if re.search(r"(найди документы.*сумма|документы.*сумма|поиск.*сумма)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду поиска документов по сумме: {user_text}")
        
        # Извлекаем сумму
        amount_match = re.search(r"сумма[а]?\s+(?:от\s+)?(\d+)(?:\s+до\s+(\d+))?", user_text, re.I)
        if amount_match:
            min_amount = int(amount_match.group(1))
            max_amount = int(amount_match.group(2)) if amount_match.group(2) else None
            
            found_docs = finances.search_documents_by_amount(min_amount, max_amount)
            
            if not found_docs:
                range_text = f"от {min_amount}" + (f" до {max_amount}" if max_amount else "")
                await update.message.reply_text(f"Документы со суммой {range_text} руб. не найдены.")
            else:
                range_text = f"от {min_amount}" + (f" до {max_amount}" if max_amount else "")
                text = f"💰 Документы со суммой {range_text} руб. ({len(found_docs)}):\n\n"
                for doc in found_docs:
                    text += f"📋 {doc['type'].title()} №{doc['number']} от {doc['date']}\n"
                    text += f"   ID: {doc['id']}\n"
                    text += f"   Сумма: {doc['amount']} руб.\n"
                    if doc.get('counterparty_name'):
                        text += f"   Контрагент: {doc['counterparty_name']}\n"
                    text += "\n"
                
                await update.message.reply_text(text)
        else:
            await update.message.reply_text("Формат: 'Найди документы сумма от 100000 до 500000' или 'Найди документы сумма от 100000'")
        return
    
    # Сводка по документам
    if re.search(r"(сводка.*документы|статистика.*документы|документы.*статистика)", user_text, re.I):
        print(f"[DEBUG] Обрабатываю команду сводки по документам: {user_text}")
        
        summary = finances.get_documents_summary()
        
        text = f"📊 Сводка по документам:\n\n"
        text += f"📄 Всего документов: {summary['total_documents']}\n"
        text += f"📎 С файлами: {summary['with_files']}\n"
        text += f"📄 Без файлов: {summary['without_files']}\n"
        text += f"💰 Общая сумма: {summary['total_amount']} руб.\n\n"
        
        text += f"📋 По типам:\n"
        for doc_type, count in summary['by_type'].items():
            text += f"   {doc_type.title()}: {count}\n"
        
        await update.message.reply_text(text)
        return
    
    # RAG поиск документов
    if re.search(r"(найди документ|поиск документов|семантический поиск|rag поиск)", user_text, re.I):
        await handle_rag_search(update, context)
        return
    
    # Поиск по типу документа
    if re.search(r"(найди по типу|поиск по типу|документы типа)", user_text, re.I):
        await handle_search_by_type(update, context)
        return
    
    # Поиск по контрагенту
    if re.search(r"(найди по контрагенту|поиск по контрагенту|документы контрагента)", user_text, re.I):
        await handle_search_by_counterparty(update, context)
        return
    
    # Статистика RAG системы
    if re.search(r"(статистика rag|rag статистика|статус rag)", user_text, re.I):
        await handle_rag_stats(update, context)
        return
    
    # Команды для целей и KPI
    if re.search(r"(создать цель|новая цель|добавить цель)", user_text, re.I):
        await handle_create_goal(update, context)
        return
    
    if re.search(r"(прогресс по цели|статус цели|как дела с целью)", user_text, re.I):
        await handle_goal_progress(update, context)
        return
    
    if re.search(r"(покажи прогресс по целям|прогресс по целям|покажи okr|покажи kpi)", user_text, re.I):
        await handle_goals_progress_summary(update, context)
        return
    
    if re.search(r"(обновить прогресс|обновить цель|прогресс)", user_text, re.I):
        await handle_update_goal_progress(update, context)
        return
    
    if re.search(r"(все цели|список целей|мои цели)", user_text, re.I):
        await handle_list_goals(update, context)
        return
    
    # Обработка действий с распознанными документами
    if context.user_data.get('processed_document'):
        await handle_document_action(update, context)
        return
    
    # Email команды
    if re.search(r"(сводка входящих|входящие|email сводка|почта сводка)", user_text, re.I):
        await handle_email_summary(update, context)
        return
    
    if re.search(r"(срочные сообщения|ожидают ответа|важные сообщения|email линзы)", user_text, re.I):
        await handle_email_lens(update, context)
        return
    
    if re.search(r"(шаблон ответа|ответить на|reply template)", user_text, re.I):
        await handle_reply_template(update, context)
        return
    
    if re.search(r"(настройка email|email настройка|конфиг email)", user_text, re.I):
        await handle_email_config(update, context)
        return
    
    # Контекст и память: что решили с ...
    if re.search(r"что решили с (.+)", user_text, re.I):
        await handle_what_decided(update, context)
        return
    # Контекст и память: с кем обсуждали ...
    if re.search(r"с кем обсуждали (.+)", user_text, re.I):
        await handle_who_discussed(update, context)
        return

    # Партнёрская сеть
    if re.search(r"(партнёры сводка|сводка партнёров|partners summary)", user_text, re.I):
        await handle_partners_summary(update, context)
        return
    
    if re.search(r"добавь партнёра", user_text, re.I):
        await handle_add_partner(update, context)
        return
    
    if re.search(r"(партнёры для прозвона|прозвон|звонить)", user_text, re.I):
        await handle_partners_for_calling(update, context)
        return
    
    if re.search(r"(партнёры для рассылки|email рассылка|рассылка)", user_text, re.I):
        await handle_partners_for_emailing(update, context)
        return
    
    if re.search(r"(предложение для|proposal для)", user_text, re.I):
        await handle_generate_proposal(update, context)
        return
    
    if re.search(r"(массовые предложения|bulk proposals|предложения для группы)", user_text, re.I):
        await handle_bulk_proposals(update, context)
        return

    # AmoCRM команды
    if re.search(r"(контакты amocrm|amocrm контакты|контакты в crm)", user_text, re.I):
        await handle_amocrm_contacts(update, context)
        return
    
    if re.search(r"(сделки amocrm|amocrm сделки|лиды в crm)", user_text, re.I):
        await handle_amocrm_leads(update, context)
        return
    
    if re.search(r"(аналитика amocrm|amocrm аналитика|crm аналитика)", user_text, re.I):
        await handle_amocrm_analytics(update, context)
        return
    
    if re.search(r"(синхронизация amocrm|amocrm синхронизация|синхронизировать партнёров)", user_text, re.I):
        await handle_amocrm_sync_partners(update, context)
        return
    
    if re.search(r"(создай контакт|добавь контакт)", user_text, re.I):
        await handle_amocrm_create_contact(update, context)
        return
    
    if re.search(r"(создай сделку|добавь сделку)", user_text, re.I):
        await handle_amocrm_create_lead(update, context)
        return
    
    if re.search(r"(воронки amocrm|amocrm воронки|воронки продаж)", user_text, re.I):
        await handle_amocrm_pipelines(update, context)
        return
    
    if re.search(r"(задачи amocrm|amocrm задачи|задачи в crm)", user_text, re.I):
        await handle_amocrm_tasks(update, context)
        return

    # Obsidian команды
    if re.search(r"(создай стратегию|новая стратегия|добавить стратегию)", user_text, re.I):
        await handle_create_strategy(update, context)
        return
    
    if re.search(r"(создай решение|новое решение|добавить решение)", user_text, re.I):
        await handle_create_decision(update, context)
        return
    
    if re.search(r"(найди заметки|поиск заметок|поиск в obsidian|obsidian поиск)", user_text, re.I):
        await handle_obsidian_search(update, context)
        return
    
    if re.search(r"(статистика obsidian|obsidian статистика|статистика заметок)", user_text, re.I):
        await handle_obsidian_stats(update, context)
        return

    # Команды для командной работы
    if re.search(r"(добавить сотрудника|добавь сотрудника|новый сотрудник)", user_text, re.I):
        await handle_add_employee(update, context)
        return
    
    if re.search(r"(назначить задачу|назначи задачу|поручи задачу)", user_text, re.I):
        await handle_assign_task(update, context)
        return
    
    if re.search(r"(мои задачи|покажи мои задачи|задачи для меня)", user_text, re.I):
        await handle_my_tasks(update, context)
        return
    
    if re.search(r"(завершить задачу|заверши задачу|задача выполнена)", user_text, re.I):
        await handle_complete_task(update, context)
        return
    
    if re.search(r"(отчёт:|ежедневный отчёт|дневной отчёт)", user_text, re.I):
        await handle_daily_report(update, context)
        return
    
    if re.search(r"(статус команды|команда статус|покажи команду)", user_text, re.I):
        await handle_team_status(update, context)
        return
    
    if re.search(r"(включить режим отпуска|режим отпуска|владелец в отпуске)", user_text, re.I):
        await handle_vacation_mode(update, context)
        return
    
    if re.search(r"(просроченные задачи|задачи просрочены|overdue tasks)", user_text, re.I):
        await handle_overdue_tasks(update, context)
        return

    # Команды для контроля платежей и документов
    if re.search(r"(контроль платежей|отчёт по контролю|деньги без документов|платежи без документов)", user_text, re.I):
        await handle_payment_control_report(update, context)
        return
    
    if re.search(r"(еженедельная сводка контроля|сводка контроля|контроль сводка)", user_text, re.I):
        await handle_weekly_control_summary(update, context)
        return
    
    if re.search(r"(критические уведомления|критические случаи|критично)", user_text, re.I):
        await handle_critical_alerts(update, context)
        return
    
    if re.search(r"(незакрытые платежи|платежи без документов|нет документов)", user_text, re.I):
        await handle_unclosed_payments_report(update, context)
        return
    
    if re.search(r"(документы без оплаты|документы без платежей|нет поступления)", user_text, re.I):
        await handle_orphaned_documents_report(update, context)
        return

    # Команды для контроля входящих сообщений
    if re.search(r"(сводка входящих|входящие сводка|контроль входящих|inbox summary)", user_text, re.I):
        await handle_inbox_summary(update, context)
        return
    
    if re.search(r"(требуют внимания|внимание|attention messages)", user_text, re.I):
        await handle_attention_messages(update, context)
        return
    
    if re.search(r"(просроченные ответы|нет ответа|overdue responses)", user_text, re.I):
        await handle_overdue_responses(update, context)
        return
    
    if re.search(r"(забытые сообщения|забыто|forgotten messages)", user_text, re.I):
        await handle_forgotten_messages(update, context)
        return
    
    if re.search(r"(предложения напоминаний|напоминания|reminder suggestions)", user_text, re.I):
        await handle_reminder_suggestions(update, context)
        return
    
    if re.search(r"отметить отвеченным ([a-zA-Z0-9_-]+)", user_text, re.I):
        await handle_mark_responded(update, context)
        return
    
    if re.search(r"отметить проигнорированным ([a-zA-Z0-9_-]+)", user_text, re.I):
        await handle_mark_ignored(update, context)
        return
    
    if re.search(r"добавить сообщение", user_text, re.I):
        await handle_add_message(update, context)
        return

    # --- Режим "Антиразрыв" ---
    if re.search(r"(устал|антиразрыв|стресс|вс[её] горит|сил нет)", user_text, re.I):
        await handle_antistress_mode(update, context)
        return

    # --- Подготовка к встрече ---
    if re.search(r"(подготовь инфу|подготовься к встрече|инфа для встречи)", user_text, re.I):
        await handle_meeting_prep(update, context)
        return

    # Если не задача и не финансы — fallback на GPT-ответ
    reply = await ask_openai(user_text)
    await update.message.reply_text(reply)

    # Глобальный поиск по всей памяти
    if re.search(r"(где обсуждали|найди файл|найди |поиск )", user_text, re.I):
        await handle_global_search(update, context)
        return

    # Прогноз cash flow
    if re.search(r"(прогноз cash ?flow|на сколько хватит денег|денег хватит на)", user_text, re.I):
        await handle_cashflow_forecast(update, context)
        return

    # Мультидашборд /дайджест и "что у нас"
    if re.search(r"(/дайджест|что у нас\??)", user_text, re.I):
        await handle_digest(update, context)
        return

    # --- Управление состоянием тихого режима ---
    STATE_FILE = 'bot_state.json'

    def get_bot_state():
        """Загружает состояние бота из файла."""
        if not os.path.exists(STATE_FILE):
            return {}
        with open(STATE_FILE, 'r') as f:
            return json.load(f)

    def save_bot_state(state):
        """Сохраняет состояние бота в файл."""
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    def set_quiet_mode(chat_id: int, enabled: bool):
        """Включает или выключает тихий режим для чата."""
        state = get_bot_state()
        if 'quiet_mode' not in state:
            state['quiet_mode'] = {}
        state['quiet_mode'][str(chat_id)] = enabled
        save_bot_state(state)

    def is_quiet_mode_enabled(chat_id: int) -> bool:
        """Проверяет, включен ли тихий режим для чата."""
        state = get_bot_state()
        return state.get('quiet_mode', {}).get(str(chat_id), False)

    async def handle_document_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка действий с распознанным документом."""
        user_text = update.message.text.lower()
        
        if 'processed_document' not in context.user_data:
            await update.message.reply_text("❌ Нет обработанного документа. Отправьте фотографию документа.")
            return
        
        doc_data = context.user_data['processed_document']
        
        try:
            if 'сохранить pdf' in user_text:
                # Создаем PDF из изображения с исправленной ориентацией
                pdf_path = f"/tmp/doc_{doc_data['timestamp']}.pdf"
                success = image_processor.create_pdf_from_image(doc_data['image_path'], pdf_path)
                
                if success:
                    # Загружаем в Google Drive
                    drive_result = drive_manager.upload_file(pdf_path, f"doc_{doc_data['timestamp']}.pdf")
                    
                    if drive_result and 'id' in drive_result:
                        await update.message.reply_text(
                            f"✅ PDF создан с исправленной ориентацией и загружен в Google Drive\n"
                            f"📁 ID файла: {drive_result['id']}\n"
                            f"📄 Имя: {drive_result.get('name', 'Не указано')}"
                        )
                    else:
                        await update.message.reply_text("❌ Ошибка загрузки в Google Drive")
                else:
                    await update.message.reply_text("❌ Ошибка создания PDF")
            
            elif 'сохранить текст' in user_text:
                # Создаем PDF из распознанного текста
                pdf_path = f"/tmp/doc_text_{doc_data['timestamp']}.pdf"
                title = f"{doc_data['doc_info']['type'].title()} {doc_data['doc_info']['number'] or ''}"
                success = image_processor.create_pdf_from_text(doc_data['text'], pdf_path, title)
                
                if success:
                    # Загружаем в Google Drive
                    drive_result = drive_manager.upload_file(pdf_path, f"doc_text_{doc_data['timestamp']}.pdf")
                    
                    if drive_result and 'id' in drive_result:
                        await update.message.reply_text(
                            f"✅ PDF из текста создан и загружен в Google Drive\n"
                            f"📁 ID файла: {drive_result['id']}\n"
                            f"📄 Имя: {drive_result.get('name', 'Не указано')}"
                        )
                    else:
                        await update.message.reply_text("❌ Ошибка загрузки в Google Drive")
                else:
                    await update.message.reply_text("❌ Ошибка создания PDF из текста")
            
            elif 'добавить в базу' in user_text:
                # Добавляем документ в систему
                doc_info = doc_data['doc_info']
                
                if doc_info['type'] and doc_info['type'] != 'неизвестно':
                    # Создаем PDF для загрузки
                    pdf_path = f"/tmp/doc_final_{doc_data['timestamp']}.pdf"
                    image_processor.images_to_pdf([doc_data['image_path']], pdf_path)
                    
                    # Добавляем документ
                    doc_id = add_document(
                        doc_type=doc_info['type'],
                        counterparty_name=doc_info['counterparty'] or 'Не указан',
                        amount=doc_info['amount'] or 0,
                        date=doc_info['date'] or datetime.now().strftime('%Y-%m-%d'),
                        description=f"Документ распознан из фотографии. Уверенность: {doc_info['confidence']}%",
                        file_path=pdf_path
                    )
                    
                    if doc_id:
                        await update.message.reply_text(
                            f"✅ Документ добавлен в систему!\n"
                            f"📄 ID: {doc_id}\n"
                            f"📋 Тип: {doc_info['type'].title()}\n"
                            f"🎯 Уверенность: {doc_info['confidence']}%"
                        )
                    else:
                        await update.message.reply_text("❌ Ошибка добавления документа в систему")
                else:
                    await update.message.reply_text("❌ Не удалось определить тип документа для добавления в базу")
            
            elif 'отмена' in user_text:
                # Очищаем данные
                del context.user_data['processed_document']
                await update.message.reply_text("❌ Обработка отменена")
            
            else:
                await update.message.reply_text(
                    "🔧 Доступные действия:\n"
                    "• 'Сохранить PDF' - создать PDF из изображения\n"
                    "• 'Сохранить текст' - создать PDF из распознанного текста\n"
                    "• 'Добавить в базу' - добавить документ в систему\n"
                    "• 'Отмена' - отменить обработку"
                )
        
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка выполнения действия: {e}")

    # --- Функции для целей и KPI ---
    async def handle_create_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка создания новой цели."""
        user_text = update.message.text
        
        # Парсим команду создания цели
        # Пример: "Создать цель выручка 3 млн до сентября"
        goal_match = re.search(r"(?:создать цель|новая цель|добавить цель)\s+(.+?)\s+(\d+(?:\.\d+)?)\s*(млн|тыс|руб|%|шт|клиентов?)?(?:\s+до\s+(.+))?", user_text, re.I)
        
        if not goal_match:
            await update.message.reply_text(
                "🎯 <b>Создание новой цели</b>\n\n"
                "Используйте формат:\n"
                "• 'Создать цель [название] [значение] [единица] до [дата]'\n\n"
                "Примеры:\n"
                "• Создать цель выручка 3 млн до сентября\n"
                "• Новая цель подписки 100 клиентов до декабря\n"
                "• Добавить цель производство 1000 шт до конца месяца",
                parse_mode='HTML'
            )
            return
        
        goal_name = goal_match.group(1).strip()
        target_value = float(goal_match.group(2))
        unit = goal_match.group(3) or ""
        end_date_str = goal_match.group(4)
        
        # Определяем тип цели
        goal_type = GoalType.CUSTOM
        if any(word in goal_name.lower() for word in ['выручка', 'доход', 'прибыль', 'млн', 'тыс']):
            goal_type = GoalType.REVENUE
        elif any(word in goal_name.lower() for word in ['подписк', 'клиент', 'пользователь']):
            goal_type = GoalType.SUBSCRIPTIONS
        elif any(word in goal_name.lower() for word in ['производство', 'продукт', 'шт', 'единиц']):
            goal_type = GoalType.PRODUCTION
        
        # Парсим дату
        end_date = None
        if end_date_str:
            end_date = parse_natural_date(end_date_str)
            if end_date:
                end_date = end_date.strftime('%Y-%m-%d')
        
        try:
            # Создаем цель
            goal_id = goals_manager.create_goal(
                name=goal_name,
                description=f"Цель: {goal_name} {target_value}{unit}",
                goal_type=goal_type,
                target_value=target_value,
                end_date=end_date
            )
            
            await update.message.reply_text(
                f"✅ <b>Цель создана!</b>\n\n"
                f"🎯 Название: {goal_name}\n"
                f"📊 Целевое значение: {target_value}{unit}\n"
                f"📅 Срок: {end_date or 'Не указан'}\n"
                f"🆔 ID: {goal_id}\n\n"
                f"Используйте:\n"
                f"• 'Прогресс по цели {goal_name}' - проверить статус\n"
                f"• 'Обновить прогресс {goal_name} [значение]' - обновить прогресс",
                parse_mode='HTML'
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка создания цели: {e}")

    async def handle_goal_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка запроса прогресса по цели."""
        user_text = update.message.text
        
        # Парсим запрос прогресса
        # Пример: "Какой прогресс по цели 3 млн выручки до сентября?"
        progress_match = re.search(r"(?:прогресс по цели|статус цели|как дела с целью)\s+(.+?)(?:\s+до\s+(.+))?", user_text, re.I)
        
        if not progress_match:
            await update.message.reply_text(
                "📊 <b>Проверка прогресса по цели</b>\n\n"
                "Используйте:\n"
                "• 'Прогресс по цели [название]' - проверить статус\n"
                "• 'Статус цели [название]' - получить детали\n\n"
                "Примеры:\n"
                "• Прогресс по цели выручка 3 млн\n"
                "• Статус цели подписки 100 клиентов",
                parse_mode='HTML'
            )
            return
        
        goal_query = progress_match.group(1).strip()
        
        try:
            # Ищем цель
            goals = goals_manager.search_goals(goal_query)
            
            if not goals:
                await update.message.reply_text(f"🔍 Цель '{goal_query}' не найдена.")
                return
            
            # Берем первую найденную цель
            goal = goals[0]
            progress_data = goals_manager.get_goal_progress(goal.id)
            
            if not progress_data:
                await update.message.reply_text(f"❌ Ошибка получения прогресса для цели '{goal.name}'")
                return
            
            # Формируем отчет
            report = f"📊 <b>Прогресс по цели: {goal.name}</b>\n\n"
            report += f"🎯 Целевое значение: {goal.target_value}\n"
            report += f"📈 Текущее значение: {goal.current_value}\n"
            report += f"📊 Прогресс: {progress_data['progress_percentage']}%\n"
            report += f"📉 Осталось: {progress_data['remaining']}\n\n"
            
            # Тренд
            trend = progress_data['trend']
            trend_emoji = "📈" if trend['direction'] == 'increasing' else "📉" if trend['direction'] == 'decreasing' else "➡️"
            report += f"{trend_emoji} <b>Тренд:</b> {trend['direction']} ({trend['rate']}/день)\n"
            
            # Прогноз
            forecast = progress_data['forecast']
            if forecast['achievable']:
                report += f"✅ <b>Прогноз:</b> Цель достижима\n"
                if forecast['estimated_completion']:
                    completion_date = datetime.fromisoformat(forecast['estimated_completion']).strftime('%d.%m.%Y')
                    report += f"📅 Ожидаемое завершение: {completion_date}\n"
            else:
                report += f"⚠️ <b>Прогноз:</b> Цель под угрозой\n"
                report += f"📊 Требуемая скорость: {forecast['required_rate']}/день\n"
            
            # Статус
            status_emoji = "🟢" if progress_data['is_on_track'] else "🔴"
            report += f"\n{status_emoji} <b>Статус:</b> {'По плану' if progress_data['is_on_track'] else 'Отставание'}"
            
            await update.message.reply_text(report, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения прогресса: {e}")

    async def handle_update_goal_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка обновления прогресса по цели."""
        user_text = update.message.text
        
        # Парсим обновление прогресса
        # Пример: "Обновить прогресс выручка 2.5 млн"
        update_match = re.search(r"(?:обновить прогресс|обновить цель|прогресс)\s+(.+?)\s+(\d+(?:\.\d+)?)\s*(млн|тыс|руб|%|шт|клиентов?)?", user_text, re.I)
        
        if not update_match:
            await update.message.reply_text(
                "📈 <b>Обновление прогресса по цели</b>\n\n"
                "Используйте формат:\n"
                "• 'Обновить прогресс [название] [новое значение] [единица]'\n\n"
                "Примеры:\n"
                "• Обновить прогресс выручка 2.5 млн\n"
                "• Обновить цель подписки 75 клиентов\n"
                "• Прогресс производство 800 шт",
                parse_mode='HTML'
            )
            return
        
        goal_query = update_match.group(1).strip()
        new_value = float(update_match.group(2))
        unit = update_match.group(3) or ""
        
        try:
            # Ищем цель
            goals = goals_manager.search_goals(goal_query)
            
            if not goals:
                await update.message.reply_text(f"🔍 Цель '{goal_query}' не найдена.")
                return
            
            # Берем первую найденную цель
            goal = goals[0]
            old_value = goal.current_value
            
            # Обновляем прогресс
            success = goals_manager.update_goal_progress(goal.id, new_value, f"Обновлено через Telegram")
            
            if success:
                change = new_value - old_value
                change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                await update.message.reply_text(
                    f"✅ <b>Прогресс обновлен!</b>\n\n"
                    f"🎯 Цель: {goal.name}\n"
                    f"📊 Было: {old_value}\n"
                    f"📈 Стало: {new_value}\n"
                    f"{change_emoji} Изменение: {change:+g}\n\n"
                    f"Используйте 'Прогресс по цели {goal.name}' для проверки статуса",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text("❌ Ошибка обновления прогресса")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка обновления прогресса: {e}")

    async def handle_list_goals(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка запроса списка всех целей."""
        try:
            goals = goals_manager.get_active_goals()
            
            if not goals:
                await update.message.reply_text(
                    "📋 <b>У вас пока нет активных целей</b>\n\n"
                    "Создайте первую цель:\n"
                    "• 'Создать цель выручка 3 млн до сентября'\n"
                    "• 'Новая цель подписки 100 клиентов до декабря'",
                    parse_mode='HTML'
                )
                return
            
            # Группируем цели по типу
            goals_by_type = {}
            for goal in goals:
                goal_type = goal.goal_type.value
                if goal_type not in goals_by_type:
                    goals_by_type[goal_type] = []
                goals_by_type[goal_type].append(goal)
            
            # Формируем отчет
            report = f"📋 <b>Активные цели ({len(goals)}):</b>\n\n"
            
            for goal_type, type_goals in goals_by_type.items():
                type_emoji = {
                    'revenue': '💰',
                    'subscriptions': '👥',
                    'production': '🏭',
                    'custom': '🎯'
                }.get(goal_type, '��')
                
                report += f"{type_emoji} <b>{goal_type.title()}:</b>\n"
                
                for goal in type_goals:
                    progress_data = goals_manager.get_goal_progress(goal.id)
                    progress_percent = progress_data['progress_percentage'] if progress_data else 0
                    
                    status_emoji = "🟢" if progress_data and progress_data['is_on_track'] else "🔴"
                    
                    report += f"  {status_emoji} {goal.name}: {goal.current_value}/{goal.target_value} ({progress_percent}%)\n"
                
                report += "\n"
            
            report += "💡 <b>Команды:</b>\n"
            report += "• 'Прогресс по цели [название]' - проверить статус\n"
            report += "• 'Обновить прогресс [название] [значение]' - обновить\n"
            report += "• 'Создать цель [название] [значение] до [дата]' - новая цель"
            
            await update.message.reply_text(report, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения списка целей: {e}")

    # --- Контекст и память ---
    async def handle_what_decided(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        match = re.search(r"что решили с (.+)", user_text, re.I)
        if not match:
            await update.message.reply_text("Уточните запрос: 'Что решили с [тема/объект]'")
            return
        topic = match.group(1).strip()
        # Поиск по истории чата
        messages = chat_memory.search(topic, limit=10)
        # Поиск по задачам
        from core.planner import get_tasks
        tasks = [t for t in get_tasks() if topic.lower() in t['description'].lower()]
        # Поиск по RAG (документы, протоколы)
        from core.rag_system import rag_system
        rag_results = rag_system.search_documents(topic, n_results=3)
        # Формируем ответ
        reply = f"🧠 <b>Контекст по запросу: {topic}</b>\n\n"
        if messages:
            reply += "💬 <b>Фрагменты переписки:</b>\n"
            for m in messages:
                reply += f"— {m['username']}: {m['text']}\n"
            reply += "\n"
        if tasks:
            reply += "📋 <b>Задачи:</b>\n"
            for t in tasks:
                reply += f"— {t['description']} ({'✅' if t.get('completed') else '⏳'})\n"
            reply += "\n"
        if rag_results:
            reply += "📄 <b>Документы/протоколы:</b>\n"
            for doc in rag_results:
                meta = doc.get('metadata', {})
                reply += f"— {meta.get('type', 'Документ')}: {meta.get('title', '')} (ID: {doc['id']})\n"
            reply += "\n"
        if not (messages or tasks or rag_results):
            reply += "❓ Нет найденных решений или обсуждений по теме."
        await update.message.reply_text(reply, parse_mode='HTML')

    async def handle_who_discussed(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        match = re.search(r"с кем обсуждали (.+)", user_text, re.I)
        if not match:
            await update.message.reply_text("Уточните запрос: 'С кем обсуждали [тема/объект]'")
            return
        topic = match.group(1).strip()
        # Поиск по истории чата
        discussions = chat_memory.get_discussions_with(topic, limit=10)
        reply = f"🧠 <b>Обсуждения по теме: {topic}</b>\n\n"
        if discussions:
            for d in discussions:
                reply += f"👤 <b>{d['username']}</b> участвовал(а):\n"
                for m in d['messages'][-3:]:
                    reply += f"— {m['text']}\n"
                reply += "\n"
        else:
            reply += "❓ Нет найденных обсуждений по теме."
        await update.message.reply_text(reply, parse_mode='HTML')

    # --- Голосовое распознавание ---
    
    # --- Email функции ---
    async def handle_email_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать сводку входящих сообщений."""
        user_text = update.message.text.lower()
        
        # Определяем провайдера
        provider = "gmail"
        if "яндекс" in user_text or "yandex" in user_text:
            provider = "yandex"
        
        await update.message.reply_text(f"📧 Получаю сводку входящих из {provider}...")
        
        try:
            summary = email_analyzer.get_inbox_summary(provider)
            
            if "error" in summary:
                await update.message.reply_text(f"❌ {summary['error']}")
                return
            
            text = f"📧 <b>Сводка входящих ({provider}):</b>\n\n"
            text += f"📊 Всего сообщений: {summary['total_messages']}\n"
            text += f"🚨 Срочных: {summary['urgent_count']}\n"
            text += f"⭐ Важных: {summary['high_priority_count']}\n"
            text += f"💬 Требуют ответа: {summary['need_reply_count']}\n\n"
            
            if summary['urgent_messages']:
                text += "🚨 <b>Срочные сообщения:</b>\n"
                for msg in summary['urgent_messages']:
                    text += f"• {msg['subject']} (от {msg['sender']})\n"
                text += "\n"
            
            if summary['need_reply_messages']:
                text += "💬 <b>Требуют ответа:</b>\n"
                for msg in summary['need_reply_messages']:
                    text += f"• {msg['subject']} (от {msg['sender']})\n"
                text += "\n"
            
            text += f"📝 <b>Анализ:</b>\n{summary['summary']}"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения сводки: {e}")

    async def handle_email_lens(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать сообщения по линзам."""
        user_text = update.message.text.lower()
        
        # Определяем линзу
        lens = None
        if "срочные" in user_text or "urgent" in user_text:
            lens = "urgent"
        elif "ответ" in user_text or "reply" in user_text:
            lens = "need_reply"
        elif "важное" in user_text or "important" in user_text:
            lens = "important"
        else:
            await update.message.reply_text(
                "�� <b>Просмотр по линзам</b>\n\n"
                "Используйте:\n"
                "• 'Срочные сообщения' - срочные и важные\n"
                "• 'Ожидают ответа' - требующие ответа\n"
                "• 'Важные сообщения' - все важные\n\n"
                "Примеры:\n"
                "• Покажи срочные сообщения\n"
                "• Что ожидает ответа\n"
                "• Важные входящие",
                parse_mode='HTML'
            )
            return
        
        # Определяем провайдера
        provider = "gmail"
        if "яндекс" in user_text or "yandex" in user_text:
            provider = "yandex"
        
        await update.message.reply_text(f"🔍 Получаю сообщения по линзе '{lens}' из {provider}...")
        
        try:
            messages = email_analyzer.get_messages_by_lens(lens, provider)
            
            if not messages:
                lens_names = {
                    "urgent": "срочных",
                    "need_reply": "ожидающих ответа",
                    "important": "важных"
                }
                await update.message.reply_text(f"📧 Сообщений {lens_names.get(lens, lens)} не найдено.")
                return
            
            # Ограничиваем количество для отображения
            display_messages = messages[:10]
            
            lens_names = {
                "urgent": "Срочные сообщения",
                "need_reply": "Ожидают ответа",
                "important": "Важные сообщения"
            }
            
            text = f"📧 <b>{lens_names.get(lens, lens)} ({provider}):</b>\n\n"
            
            for i, msg in enumerate(display_messages, 1):
                text += f"📨 <b>{i}. {msg.subject}</b>\n"
                text += f"   От: {msg.sender}\n"
                text += f"   Дата: {msg.date.strftime('%d.%m %H:%M')}\n"
                text += f"   Приоритет: {msg.priority.value}\n"
                if msg.is_reply_needed:
                    text += f"   ⚠️ Требует ответа\n"
                text += "\n"
            
            if len(messages) > 10:
                text += f"... и еще {len(messages) - 10} сообщений"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения сообщений: {e}")

    async def handle_reply_template(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация шаблона ответа."""
        user_text = update.message.text
        
        # Извлекаем тему или ID сообщения
        template_match = re.search(r"(шаблон ответа|ответить на|reply template)\s+(.+)", user_text, re.I)
        if not template_match:
            await update.message.reply_text(
                "📝 <b>Генерация шаблона ответа</b>\n\n"
                "Используйте:\n"
                "• 'Шаблон ответа [тема]' - создать шаблон\n"
                "• 'Ответить на [тема]' - предложить ответ\n"
                "• 'Reply template [subject]' - на английском\n\n"
                "Примеры:\n"
                "• Шаблон ответа на предложение о сотрудничестве\n"
                "• Ответить на запрос цены\n"
                "• Reply template meeting request",
                parse_mode='HTML'
            )
            return
        
        query = template_match.group(2).strip()
        
        await update.message.reply_text(f"📝 Генерирую шаблон ответа для '{query}'...")
        
        try:
            # Создаем временное сообщение для генерации шаблона
            temp_message = EmailMessage(
                id="temp",
                subject=query,
                sender="Unknown",
                sender_email="unknown@example.com",
                date=datetime.now(),
                content=f"Запрос на генерацию шаблона ответа для: {query}",
                priority=EmailPriority.MEDIUM,
                status=EmailStatus.NEW,
                labels=[],
                thread_id="temp",
                is_reply_needed=True
            )
            
            template = email_analyzer.generate_reply_template(temp_message)
            
            text = f"📝 <b>Шаблон ответа для '{query}':</b>\n\n"
            text += template
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка генерации шаблона: {e}")

    async def handle_email_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Настройка email аккаунтов."""
        user_text = update.message.text.lower()
        
        if "настройка" in user_text or "config" in user_text:
            await update.message.reply_text(
                "⚙️ <b>Настройка Email аккаунтов</b>\n\n"
                "Для настройки отредактируйте файл email_config.json:\n\n"
                "1. <b>Gmail:</b>\n"
                "   - Включите двухфакторную аутентификацию\n"
                "   - Создайте пароль приложения\n"
                "   - Укажите email и пароль приложения\n\n"
                "2. <b>Яндекс.Почта:</b>\n"
                "   - Включите IMAP в настройках\n"
                "   - Создайте пароль приложения\n"
                "   - Укажите email и пароль приложения\n\n"
                "После настройки используйте:\n"
                "• 'Сводка входящих' - общая сводка\n"
                "• 'Срочные сообщения' - по линзам\n"
                "• 'Шаблон ответа' - генерация ответов",
                parse_mode='HTML'
            )
        else:
            await update.message.reply_text(
                "📧 <b>Email команды</b>\n\n"
                "• 'Сводка входящих' - общая сводка\n"
                "• 'Срочные сообщения' - срочные и важные\n"
                "• 'Ожидают ответа' - требующие ответа\n"
                "• 'Важные сообщения' - все важные\n"
                "• 'Шаблон ответа [тема]' - генерация ответа\n"
                "• 'Настройка email' - инструкции по настройке\n\n"
                "Поддерживаются Gmail и Яндекс.Почта",
                parse_mode='HTML'
            )

    # --- Партнёрская сеть ---
    async def handle_partners_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать сводку по партнёрской сети."""
        try:
            summary = partners_manager.get_partners_summary()
            
            text = "🤝 <b>Сводка партнёрской сети:</b>\n\n"
            text += f"📊 Всего партнёров: {summary['total']}\n"
            text += f"✅ Активных: {summary['active_partners']}\n"
            text += f"📞 Нуждаются в контакте: {summary['needs_contact']}\n\n"
            
            text += "📈 <b>По статусам:</b>\n"
            for status, count in summary['by_status'].items():
                status_names = {
                    "prospect": "Перспективные",
                    "lead": "Лиды", 
                    "active": "Активные",
                    "partner": "Партнёры",
                    "inactive": "Неактивные"
                }
                text += f"   {status_names.get(status, status)}: {count}\n"
            
            text += "\n📡 <b>По каналам:</b>\n"
            for channel, count in summary['by_channel'].items():
                text += f"   {channel}: {count}\n"
            
            text += "\n🎯 <b>По сегментам:</b>\n"
            for segment, count in summary['by_segment'].items():
                text += f"   {segment}: {count}\n"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения сводки: {e}")

    async def handle_add_partner(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Добавление нового партнёра."""
        user_text = update.message.text
        
        # Парсим команду добавления партнёра
        pattern = r"добавь партнёра? ([^,]+), канал ([^,]+), контакты ([^,]+)(?:, статус ([^,]+))?(?:, сегмент ([^,]+))?"
        match = re.search(pattern, user_text, re.I)
        
        if not match:
            await update.message.reply_text(
                "🤝 <b>Добавление партнёра</b>\n\n"
                "Формат:\n"
                "• 'Добавь партнёра [имя], канал [канал], контакты [контакты]'\n"
                "• 'Добавь партнёра [имя], канал [канал], контакты [контакты], статус [статус]'\n"
                "• 'Добавь партнёра [имя], канал [канал], контакты [контакты], статус [статус], сегмент [сегмент]'\n\n"
                "Примеры:\n"
                "• Добавь партнёра Иван Петров, канал telegram, контакты @ivan_petrov\n"
                "• Добавь партнёра ООО Рога, канал email, контакты info@roga.ru, статус prospect, сегмент startup",
                parse_mode='HTML'
            )
            return
        
        name = match.group(1).strip()
        channel = match.group(2).strip()
        contacts = match.group(3).strip()
        status = match.group(4).strip() if match.group(4) else "prospect"
        segment = match.group(5).strip() if match.group(5) else "general"
        
        try:
            success = partners_manager.add_partner(name, channel, contacts, status, segment)
            if success:
                await update.message.reply_text(f"✅ Партнёр {name} добавлен в базу")
            else:
                await update.message.reply_text("❌ Ошибка добавления партнёра")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_partners_for_calling(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать партнёров для прозвона."""
        user_text = update.message.text.lower()
        
        # Определяем количество дней
        days = 7
        if "неделя" in user_text:
            days = 7
        elif "месяц" in user_text:
            days = 30
        elif "день" in user_text:
            days = 1
        
        try:
            partners = partners_manager.get_partners_for_calling(days)
            
            if not partners:
                await update.message.reply_text(f"📞 Нет партнёров для прозвона (не контактировали {days} дней)")
                return
            
            text = f"📞 <b>Партнёры для прозвона (не контактировали {days} дней):</b>\n\n"
            
            for i, partner in enumerate(partners[:10], 1):
                text += f"{i}. <b>{partner['name']}</b>\n"
                text += f"   Канал: {partner['channel']}\n"
                text += f"   Контакты: {partner['contacts']}\n"
                text += f"   Сегмент: {partner.get('segment', 'general')}\n"
                text += f"   Последний контакт: {partner.get('last_contact', 'Не указан')}\n\n"
            
            if len(partners) > 10:
                text += f"... и еще {len(partners) - 10} партнёров"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения списка: {e}")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка голосовых сообщений."""
        try:
            # Отправляем сообщение о начале обработки
            processing_msg = await update.message.reply_text(
                "🎤 Обрабатываю голосовое сообщение...\n"
                "🔍 Выполняю распознавание речи..."
            )
            
            # Получаем голосовое сообщение
            voice = update.message.voice
            file = await context.bot.get_file(voice.file_id)
            
            # Скачиваем аудиофайл
            temp_path = f"/tmp/voice_{voice.file_id}.ogg"
            await file.download_to_drive(temp_path)
            
            # Распознаем речь
            recognized_text = speech_recognizer.recognize_speech(temp_path)
            
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if not recognized_text:
                await processing_msg.edit_text(
                    "❌ Не удалось распознать речь.\n"
                    "Попробуйте говорить четче или отправьте текстовое сообщение."
                )
                return
            
            # Сохраняем распознанный текст в память
            user_id = update.message.from_user.id
            username = update.message.from_user.username or update.message.from_user.full_name
            chat_memory.add_message(user_id=user_id, username=username, text=recognized_text, role="user")
            
            # Показываем распознанный текст
            await processing_msg.edit_text(
                f"🎤 <b>Распознанный текст:</b>\n{recognized_text}\n\n"
                f"🔧 Обрабатываю команду...",
                parse_mode='HTML'
            )

            # --- Новый блок: парсер голосовых задач сотрудникам ---
            import re
            from core.team_manager import team_manager
            from datetime import datetime, timedelta
            
            # Примеры: "Пусть Маша проверит остатки", "Поручи Ивану отправить отчёт", "Попроси Сергея сделать ..."
            task_patterns = [
                r"пусть ([а-яёa-zA-Z]+) (.+)",
                r"поручи ([а-яёa-zA-Z]+)[уe] (.+)",
                r"попроси ([а-яёa-zA-Z]+)[уe] (.+)",
                r"назначь ([а-яёa-zA-Z]+)[уe] (.+)",
                r"([а-яёa-zA-Z]+),? (.+)"  # Маша, проверь остатки
            ]
            matched = None
            for pattern in task_patterns:
                m = re.match(pattern, recognized_text.strip(), re.I)
                if m:
                    matched = m
                    break
            
            if matched:
                employee_name = matched.group(1).strip().capitalize()
                task_text = matched.group(2).strip().capitalize()
                # Поиск сотрудника по имени (нечувствительно к регистру)
                employees = team_manager.team_data['employees']
                employee_id = None
                for eid, edata in employees.items():
                    if edata['name'].lower().startswith(employee_name.lower()):
                        employee_id = eid
                        break
                if not employee_id:
                    await update.message.reply_text(f"❌ Сотрудник '{employee_name}' не найден в команде. Добавьте его через 'добавить сотрудника'.")
                    return
                # Дедлайн по умолчанию: завтра
                deadline = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                # Назначаем задачу
                ok = team_manager.assign_task(employee_id, task_text, deadline, priority='medium')
                if ok:
                    await update.message.reply_text(f"✅ Задача для {employee_name} создана: {task_text}\nДедлайн: {deadline}")
                    # Уведомление сотруднику (если есть chat_id)
                    chat_id = employees[employee_id].get('chat_id')
                    if chat_id:
                        try:
                            await context.bot.send_message(chat_id=chat_id, text=f"📝 Новая задача: {task_text}\nДедлайн: {deadline}")
                        except Exception as e:
                            await update.message.reply_text(f"⚠️ Не удалось отправить уведомление сотруднику: {e}")
                    # Сообщение в общий чат
                    await context.bot.send_message(update.effective_chat.id, f"📢 Задача назначена: {employee_name} — {task_text} (до {deadline})")
                    return
                else:
                    await update.message.reply_text(f"❌ Не удалось назначить задачу сотруднику {employee_name}.")
                    return
            # --- Конец блока ---
            
            # Создаем фейковое сообщение для обработки
            class FakeMessage:
                def __init__(self, original_message, text):
                    self.text = text
                    self.from_user = original_message.from_user
                    self.effective_chat = original_message.effective_chat
                    self.reply_text = original_message.reply_text
            
            class FakeUpdate:
                def __init__(self, original_update, text):
                    self.message = FakeMessage(original_update.message, text)
            
            # Обрабатываем распознанный текст как обычную команду
            fake_update = FakeUpdate(update, recognized_text)
            await handle_message(fake_update, context)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка обработки голосового сообщения: {e}")


    async def handle_partners_for_emailing(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать партнёров для рассылки."""
        user_text = update.message.text.lower()
        
        # Определяем сегмент
        segment = None
        if "стартап" in user_text:
            segment = "startup"
        elif "enterprise" in user_text or "крупные" in user_text:
            segment = "enterprise"
        elif "агентство" in user_text:
            segment = "agency"
        elif "разработчик" in user_text:
            segment = "developer"
        
        try:
            partners = partners_manager.get_partners_for_emailing(segment)
            
            if not partners:
                segment_text = f" сегмента {segment}" if segment else ""
                await update.message.reply_text(f"📧 Нет партнёров для email рассылки{segment_text}")
                return
            
            text = f"📧 <b>Партнёры для email рассылки"
            if segment:
                text += f" (сегмент: {segment})"
            text += f":</b>\n\n"
            
            for i, partner in enumerate(partners[:10], 1):
                text += f"{i}. <b>{partner['name']}</b>\n"
                text += f"   Контакты: {partner['contacts']}\n"
                text += f"   Статус: {partner['status']}\n"
                text += f"   Сегмент: {partner.get('segment', 'general')}\n\n"
            
            if len(partners) > 10:
                text += f"... и еще {len(partners) - 10} партнёров"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения списка: {e}")

    async def handle_generate_proposal(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация предложения для партнёра."""
        user_text = update.message.text
        
        # Парсим команду генерации предложения
        pattern = r"(предложение|proposal) (?:для|к) ([^,]+)(?:, сегмент ([^,]+))?"
        match = re.search(pattern, user_text, re.I)
        
        if not match:
            await update.message.reply_text(
                "📝 <b>Генерация предложения</b>\n\n"
                "Формат:\n"
                "• 'Предложение для [имя партнёра]'\n"
                "• 'Предложение для [имя партнёра], сегмент [сегмент]'\n\n"
                "Примеры:\n"
                "• Предложение для Иван Петров\n"
                "• Proposal для ООО Рога, сегмент startup",
                parse_mode='HTML'
            )
            return
        
        partner_name = match.group(2).strip()
        segment = match.group(3).strip() if match.group(3) else None
        
        try:
            # Ищем партнёра
            partners = partners_manager.get_all_partners()
            partner = None
            for p in partners:
                if p['name'].lower() == partner_name.lower():
                    partner = p
                    break
            
            if not partner:
                await update.message.reply_text(f"❌ Партнёр '{partner_name}' не найден в базе")
                return
            
            await update.message.reply_text("🤖 Генерирую персонализированное предложение...")
            
            proposal = partners_manager.generate_proposal(partner, segment)
            
            text = f"📝 <b>Предложение для {partner['name']}:</b>\n\n"
            text += proposal
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка генерации предложения: {e}")

    async def handle_bulk_proposals(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация предложений для группы партнёров."""
        user_text = update.message.text.lower()
        
        # Определяем сегмент
        segment = None
        if "стартап" in user_text:
            segment = "startup"
        elif "enterprise" in user_text or "крупные" in user_text:
            segment = "enterprise"
        elif "агентство" in user_text:
            segment = "agency"
        elif "разработчик" in user_text:
            segment = "developer"
        
        # Определяем лимит
        limit = 5
        if "10" in user_text:
            limit = 10
        elif "3" in user_text:
            limit = 3
        
        try:
            await update.message.reply_text(f"🤖 Генерирую предложения для {limit} партнёров...")
            
            proposals = partners_manager.generate_bulk_proposals(segment, limit)
            
            if not proposals:
                segment_text = f" сегмента {segment}" if segment else ""
                await update.message.reply_text(f"❌ Нет партнёров для генерации предложений{segment_text}")
                return
            
            text = f"📝 <b>Предложения для партнёров"
            if segment:
                text += f" (сегмент: {segment})"
            text += f":</b>\n\n"
            
            for i, item in enumerate(proposals, 1):
                partner = item['partner']
                proposal = item['proposal']
                
                text += f"<b>{i}. {partner['name']}</b>\n"
                text += f"Канал: {partner['channel']}\n"
                text += f"Контакты: {partner['contacts']}\n\n"
                text += f"<i>Предложение:</i>\n{proposal}\n"
                text += "─" * 50 + "\n\n"
            
            # Разбиваем на части, если слишком длинное
            if len(text) > 4000:
                parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
                for i, part in enumerate(parts):
                    await update.message.reply_text(f"{part} (часть {i+1}/{len(parts)})", parse_mode='HTML')
            else:
                await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка генерации предложений: {e}")

    # --- AmoCRM Commands ---

    async def handle_amocrm_contacts(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать контакты из AmoCRM."""
        user_text = update.message.text.lower()
        
        # Определяем лимит
        limit = 10
        if "20" in user_text:
            limit = 20
        elif "50" in user_text:
            limit = 50
        
        # Поиск по имени
        query = None
        if "найди" in user_text or "поиск" in user_text:
            # Извлекаем имя после "найди" или "поиск"
            import re
            match = re.search(r'(?:найди|поиск)\s+([^\s]+)', user_text)
            if match:
                query = match.group(1)
        
        try:
            contacts = amocrm.get_contacts(limit=limit, query=query)
            
            if not contacts:
                query_text = f" по запросу '{query}'" if query else ""
                await update.message.reply_text(f"👥 Нет контактов в AmoCRM{query_text}")
                return
            
            text = f"👥 <b>Контакты в AmoCRM"
            if query:
                text += f" (поиск: {query})"
            text += f":</b>\n\n"
            
            for i, contact in enumerate(contacts[:10], 1):
                text += f"{i}. <b>{contact['name']}</b>\n"
                text += f"   ID: {contact['id']}\n"
                
                # Добавляем email и телефон если есть
                if 'custom_fields_values' in contact:
                    for field in contact['custom_fields_values']:
                        if field.get('field_id') == 1:  # Email
                            text += f"   Email: {field['values'][0]['value']}\n"
                        elif field.get('field_id') == 2:  # Phone
                            text += f"   Телефон: {field['values'][0]['value']}\n"
                
                text += f"   Создан: {datetime.fromtimestamp(contact.get('created_at', 0)).strftime('%d.%m.%Y')}\n\n"
            
            if len(contacts) > 10:
                text += f"... и еще {len(contacts) - 10} контактов"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения контактов: {e}")

    async def handle_amocrm_leads(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать сделки из AmoCRM."""
        user_text = update.message.text.lower()
        
        # Определяем лимит
        limit = 10
        if "20" in user_text:
            limit = 20
        elif "50" in user_text:
            limit = 50
        
        try:
            leads = amocrm.get_leads(limit=limit)
            
            if not leads:
                await update.message.reply_text("💼 Нет сделок в AmoCRM")
                return
            
            text = f"💼 <b>Сделки в AmoCRM:</b>\n\n"
            
            for i, lead in enumerate(leads[:10], 1):
                text += f"{i}. <b>{lead['name']}</b>\n"
                text += f"   ID: {lead['id']}\n"
                text += f"   Статус ID: {lead.get('status_id', 'Не указан')}\n"
                text += f"   Сумма: {lead.get('price', 0)} ₽\n"
                text += f"   Создана: {datetime.fromtimestamp(lead.get('created_at', 0)).strftime('%d.%m.%Y')}\n\n"
            
            if len(leads) > 10:
                text += f"... и еще {len(leads) - 10} сделок"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения сделок: {e}")

    async def handle_amocrm_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать аналитику AmoCRM."""
        user_text = update.message.text.lower()
        
        # Определяем период
        period = "month"
        if "неделя" in user_text or "неделю" in user_text:
            period = "week"
        elif "месяц" in user_text or "месяца" in user_text:
            period = "month"
        
        try:
            analytics = amocrm.get_analytics(period=period)
            
            text = f"📊 <b>Аналитика AmoCRM за {period}:</b>\n\n"
            text += f"📈 Всего лидов: {analytics['total_leads']}\n"
            text += f"✅ Выигранных сделок: {analytics['won_leads']}\n"
            text += f"📊 Конверсия: {analytics['conversion_rate']:.1f}%\n"
            text += f"💰 Общая выручка: {analytics['total_revenue']} ₽\n"
            text += f"💎 Средний чек: {analytics['avg_deal_size']:.0f} ₽\n"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения аналитики: {e}")

    async def handle_amocrm_sync_partners(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Синхронизация партнёров из Google Sheets в AmoCRM."""
        try:
            await update.message.reply_text("🔄 Синхронизирую партнёров из Google Sheets в AmoCRM...")
            
            result = amocrm.sync_partners_from_sheet(partners_manager)
            
            text = f"✅ <b>Синхронизация завершена:</b>\n\n"
            text += f"🆕 Создано контактов: {result['created']}\n"
            text += f"🔄 Обновлено контактов: {result['updated']}\n"
            text += f"❌ Ошибок: {result['errors']}\n"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка синхронизации: {e}")

    async def handle_amocrm_create_contact(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Создание контакта в AmoCRM."""
        user_text = update.message.text
        
        # Парсим команду создания контакта
        pattern = r"(?:создай|добавь)\s+контакт\s+([^,]+)(?:,\s+email\s+([^,]+))?(?:,\s+телефон\s+([^,]+))?"
        match = re.search(pattern, user_text, re.I)
        
        if not match:
            await update.message.reply_text(
                "👤 <b>Создание контакта в AmoCRM</b>\n\n"
                "Формат:\n"
                "• 'Создай контакт [имя]'\n"
                "• 'Создай контакт [имя], email [email]'\n"
                "• 'Создай контакт [имя], email [email], телефон [телефон]'\n\n"
                "Примеры:\n"
                "• Создай контакт Иван Петров\n"
                "• Создай контакт ООО Рога, email info@roga.ru\n"
                "• Создай контакт ИП Копыта, email kopyta@mail.ru, телефон +7-999-123-45-67",
                parse_mode='HTML'
            )
            return
        
        name = match.group(1).strip()
        email = match.group(2).strip() if match.group(2) else None
        phone = match.group(3).strip() if match.group(3) else None
        
        try:
            contact = amocrm.create_contact(name=name, email=email, phone=phone)
            
            if contact:
                text = f"✅ <b>Контакт создан в AmoCRM:</b>\n\n"
                text += f"👤 Имя: {contact['name']}\n"
                text += f"🆔 ID: {contact['id']}\n"
                if email:
                    text += f"📧 Email: {email}\n"
                if phone:
                    text += f"📞 Телефон: {phone}\n"
                
                await update.message.reply_text(text, parse_mode='HTML')
            else:
                await update.message.reply_text("❌ Ошибка создания контакта")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_amocrm_create_lead(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Создание сделки в AmoCRM."""
        user_text = update.message.text
        
        # Парсим команду создания сделки
        pattern = r"(?:создай|добавь)\s+сделку\s+([^,]+)(?:,\s+контакт\s+([^,]+))?(?:,\s+сумма\s+(\d+))?"
        match = re.search(pattern, user_text, re.I)
        
        if not match:
            await update.message.reply_text(
                "💼 <b>Создание сделки в AmoCRM</b>\n\n"
                "Формат:\n"
                "• 'Создай сделку [название]'\n"
                "• 'Создай сделку [название], контакт [имя]'\n"
                "• 'Создай сделку [название], контакт [имя], сумма [число]'\n\n"
                "Примеры:\n"
                "• Создай сделку Разработка сайта\n"
                "• Создай сделку Консультация, контакт Иван Петров\n"
                "• Создай сделку Дизайн логотипа, контакт ООО Рога, сумма 50000",
                parse_mode='HTML'
            )
            return
        
        name = match.group(1).strip()
        contact_name = match.group(2).strip() if match.group(2) else None
        amount = int(match.group(3)) if match.group(3) else 0
        
        try:
            contact_id = None
            if contact_name:
                # Ищем контакт по имени
                contacts = amocrm.get_contacts(query=contact_name)
                if contacts:
                    contact_id = contacts[0]['id']
                else:
                    await update.message.reply_text(f"❌ Контакт '{contact_name}' не найден")
                    return
            
            lead = amocrm.create_lead(name=name, contact_id=contact_id, custom_fields={1: amount})
            
            if lead:
                text = f"✅ <b>Сделка создана в AmoCRM:</b>\n\n"
                text += f"💼 Название: {lead['name']}\n"
                text += f"🆔 ID: {lead['id']}\n"
                if contact_name:
                    text += f"👤 Контакт: {contact_name}\n"
                if amount > 0:
                    text += f"💰 Сумма: {amount} ₽\n"
                
                await update.message.reply_text(text, parse_mode='HTML')
            else:
                await update.message.reply_text("❌ Ошибка создания сделки")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_amocrm_pipelines(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать воронки продаж в AmoCRM."""
        try:
            pipelines = amocrm.get_pipelines()
            
            if not pipelines:
                await update.message.reply_text("🔄 Нет воронок продаж в AmoCRM")
                return
            
            text = f"🔄 <b>Воронки продаж в AmoCRM:</b>\n\n"
            
            for i, pipeline in enumerate(pipelines, 1):
                text += f"{i}. <b>{pipeline['name']}</b>\n"
                text += f"   ID: {pipeline['id']}\n"
                text += f"   Активна: {'Да' if pipeline.get('is_main', False) else 'Нет'}\n\n"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения воронок: {e}")

    async def handle_amocrm_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать задачи в AmoCRM."""
        user_text = update.message.text.lower()
        
        # Определяем лимит
        limit = 10
        if "20" in user_text:
            limit = 20
        elif "50" in user_text:
            limit = 50
        
        try:
            tasks = amocrm.get_tasks(limit=limit)
            
            if not tasks:
                await update.message.reply_text("📋 Нет задач в AmoCRM")
                return
            
            text = f"📋 <b>Задачи в AmoCRM:</b>\n\n"
            
            for i, task in enumerate(tasks[:10], 1):
                text += f"{i}. <b>{task['text']}</b>\n"
                text += f"   ID: {task['id']}\n"
                text += f"   Тип: {task.get('entity_type', 'Не указан')}\n"
                text += f"   Создана: {datetime.fromtimestamp(task.get('created_at', 0)).strftime('%d.%m.%Y')}\n\n"
            
            if len(tasks) > 10:
                text += f"... и еще {len(tasks) - 10} задач"
            
            await update.message.reply_text(text, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения задач: {e}")

    async def handle_goals_progress_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать прогресс по всем целям (OKR/KPI) с трендом, статусом и прогнозом."""
        try:
            goals = goals_manager.get_active_goals()
            if not goals:
                await update.message.reply_text(
                    "🎯 <b>Нет активных целей</b>\n\nСоздайте первую цель:\n• 'Создать цель выручка 3 млн до сентября'\n• 'Новая цель подписки 100 клиентов до декабря'",
                    parse_mode='HTML'
                )
                return
            report = "<b>Прогресс по целям (OKR/KPI):</b>\n\n"
            for goal in goals:
                progress = goals_manager.get_goal_progress(goal.id)
                trend = progress['trend']
                forecast = progress['forecast']
                status_emoji = "🟢" if progress['is_on_track'] else "🔴"
                trend_emoji = "📈" if trend['direction'] == 'increasing' else "📉" if trend['direction'] == 'decreasing' else "➡️"
                forecast_text = ""
                if forecast['achievable']:
                    forecast_text = f"✅ Достижима, завершение: {datetime.fromisoformat(forecast['estimated_completion']).strftime('%d.%m.%Y') if forecast['estimated_completion'] else '—'}"
                else:
                    forecast_text = f"⚠️ Требуется скорость: {forecast.get('required_rate', '—')}/день"
                deadline = goal.end_date if goal.end_date else '—'
                report += (
                    f"<b>{goal.name}</b> ({goal.goal_type.value.title()})\n"
                    f"{status_emoji} {progress['progress_percentage']}% — {goal.current_value}/{goal.target_value}\n"
                    f"{trend_emoji} Тренд: {trend['direction']} ({trend['rate']}/день)\n"
                    f"⏳ Дедлайн: {deadline}\n"
                    f"{forecast_text}\n\n"
                )
            await update.message.reply_text(report, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения прогресса по целям: {e}")

    # --- Obsidian функции ---
    async def handle_create_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка создания стратегии в Obsidian."""
        user_text = update.message.text
        
        # Парсим команду создания стратегии
        # Пример: "Создай стратегию Развитие продаж: увеличить выручку на 50% до конца года"
        strategy_match = re.search(r"(?:создай стратегию|новая стратегия|добавь стратегию)\s+(.+?):\s*(.+)", user_text, re.I)
        
        if not strategy_match:
            await update.message.reply_text(
                "🎯 <b>Создание стратегии</b>\n\n"
                "Используйте формат:\n"
                "• 'Создай стратегию [название]: [описание]'\n\n"
                "Примеры:\n"
                "• Создай стратегию Развитие продаж: увеличить выручку на 50% до конца года\n"
                "• Новая стратегия Маркетинг: запустить рекламную кампанию в соцсетях\n"
                "• Добавь стратегию Продукт: разработать новую линейку товаров",
                parse_mode='HTML'
            )
            return
        
        title = strategy_match.group(1).strip()
        content = strategy_match.group(2).strip()
        
        try:
            # Создаем стратегию
            note_id = obsidian_manager.create_strategy_note(
                title=title,
                content=content,
                tags=["стратегия", "планирование"]
            )
            
            await update.message.reply_text(
                f"✅ <b>Стратегия создана в Obsidian!</b>\n\n"
                f"🎯 Название: {title}\n"
                f"📝 Описание: {content}\n"
                f"🆔 ID: {note_id}\n\n"
                f"Файл сохранен в папке: 01-Стратегии",
                parse_mode='HTML'
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка создания стратегии: {e}")

    async def handle_create_decision(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка создания решения в Obsidian."""
        user_text = update.message.text
        
        # Парсим команду создания решения
        # Пример: "Создай решение Выбор поставщика: проблема - нет надежного поставщика, решение - заключить договор с ООО Рога"
        decision_match = re.search(r"(?:создай решение|новое решение|добавь решение)\s+(.+?):\s*проблема\s*-\s*(.+?),\s*решение\s*-\s*(.+)", user_text, re.I)
        
        if not decision_match:
            await update.message.reply_text(
                "🤔 <b>Создание решения</b>\n\n"
                "Используйте формат:\n"
                "• 'Создай решение [название]: проблема - [описание проблемы], решение - [описание решения]'\n\n"
                "Примеры:\n"
                "• Создай решение Выбор поставщика: проблема - нет надежного поставщика, решение - заключить договор с ООО Рога\n"
                "• Новое решение Технологии: проблема - устаревшие системы, решение - внедрить новое ПО",
                parse_mode='HTML'
            )
            return
        
        title = decision_match.group(1).strip()
        problem = decision_match.group(2).strip()
        solution = decision_match.group(3).strip()
        
        try:
            # Создаем решение
            note_id = obsidian_manager.create_decision_note(
                title=title,
                problem=problem,
                solution=solution,
                reasoning="Решение принято на основе анализа ситуации"
            )
            
            await update.message.reply_text(
                f"✅ <b>Решение создано в Obsidian!</b>\n\n"
                f"🤔 Название: {title}\n"
                f"❌ Проблема: {problem}\n"
                f"✅ Решение: {solution}\n"
                f"🆔 ID: {note_id}\n\n"
                f"Файл сохранен в папке: 02-Решения",
                parse_mode='HTML'
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка создания решения: {e}")

    async def handle_obsidian_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Поиск по заметкам в Obsidian."""
        user_text = update.message.text
        
        # Извлекаем поисковый запрос
        search_match = re.search(r"(?:найди в obsidian|поиск в obsidian|obsidian поиск)\s+(.+)", user_text, re.I)
        
        if not search_match:
            await update.message.reply_text(
                "🔍 <b>Поиск в Obsidian</b>\n\n"
                "Используйте формат:\n"
                "• 'Найди в obsidian [запрос]'\n"
                "• 'Поиск в obsidian [запрос]'\n\n"
                "Примеры:\n"
                "• Найди в obsidian продажи\n"
                "• Поиск в obsidian проект ВБ\n"
                "• Obsidian поиск стратегия",
                parse_mode='HTML'
            )
            return
        
        query = search_match.group(1).strip()
        
        try:
            # Выполняем поиск
            results = obsidian_manager.search_notes(query)
            
            if not results:
                await update.message.reply_text(
                    f"🔍 <b>Поиск: '{query}'</b>\n\n"
                    f"❌ Заметки не найдены.\n\n"
                    f"Попробуйте:\n"
                    f"• Изменить запрос\n"
                    f"• Использовать другие ключевые слова\n"
                    f"• Проверить правильность написания",
                    parse_mode='HTML'
                )
                return
            
            # Формируем отчет
            report = f"🔍 <b>Поиск: '{query}'</b>\n\n"
            report += f"📊 Найдено заметок: {len(results)}\n\n"
            
            for i, result in enumerate(results[:10], 1):  # Показываем первые 10
                category_emoji = {
                    '01-Стратегии': '🎯',
                    '02-Решения': '🤔',
                    '03-Логи': '📝',
                    '04-Проекты': '📁',
                    '05-Встречи': '👥',
                    '06-Задачи': '✅',
                    '07-Финансы': '💰',
                    '08-Партнеры': '🤝',
                    '09-Клиенты': '👤',
                    '10-Документы': '📄'
                }.get(result['category'], '��')
                
                report += f"{i}. {category_emoji} <b>{result['title']}</b>\n"
                report += f"   📁 {result['category']}\n"
                report += f"   📅 {result['created_date']}\n"
                report += f"   📝 {result['content_preview']}\n\n"
            
            if len(results) > 10:
                report += f"... и еще {len(results) - 10} заметок\n\n"
            
            report += "💡 <b>Команды:</b>\n"
            report += "• 'Статистика obsidian' - общая сводка\n"
            report += "• 'Создай стратегию [название]: [описание]' - новая стратегия\n"
            report += "• 'Создай решение [название]: проблема - [описание], решение - [описание]' - решение"
            
            await update.message.reply_text(report, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка поиска: {e}")

    async def handle_obsidian_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику Obsidian."""
        try:
            stats = obsidian_manager.get_statistics()
            
            report = "📊 <b>Статистика Obsidian</b>\n\n"
            report += f"📝 Всего заметок: {stats['total_notes']}\n"
            report += f"📁 Проектов: {len(stats['projects'])}\n\n"
            
            report += "📂 <b>По категориям:</b>\n"
            for category, count in stats['by_category'].items():
                if count > 0:
                    category_name = category.replace('-', ' ').title()
                    report += f"   {category_name}: {count}\n"
            
            if stats['recent_notes']:
                report += "\n🕒 <b>Последние заметки:</b>\n"
                for note in stats['recent_notes'][:5]:
                    category_emoji = {
                        '01-Стратегии': '🎯',
                        '02-Решения': '🤔',
                        '03-Логи': '📝',
                        '04-Проекты': '📁',
                        '05-Встречи': '👥',
                        '06-Задачи': '✅',
                        '07-Финансы': '💰',
                        '08-Партнеры': '🤝',
                        '09-Клиенты': '👤',
                        '10-Документы': '📄'
                    }.get(note['category'], '📋')
                    
                    report += f"   {category_emoji} {note['title']} ({note['modified']})\n"
            
            if stats['projects']:
                report += "\n📁 <b>Проекты:</b>\n"
                for project in stats['projects'][:5]:
                    report += f"   📁 {project}\n"
            
            report += "\n💡 <b>Команды:</b>\n"
            report += "• 'Найди в obsidian [запрос]' - поиск\n"
            report += "• 'Создай стратегию [название]: [описание]' - стратегия\n"
            report += "• 'Создай решение [название]: проблема - [описание], решение - [описание]' - решение"
            
            await update.message.reply_text(report, parse_mode='HTML')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")

        # Аналитика бизнеса
        if re.search(r"(аналитика бизнеса|метрики бизнеса|выгрузи roi|выгрузи ltv|выгрузи cac)", user_text, re.I):
            import core.analytics
            metrics = core.analytics.business_analytics.get_business_metrics(period='месяц')
            text = (
                f"📊 <b>Аналитика бизнеса за месяц</b>\n\n"
                f"• ROI: {metrics['roi']}%\n"
                f"• Оборачиваемость: {metrics['turnover']} раза\n"
                f"• CAC: {metrics['cac']:.0f} ₽\n"
                f"• LTV: {metrics['ltv']:.0f} ₽\n\n"
                f"— Выручка: {metrics['revenue']:.0f} ₽\n"
                f"— Прибыль: {metrics['profit']:.0f} ₽\n"
                f"— Новых клиентов: {metrics['num_new_clients']}\n"
                f"— Средний чек: {metrics['avg_deal']:.0f} ₽"
            )
            await update.message.reply_text(text, parse_mode='HTML')
            return

    async def handle_add_employee(update, context):
        """Добавление сотрудника в команду"""
        user_text = update.message.text.lower()
        
        # Парсим: "Добавить сотрудника Иван Петров @ivan_petrov менеджер"
        match = re.search(r"добавить сотрудника (.+?) (@\w+) (.+)", user_text)
        if not match:
            await update.message.reply_text(
                "❌ Формат: 'Добавить сотрудника [Имя] [@username] [Должность]'\n"
                "Пример: 'Добавить сотрудника Иван Петров @ivan_petrov менеджер'"
            )
            return
        
        name, telegram_id, role = match.groups()
        telegram_id = telegram_id.replace('@', '')
        
        success = core.team_manager.team_manager.add_employee(
            name=name, 
            telegram_id=telegram_id, 
            role=role,
            chat_id=str(update.message.chat_id)
        )
        
        if success:
            await update.message.reply_text(
                f"✅ Сотрудник {name} добавлен в команду!\n"
                f"Должность: {role}\n"
                f"Telegram: @{telegram_id}"
            )
        else:
            await update.message.reply_text("❌ Ошибка добавления сотрудника")

    async def handle_assign_task(update, context):
        """Назначение задачи сотруднику"""
        user_text = update.message.text
        
        # Парсим: "Назначить задачу @ivan_petrov Создать презентацию до 2024-01-15 высокая"
    match = re.search(r"назначить задачу (@\w+) (.+?) до (\d{4}-\d{2}-\d{2})(?: (.+))?", user_text)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Назначить задачу [@username] [Задача] до [YYYY-MM-DD] [приоритет]'\n"
            "Пример: 'Назначить задачу @ivan_petrov Создать презентацию до 2024-01-15 высокая'"
        )
        return
    
    telegram_id, task, deadline, priority = match.groups()
    telegram_id = telegram_id.replace('@', '')
    priority = priority or 'medium'
    
    success = core.team_manager.team_manager.assign_task(
        employee_id=telegram_id,
        task=task,
        deadline=deadline,
        priority=priority
    )
    
    if success:
        await update.message.reply_text(
            f"✅ Задача назначена!\n"
            f"Сотрудник: @{telegram_id}\n"
            f"Задача: {task}\n"
            f"Дедлайн: {deadline}\n"
            f"Приоритет: {priority}"
        )
    else:
        await update.message.reply_text("❌ Сотрудник не найден в команде")

async def handle_my_tasks(update, context):
    """Показать мои задачи"""
    user_id = str(update.message.from_user.id)
    tasks = core.team_manager.team_manager.get_employee_tasks(user_id)
    
    if not tasks:
        await update.message.reply_text("✅ У вас нет активных задач!")
        return
    
    text = f"📋 Ваши активные задачи ({len(tasks)}):\n\n"
    
    for task in tasks:
        deadline = datetime.fromisoformat(task['deadline'])
        days_left = (deadline - datetime.now()).days
        status_emoji = "🔴" if days_left < 0 else "🟡" if days_left <= 2 else "🟢"
        
        text += f"{status_emoji} {task['task']}\n"
        text += f"   Дедлайн: {task['deadline']} (осталось {days_left} дн.)\n"
        text += f"   Приоритет: {task['priority']}\n"
        if task['description']:
            text += f"   Описание: {task['description']}\n"
        text += "\n"
    
    await update.message.reply_text(text)

async def handle_complete_task(update, context):
    """Завершение задачи"""
    user_text = update.message.text
    user_id = str(update.message.from_user.id)
    
    # Парсим: "Завершить задачу task_1 Отчёт готов"
    match = re.search(r"завершить задачу (\w+) (.+)", user_text)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Завершить задачу [ID_задачи] [Отчёт]'\n"
            "Пример: 'Завершить задачу task_1 Отчёт готов, презентация создана'"
        )
        return
    
    task_id, report = match.groups()
    
    success = core.team_manager.team_manager.complete_task(
        task_id=task_id,
        employee_id=user_id,
        report=report
    )
    
    if success:
        await update.message.reply_text(
            f"✅ Задача {task_id} завершена!\n"
            f"Отчёт: {report}"
        )
    else:
        await update.message.reply_text("❌ Задача не найдена или не принадлежит вам")

async def handle_daily_report(update, context):
    """Подача ежедневного отчёта"""
    user_text = update.message.text
    user_id = str(update.message.from_user.id)
    
    # Парсим: "Отчёт: Сегодня работал над презентацией, завершил 2 задачи"
    match = re.search(r"отчёт: (.+)", user_text, re.I)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Отчёт: [ваш отчёт]'\n"
            "Пример: 'Отчёт: Сегодня работал над презентацией, завершил 2 задачи'"
        )
        return
    
    report = match.group(1)
    
    success = core.team_manager.team_manager.submit_daily_report(
        employee_id=user_id,
        report=report
    )
    
    if success:
        await update.message.reply_text(
            f"✅ Ежедневный отчёт отправлен!\n\n"
            f"📝 Ваш отчёт:\n{report}"
        )
    else:
        await update.message.reply_text("❌ Вы не зарегистрированы в команде")

async def handle_team_status(update, context):
    """Показать статус команды"""
    status = core.team_manager.team_manager.get_team_status()
    
    text = f"📊 Статус команды\n\n"
    text += f"📈 Общая статистика:\n"
    text += f"• Всего задач: {status['total_tasks']}\n"
    text += f"• Завершено: {status['completed_tasks']}\n"
    text += f"• Активных: {status['active_tasks']}\n"
    text += f"• Процент выполнения: {status['completion_rate']:.1f}%\n\n"
    
    text += f"👥 Сотрудники:\n"
    for name, emp_stats in status['employees'].items():
        text += f"• {name} ({emp_stats['role']})\n"
        text += f"  Активных задач: {emp_stats['active_tasks']}\n"
        text += f"  Завершено: {emp_stats['completed_tasks']}\n\n"
    
    if status['owner_vacation_mode']:
        text += "🏖️ Режим 'Владелец в отпуске' АКТИВЕН"
    
    await update.message.reply_text(text)

async def handle_vacation_mode(update, context):
    """Управление режимом 'владелец в отпуске'"""
    user_text = update.message.text.lower()
    
    if "включить" in user_text or "активировать" in user_text:
        success = core.team_manager.team_manager.enable_vacation_mode()
        if success:
            await update.message.reply_text(
                "🏖️ Режим 'Владелец в отпуске' АКТИВИРОВАН!\n\n"
                "Ассистент будет:\n"
                "• Отправлять ежедневные напоминания\n"
                "• Контролировать просроченные задачи\n"
                "• Собирать отчёты автоматически"
            )
    elif "выключить" in user_text or "деактивировать" in user_text:
        success = core.team_manager.team_manager.disable_vacation_mode()
        if success:
            await update.message.reply_text("✅ Режим 'Владелец в отпуске' отключён")
    else:
        await update.message.reply_text(
            "❌ Формат: 'Включить режим отпуска' или 'Выключить режим отпуска'"
        )

async def handle_overdue_tasks(update, context):
    """Показать просроченные задачи"""
    overdue = core.team_manager.team_manager.get_overdue_tasks()
    
    if not overdue:
        await update.message.reply_text("✅ Просроченных задач нет!")
        return
    
    text = f"🚨 Просроченные задачи ({len(overdue)}):\n\n"
    
    for task in overdue:
        emp_name = core.team_manager.team_manager.team_data['employees'][task['employee_id']]['name']
        text += f"• {emp_name}: {task['task']}\n"
        text += f"  Просрочено на {task['days_overdue']} дн.\n"
        text += f"  Дедлайн был: {task['deadline']}\n\n"
    
    await update.message.reply_text(text)

async def handle_payment_control_report(update, context):
    """Отчёт по контролю платежей и документов"""
    report = core.payment_control.payment_control.get_control_report()
    text = core.payment_control.payment_control.format_telegram_report(report)
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_weekly_control_summary(update, context):
    """Еженедельная сводка по контролю"""
    text = core.payment_control.payment_control.get_weekly_summary()
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_critical_alerts(update, context):
    """Критические уведомления"""
    alerts = core.payment_control.payment_control.get_critical_alerts()
    
    if not alerts:
        await update.message.reply_text("✅ Критических случаев нет!")
        return
    
    text = "🚨 <b>Критические уведомления:</b>\n\n"
    for alert in alerts:
        text += f"• {alert}\n"
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_unclosed_payments_report(update, context):
    """Отчёт по незакрытым платежам"""
    unclosed = core.payment_control.payment_control.check_unclosed_payments()
    
    if not unclosed:
        await update.message.reply_text("✅ Все платежи закрыты документами!")
        return
    
    text = f"⚠️ <b>Незакрытые платежи ({len(unclosed)}):</b>\n\n"
    
    for payment in unclosed:
        critical_mark = "🚨 " if payment['days_since_payment'] > 30 else ""
        text += f"{critical_mark}💰 {payment['amount']:,} ₽ — {payment['counterparty']}\n"
        text += f"   📅 {payment['date']} ({payment['days_since_payment']} дн. назад)\n"
        text += f"   📋 Не хватает: {', '.join(payment['missing_docs'])}\n"
        text += f"   🏢 Проект: {payment['project']}\n"
        text += f"   🔗 ID: {payment['payment_id']}\n\n"
    
    total_amount = sum(p['amount'] for p in unclosed)
    text += f"💵 <b>Общая сумма: {total_amount:,} ₽</b>"
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_orphaned_documents_report(update, context):
    """Отчёт по документам без оплаты"""
    orphaned = core.payment_control.payment_control.check_documents_without_payment()
    
    if not orphaned:
        await update.message.reply_text("✅ Все документы привязаны к платежам!")
        return
    
    text = f"⚠️ <b>Документы без оплаты ({len(orphaned)}):</b>\n\n"
    
    for doc in orphaned:
        critical_mark = "🚨 " if doc['days_since_doc'] > 30 else ""
        text += f"{critical_mark}📄 {doc['doc_type'].title()} №{doc['doc_number']}\n"
        text += f"   💰 {doc['amount']:,} ₽ — {doc['counterparty']}\n"
        text += f"   📅 {doc['doc_date']} ({doc['days_since_doc']} дн. назад)\n"
        text += f"   🔗 ID: {doc['doc_id']}\n\n"
    
    total_amount = sum(d['amount'] for d in orphaned)
    text += f"💵 <b>Общая сумма: {total_amount:,} ₽</b>"
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_inbox_summary(update, context):
    """Сводка по входящим сообщениям"""
    summary = core.inbox_monitor.inbox_monitor.get_inbox_summary()
    text = core.inbox_monitor.inbox_monitor.format_telegram_summary(summary)
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_attention_messages(update, context):
    """Сообщения, требующие внимания"""
    messages = core.inbox_monitor.inbox_monitor.get_messages_requiring_attention()
    text = core.inbox_monitor.inbox_monitor.format_attention_report(messages)
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_overdue_responses(update, context):
    """Просроченные ответы (3+ дня)"""
    messages = core.inbox_monitor.inbox_monitor.get_overdue_responses()
    text = core.inbox_monitor.inbox_monitor.format_overdue_report(messages)
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_forgotten_messages(update, context):
    """Забытые сообщения (7+ дней)"""
    messages = core.inbox_monitor.inbox_monitor.get_forgotten_messages()
    text = core.inbox_monitor.inbox_monitor.format_forgotten_report(messages)
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_reminder_suggestions(update, context):
    """Предложения для напоминаний"""
    suggestions = core.inbox_monitor.inbox_monitor.get_reminder_suggestions()
    
    if not suggestions:
        await update.message.reply_text("✅ Нет просроченных ответов для напоминаний!")
        return
    
    text = "📲 <b>Предложения для напоминаний:</b>\n\n"
    for i, suggestion in enumerate(suggestions, 1):
        text += f"{i}. {suggestion}\n\n"
    
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_mark_responded(update, context):
    """Отметка сообщения как отвеченного"""
    user_text = update.message.text.lower()
    
    # Парсим: "Отметить отвеченным [ID сообщения]"
    match = re.search(r"отметить отвеченным ([a-zA-Z0-9_-]+)", user_text)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Отметить отвеченным [ID сообщения]'\n"
            "Пример: 'Отметить отвеченным msg_20240120_143022'"
        )
        return
    
    message_id = match.group(1)
    core.inbox_monitor.inbox_monitor.mark_as_responded(message_id)
    
    await update.message.reply_text(f"✅ Сообщение {message_id} отмечено как отвеченное!")

async def handle_mark_ignored(update, context):
    """Отметка сообщения как проигнорированного"""
    user_text = update.message.text.lower()
    
    # Парсим: "Отметить проигнорированным [ID сообщения]"
    match = re.search(r"отметить проигнорированным ([a-zA-Z0-9_-]+)", user_text)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Отметить проигнорированным [ID сообщения]'\n"
            "Пример: 'Отметить проигнорированным msg_20240120_143022'"
        )
        return
    
    message_id = match.group(1)
    core.inbox_monitor.inbox_monitor.mark_as_ignored(message_id)
    
    await update.message.reply_text(f"✅ Сообщение {message_id} отмечено как проигнорированное!")

async def handle_add_message(update, context):
    """Добавление сообщения для мониторинга"""
    user_text = update.message.text.lower()
    
    # Парсим: "Добавить сообщение email Тимур КП по проекту высокий"
    match = re.search(r"добавить сообщение (\w+) ([^:]+): (.+)", user_text)
    if not match:
        await update.message.reply_text(
            "❌ Формат: 'Добавить сообщение [канал] [отправитель]: [тема]'\n"
            "Пример: 'Добавить сообщение email Тимур: КП по проекту'\n"
            "Каналы: email, telegram, crm, google_docs"
        )
        return
    
    channel, sender, subject = match.groups()
    channel = channel.strip()
    sender = sender.strip()
    subject = subject.strip()
    
    # Определяем приоритет
    priority = 'normal'
    if any(word in user_text for word in ['высокий', 'high', 'срочно', 'urgent']):
        priority = 'high'
    elif any(word in user_text for word in ['средний', 'medium']):
        priority = 'medium'
    
    message_id = core.inbox_monitor.inbox_monitor.add_message(
        channel=channel,
        sender=sender,
        subject=subject,
        content=subject,  # Используем тему как содержимое
        timestamp=datetime.now().isoformat(),
        priority=priority,
        requires_response=True
    )
    
    await update.message.reply_text(
        f"✅ Сообщение добавлено для мониторинга!\n"
        f"📝 ID: {message_id}\n"
        f"📧 Канал: {channel}\n"
        f"👤 Отправитель: {sender}\n"
        f"🎯 Приоритет: {priority}"
    )

async def handle_global_search(update, context):
    user_text = update.message.text
    import re
    # Извлекаем поисковый запрос
    m = re.search(r'(где обсуждали|найди файл|найди|поиск)\s+(.+)', user_text, re.I)
    if not m:
        await update.message.reply_text("❌ Не удалось определить поисковый запрос.")
        return
    query = m.group(2).strip()
    results = core.global_search.global_search(query)
    text = core.global_search.format_global_search_results(results, query)
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_cashflow_forecast(update, context):
    result = core.finances.cashflow_forecast()
    text = (
        f"💸 <b>Прогноз Cash Flow</b>\n\n"
        f"Текущий баланс: <b>{result['balance']:,}</b> руб.\n"
        f"Средний обязательный расход/день: <b>{result['avg_daily_expense']:,}</b> руб.\n\n"
        f"<b>{result['comment']}</b>"
    )
    await update.message.reply_text(text, parse_mode='HTML')

async def handle_digest(update, context):
    digest = core.digest.get_digest()
    text = core.digest.format_digest(digest)
    await update.message.reply_text(text, parse_mode='HTML')

# --- Новые обработчики для режима "Антиразрыв" ---
async def handle_antistress_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Предлагает действия в режиме 'Антиразрыв'."""
    chat_id = update.effective_chat.id
    
    quiet_mode_enabled = is_quiet_mode_enabled(chat_id)
    quiet_mode_text = "Выключить тихий режим" if quiet_mode_enabled else "Включить тихий режим"
    quiet_mode_callback = "antistress_quiet_off" if quiet_mode_enabled else "antistress_quiet_on"
    
    keyboard = [
        [InlineKeyboardButton("Приоритеты на сегодня", callback_data='antistress_prioritize')],
        [InlineKeyboardButton("Разгрузить календарь", callback_data='antistress_meetings')],
        [InlineKeyboardButton(quiet_mode_text, callback_data=quiet_mode_callback)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Похоже, у тебя стресс. Давай помогу. Что делаем?",
        reply_markup=reply_markup
    )

async def antistress_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатия кнопок режима 'Антиразрыв'."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if query.data == 'antistress_prioritize':
        tasks_summary = core.antistress.get_tasks_for_prioritization()
        await query.edit_message_text(text=tasks_summary, parse_mode="HTML")
        
    elif query.data == 'antistress_meetings':
        meetings = core.antistress.get_todays_meetings()
        if not meetings:
            await query.edit_message_text(text="На сегодня встреч нет. Можно расслабиться.")
            return
        
        context.user_data['meetings_to_cancel'] = [m['id'] for m in meetings]
        meetings_text = "<b>Встречи на сегодня:</b>\n"
        for meeting in meetings:
            start_time = meeting['start'].get('dateTime', meeting['start'].get('date'))
            meetings_text += f"• {meeting['summary']} ({start_time})\n"
        meetings_text += "\nОтменить все встречи на сегодня?"
        
        keyboard = [
            [InlineKeyboardButton("Да, отменить все", callback_data='antistress_cancel_confirm')],
            [InlineKeyboardButton("Нет, оставить", callback_data='antistress_cancel_decline')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=meetings_text, reply_markup=reply_markup, parse_mode="HTML")

    elif query.data == 'antistress_cancel_confirm':
        event_ids = context.user_data.get('meetings_to_cancel', [])
        if not event_ids:
            await query.edit_message_text("Не нашел, что отменять. Возможно, список устарел.")
            return
        result = core.antistress.cancel_meetings_by_ids(event_ids)
        await query.edit_message_text(f"Отменено {result['cancelled']} встреч. Календарь свободен.")
        context.user_data.pop('meetings_to_cancel', None)

    elif query.data == 'antistress_cancel_decline':
        await query.edit_message_text("Хорошо, встречи остаются в календаре.")
        context.user_data.pop('meetings_to_cancel', None)

    elif query.data == 'antistress_quiet_on':
        set_quiet_mode(chat_id, True)
        summary = core.antistress.get_critical_summary()
        await query.edit_message_text(
            "Тихий режим включен. Буду присылать только критичные уведомления.\n\n" + summary,
            parse_mode="HTML"
        )
        
    elif query.data == 'antistress_quiet_off':
        set_quiet_mode(chat_id, False)
        await query.edit_message_text("Тихий режим выключен. Уведомления возвращены.")

async def handle_meeting_prep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Готовит сводку для встречи."""
    user_text = update.message.text
    match = re.search(r"(?:подготовь инфу|подготовься к встрече|инфа для встречи) с (.+)", user_text, re.I)
    if not match:
        await update.message.reply_text("Не могу понять, с кем встреча. Укажите имя или название компании.")
        return
        
    person_name = match.group(1).strip()
    await update.message.reply_text(f"Собираю информацию по '{person_name}'... Это может занять до минуты.")
    
    try:
        # Убедимся, что модуль импортирован
        from core import meeting_prep
        report_data = meeting_prep.prepare_for_meeting(person_name)
        report_text = meeting_prep.format_meeting_prep(report_data)
        await update.message.reply_text(report_text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при подготовке отчета: {e}")

async def ask_daily_focus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Спрашивает пользователя о фокусе на день"""
    keyboard = [
        [InlineKeyboardButton("Пропустить", callback_data="focus_skip")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎯 На чём сегодня фокус?\n\n"
        "Напиши главную задачу или направление работы, "
        "и я помогу не отвлекаться на второстепенное.",
        reply_markup=reply_markup
    )

async def handle_focus_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает ответ пользователя о фокусе дня"""
    focus_text = update.message.text
    focus_manager.set_daily_focus(focus_text)
    
    await update.message.reply_text(
        f"✅ Отлично! Сегодня фокусируемся на:\n"
        f"<b>{focus_text}</b>\n\n"
        f"Я буду следить, чтобы ты не отвлекался, и напомню "
        f"если задача будет не по фокусу.",
        parse_mode='HTML'
    )

async def handle_task_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверяет, соответствует ли новая задача дневному фокусу"""
    message_text = update.message.text.lower()
    
    # Проверяем, что это похоже на добавление задачи
    if not any(keyword in message_text for keyword in ["добавить", "задача", "запланировать", "встреча"]):
        return
    
    # Парсим задачу через GPT
    task_data = await parse_task_intent(message_text)
    if not task_data or task_data.get('intent') != 'add':
        return
    
    task_text = task_data.get('task_text', '')
    if not focus_manager.is_task_in_focus(task_text):
        keyboard = [
            [
                InlineKeyboardButton("Отложить ➡️", callback_data=f"postpone_task:{task_text}"),
                InlineKeyboardButton("Всё равно добавить ✅", callback_data=f"force_add_task:{task_text}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        current_focus = focus_manager.get_daily_focus() or "не задан"
        await update.message.reply_text(
            f"⚠️ Кажется, эта задача не связана с текущим фокусом:\n"
            f"<b>{current_focus}</b>\n\n"
            f"Хочешь, отложим её на другой день?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        return False
    return True

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback-кнопок"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "focus_skip":
        await query.message.edit_text(
            "Хорошо! Сегодня работаем без строгого фокуса. "
            "Но я всё равно помогу с приоритизацией задач."
        )
    
    elif query.data.startswith("postpone_task:"):
        task_text = query.data.replace("postpone_task:", "")
        keyboard = [
            [
                InlineKeyboardButton("Завтра", callback_data=f"set_task_date:{task_text}:tomorrow"),
                InlineKeyboardButton("Послезавтра", callback_data=f"set_task_date:{task_text}:day_after")
            ],
            [
                InlineKeyboardButton("Через неделю", callback_data=f"set_task_date:{task_text}:next_week"),
                InlineKeyboardButton("Выбрать дату", callback_data=f"choose_date:{task_text}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            f"📅 Выбери, на когда перенести задачу:\n"
            f"<b>{task_text}</b>",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    
    elif query.data.startswith("set_task_date:"):
        _, task_text, when = query.data.split(":", 2)
        today = datetime.now().date()
        
        if when == "tomorrow":
            new_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif when == "day_after":
            new_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        elif when == "next_week":
            new_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Добавляем задачу на новую дату
        add_task(task_text, new_date)
        
        await query.message.edit_text(
            f"✅ Задача перенесена на {new_date}:\n"
            f"<b>{task_text}</b>",
            parse_mode='HTML'
        )
    
    elif query.data.startswith("force_add_task:"):
        task_text = query.data.replace("force_add_task:", "")
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Добавляем задачу на сегодня
        add_task(task_text, today)
        
        await query.message.edit_text(
            f"✅ Задача добавлена на сегодня:\n"
            f"<b>{task_text}</b>",
            parse_mode='HTML'
        )
    
    # ... остальные обработчики callback ...

async def initialize_scheduler(app):
    """Инициализация планировщика задач"""
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Europe/Moscow'))
    
    # Запрашиваем фокус дня в 9:00
    scheduler.add_job(
        ask_daily_focus_to_all_chats,
        'cron',
        hour=9,
        minute=0,
        args=[app]
    )
    
    scheduler.start()

async def ask_daily_focus_to_all_chats(app):
    """Отправляет запрос фокуса дня во все активные чаты"""
    chat_ids = load_active_chat_ids()
    for chat_id in chat_ids:
        if not is_quiet_mode_enabled(chat_id):
            try:
                keyboard = [
                    [InlineKeyboardButton("Пропустить", callback_data="focus_skip")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await app.bot.send_message(
                    chat_id=chat_id,
                    text="🎯 Доброе утро! На чём сегодня фокус?\n\n"
                         "Напиши главную задачу или направление работы, "
                         "и я помогу не отвлекаться на второстепенное.",
                    reply_markup=reply_markup
                )
            except Exception as e:
                print(f"Ошибка отправки запроса фокуса в чат {chat_id}: {e}")

async def handle_on_site_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команд режима 'На объекте'"""
    user_text = update.message.text.lower()
    
    # Активация режима
    if re.search(r"(я на объекте|на стройке|на выезде|не трогай лишним)", user_text):
        # Пытаемся найти название проекта
        project_match = re.search(r"проект[а-я]* [\«\"]?([^\»\"\n]+)[\»\"]?", user_text)
        project = project_match.group(1) if project_match else None
        
        if not project:
            # Спрашиваем проект
            keyboard = []
            # Получаем список активных проектов
            projects = finances.get_active_projects()
            for i in range(0, len(projects), 2):
                row = [
                    InlineKeyboardButton(projects[i], callback_data=f"on_site_project:{projects[i]}")
                ]
                if i + 1 < len(projects):
                    row.append(InlineKeyboardButton(projects[i+1], callback_data=f"on_site_project:{projects[i+1]}"))
                keyboard.append(row)
            keyboard.append([InlineKeyboardButton("❌ Без проекта", callback_data="on_site_project:none")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "🏗️ Включаю режим «На объекте»\n\n"
                "Выбери проект, чтобы я фильтровал уведомления:",
                reply_markup=reply_markup
            )
        else:
            # Активируем режим с указанным проектом
            mode_info = work_mode_manager.activate_on_site_mode(project)
            await update.message.reply_text(
                f"🏗️ Включен режим «На объекте»\n"
                f"Проект: <b>{project}</b>\n\n"
                f"Буду беспокоить только по критичным вопросам.\n"
                f"Чтобы выключить режим, напиши: <i>я в офисе</i>",
                parse_mode='HTML'
            )
        return True
    
    # Деактивация режима
    if re.search(r"(я в офисе|вернулся|закончил|выключи режим)", user_text):
        if work_mode_manager.deactivate_mode():
            # Отправляем накопленные уведомления
            await notification_manager.send_queued_notifications(update.effective_chat.id)
            await update.message.reply_text(
                "✅ Режим «На объекте» выключен.\n"
                "Возвращаюсь к обычному режиму работы."
            )
        return True
    
    return False

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback-кнопок"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("on_site_project:"):
        project = query.data.replace("on_site_project:", "")
        project = None if project == "none" else project
        
        mode_info = work_mode_manager.activate_on_site_mode(project)
        project_text = f"Проект: <b>{project}</b>" if project else "Без привязки к проекту"
        
        await query.message.edit_text(
            f"🏗️ Включен режим «На объекте»\n"
            f"{project_text}\n\n"
            f"Буду беспокоить только по критичным вопросам.\n"
            f"Чтобы выключить режим, напиши: <i>я в офисе</i>",
            parse_mode='HTML'
        )
        return
    
    # ... остальные обработчики callback ...

async def send_notification(chat_id: int, text: str, priority: str = 'normal', project: Optional[str] = None, **kwargs):
    """Отправка уведомления с учетом режима работы"""
    if not work_mode_manager.should_notify(priority, project):
        print(f"[DEBUG] Уведомление пропущено из-за режима работы: {text}")
        return
    
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, **kwargs)
    except Exception as e:
        print(f"Ошибка отправки уведомления: {e}")

async def handle_document_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик поиска документов"""
    user_text = update.message.text.lower()
    
    # Поиск по имени и дате
    if "найди" in user_text or "покажи" in user_text:
        # Извлекаем параметры поиска
        doc_type = None
        date_from = None
        date_to = None
        has_signatures = None
        
        # Определяем тип документа
        if "акт" in user_text:
            doc_type = "акт"
        elif "счет" in user_text or "счёт" in user_text:
            doc_type = "счет"
        elif "договор" in user_text:
            doc_type = "договор"
        elif "накладная" in user_text:
            doc_type = "накладная"
        
        # Ищем упоминание подписей
        if "без подписи" in user_text or "неподписанные" in user_text:
            has_signatures = False
        elif "подписанные" in user_text:
            has_signatures = True
        
        # Ищем даты
        months = {
            'январ': '01', 'феврал': '02', 'март': '03', 'апрел': '04',
            'май': '05', 'мая': '05', 'июн': '06', 'июл': '07',
            'август': '08', 'сентябр': '09', 'октябр': '10',
            'ноябр': '11', 'декабр': '12'
        }
        
        # Поиск месяца
        for month_name, month_num in months.items():
            if month_name in user_text:
                # Ищем год
                year_match = re.search(r'20\d{2}', user_text)
                year = year_match.group(0) if year_match else str(datetime.now().year)
                
                date_from = f"{year}-{month_num}-01"
                # Определяем последний день месяца
                last_day = "31" if month_num in ['01', '03', '05', '07', '08', '10', '12'] else \
                          "30" if month_num in ['04', '06', '09', '11'] else \
                          "29" if month_num == "02" and int(year) % 4 == 0 else "28"
                date_to = f"{year}-{month_num}-{last_day}"
                break
        
        # Поиск документов
        docs = document_assistant.search_documents(
            query=user_text,
            doc_type=doc_type,
            date_from=date_from,
            date_to=date_to,
            has_signatures=has_signatures
        )
        
        if not docs:
            await update.message.reply_text(
                "🔍 Не нашел документов по вашему запросу.\n"
                "Попробуйте уточнить поиск или изменить критерии."
            )
            return
        
        # Формируем ответ
        response = "📄 Найденные документы:\n\n"
        for i, doc in enumerate(docs[:10], 1):
            meta = doc['metadata']
            date_str = f" от {meta['date']}" if meta.get('date') else ""
            number_str = f" №{meta['number']}" if meta.get('number') else ""
            amount_str = f"\nСумма: {meta['amount']} руб." if meta.get('amount') else ""
            
            signature_status = "✅" if doc['has_signatures'] else "❌"
            
            response += (
                f"{i}. {doc['type'].capitalize()}{number_str}{date_str}\n"
                f"📎 {doc['filename']}\n"
                f"✍️ Подписи: {signature_status}{amount_str}\n"
                f"🏷️ Теги: {', '.join(doc['tags']) if doc['tags'] else 'нет'}\n\n"
            )
        
        if len(docs) > 10:
            response += f"... и еще {len(docs) - 10} документов"
        
        # Добавляем кнопки действий
        keyboard = []
        if len(docs) == 1:
            # Для одного документа - полный набор действий
            keyboard = [
                [
                    InlineKeyboardButton("📤 Отправить файл", callback_data=f"send_doc:{docs[0]['id']}"),
                    InlineKeyboardButton("📝 Редактировать", callback_data=f"edit_doc:{docs[0]['id']}")
                ],
                [
                    InlineKeyboardButton("🔗 Связать с задачей", callback_data=f"link_doc:{docs[0]['id']}"),
                    InlineKeyboardButton("🏷️ Добавить теги", callback_data=f"tag_doc:{docs[0]['id']}")
                ]
            ]
        else:
            # Для списка - только отправка файлов
            keyboard = [
                [InlineKeyboardButton(f"📤 Документ {i}", callback_data=f"send_doc:{doc['id']}")]
                for i, doc in enumerate(docs[:5], 1)
            ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(response, reply_markup=reply_markup)
        return

async def handle_document_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback-кнопок для документов"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("send_doc:"):
        doc_id = query.data.replace("send_doc:", "")
        doc = document_assistant.get_document_by_id(doc_id)
        if doc:
            try:
                with open(doc['path'], 'rb') as file:
                    await query.message.reply_document(
                        document=file,
                        filename=doc['filename'],
                        caption=f"📄 {doc['type'].capitalize()}\n"
                                f"Добавлен: {doc['added_at'][:10]}"
                    )
            except Exception as e:
                await query.message.reply_text(f"Ошибка при отправке файла: {e}")
    
    elif query.data.startswith("edit_doc:"):
        doc_id = query.data.replace("edit_doc:", "")
        # Сохраняем ID документа для редактирования
        context.user_data['editing_doc'] = doc_id
        keyboard = [
            [
                InlineKeyboardButton("📝 Тип документа", callback_data=f"edit_doc_type:{doc_id}"),
                InlineKeyboardButton("📅 Дата", callback_data=f"edit_doc_date:{doc_id}")
            ],
            [
                InlineKeyboardButton("🔢 Номер", callback_data=f"edit_doc_number:{doc_id}"),
                InlineKeyboardButton("💰 Сумма", callback_data=f"edit_doc_amount:{doc_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.edit_text(
            "Выберите, что хотите отредактировать:",
            reply_markup=reply_markup
        )
    
    elif query.data.startswith("link_doc:"):
        doc_id = query.data.replace("link_doc:", "")
        # TODO: Показать список активных задач для связывания
        await query.message.edit_text(
            "🔄 Функция связывания с задачами в разработке"
        )
    
    elif query.data.startswith("tag_doc:"):
        doc_id = query.data.replace("tag_doc:", "")
        context.user_data['tagging_doc'] = doc_id
        await query.message.edit_text(
            "🏷️ Отправьте теги через запятую или пробел.\n"
            "Например: договор важное срочное"
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    # Обработка тегов для документа
    if 'tagging_doc' in context.user_data:
        doc_id = context.user_data.pop('tagging_doc')
        tags = [tag.strip() for tag in re.split(r'[,\s]+', user_text) if tag.strip()]
        doc = document_assistant.update_document(doc_id, {'tags': tags})
        if doc:
            await update.message.reply_text(
                f"✅ Теги обновлены:\n"
                f"🏷️ {', '.join(tags)}"
            )
        return
    
    # Обработка поиска документов
    if await handle_document_search(update, context):
        return
    
    # ... остальные обработчики ...

from core.speech_synthesizer import speech_synthesizer

async def handle_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команд озвучки текста"""
    user_text = update.message.text.lower()
    
    # Проверяем, что это команда озвучки
    voice_commands = ["озвучь", "проговори", "прочитай", "скажи"]
    if not any(cmd in user_text for cmd in voice_commands):
        return False
    
    # Определяем, что нужно озвучить
    text_to_speak = None
    voice_params = {
        "rate": "+0%",
        "volume": "+0%"
    }
    
    # Проверяем ключевые слова для разных типов контента
    if "сводку" in user_text or "отчет" in user_text or "отчёт" in user_text:
        if "утренн" in user_text:
            text_to_speak = await get_morning_summary()
        elif "финанс" in user_text:
            text_to_speak = await get_financial_summary()
        elif "задач" in user_text:
            text_to_speak = await get_tasks_summary()
    
    # Если не нашли специальный контент, проверяем предыдущее сообщение
    if not text_to_speak and update.message.reply_to_message:
        text_to_speak = update.message.reply_to_message.text
    
    if not text_to_speak:
        await update.message.reply_text(
            "🎙 Не могу понять, что нужно озвучить. Пожалуйста:\n"
            "1. Ответьте на сообщение, которое нужно озвучить\n"
            "2. Или укажите тип сводки (утренняя/финансы/задачи)"
        )
        return True
    
    # Проверяем параметры голоса в запросе
    if "быстр" in user_text:
        voice_params["rate"] = "+30%"
    elif "медленн" in user_text:
        voice_params["rate"] = "-30%"
    
    if "громч" in user_text:
        voice_params["volume"] = "+30%"
    elif "тих" in user_text:
        voice_params["volume"] = "-30%"
    
    # Выбор голоса
    voice = None
    if "мужск" in user_text or "дмитри" in user_text:
        voice = "ru-RU-DmitryNeural"
    elif "женск" in user_text or "светлан" in user_text:
        voice = "ru-RU-SvetlanaNeural"
    
    # Подготавливаем текст для озвучки
    prepared_text = speech_synthesizer.prepare_text_for_tts(text_to_speak)
    
    try:
        # Отправляем сообщение о начале генерации
        status_message = await update.message.reply_text(
            "🎙 Генерирую аудио...",
            reply_to_message_id=update.message.message_id
        )
        
        # Генерируем аудио
        audio_path = await speech_synthesizer.text_to_speech(
            prepared_text,
            voice=voice,
            rate=voice_params["rate"],
            volume=voice_params["volume"]
        )
        
        # Отправляем аудио
        with open(audio_path, 'rb') as audio:
            await update.message.reply_voice(
                voice=audio,
                reply_to_message_id=update.message.message_id,
                caption="🎙 Вот ваше аудио"
            )
        
        # Удаляем статусное сообщение
        await status_message.delete()
        
        # Запускаем очистку старых файлов
        speech_synthesizer.cleanup_temp_files()
        
    except Exception as e:
        await status_message.edit_text(
            f"❌ Не удалось создать аудио: {str(e)}"
        )
    
    return True

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик всех текстовых сообщений"""
    
    # Сначала проверяем команды озвучки
    if await handle_voice_command(update, context):
        return
    
    # ... остальные обработчики ...

async def handle_mail_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Почтовая сводка с поддержкой выбора ящика"""
    # Получаем список ящиков из oauth_settings
    try:
        with open("email_oauth_settings.json", "r", encoding="utf-8") as f:
            oauth_settings = json.load(f)
        mailboxes = list(oauth_settings.keys())
    except Exception:
        mailboxes = ["gmail"]
    
    # Если несколько ящиков — предлагаем выбрать
    if len(mailboxes) > 1 and not context.user_data.get('selected_mailbox'):
        keyboard = [[InlineKeyboardButton(box, callback_data=f"mailbox:{box}")] for box in mailboxes]
        await update.message.reply_text(
            "Выберите почтовый ящик:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
    
    # Определяем ящик
    mailbox = context.user_data.get('selected_mailbox', mailboxes[0])
    # Загружаем новые письма
    new_count = inbox_monitor.fetch_emails_via_oauth(provider=mailbox)
    # Формируем сводку
    summary = inbox_monitor.get_inbox_summary()
    text = inbox_monitor.format_telegram_summary(summary)
    await update.message.reply_text(
        f"📬 Почтовая сводка ({mailbox}):\n\n{text}"
    )
    # Кнопки для писем, требующих действий
    actionable = summary.get('action_required', [])
    if actionable:
        for msg in actionable[:5]:
            keyboard = [
                [
                    InlineKeyboardButton("Ответить", callback_data=f"mail_reply:{msg['id']}"),
                    InlineKeyboardButton("Игнорировать", callback_data=f"mail_ignore:{msg['id']}")
                ]
            ]
            preview = f"✉️ {msg['subject']}\nОт: {msg['sender']}\n{msg['content'][:100]}..."
            await update.message.reply_text(preview, reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_mailbox_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора почтового ящика"""
    query = update.callback_query
    await query.answer()
    mailbox = query.data.replace("mailbox:", "")
    context.user_data['selected_mailbox'] = mailbox
    # Перезапускаем сводку для выбранного ящика
    class DummyMsg:
        def __init__(self, chat_id):
            self.chat_id = chat_id
            self.message_id = None
    update.message = DummyMsg(query.message.chat_id)
    await handle_mail_summary(update, context)

async def handle_mail_reply_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка кнопки 'Ответить' — предлагает шаблон ответа"""
    query = update.callback_query
    await query.answer()
    msg_id = query.data.replace("mail_reply:", "")
    reply_text = inbox_monitor.suggest_reply(msg_id)
    context.user_data['replying_to_mail'] = msg_id
    keyboard = [
        [InlineKeyboardButton("Отправить этот ответ", callback_data=f"mail_send:{msg_id}")],
        [InlineKeyboardButton("Ввести свой ответ", callback_data=f"mail_custom:{msg_id}")]
    ]
    await query.message.reply_text(
        f"💬 Шаблон ответа:\n\n{reply_text}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_mail_ignore_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка кнопки 'Игнорировать'"""
    query = update.callback_query
    await query.answer()
    msg_id = query.data.replace("mail_ignore:", "")
    inbox_monitor.mark_as_ignored(msg_id)
    await query.message.reply_text("Письмо отмечено как проигнорированное.")

async def handle_mail_send_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправка шаблонного ответа (заглушка)"""
    query = update.callback_query
    await query.answer()
    msg_id = query.data.replace("mail_send:", "")
    # Здесь должна быть интеграция с SMTP для отправки письма
    inbox_monitor.mark_as_responded(msg_id)
    await query.message.reply_text("✅ Ответ отправлен (заглушка)")

async def handle_mail_custom_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь вводит свой ответ"""
    query = update.callback_query
    await query.answer()
    msg_id = query.data.replace("mail_custom:", "")
    context.user_data['replying_to_mail'] = msg_id
    await query.message.reply_text("Введите свой вариант ответа на письмо:")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... существующие обработчики ...
    # Проверка на режим ответа на письмо
    if 'replying_to_mail' in context.user_data:
        msg_id = context.user_data.pop('replying_to_mail')
        # Здесь должна быть интеграция с SMTP для отправки письма
        inbox_monitor.mark_as_responded(msg_id)
        await update.message.reply_text("✅ Ваш ответ отправлен (заглушка)")
        return
    # ... остальные обработчики ...

from core.deadline_monitor import deadline_monitor

async def handle_deadline_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка дедлайнов и рисковых задач"""
    report = deadline_monitor.format_risk_report()
    await update.message.reply_text(report)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.lower()
    # ... существующие обработчики ...
    if any(cmd in user_text for cmd in ["проверь дедлайны", "что под угрозой", "дедлайн", "риск просрочки"]):
        await handle_deadline_check(update, context)
        return
    # ... остальные обработчики ...

from core.meeting_assistant import meeting_assistant

async def handle_meeting_assistant(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ассистент на звонке: аудио или текст, summary, действия, задачи"""
    # Если аудиофайл
    if update.message.voice or update.message.audio or update.message.document:
        file = update.message.voice or update.message.audio or update.message.document
        file_obj = await file.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as tmp:
            await file_obj.download_to_drive(tmp.name)
            audio_path = tmp.name
        text = meeting_assistant.transcribe_audio(audio_path)
        os.remove(audio_path)
    else:
        text = update.message.text
    # Анализируем встречу
    result = meeting_assistant.analyze_meeting_text(text)
    summary = result.get('summary', '')
    actions = result.get('actions', [])
    tasks = result.get('tasks', [])
    # Формируем ответ
    response = f"📝 Краткие заметки:\n{summary}\n\n"
    if actions:
        response += "💡 Действия:\n" + "\n".join(f"— {a}" for a in actions) + "\n\n"
    if tasks:
        response += "📋 Задачи:\n" + "\n".join(f"• {t}" for t in tasks)
    await update.message.reply_text(response)
    # Предложить добавить задачи
    if tasks:
        keyboard = [[InlineKeyboardButton("Добавить задачи", callback_data="meeting_add_tasks")]]
        context.user_data['meeting_tasks_to_add'] = tasks
        await update.message.reply_text("Добавить эти задачи в список?", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_meeting_add_tasks_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    tasks = context.user_data.pop('meeting_tasks_to_add', [])
    meeting_assistant.add_action_tasks(tasks)
    await query.message.reply_text("✅ Задачи добавлены!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.lower() if update.message.text else ''
    # ... существующие обработчики ...
    if any(cmd in user_text for cmd in ["ассистент на звонке", "заметки встречи", "итоги звонка", "итоги встречи"]):
        await handle_meeting_assistant(update, context)
        return
    # Если аудиофайл с пометкой "встреча"
    if (update.message.voice or update.message.audio or update.message.document) and (
        update.message.caption and "встреча" in update.message.caption.lower()):
        await handle_meeting_assistant(update, context)
        return
    # ... остальные обработчики ...

def run_bot():
    import asyncio
    from telegram.ext import ApplicationBuilder, MessageHandler, filters, CallbackQueryHandler
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Основной обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    # Callback-кнопки
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    # Голосовые сообщения
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    # Документы и аудио (например, для встреч)
    app.add_handler(MessageHandler(filters.Document.ALL | filters.AUDIO, handle_meeting_assistant))

    # Планировщик и polling календаря
    start_scheduler(app)
    start_calendar_polling(app)

    # Запуск бота
    print("[TelegramBot] Запуск polling...")
    asyncio.run(app.run_polling())