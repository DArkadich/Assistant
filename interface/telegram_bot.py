import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
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
from core.rag_system import rag_system

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
SMART_MODEL = os.getenv("OPENAI_SMART_MODEL", "gpt-4-1106-preview")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
    # Задачи
    tasks = calendar.get_daily_plan(today)
    if tasks:
        tasks_text = "\n".join([f"- {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
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
    summary = f"🗓️ План на сегодня:\n{tasks_text}\n\n🎯 Цели:\n{goals_text}\n\n💰 Финансы за сегодня:\n{finance_text}"
    await update.message.reply_text(summary, parse_mode='HTML')

async def send_weekly_summary(update: Update):
    from core import calendar, planner, finances
    import pytz
    today = datetime.now(pytz.timezone('Europe/Moscow')).date()
    week_dates = [(today + timedelta(days=i)) for i in range(7)]
    week_strs = [d.strftime('%Y-%m-%d') for d in week_dates]
    week_days = [d.strftime('%A, %d %B').capitalize() for d in week_dates]

    # 1. Группировка задач по дням недели (только невыполненные)
    tasks_by_day = {d: [] for d in week_strs}
    for d in week_strs:
        day_tasks = [t for t in calendar.get_daily_plan(d) if not t.get('done')]
        tasks_by_day[d] = day_tasks

    # 2. Дедлайны на неделе (по задачам и целям)
    deadlines = []
    # Задачи с дедлайном на неделе
    for d in week_strs:
        for t in tasks_by_day[d]:
            if not t.get('from_google_calendar'):
                deadlines.append((d, t['task_text']))
    # Цели с дедлайном на неделе
    for goal in planner.get_goals():
        deadline = goal.get('deadline')
        if deadline and deadline in week_strs:
            deadlines.append((deadline, f"Цель: {goal['goal_text']}"))

    # 3. Важные события из Google Calendar
    events = []
    for d in week_strs:
        for t in tasks_by_day[d]:
            if t.get('from_google_calendar'):
                events.append((d, t['task_text'], t.get('time')))

    # 4. Цели и аналитика
    goals = planner.get_goals()
    goals_text = []
    for goal in goals:
        deadline = goal.get('deadline')
        progress = goal.get('progress', 0)
        days_left = (datetime.strptime(deadline, '%Y-%m-%d').date() - today).days if deadline else None
        tasks_left = 0
        if hasattr(planner, 'get_goal_tasks'):
            tasks_left = len([t for t in planner.get_goal_tasks(goal['goal_text']) if not t.get('done')])
        goals_text.append(f"- {goal['goal_text']} — {progress}% (до {deadline or '—'}, осталось {days_left if days_left is not None else '?'} дн., {tasks_left} задач)")
    if not goals_text:
        goals_text = ["Нет целей."]

    # 5. Финансы за месяц
    period = today.strftime("%Y-%m")
    report = finances.get_report(period=period)
    finance_text = f"Доход: {report['income']}, Расход: {report['expense']}, Прибыль: {report['profit']}"
    # --- Чистый остаток общий ---
    total_balance = finances.get_total_balance()
    finance_text += f"\nЧистый остаток: {total_balance}"
    # --- Разметка по проектам ---
    project_reports = finances.get_report_by_project(period=period)
    if project_reports:
        finance_text += "\n\n<b>По проектам:</b>"
        for project, rep in project_reports.items():
            balance = finances.get_total_balance(project=project)
            finance_text += f"\n- {project}: Доход {rep['income']}, Расход {rep['expense']}, Прибыль {rep['profit']}, Остаток {balance}"

    # Формируем итоговый текст
    summary = "🗓️ <b>План на неделю</b>\n"
    for i, d in enumerate(week_strs):
        day_header = f"<b>{week_days[i]}</b>"
        day_tasks = tasks_by_day[d]
        if day_tasks:
            summary += f"\n{day_header}:\n"
            for t in day_tasks:
                time = t.get('time')
                summary += f"- [{time or '--:--'}] {t['task_text']}\n"
    # Дедлайны
    if deadlines:
        summary += "\n⏰ <b>Дедлайны на неделе:</b>\n"
        for d, text in deadlines:
            summary += f"- {d}: {text}\n"
    # События
    if events:
        summary += "\n📅 <b>События:</b>\n"
        for d, text, time in events:
            summary += f"- {d} {time or ''}: {text}\n"
    # Цели
    summary += "\n🎯 <b>Цели и аналитика:</b>\n" + "\n".join(goals_text)
    # Финансы
    summary += f"\n\n💰 <b>Финансы за {today.strftime('%B')}:</b>\n{finance_text}"

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
    print(f"[DEBUG] Получено сообщение: {user_text}")
    save_last_chat_id(update.effective_chat.id)

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
                if act['payment_ids']:
                    payment = finances.find_payment_by_id(act['payment_ids'][0])
                    if payment:
                        text += f"   Платёж: {payment['counterparty']} ({payment['amount']} руб.)\n"
                        text += f"   Проект: {payment['project']}\n"
                        text += f"   Направление: {'входящий' if payment['direction'] == 'in' else 'исходящий'}\n"
                else:
                    text += f"   Платёж: не привязан\n"
                if act.get('file_url'):
                    text += f"   📎 Файл: {act['file_url']}\n"
            
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
    
    # Если не задача и не финансы — fallback на GPT-ответ
    reply = await ask_openai(user_text)
    await update.message.reply_text(reply)

def extract_date_phrase_for_finance(text):
    import re
    patterns = [
        r"вчера", r"сегодня", r"завтра", r"позавчера", r"послезавтра",
        r"\d{1,2} [а-я]+", r"\d{1,2}\.\d{1,2}\.\d{2,4}", r"\d{4}-\d{2}-\d{2}",
        r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    start_scheduler(app)
    start_calendar_polling(app)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

# --- RAG функции ---
async def handle_rag_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка семантического поиска документов."""
    user_text = update.message.text
    
    # Извлекаем запрос для поиска
    query_match = re.search(r"(найди документ|поиск документов|семантический поиск|rag поиск)\s+(.+)", user_text, re.I)
    if not query_match:
        await update.message.reply_text(
            "🔍 <b>Семантический поиск документов</b>\n\n"
            "Используйте:\n"
            "• 'Найди документ [запрос]' - поиск по содержимому\n"
            "• 'Поиск документов [запрос]' - семантический поиск\n"
            "• 'RAG поиск [запрос]' - векторный поиск\n\n"
            "Примеры:\n"
            "• Найди документ про оплату услуг\n"
            "• Поиск документов контракт разработка\n"
            "• RAG поиск накладная поставка товаров",
            parse_mode='HTML'
        )
        return
    
    query = query_match.group(2).strip()
    
    try:
        # Выполняем поиск
        results = rag_system.search_documents(query, n_results=5)
        
        if not results:
            await update.message.reply_text(f"🔍 По запросу '{query}' документы не найдены.")
            return
        
        # Формируем ответ
        text = f"🔍 <b>Результаты поиска по запросу '{query}':</b>\n\n"
        
        for i, doc in enumerate(results, 1):
            metadata = doc.get('metadata', {})
            distance = doc.get('distance', 0)
            # Более точный расчет релевантности (distance обычно от 0 до 2)
            relevance = max(0, min(100, int((1 - distance) * 100)))
            
            text += f"📋 <b>{i}. {metadata.get('type', 'Документ').title()}</b>\n"
            text += f"   Контрагент: {metadata.get('counterparty_name', 'Не указан')}\n"
            text += f"   Сумма: {metadata.get('amount', 'Не указана')} руб.\n"
            text += f"   Дата: {metadata.get('date', 'Не указана')}\n"
            text += f"   Релевантность: {relevance}%\n"
            text += f"   ID: {doc['id']}\n\n"
        
        await update.message.reply_text(text, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")

async def handle_search_by_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка поиска документов по типу."""
    user_text = update.message.text
    
    # Извлекаем тип документа
    type_match = re.search(r"(найди по типу|поиск по типу|документы типа)\s+(накладная|упд|гтд|счёт|контракт|акт)(?:\s+(.+))?", user_text, re.I)
    if not type_match:
        await update.message.reply_text(
            "📋 <b>Поиск документов по типу</b>\n\n"
            "Используйте:\n"
            "• 'Найди по типу [тип] [запрос]' - поиск документов определенного типа\n"
            "• 'Поиск по типу [тип] [запрос]' - семантический поиск по типу\n\n"
            "Типы документов:\n"
            "• накладная, упд, гтд, счёт, контракт, акт\n\n"
            "Примеры:\n"
            "• Найди по типу контракт разработка\n"
            "• Поиск по типу накладная поставка",
            parse_mode='HTML'
        )
        return
    
    doc_type = type_match.group(2).lower()
    query = type_match.group(3).strip() if type_match.group(3) else ""
    
    try:
        # Выполняем поиск по типу
        results = rag_system.search_by_type(doc_type, query, n_results=5)
        
        if not results:
            type_text = f" типа '{doc_type}'"
            query_text = f" по запросу '{query}'" if query else ""
            await update.message.reply_text(f"📋 Документы{type_text}{query_text} не найдены.")
            return
        
        # Формируем ответ
        text = f"📋 <b>Документы типа '{doc_type}'"
        if query:
            text += f" по запросу '{query}'"
        text += f":</b>\n\n"
        
        for i, doc in enumerate(results, 1):
            metadata = doc.get('metadata', {})
            distance = doc.get('distance', 0)
            # Более точный расчет релевантности (distance обычно от 0 до 2)
            relevance = max(0, min(100, int((1 - distance) * 100)))
            
            text += f"📄 <b>{i}. {metadata.get('type', 'Документ').title()}</b>\n"
            text += f"   Контрагент: {metadata.get('counterparty_name', 'Не указан')}\n"
            text += f"   Сумма: {metadata.get('amount', 'Не указана')} руб.\n"
            text += f"   Дата: {metadata.get('date', 'Не указана')}\n"
            text += f"   Релевантность: {relevance}%\n"
            text += f"   ID: {doc['id']}\n\n"
        
        await update.message.reply_text(text, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")

async def handle_search_by_counterparty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка поиска документов по контрагенту."""
    user_text = update.message.text
    
    # Извлекаем название контрагента и запрос
    counterparty_match = re.search(r"(найди по контрагенту|поиск по контрагенту|документы контрагента)\s+([а-яёa-z0-9\s]+?)(?:\s+(.+))?$", user_text, re.I)
    if not counterparty_match:
        await update.message.reply_text(
            "🏢 <b>Поиск документов по контрагенту</b>\n\n"
            "Используйте:\n"
            "• 'Найди по контрагенту [название] [запрос]' - поиск документов контрагента\n"
            "• 'Поиск по контрагенту [название] [запрос]' - семантический поиск\n\n"
            "Примеры:\n"
            "• Найди по контрагенту ООО Рога и Копыта\n"
            "• Поиск по контрагенту ИП Иванов контракт",
            parse_mode='HTML'
        )
        return
    
    counterparty = counterparty_match.group(2).strip()
    query = counterparty_match.group(3).strip() if counterparty_match.group(3) else ""
    
    try:
        # Выполняем поиск по контрагенту
        results = rag_system.search_by_counterparty(counterparty, query, n_results=5)
        
        if not results:
            query_text = f" по запросу '{query}'" if query else ""
            await update.message.reply_text(f"🏢 Документы контрагента '{counterparty}'{query_text} не найдены.")
            return
        
        # Формируем ответ
        text = f"🏢 <b>Документы контрагента '{counterparty}'"
        if query:
            text += f" по запросу '{query}'"
        text += f":</b>\n\n"
        
        for i, doc in enumerate(results, 1):
            metadata = doc.get('metadata', {})
            distance = doc.get('distance', 0)
            # Более точный расчет релевантности (distance обычно от 0 до 2)
            relevance = max(0, min(100, int((1 - distance) * 100)))
            
            text += f"📄 <b>{i}. {metadata.get('type', 'Документ').title()}</b>\n"
            text += f"   Сумма: {metadata.get('amount', 'Не указана')} руб.\n"
            text += f"   Дата: {metadata.get('date', 'Не указана')}\n"
            text += f"   Проект: {metadata.get('project', 'Не указан')}\n"
            text += f"   Релевантность: {relevance}%\n"
            text += f"   ID: {doc['id']}\n\n"
        
        await update.message.reply_text(text, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")

async def handle_rag_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка запроса статистики RAG системы."""
    try:
        stats = rag_system.get_collection_stats()
        
        if 'error' in stats:
            await update.message.reply_text(f"❌ Ошибка получения статистики: {stats['error']}")
            return
        
        text = f"📊 <b>Статистика RAG системы:</b>\n\n"
        text += f"📄 Всего документов: {stats['total_documents']}\n"
        text += f"📁 Коллекция: {stats['collection_name']}\n"
        text += f"🟢 Статус: {stats['status']}\n\n"
        
        if stats['total_documents'] > 0:
            text += f"✅ RAG система активна и готова к поиску\n"
            text += f"🔍 Используйте команды:\n"
            text += f"• 'Найди документ [запрос]'\n"
            text += f"• 'Найди по типу [тип] [запрос]'\n"
            text += f"• 'Найди по контрагенту [название] [запрос]'"
        else:
            text += f"⚠️ Коллекция пуста\n"
            text += f"📝 Добавьте документы для активации поиска"
        
        await update.message.reply_text(text, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")