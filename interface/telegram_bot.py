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
    # Итог
    summary = f"🗓️ План на сегодня:\n{tasks_text}\n\n🎯 Цели:\n{goals_text}\n\n💰 Финансы за сегодня:\n{finance_text}"
    await update.message.reply_text(summary)

async def send_weekly_summary(update: Update):
    today = datetime.now().date()
    week = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    # Задачи
    tasks = calendar.get_week_plan(today.strftime("%Y-%m-%d"))
    if tasks:
        tasks_text = "\n".join([f"- {t['task_text']} ({t['date']}) {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
    else:
        tasks_text = "Нет задач."
    # Цели
    goals = planner.get_goals()
    if goals:
        goals_text = "\n".join([f"- {g['goal_text']} — {g['progress']}% (до {g['deadline']})" for g in goals])
    else:
        goals_text = "Нет целей."
    # Финансы (за неделю)
    period = today.strftime("%Y-%m-%d")[:7]  # ГГГГ-ММ
    report = finances.get_report(period=period)
    finance_text = f"Доход: {report['income']}, Расход: {report['expense']}, Прибыль: {report['profit']}"
    # Итог
    summary = f"🗓️ План на неделю:\n{tasks_text}\n\n🎯 Цели:\n{goals_text}\n\n💰 Финансы за месяц:\n{finance_text}"
    await update.message.reply_text(summary)

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
        loop = asyncio.get_event_loop()
        if new_events:
            for summary, start in new_events:
                loop.create_task(app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Новое событие: {summary} ({start})"))
        if changed_events:
            for eid in changed_events:
                summary, start = event_map[eid]
                loop.create_task(app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Изменено событие: {summary} ({start})"))
        if deleted_events:
            for eid in deleted_events:
                summary, start = last_polled_events[eid]
                loop.create_task(app.bot.send_message(chat_id=chat_id, text=f"[Календарь] Удалено событие: {summary} ({start})"))
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
    # --- Финансы через естественный язык ---
    fin_intent = await parse_finance_intent(user_text)
    if fin_intent:
        intent = fin_intent.get("intent")
        if intent in ("income", "expense"):
            import dateparser
            date_from_gpt = fin_intent.get("date")
            date_phrase = extract_date_phrase_for_finance(user_text)
            date_from_text = dateparser.parse(date_phrase, languages=['ru']) if date_phrase else None
            if date_from_text:
                date_from_text = date_from_text.strftime('%Y-%m-%d')
            print(f"[DEBUG] date_from_gpt: {date_from_gpt}, date_phrase: {date_phrase}, date_from_text: {date_from_text}", flush=True)
            # Если dateparser распознал дату — всегда используем её
            if date_from_text:
                fin_intent['date'] = date_from_text
            # Если всё равно нет даты — уточнить у пользователя
            if not fin_intent.get('date'):
                await update.message.reply_text("Не удалось определить дату операции. Уточни, пожалуйста, дату для записи дохода/расхода.")
                context.user_data['awaiting_fin_date'] = fin_intent
                return
        if intent == "income":
            op = finances.add_income(
                fin_intent.get("amount"),
                fin_intent.get("project"),
                description=fin_intent.get("description"),
                date=fin_intent.get("date")
            )
            await update.message.reply_text(f"Доход добавлен: {op['amount']} ({op['project']}) — {op['description']} ({op['date']})")
            return
        elif intent == "expense":
            op = finances.add_expense(
                fin_intent.get("amount"),
                fin_intent.get("project"),
                description=fin_intent.get("description"),
                date=fin_intent.get("date"),
                category=fin_intent.get("category")
            )
            await update.message.reply_text(f"Расход добавлен: {op['amount']} ({op['project']}) — {op['description']} ({op['date']})")
            return
        elif intent == "report":
            report = finances.get_report(
                period=fin_intent.get("period"),
                project=fin_intent.get("project")
            )
            await update.message.reply_text(
                f"Отчёт: Доход {report['income']}, Расход {report['expense']}, Прибыль {report['profit']}"
            )
            return
        elif intent == "unclassified":
            # Только если в сообщении явно есть слова про траты/категории
            if "категор" in user_text.lower() or "траты" in user_text.lower():
                unclassified = finances.get_unclassified_expenses()
                if not unclassified:
                    await update.message.reply_text("Нет трат без категории.")
                else:
                    text = "\n".join([f"{op['amount']} ({op['project']}) — {op['description']} ({op['date']})" for op in unclassified])
                    await update.message.reply_text(f"Траты без категории:\n{text}")
                return
    # --- Цели (как раньше) ---
    if re.match(r"установи цель|добавь цель", user_text, re.I):
        # Пример: "Установи цель 3 млн выручки до сентября"
        match = re.search(r"цель (.+?)( до ([^\n]+))?$", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            deadline = match.group(3).strip() if match.group(3) else None
            goal = planner.set_goal(goal_text, deadline)
            await update.message.reply_text(f"Цель добавлена: {goal['goal_text']} (до {goal['deadline']})")
        else:
            await update.message.reply_text("Формат: 'Установи цель <текст> до <дата/срок>'")
        return
    elif re.match(r"какие цели|покажи цели|список целей", user_text, re.I):
        goals = planner.get_goals()
        if not goals:
            await update.message.reply_text("Целей пока нет.")
        else:
            text = "\n".join([f"- {g['goal_text']} (до {g['deadline']})" for g in goals])
            await update.message.reply_text(f"Текущие цели:\n{text}")
    elif re.match(r"прогресс по цели", user_text, re.I):
        # Пример: "Прогресс по цели 3 млн выручки"
        match = re.search(r"прогресс по цели (.+)$", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            progress = planner.get_goal_progress(goal_text)
            if progress:
                await update.message.reply_text(f"Прогресс по цели '{progress['goal_text']}': {progress['progress']}% (до {progress['deadline']})")
            else:
                await update.message.reply_text("Цель не найдена.")
        else:
            await update.message.reply_text("Формат: 'Прогресс по цели <текст цели>'")
    elif re.match(r"обнови прогресс по цели", user_text, re.I):
        # Пример: "Обнови прогресс по цели 3 млн выручки: 40%"
        match = re.search(r"обнови прогресс по цели (.+?):\s*(\d+)%?", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            progress = int(match.group(2))
            updated = planner.update_goal_progress(goal_text, progress)
            if updated:
                await update.message.reply_text(f"Прогресс по цели '{goal_text}' обновлён: {progress}%")
            else:
                await update.message.reply_text("Цель не найдена.")
        else:
            await update.message.reply_text("Формат: 'Обнови прогресс по цели <текст>: <число>%'")
    elif re.match(r"добавь задачу к цели", user_text, re.I):
        # Пример: "Добавь задачу к цели 3 млн выручки: Позвонить 10 клиентам"
        match = re.search(r"добавь задачу к цели (.+?):\s*(.+)$", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            task_text = match.group(2).strip()
            task = planner.add_goal_task(goal_text, task_text)
            await update.message.reply_text(f"Задача добавлена к цели '{goal_text}': {task['task_text']}")
        else:
            await update.message.reply_text("Формат: 'Добавь задачу к цели <цель>: <текст задачи>'")
    elif re.match(r"отметь задачу по цели.*как выполненную", user_text, re.I):
        # Пример: "Отметь задачу по цели 3 млн выручки как выполненную: Позвонить 10 клиентам"
        match = re.search(r"отметь задачу по цели (.+?) как выполненную: (.+)$", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            task_text = match.group(2).strip()
            task = planner.mark_goal_task_done(goal_text, task_text)
            if task:
                await update.message.reply_text(f"Задача '{task_text}' по цели '{goal_text}' отмечена как выполненная.")
            else:
                await update.message.reply_text("Задача не найдена.")
        else:
            await update.message.reply_text("Формат: 'Отметь задачу по цели <цель> как выполненную: <текст задачи>'")
    elif re.match(r"промежуточные задачи по цели", user_text, re.I):
        # Пример: "Промежуточные задачи по цели 3 млн выручки"
        match = re.search(r"промежуточные задачи по цели (.+)$", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            tasks = planner.get_goal_tasks(goal_text)
            if not tasks:
                await update.message.reply_text("Промежуточных задач по этой цели нет.")
            else:
                text = "\n".join([f"- {t['task_text']} ({t['date'] if t['date'] else 'без даты'}) {'[Выполнено]' if t['done'] else ''}" for t in tasks])
                await update.message.reply_text(f"Промежуточные задачи по цели '{goal_text}':\n{text}")
        else:
            await update.message.reply_text("Формат: 'Промежуточные задачи по цели <текст цели>'")
    elif re.match(r"разбей цель на ежедневные задачи", user_text, re.I):
        # Пример: "Разбей цель 3 млн выручки на ежедневные задачи: 3000000 с 2024-06-01 по 2024-06-30 руб."
        match = re.search(r"разбей цель (.+?) на ежедневные задачи: (\d+) с (\d{4}-\d{2}-\d{2}) по (\d{4}-\d{2}-\d{2}) ?([\w.]+)?", user_text, re.I)
        if match:
            goal_text = match.group(1).strip()
            total_value = int(match.group(2))
            start_date = match.group(3)
            end_date = match.group(4)
            unit = match.group(5) if match.group(5) else "единиц"
            tasks = planner.suggest_daily_tasks(goal_text, total_value, start_date, end_date, unit)
            await update.message.reply_text(f"Создано {len(tasks)} ежедневных задач по цели '{goal_text}'.")
        else:
            await update.message.reply_text("Формат: 'Разбей цель <цель> на ежедневные задачи: <число> с <дата> по <дата> <единицы>'")
    # --- Естественный язык для задач ---
    task_intent = await parse_task_intent(user_text)
    if task_intent:
        intent = task_intent.get("intent")
        if intent == "add":
            # Проверяем дату задачи
            date = validate_task_date(task_intent.get("date"))
            time = task_intent.get("time")
            if not date:
                # Если дата не указана явно, пробуем распознать естественную дату и время
                date, parsed_time = parse_natural_date_and_time(user_text)
                if parsed_time:
                    time = parsed_time
            if not date and task_intent.get("date"):
                # Если дата была, но она в прошлом — сообщаем и ставим на сегодня
                date = datetime.now().strftime('%Y-%m-%d')
                msg = "Дата задачи была в прошлом или не распознана, задача записана на сегодня."
            elif not date:
                # Fallback: спросить пользователя явно
                await update.message.reply_text("Не удалось определить дату задачи. На какой день поставить задачу?")
                return
            else:
                msg = None
            # Логирование для отладки
            print(f"[DEBUG] add_task: text={task_intent.get('task_text')}, date={date}, time={time}")
            task = calendar.add_task(
                task_intent.get("task_text"),
                date=date,
                time=time
            )
            reply = f"Задача добавлена: {task['task_text']} ({task['date']} {task['time'] or ''})"
            if msg:
                reply = msg + "\n" + reply
            await update.message.reply_text(reply)
            return
        elif intent == "view":
            date = task_intent.get("date")
            if date:
                tasks = calendar.get_daily_plan(date)
                if not tasks:
                    await update.message.reply_text("На этот день задач нет.")
                else:
                    text = "\n".join([f"[{t['id']}] {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
                    await update.message.reply_text(f"Задачи на {date}:\n{text}")
            else:
                tasks = calendar.get_tasks()
                if not tasks:
                    await update.message.reply_text("Задач нет.")
                else:
                    text = "\n".join([f"[{t['id']}] {t['task_text']} {t['date']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
                    await update.message.reply_text(f"Все задачи:\n{text}")
            return
        elif intent == "delete":
            task_id = task_intent.get("task_id")
            if task_id:
                calendar.delete_task(task_id)
                await update.message.reply_text(f"Задача {task_id} удалена.")
            else:
                await update.message.reply_text("Не удалось определить задачу для удаления.")
            return
        elif intent == "move":
            task_id = task_intent.get("task_id")
            new_date = task_intent.get("new_date")
            if task_id and new_date:
                calendar.move_task(task_id, new_date)
                await update.message.reply_text(f"Задача {task_id} перенесена на {new_date}.")
            else:
                await update.message.reply_text("Не удалось определить задачу или новую дату.")
            return
        elif intent == "done":
            task_id = task_intent.get("task_id")
            if task_id:
                calendar.mark_task_done(task_id)
                await update.message.reply_text(f"Задача {task_id} отмечена как выполненная.")
            else:
                await update.message.reply_text("Не удалось определить задачу для отметки.")
            return
        elif intent == "summary":
            today = datetime.now().strftime("%Y-%m-%d")
            tasks = calendar.get_daily_plan(today)
            if not tasks:
                await update.message.reply_text("На сегодня задач нет.")
            else:
                text = "\n".join([f"[{t['id']}] {t['task_text']} {t['time'] or ''} {'[Выполнено]' if t['done'] else ''}" for t in tasks])
                await update.message.reply_text(f"План на сегодня:\n{text}")
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