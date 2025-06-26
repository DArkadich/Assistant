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
    "task_text, date (ГГГГ-ММ-ДД или null), time (ЧЧ:ММ или null), new_date (если перенос), task_id (если есть), "
    "пример: {\"intent\": \"add\", \"task_text\": \"Встреча с Тигрой\", \"date\": \"2024-06-10\", \"time\": \"15:00\"}"
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
            import asyncio
            asyncio.run(send_daily_summary_to_chat(app, chat_id))
    scheduler.add_job(job, 'cron', hour=8, minute=0)
    scheduler.start()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Сохраняем chat_id для рассылки сводки
    save_last_chat_id(update.effective_chat.id)
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
    # Прежняя маршрутизация
    elif "задач" in user_text or "встреча" in user_text or "напомни" in user_text:
        await update.message.reply_text("[Задачи] (заглушка)" )
    elif "прибыль" in user_text or "доход" in user_text or "расход" in user_text or "финанс" in user_text:
        await update.message.reply_text("[Финансы] (заглушка)")
    elif "цель" in user_text or "KPI" in user_text or "прогресс" in user_text:
        await update.message.reply_text("[Цели/KPI] (заглушка)")
    # --- Естественный язык для задач ---
    task_intent = await parse_task_intent(user_text)
    if task_intent:
        intent = task_intent.get("intent")
        if intent == "add":
            task = calendar.add_task(
                task_intent.get("task_text"),
                date=task_intent.get("date"),
                time=task_intent.get("time")
            )
            await update.message.reply_text(f"Задача добавлена: {task['task_text']} ({task['date']} {task['time'] or ''})")
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

def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    start_scheduler(app)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling() 