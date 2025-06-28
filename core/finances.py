# finances.py — учёт доходов и расходов через Google Sheets или локально
import json
import os
from datetime import datetime
import gspread

operations = []
FIN_FILE = 'finances.json'
SHEET_ID = '10Tu5b40FPKrDi8M7sXTv8SUAbN2c6UKmJGBhsCK_u94'
SHEET_NAME = 'Финансы'

# --- Persistence ---
def save_finances():
    with open(FIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(operations, f, ensure_ascii=False, indent=2)

def load_finances():
    global operations
    if os.path.exists(FIN_FILE):
        with open(FIN_FILE, 'r', encoding='utf-8') as f:
            operations.clear()
            operations.extend(json.load(f))

load_finances()

# --- Google Sheets ---
def get_gsheet():
    gc = gspread.service_account(filename='credentials.json')
    sh = gc.open_by_key(SHEET_ID)
    try:
        worksheet = sh.worksheet(SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=10)
    return worksheet

def append_to_gsheet(op):
    ws = get_gsheet()
    # Если таблица пуста — добавим заголовки
    if ws.row_count == 0 or not ws.get_all_values():
        ws.append_row(['Тип', 'Сумма', 'Проект', 'Описание', 'Дата', 'Категория'])
    ws.append_row([
        'Доход' if op['type'] == 'income' else 'Расход',
        op['amount'],
        op['project'],
        op.get('description', ''),
        op['date'],
        op.get('category', '')
    ])

# --- Finance logic ---
def add_income(amount, project, description=None, date=None):
    """
    Добавить доход.
    Пример: add_income(400000, 'ВБ', 'Поступили от ВБ', date='2024-06-10')
    """
    op = {
        'type': 'income',
        'amount': amount,
        'project': project,
        'description': description,
        'date': date or datetime.now().strftime('%Y-%m-%d')
    }
    operations.append(op)
    save_finances()
    try:
        append_to_gsheet(op)
    except Exception as e:
        print(f"Ошибка при записи дохода в Google Sheets: {e}", flush=True)
    return op

def add_expense(amount, project, description=None, date=None, category=None):
    """
    Добавить расход.
    Пример: add_expense(15000, 'Horien', 'Закупка коробок', date='2024-06-10', category='упаковка')
    """
    op = {
        'type': 'expense',
        'amount': amount,
        'project': project,
        'description': description,
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'category': category
    }
    operations.append(op)
    save_finances()
    try:
        append_to_gsheet(op)
    except Exception as e:
        print(f"Ошибка при записи расхода в Google Sheets: {e}", flush=True)
    return op

def get_report(period=None, project=None):
    """
    Получить финансовый отчёт за период (месяц, неделя, всё).
    Пример: get_report('июнь', 'Horien')
    """
    # period: 'июнь', '2024-06', '2024', None
    # project: 'Horien', 'ВБ', None
    result = {'income': 0, 'expense': 0, 'profit': 0, 'details': []}
    for op in operations:
        if project and op['project'] != project:
            continue
        if period:
            if len(period) == 4 and op['date'][:4] != period:
                continue
            if len(period) == 7 and op['date'][:7] != period:
                continue
            if len(period) == 2 and period.lower() in month_name(op['date'][5:7]):
                continue
        if op['type'] == 'income':
            result['income'] += op['amount']
        elif op['type'] == 'expense':
            result['expense'] += op['amount']
        result['details'].append(op)
    result['profit'] = result['income'] - result['expense']
    return result

def get_unclassified_expenses():
    """
    Получить расходы без категории.
    Пример: get_unclassified_expenses()
    """
    return [op for op in operations if op['type'] == 'expense' and not op.get('category')]

def month_name(mm):
    months = {
        '01': 'янв', '02': 'фев', '03': 'мар', '04': 'апр', '05': 'мая', '06': 'июн',
        '07': 'июл', '08': 'авг', '09': 'сен', '10': 'окт', '11': 'ноя', '12': 'дек'
    }
    return months.get(mm, mm) 