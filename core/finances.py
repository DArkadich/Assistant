# finances.py — учёт доходов и расходов через Google Sheets или локально
import json
import os
from datetime import datetime

operations = []
FIN_FILE = 'finances.json'

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