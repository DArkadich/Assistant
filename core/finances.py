# finances.py — учёт доходов и расходов через Google Sheets или локально
import json
import os
from datetime import datetime
import gspread
import traceback
import uuid

operations = []
FIN_FILE = 'finances.json'
SHEET_ID = '10Tu5b40FPKrDi8M7sXTv8SUAbN2c6UKmJGBhsCK_u94'
SHEET_NAME = 'Финансы'

# --- Новые структуры для документооборота и учёта документов ---
payments = []  # список платежей
purchases = []  # список закупок
documents = []  # список документов

DOC_FILE = 'doc_data.json'

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
        print(f"Ошибка при записи дохода в Google Sheets: {e}\n{traceback.format_exc()}", flush=True)
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
        print(f"Ошибка при записи расхода в Google Sheets: {e}\n{traceback.format_exc()}", flush=True)
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

def get_report_by_project(period=None):
    """Вернуть отчёты по каждому проекту за период."""
    projects = set(op['project'] for op in operations)
    reports = {}
    for project in projects:
        reports[project] = get_report(period=period, project=project)
    return reports

def get_total_balance(project=None):
    """Вернуть чистый остаток (доходы - расходы) по всем операциям или по проекту."""
    income = sum(op['amount'] for op in operations if op['type'] == 'income' and (not project or op['project'] == project))
    expense = sum(op['amount'] for op in operations if op['type'] == 'expense' and (not project or op['project'] == project))
    return income - expense

def save_doc():
    data = {
        'payments': payments,
        'purchases': purchases,
        'documents': documents
    }
    with open(DOC_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_doc():
    global payments, purchases, documents
    if os.path.exists(DOC_FILE):
        with open(DOC_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            payments = data.get('payments', [])
            purchases = data.get('purchases', [])
            documents = data.get('documents', [])

load_doc()

# --- Типы документов ---
VED_DOC_TYPES = ['накладная', 'упд', 'гтд', 'счёт', 'контракт', 'акт']

# --- Добавление платежа ---
def add_payment(amount, date, direction, country, project, counterparty, purpose, purchases_ids=None):
    """
    direction: 'in' (входящий) или 'out' (исходящий)
    country: 'RU' или 'INT' (Россия или за границу)
    purchases_ids: список id закупок (может быть пустым)
    """
    payment = {
        'id': str(uuid.uuid4()),
        'amount': amount,
        'date': date,
        'direction': direction,
        'country': country,
        'project': project,
        'counterparty': counterparty,
        'purpose': purpose,
        'purchases_ids': purchases_ids or [],
        'documents_ids': [],  # будут добавляться позже
        'closed': False
    }
    payments.append(payment)
    save_doc()
    return payment

# --- Добавление закупки ---
def add_purchase(name, amount, date, payment_ids=None):
    purchase = {
        'id': str(uuid.uuid4()),
        'name': name,
        'amount': amount,
        'date': date,
        'payment_ids': payment_ids or [],
        'documents_ids': []
    }
    purchases.append(purchase)
    save_doc()
    return purchase

# --- Добавление документа ---
def add_document(doc_type, number, date, payment_ids=None, purchase_ids=None, file_url=None):
    doc = {
        'id': str(uuid.uuid4()),
        'type': doc_type,
        'number': number,
        'date': date,
        'payment_ids': payment_ids or [],
        'purchase_ids': purchase_ids or [],
        'file_url': file_url,
        'received': bool(file_url)
    }
    documents.append(doc)
    
    # Обновляем платежи, добавляя ID документа
    for payment_id in doc['payment_ids']:
        payment = find_payment_by_id(payment_id)
        if payment:
            if doc['id'] not in payment['documents_ids']:
                payment['documents_ids'].append(doc['id'])
    
    # Обновляем закупки, добавляя ID документа
    for purchase_id in doc['purchase_ids']:
        purchase = find_purchase_by_id(purchase_id)
        if purchase:
            if doc['id'] not in purchase['documents_ids']:
                purchase['documents_ids'].append(doc['id'])
    
    save_doc()
    return doc

# --- Поиск платежей, закупок, документов ---
def find_payment_by_id(payment_id):
    return next((p for p in payments if p['id'] == payment_id), None)

def find_purchase_by_id(purchase_id):
    return next((p for p in purchases if p['id'] == purchase_id), None)

def find_document_by_id(doc_id):
    return next((d for d in documents if d['id'] == doc_id), None)

# --- Получение обязательных документов по типу платежа ---
def get_required_docs_for_payment(payment):
    if payment['direction'] == 'in':
        # Входящий платёж (платили мне)
        return ['накладная/упд', 'счёт', 'контракт', 'акт']
    elif payment['direction'] == 'out':
        if payment['country'] == 'RU':
            return ['накладная/упд', 'счёт', 'контракт', 'акт']
        else:
            return ['гтд', 'счёт', 'контракт', 'акт']
    return []

# --- Проверка закрытости платежа ---
def is_payment_closed(payment):
    required = get_required_docs_for_payment(payment)
    docs = [find_document_by_id(doc_id) for doc_id in payment['documents_ids']]
    doc_types = [d['type'] for d in docs if d]
    
    # Для накладная/упд — достаточно одного из них
    if 'накладная/упд' in required:
        if not any(t in doc_types for t in ['накладная', 'упд']):
            return False
    
    # Для гтд — должен быть
    if 'гтд' in required:
        if 'гтд' not in doc_types:
            return False
    
    # Остальные — все должны быть
    for t in required:
        if t in ['накладная/упд', 'гтд']:
            continue
        if t not in doc_types:
            return False
    
    return True

# --- Поиск незакрытых платежей ---
def get_unclosed_payments():
    return [p for p in payments if not is_payment_closed(p)] 