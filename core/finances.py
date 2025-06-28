# finances.py — учёт доходов и расходов через Google Sheets или локально
import json
import os
from datetime import datetime
import gspread
import traceback
import uuid
from google.oauth2.service_account import Credentials

# Импортируем менеджер Google Drive
from .drive_manager import drive_manager

operations = []
FIN_FILE = 'finances.json'
SHEET_ID = '10Tu5b40FPKrDi8M7sXTv8SUAbN2c6UKmJGBhsCK_u94'
SHEET_NAME = 'Финансы'

# --- Новые структуры для документооборота и учёта документов ---
payments = []  # список платежей
purchases = []  # список закупок
documents = []  # список документов

DOC_FILE = 'documents.json'

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

def get_income_for_week(project=None):
    """Получить приходы за текущую неделю (с понедельника по воскресенье)."""
    from datetime import datetime, timedelta
    
    today = datetime.now().date()
    # Находим понедельник текущей недели
    monday = today - timedelta(days=today.weekday())
    # Находим воскресенье текущей недели
    sunday = monday + timedelta(days=6)
    
    week_income = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'income':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d').date()
            if monday <= op_date <= sunday:
                week_income.append(op)
                total_amount += op['amount']
    
    return {
        'income_list': week_income,
        'total_amount': total_amount,
        'week_start': monday.strftime('%Y-%m-%d'),
        'week_end': sunday.strftime('%Y-%m-%d')
    }

def get_expense_for_week(project=None):
    """Получить расходы за текущую неделю (с понедельника по воскресенье)."""
    from datetime import datetime, timedelta
    
    today = datetime.now().date()
    # Находим понедельник текущей недели
    monday = today - timedelta(days=today.weekday())
    # Находим воскресенье текущей недели
    sunday = monday + timedelta(days=6)
    
    week_expense = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'expense':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d').date()
            if monday <= op_date <= sunday:
                week_expense.append(op)
                total_amount += op['amount']
    
    return {
        'expense_list': week_expense,
        'total_amount': total_amount,
        'week_start': monday.strftime('%Y-%m-%d'),
        'week_end': sunday.strftime('%Y-%m-%d')
    }

def get_income_for_month(month=None, year=None, project=None):
    """Получить приходы за месяц."""
    from datetime import datetime
    
    if month is None or year is None:
        today = datetime.now()
        month = today.month
        year = today.year
    
    month_income = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'income':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            if op_date.month == month and op_date.year == year:
                month_income.append(op)
                total_amount += op['amount']
    
    month_name = datetime(year, month, 1).strftime('%B %Y')
    return {
        'income_list': month_income,
        'total_amount': total_amount,
        'period': month_name
    }

def get_expense_for_month(month=None, year=None, project=None):
    """Получить расходы за месяц."""
    from datetime import datetime
    
    if month is None or year is None:
        today = datetime.now()
        month = today.month
        year = today.year
    
    month_expense = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'expense':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            if op_date.month == month and op_date.year == year:
                month_expense.append(op)
                total_amount += op['amount']
    
    month_name = datetime(year, month, 1).strftime('%B %Y')
    return {
        'expense_list': month_expense,
        'total_amount': total_amount,
        'period': month_name
    }

def get_income_for_quarter(quarter=None, year=None, project=None):
    """Получить приходы за квартал."""
    from datetime import datetime
    
    if quarter is None or year is None:
        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        year = today.year
    
    quarter_income = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'income':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            op_quarter = (op_date.month - 1) // 3 + 1
            if op_quarter == quarter and op_date.year == year:
                quarter_income.append(op)
                total_amount += op['amount']
    
    quarter_name = f"Q{quarter} {year}"
    return {
        'income_list': quarter_income,
        'total_amount': total_amount,
        'period': quarter_name
    }

def get_expense_for_quarter(quarter=None, year=None, project=None):
    """Получить расходы за квартал."""
    from datetime import datetime
    
    if quarter is None or year is None:
        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        year = today.year
    
    quarter_expense = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'expense':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            op_quarter = (op_date.month - 1) // 3 + 1
            if op_quarter == quarter and op_date.year == year:
                quarter_expense.append(op)
                total_amount += op['amount']
    
    quarter_name = f"Q{quarter} {year}"
    return {
        'expense_list': quarter_expense,
        'total_amount': total_amount,
        'period': quarter_name
    }

def get_income_for_year(year=None, project=None):
    """Получить приходы за год."""
    from datetime import datetime
    
    if year is None:
        year = datetime.now().year
    
    year_income = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'income':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            if op_date.year == year:
                year_income.append(op)
                total_amount += op['amount']
    
    return {
        'income_list': year_income,
        'total_amount': total_amount,
        'period': str(year)
    }

def get_expense_for_year(year=None, project=None):
    """Получить расходы за год."""
    from datetime import datetime
    
    if year is None:
        year = datetime.now().year
    
    year_expense = []
    total_amount = 0
    
    for op in operations:
        if op['type'] == 'expense':
            if project and op['project'] != project:
                continue
            
            op_date = datetime.strptime(op['date'], '%Y-%m-%d')
            if op_date.year == year:
                year_expense.append(op)
                total_amount += op['amount']
    
    return {
        'expense_list': year_expense,
        'total_amount': total_amount,
        'period': str(year)
    }

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
def add_document(doc_type, number, date, payment_ids=None, purchase_ids=None, file_url=None, description=None, counterparty_name=None, amount=None, keywords=None, file_path=None):
    """
    Добавить документ с возможностью загрузки файла в Google Drive.
    
    Args:
        file_path: Путь к локальному файлу для загрузки в Drive
    """
    # Если передан файл, загружаем его в Google Drive
    drive_file_info = None
    if file_path and os.path.exists(file_path):
        try:
            # Создаем папку для документа
            folder_id = drive_manager.create_document_folder(doc_type, date)
            
            # Загружаем файл
            file_name = f"{doc_type}_{number}_{date}.{file_path.split('.')[-1]}"
            drive_file_info = drive_manager.upload_file(
                file_path=file_path,
                file_name=file_name,
                folder_id=folder_id,
                description=f"{doc_type.title()} №{number} от {date}"
            )
            
            if drive_file_info:
                file_url = drive_file_info['url']
                print(f"✅ Файл загружен в Google Drive: {drive_file_info['url']}")
            else:
                print("❌ Ошибка загрузки файла в Google Drive")
                
        except Exception as e:
            print(f"❌ Ошибка при работе с Google Drive: {e}")
    
    # Если не передан file_path, но есть file_url, извлекаем текст
    extracted_text = None
    if file_url and not file_path:
        try:
            # Пытаемся извлечь текст из файла в Drive
            if 'drive.google.com' in file_url:
                file_id = extract_file_id_from_url(file_url)
                if file_id:
                    extracted_text = drive_manager.extract_text_from_file(file_id)
                    if extracted_text:
                        print(f"✅ Текст извлечен из файла в Drive")
        except Exception as e:
            print(f"❌ Ошибка извлечения текста: {e}")
    
    # Автоматически создаем описание, если его нет
    if not description and extracted_text:
        description = create_document_description(extracted_text, doc_type, number, date)
    
    # Автоматически извлекаем ключевые слова
    if not keywords and extracted_text:
        keywords = extract_keywords_from_text(extracted_text, doc_type)
    
    doc = {
        'id': str(uuid.uuid4()),
        'type': doc_type,
        'number': number,
        'date': date,
        'payment_ids': payment_ids or [],
        'purchase_ids': purchase_ids or [],
        'file_url': file_url,
        'received': bool(file_url),
        'description': description,
        'counterparty_name': counterparty_name,
        'amount': amount,
        'keywords': keywords or [],
        'drive_file_id': drive_file_info['id'] if drive_file_info else None,
        'drive_folder_id': drive_file_info.get('folder_id') if drive_file_info else None,
        'extracted_text': extracted_text[:1000] if extracted_text else None  # Сохраняем первые 1000 символов
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

def extract_file_id_from_url(url):
    """Извлечь ID файла из URL Google Drive."""
    import re
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def create_document_description(text, doc_type, number, date):
    """Создать описание документа на основе извлеченного текста."""
    # Простая логика создания описания
    lines = text.split('\n')[:10]  # Берем первые 10 строк
    summary = ' '.join([line.strip() for line in lines if line.strip()])
    
    if len(summary) > 200:
        summary = summary[:200] + "..."
    
    return f"{doc_type.title()} №{number} от {date}: {summary}"

def extract_keywords_from_text(text, doc_type):
    """Извлечь ключевые слова из текста документа."""
    # Простая логика извлечения ключевых слов
    keywords = []
    
    # Добавляем тип документа
    keywords.append(doc_type)
    
    # Ищем суммы
    import re
    amounts = re.findall(r'\d{1,3}(?:\s\d{3})*(?:\sруб|\s₽|руб|₽)', text)
    if amounts:
        keywords.extend(amounts[:3])  # Максимум 3 суммы
    
    # Ищем даты
    dates = re.findall(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', text)
    if dates:
        keywords.extend(dates[:2])  # Максимум 2 даты
    
    # Ищем организации (слова с заглавной буквы)
    orgs = re.findall(r'[А-ЯЁ][а-яё]*(?:\s+[А-ЯЁ][а-яё]*)*\s+(?:ООО|ИП|АО|ЗАО)', text)
    if orgs:
        keywords.extend(orgs[:2])  # Максимум 2 организации
    
    return list(set(keywords))  # Убираем дубликаты

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

# --- Удаление платежа ---
def delete_payment(payment_id):
    """
    Удаляет платёж и все связанные с ним документы.
    Возвращает True если платёж был найден и удалён, False если не найден.
    """
    payment = find_payment_by_id(payment_id)
    if not payment:
        return False
    
    # Удаляем связанные документы
    documents_to_remove = []
    for doc_id in payment['documents_ids']:
        doc = find_document_by_id(doc_id)
        if doc:
            # Удаляем ссылку на этот платёж из документа
            if payment_id in doc['payment_ids']:
                doc['payment_ids'].remove(payment_id)
            # Если у документа больше нет связанных платежей, удаляем его
            if not doc['payment_ids'] and not doc['purchase_ids']:
                documents_to_remove.append(doc_id)
    
    # Удаляем документы без связей
    for doc_id in documents_to_remove:
        documents[:] = [d for d in documents if d['id'] != doc_id]
    
    # Удаляем платёж
    payments[:] = [p for p in payments if p['id'] != payment_id]
    
    # Сохраняем изменения
    save_doc()
    return True

# --- Функции поиска документов ---
def search_documents_by_counterparty(counterparty_name):
    """Поиск документов по названию контрагента."""
    return [doc for doc in documents if doc.get('counterparty_name') and counterparty_name.lower() in doc['counterparty_name'].lower()]

def search_documents_by_amount(min_amount=None, max_amount=None):
    """Поиск документов по сумме."""
    results = []
    for doc in documents:
        if doc.get('amount'):
            if min_amount and max_amount:
                if min_amount <= doc['amount'] <= max_amount:
                    results.append(doc)
            elif min_amount:
                if doc['amount'] >= min_amount:
                    results.append(doc)
            elif max_amount:
                if doc['amount'] <= max_amount:
                    results.append(doc)
    return results

def search_documents_by_keywords(keywords):
    """Поиск документов по ключевым словам."""
    if isinstance(keywords, str):
        keywords = [keywords.lower()]
    else:
        keywords = [k.lower() for k in keywords]
    
    results = []
    for doc in documents:
        # Поиск в ключевых словах
        doc_keywords = [k.lower() for k in doc.get('keywords', [])]
        if any(kw in doc_keywords for kw in keywords):
            results.append(doc)
            continue
        
        # Поиск в описании
        if doc.get('description') and any(kw in doc['description'].lower() for kw in keywords):
            results.append(doc)
            continue
        
        # Поиск в названии контрагента
        if doc.get('counterparty_name') and any(kw in doc['counterparty_name'].lower() for kw in keywords):
            results.append(doc)
            continue
    
    return results

def search_documents_by_date_range(start_date=None, end_date=None):
    """Поиск документов по диапазону дат."""
    from datetime import datetime
    
    results = []
    for doc in documents:
        doc_date = datetime.strptime(doc['date'], '%Y-%m-%d').date()
        
        if start_date and end_date:
            if start_date <= doc_date <= end_date:
                results.append(doc)
        elif start_date:
            if doc_date >= start_date:
                results.append(doc)
        elif end_date:
            if doc_date <= end_date:
                results.append(doc)
    
    return results

def get_documents_summary():
    """Получить сводку по документам."""
    total_docs = len(documents)
    docs_by_type = {}
    total_amount = 0
    
    for doc in documents:
        # Подсчет по типам
        doc_type = doc['type']
        docs_by_type[doc_type] = docs_by_type.get(doc_type, 0) + 1
        
        # Подсчет общей суммы
        if doc.get('amount'):
            total_amount += doc['amount']
    
    return {
        'total_documents': total_docs,
        'by_type': docs_by_type,
        'total_amount': total_amount,
        'with_files': len([d for d in documents if d.get('file_url')]),
        'without_files': len([d for d in documents if not d.get('file_url')])
    } 