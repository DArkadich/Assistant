"""
Глобальный поиск по всей памяти ассистента: задачи, сообщения, документы, CRM, Google Drive, Obsidian
"""
from typing import List, Dict

# Импортируем необходимые модули
from core.team_manager import team_manager
from core.memory import chat_memory
from core.finances import finances
from core.obsidian_manager import obsidian_manager
from core.amocrm import amocrm
try:
    from core.drive_manager import drive_manager
except ImportError:
    drive_manager = None


def global_search(query: str, limit: int = 10) -> Dict[str, List[Dict]]:
    """
    Глобальный поиск по задачам, сообщениям, документам, CRM, Google Drive, Obsidian
    Возвращает словарь с результатами по категориям
    """
    results = {}

    # 1. Поиск по задачам
    task_hits = []
    for task in team_manager.team_data.get('tasks', []):
        if query.lower() in task.get('task', '').lower() or query.lower() in task.get('description', '').lower():
            task_hits.append(task)
    if task_hits:
        results['tasks'] = task_hits[:limit]

    # 2. Поиск по сообщениям (чат-память)
    message_hits = []
    for msg in chat_memory.get_all_messages():
        if query.lower() in msg['text'].lower():
            message_hits.append(msg)
    if message_hits:
        results['messages'] = message_hits[:limit]

    # 3. Поиск по документам (финансы)
    doc_hits = []
    for doc in finances.documents:
        if any(query.lower() in str(doc.get(field, '')).lower() for field in ['type', 'number', 'date', 'description', 'counterparty_name', 'file_url']):
            doc_hits.append(doc)
    if doc_hits:
        results['documents'] = doc_hits[:limit]

    # 4. Поиск по Obsidian (заметки)
    obsidian_hits = []
    for note in obsidian_manager.list_notes():
        if query.lower() in note['title'].lower() or query.lower() in note.get('content', '').lower():
            obsidian_hits.append(note)
    if obsidian_hits:
        results['obsidian'] = obsidian_hits[:limit]

    # 5. Поиск по CRM (AmoCRM)
    crm_hits = []
    for contact in amocrm.get_contacts():
        if query.lower() in contact.get('name', '').lower() or query.lower() in str(contact.get('custom_fields', '')).lower():
            crm_hits.append({'type': 'contact', **contact})
    for lead in amocrm.get_leads():
        if query.lower() in lead.get('name', '').lower() or query.lower() in str(lead.get('custom_fields', '')).lower():
            crm_hits.append({'type': 'lead', **lead})
    if crm_hits:
        results['crm'] = crm_hits[:limit]

    # 6. Поиск по Google Drive (если есть)
    drive_hits = []
    if drive_manager:
        files = drive_manager.search_files(query)
        for f in files:
            drive_hits.append(f)
    if drive_hits:
        results['drive'] = drive_hits[:limit]

    return results

def format_global_search_results(results: Dict[str, List[Dict]], query: str) -> str:
    """Форматирование результатов глобального поиска для Telegram"""
    if not results:
        return f"❌ По запросу <b>{query}</b> ничего не найдено."
    text = f"🔎 <b>Результаты поиска по: {query}</b>\n\n"
    for section, items in results.items():
        if section == 'tasks':
            text += f"📝 <b>Задачи:</b> ({len(items)})\n"
            for t in items:
                text += f"• {t.get('task')} (до {t.get('deadline')})\n"
            text += "\n"
        elif section == 'messages':
            text += f"💬 <b>Сообщения:</b> ({len(items)})\n"
            for m in items:
                text += f"• {m.get('username', 'user')}: {m.get('text')[:80]}\n"
            text += "\n"
        elif section == 'documents':
            text += f"📄 <b>Документы:</b> ({len(items)})\n"
            for d in items:
                text += f"• {d.get('type', '').title()} №{d.get('number', '')} {d.get('description', '')}\n"
            text += "\n"
        elif section == 'obsidian':
            text += f"🗒️ <b>Заметки Obsidian:</b> ({len(items)})\n"
            for n in items:
                text += f"• {n.get('title')}\n"
            text += "\n"
        elif section == 'crm':
            text += f"📋 <b>CRM:</b> ({len(items)})\n"
            for c in items:
                if c['type'] == 'contact':
                    text += f"• Контакт: {c.get('name')}\n"
                else:
                    text += f"• Сделка: {c.get('name')}\n"
            text += "\n"
        elif section == 'drive':
            text += f"☁️ <b>Google Drive:</b> ({len(items)})\n"
            for f in items:
                text += f"• {f.get('name')} ({f.get('webViewLink', f.get('id', ''))})\n"
            text += "\n"
    return text.strip() 