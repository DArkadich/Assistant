"""
Мультидашборд: дайджест по финансам, остаткам, KPI, календарю, задачам, проблемам
"""
from datetime import datetime, timedelta

def get_digest():
    from core.finances import get_total_balance, get_report, purchases
    from core.analytics import BusinessAnalytics
    from core.team_manager import team_manager
    from core.calendar import calendar
    # Финансы
    balance = get_total_balance()
    report = get_report(period=None)
    # Остатки (по закупкам)
    stock = sum(p['amount'] for p in purchases if 'остаток' in p.get('category', '').lower() or 'склад' in p.get('category', '').lower())
    # KPI
    analytics = BusinessAnalytics()
    kpi = analytics.get_kpi_summary()
    # Календарь (ближайшие события)
    events = calendar.get_upcoming_events(days=3)
    # Выполненные задачи (за неделю)
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    completed = [t for t in team_manager.team_data['tasks'] if t['status'] == 'completed' and t['completed_at'] >= week_ago]
    # Проблемы (просроченные задачи)
    overdue = team_manager.get_overdue_tasks()
    # Формируем дайджест
    return {
        'balance': balance,
        'income': report['income'],
        'expense': report['expense'],
        'stock': stock,
        'kpi': kpi,
        'events': events,
        'completed_tasks': completed,
        'overdue_tasks': overdue
    }

def format_digest(digest: dict) -> str:
    """Форматирование дайджеста для Telegram"""
    text = "<b>📊 Дайджест компании</b>\n\n"
    text += f"💵 <b>Финансы:</b>\nОстаток: <b>{digest['balance']:,}</b> руб.\nДоходы: <b>{digest['income']:,}</b> руб. | Расходы: <b>{digest['expense']:,}</b> руб.\n\n"
    text += f"📦 <b>Остатки:</b> {digest['stock']:,} (по закупкам)\n\n"
    text += f"📈 <b>KPI:</b>\n"
    for k, v in digest['kpi'].items():
        text += f"• {k}: <b>{v}</b>\n"
    text += "\n"
    text += f"📅 <b>Календарь (3 дня):</b>\n"
    for e in digest['events'][:5]:
        text += f"• {e['date']} — {e['title']}\n"
    text += "\n"
    text += f"✅ <b>Выполнено задач за неделю:</b> {len(digest['completed_tasks'])}\n"
    for t in digest['completed_tasks'][:3]:
        text += f"• {t['task']} ({t['completed_at'][:10]})\n"
    text += "\n"
    if digest['overdue_tasks']:
        text += f"🚨 <b>Проблемы (просрочено):</b> {len(digest['overdue_tasks'])}\n"
        for t in digest['overdue_tasks'][:3]:
            text += f"• {t['task']} (дедлайн: {t['deadline']})\n"
    else:
        text += "✅ <b>Проблем не обнаружено!</b>\n"
    return text.strip() 