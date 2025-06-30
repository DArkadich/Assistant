"""
AI-контроль дедлайнов: сбор задач, прогноз риска, предложения по действиям
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from core.task_manager import task_manager

class DeadlineMonitor:
    def get_all_deadlines(self) -> List[Dict]:
        """Получает все активные задачи из TaskManager."""
        # Берем только задачи, которые еще не завершены
        return task_manager.get_all_tasks(status_filter=['pending', 'in_progress'])

    def analyze_deadlines(self) -> List[Dict]:
        """
        Возвращает список задач с риском просрочки и рекомендациями.
        """
        now = datetime.now()
        risky = []
        for task in self.get_all_deadlines():
            due_str = task.get('due_date')
            if not due_str:
                continue

            try:
                due = datetime.fromisoformat(due_str.replace('Z', ''))
            except (ValueError, TypeError):
                continue
                
            days_left = (due - now).days
            progress = task.get('progress', 0)
            risk = False
            reason = []

            # Новая, более точная эвристика риска
            if days_left < 0:
                risk = True
                reason.append(f"Просрочено на {-days_left} дн.")
            elif days_left == 0:
                risk = True
                reason.append('Дедлайн сегодня')
            elif days_left <= 2 and progress < 70:
                risk = True
                reason.append('Мало времени, низкий прогресс')
            elif days_left <= 5 and progress < 30:
                risk = True
                reason.append('Слабый старт, высокий риск')
            
            if risk:
                task['risk_reason'] = ', '.join(reason)
                risky.append(task)
        
        # Сортируем по срочности
        risky.sort(key=lambda x: datetime.fromisoformat(x['due_date'].replace('Z', '')))
        return risky

    def format_risk_report(self, risky: Optional[List[Dict]] = None) -> str:
        if risky is None:
            risky = self.analyze_deadlines()
            
        if not risky:
            return "✅ Все дедлайны под контролем! Нет задач с риском просрочки."
            
        report = "⚠️ <b>Под угрозой дедлайна:</b>\n\n"
        for t in risky:
            report += (
                f"🚨 <b>{t['title']}</b>\n"
                f"   - <b>Срок:</b> {datetime.fromisoformat(t['due_date'].replace('Z', '')).strftime('%d.%m.%Y')} ({t.get('risk_reason', 'Риск')})\n"
                f"   - <b>Источник:</b> {t['source'].capitalize()}\n"
                f"   - <b>Статус:</b> {t['status']}, Прогресс: {t['progress']}%\n"
            )
            # Предложения
            report += "   - <b>Что делать:</b> "
            suggestions = []
            if t.get('assigned_to'):
                suggestions.append("Делегировать")
            if t.get('progress', 0) < 50:
                suggestions.append("Разбить на подзадачи")
            suggestions.append("Перенести срок")
            report += ", ".join(suggestions) + "?\n\n"
            
        return report

deadline_monitor = DeadlineMonitor() 