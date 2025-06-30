"""
Модуль контроля "деньги без документов"
Отслеживание незакрытых платежей и документов без оплаты
"""
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from core.task_manager import task_manager

class PaymentControl:
    def __init__(self):
        self.data_file = Path("payment_control.json")
        self.control_data = self._load_data()
        
    def _load_data(self) -> dict:
        """Загрузка данных контроля"""
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'last_check': None,
            'notifications_sent': [],
            'rules': {
                'check_frequency_days': 7,
                'auto_notify': True,
                'notify_owner': True,
                'notify_accountant': True
            }
        }
    
    def _save_data(self):
        """Сохранение данных контроля"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.control_data, f, ensure_ascii=False, indent=2)
    
    def check_unclosed_payments(self) -> List[Dict]:
        """Проверка незакрытых платежей (есть платёж, нет документов)"""
        unclosed = []
        
        # Импортируем finances только при необходимости
        try:
            from core import finances
            
            for payment in finances.payments:
                # Проверяем, есть ли все необходимые документы
                required_docs = finances.get_required_docs_for_payment(payment)
                existing_docs = [finances.find_document_by_id(doc_id) for doc_id in payment['documents_ids']]
                existing_doc_types = [d['type'] for d in existing_docs if d]
                
                missing_docs = []
                for req_doc in required_docs:
                    if req_doc == 'накладная/упд':
                        if not any(t in existing_doc_types for t in ['накладная', 'упд']):
                            missing_docs.append(req_doc)
                    elif req_doc not in existing_doc_types:
                        missing_docs.append(req_doc)
                
                if missing_docs:
                    unclosed.append({
                        'payment_id': payment['id'],
                        'amount': payment['amount'],
                        'counterparty': payment['counterparty'],
                        'date': payment['date'],
                        'project': payment['project'],
                        'direction': payment['direction'],
                        'missing_docs': missing_docs,
                        'existing_docs': existing_doc_types,
                        'days_since_payment': (datetime.now() - datetime.strptime(payment['date'], '%Y-%m-%d')).days
                    })
        except ImportError:
            print("Модуль finances не найден")
        
        return unclosed
    
    def check_documents_without_payment(self) -> List[Dict]:
        """Проверка документов без оплаты (есть документ, нет платежа)"""
        orphaned_docs = []
        
        try:
            from core import finances
            
            for doc in finances.documents:
                # Если документ не привязан к платежу
                if not doc.get('payment_ids') or len(doc['payment_ids']) == 0:
                    orphaned_docs.append({
                        'doc_id': doc['id'],
                        'doc_type': doc['type'],
                        'doc_number': doc['number'],
                        'doc_date': doc['date'],
                        'amount': doc.get('amount', 0),
                        'counterparty': doc.get('counterparty_name', 'Не указан'),
                        'days_since_doc': (datetime.now() - datetime.strptime(doc['date'], '%Y-%m-%d')).days
                    })
                else:
                    # Проверяем, существует ли привязанный платёж
                    payment_exists = False
                    for payment_id in doc['payment_ids']:
                        if finances.find_payment_by_id(payment_id):
                            payment_exists = True
                            break
                    
                    if not payment_exists:
                        orphaned_docs.append({
                            'doc_id': doc['id'],
                            'doc_type': doc['type'],
                            'doc_number': doc['number'],
                            'doc_date': doc['date'],
                            'amount': doc.get('amount', 0),
                            'counterparty': doc.get('counterparty_name', 'Не указан'),
                            'days_since_doc': (datetime.now() - datetime.strptime(doc['date'], '%Y-%m-%d')).days,
                            'orphaned_payment_ids': doc['payment_ids']
                        })
        except ImportError:
            print("Модуль finances не найден")
        
        return orphaned_docs
    
    def get_control_report(self) -> Dict:
        """Получение полного отчёта по контролю"""
        unclosed_payments = self.check_unclosed_payments()
        orphaned_docs = self.check_documents_without_payment()
        
        # Статистика
        total_unclosed_amount = sum(p['amount'] for p in unclosed_payments)
        total_orphaned_amount = sum(d['amount'] for d in orphaned_docs)
        
        # Критические случаи (более 30 дней)
        critical_unclosed = [p for p in unclosed_payments if p['days_since_payment'] > 30]
        critical_orphaned = [d for d in orphaned_docs if d['days_since_doc'] > 30]
        
        return {
            'unclosed_payments': unclosed_payments,
            'orphaned_documents': orphaned_docs,
            'total_unclosed_amount': total_unclosed_amount,
            'total_orphaned_amount': total_orphaned_amount,
            'critical_unclosed_count': len(critical_unclosed),
            'critical_orphaned_count': len(critical_orphaned),
            'report_date': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(unclosed_payments) + len(orphaned_docs),
                'total_amount_at_risk': total_unclosed_amount + total_orphaned_amount
            }
        }
    
    def format_telegram_report(self, report: Dict) -> str:
        """Форматирование отчёта для Telegram"""
        text = "🔍 <b>Отчёт по контролю платежей и документов</b>\n\n"
        
        # Сводка
        text += f"📊 <b>Сводка:</b>\n"
        text += f"• Всего проблем: {report['summary']['total_issues']}\n"
        text += f"• Сумма под риском: {report['summary']['total_amount_at_risk']:,} ₽\n"
        text += f"• Критических случаев: {report['critical_unclosed_count'] + report['critical_orphaned_count']}\n\n"
        
        # Незакрытые платежи
        if report['unclosed_payments']:
            text += f"⚠️ <b>Незакрытые платежи ({len(report['unclosed_payments'])}):</b>\n"
            for payment in report['unclosed_payments'][:5]:  # Показываем первые 5
                critical_mark = "🚨 " if payment['days_since_payment'] > 30 else ""
                text += f"{critical_mark}💰 {payment['amount']:,} ₽ — {payment['counterparty']}\n"
                text += f"   📅 {payment['date']} ({payment['days_since_payment']} дн. назад)\n"
                text += f"   📋 Не хватает: {', '.join(payment['missing_docs'])}\n"
                text += f"   🏢 Проект: {payment['project']}\n\n"
            
            if len(report['unclosed_payments']) > 5:
                text += f"... и ещё {len(report['unclosed_payments']) - 5} платежей\n\n"
        
        # Документы без оплаты
        if report['orphaned_documents']:
            text += f"⚠️ <b>Документы без оплаты ({len(report['orphaned_documents'])}):</b>\n"
            for doc in report['orphaned_documents'][:5]:  # Показываем первые 5
                critical_mark = "🚨 " if doc['days_since_doc'] > 30 else ""
                text += f"{critical_mark}📄 {doc['doc_type'].title()} №{doc['doc_number']}\n"
                text += f"   💰 {doc['amount']:,} ₽ — {doc['counterparty']}\n"
                text += f"   📅 {doc['doc_date']} ({doc['days_since_doc']} дн. назад)\n\n"
            
            if len(report['orphaned_documents']) > 5:
                text += f"... и ещё {len(report['orphaned_documents']) - 5} документов\n\n"
        
        # Рекомендации
        text += "💡 <b>Рекомендации:</b>\n"
        if report['unclosed_payments']:
            text += "• Запросите недостающие документы у контрагентов\n"
        if report['orphaned_documents']:
            text += "• Проверьте, были ли произведены платежи по документам\n"
        if report['critical_unclosed_count'] > 0 or report['critical_orphaned_count'] > 0:
            text += "• Обратите внимание на критические случаи (более 30 дней)\n"
        
        text += f"\n📅 Отчёт сформирован: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        
        return text
    
    def get_weekly_summary(self) -> str:
        """Получение еженедельной сводки"""
        report = self.get_control_report()
        
        text = "📈 <b>Еженедельная сводка по контролю</b>\n\n"
        
        # Основные метрики
        text += f"🔢 <b>Метрики за неделю:</b>\n"
        text += f"• Незакрытых платежей: {len(report['unclosed_payments'])}\n"
        text += f"• Документов без оплаты: {len(report['orphaned_documents'])}\n"
        text += f"• Сумма под риском: {report['summary']['total_amount_at_risk']:,} ₽\n\n"
        
        # Топ проблемных контрагентов
        counterparty_issues = {}
        for payment in report['unclosed_payments']:
            counterparty = payment['counterparty']
            if counterparty not in counterparty_issues:
                counterparty_issues[counterparty] = {'payments': 0, 'amount': 0}
            counterparty_issues[counterparty]['payments'] += 1
            counterparty_issues[counterparty]['amount'] += payment['amount']
        
        if counterparty_issues:
            text += "🏢 <b>Топ проблемных контрагентов:</b>\n"
            sorted_counterparties = sorted(counterparty_issues.items(), 
                                         key=lambda x: x[1]['amount'], reverse=True)
            
            for counterparty, data in sorted_counterparties[:3]:
                text += f"• {counterparty}: {data['payments']} платежей, {data['amount']:,} ₽\n"
        
        return text
    
    def should_send_notification(self) -> bool:
        """Проверка, нужно ли отправлять уведомление"""
        if not self.control_data['rules']['auto_notify']:
            return False
        
        last_check = self.control_data.get('last_check')
        if not last_check:
            return True
        
        last_check_date = datetime.fromisoformat(last_check)
        days_since_check = (datetime.now() - last_check_date).days
        
        return days_since_check >= self.control_data['rules']['check_frequency_days']
    
    def mark_notification_sent(self):
        """Отметка о том, что уведомление отправлено"""
        self.control_data['last_check'] = datetime.now().isoformat()
        self.control_data['notifications_sent'].append({
            'date': datetime.now().isoformat(),
            'type': 'weekly_report'
        })
        self._save_data()
    
    def get_critical_alerts(self) -> List[str]:
        """Получение критических уведомлений"""
        alerts = []
        report = self.get_control_report()
        
        # Критические незакрытые платежи
        critical_payments = [p for p in report['unclosed_payments'] if p['days_since_payment'] > 30]
        if critical_payments:
            total_amount = sum(p['amount'] for p in critical_payments)
            alerts.append(f"🚨 КРИТИЧНО! {len(critical_payments)} незакрытых платежей на {total_amount:,} ₽ (более 30 дней)")
        
        # Критические документы без оплаты
        critical_docs = [d for d in report['orphaned_documents'] if d['days_since_doc'] > 30]
        if critical_docs:
            total_amount = sum(d['amount'] for d in critical_docs)
            alerts.append(f"🚨 КРИТИЧНО! {len(critical_docs)} документов без оплаты на {total_amount:,} ₽ (более 30 дней)")
        
        return alerts

    def create_tasks_from_report(self):
        """Создает задачи в TaskManager на основе отчета о контроле."""
        report = self.get_control_report()
        now = datetime.now()
        
        # Задачи по незакрытым платежам
        for payment in report.get('unclosed_payments', []):
            due_date = (datetime.strptime(payment['date'], '%Y-%m-%d') + timedelta(days=14)).isoformat()
            task_manager.add_task(
                title=f"Запросить документы по платежу для {payment['counterparty']}",
                description=(
                    f"Платеж на сумму {payment['amount']:,} ₽ от {payment['date']}. "
                    f"Отсутствуют документы: {', '.join(payment['missing_docs'])}."
                ),
                source='payment_control',
                source_id=f"payment_{payment['payment_id']}",
                due_date=due_date,
                tags=['финансы', 'документы'],
                priority=2 # Высокий приоритет
            )
            
        # Задачи по документам без оплаты
        for doc in report.get('orphaned_documents', []):
            due_date = (datetime.strptime(doc['doc_date'], '%Y-%m-%d') + timedelta(days=14)).isoformat()
            task_manager.add_task(
                title=f"Проверить оплату по документу от {doc['counterparty']}",
                description=(
                    f"Документ {doc['doc_type']} №{doc['doc_number']} на сумму {doc['amount']:,} ₽ от {doc['doc_date']}. "
                    "Не найден связанный платеж."
                ),
                source='payment_control',
                source_id=f"doc_{doc['doc_id']}",
                due_date=due_date,
                tags=['финансы', 'оплата'],
                priority=2
            )

# Глобальный экземпляр
payment_control = PaymentControl() 