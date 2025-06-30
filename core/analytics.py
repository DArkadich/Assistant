"""
Модуль бизнес-аналитики: ROI, оборачиваемость, CAC, LTV
"""
from core import pnl, amocrm, finances
from datetime import datetime

class BusinessAnalytics:
    def __init__(self):
        self.pnl = pnl.PnLManager()
        self.amocrm = amocrm.amocrm
        self.finances = finances

    def get_business_metrics(self, period: str = 'месяц') -> dict:
        # 1. P&L данные
        pnl_data = self.pnl.get_profit(period)
        revenue = pnl_data['revenue']
        cost = pnl_data['cost']
        profit = pnl_data['profit']
        # 2. Маркетинговые расходы (по категории)
        marketing_expense = 0
        for op in self.finances.operations:
            if op['type'] == 'expense' and 'маркет' in op.get('category', '').lower():
                op_date = datetime.strptime(op['date'], '%Y-%m-%d')
                # Фильтр по периоду (грубо: месяц)
                if period == 'месяц' and op_date.month == datetime.now().month:
                    marketing_expense += op['amount']
        # 3. AmoCRM — клиенты и сделки
        leads = self.amocrm.get_leads(limit=1000)
        contacts = self.amocrm.get_contacts()
        # Новые клиенты за период (по дате создания)
        now = datetime.now()
        new_clients = [c for c in contacts if 'created_at' in c and datetime.fromtimestamp(c['created_at']).month == now.month]
        num_new_clients = len(new_clients)
        # Сделки за период
        period_leads = [l for l in leads if 'created_at' in l and datetime.fromtimestamp(l['created_at']).month == now.month]
        won_leads = [l for l in period_leads if l.get('status_id') == 142]
        total_revenue = sum(l.get('price', 0) for l in won_leads)
        avg_deal = (total_revenue / len(won_leads)) if won_leads else 0
        # 4. Метрики
        roi = (profit / (cost + marketing_expense) * 100) if (cost + marketing_expense) > 0 else 0
        # Оборачиваемость (продажи/средний запас — если нет запаса, считаем как revenue/cost)
        turnover = (revenue / cost) if cost > 0 else 0
        cac = (marketing_expense / num_new_clients) if num_new_clients > 0 else 0
        # LTV: средний чек * среднее число покупок на клиента (грубо: total_revenue/num_new_clients)
        ltv = (total_revenue / num_new_clients) if num_new_clients > 0 else 0
        return {
            'roi': round(roi, 2),
            'turnover': round(turnover, 2),
            'cac': round(cac, 2),
            'ltv': round(ltv, 2),
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'num_new_clients': num_new_clients,
            'avg_deal': round(avg_deal, 2),
        }

business_analytics = BusinessAnalytics() 