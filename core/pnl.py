"""
PnLManager — учет прибыли и убытков (P&L) с интеграцией Google Sheets
"""
import gspread
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import dateparser

class PnLManager:
    def __init__(self, sheet_name: str = "P&L"):
        self.sheet_name = sheet_name
        self.gc = gspread.service_account(filename="service_account.json")
        # Открываем лист по названию
        self.workbook = self.gc.open(sheet_name)
        self.sheet = self.workbook.sheet1
        self.header = self.sheet.row_values(1)

    def _parse_row(self, row: List[str]) -> Dict:
        # Ожидается: Дата, Выручка, Себестоимость, ФОТ, Аренда, Прочие, Прибыль
        try:
            date = dateparser.parse(row[0], languages=["ru"])
            revenue = float(row[1]) if row[1] else 0
            cost = float(row[2]) if row[2] else 0
            fot = float(row[3]) if row[3] else 0
            rent = float(row[4]) if row[4] else 0
            other = float(row[5]) if row[5] else 0
            profit = float(row[6]) if len(row) > 6 and row[6] else revenue - cost - fot - rent - other
            return {
                "date": date,
                "revenue": revenue,
                "cost": cost,
                "fot": fot,
                "rent": rent,
                "other": other,
                "profit": profit
            }
        except Exception as e:
            return None

    def get_rows(self) -> List[Dict]:
        rows = self.sheet.get_all_values()[1:]
        return [self._parse_row(row) for row in rows if self._parse_row(row)]

    def filter_by_period(self, period: str) -> List[Dict]:
        now = datetime.now()
        if period == "вчера":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == "неделя":
            start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        elif period == "месяц":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = (start + timedelta(days=32)).replace(day=1)
        else:
            # Попробуем парсить диапазон дат
            dt = dateparser.parse(period, languages=["ru"])
            if dt:
                start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
            else:
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end = now
        rows = self.get_rows()
        return [row for row in rows if row["date"] and start <= row["date"] < end]

    def get_profit(self, period: str = "месяц") -> Dict:
        rows = self.filter_by_period(period)
        total = {"revenue": 0, "cost": 0, "fot": 0, "rent": 0, "other": 0, "profit": 0}
        for row in rows:
            for k in total:
                total[k] += row[k]
        return total

    def get_full_report(self, period: str = "месяц") -> str:
        total = self.get_profit(period)
        text = f"<b>P&L отчет за {period}:</b>\n"
        text += f"Выручка: <b>{total['revenue']:.2f}</b>\n"
        text += f"Себестоимость: {total['cost']:.2f}\n"
        text += f"ФОТ: {total['fot']:.2f}\n"
        text += f"Аренда: {total['rent']:.2f}\n"
        text += f"Прочие расходы: {total['other']:.2f}\n"
        text += f"<b>Прибыль: {total['profit']:.2f}</b>"
        return text

# Глобальный экземпляр
pnl_manager = PnLManager() 