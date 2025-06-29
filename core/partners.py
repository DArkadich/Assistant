"""
PartnersManager — ведение партнёрской сети
Управление партнёрами, фильтрация, автогенерация предложений
"""

import gspread
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import openai
import os
import re

class PartnerStatus:
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROSPECT = "prospect"
    LEAD = "lead"
    PARTNER = "partner"

class PartnerChannel:
    TELEGRAM = "telegram"
    EMAIL = "email"
    PHONE = "phone"
    LINKEDIN = "linkedin"
    WEBSITE = "website"
    REFERRAL = "referral"

class PartnersManager:
    def __init__(self, sheet_name: str = "Partners"):
        self.sheet_name = sheet_name
        self.gc = gspread.service_account(filename="service_account.json")
        self.workbook = self.gc.open(sheet_name)
        self.sheet = self.workbook.sheet1
        self.header = self.sheet.row_values(1)
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Инициализация OpenAI клиента."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI клиент инициализирован для партнёров")
            else:
                print("⚠️ OpenAI API ключ не найден")
        except Exception as e:
            print(f"❌ Ошибка инициализации OpenAI: {e}")
    
    def _parse_partner_row(self, row: List[str]) -> Optional[Dict]:
        """Парсинг строки партнёра из таблицы."""
        try:
            if len(row) < 6:
                return None
            
            return {
                "id": row[0] if row[0] else str(len(self.sheet.get_all_values())),
                "name": row[1] if row[1] else "",
                "channel": row[2] if row[2] else "",
                "result": row[3] if row[3] else "",
                "contacts": row[4] if row[4] else "",
                "status": row[5] if row[5] else PartnerStatus.PROSPECT,
                "segment": row[6] if len(row) > 6 and row[6] else "general",
                "last_contact": row[7] if len(row) > 7 and row[7] else "",
                "notes": row[8] if len(row) > 8 and row[8] else ""
            }
        except Exception as e:
            print(f"❌ Ошибка парсинга партнёра: {e}")
            return None
    
    def get_all_partners(self) -> List[Dict]:
        """Получение всех партнёров."""
        try:
            rows = self.sheet.get_all_values()[1:]  # Пропускаем заголовок
            partners = []
            for row in rows:
                partner = self._parse_partner_row(row)
                if partner:
                    partners.append(partner)
            return partners
        except Exception as e:
            print(f"❌ Ошибка получения партнёров: {e}")
            return []
    
    def add_partner(self, name: str, channel: str, contacts: str, 
                   status: str = PartnerStatus.PROSPECT, segment: str = "general") -> bool:
        """Добавление нового партнёра."""
        try:
            partner_id = str(len(self.sheet.get_all_values()) + 1)
            row = [
                partner_id,
                name,
                channel,
                "",  # result
                contacts,
                status,
                segment,
                datetime.now().strftime("%Y-%m-%d"),  # last_contact
                ""  # notes
            ]
            self.sheet.append_row(row)
            print(f"✅ Партнёр {name} добавлен")
            return True
        except Exception as e:
            print(f"❌ Ошибка добавления партнёра: {e}")
            return False
    
    def update_partner(self, partner_id: str, **kwargs) -> bool:
        """Обновление данных партнёра."""
        try:
            # Находим строку партнёра
            rows = self.sheet.get_all_values()
            for i, row in enumerate(rows):
                if row[0] == partner_id:
                    # Обновляем поля
                    for key, value in kwargs.items():
                        if key == "name" and len(row) > 1:
                            self.sheet.update_cell(i + 1, 2, value)
                        elif key == "channel" and len(row) > 2:
                            self.sheet.update_cell(i + 1, 3, value)
                        elif key == "result" and len(row) > 3:
                            self.sheet.update_cell(i + 1, 4, value)
                        elif key == "contacts" and len(row) > 4:
                            self.sheet.update_cell(i + 1, 5, value)
                        elif key == "status" and len(row) > 5:
                            self.sheet.update_cell(i + 1, 6, value)
                        elif key == "segment" and len(row) > 6:
                            self.sheet.update_cell(i + 1, 7, value)
                        elif key == "last_contact" and len(row) > 7:
                            self.sheet.update_cell(i + 1, 8, value)
                        elif key == "notes" and len(row) > 8:
                            self.sheet.update_cell(i + 1, 9, value)
                    
                    print(f"✅ Партнёр {partner_id} обновлён")
                    return True
            
            print(f"❌ Партнёр {partner_id} не найден")
            return False
        except Exception as e:
            print(f"❌ Ошибка обновления партнёра: {e}")
            return False
    
    def filter_partners(self, **filters) -> List[Dict]:
        """Фильтрация партнёров по критериям."""
        partners = self.get_all_partners()
        filtered = []
        
        for partner in partners:
            match = True
            
            # Фильтр по статусу
            if "status" in filters and partner["status"] != filters["status"]:
                match = False
            
            # Фильтр по каналу
            if "channel" in filters and partner["channel"] != filters["channel"]:
                match = False
            
            # Фильтр по сегменту
            if "segment" in filters and partner["segment"] != filters["segment"]:
                match = False
            
            # Фильтр по результату (пустой/непустой)
            if "has_result" in filters:
                if filters["has_result"] and not partner["result"]:
                    match = False
                elif not filters["has_result"] and partner["result"]:
                    match = False
            
            # Фильтр по последнему контакту
            if "days_since_contact" in filters:
                if partner["last_contact"]:
                    try:
                        last_contact = datetime.strptime(partner["last_contact"], "%Y-%m-%d")
                        days_diff = (datetime.now() - last_contact).days
                        if days_diff < filters["days_since_contact"]:
                            match = False
                    except:
                        pass
            
            if match:
                filtered.append(partner)
        
        return filtered
    
    def get_partners_for_calling(self, days_since_contact: int = 7) -> List[Dict]:
        """Получение партнёров для прозвона."""
        return self.filter_partners(
            status=PartnerStatus.PROSPECT,
            days_since_contact=days_since_contact
        )
    
    def get_partners_for_emailing(self, segment: str = None) -> List[Dict]:
        """Получение партнёров для рассылки."""
        filters = {"channel": PartnerChannel.EMAIL}
        if segment:
            filters["segment"] = segment
        return self.filter_partners(**filters)
    
    def generate_proposal(self, partner: Dict, segment: str = None) -> str:
        """Генерация предложения для партнёра."""
        if not self.openai_client:
            return "OpenAI не настроен для генерации предложений"
        
        try:
            segment_info = self._get_segment_info(segment or partner.get("segment", "general"))
            
            prompt = f"""
Создай персонализированное предложение для партнёрства:

Партнёр: {partner['name']}
Канал: {partner['channel']}
Сегмент: {partner.get('segment', 'general')}
Результат: {partner.get('result', 'Нет')}

Информация о сегменте:
{segment_info}

Предложение должно быть:
1. Персонализированным под партнёра
2. Кратким и по делу
3. С конкретными выгодами
4. С призывом к действию
5. Профессиональным тоном

Формат: приветствие + предложение + выгоды + призыв к действию
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты эксперт по партнёрским отношениям. Создавай убедительные и персонализированные предложения."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Ошибка генерации предложения: {e}"
    
    def _get_segment_info(self, segment: str) -> str:
        """Получение информации о сегменте."""
        segment_info = {
            "startup": "Стартапы и инновационные проекты. Предлагаем техническую экспертизу, доступ к ресурсам, совместное развитие продуктов.",
            "enterprise": "Крупные компании. Предлагаем масштабируемые решения, интеграцию с существующими системами, долгосрочное партнёрство.",
            "agency": "Маркетинговые и консалтинговые агентства. Предлагаем реферальную программу, совместные проекты, обмен клиентами.",
            "developer": "Разработчики и IT-специалисты. Предлагаем техническое партнёрство, API доступ, совместную разработку.",
            "general": "Общие партнёры. Предлагаем взаимовыгодное сотрудничество, обмен опытом, совместные проекты."
        }
        return segment_info.get(segment, segment_info["general"])
    
    def get_partners_summary(self) -> Dict:
        """Получение сводки по партнёрам."""
        partners = self.get_all_partners()
        
        summary = {
            "total": len(partners),
            "by_status": {},
            "by_channel": {},
            "by_segment": {},
            "needs_contact": 0,
            "active_partners": 0
        }
        
        for partner in partners:
            # По статусу
            status = partner["status"]
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            # По каналу
            channel = partner["channel"]
            summary["by_channel"][channel] = summary["by_channel"].get(channel, 0) + 1
            
            # По сегменту
            segment = partner.get("segment", "general")
            summary["by_segment"][segment] = summary["by_segment"].get(segment, 0) + 1
            
            # Активные партнёры
            if status in [PartnerStatus.ACTIVE, PartnerStatus.PARTNER]:
                summary["active_partners"] += 1
            
            # Нуждаются в контакте
            if not partner["result"] and status == PartnerStatus.PROSPECT:
                summary["needs_contact"] += 1
        
        return summary
    
    def generate_bulk_proposals(self, segment: str = None, limit: int = 10) -> List[Dict]:
        """Генерация предложений для группы партнёров."""
        if segment:
            partners = self.filter_partners(segment=segment, has_result=False)
        else:
            partners = self.filter_partners(has_result=False)
        
        proposals = []
        for partner in partners[:limit]:
            proposal = self.generate_proposal(partner, segment)
            proposals.append({
                "partner": partner,
                "proposal": proposal
            })
        
        return proposals

# Глобальный экземпляр
partners_manager = PartnersManager() 