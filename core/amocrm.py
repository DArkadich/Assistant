"""
AmoCRM Integration — интеграция с AmoCRM
Управление контактами, сделками, воронкой продаж
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

class AmoCRM:
    def __init__(self):
        self.base_url = os.getenv('AMOCRM_BASE_URL', 'https://your-domain.amocrm.ru')
        self.client_id = os.getenv('AMOCRM_CLIENT_ID')
        self.client_secret = os.getenv('AMOCRM_CLIENT_SECRET')
        self.access_token = os.getenv('AMOCRM_ACCESS_TOKEN')
        self.refresh_token = os.getenv('AMOCRM_REFRESH_TOKEN')
        
        if not all([self.base_url, self.client_id, self.client_secret]):
            print("⚠️ AmoCRM не настроен. Установите переменные окружения:")
            print("AMOCRM_BASE_URL, AMOCRM_CLIENT_ID, AMOCRM_CLIENT_SECRET")
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Optional[Dict]:
        """Выполнение запроса к AmoCRM API."""
        if not self.access_token:
            print("❌ AmoCRM access token не найден")
            return None
        
        url = f"{self.base_url}/api/v4/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PATCH':
                response = requests.patch(url, headers=headers, json=data)
            else:
                print(f"❌ Неподдерживаемый метод: {method}")
                return None
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Попытка обновить токен
                if self._refresh_access_token():
                    return self._make_request(method, endpoint, data, params)
                else:
                    print("❌ Не удалось обновить access token")
                    return None
            else:
                print(f"❌ Ошибка AmoCRM API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка запроса к AmoCRM: {e}")
            return None
    
    def _refresh_access_token(self) -> bool:
        """Обновление access token."""
        if not self.refresh_token:
            return False
        
        try:
            url = f"{self.base_url}/oauth2/access_token"
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token
            }
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data['refresh_token']
                
                # Обновляем переменные окружения (для текущей сессии)
                os.environ['AMOCRM_ACCESS_TOKEN'] = self.access_token
                os.environ['AMOCRM_REFRESH_TOKEN'] = self.refresh_token
                
                print("✅ Access token обновлён")
                return True
            else:
                print(f"❌ Ошибка обновления токена: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка обновления токена: {e}")
            return False
    
    def get_contacts(self, limit: int = 50, query: str = None) -> List[Dict]:
        """Получение контактов."""
        params = {'limit': limit}
        if query:
            params['query'] = query
        
        response = self._make_request('GET', 'contacts', params=params)
        if response and '_embedded' in response:
            return response['_embedded']['contacts']
        return []
    
    def create_contact(self, name: str, email: str = None, phone: str = None, 
                      custom_fields: Dict = None) -> Optional[Dict]:
        """Создание нового контакта."""
        data = [{
            'name': name,
            'custom_fields_values': []
        }]
        
        # Добавляем email
        if email:
            data[0]['custom_fields_values'].append({
                'field_id': 1,  # ID поля email (может отличаться)
                'values': [{'value': email, 'enum_code': 'WORK'}]
            })
        
        # Добавляем телефон
        if phone:
            data[0]['custom_fields_values'].append({
                'field_id': 2,  # ID поля телефон (может отличаться)
                'values': [{'value': phone, 'enum_code': 'WORK'}]
            })
        
        # Добавляем кастомные поля
        if custom_fields:
            for field_id, value in custom_fields.items():
                data[0]['custom_fields_values'].append({
                    'field_id': field_id,
                    'values': [{'value': value}]
                })
        
        response = self._make_request('POST', 'contacts', data=data)
        if response and '_embedded' in response:
            return response['_embedded']['contacts'][0]
        return None
    
    def update_contact(self, contact_id: int, **kwargs) -> bool:
        """Обновление контакта."""
        data = [{'id': contact_id}]
        
        if 'name' in kwargs:
            data[0]['name'] = kwargs['name']
        
        if 'custom_fields_values' in kwargs:
            data[0]['custom_fields_values'] = kwargs['custom_fields_values']
        
        response = self._make_request('PATCH', 'contacts', data=data)
        return response is not None
    
    def get_leads(self, limit: int = 50, status_id: int = None) -> List[Dict]:
        """Получение сделок."""
        params = {'limit': limit}
        if status_id:
            params['filter[statuses][0][pipeline_id]'] = status_id
        
        response = self._make_request('GET', 'leads', params=params)
        if response and '_embedded' in response:
            return response['_embedded']['leads']
        return []
    
    def create_lead(self, name: str, contact_id: int = None, 
                   custom_fields: Dict = None) -> Optional[Dict]:
        """Создание новой сделки."""
        data = [{
            'name': name,
            'price': 0
        }]
        
        if contact_id:
            data[0]['_embedded'] = {
                'contacts': [{'id': contact_id}]
            }
        
        if custom_fields:
            data[0]['custom_fields_values'] = []
            for field_id, value in custom_fields.items():
                data[0]['custom_fields_values'].append({
                    'field_id': field_id,
                    'values': [{'value': value}]
                })
        
        response = self._make_request('POST', 'leads', data=data)
        if response and '_embedded' in response:
            return response['_embedded']['leads'][0]
        return None
    
    def update_lead_status(self, lead_id: int, status_id: int, pipeline_id: int = None) -> bool:
        """Обновление статуса сделки."""
        data = [{
            'id': lead_id,
            'status_id': status_id
        }]
        
        if pipeline_id:
            data[0]['pipeline_id'] = pipeline_id
        
        response = self._make_request('PATCH', 'leads', data=data)
        return response is not None
    
    def get_pipelines(self) -> List[Dict]:
        """Получение воронок продаж."""
        response = self._make_request('GET', 'leads/pipelines')
        if response and '_embedded' in response:
            return response['_embedded']['pipelines']
        return []
    
    def get_pipeline_statuses(self, pipeline_id: int) -> List[Dict]:
        """Получение статусов воронки."""
        response = self._make_request('GET', f'leads/pipelines/{pipeline_id}/statuses')
        if response and '_embedded' in response:
            return response['_embedded']['statuses']
        return []
    
    def get_tasks(self, limit: int = 50, responsible_user_id: int = None) -> List[Dict]:
        """Получение задач."""
        params = {'limit': limit}
        if responsible_user_id:
            params['filter[responsible_user_id]'] = responsible_user_id
        
        response = self._make_request('GET', 'tasks', params=params)
        if response and '_embedded' in response:
            return response['_embedded']['tasks']
        return []
    
    def create_task(self, text: str, entity_type: str, entity_id: int, 
                   complete_till: int = None, responsible_user_id: int = None) -> Optional[Dict]:
        """Создание новой задачи."""
        data = [{
            'text': text,
            'entity_type': entity_type,
            'entity_id': entity_id
        }]
        
        if complete_till:
            data[0]['complete_till'] = complete_till
        
        if responsible_user_id:
            data[0]['responsible_user_id'] = responsible_user_id
        
        response = self._make_request('POST', 'tasks', data=data)
        if response and '_embedded' in response:
            return response['_embedded']['tasks'][0]
        return None
    
    def get_analytics(self, period: str = "month") -> Dict:
        """Получение аналитики."""
        # Определяем период
        end_date = datetime.now()
        if period == "week":
            start_date = end_date - timedelta(days=7)
        elif period == "month":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Получаем сделки за период
        leads = self.get_leads(limit=1000)
        period_leads = []
        
        for lead in leads:
            created_at = datetime.fromtimestamp(lead.get('created_at', 0))
            if start_date <= created_at <= end_date:
                period_leads.append(lead)
        
        # Аналитика
        total_leads = len(period_leads)
        won_leads = len([l for l in period_leads if l.get('status_id') == 142])  # ID статуса "Успешно реализовано"
        total_revenue = sum([l.get('price', 0) for l in period_leads if l.get('status_id') == 142])
        
        return {
            'period': period,
            'total_leads': total_leads,
            'won_leads': won_leads,
            'conversion_rate': (won_leads / total_leads * 100) if total_leads > 0 else 0,
            'total_revenue': total_revenue,
            'avg_deal_size': (total_revenue / won_leads) if won_leads > 0 else 0
        }
    
    def sync_partners_from_sheet(self, partners_manager) -> Dict:
        """Синхронизация партнёров из Google Sheets в AmoCRM."""
        partners = partners_manager.get_all_partners()
        synced = {'created': 0, 'updated': 0, 'errors': 0}
        
        for partner in partners:
            try:
                # Проверяем, есть ли уже контакт с таким именем
                existing_contacts = self.get_contacts(query=partner['name'])
                
                if existing_contacts:
                    # Обновляем существующий контакт
                    contact = existing_contacts[0]
                    success = self.update_contact(
                        contact['id'],
                        custom_fields_values=[
                            {
                                'field_id': 3,  # Канал
                                'values': [{'value': partner['channel']}]
                            },
                            {
                                'field_id': 4,  # Результат
                                'values': [{'value': partner.get('result', '')}]
                            },
                            {
                                'field_id': 5,  # Сегмент
                                'values': [{'value': partner.get('segment', 'general')}]
                            }
                        ]
                    )
                    if success:
                        synced['updated'] += 1
                    else:
                        synced['errors'] += 1
                else:
                    # Создаём новый контакт
                    contact = self.create_contact(
                        name=partner['name'],
                        custom_fields={
                            3: partner['channel'],  # Канал
                            4: partner.get('result', ''),  # Результат
                            5: partner.get('segment', 'general')  # Сегмент
                        }
                    )
                    if contact:
                        synced['created'] += 1
                    else:
                        synced['errors'] += 1
                        
            except Exception as e:
                print(f"❌ Ошибка синхронизации партнёра {partner['name']}: {e}")
                synced['errors'] += 1
        
        return synced

# Глобальный экземпляр
amocrm = AmoCRM() 