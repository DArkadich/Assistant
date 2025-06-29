"""
Тест AmoCRM интеграции
Проверка подключения и основных функций
"""

import os
import sys
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.amocrm import amocrm

def test_amocrm_connection():
    """Тест подключения к AmoCRM."""
    print("🔗 Тестирование подключения к AmoCRM...")
    
    # Проверяем наличие переменных окружения
    required_vars = [
        'AMOCRM_BASE_URL',
        'AMOCRM_CLIENT_ID', 
        'AMOCRM_CLIENT_SECRET',
        'AMOCRM_ACCESS_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("📝 Создайте файл .env с переменными AmoCRM")
        return False
    
    print("✅ Переменные окружения настроены")
    return True

def test_amocrm_contacts():
    """Тест получения контактов."""
    print("\n👥 Тестирование получения контактов...")
    
    try:
        contacts = amocrm.get_contacts(limit=5)
        
        if contacts is None:
            print("❌ Ошибка получения контактов")
            return False
        
        print(f"✅ Получено контактов: {len(contacts)}")
        
        if contacts:
            print("📋 Примеры контактов:")
            for i, contact in enumerate(contacts[:3], 1):
                print(f"  {i}. {contact['name']} (ID: {contact['id']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_amocrm_leads():
    """Тест получения сделок."""
    print("\n💼 Тестирование получения сделок...")
    
    try:
        leads = amocrm.get_leads(limit=5)
        
        if leads is None:
            print("❌ Ошибка получения сделок")
            return False
        
        print(f"✅ Получено сделок: {len(leads)}")
        
        if leads:
            print("📋 Примеры сделок:")
            for i, lead in enumerate(leads[:3], 1):
                print(f"  {i}. {lead['name']} (ID: {lead['id']}, Сумма: {lead.get('price', 0)} ₽)")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_amocrm_pipelines():
    """Тест получения воронок."""
    print("\n🔄 Тестирование получения воронок...")
    
    try:
        pipelines = amocrm.get_pipelines()
        
        if pipelines is None:
            print("❌ Ошибка получения воронок")
            return False
        
        print(f"✅ Получено воронок: {len(pipelines)}")
        
        if pipelines:
            print("📋 Воронки продаж:")
            for i, pipeline in enumerate(pipelines, 1):
                print(f"  {i}. {pipeline['name']} (ID: {pipeline['id']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_amocrm_analytics():
    """Тест получения аналитики."""
    print("\n📊 Тестирование получения аналитики...")
    
    try:
        analytics = amocrm.get_analytics(period="month")
        
        if analytics is None:
            print("❌ Ошибка получения аналитики")
            return False
        
        print("✅ Аналитика получена:")
        print(f"  📈 Всего лидов: {analytics['total_leads']}")
        print(f"  ✅ Выигранных сделок: {analytics['won_leads']}")
        print(f"  📊 Конверсия: {analytics['conversion_rate']:.1f}%")
        print(f"  💰 Общая выручка: {analytics['total_revenue']} ₽")
        print(f"  💎 Средний чек: {analytics['avg_deal_size']:.0f} ₽")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_amocrm_create_contact():
    """Тест создания контакта."""
    print("\n👤 Тестирование создания контакта...")
    
    try:
        test_name = f"Тест контакт {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_email = "test@example.com"
        test_phone = "+7-999-123-45-67"
        
        contact = amocrm.create_contact(
            name=test_name,
            email=test_email,
            phone=test_phone
        )
        
        if contact is None:
            print("❌ Ошибка создания контакта")
            return False
        
        print(f"✅ Контакт создан:")
        print(f"  👤 Имя: {contact['name']}")
        print(f"  🆔 ID: {contact['id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_amocrm_create_lead():
    """Тест создания сделки."""
    print("\n💼 Тестирование создания сделки...")
    
    try:
        test_name = f"Тест сделка {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        lead = amocrm.create_lead(
            name=test_name,
            custom_fields={1: 50000}  # Сумма сделки
        )
        
        if lead is None:
            print("❌ Ошибка создания сделки")
            return False
        
        print(f"✅ Сделка создана:")
        print(f"  💼 Название: {lead['name']}")
        print(f"  🆔 ID: {lead['id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🧪 Тестирование AmoCRM интеграции")
    print("=" * 50)
    
    tests = [
        ("Подключение", test_amocrm_connection),
        ("Контакты", test_amocrm_contacts),
        ("Сделки", test_amocrm_leads),
        ("Воронки", test_amocrm_pipelines),
        ("Аналитика", test_amocrm_analytics),
        ("Создание контакта", test_amocrm_create_contact),
        ("Создание сделки", test_amocrm_create_lead),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчёт
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены! AmoCRM интеграция работает корректно.")
    else:
        print("⚠️ Некоторые тесты не пройдены. Проверьте настройки AmoCRM.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 