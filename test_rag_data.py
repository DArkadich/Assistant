#!/usr/bin/env python3
"""
Скрипт для добавления тестовых данных в RAG систему
"""

import os
import sys
sys.path.append('.')

from core.finances import *
from core.rag_system import rag_system

def create_test_documents():
    """Создание тестовых документов для демонстрации RAG системы."""
    
    print("🚀 Создание тестовых документов для RAG системы...")
    
    # Тестовые документы
    test_docs = [
        {
            'type': 'контракт',
            'counterparty_name': 'ООО ТехноСофт',
            'amount': 500000,
            'date': '2024-01-15',
            'description': 'Контракт на разработку веб-приложения для управления складом',
            'project': 'СкладСистема',
            'extracted_text': 'Договор на разработку программного обеспечения для автоматизации складского учета. Включает модули приема, отгрузки, инвентаризации и отчетности. Срок выполнения 3 месяца.'
        },
        {
            'type': 'накладная',
            'counterparty_name': 'ИП Петров А.В.',
            'amount': 150000,
            'date': '2024-01-20',
            'description': 'Накладная на поставку компьютерного оборудования',
            'project': 'Офис',
            'extracted_text': 'Товарная накладная на поставку 5 ноутбуков Dell, 3 монитора Samsung, 2 принтера HP. Оплата по факту поставки.'
        },
        {
            'type': 'счёт',
            'counterparty_name': 'ООО Рога и Копыта',
            'amount': 75000,
            'date': '2024-02-01',
            'description': 'Счет за консультационные услуги по бухгалтерскому учету',
            'project': 'Бухгалтерия',
            'extracted_text': 'Счет за консультационные услуги по ведению бухгалтерского учета за январь 2024 года. Включает консультации по налогообложению и подготовку отчетности.'
        },
        {
            'type': 'акт',
            'counterparty_name': 'ООО ТехноСофт',
            'amount': 500000,
            'date': '2024-02-15',
            'description': 'Акт выполненных работ по разработке веб-приложения',
            'project': 'СкладСистема',
            'extracted_text': 'Акт выполненных работ по разработке веб-приложения для управления складом. Работы выполнены в полном объеме и в установленные сроки. Система готова к внедрению.'
        },
        {
            'type': 'гтд',
            'counterparty_name': 'ООО ИмпортТрейд',
            'amount': 300000,
            'date': '2024-02-20',
            'description': 'ГТД на импорт серверного оборудования',
            'project': 'Инфраструктура',
            'extracted_text': 'Грузовая таможенная декларация на импорт серверного оборудования из Китая. Включает 2 сервера Dell PowerEdge, сетевое оборудование Cisco.'
        },
        {
            'type': 'упд',
            'counterparty_name': 'ИП Сидорова Е.М.',
            'amount': 45000,
            'date': '2024-03-01',
            'description': 'УПД за услуги по дизайну логотипа и фирменного стиля',
            'project': 'Маркетинг',
            'extracted_text': 'Универсальный передаточный документ за услуги по разработке дизайна логотипа компании и фирменного стиля. Включает макеты для печати и цифрового использования.'
        },
        {
            'type': 'контракт',
            'counterparty_name': 'ООО МаркетПлюс',
            'amount': 200000,
            'date': '2024-03-10',
            'description': 'Контракт на проведение маркетинговой кампании',
            'project': 'Маркетинг',
            'extracted_text': 'Договор на проведение комплексной маркетинговой кампании. Включает разработку стратегии, создание рекламных материалов, размещение в СМИ и социальных сетях.'
        },
        {
            'type': 'накладная',
            'counterparty_name': 'ООО ОфисМаркет',
            'amount': 25000,
            'date': '2024-03-15',
            'description': 'Накладная на поставку канцелярских товаров',
            'project': 'Офис',
            'extracted_text': 'Товарная накладная на поставку канцелярских товаров: бумага А4, ручки, блокноты, папки, степлеры. Для обеспечения офисной работы.'
        }
    ]
    
    # Добавляем документы
    for i, doc_data in enumerate(test_docs, 1):
        try:
            # Генерируем уникальный ID
            doc_id = f"test_{doc_data['type']}_{i}_{int(os.urandom(4).hex(), 16)}"
            
            # Добавляем в JSON
            document = {
                'id': doc_id,
                'type': doc_data['type'],
                'counterparty_name': doc_data['counterparty_name'],
                'amount': doc_data['amount'],
                'date': doc_data['date'],
                'description': doc_data['description'],
                'project': doc_data['project'],
                'created_at': '2024-01-01T00:00:00',
                'drive_file_id': None,
                'extracted_text': doc_data['extracted_text'],
                'keywords': [doc_data['type'], doc_data['project'], doc_data['counterparty_name']]
            }
            
            documents.append(document)
            
            # Добавляем в RAG систему
            rag_content = f"{doc_data['type']} {doc_data['counterparty_name']} {doc_data['description']} {doc_data['extracted_text']}"
            
            rag_metadata = {
                'type': doc_data['type'],
                'counterparty_name': doc_data['counterparty_name'],
                'amount': str(doc_data['amount']),
                'date': doc_data['date'],
                'project': doc_data['project'],
                'drive_file_id': '',
                'has_extracted_text': 'yes'
            }
            
            rag_system.add_document(doc_id, rag_content, rag_metadata)
            print(f"✅ Добавлен тестовый документ {i}: {doc_data['type']} - {doc_data['counterparty_name']}")
            
        except Exception as e:
            print(f"❌ Ошибка добавления документа {i}: {e}")
    
    # Сохраняем в JSON
    save_doc()
    
    print(f"\n🎉 Создано {len(test_docs)} тестовых документов!")
    print("📊 Статистика RAG системы:")
    stats = rag_system.get_collection_stats()
    print(f"   - Всего документов: {stats.get('total_documents', 0)}")
    print(f"   - Коллекция: {stats.get('collection_name', 'N/A')}")
    print(f"   - Статус: {stats.get('status', 'N/A')}")

if __name__ == "__main__":
    create_test_documents() 