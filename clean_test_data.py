#!/usr/bin/env python3
"""
Скрипт для очистки тестовых данных из RAG системы
"""

import os
import sys
sys.path.append('.')

from core.finances import *
from core.rag_system import rag_system

def clean_test_data():
    """Очистка тестовых данных из RAG системы."""
    
    print("🧹 Очистка тестовых данных из RAG системы...")
    
    # Находим тестовые документы
    test_docs = [doc for doc in documents if doc['id'].startswith('test_')]
    
    if not test_docs:
        print("ℹ️ Тестовые документы не найдены.")
        return
    
    print(f"📄 Найдено {len(test_docs)} тестовых документов для удаления...")
    
    # Удаляем из RAG системы и JSON
    removed_count = 0
    for doc in test_docs:
        try:
            # Удаляем из RAG системы
            rag_system.delete_document(doc['id'])
            
            # Удаляем из JSON
            documents.remove(doc)
            
            removed_count += 1
            print(f"✅ Удален тестовый документ: {doc['type']} - {doc['counterparty_name']}")
            
        except Exception as e:
            print(f"❌ Ошибка удаления документа {doc['id']}: {e}")
    
    # Сохраняем изменения в JSON
    save_doc()
    
    print(f"\n🎉 Удалено {removed_count} тестовых документов!")
    print("📊 Статистика RAG системы:")
    stats = rag_system.get_collection_stats()
    print(f"   - Всего документов: {stats.get('total_documents', 0)}")
    print(f"   - Коллекция: {stats.get('collection_name', 'N/A')}")
    print(f"   - Статус: {stats.get('status', 'N/A')}")

def clean_all_data():
    """Полная очистка всех данных из RAG системы."""
    
    print("⚠️ ВНИМАНИЕ: Полная очистка всех данных из RAG системы!")
    confirm = input("Вы уверены? Введите 'yes' для подтверждения: ")
    
    if confirm.lower() != 'yes':
        print("❌ Операция отменена.")
        return
    
    try:
        # Очищаем RAG коллекцию
        rag_system.clear_collection()
        
        # Очищаем JSON документы
        documents.clear()
        save_doc()
        
        print("✅ Все данные очищены!")
        
    except Exception as e:
        print(f"❌ Ошибка очистки: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Очистка тестовых данных из RAG системы')
    parser.add_argument('--all', action='store_true', help='Очистить все данные (не только тестовые)')
    
    args = parser.parse_args()
    
    if args.all:
        clean_all_data()
    else:
        clean_test_data() 