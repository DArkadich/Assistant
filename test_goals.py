#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации модуля целей и KPI
"""

from core.goals import goals_manager, GoalType, GoalPeriod
from datetime import datetime, timedelta

def test_goals_module():
    """Тестирование модуля целей."""
    print("🎯 Тестирование модуля целей и KPI")
    print("=" * 50)
    
    # Очищаем существующие цели
    goals_manager.goals.clear()
    
    # Создаем тестовые цели
    print("\n1. Создание целей:")
    
    # Цель по выручке
    revenue_goal_id = goals_manager.create_goal(
        name="выручка 3 млн",
        description="Цель по выручке 3 млн рублей до сентября",
        goal_type=GoalType.REVENUE,
        target_value=3000000,
        end_date=(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
    )
    print(f"✅ Создана цель по выручке: {revenue_goal_id}")
    
    # Цель по подпискам
    subs_goal_id = goals_manager.create_goal(
        name="подписки 100 клиентов",
        description="Цель по привлечению 100 клиентов",
        goal_type=GoalType.SUBSCRIPTIONS,
        target_value=100,
        end_date=(datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
    )
    print(f"✅ Создана цель по подпискам: {subs_goal_id}")
    
    # Цель по производству
    prod_goal_id = goals_manager.create_goal(
        name="производство 1000 шт",
        description="Цель по производству 1000 единиц продукции",
        goal_type=GoalType.PRODUCTION,
        target_value=1000,
        end_date=(datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d')
    )
    print(f"✅ Создана цель по производству: {prod_goal_id}")
    
    # Обновляем прогресс
    print("\n2. Обновление прогресса:")
    
    goals_manager.update_goal_progress(revenue_goal_id, 1500000, "Первое обновление")
    goals_manager.update_goal_progress(subs_goal_id, 45, "Текущие подписки")
    goals_manager.update_goal_progress(prod_goal_id, 600, "Произведено единиц")
    
    print("✅ Прогресс обновлен")
    
    # Проверяем прогресс
    print("\n3. Проверка прогресса:")
    
    for goal_id in [revenue_goal_id, subs_goal_id, prod_goal_id]:
        goal = goals_manager.get_goal(goal_id)
        progress = goals_manager.get_goal_progress(goal_id)
        
        print(f"\n🎯 {goal.name}:")
        print(f"   Прогресс: {progress['progress_percentage']}%")
        print(f"   Тренд: {progress['trend']['direction']} ({progress['trend']['rate']}/день)")
        print(f"   Статус: {'По плану' if progress['is_on_track'] else 'Отставание'}")
    
    # Список всех целей
    print("\n4. Список всех целей:")
    goals = goals_manager.get_all_goals()
    for goal in goals:
        print(f"   • {goal.name}: {goal.current_value}/{goal.target_value}")
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    test_goals_module() 