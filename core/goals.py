import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass, asdict
from enum import Enum

class GoalType(Enum):
    REVENUE = "revenue"
    SUBSCRIPTIONS = "subscriptions"
    PRODUCTION = "production"
    CUSTOM = "custom"

class GoalPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class Goal:
    id: str
    name: str
    description: str
    goal_type: GoalType
    target_value: float
    current_value: float = 0.0
    period: GoalPeriod = GoalPeriod.MONTHLY
    start_date: str = None
    end_date: str = None
    created_at: str = None
    updated_at: str = None
    is_active: bool = True
    progress_updates: List[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        if self.progress_updates is None:
            self.progress_updates = []
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['goal_type'] = self.goal_type.value
        data['period'] = self.period.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        data['goal_type'] = GoalType(data['goal_type'])
        data['period'] = GoalPeriod(data['period'])
        return cls(**data)

class GoalsManager:
    def __init__(self, file_path: str = "goals.json"):
        self.file_path = file_path
        self.goals: Dict[str, Goal] = {}
        self._load_goals()
    
    def _load_goals(self):
        """Загрузить цели из файла."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.goals = {
                        goal_id: Goal.from_dict(goal_data)
                        for goal_id, goal_data in data.items()
                    }
        except Exception as e:
            print(f"Ошибка загрузки целей: {e}")
            self.goals = {}
    
    def _save_goals(self):
        """Сохранить цели в файл."""
        try:
            data = {
                goal_id: goal.to_dict()
                for goal_id, goal in self.goals.items()
            }
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения целей: {e}")
    
    def create_goal(self, name: str, description: str, goal_type: GoalType, 
                   target_value: float, period: GoalPeriod = GoalPeriod.MONTHLY,
                   start_date: str = None, end_date: str = None) -> str:
        """Создать новую цель."""
        goal_id = f"goal_{len(self.goals) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        goal = Goal(
            id=goal_id,
            name=name,
            description=description,
            goal_type=goal_type,
            target_value=target_value,
            period=period,
            start_date=start_date,
            end_date=end_date
        )
        
        self.goals[goal_id] = goal
        self._save_goals()
        return goal_id
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Получить цель по ID."""
        return self.goals.get(goal_id)
    
    def get_all_goals(self, active_only: bool = True) -> List[Goal]:
        """Получить все цели."""
        goals = list(self.goals.values())
        if active_only:
            goals = [goal for goal in goals if goal.is_active]
        return goals
    
    def update_goal_progress(self, goal_id: str, new_value: float, 
                           update_note: str = None) -> bool:
        """Обновить прогресс по цели."""
        goal = self.goals.get(goal_id)
        if not goal:
            return False
        
        old_value = goal.current_value
        goal.current_value = new_value
        goal.updated_at = datetime.now().isoformat()
        
        # Добавляем запись об обновлении
        update_record = {
            'date': datetime.now().isoformat(),
            'old_value': old_value,
            'new_value': new_value,
            'change': new_value - old_value,
            'note': update_note
        }
        goal.progress_updates.append(update_record)
        
        self._save_goals()
        return True
    
    def delete_goal(self, goal_id: str) -> bool:
        """Удалить цель."""
        if goal_id in self.goals:
            del self.goals[goal_id]
            self._save_goals()
            return True
        return False
    
    def get_goal_progress(self, goal_id: str) -> Dict[str, Any]:
        """Получить прогресс по цели."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
        
        progress_percentage = (goal.current_value / goal.target_value) * 100 if goal.target_value > 0 else 0
        remaining = goal.target_value - goal.current_value
        
        # Рассчитываем тренд
        trend = self._calculate_trend(goal)
        
        # Рассчитываем прогноз
        forecast = self._calculate_forecast(goal)
        
        return {
            'goal': goal,
            'progress_percentage': round(progress_percentage, 2),
            'remaining': remaining,
            'trend': trend,
            'forecast': forecast,
            'is_on_track': self._is_on_track(goal, forecast)
        }
    
    def _calculate_trend(self, goal: Goal) -> Dict[str, Any]:
        """Рассчитать тренд прогресса."""
        if len(goal.progress_updates) < 2:
            return {'direction': 'stable', 'rate': 0, 'confidence': 'low'}
        
        # Берем последние 5 обновлений
        recent_updates = goal.progress_updates[-5:]
        
        if len(recent_updates) < 2:
            return {'direction': 'stable', 'rate': 0, 'confidence': 'low'}
        
        # Рассчитываем среднюю скорость изменения
        total_change = 0
        total_days = 0
        
        for i in range(1, len(recent_updates)):
            prev_update = recent_updates[i-1]
            curr_update = recent_updates[i]
            
            prev_date = datetime.fromisoformat(prev_update['date'])
            curr_date = datetime.fromisoformat(curr_update['date'])
            
            days_diff = (curr_date - prev_date).days
            if days_diff > 0:
                change_per_day = curr_update['change'] / days_diff
                total_change += change_per_day
                total_days += days_diff
        
        if total_days == 0:
            return {'direction': 'stable', 'rate': 0, 'confidence': 'low'}
        
        avg_rate = total_change / len(recent_updates)
        
        if avg_rate > 0.01:
            direction = 'increasing'
        elif avg_rate < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        confidence = 'high' if len(recent_updates) >= 3 else 'medium'
        
        return {
            'direction': direction,
            'rate': round(avg_rate, 2),
            'confidence': confidence
        }
    
    def _calculate_forecast(self, goal: Goal) -> Dict[str, Any]:
        """Рассчитать прогноз достижения цели."""
        if not goal.end_date:
            return {'achievable': False, 'estimated_completion': None, 'confidence': 'low'}
        
        end_date = datetime.fromisoformat(goal.end_date)
        current_date = datetime.now()
        
        if current_date >= end_date:
            return {'achievable': False, 'estimated_completion': end_date.isoformat(), 'confidence': 'high'}
        
        days_remaining = (end_date - current_date).days
        remaining_value = goal.target_value - goal.current_value
        
        if remaining_value <= 0:
            return {'achievable': True, 'estimated_completion': current_date.isoformat(), 'confidence': 'high'}
        
        # Рассчитываем необходимую скорость
        required_rate = remaining_value / days_remaining
        
        # Получаем текущий тренд
        trend = self._calculate_trend(goal)
        current_rate = trend['rate']
        
        if current_rate <= 0:
            return {'achievable': False, 'estimated_completion': None, 'confidence': 'medium'}
        
        # Рассчитываем время достижения при текущей скорости
        estimated_days = remaining_value / current_rate
        estimated_completion = current_date + timedelta(days=estimated_days)
        
        achievable = estimated_completion <= end_date
        
        confidence = 'high' if trend['confidence'] == 'high' else 'medium'
        
        return {
            'achievable': achievable,
            'estimated_completion': estimated_completion.isoformat(),
            'required_rate': round(required_rate, 2),
            'current_rate': round(current_rate, 2),
            'confidence': confidence
        }
    
    def _is_on_track(self, goal: Goal, forecast: Dict[str, Any]) -> bool:
        """Определить, идет ли цель по плану."""
        if not forecast['achievable']:
            return False
        
        if forecast['confidence'] == 'low':
            return True  # Недостаточно данных для точной оценки
        
        # Если прогноз показывает достижение цели в срок
        return forecast['achievable']
    
    def search_goals(self, query: str) -> List[Goal]:
        """Поиск целей по запросу."""
        query_lower = query.lower()
        results = []
        
        for goal in self.goals.values():
            if (query_lower in goal.name.lower() or 
                query_lower in goal.description.lower() or
                query_lower in goal.goal_type.value.lower()):
                results.append(goal)
        
        return results
    
    def get_goals_by_type(self, goal_type: GoalType) -> List[Goal]:
        """Получить цели определенного типа."""
        return [goal for goal in self.goals.values() if goal.goal_type == goal_type]
    
    def get_active_goals(self) -> List[Goal]:
        """Получить активные цели."""
        return [goal for goal in self.goals.values() if goal.is_active]

# Глобальный экземпляр менеджера целей
goals_manager = GoalsManager() 