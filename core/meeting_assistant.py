"""
Ассистент на звонке: распознавание речи, summary, action items, задачи
"""
import os
import tempfile
import openai
import speech_recognition as sr
from typing import List, Dict, Optional
from core.inbox_monitor import inbox_monitor
from core.task_manager import task_manager

class MeetingAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_path: str, lang: str = 'ru-RU') -> str:
        """Распознаёт речь из аудиофайла (wav/mp3/ogg)"""
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio, language=lang)
            return text
        except Exception as e:
            return f"[Ошибка распознавания: {e}]"
    
    def analyze_meeting_text(self, text: str) -> Dict:
        """
        Генерирует summary и action items по тексту встречи
        Возвращает: {'summary': ..., 'actions': [...], 'tasks': [...]} 
        """
        prompt = (
            "Ты — бизнес-ассистент на встрече.\n"
            "1. Составь краткое резюме (summary) по тексту встречи.\n"
            "2. Выдели отдельным списком все действия (action items), которые нужно выполнить (начинай с глагола: отправить, согласовать, проверить и т.д.).\n"
            "3. Для каждого действия предложи короткое название задачи.\n"
            "Ответ верни в формате JSON: {summary: ..., actions: [...], tasks: [...]}\n"
            f"Текст встречи: {text}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            import json
            # Находим JSON в ответе
            import re
            match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {'summary': '', 'actions': [], 'tasks': []}
        except Exception as e:
            return {'summary': f'Ошибка анализа: {e}', 'actions': [], 'tasks': []}
    
    def add_action_tasks(self, tasks: List[str]):
        """Добавляет задачи из встречи в TaskManager."""
        for task_title in tasks:
            task_manager.add_task(
                title=task_title,
                source='meeting',
                tags=['meeting']
            )

meeting_assistant = MeetingAssistant() 