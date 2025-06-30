"""
Модуль для синтеза речи из текста с поддержкой русского языка
"""
import os
import tempfile
from pathlib import Path
import edge_tts  # Используем Microsoft Edge TTS для качественного русского языка
import asyncio
from typing import Optional, Union

class SpeechSynthesizer:
    def __init__(self):
        # Настройки голоса по умолчанию
        self.default_voice = "ru-RU-SvetlanaNeural"  # Женский голос
        self.alternative_voice = "ru-RU-DmitryNeural"  # Мужской голос
        self.rate = "+0%"  # Скорость речи
        self.volume = "+0%"  # Громкость
        
        # Создаем временную директорию для аудиофайлов
        self.temp_dir = Path(tempfile.gettempdir()) / "tts_cache"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def text_to_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None
    ) -> Path:
        """
        Преобразует текст в речь и сохраняет в аудиофайл
        
        Args:
            text: Текст для озвучки
            output_path: Путь для сохранения аудиофайла (если None, создается временный файл)
            voice: Голос для озвучки (если None, используется голос по умолчанию)
            rate: Скорость речи (например, "+10%", "-20%")
            volume: Громкость (например, "+10%", "-20%")
        
        Returns:
            Path: Путь к созданному аудиофайлу
        """
        # Подготовка параметров
        voice = voice or self.default_voice
        rate = rate or self.rate
        volume = volume or self.volume
        
        # Если путь не указан, создаем временный файл
        if output_path is None:
            output_path = self.temp_dir / f"tts_{hash(text)}.mp3"
        else:
            output_path = Path(output_path)
        
        # Создаем коммуникатор для TTS
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            volume=volume
        )
        
        try:
            # Синтезируем речь и сохраняем в файл
            await communicate.save(str(output_path))
            return output_path
        except Exception as e:
            # В случае ошибки с основным голосом, пробуем альтернативный
            if voice == self.default_voice:
                communicate = edge_tts.Communicate(
                    text,
                    self.alternative_voice,
                    rate=rate,
                    volume=volume
                )
                await communicate.save(str(output_path))
                return output_path
            raise e
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Очистка временных файлов старше указанного возраста"""
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file in self.temp_dir.glob("*.mp3"):
            if current_time - file.stat().st_mtime > max_age_seconds:
                try:
                    file.unlink()
                except Exception:
                    pass
    
    @staticmethod
    def prepare_text_for_tts(text: str) -> str:
        """
        Подготавливает текст для более естественного звучания
        
        Args:
            text: Исходный текст
        
        Returns:
            str: Подготовленный текст
        """
        # Замена специальных символов
        replacements = {
            "₽": " рублей",
            "$": " долларов",
            "€": " евро",
            "%": " процентов",
            "+": " плюс ",
            "-": " минус ",
            "*": " умножить на ",
            "/": " разделить на ",
            "=": " равно ",
            "&": " и ",
            "@": " собака ",
            "#": " решётка ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Добавляем паузы после предложений
        text = text.replace(". ", "... ")
        text = text.replace("! ", "... ")
        text = text.replace("? ", "... ")
        
        return text

# Глобальный экземпляр для использования в других модулях
speech_synthesizer = SpeechSynthesizer() 