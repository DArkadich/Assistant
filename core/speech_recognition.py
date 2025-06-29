import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from typing import Optional

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Настройки для лучшего распознавания
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def convert_audio_to_wav(self, audio_path: str) -> str:
        """Конвертировать аудиофайл в WAV формат."""
        try:
            # Определяем формат по расширению
            file_ext = os.path.splitext(audio_path)[1].lower()
            
            if file_ext == '.wav':
                return audio_path
            
            # Конвертируем в WAV
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.replace(file_ext, '.wav')
            audio.export(wav_path, format='wav')
            
            return wav_path
            
        except Exception as e:
            print(f"Ошибка конвертации аудио: {e}")
            return audio_path
    
    def recognize_speech(self, audio_path: str, language: str = 'ru-RU') -> Optional[str]:
        """Распознать речь из аудиофайла."""
        try:
            # Конвертируем в WAV если нужно
            wav_path = self.convert_audio_to_wav(audio_path)
            
            # Загружаем аудио
            with sr.AudioFile(wav_path) as source:
                # Настраиваем распознаватель для шумного аудио
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Получаем аудиоданные
                audio_data = self.recognizer.record(source)
                
                # Распознаем речь
                text = self.recognizer.recognize_google(
                    audio_data, 
                    language=language,
                    show_all=False
                )
                
                return text.strip()
                
        except sr.UnknownValueError:
            print("Речь не распознана")
            return None
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
            return None
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            return None
    
    def recognize_from_bytes(self, audio_bytes: bytes, language: str = 'ru-RU') -> Optional[str]:
        """Распознать речь из байтов аудио."""
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Распознаем
            result = self.recognize_speech(temp_path, language)
            
            # Удаляем временный файл
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Ошибка распознавания из байтов: {e}")
            return None

# Глобальный экземпляр распознавателя
speech_recognizer = SpeechRecognizer() 