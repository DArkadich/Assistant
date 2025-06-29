FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/* \
    && rm -rf /tmp/* /var/tmp/*

# Копирование кода приложения
COPY . .

# Создание placeholder для service_account.json
RUN echo '{"type": "service_account", "project_id": "placeholder"}' > service_account.json

# Запуск приложения
CMD ["python", "main.py"] 