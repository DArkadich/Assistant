# Business Assistant Bot

🤖 **Умный Telegram бот для управления бизнесом** с интеграцией OpenAI GPT-4, Google Calendar, Google Sheets, AmoCRM и автоматизацией документооборота.

## 🚀 Возможности

### 📅 **Календарь и планирование**
- Синхронизация с Google Calendar
- Естественный язык для добавления задач
- Автоматические уведомления о событиях
- Ежедневные и еженедельные сводки

### 💰 **Финансовый учёт**
- Отслеживание доходов и расходов по проектам
- Интеграция с Google Sheets
- Автоматические отчёты (день/неделя/месяц/квартал/год)
- P&L анализ в реальном времени

### 📄 **Документооборот**
- Автоматическое распознавание документов (OCR)
- Семантический поиск по документам (RAG)
- Интеграция с Google Drive
- Поддержка: накладные, УПД, ГТД, счета, контракты, акты

### 🎯 **Цели и KPI**
- Постановка и отслеживание целей
- Автоматический расчёт прогресса
- Уведомления о дедлайнах

### 📧 **Email аналитика**
- Интеграция с Gmail и Yandex.Mail
- Автоматические сводки входящих
- Генерация шаблонов ответов
- Выделение срочных сообщений

### 🤝 **Партнёрская сеть**
- Управление партнёрами через Google Sheets
- Автоматическая генерация предложений
- Фильтрация по сегментам
- Списки для прозвона и рассылки

### 🏢 **AmoCRM интеграция**
- Синхронизация контактов и сделок
- Автоматическая аналитика продаж
- Управление воронкой продаж
- Создание задач и сделок
- Синхронизация партнёров из Google Sheets

### 🗣️ **Голосовые команды**
- Распознавание речи
- Обработка голосовых сообщений
- Поддержка русского языка

### 🧠 **Контекст и память**
- Сохранение истории обсуждений
- Поиск по принятым решениям
- Анализ участников обсуждений

## 🛠️ Технологии

- **Python 3.9+**
- **OpenAI GPT-4/3.5** - обработка естественного языка
- **Google APIs** - Calendar, Sheets, Drive
- **AmoCRM API** - управление CRM
- **ChromaDB** - векторная база данных для RAG
- **OpenCV + Tesseract** - обработка изображений и OCR
- **Docker** - контейнеризация
- **GitHub Actions** - автоматический деплой

## 📦 Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/your-username/business-assistant-bot.git
cd business-assistant-bot
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения
Создайте файл `.env`:
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo
OPENAI_SMART_MODEL=gpt-4-1106-preview

# Google APIs
GOOGLE_APPLICATION_CREDENTIALS=service_account.json
GOOGLE_CALENDAR_ID=your_calendar_id
GOOGLE_SHEETS_ID=your_sheets_id

# AmoCRM
AMOCRM_BASE_URL=https://your-domain.amocrm.ru
AMOCRM_CLIENT_ID=your_client_id
AMOCRM_CLIENT_SECRET=your_client_secret
AMOCRM_ACCESS_TOKEN=your_access_token
AMOCRM_REFRESH_TOKEN=your_refresh_token

# Email
GMAIL_USERNAME=your_email@gmail.com
GMAIL_PASSWORD=your_app_password
YANDEX_USERNAME=your_email@yandex.ru
YANDEX_PASSWORD=your_app_password
```

### 4. Настройка Google APIs
1. Создайте проект в Google Cloud Console
2. Включите APIs: Calendar, Sheets, Drive, Gmail
3. Создайте Service Account и скачайте JSON ключ
4. Переименуйте в `service_account.json`

### 5. Настройка AmoCRM
Следуйте инструкции в [AMOCRM_SETUP.md](AMOCRM_SETUP.md)

## 🚀 Запуск

### Локальный запуск
```bash
python main.py
```

### Docker запуск
```bash
docker-compose up -d
```

## 📱 Команды бота

### Календарь
- `"Встреча с клиентом завтра в 15:00"` - добавить задачу
- `"План на сегодня"` - дневная сводка
- `"План на неделю"` - недельная сводка

### Финансы
- `"Доход 100000 проект ВБ описание оплата за разработку"` - добавить доход
- `"Расход 5000 проект Horien категория упаковка описание коробки"` - добавить расход
- `"Покажи доходы за неделю"` - отчёт по доходам
- `"Покажи расходы за месяц проект ВБ"` - отчёт по расходам

### Документы
- `"Найди документ про оплату услуг"` - семантический поиск
- `"Найди по типу контракт разработка"` - поиск по типу
- `"Найди по контрагенту ООО Рога"` - поиск по контрагенту

### Цели
- `"Создать цель выручка 3 млн до сентября"` - создать цель
- `"Прогресс по цели выручка"` - проверить прогресс
- `"Обновить прогресс выручка 2.5 млн"` - обновить прогресс

### Email
- `"Сводка входящих"` - сводка по email
- `"Срочные сообщения"` - важные письма
- `"Шаблон ответа на письмо про сотрудничество"` - генерация ответа

### Партнёры
- `"Партнёры сводка"` - общая сводка
- `"Добавь партнёра Иван Петров, канал LinkedIn, контакты +7-999-123-45-67"` - добавить партнёра
- `"Партнёры для прозвона"` - список для звонков
- `"Предложение для ООО Рога"` - генерация предложения

### AmoCRM
- `"Контакты amocrm"` - показать контакты
- `"Сделки amocrm"` - показать сделки
- `"Аналитика amocrm"` - аналитика продаж
- `"Создай контакт Иван Петров, email ivan@example.com"` - создать контакт
- `"Создай сделку Разработка сайта, контакт Иван Петров, сумма 100000"` - создать сделку
- `"Синхронизация amocrm"` - синхронизировать партнёров

## 🧪 Тестирование

### Тест AmoCRM интеграции
```bash
python test_amocrm.py
```

### Тест других модулей
```bash
python test_calendar.py
python test_goals.py
python test_rag_data.py
```

## 📊 Мониторинг

### Логи
```bash
docker-compose logs -f bot
```

### Статистика RAG
```bash
# В боте: "статистика rag"
```

## 🔧 Разработка

### Структура проекта
```
Assistant/
├── core/                    # Основные модули
│   ├── calendar.py         # Google Calendar
│   ├── finances.py         # Финансовый учёт
│   ├── amocrm.py          # AmoCRM интеграция
│   ├── partners.py        # Партнёрская сеть
│   ├── goals.py           # Цели и KPI
│   ├── email_analyzer.py  # Email аналитика
│   ├── rag_system.py      # Семантический поиск
│   └── ...
├── interface/              # Интерфейсы
│   ├── telegram_bot.py    # Telegram бот
│   └── webhook.py         # Webhook обработчики
├── docker-compose.yml     # Docker конфигурация
├── requirements.txt       # Зависимости
└── README.md             # Документация
```

### Добавление новых команд
1. Создайте функцию в соответствующем модуле
2. Добавьте обработчик в `interface/telegram_bot.py`
3. Добавьте регулярное выражение для распознавания команды
4. Протестируйте функциональность

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🆘 Поддержка

- 📧 Email: support@example.com
- 💬 Telegram: @business_assistant_support
- 📖 Документация: [Wiki](https://github.com/your-username/business-assistant-bot/wiki)

## 🙏 Благодарности

- OpenAI за GPT модели
- Google за APIs
- AmoCRM за CRM платформу
- Сообществу open source за библиотеки

---

⭐ **Если проект вам понравился, поставьте звезду!** 