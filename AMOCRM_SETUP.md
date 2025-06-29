# Настройка интеграции с AmoCRM

## 1. Создание интеграции в AmoCRM

### Шаг 1: Создание интеграции
1. Войдите в ваш AmoCRM аккаунт
2. Перейдите в **Настройки** → **Интеграции** → **Другие интеграции**
3. Нажмите **Создать интеграцию**
4. Заполните форму:
   - **Название**: Business Assistant Bot
   - **Описание**: Telegram бот для управления бизнесом
   - **Ссылка на сайт**: https://github.com/your-repo
   - **Права доступа**: выберите необходимые права

### Шаг 2: Настройка прав доступа
Выберите следующие права:
- ✅ **Контакты**: чтение, добавление, редактирование
- ✅ **Сделки**: чтение, добавление, редактирование
- ✅ **Задачи**: чтение, добавление, редактирование
- ✅ **Воронки продаж**: чтение
- ✅ **Пользователи**: чтение

### Шаг 3: Получение данных для интеграции
После создания интеграции вы получите:
- **Client ID** (ID интеграции)
- **Client Secret** (Секретный ключ)
- **Authorization Code** (код авторизации)

## 2. Настройка переменных окружения

Добавьте следующие переменные в ваш `.env` файл:

```env
# AmoCRM Configuration
AMOCRM_BASE_URL=https://your-domain.amocrm.ru
AMOCRM_CLIENT_ID=your_client_id_here
AMOCRM_CLIENT_SECRET=your_client_secret_here
AMOCRM_ACCESS_TOKEN=your_access_token_here
AMOCRM_REFRESH_TOKEN=your_refresh_token_here
```

## 3. Получение Access Token

### Метод 1: Через Postman или curl

```bash
curl -X POST "https://your-domain.amocrm.ru/oauth2/access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "grant_type": "authorization_code",
    "code": "your_authorization_code",
    "redirect_uri": "https://your-domain.amocrm.ru"
  }'
```

### Метод 2: Через Python скрипт

Создайте файл `get_amocrm_token.py`:

```python
import requests
import json

def get_amocrm_token():
    url = "https://your-domain.amocrm.ru/oauth2/access_token"
    
    data = {
        "client_id": "your_client_id",
        "client_secret": "your_client_secret", 
        "grant_type": "authorization_code",
        "code": "your_authorization_code",
        "redirect_uri": "https://your-domain.amocrm.ru"
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        token_data = response.json()
        print("Access Token:", token_data['access_token'])
        print("Refresh Token:", token_data['refresh_token'])
        return token_data
    else:
        print("Error:", response.status_code, response.text)
        return None

if __name__ == "__main__":
    get_amocrm_token()
```

## 4. Настройка кастомных полей

Для корректной работы синхронизации партнёров создайте кастомные поля в AmoCRM:

### Поля для контактов:
1. **Канал** (field_id: 3) - текстовое поле
2. **Результат** (field_id: 4) - текстовое поле  
3. **Сегмент** (field_id: 5) - текстовое поле

### Поля для сделок:
1. **Сумма** (field_id: 1) - числовое поле

## 5. Команды Telegram бота

После настройки вы сможете использовать следующие команды:

### Контакты
- `контакты amocrm` - показать контакты
- `найди контакт [имя]` - поиск контакта
- `создай контакт [имя], email [email], телефон [телефон]` - создать контакт

### Сделки
- `сделки amocrm` - показать сделки
- `создай сделку [название], контакт [имя], сумма [число]` - создать сделку

### Аналитика
- `аналитика amocrm` - общая аналитика
- `аналитика amocrm за неделю` - аналитика за неделю
- `аналитика amocrm за месяц` - аналитика за месяц

### Синхронизация
- `синхронизация amocrm` - синхронизировать партнёров из Google Sheets

### Воронки и задачи
- `воронки amocrm` - показать воронки продаж
- `задачи amocrm` - показать задачи

## 6. Тестирование интеграции

1. Запустите бота
2. Отправьте команду `контакты amocrm`
3. Если всё настроено правильно, вы увидите список контактов

## 7. Устранение неполадок

### Ошибка "Access token не найден"
- Проверьте переменные окружения
- Убедитесь, что токены не истекли

### Ошибка "401 Unauthorized"
- Обновите access token через refresh token
- Проверьте правильность client_id и client_secret

### Ошибка "403 Forbidden"
- Проверьте права доступа интеграции
- Убедитесь, что интеграция активна

## 8. Автоматическое обновление токенов

Бот автоматически обновляет access token при истечении срока действия. 
Refresh token сохраняется в переменных окружения.

## 9. Безопасность

- Никогда не публикуйте client_secret в открытом доступе
- Используйте переменные окружения для хранения токенов
- Регулярно обновляйте токены
- Ограничьте права доступа интеграции только необходимыми

## 10. Дополнительные возможности

### Webhook интеграция
Для получения уведомлений о новых лидах можно настроить webhook:

```python
# В настройках интеграции добавьте webhook URL
WEBHOOK_URL = "https://your-bot-domain.com/webhook/amocrm"
```

### Автоматическая синхронизация
Настройте автоматическую синхронизацию партнёров:

```python
# В планировщике задач
scheduler.add_job(
    amocrm.sync_partners_from_sheet,
    'interval',
    hours=24,
    args=[partners_manager]
)
``` 