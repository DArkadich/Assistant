"""
AI-критик решений: анализ, вопросы, плюсы, риски, рекомендации
"""
from core.finances import cashflow_forecast, get_total_balance
from core.analytics import BusinessAnalytics
from datetime import datetime

def analyze_decision(text: str) -> dict:
    """
    Анализирует бизнес-решение, задаёт вопросы, даёт плюсы, риски, рекомендации
    Возвращает dict: {'questions': [...], 'pros': [...], 'cons': [...], 'risks': [...], 'summary': str}
    """
    text_l = text.lower()
    result = {'questions': [], 'pros': [], 'cons': [], 'risks': [], 'summary': ''}
    # 1. Распознаём тип решения
    if 'кредит' in text_l or 'займ' in text_l:
        # Вопросы
        result['questions'] = [
            'На какой срок нужен кредит?',
            'Какова ставка и условия погашения?',
            'Для каких целей берёте кредит?',
            'Есть ли альтернативы (отсрочка, инвестор, оптимизация)?',
            'Какой будет ежемесячный платёж?'
        ]
        # Анализируем финансы
        cashflow = cashflow_forecast()
        analytics = BusinessAnalytics()
        kpi = analytics.get_kpi_summary()
        balance = get_total_balance()
        # Плюсы
        result['pros'] = [
            'Дополнительные оборотные средства',
            'Возможность реализовать планы быстрее',
            'Повышение кредитной истории (при успешном обслуживании)'
        ]
        # Минусы и риски
        result['cons'] = [
            'Увеличение долговой нагрузки',
            'Платежи по кредиту уменьшат свободный cash flow',
            'Риски просрочки и штрафов'
        ]
        result['risks'] = [
            'Если сейчас кассовый разрыв — кредит может усилить просадку',
            'Возможное ухудшение финансовых показателей к концу квартала',
            'Рост процентных ставок или изменение условий банка',
            'Падение выручки или непредвиденные расходы'
        ]
        # Сводка
        summary = []
        if cashflow['days_left'] < 60:
            summary.append(f"У тебя сейчас кассовый разрыв — дополнительный кредит усилит просадку по обороту к {datetime.now().strftime('%B')}")
        elif balance < 0:
            summary.append("Баланс отрицательный — кредит может быть рискованным без чёткого плана возврата.")
        else:
            summary.append("Кредит может помочь, если есть чёткий план возврата и рост выручки.")
        summary.append(f"Текущий баланс: {balance:,} руб. | Прогноз: хватит на {cashflow['days_left']} дней.")
        if kpi.get('ROI') and kpi['ROI'] < 0:
            summary.append("ROI бизнеса сейчас отрицательный — стоит пересмотреть стратегию.")
        result['summary'] = '\n'.join(summary)
        return result
    # --- Другие типы решений (пример: покупка оборудования) ---
    if 'купить' in text_l or 'покупка' in text_l or 'оборудование' in text_l:
        result['questions'] = [
            'Какова стоимость и срок окупаемости?',
            'Есть ли альтернативы (аренда, лизинг)?',
            'Как покупка повлияет на cash flow?',
            'Есть ли резерв на непредвиденные расходы?',
            'Как изменится производительность/выручка?'
        ]
        result['pros'] = [
            'Рост производительности',
            'Снижение издержек в долгосрочной перспективе',
            'Улучшение качества продукта/услуги'
        ]
        result['cons'] = [
            'Высокие единовременные расходы',
            'Риски недозагрузки оборудования',
            'Увеличение амортизации и расходов на обслуживание'
        ]
        result['risks'] = [
            'Недостаток оборотных средств после покупки',
            'Долгий срок окупаемости',
            'Неожиданные поломки или простои'
        ]
        result['summary'] = "Проверь, не приведёт ли покупка к кассовому разрыву. Оцени срок окупаемости и влияние на KPI."
        return result
    # --- Найм сотрудника ---
    if 'нанять' in text_l or 'найм' in text_l or 'сотрудник' in text_l:
        result['questions'] = [
            'Какую задачу будет решать новый сотрудник?',
            'Какова полная стоимость найма (зарплата, налоги, обучение)?',
            'Как быстро сотрудник выйдет на окупаемость?',
            'Есть ли альтернатива (аутсорс, автоматизация)?',
            'Как изменится нагрузка на команду?'
        ]
        result['pros'] = [
            'Рост производительности',
            'Снижение нагрузки на команду',
            'Возможность масштабирования бизнеса'
        ]
        result['cons'] = [
            'Рост постоянных расходов',
            'Риски неэффективного найма',
            'Время на адаптацию и обучение'
        ]
        result['risks'] = [
            'Кассовый разрыв при недостатке выручки',
            'Снижение мотивации команды',
            'Ошибки в подборе персонала'
        ]
        result['summary'] = "Оцени, как найм повлияет на cash flow и KPI. Проверь, есть ли бюджет на 3-6 месяцев вперёд."
        return result
    # --- По умолчанию ---
    result['questions'] = [
        'Какова цель этого решения?',
        'Как это повлияет на финансы и KPI?',
        'Есть ли альтернативы?',
        'Каковы риски и план действий при негативном сценарии?'
    ]
    result['summary'] = "Рекомендуется проанализировать влияние на cash flow, KPI и риски."
    return result

def format_critic_result(result: dict) -> str:
    """Форматирование ответа AI-критика для Telegram"""
    text = "<b>🤖 AI-критик решения</b>\n\n"
    if result['questions']:
        text += "<b>Уточняющие вопросы:</b>\n"
        for q in result['questions']:
            text += f"• {q}\n"
        text += "\n"
    if result['pros']:
        text += "<b>Плюсы:</b>\n"
        for p in result['pros']:
            text += f"+ {p}\n"
    if result['cons']:
        text += "<b>Минусы:</b>\n"
        for c in result['cons']:
            text += f"- {c}\n"
    if result['risks']:
        text += "<b>Риски:</b>\n"
        for r in result['risks']:
            text += f"⚠️ {r}\n"
    if result['summary']:
        text += f"\n<b>Сводка:</b>\n{result['summary']}"
    return text.strip() 