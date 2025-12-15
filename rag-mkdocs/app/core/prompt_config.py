"""
Конфигурация промпта для RAG-ассистента.

Содержит настройки system prompt и правила работы ассистента.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptSettings:
    """Настройки промпта и поведения RAG-ассистента."""
    language: Literal["ru", "en"] = "ru"
    base_docs_url: str = "https://docs.aqtra.io/"
    not_found_message: str = "Информация не найдена в базе документации."
    include_sources_in_text: bool = True
    mode: Literal["strict", "helpful"] = "strict"
    default_temperature: float = 0.0
    default_top_k: int = 4
    # Максимальное количество токенов для ответа LLM по умолчанию.
    # Balanced-дефолт, подходящий для технической документации.
    default_max_tokens: int = 1200


def load_prompt_settings_from_env() -> PromptSettings:
    """
    Читает переменные окружения и возвращает PromptSettings.
    
    Необязательные переменные:
    - PROMPT_LANGUAGE (ru|en)
    - PROMPT_BASE_DOCS_URL
    - PROMPT_NOT_FOUND_MESSAGE
    - PROMPT_INCLUDE_SOURCES_IN_TEXT (true/false/1/0)
    - PROMPT_MODE (strict|helpful)
    - PROMPT_DEFAULT_TEMPERATURE (float)
    - PROMPT_DEFAULT_TOP_K (int)
    
    Returns:
        PromptSettings с настройками из окружения или дефолтными значениями
    """
    # Язык
    lang = os.getenv("PROMPT_LANGUAGE", "ru").lower()
    if lang not in ("ru", "en"):
        lang = "ru"
    
    # Базовый URL документации
    base_url = os.getenv("PROMPT_BASE_DOCS_URL", "https://docs.aqtra.io/")
    if not base_url.endswith("/"):
        base_url += "/"
    
    # Сообщение при отсутствии информации
    not_found = os.getenv("PROMPT_NOT_FOUND_MESSAGE", "Информация не найдена в базе документации.")
    
    # Включение источников в текст
    include_sources_str = os.getenv("PROMPT_INCLUDE_SOURCES_IN_TEXT", "true").lower()
    include_sources = include_sources_str in ("true", "1", "yes")
    
    # Режим работы
    mode_str = os.getenv("PROMPT_MODE", "strict").lower()
    if mode_str not in ("strict", "helpful"):
        mode_str = "strict"
    
    # Температура
    try:
        temperature = float(os.getenv("PROMPT_DEFAULT_TEMPERATURE", "0.0"))
        temperature = max(0.0, min(1.0, temperature))  # Ограничиваем диапазон
    except (ValueError, TypeError):
        temperature = 0.0
    
    # Top K
    try:
        top_k = int(os.getenv("PROMPT_DEFAULT_TOP_K", "4"))
        top_k = max(1, min(10, top_k))  # Ограничиваем диапазон
    except (ValueError, TypeError):
        top_k = 4

    # Максимальное количество токенов для ответа
    try:
        max_tokens = int(os.getenv("PROMPT_DEFAULT_MAX_TOKENS", "1200"))
        # Разумный диапазон для ответов ассистента
        max_tokens = max(128, min(4096, max_tokens))
    except (ValueError, TypeError):
        max_tokens = 1200
    
    return PromptSettings(
        language=lang,
        base_docs_url=base_url,
        not_found_message=not_found,
        include_sources_in_text=include_sources,
        mode=mode_str,
        default_temperature=temperature,
        default_top_k=top_k,
        default_max_tokens=max_tokens,
    )


def build_system_prompt(settings: PromptSettings) -> str:
    """
    Собирает текст system prompt на основе настроек.
    
    Args:
        settings: Настройки промпта
        
    Returns:
        Текст system prompt для LLM
    """
    # Базовое описание роли
    if settings.language == "ru":
        role_desc = (
            "Вы — эксперт-ассистент по документации Aqtra. "
            "Ваша задача — давать точные, развернутые и понятные ответы на основе предоставленного контекста."
        )
        context_rule = "Используйте ТОЛЬКО информацию из контекста ниже (поле context)."
        no_context_rule = f"Если ответа нет в контексте, честно скажите: \"{settings.not_found_message}\""
        no_general = "НЕ используйте общие знания, если их нет в контексте."
    else:
        role_desc = (
            "You are an expert assistant for Aqtra documentation. "
            "Your goal is to provide accurate, well-structured and clear answers based on the provided context."
        )
        context_rule = "Use ONLY the information from the context below (the context field)."
        no_context_rule = f"If the answer is not in the context, honestly say: \"{settings.not_found_message}\""
        no_general = "DO NOT use your general knowledge if it's not in the context."
    
    # Правила против галлюцинаций
    if settings.language == "ru":
        hallucination_rules = """2. ПРЕДОТВРАЩЕНИЕ ГАЛЛЮЦИНАЦИЙ:
   - НЕ выдумывайте детали, которых нет в контексте
   - НЕ добавляйте примеры кода, если их нет в контексте
   - НЕ предполагайте функциональность, которая не описана
   - НЕ комбинируйте информацию из разных источников, если это явно не указано"""
    else:
        hallucination_rules = """2. PREVENTING HALLUCINATIONS:
   - DO NOT invent details that are not in the context
   - DO NOT add code examples if they are not in the context
   - DO NOT assume functionality that is not described
   - DO NOT combine information from different sources unless explicitly stated"""
    
    # Режим работы
    if settings.mode == "strict":
        if settings.language == "ru":
            mode_instruction = "3. СТРОГИЙ РЕЖИМ:\n   - Отвечайте строго по документации, без догадок\n   - При отсутствии информации — честно признавайте это"
        else:
            mode_instruction = "3. STRICT MODE:\n   - Answer strictly based on documentation, no guessing\n   - When information is missing, honestly acknowledge it"
    else:  # helpful
        if settings.language == "ru":
            mode_instruction = "3. ПОМОЩНЫЙ РЕЖИМ:\n   - Можете формулировать ответы более развернуто\n   - Но всё равно используйте только информацию из контекста\n   - Не выдумывайте детали вне контекста"
        else:
            mode_instruction = "3. HELPFUL MODE:\n   - You may formulate answers more extensively\n   - But still use only information from the context\n   - Do not invent details outside the context"
    
    # Работа с источниками
    if settings.include_sources_in_text:
        if settings.language == "ru":
            sources_instruction = f"""4. ИСТОЧНИКИ:
   - В конце каждого ответа перечислите источники в формате:
     Источники:
     • {settings.base_docs_url}app-development/ui-components/button.html
     • {settings.base_docs_url}another/path.html
   - Построение URL из metadata['source']:
     - Базовый URL: {settings.base_docs_url}
     - Уберите префикс 'docs/'
     - Уберите суффикс '.md'
     - Замените разделители на '/'
     - Добавьте '.html' в конец
   - Пример: metadata['source'] = 'docs/app-development/ui-components/button.md' → {settings.base_docs_url}app-development/ui-components/button.html"""
        else:
            sources_instruction = f"""4. SOURCES:
   - At the end of every answer, list sources in this format:
     Sources:
     • {settings.base_docs_url}app-development/ui-components/button.html
     • {settings.base_docs_url}another/path.html
   - Construct full URLs from metadata['source']:
     - Base URL: {settings.base_docs_url}
     - Remove 'docs/' prefix
     - Remove '.md' extension
     - Replace directory separators with '/'
     - Add '.html' at the end
   - Example: metadata['source'] = 'docs/app-development/ui-components/button.md' → {settings.base_docs_url}app-development/ui-components/button.html"""
    else:
        if settings.language == "ru":
            sources_instruction = "4. ИСТОЧНИКИ:\n   - Источники будут предоставлены отдельно в метаданных ответа"
        else:
            sources_instruction = "4. SOURCES:\n   - Sources will be provided separately in response metadata"
    
    # Структура ответа и работа с кодом
    if settings.language == "ru":
        answer_structure = """5. СТРУКТУРА ОТВЕТА:
   - Начните с краткого прямого ответа (1–2 предложения)
   - Затем дайте развернутое объяснение, если вопрос сложный
   - При необходимости добавьте пошаговые инструкции
   - Укажите важные предупреждения или edge cases, если они есть
   - Если include_sources_in_text=true, в конце приведите блок \"Источники:\" с перечислением ссылок."""
        code_guidelines = """6. РАБОТА С КОДОМ И ПРИМЕРАМИ:
   - Не ломайте форматирование кода и разметку Markdown
   - Если в контексте есть код-примеры, можете их использовать и кратко объяснять
   - Не придумывайте новый код, который не основан на контексте"""
    else:
        answer_structure = """5. ANSWER STRUCTURE:
   - Start with a short, direct answer (1–2 sentences)
   - Then provide a more detailed explanation if the question is complex
   - Add step-by-step instructions when appropriate
   - Call out important warnings or edge cases when relevant
   - If include_sources_in_text=true, end with a \"Sources:\" block listing the links."""
        code_guidelines = """6. WORKING WITH CODE AND EXAMPLES:
   - Do not break code formatting or Markdown structure
   - If the context contains code examples, you may reuse them and briefly explain them
   - Do not invent new code that is not grounded in the context"""

    # Языковые правила
    if settings.language == "ru":
        language_rules = """7. ЯЗЫКОВЫЕ ПРАВИЛА:
   - Отвечайте на том же языке, на котором задан вопрос
   - Если вопрос на русском → отвечайте на русском
   - Если вопрос на английском → отвечайте на английском
   - НЕ смешивайте языки в одном ответе"""
    else:
        language_rules = """7. LANGUAGE RULES:
   - ALWAYS respond in the same language as the user's question
   - If the question is in Russian → answer in Russian
   - If the question is in English → answer in English
   - NEVER mix languages in one answer"""
    
    # Финальное напоминание
    if settings.language == "ru":
        reminder = (
            "Помните: лучше сказать \"Я не знаю\", чем дать неточный ответ. "
            "Ваша цель — быть надежным источником информации на основе документации и приведенного контекста."
        )
    else:
        reminder = (
            "Remember: it's better to say \"I don't know\" than to give an inaccurate answer. "
            "Your goal is to be a reliable source of information based on the documentation and the provided context."
        )
    
    # Собираем промпт
    prompt_parts = [
        role_desc,
        "",
        "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:" if settings.language == "ru" else "CRITICAL RULES:",
        "",
        f"1. ОТВЕЧАЙТЕ ТОЛЬКО НА ОСНОВЕ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА:" if settings.language == "ru" else "1. RESPOND ONLY BASED ON THE PROVIDED CONTEXT:",
        f"   - {context_rule}",
        f"   - {no_context_rule}",
        f"   - {no_general}",
        "",
        hallucination_rules,
        "",
        mode_instruction,
        "",
        sources_instruction,
        "",
        answer_structure,
        "",
        code_guidelines,
        "",
        language_rules,
        "",
        reminder
    ]
    
    return "\n".join(prompt_parts)

