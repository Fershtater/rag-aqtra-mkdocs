"""
Утилиты для работы с Markdown документами.

Функции для извлечения секций, якорей и улучшенного чанкинга.
"""

import re
from typing import List, Tuple, Optional


def extract_sections(text: str) -> List[Tuple[int, str, str]]:
    """
    Извлекает секции из Markdown текста.
    
    Args:
        text: Markdown текст
        
    Returns:
        Список кортежей (уровень, заголовок, содержимое_секции)
    """
    sections = []
    lines = text.split('\n')
    current_section_level = 0
    current_section_title = ""
    current_section_content = []
    
    for line in lines:
        # Проверяем, является ли строка заголовком
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            # Сохраняем предыдущую секцию, если она есть
            if current_section_content:
                sections.append((
                    current_section_level,
                    current_section_title,
                    '\n'.join(current_section_content)
                ))
            
            # Начинаем новую секцию
            current_section_level = len(header_match.group(1))
            current_section_title = header_match.group(2).strip()
            current_section_content = []
        else:
            current_section_content.append(line)
    
    # Добавляем последнюю секцию
    if current_section_content:
        sections.append((
            current_section_level,
            current_section_title,
            '\n'.join(current_section_content)
        ))
    
    return sections


def slugify(text: str) -> str:
    """
    Преобразует текст в slug для якорей.
    
    Args:
        text: Текст заголовка
        
    Returns:
        Slug (например, "app-development" из "App Development")
    """
    # Приводим к нижнему регистру
    text = text.lower()
    # Заменяем пробелы и специальные символы на дефисы
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Убираем дефисы в начале и конце
    return text.strip('-')


def find_section_for_text(text: str, position: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Находит секцию, к которой относится текст на указанной позиции.
    
    Args:
        text: Полный текст документа
        position: Позиция в тексте
        
    Returns:
        Кортеж (заголовок, уровень, anchor) или (None, None, None)
    """
    sections = extract_sections(text)
    current_pos = 0
    last_title = None
    last_level = None
    last_anchor = None
    
    for level, title, content in sections:
        section_start = current_pos
        section_end = current_pos + len(content)
        
        if section_start <= position <= section_end:
            return title, level, slugify(title)
        
        # Обновляем последнюю встреченную секцию
        if level <= (last_level or 999):  # Более высокий уровень заголовка
            last_title = title
            last_level = level
            last_anchor = slugify(title)
        
        current_pos = section_end + len(title) + level + 2  # +2 для "# " и "\n"
    
    return last_title, last_level, last_anchor


def build_doc_url(base_url: str, source: str, section_anchor: Optional[str] = None) -> str:
    """
    Строит полный URL документа с опциональным якорем секции.
    
    Args:
        base_url: Базовый URL документации (например, "https://docs.aqtra.io/")
        source: Путь к файлу относительно docs/ (например, "docs/app-development/button.md")
        section_anchor: Якорь секции (например, "primary-button") или None
        
    Returns:
        Полный URL (например, "https://docs.aqtra.io/app-development/button.html#primary-button")
    """
    # Убираем префикс docs/ если есть
    if source.startswith("docs/"):
        path = source[5:]  # Убираем "docs/"
    else:
        path = source
    
    # Убираем .md
    if path.endswith(".md"):
        path = path[:-3]
    
    # Заменяем разделители на /
    path = path.replace("\\", "/")
    
    # Собираем URL
    url = f"{base_url.rstrip('/')}/{path}.html"
    
    # Добавляем якорь если есть
    if section_anchor:
        url += f"#{section_anchor}"
    
    return url

