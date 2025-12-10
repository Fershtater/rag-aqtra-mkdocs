# Исправление конфликта версий

## Проблема
Исходные версии в requirements.txt были несовместимы:
- langchain-community 0.2.0 требует langchain-core >=0.2.0
- langchain-openai 0.1.1 требует langchain-core <0.2.0

## Решение
В pyproject.toml используются гибкие версии (^), позволяющие Poetry найти совместимые версии:
- langchain ^0.2.0 → установится 0.2.17
- langchain-openai ^0.1.0 → установится 0.1.25
- langchain-community ^0.2.0 → установится 0.2.19

## Важно
- Используйте Python 3.12 (не 3.14!)
- Используйте Poetry для установки: `poetry install`
- Poetry автоматически разрешит все конфликты зависимостей
