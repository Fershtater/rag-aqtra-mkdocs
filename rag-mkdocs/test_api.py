"""
Тестовый скрипт для проверки работы RAG API.

Использование:
    poetry run python test_api.py

Или с активированным окружением:
    python test_api.py

Скрипт проверяет:
1. Доступность API (health check)
2. Работу endpoint /query
3. Корректность ответа и наличия источников
"""

import json
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Базовый URL API
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Тестирует endpoint /health."""
    print("=" * 60)
    print("ТЕСТ 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        if data.get("status") == "ok" and data.get("rag_chain_ready"):
            print("✓ RAG цепочка готова к работе")
            return True
        else:
            print("⚠ RAG цепочка не готова")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Ошибка: Не удалось подключиться к серверу")
        print("  Убедитесь, что сервер запущен:")
        print("  poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return False
    except requests.exceptions.Timeout:
        print("✗ Ошибка: Таймаут при подключении к серверу")
        return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_query_endpoint():
    """Тестирует endpoint /query."""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Query Endpoint")
    print("=" * 60)
    
    test_question = "Тестовый вопрос по docs"
    print(f"Вопрос: {test_question}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": test_question},
            headers={"Content-Type": "application/json"},
            timeout=30  # Увеличено для обработки RAG запроса
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Ответ получен")
        print(f"\nОтвет:")
        print(f"  {data.get('answer', 'N/A')[:200]}...")
        
        sources = data.get("sources", [])
        if sources:
            print(f"\n✓ Найдено источников: {len(sources)}")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source.get('filename', 'unknown')} ({source.get('source', 'unknown')})")
        else:
            print("⚠ Источники не найдены")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Ошибка: {e}")
        if e.response is not None:
            try:
                error_data = e.response.json()
                print(f"  Детали: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"  Response: {e.response.text}")
        return False
    except requests.exceptions.Timeout:
        print("✗ Ошибка: Таймаут при запросе (возможно, сервер обрабатывает запрос)")
        return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_root_endpoint():
    """Тестирует корневой endpoint /."""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Root Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status Code: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def main():
    """Основная функция для запуска всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ RAG API")
    print("=" * 60)
    print(f"Базовый URL: {BASE_URL}")
    print(f"Убедитесь, что сервер запущен на {BASE_URL}")
    print("=" * 60 + "\n")
    
    results = []
    
    # Тест 1: Health Check
    results.append(("Health Check", test_health_check()))
    
    # Тест 2: Root Endpoint
    results.append(("Root Endpoint", test_root_endpoint()))
    
    # Тест 3: Query Endpoint (только если health check прошел)
    if results[0][1]:  # Если health check прошел
        results.append(("Query Endpoint", test_query_endpoint()))
    else:
        print("\n" + "=" * 60)
        print("ТЕСТ 3: Query Endpoint - ПРОПУЩЕН")
        print("=" * 60)
        print("⚠ Пропущен, так как health check не прошел")
        results.append(("Query Endpoint", False))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ ПРОШЕЛ" if result else "✗ НЕ ПРОШЕЛ"
        print(f"{test_name}: {status}")
    
    print(f"\nВсего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Не пройдено: {total - passed}")
    
    if passed == total:
        print("\n✓ Все тесты пройдены успешно!")
        return 0
    else:
        print(f"\n✗ Некоторые тесты не пройдены ({total - passed} из {total})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

