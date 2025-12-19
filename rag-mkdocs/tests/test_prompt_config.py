"""
Unit tests for prompt_config module.
"""

import pytest
from app.core.prompt_config import detect_response_language


def test_detect_response_language_english():
    """Test detection of English questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("How do I create a button?", supported) == "en"
    assert detect_response_language("What is the API endpoint?", supported) == "en"
    assert detect_response_language("The user wants to know about components", supported) == "en"


def test_detect_response_language_german():
    """Test detection of German questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("Wie erstelle ich eine Schaltfläche?", supported) == "de"
    assert detect_response_language("Was ist der API-Endpunkt?", supported) == "de"
    assert detect_response_language("Der Benutzer möchte wissen, wie man eine Komponente erstellt", supported) == "de"


def test_detect_response_language_french():
    """Test detection of French questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("Comment créer un bouton?", supported) == "fr"
    assert detect_response_language("Quel est le point de terminaison de l'API?", supported) == "fr"
    assert detect_response_language("L'utilisateur veut savoir comment créer une application", supported) == "fr"


def test_detect_response_language_spanish():
    """Test detection of Spanish questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("¿Cómo creo un botón?", supported) == "es"
    assert detect_response_language("¿Cuál es el endpoint de la API?", supported) == "es"
    assert detect_response_language("El usuario quiere saber cómo crear una aplicación", supported) == "es"


def test_detect_response_language_portuguese():
    """Test detection of Portuguese questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("Como criar um botão?", supported) == "pt"
    assert detect_response_language("Qual é o endpoint da API?", supported) == "pt"
    assert detect_response_language("O usuário quer saber como criar uma aplicação", supported) == "pt"


def test_detect_response_language_russian_fallback():
    """Test that Russian (Cyrillic) questions fallback to English."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("Как создать кнопку?", supported) == "en"
    assert detect_response_language("Что такое API?", supported) == "en"
    assert detect_response_language("Пользователь хочет узнать о компонентах", supported) == "en"


def test_detect_response_language_unsupported_latin_fallback():
    """Test that unsupported languages (e.g., Finnish, Italian) fallback to English."""
    supported = {"en", "de", "fr", "es", "pt"}
    # Finnish
    assert detect_response_language("Miten luon painikkeen?", supported) == "en"
    # Italian
    assert detect_response_language("Come creo un pulsante?", supported) == "en"
    # Mixed/ambiguous
    assert detect_response_language("Test question without clear language markers", supported) == "en"


def test_detect_response_language_custom_fallback():
    """Test that custom fallback language is used when specified."""
    supported = {"de", "fr"}
    # English question but English not in supported
    assert detect_response_language("How do I create a button?", supported, fallback="de") == "de"
    # Russian question
    assert detect_response_language("Как создать кнопку?", supported, fallback="fr") == "fr"


def test_detect_response_language_minimum_hits():
    """Test that at least 2 language markers are required."""
    supported = {"en", "de", "fr", "es", "pt"}
    # Single word might not be enough
    assert detect_response_language("der", supported) == "en"  # Single word, likely fallback
    # But multiple words should work
    assert detect_response_language("der Benutzer und die Anwendung", supported) == "de"


def test_detect_response_language_empty_question():
    """Test handling of empty or very short questions."""
    supported = {"en", "de", "fr", "es", "pt"}
    assert detect_response_language("", supported) == "en"
    assert detect_response_language("?", supported) == "en"
    assert detect_response_language("a", supported) == "en"

