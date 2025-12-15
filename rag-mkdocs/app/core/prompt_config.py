"""
Prompt configuration for RAG assistant.

Contains system prompt settings and assistant behavior rules.
"""

import os
import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class PromptSettings:
    """Prompt settings and RAG assistant behavior."""
    supported_languages: tuple[str, ...] = ("en", "de", "fr", "es", "pt")
    fallback_language: Literal["en"] = "en"
    base_docs_url: str = "https://docs.aqtra.io/"
    not_found_message: str = "Information not found in documentation database."
    include_sources_in_text: bool = True
    mode: Literal["strict", "helpful"] = "strict"
    default_temperature: float = 0.0
    default_top_k: int = 4
    # Maximum number of tokens for LLM response by default.
    # Balanced default suitable for technical documentation.
    default_max_tokens: int = 1200


def load_prompt_settings_from_env() -> PromptSettings:
    """
    Reads environment variables and returns PromptSettings.
    
    Optional variables:
    - PROMPT_SUPPORTED_LANGUAGES (comma-separated, default: "en,de,fr,es,pt")
    - PROMPT_FALLBACK_LANGUAGE (default: "en")
    - PROMPT_BASE_DOCS_URL
    - PROMPT_NOT_FOUND_MESSAGE
    - PROMPT_INCLUDE_SOURCES_IN_TEXT (true/false/1/0)
    - PROMPT_MODE (strict|helpful)
    - PROMPT_DEFAULT_TEMPERATURE (float)
    - PROMPT_DEFAULT_TOP_K (int)
    - PROMPT_DEFAULT_MAX_TOKENS (int)
    
    Returns:
        PromptSettings with settings from environment or default values
    """
    # Supported languages
    allowed_languages = {"en", "de", "fr", "es", "pt"}
    supported_langs_str = os.getenv("PROMPT_SUPPORTED_LANGUAGES", "en,de,fr,es,pt")
    supported_langs_list = [lang.strip().lower() for lang in supported_langs_str.split(",")]
    # Filter to only allowed languages
    supported_langs = tuple(lang for lang in supported_langs_list if lang in allowed_languages)
    if not supported_langs:
        supported_langs = ("en",)  # Ensure at least English is supported
    
    # Fallback language
    fallback = os.getenv("PROMPT_FALLBACK_LANGUAGE", "en").lower()
    if fallback not in allowed_languages:
        fallback = "en"
    
    # Base documentation URL
    base_url = os.getenv("PROMPT_BASE_DOCS_URL", "https://docs.aqtra.io/")
    if not base_url.endswith("/"):
        base_url += "/"
    
    # Message when information is missing
    not_found = os.getenv("PROMPT_NOT_FOUND_MESSAGE", "Information not found in documentation database.")
    
    # Include sources in text
    include_sources_str = os.getenv("PROMPT_INCLUDE_SOURCES_IN_TEXT", "true").lower()
    include_sources = include_sources_str in ("true", "1", "yes")
    
    # Mode
    mode_str = os.getenv("PROMPT_MODE", "strict").lower()
    if mode_str not in ("strict", "helpful"):
        mode_str = "strict"
    
    # Temperature
    try:
        temperature = float(os.getenv("PROMPT_DEFAULT_TEMPERATURE", "0.0"))
        temperature = max(0.0, min(1.0, temperature))  # Limit range
    except (ValueError, TypeError):
        temperature = 0.0
    
    # Top K
    try:
        top_k = int(os.getenv("PROMPT_DEFAULT_TOP_K", "4"))
        top_k = max(1, min(10, top_k))  # Limit range
    except (ValueError, TypeError):
        top_k = 4

    # Maximum number of tokens for response
    try:
        max_tokens = int(os.getenv("PROMPT_DEFAULT_MAX_TOKENS", "1200"))
        # Reasonable range for assistant responses
        max_tokens = max(128, min(4096, max_tokens))
    except (ValueError, TypeError):
        max_tokens = 1200
    
    return PromptSettings(
        supported_languages=supported_langs,
        fallback_language=fallback,
        base_docs_url=base_url,
        not_found_message=not_found,
        include_sources_in_text=include_sources,
        mode=mode_str,
        default_temperature=temperature,
        default_top_k=top_k,
        default_max_tokens=max_tokens,
    )


def detect_response_language(user_question: str, supported: set[str], fallback: str = "en") -> str:
    """
    Detects the response language based on user question.
    
    Uses lightweight heuristics to detect German, French, Spanish, Portuguese, or English.
    If question contains Cyrillic or other unsupported languages, returns fallback.
    
    Args:
        user_question: The user's question text
        supported: Set of supported language codes (e.g., {"en", "de", "fr", "es", "pt"})
        fallback: Language code to use if detection fails or language is unsupported (default: "en")
        
    Returns:
        Language code (one of supported languages or fallback)
    """
    # Check for Cyrillic characters (Russian, etc.) - immediate fallback
    if re.search(r'[\u0400-\u04FF]', user_question):
        return fallback
    
    # Normalize: lowercase, collapse whitespace, add spaces around
    normalized = " " + re.sub(r'\s+', ' ', user_question.lower().strip()) + " "
    
    # Language detection patterns (common stopwords/phrases)
    patterns = {
        "de": [" der ", " die ", " und ", " nicht ", " ich ", " sie ", " ein ", " eine ", " das ", " ist ", " für ", " auf "],
        "fr": [" le ", " la ", " les ", " des ", " une ", " et ", " pas ", " vous ", " avec ", " dans ", " pour ", " est "],
        "es": [" el ", " la ", " los ", " las ", " de ", " que ", " y ", " para ", " con ", " en ", " un ", " una "],
        "pt": [" o ", " a ", " os ", " as ", " de ", " que ", " e ", " para ", " com ", " não ", " em ", " um ", " uma "],
        "en": [" the ", " and ", " to ", " of ", " you ", " with ", " for ", " are ", " is ", " in ", " on ", " at "],
    }
    
    # Count matches for each language
    scores = {}
    for lang_code, lang_patterns in patterns.items():
        if lang_code not in supported:
            continue
        count = sum(1 for pattern in lang_patterns if pattern in normalized)
        if count > 0:
            scores[lang_code] = count
    
    # Require at least 2 hits to avoid false positives
    if scores:
        best_lang = max(scores.items(), key=lambda x: x[1])
        if best_lang[1] >= 2:
            return best_lang[0]
    
    # Default to English if no strong signal
    return "en" if "en" in supported else fallback


def build_system_prompt(settings: PromptSettings, response_language: str = "{response_language}") -> str:
    """
    Builds system prompt text based on settings.
    
    The prompt text is kept in English only (repository English-only policy).
    Language output is controlled via LANGUAGE OUTPUT RULE instruction.
    
    Args:
        settings: Prompt settings
        response_language: Language code for response output (en/de/fr/es/pt) or "{response_language}" placeholder
        
    Returns:
        System prompt text for LLM with {response_language} placeholder that will be filled at runtime
    """
    # If response_language is a placeholder string, use it as-is
    # Otherwise validate and format it
    if response_language == "{response_language}":
        lang_upper = "{response_language}"
    else:
        # Validate response_language
        if response_language not in settings.supported_languages:
            response_language = settings.fallback_language
        lang_upper = response_language.upper()
    
    # Base role description (always in English)
    role_desc = (
        "You are an expert assistant for Aqtra documentation. "
        "Your goal is to provide accurate, well-structured and clear answers based on the provided context."
    )
    context_rule = "Use ONLY the information from the context below (the context field)."
    no_context_rule = "If the answer is not in the context, clearly state that the documentation does not contain the information. Follow the response language rule."
    no_general = "DO NOT use your general knowledge if it's not in the context."
    
    # Rules against hallucinations
    hallucination_rules = """2. PREVENTING HALLUCINATIONS:
   - DO NOT invent details that are not in the context
   - DO NOT add code examples if they are not in the context
   - DO NOT assume functionality that is not described
   - DO NOT combine information from different sources unless explicitly stated"""
    
    # Mode
    if settings.mode == "strict":
        mode_instruction = "3. STRICT MODE:\n   - Answer strictly based on documentation, no guessing\n   - When information is missing, honestly acknowledge it"
    else:  # helpful
        mode_instruction = "3. HELPFUL MODE:\n   - You may formulate answers more extensively\n   - But still use only information from the context\n   - Do not invent details outside the context"
    
    # Working with sources
    if settings.include_sources_in_text:
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
        sources_instruction = "4. SOURCES:\n   - Sources will be provided separately in response metadata"
    
    # Answer structure and working with code
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

    # Language rules - use placeholder for response_language
    language_rules = f"""7. LANGUAGE RULES:
   - Allowed output languages: {', '.join(settings.supported_languages).upper()}
   - Determine the output language from the user's question
   - If the user's language is not one of the allowed languages, respond in {settings.fallback_language.upper()}
   - LANGUAGE OUTPUT RULE: For this request, respond in {lang_upper} only (must be the only language in the answer)"""
    
    # Final reminder
    reminder = (
        "Remember: it's better to say \"I don't know\" than to give an inaccurate answer. "
        "Your goal is to be a reliable source of information based on the documentation and the provided context."
    )
    
    # Build prompt (all text in English, language output controlled by instruction)
    prompt_parts = [
        role_desc,
        "",
        "CRITICAL RULES:",
        "",
        "1. RESPOND ONLY BASED ON THE PROVIDED CONTEXT:",
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

