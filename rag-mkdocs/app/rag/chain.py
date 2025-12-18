"""
Chain module: RAG chain building logic.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    # Fallback for older LangChain versions
    def create_stuff_documents_chain(llm, prompt):
        def chain(inputs):
            context = "\n\n".join([doc.page_content for doc in inputs.get("context", [])])
            return llm.invoke(prompt.format_messages(context=context, input=inputs.get("input", "")))
        return chain
    
    def create_retrieval_chain(retriever, combine_docs_chain):
        def chain(inputs):
            docs = retriever.invoke(inputs.get("input", ""))
            return combine_docs_chain({"context": docs, "input": inputs.get("input", "")})
        return chain

from app.core.prompt_config import (
    PromptSettings,
    load_prompt_settings_from_env,
    build_system_prompt,
)
from app.infra.openai_utils import get_chat_llm
from app.rag.indexing import build_or_load_vectorstore, get_vectorstore_dir
from app.rag.retrieval import build_retriever

logger = logging.getLogger(__name__)


def build_rag_chain(
    vectorstore,
    prompt_settings: Optional[PromptSettings] = None,
    k: Optional[int] = None,
    model: str = "gpt-4o",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Creates RAG chain from ready vectorstore.
    
    Uses settings from PromptSettings for system prompt and LLM parameters.
    
    Args:
        vectorstore: Ready FAISS vectorstore
        prompt_settings: Prompt settings (if None, loaded from environment)
        k: Number of relevant chunks (if None, used from prompt_settings)
        model: OpenAI model (default "gpt-4o")
        temperature: Generation temperature (if None, used from prompt_settings)
        max_tokens: Maximum number of tokens (optional)
        
    Returns:
        RAG chain for answering questions
    """
    if prompt_settings is None:
        prompt_settings = load_prompt_settings_from_env()
    
    effective_k = k if k is not None else prompt_settings.default_top_k
    effective_temperature = temperature if temperature is not None else prompt_settings.default_temperature
    effective_max_tokens = max_tokens if max_tokens is not None else prompt_settings.default_max_tokens

    effective_k = max(1, min(10, effective_k))
    effective_temperature = max(0.0, min(1.0, effective_temperature))
    if effective_max_tokens is not None:
        effective_max_tokens = max(128, min(4096, effective_max_tokens))

    # Build retriever
    retriever = build_retriever(
        vectorstore,
        k=effective_k,
        model=model,
        temperature=effective_temperature,
        max_tokens=effective_max_tokens
    )
    
    # Build LLM
    llm = get_chat_llm(
        temperature=effective_temperature,
        model=model,
        max_tokens=effective_max_tokens,
    )
    
    # Build system prompt template as a variable placeholder
    # The actual system_prompt will be provided at runtime (allows Jinja2 rendering)
    default_system_prompt = build_system_prompt(prompt_settings, response_language="{response_language}")
    
    system_template = "{system_prompt}"
    
    human_template = (
        "Documentation context (relevant fragments):\n\n"
        "{context}\n\n"
        "---\n\n"
        "Conversation history:\n{chat_history}\n\n"
        "User question: {input}\n\n"
        "Instructions:\n"
        "- Use ONLY the information from the context\n"
        "- Answer as clearly and structurally as possible\n"
        "- Provide examples and step-by-step instructions when helpful\n"
        "- If the context is insufficient, explain what exactly is missing"
    )

    logger.info("Creating prompt template with dynamic system_prompt...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])
    
    logger.info("Creating Stuff Documents Chain...")
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    logger.info("Creating Retrieval Chain...")
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    logger.info("RAG chain successfully created")
    return rag_chain


def get_rag_chain(
    index_path: Optional[str] = None,
    k: Optional[int] = None,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None
):
    """
    Creates RAG chain by loading vectorstore and building chain.
    
    Helper function for backward compatibility. Uses build_rag_chain().
    
    Args:
        index_path: Path to directory with FAISS index
        k: Number of relevant chunks (if None, used from settings)
        model: OpenAI model (default "gpt-4o-mini")
        temperature: Generation temperature (if None, used from settings)
        
    Returns:
        RAG chain for answering questions
    """
    logger.info("=" * 60)
    logger.info("RAG CHAIN INITIALIZATION")
    logger.info("=" * 60)
    
    logger.info("Loading vector store...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    rag_chain = build_rag_chain(vectorstore, k=k, model=model, temperature=temperature)
    
    logger.info("=" * 60)
    return rag_chain


def build_rag_chain_and_settings(
    index_path: Optional[str] = None
):
    """
    Build RAG chain and load settings in one call.
    
    Convenience function for application startup.
    
    Args:
        index_path: Optional path to index directory (uses VECTORSTORE_DIR if None)
        
    Returns:
        Tuple (rag_chain, vectorstore, prompt_settings)
    """
    from app.core.prompt_config import load_prompt_settings_from_env
    
    if index_path is None:
        index_path = get_vectorstore_dir()
    
    logger.info("=" * 60)
    logger.info("RAG CHAIN INITIALIZATION")
    logger.info("=" * 60)
    
    logger.info("Loading vector store...")
    vectorstore = build_or_load_vectorstore(chunks=None, index_path=index_path)
    
    logger.info("Loading prompt settings...")
    prompt_settings = load_prompt_settings_from_env()
    
    logger.info("Building RAG chain...")
    rag_chain = build_rag_chain(
        vectorstore,
        prompt_settings=prompt_settings,
        k=prompt_settings.default_top_k,
        temperature=prompt_settings.default_temperature
    )
    
    logger.info("=" * 60)
    return rag_chain, vectorstore, prompt_settings

