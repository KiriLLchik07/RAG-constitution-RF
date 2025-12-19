from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from typing import Any, Optional
from pathlib import Path

def load_system_prompt() -> str:
    """
    Загружает системный промпт из файла конфигурации
    
    Returns:
        Строка с системным промптом
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    prompt_path = project_root / "configs" / "prompts" / "system_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Файл системного промпта не найден: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    return system_prompt

def create_constitution_prompt_template() -> ChatPromptTemplate:
    """
    Создает шаблон промпта для работы с Конституцией РФ
    
    Returns:
        ChatPromptTemplate для формирования запросов к LLM
    """
    system_prompt = load_system_prompt()
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(
            "Пользовательский запрос: {query}\n\n"
            "Контекст (релевантные статьи Конституции РФ):\n{context}"
        )
    ])
    
    return prompt_template

def format_context(documents: list[dict[str, Any]]) -> str:
    """
    Форматирует найденные документы в читаемый формат для промпта
    
    Args:
        documents: Список документов с метаданными
        
    Returns:
        Отформатированная строка с контекстом
    """
    formatted_context = []
    
    for i, doc in enumerate(documents, 1):
        chapter = doc["metadata"].get("chapter", "не указана")
        article = doc["metadata"].get("article_number", "не указан")
        text = doc["text"].strip()
        
        formatted_context.append(
            f"Источник {i}:\n"
            f"Глава: {chapter}\n"
            f"Статья: {article}\n"
            f"Текст статьи:\n{text}\n"
            f"{'='*50}"
        )
    
    return "\n\n".join(formatted_context)

def create_system_prompt(
    query: str, 
    retrieved_docs: list[dict[str, Any]], 
    chat_history: Optional[list] = None
) -> dict[str, Any]:
    """
    Формирует полный системный промпт для LLM
    
    Args:
        query: Запрос пользователя
        retrieved_docs: Релевантные документы из векторной БД
        chat_history: История диалога (опционально)
        
    Returns:
        Словарь с переменными для промпта
    """
    context = format_context(retrieved_docs)
    
    prompt_variables = {
        "query": query,
        "context": context
    }
    
    if chat_history:
        prompt_variables["chat_history"] = chat_history
    
    return prompt_variables
