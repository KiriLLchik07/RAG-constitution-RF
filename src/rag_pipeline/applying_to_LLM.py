import logging
from typing import Any, Optional
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from .prompt_engineering import create_constitution_prompt_template, create_system_prompt
from .retriever import ConstitutionRetriever
import time
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/logs/applying_to_LLM.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConstitutionQA")

class ConstitutionQA:
    """
    Система вопросов и ответов по Конституции РФ с использованием RAG
    """

    def __init__(self, retriever: ConstitutionRetriever, model_name: str = "mistral:instruct", temperature: float = 0.1,
                max_retries: int = 3, timeout: int = 120):
        """
        Инициализация системы вопросов и ответов
        
        Args:
            retriever: Экземпляр ConstitutionRetriever для поиска документов
            model_name: Название модели в Ollama
            temperature: Температура генерации (0.0-1.0)
            max_retries: Максимальное количество попыток при ошибках
            timeout: Таймаут запроса в секундах
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        Path("logs").mkdir(exist_ok=True)
        
        logger.info(f"Инициализация LLM: {model_name} с temperature={temperature}")
        
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            timeout=timeout
        )
        
        self.prompt_template = create_constitution_prompt_template()
        
        self.qa_chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Система вопросов и ответов успешно инициализирована")
    
    def _log_interaction(self, query: str, context: str, response: str, sources: list[dict[str, Any]], 
                         execution_time: float ) -> None:
        """
        Логирование взаимодействия с LLM
        
        Args:
            query: Запрос пользователя
            context: Контекст, переданный в LLM
            response: Ответ LLM
            sources: Источники контекста
            execution_time: Время выполнения в секундах
        """
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "context": context,
            "response": response,
            "sources": sources,
            "execution_time": execution_time,
            "model": self.model_name,
            "temperature": self.temperature
        }
        
        with open("data/logs/interactions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def answer_question(self, query: str, chat_history: Optional[list[BaseMessage]] = None,
                        n_initial: int = 10, n_final: int = 5) -> dict[str, Any]:
        """
        Ответ на вопрос пользователя по Конституции РФ
        
        Args:
            query: Запрос пользователя
            chat_history: История диалога (опционально)
            n_initial: Количество документов для первичного поиска
            n_final: Количество документов для финального контекста
            
        Returns:
            Словарь с ответом, источниками и метаданными
        """
        start_time = time.time()
        logger.info(f"Обработка запроса: '{query}'")
        
        try:
            documents = self.retriever.retrieve(
                query=query,
                n_initial=n_initial,
                n_final=n_final
            )
            logger.info(f"Получено {len(documents)} релевантных документов")
            
            prompt_vars = create_system_prompt(
                query=query,
                retrieved_docs=documents,
                chat_history=chat_history
            )
            
            response_text = None
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Попытка {attempt + 1}/{self.max_retries} генерации ответа")
                    response = self.qa_chain.invoke(prompt_vars)
                    response_text = response.strip() if isinstance(response, str) else str(response)
                    break
                except Exception as e:
                    logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1 * (attempt + 1))
            
            execution_time = time.time() - start_time
            sources = [
                {
                    "article_number": doc["metadata"]["article_number"],
                    "chapter": doc["metadata"]["chapter"],
                    "text_excerpt": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "score": doc.get("rerank_score", doc.get("score", 0))
                }
                for doc in documents
            ]
            
            result = {
                "query": query,
                "answer": response_text or "Не удалось получить ответ от LLM",
                "sources": sources,
                "execution_time": execution_time,
                "model": self.model_name,
                "temperature": self.temperature
            }
            
            context = prompt_vars["context"]
            self._log_interaction(
                query=query,
                context=context,
                response=result["answer"],
                sources=sources,
                execution_time=execution_time
            )
            
            logger.info(f"Запрос успешно обработан за {execution_time:.2f} секунд")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            
            error_result = {
                "query": query,
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}. Пожалуйста, попробуйте позже.",
                "sources": [],
                "execution_time": execution_time,
                "error": str(e)
            }
            
            self._log_interaction(
                query=query,
                context="",
                response=error_result["answer"],
                sources=[],
                execution_time=execution_time
            )
            
            return error_result
    
    def answer_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """
        Пакетная обработка запросов
        
        Args:
            queries: Список запросов
            
        Returns:
            Список результатов для каждого запроса
        """
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Обработка запроса {i}/{len(queries)}: '{query}'")
            result = self.answer_question(query)
            results.append(result)
            time.sleep(0.5)
        
        return results
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

LOG_DIR.mkdir(exist_ok=True)

try:
    logger.info("Инициализация ретривера...")
    retriever = ConstitutionRetriever(str(DB_PATH), use_reranker=True)
    
    logger.info("Инициализация QA системы...")
    qa_system = ConstitutionQA(
        retriever=retriever,
        model_name="mistral:instruct",
        temperature=0.1,
        timeout=120
    )
    
    test_questions = [
        "Какие права имеет гражданин Российской Федерации?",
        "Что такое светское государство?",
        "Как происходит избрание Президента РФ?",
        "Какие полномочия у Государственной Думы?"
    ]
    
    print("\n" + "="*80)
    print("Тестирование системы вопросов и ответов по коснтитуции РФ")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-'*80}")
        print(f"Вопрос {i}/{len(test_questions)}: {question}")
        print("-"*80)
        
        result = qa_system.answer_question(question)
        
        print("\Ответ:")
        print("-" * 40)
        print(result["answer"])
        
        print("\Источники:")
        print("-" * 40)
        for j, source in enumerate(result["sources"], 1):
            print(f"{j}. {source['article_number']} ({source['chapter']})")
            print(f"   Релевантность: {source['score']:.4f}")
            print(f"   Текст: {source['text_excerpt']}")
        
        print(f"\nВремя обработки: {result['execution_time']:.2f} секунд")

except Exception as e:
    logger.error(f"Критическая ошибка в основном скрипте: {e}", exc_info=True)
    print(f"Критическая ошибка: {e}")
