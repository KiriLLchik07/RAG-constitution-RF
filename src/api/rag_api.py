import logging
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from pathlib import Path
from rag_pipeline.applying_to_LLM import ConstitutionQA
from rag_pipeline.retriever import ConstitutionRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ROOT_DIR / "data" / "logs" / "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConstitutionAPI")

DB_PATH = ROOT_DIR / "data" / "vector_db" / "chroma_db"

try:
    logger.info("Инициализация ретривера для API...")
    retriever = ConstitutionRetriever(str(DB_PATH), use_reranker=True)
    
    logger.info("Инициализация QA системы для API...")
    qa_system = ConstitutionQA(
        retriever=retriever,
        model_name="mistral:instruct",
        temperature=0.1,
        timeout=120
    )
    logger.info("API успешно инициализировано")
except Exception as e:
    logger.critical(f"Критическая ошибка при инициализации API: {e}", exc_info=True)
    raise

app = FastAPI(
    title="ConstitutionQA API",
    description="API для вопросов и ответов по Конституции РФ"
)

class QuestionRequest(BaseModel):
    question: str
    n_initial: int = 10
    n_final: int = 5
    chat_history: Optional[list[dict[str, Any]]] = None

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    execution_time: float
    model: str
    temperature: float
    error: Optional[str] = None

@app.post("/ask", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Получает ответ на вопрос по Конституции РФ
    
    Args:
        request: Вопрос пользователя и параметры поиска
        
    Returns:
        Ответ с источниками и метаданными
    """
    logger.info(f"Получен запрос: '{request.question}'")
    
    try:
        result = qa_system.answer_question(
            query=request.question,
            chat_history=request.chat_history,
            n_initial=request.n_initial,
            n_final=request.n_final
        )
        
        if "error" in result:
            logger.error(f"Ошибка при обработке запроса: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
            
        logger.info(f"Запрос успешно обработан за {result['execution_time']:.2f} секунд")
        return result
        
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Проверка работоспособности API
    """
    return {"status": "healthy", "service": "ConstitutionQA API"}
