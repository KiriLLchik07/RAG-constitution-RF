from pathlib import Path
from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer
from .reranker import CrossEncoderReranker

class ConstitutionRetriever:
    """
    Класс для поиска релевантных статей Конституции РФ по запросу пользователя
    с поддержкой реранжирования
    """
    
    def __init__(self, persist_directory: str, use_reranker: bool = True, embedding_model_name: str = "deepvk/USER-bge-m3"):
        """
        Инициализация ретривера
        
        Args:
            persist_directory: Путь к директории с сохраненной векторной БД
            use_reranker: Флаг использования реранкера
            embedding_model_name: Модель для векторизации запросов
        """
        self.persist_directory = Path(persist_directory)
        self.use_reranker = use_reranker
        
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Директория векторной БД не найдена: {self.persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_collection(
            name="constitution_rag"
        )
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        if self.use_reranker:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        print(f"Ретривер инициализирован. В коллекции {self.collection.count()} документов.")
        if self.use_reranker:
            print("Реранкер активирован для улучшения качества поиска")
    
    def retrieve(self, query: str, n_initial: int = 10, n_final: int = 5, relevance_threshold: float = 0.5) -> list[dict[str, Any]]:
        """
        Поиск релевантных документов по запросу с реранжированием
        
        Args:
            query: Запрос пользователя
            n_initial: Количество документов для первичного поиска
            n_final: Количество документов для финального результата
        
        Returns:
            Список релевантных документов с метаданными
        """
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_initial,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            doc = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            retrieved_docs.append(doc)
        
        if not retrieved_docs:
            return []
        
        print(f"Найдено {len(retrieved_docs)} документов на первичном этапе для запроса: '{query}'")
         
        if self.use_reranker and self.reranker is not None:
            reranked_docs = self.reranker.rerank(query, retrieved_docs.copy())
            filtered_docs = [
                doc for doc in reranked_docs
                if doc["rerank_score"] >= relevance_threshold
            ]

            if not filtered_docs:
                return []

            return filtered_docs[:n_final]
    
    def get_context_for_llm(self, query: str, n_initial: int = 10, n_final: int = 5) -> str:
        """
        Формирование контекста для LLM из релевантных документов
        
        Args:
            query: Запрос пользователя
            n_initial: Количество документов для первичного поиска
            n_final: Количество документов для контекста
        
        Returns:
            Строка с контекстом для LLM
        """
        documents = self.retrieve(query, n_initial, n_final)
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"Источник {i} ({doc['metadata']['article_number']}, {doc['metadata']['chapter']}):\n"
                f"{doc['text']}\n"
            )
        
        return "\n\n".join(context_parts)
