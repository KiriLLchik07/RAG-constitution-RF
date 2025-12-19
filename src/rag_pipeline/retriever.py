from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class ConstitutionRetriever:
    """
    Класс для поиска релевантных статей Конституции РФ по запросу пользователя
    """
    
    def __init__(self, persist_directory: str):
        """
        Инициализация ретривера
        
        Args:
            persist_directory: Путь к директории с сохраненной векторной БД
        """
        self.persist_directory = Path(persist_directory)
        
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Директория векторной БД не найдена: {self.persist_directory}")
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=chromadb.Settings(
                anonymized_telemetry=False
            )
        )
        
        self.collection = self.client.get_collection(
            name="constitution_rag",
            embedding_function=self.embedding_function
        )
        
        print(f"Ретривер инициализирован. В коллекции {self.collection.count()} документов.")
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу
        
        Args:
            query: Запрос пользователя
            n_results: Количество возвращаемых результатов
        
        Returns:
            Список релевантных документов с метаданными
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            doc = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - (results["distances"][0][i] / 2)
            }
            retrieved_docs.append(doc)
        
        print(f"Найдено {len(retrieved_docs)} релевантных документов для запроса: '{query}'")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  {i}. {doc['metadata']['article_number']} ({doc['metadata']['chapter']}) - score: {doc['score']:.4f}")
        
        return retrieved_docs
    
    def get_context_for_llm(self, query: str, n_results: int = 3) -> str:
        """
        Формирование контекста для LLM из релевантных документов
        
        Args:
            query: Запрос пользователя
            n_results: Количество документов для контекста
        
        Returns:
            Строка с контекстом для LLM
        """
        documents = self.retrieve(query, n_results)
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"Источник {i} ({doc['metadata']['article_number']}, {doc['metadata']['chapter']}):\n"
                f"{doc['text']}\n"
            )
        
        return "\n\n".join(context_parts)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_db"

try:
    retriever = ConstitutionRetriever(str(DB_PATH))
    
    test_queries = [
        "Какие права имеет гражданин Российской Федерации?",
        "Что такое светское государство?",
        "Как происходит избрание Президента РФ?",
        "Какие полномочия у Государственной Думы?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Тестовый запрос: {query}")
        print(f"{'='*50}")
        context = retriever.get_context_for_llm(query, n_results=3)
        print("\nСформированный контекст для LLM:")
        print("-"*50)
        print(context[:500] + "..." if len(context) > 500 else context)
        
except Exception as e:
    print(f"Ошибка при инициализации ретривера: {e}")
