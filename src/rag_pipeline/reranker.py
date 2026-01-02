import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any
from tqdm import tqdm

class CrossEncoderReranker:
    """
    Реранкер на основе кросс-энкодера для переранжирования релевантных документов
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Инициализация реранкера
        
        Args:
            model_name: Название модели для реранжирования
        """
        print(f"Загрузка модели реранкера: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
            
    def rerank(self, query: str, documents: list[dict[str, Any]], batch_size: int = 8) -> list[dict[str, Any]]:
        """
        Переранжирование документов на основе их релевантности запросу
        
        Args:
            query: Запрос пользователя
            documents: Список документов для реранжирования
            batch_size: Размер батча для обработки
            
        Returns:
            Отсортированный по релевантности список документов
        """
        
        print(f"\nРеранжирование {len(documents)} документов для запроса: '{query}'")
        
        pairs = [(query, doc["text"]) for doc in documents]
        
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), batch_size), desc="Реранжирование"):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.view(-1).float().cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        print("Топ-3 документа после реранжирования:")
        for i, doc in enumerate(reranked_docs[:3], 1):
            article = doc["metadata"]["article_number"]
            chapter = doc["metadata"]["chapter"]
            print(f"  {i}. {article} ({chapter}) - rerank_score: {doc['rerank_score']:.4f}")
        
        return reranked_docs
