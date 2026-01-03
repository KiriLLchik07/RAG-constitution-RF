from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="qilowoq/bge-reranker-v2-m3-en-ru"):
        self.model = HuggingFaceCrossEncoder(model_name=model_name)

    def rerank(self, query, documents):
        pairs = [(query, d["text"]) for d in documents]
        scores = self.model.score(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents
