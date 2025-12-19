import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

def generate_embeddings(chunks_file_path: str, output_dir: str):
    """
    Генерация эмбеддингов для чанков с использованием модели all-MiniLM-L6-v2
    
    Args:
        chunks_file_path: Путь к файлу с метаданными чанков
        output_dir: Директория для сохранения эмбеддингов
    """
    print("Загрузка чанков из файла метаданных...")
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        chunks_metadata = json.load(f)
    
    chunk_texts = [chunk["text_preview"].rstrip("...") for chunk in chunks_metadata]
    print(f"Загружено {len(chunk_texts)} чанков для векторизации")
    
    print("\nЗагрузка модели sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print(f"Модель загружена")
    
    print("\nГенерация эмбеддингов для чанков...")
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Векторизация"):
        batch_texts = chunk_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    print(f"Сгенерировано {len(embeddings)} эмбеддингов")
    print(f"Размерность эмбеддингов: {embeddings.shape[1]}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    embeddings_path = Path(output_dir) / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Эмбеддинги сохранены в: {embeddings_path}")
    
    metadata_path = Path(output_dir) / "embeddings_metadata.json"
    metadata = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": embeddings.shape[1],
        "chunk_count": len(embeddings),
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Метаданные эмбеддингов сохранены в: {metadata_path}")
    
    return embeddings, chunk_texts, chunks_metadata

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_metadata.json"
EMBEDDINGS_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

embeddings, chunk_texts, chunks_metadata = generate_embeddings(
    str(CHUNKS_METADATA_PATH),
    str(EMBEDDINGS_OUTPUT_DIR)
)
