import json
from pathlib import Path
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def initialize_vector_db(chunks_metadata_path: str, persist_directory: str):
    """
    Инициализация и заполнение векторной базы данных ChromaDB
    
    Args:
        chunks_metadata_path: Путь к файлу с метаданными чанков
        persist_directory: Директория для сохранения векторной БД
    """

    with open(chunks_metadata_path, "r", encoding="utf-8") as f:
        chunks_metadata = json.load(f)
    
    print(f"Загружено {len(chunks_metadata)} чанков для векторизации")
    
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    collection_name = "constitution_rag"
    
    try:
        client.delete_collection(collection_name)
        print(f"Удалена существующая коллекция '{collection_name}'")
    except NotFoundError:
        print(f"Коллекция '{collection_name}' не существует. Создаем новую...")

    print(f"Коллекция '{collection_name}' создана")
        
    ids = [f"id_{i}" for i in range(len(chunks_metadata))]
    documents = [chunk["full_text"] for chunk in chunks_metadata]

    model = SentenceTransformer("deepvk/USER-bge-m3")

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
        get_or_create=True
    )
    
    metadatas = []
    for chunk in chunks_metadata:
        metadata = {
            "chapter": chunk["chapter"],
            "article_number": chunk["article"],
            "source": "Конституция РФ"
        }
        metadatas.append(metadata)
    
    batch_size = 50
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(ids), batch_size), desc="Заполнение БД", total=total_batches):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
    
    print(f"Векторная база данных успешно заполнена {len(ids)} документами")
    return client, collection
    
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKS_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_metadata.json"
PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "vector_db" / "chroma_db"

if not CHUNKS_METADATA_PATH.exists():
    raise FileNotFoundError(f"Файл метаданных не найден: {CHUNKS_METADATA_PATH}")

client, collection = initialize_vector_db(
    str(CHUNKS_METADATA_PATH),
    str(PERSIST_DIRECTORY)
)

print(f"\nВекторная база данных успешно инициализирована и сохранена в: {PERSIST_DIRECTORY}")
