import re
from pathlib import Path
from langchain_core.documents import Document
import json

def chunk_constitution(text_path: str) -> list[Document]:
    """Чанкирование Конституции РФ с автоматическим определением глав и поддержкой дополнительных номеров статей"""
    
    file_path = Path(text_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    print(f"Текст загружен. Длина: {len(text)} символов")
    
    chapter_pattern = r'(?:^|\n)(ГЛАВА\s+\d+\.)(?:\s*([^\n]+))?'
    chapters = re.split(chapter_pattern, text)[1:]
    
    chunks = []
    current_chapter = "Преамбула"
    
    for i in range(0, len(chapters), 3):
        chapter_title = chapters[i].strip()
        chapter_subtitle = chapters[i+1].strip() if i+1 < len(chapters) and chapters[i+1] else ""
        
        if chapter_subtitle:
            current_chapter = f"{chapter_title} {chapter_subtitle}".strip()
        else:
            current_chapter = chapter_title
        
        chapter_content = chapters[i+2] if i+2 < len(chapters) else ""
        
        if "ЗАКЛЮЧИТЕЛЬНЫЕ И ПЕРЕХОДНЫЕ ПОЛОЖЕНИЯ" in current_chapter:
            continue
            
        article_blocks = re.split(r'(\nСтатья\s+\d+(?:\.\d+)*[^.]*)', chapter_content)[1:]
        
        for j in range(0, len(article_blocks), 2):
            if j+1 >= len(article_blocks):
                break
                
            article_header = article_blocks[j].strip()
            article_content = article_blocks[j+1].strip()
            
            article_num_match = re.search(r'Статья\s+(\d+(?:\.\d+)*)', article_header)
            if not article_num_match:
                continue
                
            article_number = f"Статья {article_num_match.group(1)}"
            
            article_content_clean = re.sub(
                r'^\s*Статья\s+\d+(?:\.\d+)*[^\n]*\n?',
                '',
                article_content,
                count=1,
                flags=re.IGNORECASE | re.MULTILINE
            )

            article_content_clean = re.sub(r'<\d+>.*?(?=\n|$)', '', article_content_clean, flags=re.DOTALL)
            article_content_clean = re.sub(r'<\*>.*?(?=\n|$)', '', article_content_clean, flags=re.DOTALL)
            article_content_clean = re.sub(r'\s+', ' ', article_content_clean).strip()

            full_text = f"{article_header}\n{article_content_clean}"
            full_text = re.sub(r'(\d+)\s*\.\s*', r'\1. ', full_text)
            
            chunks.append(Document(
                page_content=full_text,
                metadata={
                    "chapter": current_chapter,
                    "article_number": article_number,
                    "source": "Конституция РФ"
                }
            ))
            
    print(f"Создано {len(chunks)} чанков (статей)")
    return chunks

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEXT_PATH = PROJECT_ROOT / "data" / "raw" / "constitution_rf_clean.txt"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_metadata.json"

chunks = chunk_constitution(str(TEXT_PATH))

unique_chunks = {}
for chunk in chunks:
    key = (chunk.metadata["chapter"], chunk.metadata["article_number"])
    unique_chunks[key] = chunk
chunks = list(unique_chunks.values())

metadata = [{
    "chapter": chunk.metadata["chapter"],
    "article": chunk.metadata["article_number"],
    "full_text": chunk.page_content
} for chunk in chunks]

METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nМетаданные сохранены: {METADATA_PATH}")
print(f"Статистика по главам:")

chapter_stats = {}
for chunk in chunks:
    chapter = chunk.metadata["chapter"]
    chapter_stats[chapter] = chapter_stats.get(chapter, 0) + 1

for chapter, count in chapter_stats.items():
    print(f"  {chapter}: {count} статей")
