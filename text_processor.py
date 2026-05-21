import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import JSON_DIR, CHUNK_SIZE, CHUNK_OVERLAP


@dataclass(frozen=True)
class ProcessResult:
    splits: List[Document]
    json_data: List[dict]
    json_path: str
    chunk_count: int
    total_chars: int


def split_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(docs)


def docs_to_json(docs: List[Document]) -> List[dict]:
    return [
        {
            "chunk_id": i,
            "content": doc.page_content,
            "char_count": len(doc.page_content),
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(docs)
    ]


def save_json(data: List[dict], filename: str) -> str:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output_name = Path(filename).stem + ".json"
    output_path = JSON_DIR / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(output_path)


def process_documents(docs: List[Document], source_filename: str) -> ProcessResult:
    splits = split_documents(docs)
    json_data = docs_to_json(splits)
    json_path = save_json(json_data, source_filename)
    return ProcessResult(
        splits=splits,
        json_data=json_data,
        json_path=json_path,
        chunk_count=len(splits),
        total_chars=sum(d["char_count"] for d in json_data),
    )
