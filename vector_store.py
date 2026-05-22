from functools import lru_cache
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL, CHROMA_DIR, RETRIEVAL_K


@lru_cache(maxsize=1)
def _get_embeddings():
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )


def create_vectorstore(docs: List[Document]) -> Chroma:
    embedding = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def load_vectorstore() -> Chroma | None:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return None
    embedding = _get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding,
    )


def get_retriever(vectorstore: Chroma, k: int | None = None):
    if k is None:
        k = RETRIEVAL_K
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def add_documents(vectorstore: Chroma, docs: List[Document]):
    vectorstore.add_documents(docs)
