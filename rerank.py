import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from config import DASHSCOPE_API_KEY, RETRIEVAL_K

logger = logging.getLogger(__name__)

RERANK_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


@dataclass(frozen=True)
class RerankResult:
    docs: List[Document]
    degraded: bool


def rerank_documents(query: str, docs: List[Document], top_k: int = RETRIEVAL_K) -> RerankResult:
    """Use DashScope gte-rerank (cross-encoder) to rerank retrieval results.

    Returns RerankResult with degraded=True when the API call fails and
    results fall back to original vector-similarity ordering.
    """
    if len(docs) == 0:
        return RerankResult(docs=[], degraded=False)
    if len(docs) <= top_k:
        return RerankResult(docs=list(docs), degraded=False)

    documents = [doc.page_content for doc in docs]

    payload = json.dumps({
        "model": "gte-rerank",
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            "top_n": top_k,
            "return_documents": False,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        RERANK_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.load(resp)

        ranked_indices = [
            item["index"]
            for item in result.get("output", {}).get("results", [])
        ]
        reranked = [docs[i] for i in ranked_indices if i < len(docs)]
        return RerankResult(docs=reranked, degraded=False)

    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"Rerank API 调用失败，降级为原始顺序: {e}")
        return RerankResult(docs=docs[:top_k], degraded=True)
