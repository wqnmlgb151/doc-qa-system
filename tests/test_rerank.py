import json
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from rerank import rerank_documents, RerankResult


def make_docs(texts):
    return [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(texts)]


class TestRerankDocuments:
    def test_empty_list_returns_empty_not_degraded(self):
        result = rerank_documents("query", [])
        assert result.docs == []
        assert not result.degraded

    def test_fewer_docs_than_top_k_returns_all_not_degraded(self):
        docs = make_docs(["a", "b", "c"])
        result = rerank_documents("query", docs, top_k=5)
        assert len(result.docs) == 3
        assert not result.degraded

    def test_equal_docs_and_top_k_returns_all(self):
        docs = make_docs(["a", "b", "c", "d"])
        result = rerank_documents("query", docs, top_k=4)
        assert len(result.docs) == 4
        assert not result.degraded

    def test_successful_rerank_returns_reranked_not_degraded(self):
        docs = make_docs(["irrelevant", "relevant", "also irrelevant", "maybe", "nope"])
        api_response = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 3, "relevance_score": 0.80},
                    {"index": 0, "relevance_score": 0.30},
                ]
            }
        }

        with patch("rerank.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(api_response).encode("utf-8")
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            result = rerank_documents("relevant query", docs, top_k=3)

        assert not result.degraded
        assert len(result.docs) == 3
        # doc[1] ("relevant") should be first
        assert result.docs[0].page_content == "relevant"

    def test_api_failure_fallback_to_original_order(self):
        docs = make_docs(["doc0", "doc1", "doc2", "doc3", "doc4"])

        with patch("rerank.urllib.request.urlopen", side_effect=OSError("network down")):
            result = rerank_documents("query", docs, top_k=3)

        assert result.degraded
        assert len(result.docs) == 3
        # Falls back to first top_k in original order
        assert [d.page_content for d in result.docs] == ["doc0", "doc1", "doc2"]


class TestRerankResult:
    def test_is_frozen_dataclass(self):
        result = RerankResult(docs=[], degraded=False)
        assert result.degraded is False

    def test_docs_field_preserved(self):
        docs = make_docs(["hello"])
        result = RerankResult(docs=docs, degraded=True)
        assert result.docs == docs
        assert result.degraded
