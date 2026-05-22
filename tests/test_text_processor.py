import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from text_processor import (
    ProcessResult,
    docs_to_json,
    process_documents,
    save_json,
    split_documents,
)


class TestSplitDocuments:
    def test_returns_list_of_documents(self):
        docs = [Document(page_content="Hello world", metadata={"source": "test.txt"})]
        result = split_documents(docs)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_preserves_source_metadata_in_output(self):
        docs = [Document(page_content="Hello world", metadata={"source": "test.txt", "page": 1})]
        result = split_documents(docs)
        for doc in result:
            assert "source" in doc.metadata

    def test_handles_multiple_input_documents(self):
        docs = [
            Document(page_content="First document content with enough text to be meaningful.", metadata={"id": 1}),
            Document(page_content="Second document content also with sufficient text here.", metadata={"id": 2}),
        ]
        result = split_documents(docs)
        assert len(result) >= 2

    def test_handles_empty_list(self):
        result = split_documents([])
        assert result == []

    def test_splits_long_content_into_multiple_chunks(self):
        long_text = "This is a test sentence. " * 100
        docs = [Document(page_content=long_text, metadata={"source": "long.txt"})]
        result = split_documents(docs)
        assert len(result) > 1


class TestDocsToJson:
    def test_converts_document_to_json_serializable_dict(self):
        docs = [Document(page_content="Content 1", metadata={"src": "a.txt"})]
        result = docs_to_json(docs)
        assert len(result) == 1
        assert result[0]["chunk_id"] == 0
        assert result[0]["content"] == "Content 1"
        assert result[0]["char_count"] == 9
        assert result[0]["metadata"] == {"src": "a.txt"}

    def test_generates_sequential_chunk_ids(self):
        docs = [
            Document(page_content="A", metadata={}),
            Document(page_content="BB", metadata={}),
            Document(page_content="CCC", metadata={}),
        ]
        result = docs_to_json(docs)
        assert [r["chunk_id"] for r in result] == [0, 1, 2]

    def test_char_count_matches_content_length(self):
        docs = [Document(page_content="hello world", metadata={})]
        result = docs_to_json(docs)
        assert result[0]["char_count"] == 11

    def test_empty_list_returns_empty_list(self):
        result = docs_to_json([])
        assert result == []


class TestSaveJson:
    def test_creates_json_file_with_correct_content(self, tmp_path):
        data = [{"chunk_id": 0, "content": "test", "char_count": 4, "metadata": {}}]
        with patch("text_processor.JSON_DIR", tmp_path):
            path = save_json(data, "test_file.pdf")
            assert path == str(tmp_path / "test_file.json")
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            assert saved == data

    def test_creates_parent_directory_if_needed(self, tmp_path):
        nested_dir = tmp_path / "sub" / "nested"
        data = [{"chunk_id": 0, "content": "x", "char_count": 1, "metadata": {}}]
        with patch("text_processor.JSON_DIR", nested_dir):
            path = save_json(data, "doc.txt")
            assert Path(path).exists()

    def test_strips_original_extension_and_uses_json(self, tmp_path):
        data: list = []
        with patch("text_processor.JSON_DIR", tmp_path):
            path = save_json(data, "report.final.docx")
            assert path.endswith(".json")
            assert "report.final" in path


class TestProcessDocuments:
    def test_returns_process_result_with_correct_fields(self):
        docs = [Document(page_content="Hello world. This is a test document.", metadata={"source": "test.txt"})]
        result = process_documents(docs, "test.txt")
        assert isinstance(result, ProcessResult)
        assert result.chunk_count > 0
        assert result.total_chars > 0
        assert result.json_path.endswith(".json")
        assert len(result.splits) == result.chunk_count
        assert len(result.json_data) == result.chunk_count

    def test_total_chars_equals_sum_of_char_counts(self):
        docs = [Document(page_content="This is a test document with some content.", metadata={})]
        result = process_documents(docs, "test.txt")
        expected = sum(d["char_count"] for d in result.json_data)
        assert result.total_chars == expected

    def test_handles_multiple_input_docs(self):
        docs = [
            Document(page_content="First document with enough text to split properly.", metadata={"id": 1}),
            Document(page_content="Second document also with sufficient text here.", metadata={"id": 2}),
        ]
        result = process_documents(docs, "multi.txt")
        assert result.chunk_count >= 2


class TestProcessResult:
    def test_is_frozen_dataclass(self):
        pr = ProcessResult(splits=[], json_data=[], json_path="a.json", chunk_count=0, total_chars=0)
        with pytest.raises(Exception):
            pr.chunk_count = 5

    def test_all_fields_accessible(self):
        splits = [Document(page_content="x", metadata={})]
        json_data = [{"chunk_id": 0}]
        pr = ProcessResult(
            splits=splits, json_data=json_data, json_path="/tmp/x.json", chunk_count=1, total_chars=1
        )
        assert pr.splits == splits
        assert pr.json_data == json_data
        assert pr.json_path == "/tmp/x.json"
        assert pr.chunk_count == 1
        assert pr.total_chars == 1
