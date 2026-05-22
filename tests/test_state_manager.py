import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from state_manager import KnowledgeBase, kb


class TestKnowledgeBaseInit:
    def test_starts_not_ready(self):
        k = KnowledgeBase()
        assert not k.is_ready
        assert k.file_count == 0
        assert k.files == []

    def test_has_file_returns_false_for_unknown(self):
        k = KnowledgeBase()
        assert not k.has_file("nonexistent")

    def test_get_file_returns_none_for_unknown(self):
        k = KnowledgeBase()
        assert k.get_file("missing") is None

    def test_get_retriever_returns_none_when_not_ready(self):
        k = KnowledgeBase()
        assert k.get_retriever() is None

    def test_vectorstore_property_returns_none_initially(self):
        k = KnowledgeBase()
        assert k.vectorstore is None


class TestKnowledgeBaseFiles:
    def test_add_file_record(self, tmp_path):
        k = KnowledgeBase()
        with patch("state_manager.KNOWLEDGE_STATE_FILE", tmp_path / "state.json"):
            k.add_file_record("abc", {"id": "abc", "filename": "test.pdf"})

        assert k.has_file("abc")
        assert k.file_count == 1
        assert k.get_file("abc") == {"id": "abc", "filename": "test.pdf"}

    def test_remove_file(self, tmp_path):
        k = KnowledgeBase()
        with patch("state_manager.KNOWLEDGE_STATE_FILE", tmp_path / "state.json"):
            k.add_file_record("abc", {"id": "abc", "filename": "test.pdf"})
            info = k.remove_file("abc")

        assert info == {"id": "abc", "filename": "test.pdf"}
        assert not k.has_file("abc")
        assert k.file_count == 0

    def test_remove_nonexistent_returns_none(self):
        k = KnowledgeBase()
        assert k.remove_file("nope") is None

    def test_files_is_a_snapshot(self, tmp_path):
        k = KnowledgeBase()
        with patch("state_manager.KNOWLEDGE_STATE_FILE", tmp_path / "state.json"):
            k.add_file_record("1", {"id": "1"})

        snap = k.files
        # Mutating the snapshot should not affect internal state
        snap.append({"id": "2"})
        assert k.file_count == 1


class TestKnowledgeBaseThreadSafety:
    def test_concurrent_add_file_records(self, tmp_path):
        k = KnowledgeBase()
        errors = []

        with patch("state_manager.KNOWLEDGE_STATE_FILE", tmp_path / "state.json"):
            def add_one(i):
                try:
                    k.add_file_record(str(i), {"id": str(i), "n": i})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_one, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0
        assert k.file_count == 20

    def test_concurrent_remove(self, tmp_path):
        k = KnowledgeBase()
        with patch("state_manager.KNOWLEDGE_STATE_FILE", tmp_path / "state.json"):
            for i in range(10):
                k.add_file_record(str(i), {"id": str(i), "filename": f"doc_{i}.pdf"})

            def remove_one(i):
                k.remove_file(str(i))

            threads = [threading.Thread(target=remove_one, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert k.file_count == 0


class TestKnowledgeBaseInitVectorstore:
    def test_loads_existing_vectorstore(self):
        k = KnowledgeBase()
        with patch("state_manager.load_vectorstore") as mock_load:
            mock_vs = MagicMock()
            mock_load.return_value = mock_vs

            with patch("state_manager.KNOWLEDGE_STATE_FILE", Path("/nonexistent")):
                k.init_vectorstore()

            assert k.is_ready
            assert k.vectorstore is mock_vs

    def test_no_existing_vectorstore_stays_not_ready(self):
        k = KnowledgeBase()
        with patch("state_manager.load_vectorstore", return_value=None):
            k.init_vectorstore()

        assert not k.is_ready

    def test_load_failure_stays_not_ready(self):
        k = KnowledgeBase()
        with patch("state_manager.load_vectorstore", side_effect=RuntimeError("boom")):
            k.init_vectorstore()

        assert not k.is_ready


class TestKnowledgeBaseSingleton:
    def test_kb_is_knowledge_base_instance(self):
        assert isinstance(kb, KnowledgeBase)
