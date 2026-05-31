import json
import logging
import threading
from pathlib import Path
from typing import Optional

from config import (
    UPLOAD_DIR,
    JSON_DIR,
    KNOWLEDGE_STATE_FILE,
    RETRIEVAL_K_PRE_RERANK,
    ensure_dir,
)
from document_loader import load_file
from text_processor import ProcessResult, process_documents
from vector_store import (
    create_vectorstore,
    load_vectorstore,
    get_retriever,
    add_documents,
)
from middleware import compose_safe_name

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Thread-safe knowledge base encapsulating vectorstore, retriever, and file state.

    Uses a single lock — upload operations hold it longer (embedding API calls)
    but uploads are infrequent. The hot path (chat) only needs the lock briefly.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vectorstore = None
        self._retriever = None
        self._processed_files: dict[str, dict] = {}

    # ── read-only queries (brief lock) ──────────────────────────

    @property
    def vectorstore(self):
        """Return the raw vectorstore for direct access (e.g. deletion by metadata)."""
        with self._lock:
            return self._vectorstore

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._vectorstore is not None

    @property
    def files(self) -> list[dict]:
        with self._lock:
            return list(self._processed_files.values())

    @property
    def file_count(self) -> int:
        with self._lock:
            return len(self._processed_files)

    def has_file(self, file_id: str) -> bool:
        with self._lock:
            return file_id in self._processed_files

    def get_file(self, file_id: str) -> dict | None:
        with self._lock:
            return self._processed_files.get(file_id)

    # ── retriever access (cached, thread-safe) ──────────────────

    def get_retriever(self) -> "Chroma | None":
        """Return a cached retriever for the current vectorstore, or None."""
        with self._lock:
            if self._retriever is not None:
                return self._retriever
            if self._vectorstore is None:
                return None
            self._retriever = get_retriever(self._vectorstore, k=RETRIEVAL_K_PRE_RERANK)
            return self._retriever

    # ── initialisation ──────────────────────────────────────────

    def init_vectorstore(self) -> None:
        """Load existing vectorstore and knowledge state on startup."""
        with self._lock:
            try:
                self._vectorstore = load_vectorstore()
                if self._vectorstore:
                    logger.info("已加载现有向量数据库")
                    self._load_knowledge_state_locked()
                else:
                    logger.info("向量数据库为空，等待文件上传")
            except (OSError, RuntimeError) as e:
                logger.warning(f"加载向量数据库失败: {e}")
                self._vectorstore = None
            self._retriever = None

    # ── file processing ─────────────────────────────────────────

    def process_and_index_file(self, save_path: Path, original_filename: str, file_id: str) -> ProcessResult:
        """Load a file, create chunks, and index into the vectorstore.

        Returns the ProcessResult for the caller to extract metadata.
        The caller is responsible for calling add_file_record afterwards.
        """
        # Phase 1: document loading (may do OCR — slow, no lock)
        docs = load_file(str(save_path))
        logger.info(f"文档加载成功，共 {len(docs)} 页/段")

        result = process_documents(docs, original_filename, file_id)
        logger.info(f"文档处理完成: {result.chunk_count} chunks → {result.json_path}")

        if result.chunk_count == 0:
            from document_loader import EmptyDocumentError
            raise EmptyDocumentError(
                "文档内容为空或无法提取有效文本。"
                "PDF 可能是扫描版（图片），请使用带有文字层的文档。"
            )

        # Phase 2: index into vectorstore (embedding API calls — slow, hold lock)
        with self._lock:
            if self._vectorstore is None:
                self._vectorstore = create_vectorstore(result.splits)
                logger.info("已创建新的向量数据库")
            else:
                add_documents(self._vectorstore, result.splits)
                logger.info("已将文档添加到现有向量数据库")
            self._retriever = None

        return result

    def add_file_record(self, file_id: str, record: dict) -> None:
        """Register a file record and persist to disk."""
        with self._lock:
            self._processed_files[file_id] = record
            data = list(self._processed_files.values())
        self._write_state_file(data)

    def remove_file(self, file_id: str) -> dict | None:
        """Remove a file record and its vector entries."""
        with self._lock:
            info = self._processed_files.pop(file_id, None)
            if info is None:
                return None
            fname = info.get("filename", "")
            if fname and self._vectorstore:
                safe_name = compose_safe_name(file_id, fname)
                try:
                    self._vectorstore.delete(where={"filename": safe_name})
                except Exception:
                    logger.warning(f"从向量库删除文件失败: {fname}", exc_info=True)
            self._retriever = None
            data = list(self._processed_files.values())
        self._write_state_file(data)
        return info

    # ── persistence ─────────────────────────────────────────────

    def _write_state_file(self, data: list[dict]) -> None:
        """Persist file records to disk. I/O is outside the lock."""
        try:
            ensure_dir(KNOWLEDGE_STATE_FILE.parent)
            with open(KNOWLEDGE_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"保存知识库状态失败: {e}")

    def _load_knowledge_state_locked(self) -> None:
        """Must be called with _lock held."""
        if not KNOWLEDGE_STATE_FILE.exists():
            return
        try:
            with open(KNOWLEDGE_STATE_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"读取知识库状态文件失败: {e}")
            return

        for rec in records:
            fid = rec.get("id", "")
            fname = rec.get("filename", "")
            upload_path = UPLOAD_DIR / compose_safe_name(fid, fname)
            stem = Path(fname).stem
            # Try new naming first, then old (backwards compat)
            json_path = JSON_DIR / f"{fid}_{stem}.json"
            if not json_path.exists():
                json_path = JSON_DIR / f"{stem}.json"
            if upload_path.exists() and json_path.exists():
                self._processed_files[fid] = rec
            else:
                logger.info(f"跳过已丢失的文件: {fname}")

        if self._processed_files:
            logger.info(f"已恢复 {len(self._processed_files)} 个已处理文件的知识库状态")


# Module-level singleton — the single source of truth for all state.
kb = KnowledgeBase()


def init_vectorstore():
    """Convenience function for backward-compatible startup."""
    kb.init_vectorstore()
