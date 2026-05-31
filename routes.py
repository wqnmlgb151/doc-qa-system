import json
import logging
import re
import uuid
from pathlib import Path

from flask import Blueprint, request, jsonify, send_from_directory, Response, stream_with_context

from config import ALLOWED_EXTENSIONS, UPLOAD_DIR, JSON_DIR, MAX_QUESTION_LENGTH
from middleware import check_rate_limit, require_auth, compose_safe_name
from document_loader import DocumentLoaderError
from rerank import rerank_documents
from qa_chain import create_chain_from_docs
from state_manager import kb

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__)

# ── stream cancellation ────────────────────────────────────────

import threading
import time

_active_streams: dict[str, tuple[threading.Event, float]] = {}
_STREAM_TTL = 300  # 5 minutes — stale entries evicted on next registration


def _prune_stale_streams(now: float | None = None) -> None:
    """Remove stream entries that have exceeded the TTL."""
    if now is None:
        now = time.time()
    stale = [sid for sid, (_, ts) in _active_streams.items() if now - ts > _STREAM_TTL]
    for sid in stale:
        _active_streams.pop(sid, None)


def _cancel_stream(stream_id: str):
    entry = _active_streams.get(stream_id)
    if entry:
        entry[0].set()


# ── routes ─────────────────────────────────────────────────────

@api.route("/")
def index():
    return send_from_directory("static", "index.html")


@api.route("/api/status")
def api_status():
    return jsonify({
        "ready": kb.is_ready,
        "files": kb.files,
        "file_count": kb.file_count,
    })


@api.route("/upload", methods=["POST"])
@require_auth
def upload():
    if not check_rate_limit("upload"):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "不支持的文件类型", "allowed": list(ALLOWED_EXTENSIONS)}), 400

    file_id = uuid.uuid4().hex[:8]
    safe_name = compose_safe_name(file_id, file.filename)
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))
    logger.info(f"文件已保存: {save_path}")

    try:
        result = kb.process_and_index_file(save_path, file.filename, file_id)
    except DocumentLoaderError as e:
        logger.error(f"文档加载失败: {e} — {e.detail}")
        _cleanup_upload(save_path)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"处理文件失败: {e}", exc_info=True)
        _cleanup_upload(save_path)
        return jsonify({"error": "处理文件失败，请检查文件格式"}), 500

    record = {
        "id": file_id,
        "filename": file.filename,
        "chunks": result.chunk_count,
        "chars": result.total_chars,
        "type": ext,
    }
    kb.add_file_record(file_id, record)

    return jsonify({
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "chunks": result.chunk_count,
        "total_chars": result.total_chars,
    })


@api.route("/chat", methods=["POST"])
def chat():
    if not check_rate_limit("chat"):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

    retriever = kb.get_retriever()
    if retriever is None:
        return jsonify({"error": "请先上传文件建立知识库"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请输入问题"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"问题长度不能超过{MAX_QUESTION_LENGTH}字符"}), 400

    # ── Phase 1: Retrieve + Rerank (synchronous, fast) ──
    try:
        raw_docs = retriever.invoke(question)
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        return jsonify({"error": "检索服务异常，请稍后重试"}), 500

    rerank_result = rerank_documents(question, raw_docs)

    if not rerank_result.docs:
        return jsonify({"error": "未找到相关内容，请尝试换个问题"}), 404

    # ── Phase 2: LLM generation (streaming) ──
    chain = create_chain_from_docs(rerank_result.docs, streaming=True)
    stream_id = uuid.uuid4().hex
    cancel_event = threading.Event()
    _prune_stale_streams()
    _active_streams[stream_id] = (cancel_event, time.time())

    def generate():
        try:
            # Send metadata first (includes stream_id for cancellation)
            yield f"data: {json.dumps({'stream_id': stream_id})}\n\n"

            if rerank_result.degraded:
                yield f"data: {json.dumps({'warning': 'Rerank 精排服务暂不可用，当前使用向量相似度排序，检索质量可能降低'})}\n\n"

            full_response = ""
            for chunk in chain.stream(question):
                if cancel_event.is_set():
                    logger.info(f"流式生成已取消 [{stream_id}]")
                    break
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            if not cancel_event.is_set():
                yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"

        except GeneratorExit:
            logger.info(f"客户端断开连接 [{stream_id}]")
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': '服务内部错误，请重试'})}\n\n"
        finally:
            _active_streams.pop(stream_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api.route("/api/stream/<stream_id>/cancel", methods=["POST"])
def cancel_stream_route(stream_id: str):
    if not check_rate_limit("cancel"):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
    _cancel_stream(stream_id)
    return jsonify({"success": True})


@api.route("/api/files", methods=["GET"])
def list_files():
    return jsonify(kb.files)


@api.route("/api/files/<file_id>", methods=["DELETE"])
@require_auth
def delete_file(file_id: str):
    if not re.fullmatch(r"[0-9a-fA-F]{8}", file_id):
        return jsonify({"error": "无效的文件 ID"}), 400
    info = kb.remove_file(file_id)
    if info is None:
        return jsonify({"error": "文件不存在"}), 404

    _rm_glob(UPLOAD_DIR, f"{file_id}_*")
    _rm_glob(JSON_DIR, f"{file_id}_*")

    return jsonify({"success": True})


# ── helpers ────────────────────────────────────────────────────

def _rm_glob(directory: Path, pattern: str):
    for f in directory.glob(pattern):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            logger.warning(f"删除文件失败: {f}", exc_info=True)


def _cleanup_upload(save_path: Path):
    try:
        save_path.unlink(missing_ok=True)
    except OSError:
        logger.warning(f"清理上传文件失败: {save_path}", exc_info=True)
