import os
import re
import json
import time
import uuid
import logging
import unicodedata
from functools import wraps
from pathlib import Path
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

from config import UPLOAD_DIR, JSON_DIR, DASHSCOPE_API_KEY, ALLOWED_EXTENSIONS, KNOWLEDGE_STATE_FILE
from document_loader import load_file
from text_processor import process_documents
from vector_store import create_vectorstore, load_vectorstore, get_retriever, add_documents
from qa_chain import create_rag_chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

API_TOKEN = os.environ.get("API_TOKEN", "")
MAX_QUESTION_LENGTH = 2000

vectorstore = None
_cached_retriever = None
_cached_chain = None
processed_files: dict[str, dict] = {}

_requests_store: dict[str, list[float]] = defaultdict(list)

_rate_limits: dict[str, tuple[int, int]] = {
    "upload": (10, 3600),
    "chat": (30, 60),
}


def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    if len(filename) > 200:
        name_part, ext_part = os.path.splitext(filename)
        filename = name_part[:195] + ext_part
    return filename.strip()


def check_rate_limit(endpoint: str) -> bool:
    if endpoint not in _rate_limits:
        return True
    max_req, window = _rate_limits[endpoint]
    now = time.time()
    key = f"{request.remote_addr}:{endpoint}"
    _requests_store[key] = [t for t in _requests_store[key] if now - t < window]
    if not _requests_store[key]:
        del _requests_store[key]
    if len(_requests_store.get(key, [])) >= max_req:
        return False
    _requests_store[key].append(now)
    return True


def _invalidate_cache():
    global _cached_retriever, _cached_chain
    _cached_retriever = None
    _cached_chain = None


def _get_chain():
    global _cached_retriever, _cached_chain
    if _cached_chain is None and vectorstore is not None:
        _cached_retriever = get_retriever(vectorstore)
        _cached_chain = create_rag_chain(_cached_retriever, streaming=True)
    return _cached_chain


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)
        token = request.headers.get("Authorization", "")
        if token != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    return response


def _save_knowledge_state():
    try:
        KNOWLEDGE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(list(processed_files.values()), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存知识库状态失败: {e}")


def _load_knowledge_state():
    global processed_files
    if not KNOWLEDGE_STATE_FILE.exists():
        return
    try:
        with open(KNOWLEDGE_STATE_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        valid_records = {}
        for rec in records:
            fid = rec.get("id", "")
            fname = rec.get("filename", "")
            safe_name = sanitize_filename(fname)
            upload_path = UPLOAD_DIR / f"{fid}_{safe_name}"
            json_path = JSON_DIR / f"{Path(fname).stem}.json"
            if upload_path.exists() and json_path.exists():
                valid_records[fid] = rec
            else:
                logger.info(f"跳过已丢失的文件: {fname}")
        processed_files = valid_records
        if valid_records:
            logger.info(f"已恢复 {len(valid_records)} 个已处理文件的知识库状态")
    except Exception as e:
        logger.warning(f"加载知识库状态失败: {e}")


def _init_vectorstore():
    global vectorstore
    try:
        vectorstore = load_vectorstore()
        if vectorstore:
            logger.info("已加载现有向量数据库")
            _load_knowledge_state()
        else:
            logger.info("向量数据库为空，等待文件上传")
    except Exception as e:
        logger.warning(f"加载向量数据库失败: {e}")
        vectorstore = None
    _invalidate_cache()


def _process_and_index_file(save_path: Path, original_filename: str) -> dict:
    global vectorstore
    docs = load_file(str(save_path))
    logger.info(f"文档加载成功，共 {len(docs)} 页/段")

    result = process_documents(docs, original_filename)
    logger.info(f"文档处理完成: {result.chunk_count} chunks → {result.json_path}")

    if result.chunk_count == 0:
        raise ValueError(
            "文档内容为空或无法提取有效文本。"
            "PDF 可能是扫描版（图片），请使用带有文字层的文档。"
        )

    if vectorstore is None:
        vectorstore = create_vectorstore(result.splits)
        logger.info("已创建新的向量数据库")
    else:
        add_documents(vectorstore, result.splits)
        logger.info("已将文档添加到现有向量数据库")
    _invalidate_cache()
    return result


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready": vectorstore is not None,
        "files": list(processed_files.values()),
        "file_count": len(processed_files),
    })


@app.route("/upload", methods=["POST"])
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
    safe_name = f"{file_id}_{sanitize_filename(file.filename)}"
    save_path = UPLOAD_DIR / safe_name
    file.save(str(save_path))
    logger.info(f"文件已保存: {save_path}")

    try:
        result = _process_and_index_file(save_path, file.filename)
        processed_files[file_id] = {
            "id": file_id,
            "filename": file.filename,
            "chunks": result.chunk_count,
            "chars": result.total_chars,
            "type": ext,
        }
        _save_knowledge_state()
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "chunks": result.chunk_count,
            "total_chars": result.total_chars,
            "json_path": result.json_path,
        })

    except Exception as e:
        logger.error(f"处理文件失败: {e}", exc_info=True)
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            pass
        return jsonify({"error": "处理文件失败，请检查文件格式"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    if not check_rate_limit("chat"):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

    chain = _get_chain()
    if chain is None:
        return jsonify({"error": "请先上传文件建立知识库"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请输入问题"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({"error": f"问题长度不能超过{MAX_QUESTION_LENGTH}字符"}), 400

    def generate():
        try:
            full_response = ""
            for chunk in chain.stream(question):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"
        except Exception as e:
            logger.error(f"流式生成失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': '服务内部错误，请重试'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify(list(processed_files.values()))


@app.route("/api/files/<file_id>", methods=["DELETE"])
@require_auth
def delete_file(file_id: str):
    global vectorstore
    if file_id not in processed_files:
        return jsonify({"error": "文件不存在"}), 404

    info = processed_files[file_id]
    safe_name = sanitize_filename(info["filename"])
    disk_prefix = f"{file_id}_{safe_name}"
    for f in UPLOAD_DIR.glob(f"{file_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in JSON_DIR.glob(f"{Path(safe_name).stem}*"):
        try:
            f.unlink()
        except Exception:
            pass
    if vectorstore:
        try:
            vectorstore.delete(where={"filename": disk_prefix})
        except Exception:
            pass
    del processed_files[file_id]
    _save_knowledge_state()
    return jsonify({"success": True})


if __name__ == "__main__":
    logger.info("正在初始化向量数据库...")
    _init_vectorstore()

    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() == "true"
    if debug_mode:
        logger.warning("Debug 模式已启用，切勿在生产环境使用")
    logger.info("服务启动: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
