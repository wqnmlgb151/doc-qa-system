import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent

# DashScope API
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is required. Set it in .env file.")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "deepseek-v3.2-exp"
EMBEDDING_MODEL = "text-embedding-v2"

# 目录
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
JSON_DIR = BASE_DIR / "data" / "json"
CHROMA_DIR = BASE_DIR / "chroma_db"
KNOWLEDGE_STATE_FILE = BASE_DIR / "data" / "knowledge_state.json"

# 文本拆分
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# 检索
RETRIEVAL_K = 4

# 支持的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".txt", ".md", ".csv", ".json", ".xml", ".log"}
