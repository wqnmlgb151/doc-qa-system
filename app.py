import os
import logging

from flask import Flask

from config import UPLOAD_DIR, JSON_DIR, ensure_dir
from middleware import register_security_headers
from routes import api
from state_manager import init_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ensure_dir(UPLOAD_DIR)
ensure_dir(JSON_DIR)

register_security_headers(app)
app.register_blueprint(api)

logger.info("正在初始化向量数据库...")
init_vectorstore()

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() == "true"
    if debug_mode:
        logger.warning("Debug 模式已启用，切勿在生产环境使用")
    logger.info("服务启动: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
