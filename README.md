# 文档智能分析系统

基于 **RAG（检索增强生成）** 的文档问答系统。上传文档后自动建立向量知识库，支持自然语言提问，AI 基于文档内容实时回答。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（阿里云百炼平台）
echo "DASHSCOPE_API_KEY=your-key-here" > .env

# 启动
python app.py
# → http://localhost:5000
```

## 功能

| 功能 | 说明 |
|------|------|
| 多格式上传 | PDF / Word / PPT / Excel / HTML / 图片 / 文本（5 大类 19 种格式） |
| PDF OCR | 扫描版 PDF 三级降级：PyPDF → PyMuPDF → DashScope 多模态 OCR |
| Rerank 重排序 | Cross-encoder 精排，初检 12 → 精选 4，检索质量最高单点改进 |
| 流式回答 | SSE 逐 token 渲染，支持取消生成 |
| 知识库持久化 | 重启自动恢复，无需重新上传 |
| 安全防护 | 8层防护：限流 / 认证 / HSTS / CSP / XSS 防护 / 文件净化 / 日志脱敏 |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/api/status` | 知识库就绪状态 + 文件列表 |
| `POST` | `/upload` | 上传文件 |
| `POST` | `/chat` | 问答（SSE 流式返回） |
| `GET` | `/api/files` | 已上传文件列表 |
| `DELETE` | `/api/files/<id>` | 删除文件 |
| `POST` | `/api/stream/<id>/cancel` | 取消正在生成的回答 |

## 架构

```
static/index.html      前端 SPA（上传 + 聊天 + 文件管理）
        │
routes.py              路由层 — HTTP 请求/响应 + pre-flight 检索
middleware.py           中间件层 — 限流 / 认证 / 安全头 / 净化
state_manager.py        状态管理 — KnowledgeBase Class + threading.Lock
        │
document_loader.py      文档加载 — 5 大类格式 + PDF 三级 OCR
text_processor.py       文本处理 — 拆分 + JSON 结构化
vector_store.py         向量存储 — ChromaDB + DashScope 嵌入
rerank.py               Cross-encoder 重排序（gte-rerank）
qa_chain.py             RAG chain — LLM 生成（DeepSeek-V3）
```

## 测试

```bash
pytest tests/ -v                 # 46 项用例
pytest tests/ --cov=. --cov-report=term-missing
```

## 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | Flask 3.x |
| LLM | DeepSeek-V3（DashScope） |
| 嵌入 | DashScope text-embedding-v2 |
| 向量库 | ChromaDB（持久化） |
| 重排序 | DashScope gte-rerank |
| OCR | DashScope qwen-vl-max |
| PDF | PyPDF + PyMuPDF |
| 文档解析 | python-docx / python-pptx / openpyxl / BeautifulSoup4 |
| 测试 | pytest |

## 环境要求

- Python 3.10+
- [阿里云百炼](https://bailian.console.aliyun.com/) API Key（云端模式）**或** Ollama 等本地模型服务（离线模式）

## 使用本地模型（可选）

系统支持接入 **Ollama / Xinference / vLLM / LM Studio** 等任何 OpenAI 兼容接口的本地模型服务。

### Ollama 方案（推荐）

```bash
# 1. 安装 Ollama → https://ollama.com/download/windows
# 2. 拉取模型
ollama pull qwen2.5:14b              # LLM 问答（中文优秀）
ollama pull nomic-embed-text         # 文本嵌入

# 3. 配置 .env
DASHSCOPE_API_KEY=ollama
LLM_MODEL=qwen2.5:14b
EMBEDDING_MODEL=nomic-embed-text
BASE_URL=http://localhost:11434/v1

# 4. 启动
python app.py
```

### 其他本地服务

任何提供 OpenAI 兼容 `/v1/chat/completions` + `/v1/embeddings` 端点的服务均可接入：

| 服务 | 默认地址 | 配置方式 |
|------|---------|---------|
| Xinference | `http://localhost:9997/v1` | 同上，修改 BASE_URL / LLM_MODEL |
| vLLM | `http://localhost:8000/v1` | 同上 |
| LM Studio | `http://localhost:1234/v1` | 同上 |

> Rerank 和 OCR 在纯本地模式下会自动降级（Rerank 降级→向量排序；OCR 不可用→仅支持文字型文档）。
> 详见 [答辩文档.md](./答辩文档.md) 第十章。

## 许可证

MIT
