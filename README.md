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
| 安全防护 | 限流 / 认证 / CSP / XSS 防护 / 文件净化 |

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
- [阿里云百炼](https://bailian.console.aliyun.com/) API Key

## 许可证

MIT
