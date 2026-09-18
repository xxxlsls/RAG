# 掌柜智库（Knowledge Brain）

基于 RAG（检索增强生成）架构的智能知识库系统，支持 PDF/Markdown 文档导入、智能切分、向量检索与多轮对话问答。

## 项目架构

```
┌──────────────┐      ┌──────────────────────────────────────────┐
│   前端页面    │      │              后端服务                     │
│  import.html │─POST─▶│  导入服务 (FastAPI :8000)                │
│  chat.html   │─POST─▶│  查询服务 (FastAPI :8001)                │
└──────────────┘      └──────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │   导入流程       │  │   查询流程       │
          │   (LangGraph)    │  │   (LangGraph)    │
          │                  │  │                  │
          │ entry            │  │ item_name_confirm│
          │  ↓               │  │        ↓         │
          │ pdf_to_md        │  │ ┌────┴────┐      │
          │  ↓               │  │ ▼         ▼      │
          │ md_img           │  │vector_search    │
          │  ↓               │  │hyde_search      │
          │ document_split   │  │web_search_mcp   │
          │  ↓               │  │ └────┬────┘      │
          │ item_name_recog  │  │      ▼          │
          │  ↓               │  │    rrf 融合      │
          │ bge_embedding    │  │      ↓          │
          │  ↓               │  │   rerank        │
          │ import_milvus    │  │      ↓          │
          └─────────────────┘  │ answer_output   │
                               └─────────────────┘
```

## 技术栈

| 类别 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| 流程编排 | LangGraph |
| 向量数据库 | Milvus（混合检索：dense + sparse） |
| 嵌入模型 | BGE-M3（本地） / text-embedding-v4（远程） |
| 重排序模型 | BGE-Reranker-Large |
| 文档解析 | MinerU（PDF → Markdown） |
| LLM | 阿里云 DashScope（OpenAI 兼容接口） |
| 网络搜索 | DashScope MCP WebSearch |
| 对象存储 | MinIO |
| 文档数据库 | MongoDB（历史会话） |
| 前端 | 原生 HTML + JS（SSE 流式输出） |

## 目录结构

```
knowledge/
├── api/                    # FastAPI 路由层
│   ├── import_router.py    # 导入服务路由（:8000）
│   └── query_router.py     # 查询服务路由（:8001）
├── core/                   # 核心配置
│   ├── deps.py             # 依赖注入
│   └── paths.py            # 路径常量
├── front/                  # 前端页面
│   ├── import.html         # 文档导入页
│   └── chat.html           # 对话查询页
├── processor/              # 核心处理流程
│   ├── import_process/     # 导入流程（LangGraph）
│   │   ├── nodes/          # 导入节点
│   │   │   ├── entry.py              # 入口：文件类型判断
│   │   │   ├── pdf_to_md.py          # PDF → Markdown 转换
│   │   │   ├── md_img.py             # 图片扫描与上传
│   │   │   ├── ducment_split.py      # 文档智能切分
│   │   │   ├── item_name_recognition.py  # 商品名识别
│   │   │   ├── bge_embedding.py      # 向量化
│   │   │   └── import_milvus.py      # 写入 Milvus
│   │   ├── main_graph.py   # 导入流程图定义
│   │   ├── state.py        # 导入状态定义
│   │   └── config.py       # 导入配置
│   └── query_process/      # 查询流程（LangGraph）
│       ├── nodes/          # 查询节点
│       │   ├── item_name_confirm.py  # 商品名确认
│       │   ├── vector_search.py      # 向量检索
│       │   ├── hyde_search.py        # HyDE 检索
│       │   ├── web_search_mcp.py     # 网络搜索（MCP）
│       │   ├── rrf.py                # RRF 多路融合
│       │   ├── rerank.py             # 重排序
│       │   └── answer_output.py      # 答案生成
│       ├── main_graph.py   # 查询流程图定义
│       ├── prompt.py       # 查询提示词
│       └── state.py        # 查询状态定义
├── schema/                 # Pydantic 数据模型
├── services/               # 业务服务层
├── utils/                  # 工具类
│   ├── client/             # AI/存储客户端封装
│   ├── embedding_util.py   # 嵌入向量工具
│   ├── milvus_util.py      # Milvus 操作工具
│   ├── mongo_history_util.py # MongoDB 历史工具
│   ├── sse_util.py         # SSE 流式工具
│   └── task_util.py        # 任务状态工具
├── prompt/                 # 导入流程提示词
├── test/                   # 测试脚本
├── temp_data/              # 临时数据（已忽略）
├── .env                    # 环境变量配置
└── requirements.txt        # Python 依赖
```

## 环境要求

- Python 3.10+
- CUDA 12.8（GPU 推理加速）
- Milvus 向量数据库
- MongoDB 文档数据库
- MinIO 对象存储

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> PyTorch 需单独安装，推荐使用 CUDA 12.8 版本：
> ```bash
> pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
> ```

### 2. 配置环境变量

复制 `.env` 文件并修改以下关键配置：

```env
# LLM API（DashScope）
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash
ITEM_MODEL=qwen-flash

# BGE 嵌入模型
BGE_M3=E:/ai_models/BAAI--bge-m3
BGE_DEVICE=cuda:0

# Milvus
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks_v1
ITEM_NAME_COLLECTION=kb_item_names_v1

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=kb001

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files
```

### 3. 启动服务

**导入服务**（端口 8000）：

```bash
python -m api.import_router
```

访问 http://127.0.0.1:8000/import 上传文档。

**查询服务**（端口 8001）：

```bash
python -m api.query_router
```

访问 http://127.0.0.1:8001/chat 开始对话。

## API 接口

### 导入服务（:8000）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/import` | 导入页面 |
| POST | `/upload` | 上传文件（PDF/MD） |
| GET | `/status/{task_id}` | 查询导入任务状态 |

### 查询服务（:8001）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat` | 对话页面 |
| POST | `/query` | 提交查询（支持流式/非流式） |
| GET | `/stream/{task_id}` | SSE 流式获取答案 |
| GET | `/history/{session_id}` | 获取历史对话 |
| DELETE | `/history/{session_id}` | 清空历史对话 |

## 核心流程

### 导入流程

1. **entry** — 判断文件类型（PDF/MD），决定后续处理链路
2. **pdf_to_md** — 调用 MinerU 将 PDF 转为 Markdown
3. **md_img** — 扫描文档中的图片，上传至 MinIO 并替换路径
4. **document_split** — 按标题层级智能切分，短章节合并，长段落装箱
5. **item_name_recognition** — LLM 识别文档中的商品/产品名称
6. **bge_embedding** — 使用 BGE-M3 生成 dense + sparse 向量
7. **import_milvus** — 写入 Milvus，附带商品名标量字段

### 查询流程

1. **item_name_confirm** — 从用户问题中识别相关商品名
2. **三路并行检索**：
   - `vector_search` — BGE-M3 混合检索（按商品名过滤）
   - `hyde_search` — HyDE 假设文档嵌入检索
   - `web_search_mcp` — MCP 网络搜索兜底
3. **rrf** — RRF（Reciprocal Rank Fusion）融合三路结果
4. **rerank** — BGE-Reranker 重排序
5. **answer_output** — 组装提示词，LLM 生成答案（支持流式输出）
