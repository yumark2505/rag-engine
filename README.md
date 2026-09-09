# FastAPI RAG System

Hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh sử dụng FastAPI + Ollama + pgvector + LangChain.

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT / FRONTEND                              │
│                                  Streamlit                                  │
│                              streamlit/app.py                               │
│                                                                             │
│        Upload PDF  •  Lọc theo tài liệu  •  Chọn chiến lược  •  Chat UI      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP REST / Streaming SSE
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               FASTAPI SERVER                                │
│                                  main.py                                    │
│                                                                             │
│  ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────────┐  │
│  │      Lifespan      │   │      RAG API       │   │    Document API     │  │
│  │                    │   │     RagApi.py      │   │     document.py     │  │
│  │ • Load Registry    │   │                    │   │                     │  │
│  │ • Init Pipelines   │   │ • /rag/query       │   │ • /documents/list   │  │
│  │ • CORS Middleware  │   │ • /rag/stream      │   │ • /documents/folder │  │
│  │                    │   │ • /rag/query/stream│   │ • /documents/upload │  │
│  └────────────────────┘   └─────────┬──────────┘   │ • /documents/reindex│  │
│                                     │              └──────────┬──────────┘  │
│            ┌────────────────────────┘                         │             │
│            │ (Online Serving Request)                         │ (Async Task)│
│            ▼                                                  ▼             │
│  ┌───────────────────────────────────┐      ┌────────────────────────────┐  │
│  │       RAG SERVING PIPELINE        │      │   RAG INGESTION PIPELINE   │  │
│  │        RAGServingPipeline         │      │    RAGIngestionPipeline    │  │
│  │                                   │      │                            │  │
│  │  1. PRE-RETRIEVAL                 │      │  1. Loader (Unstructured)  │  │
│  │     • Identity • Transform • HyDE │      │             │              │  │
│  │     • Contextualize (Chat Memory) │      │             ▼              │  │
│  │             │                     │      │  2. Chunker                │  │
│  │             ▼                     │      │     (Recursive / Semantic) │  │
│  │  2. RETRIEVAL (Metadata Filter)   │      │             │              │  │
│  │     • Dense Vector (pgvector)     │      │             ▼              │  │
│  │     • BM25 (Sparse)               │      │  3. Embedding              │  │
│  │     • Hybrid Search (RRF Fusion)  │      │     (Ollama / OpenAI...)   │  │
│  │             │                     │      │             │              │  │
│  │             ▼                     │      │             ▼              │  │
│  │  3. POST-RETRIEVAL                │      │  4. Vector Store Manager   │  │
│  │     • FlashRank Reranker          │      │     (PGVector / Metadata)  │  │
│  │     • Contextual Compression      │      └─────────────┬──────────────┘  │
│  │             │                     │                    │                 │
│  │             ▼                     │                    │                 │
│  │  4. GENERATION                    │                    │                 │
│  │     • Ollama / OpenAI / vLLM      │                    │                 │
│  │     • Multi-turn Memory update    │                    │                 │
│  └─────────────┬─────────────────────┘                    │                 │
│                │                                          │                 │
└────────────────┼──────────────────────────────────────────┼─────────────────┘
                 │ Read Vectors / Filter Query              │ Write Vectors
                 ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE & VECTOR STORE                           │
│                   PostgreSQL + pgvector (Container Docker)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục

```
Fastapi-rag/
│
├── .env                              # Biến môi trường (DB URL, LLM, Chunking...)
├── .venv/                            # Môi trường ảo Python
├── config.py                         # Cấu hình tập trung (Pydantic BaseSettings)
├── main.py                           # FastAPI entrypoint, lifespan, CORS
├── requirements.txt                  # Danh mục dependencies
├── test_ingest.py                    # Script test nhanh ingestion
│
├── data/
│   └── Document/                     # Tài liệu nạp vào (.pdf, .docx, .txt...)
│
├── frontend/                         # Giao diện Streamlit
│   └── app.py                        # Streamlit UI đa chiến lược + Chat Memory
│
├── opt/                              # FlashRank cache directory
│
├── rag/                              # CORE PIPELINE ENGINE
│   ├── __init__.py                   # Khởi tạo package, nạp modules
│   ├── registry.py                   # Registry Pattern trung tâm
│   ├── memory.py                     # ChatMessage + ChatMemory + chat_memory
│   ├── pipeline.py                   # RAGIngestionPipeline & RAGServingPipeline
│   │
│   ├── ingestion/                    # GIAI ĐOẠN 1: NẠP & ĐÁNH CHỈ MỤC
│   │   ├── Loader.py                 # @registry.loader("unstructured")
│   │   ├── Chunker.py                # @registry.chunker("recursive", "tiktoken", "semantic")
│   │   ├── Embedding.py              # @registry.embedder("ollama")
│   │   └── VectorStore.py            # @registry.vector_store("pgvector")
│   │
│   ├── retrieval/                    # GIAI ĐOẠN 2: TRUY VẤN 
│   │   ├── PreRetrieval.py           # IdentityQueryOptimizer, QueryOptimizer, HyDEOptimizer
│   │   ├── Retrieval.py              # VectorSearchRetriever, BM25SearchRetriever, HybridSearchRetriever
│   │   └── PostRetrieval.py          # FlashReranker, ContextualCompressor
│   │
│   ├── prompt/                       # GIAI ĐOẠN 3: PROMPT TEMPLATES
│   │   └── prompt.py                 # BasePrompt, BasicPrompt, ConversationalPrompt
│   │
│   └── generation/                   # GIAI ĐOẠN 4: SINH PHẢN HỒI (LLM GENERATION)
│       ├── base.py                   # BaseGenerator (ABC): _build_prompt_text, _build_messages
│       ├── ollama.py                 # @registry.generator("ollama")
│       ├── openai.py                 # @registry.generator("openai")
│       ├── google.py                 # @registry.generator("google")
│       └── vllm.py                   # @registry.generator("vllm")
│
├── routers/                          # FASTAPI ROUTE CONTROLLERS
│   ├── RagApi.py                     # POST /rag/query, /rag/stream, /rag/query/stream
│   └── document.py                   # CRUD documents + upload + reindex + status
│
├── schemas/                          # DATA SCHEMAS & DTO (Pydantic V2)
│   └── RagSchema.py                  # RagQueryRequest, RagQueryResponse, IngestionRequest
│
└── tests/                            # Test directory (chưa có file)
```

## Yêu cầu hệ thống

- Python 3.11+
- Ollama đã cài đặt và chạy (`ollama serve`)
- Docker Desktop (Chỉ dùng duy nhất để chạy dịch vụ PostgreSQL có cài sẵn `pgvector`)

## Cài đặt

### 1. Clone và cài dependencies

```bash
cd Fastapi-rag
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```env
# ── App & Server ─────────────────────────────────────────────
APP_NAME="Fastapi-rag"
DEBUG=False
HOST="0.0.0.0"
PORT=8000

# ── Database (pgvector) ──────────────────────────────────────
PGVECTOR_URL="postgresql://postgres:postgres@localhost:5432/rag"
TOP_K=8

# ── Chunking Settings ────────────────────────────────────────
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# ── Local LLM & Embedding (Ollama) ───────────────────────────
OLLAMA_BASE_URL="http://127.0.0.1:11434"
LLM_MODEL="qwen3:8b"
EMBEDDING_MODEL="nomic-embed-text:latest"
BATCH_SIZE=64

# ── OpenAI ───────────────────────────────────────────────────
OPENAI_API_KEY=""
OPENAI_MODEL=""

# ── Google AI (Gemini) ───────────────────────────────────────
GEMINI_API_KEY=""
GEMINI_MODEL=""

# ── Private GPU Server (vLLM) ────────────────────────────────
VLLM_BASE_URL="http://127.0.0.1:8001/v1"
VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

### 3. Khởi tạo PostgreSQL + pgvector

```bash
docker run -d --name fastapi-rag -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag \
  --restart unless-stopped \
  pgvector/pgvector:pg16
```

```sql
CREATE DATABASE rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Tải model Ollama

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text:latest
```

### 5. Nạp dữ liệu Offline

Đặt các tệp PDF vào thư mục `data/Document/`. Server sẽ tự động nạp khi khởi động.

## Chạy

### Khởi động Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ tự động:
1. **Phase 1 (Ingestion)**: Nạp toàn bộ tài liệu trong `data/Document/` → chunk → embed → lưu vào pgvector
2. **Phase 2 (Serving)**: Khởi tạo `RAGServingPipeline` để phục vụ truy vấn

### Khởi động Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Mở trình duyệt tại `http://localhost:8501`

## API Reference

### Health Check

```http
GET /
```

**Response:**
```json
{"message": "FastAPI RAG Service is running!"}
```

---

### RAG Query Endpoints

#### POST /rag/query — Truy vấn RAG (JSON)

Trả về kết quả JSON hoàn chỉnh kèm trích dẫn nguồn.

```http
POST /rag/query
Content-Type: application/json

{
    "query": "Câu hỏi của bạn?",
    "session_id": "default_session",
    "selected_document": null,
    "llm_provider": "ollama",
    "top_k": 8,
    "top_n": 3,
    "pre_retrieval_strategy": "identity",
    "retrieval_strategy": "vector",
    "post_retrieval_strategy": "rerank"
}
```

**Request Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | **required** | Câu hỏi gửi lên RAG |
| `session_id` | string | `"default_session"` | Định danh phiên hội thoại |
| `selected_document` | string/null | `null` | Lọc theo tên tệp tài liệu cụ thể |
| `llm_provider` | string | `"ollama"` | LLM: `ollama`, `openai`, `google`, `vllm` |
| `top_k` | integer | `5` | Số chunk tìm kiếm ban đầu (1-20) |
| `top_n` | integer | `3` | Số chunk giữ lại sau rerank (1-10) |
| `pre_retrieval_strategy` | string | `"identity"` | `identity`, `query_transform`, `query_optimizer`, `hyde` |
| `retrieval_strategy` | string | `"vector"` | `vector`, `bm25`, `hybrid` |
| `post_retrieval_strategy` | string | `"rerank"` | `rerank`, `contextual_compression` |

**Response (200 OK):**
```json
{
    "query": "Câu hỏi của bạn?",
    "sub_queries": ["Câu hỏi của bạn?"],
    "answer": "Câu trả lời từ hệ thống RAG...",
    "sources": [
        {
            "content": "Nội dung trích dẫn...",
            "metadata": {"filename": "report.pdf", "rerank_score": 0.95}
        }
    ]
}
```

---

#### POST /rag/stream — Truy vấn RAG (Streaming)

Streaming từng token phục vụ Streamlit/Client.

```http
POST /rag/stream
Content-Type: application/json

{ ... giống /rag/query ... }
```

**Response:** Streaming text/plain ( từng token JSON ).

---

#### POST /rag/query/stream — Truy vấn RAG (SSE)

Server-Sent Events chuẩn `data: <token>`.

```http
POST /rag/query/stream
Content-Type: application/json

{ ... giống /rag/query ... }
```

**Response:** `text/event-stream` format.

---

### Document Endpoints

#### POST /documents/upload — Tải lên tài liệu (Background)

Tải PDF lên và xử lý ngầm (chunk + embed + vectorize).

```http
POST /documents/upload
Content-Type: multipart/form-data

- file: <PDF file>           # Bắt buộc, chỉ chấp nhận .pdf
- chunker_name: "recursive"  # tùy chọn: "recursive", "tiktoken", "semantic"
- embedder_name: "ollama"    # tùy chọn: "ollama", "openai", "google", "huggingface"
```

**Response (202 Accepted):**
```json
{
    "task_id": "abc-123",
    "filename": "report.pdf",
    "status": "PENDING",
    "message": "Tệp đã tiếp nhận. Phân đoạn: 'recursive', Embedding: 'ollama'."
}
```

---

#### POST /documents/reindex — Re-index toàn bộ

Quét lại toàn bộ tệp trong `data/Document/` và index lại theo chiến lược mới.

```http
POST /documents/reindex
Content-Type: application/json

{
    "chunker_name": "recursive",
    "embedder_name": "ollama"
}
```

**Response (202 Accepted):**
```json
{
    "task_id": "abc-123",
    "total_files": 3,
    "status": "PENDING",
    "message": "Bắt đầu re-index 3 tệp với Chunker: 'recursive', Embedder: 'ollama'."
}
```

---

#### GET /documents/folder — Liệt kê tệp PDF trong thư mục

```http
GET /documents/folder
```

**Response:**
```json
{
    "files": ["report1.pdf", "report2.pdf"]
}
```

---

#### GET /documents/list — Liệt kê tài liệu đã nạp vào pgvector

```http
GET /documents/list
```

**Response:**
```json
{
    "documents": ["report1.pdf", "report2.pdf"]
}
```

---

#### DELETE /documents/{filename} — Xóa tài liệu

Xóa toàn bộ chunks của một tệp khỏi PostgreSQL và thư mục vật lý.

```http
DELETE /documents/report.pdf
```

**Response:**
```json
{
    "status": "success",
    "message": "Đã xóa hoàn toàn dữ liệu và tệp 'report.pdf'."
}
```

---

#### GET /documents/status/{task_id} — Kiểm tra trạng thái tác vụ

```http
GET /documents/status/abc-123
```

**Response:**
```json
{
    "task_id": "abc-123",
    "status": "COMPLETED"
}
```

Status có thể: `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED: <reason>`.

---

## Chiến lược truy vấn

### Pre-Retrieval (Tiền xử lý truy vấn)

| Strategy | Registry Key | Mô tả |
|----------|-------------|--------|
| `identity` | `pre_retrieval_identity` | Giữ nguyên câu hỏi (mặc định) — `IdentityQueryOptimizer` |
| `query_transform` | `pre_retrieval_query_optimizer` | Phân tách câu hỏi thành sub-questions — `QueryOptimizer` |
| `query_optimizer` | `pre_retrieval_query_optimizer` | Truy cập trực tiếp `QueryOptimizer` |
| `hyde` | `pre_retrieval_hyde` | Tạo hypothetical document — `HyDEOptimizer` |

### Retrieval (Tìm kiếm ngữ cảnh)

| Strategy | Registry Key | Mô tả |
|----------|-------------|--------|
| `vector` | `standard_retriever` | Dense Vector Search qua pgvector (mặc định) |
| `bm25` | `retrieval_bm25` | Sparse Keyword Search qua BM25 |
| `hybrid` | `retrieval_hybrid` | Kết hợp Vector + BM25 qua RRF (Reciprocal Rank Fusion) |

### Post-Retrieval (Hậu xử lý)

| Strategy | Registry Key | Mô tả |
|----------|-------------|--------|
| `rerank` | `post_retrieval_reranker` | FlashRank Reranker để sắp xếp lại kết quả (mặc định) |
| `contextual_compression` | `post_retrieval_contextual_compression` | Nén ngữ cảnh — lọc câu không chứa keyword |

### LLM Provider

| Provider | Registry Key | Mô tả |
|----------|-------------|--------|
| `ollama` | `ollama` | Ollama local (Qwen3:8b) (mặc định) |
| `openai` | `openai` | OpenAI GPT-4o-mini |
| `google` | `google` | Google Gemini 2.5 Flash |
| `vllm` | `vllm` | vLLM (Private GPU) |

## Chunker Options

| Chunker | Registry Key | Mô tả |
|---------|-------------|--------|
| `recursive` | `recursive` | RecursiveCharacterTextSplitter (mặc định) |
| `tiktoken` | `tiktoken` | TokenTextSplitter theo token |
| `semantic` | `semantic` | SemanticChunker theo ngữ nghĩa (dùng Ollama embeddings) |

## Registry Pattern

Hệ thống sử dụng **Registry Pattern** để đăng ký và quản lý tất cả component:

```python
from rag.registry import registry

# Đăng ký component bằng decorator
@registry.loader("unstructured")
class UnstructuredLoader: ...

@registry.chunker("recursive")
class RecursiveChunker: ...

@registry.generator("ollama")
class OllamaGenerator: ...

# Lấy component từ registry
loader_cls = registry.get_loader("unstructured")
loader = loader_cls()

# Liệt kê tất cả component
print(registry.generators)    # {'chain', 'ollama', 'openai', 'google', 'vllm'}
print(registry.retrievers)    # {'pre_retrieval_identity', 'pre_retrieval_query_optimizer', ...}
print(registry.prompts)       # {'basic', 'conversational'}
print(registry.loaders)       # {'unstructured'}
```

8 loại registry:
- `loader` — Đọc tài liệu
- `chunker` — Phân đoạn văn bản
- `embedder` — Tạo embedding
- `vector_store` — Lưu trữ vector
- `retriever` — Truy xuất (pre / post / retrieval)
- `prompt` — Prompt templates
- `generator` — LLM generators

## Chat Memory

Hệ thống hỗ trợ **đa lượt hội thoại** (multi-turn) với `ChatMemory`:

```python
from rag.memory import chat_memory

# Thêm tin nhắn
chat_memory.add_user_message("session_123", "Câu hỏi?")
chat_memory.add_ai_message("session_123", "Câu trả lời!")

# Lấy lịch sử
history = chat_memory.get_messages("session_123")
```

History được tự động quản lý qua `session_id` và truyền vào `ConversationalPrompt`.

## Prompt Templates

| Prompt | Registry Key | Mô tả |
|--------|-------------|--------|
| `BasicPrompt` | `basic` | Single-turn, không có chat history |
| `ConversationalPrompt` | `conversational` | Multi-turn, kèm lịch sử chat (mặc định) |

## Dependencies chính

| Package | Mục đích |
|---------|----------|
| FastAPI | REST API framework |
| uvicorn | ASGI server |
| Ollama | Local LLM + Embeddings |
| pgvector | Vector database cho PostgreSQL |
| LangChain | Framework cho RAG pipeline |
| Streamlit | Frontend UI |
| rank-bm25 | BM25 sparse retrieval |
| flashrank | Learning-to-rank reranker |
| unstructured | Document parsing |
| pydantic | Data validation |
| SQLAlchemy | ORM cho pgvector |
| langgraph | LangGraph orchestration |
## Troubleshooting

### Không kết nối được Ollama
- Đảm bảo `ollama serve` đang chạy
- Kiểm tra `OLLAMA_BASE_URL` trong `.env`

### PostgreSQL lỗi
- Đảm bảo PostgreSQL đang chạy
- Kiểm tra `PGVECTOR_URL` trong `.env`
- Đảm bảo extension `pgvector` đã được cài

### Thiếu `unstructured` module
```bash
pip install unstructured
```

### FlashRank không cài được (Python 3.11)
```bash
pip install flashrank==0.2.10
```

### Pylance errors với `api_key` parameter
- Đã fix bằng `SecretStr(api_key)` trong `openai.py` và `vllm.py`
```bash
pip install pydantic  # SecretStr is from pydantic
```
