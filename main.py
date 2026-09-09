
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag import registry, RAGIngestionPipeline, RAGServingPipeline
from config import settings
from routers.RagApi import router as rag_router
from routers.document import router as document_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--> [LIFESPAN] Khởi tạo tài nguyên RAG...")

    embedder_cls = registry.get_embedder("ollama")
    store_cls = registry.get_vector_store("pgvector")

    if embedder_cls and store_cls:
        # ── Phase 1: Offline Ingestion ──
        ingestion = RAGIngestionPipeline()
        ingest_result = ingestion.ingest()
        print(f"--> [INGESTION] {ingest_result}")

        # ── Phase 2: Online Serving (reuse same vector_store) ──
        app.state.rag_pipeline = RAGServingPipeline(
            vector_store=ingestion.vector_store,
            top_k=settings.TOP_K,
            top_n=3,
        )
        print("--> [LIFESPAN] RAG Serving Pipeline đã sẵn sàng!")

    yield
    print("--> [LIFESPAN] Tắt server...")


app = FastAPI(
    title="FastAPI RAG Service",
    description="API RAG phục vụ suy luận với Ollama và PGVector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(document_router)

@app.get("/")
def root():
    return {"message": "FastAPI RAG Service is running!"}
