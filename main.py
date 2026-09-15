
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag import registry, Pipeline
from config import settings
from routers.RagApi import router as rag_router
from routers.document import router as document_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--> [LIFESPAN] Khởi tạo tài nguyên RAG...")

    embedder_cls = registry.get_embedder("ollama")
    store_cls = registry.get_vector_store("pgvector")

    if embedder_cls and store_cls:
        app.state.rag_pipeline = Pipeline()
        ingest_result = app.state.rag_pipeline.ingest()
        print(f"--> [INGESTION] {ingest_result}")
        print("--> [LIFESPAN] RAG Pipeline đã sẵn sàng!")

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
