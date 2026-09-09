import shutil
import uuid
from pathlib import Path
from typing import Dict, Literal
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import create_engine, text
from config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])

task_status: Dict[str, str] = {}
UPLOAD_DIR = Path("./data/Document")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_document_task(
    target_path: Path,
    task_id: str,
    chunker_name: str,
    embedder_name: str,
):
    """Hàm chạy ngầm: phân đoạn theo chunker và vector hóa theo embedder đã chọn."""
    try:
        task_status[task_id] = "PROCESSING"
        from rag.pipeline import RAGIngestionPipeline

        ingestion_pipeline = RAGIngestionPipeline(
            data_dir=str(UPLOAD_DIR),
            chunker_name=chunker_name,
            embedder_name=embedder_name,
        )
        ingestion_pipeline.ingest(overwrite=False)
        task_status[task_id] = "COMPLETED"
    except Exception as e:
        task_status[task_id] = f"FAILED: {str(e)}"


@router.get("/folder")
async def list_folder_documents():
    """Liệt kê danh sách các tệp PDF thực tế đang nằm trong thư mục data/Document/."""
    try:
        pdf_files = [f.name for f in UPLOAD_DIR.glob("*.pdf")]
        return {"files": pdf_files}
    except Exception as e:
        return {"files": [], "error": str(e)}


@router.get("/list")
async def list_documents():
    """Liệt kê danh sách tên tệp đã nạp vào PostgreSQL pgvector."""
    engine = create_engine(settings.PGVECTOR_URL)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT DISTINCT cmetadata->>'filename' FROM langchain_pg_embedding;")
            ).fetchall()
            files = [r[0] for r in result if r[0]]
        return {"documents": files}
    except Exception as e:
        return {"documents": [], "error": str(e)}


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Xóa toàn bộ chunks của một tệp cụ thể khỏi cơ sở dữ liệu và thư mục."""
    engine = create_engine(settings.PGVECTOR_URL)
    try:
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'filename' = :filename;"),
                {"filename": filename},
            )
            conn.commit()

        # Dọn dẹp tệp vật lý nếu tồn tại
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()

        return {
            "status": "success",
            "message": f"Đã xóa hoàn toàn dữ liệu và tệp '{filename}'.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa tài liệu: {str(e)}",
        )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    chunker_name: Literal["recursive", "tiktoken", "semantic"] = Form("recursive"),
    embedder_name: Literal["ollama", "openai", "google", "huggingface"] = Form("ollama"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hệ thống hiện chỉ chấp nhận tệp PDF.",
        )

    task_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / file.filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        process_document_task,
        saved_path,
        task_id,
        chunker_name,
        embedder_name,
    )
    task_status[task_id] = "PENDING"

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "PENDING",
        "message": f"Tệp đã tiếp nhận. Phân đoạn: '{chunker_name}', Embedding: '{embedder_name}'.",
    }


@router.post("/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_documents(
    background_tasks: BackgroundTasks,
    chunker_name: Literal["recursive", "tiktoken", "semantic"] = Query("recursive"),
    embedder_name: Literal["ollama", "openai", "google", "huggingface"] = Query("ollama"),
):
    """Quét lại toàn bộ tệp trong data/Document và index lại theo chiến lược mới."""
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kho dữ liệu './data/Document' không có tệp PDF nào để nạp lại.",
        )

    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        process_document_task,
        UPLOAD_DIR,
        task_id,
        chunker_name,
        embedder_name,
    )
    task_status[task_id] = "PENDING"

    return {
        "task_id": task_id,
        "total_files": len(pdf_files),
        "status": "PENDING",
        "message": f"Bắt đầu re-index {len(pdf_files)} tệp với Chunker: '{chunker_name}', Embedder: '{embedder_name}'.",
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    curr_status = task_status.get(task_id)
    if not curr_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy mã tác vụ.",
        )
    return {"task_id": task_id, "status": curr_status}