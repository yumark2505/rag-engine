import traceback
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from schemas.RagSchema import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=RagQueryResponse, status_code=status.HTTP_200_OK)
def query_rag(request_body: RagQueryRequest, request: Request):
    """Truy vấn RAG trả về kết quả JSON hoàn chỉnh kèm trích dẫn nguồn."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Pipeline chưa sẵn sàng hoặc chưa được khởi tạo.",
        )

    try:
        result = pipeline.run(
            query=request_body.query,
            session_id=request_body.session_id,
            selected_document=request_body.selected_document,
            llm_provider=request_body.llm_provider,
            top_k=request_body.top_k,
            top_n=request_body.top_n,
            pre_retrieval_strategy=request_body.pre_retrieval_strategy,
            retrieval_strategy=request_body.retrieval_strategy,
            post_retrieval_strategy=request_body.post_retrieval_strategy,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi thực thi RAG: {str(e)}",
        )


@router.post("/stream")
async def rag_stream_endpoint(request: Request, body: RagQueryRequest):
    """Truy vấn RAG chế độ Streaming từng token phục vụ giao diện Streamlit/Client."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Pipeline chưa sẵn sàng hoặc chưa được khởi tạo.",
        )

    def event_generator():
        try:
            for token in pipeline.stream_answer(
                query=body.query,
                session_id=body.session_id,
                selected_document=body.selected_document,
                llm_provider=body.llm_provider,
                top_k=body.top_k,
                top_n=body.top_n,
                pre_retrieval_strategy=body.pre_retrieval_strategy,
                retrieval_strategy=body.retrieval_strategy,
                post_retrieval_strategy=body.post_retrieval_strategy,
            ):
                yield token
        except Exception as e:
            traceback.print_exc()
            yield f"\n[STREAM ERROR]: {str(e)}"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/query/stream")
def query_rag_stream(request_body: RagQueryRequest, request: Request):
    """Truy vấn RAG dạng Server-Sent Events (SSE) chuẩn `data: <token>`."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Pipeline chưa sẵn sàng.",
        )

    def event_stream():
        try:
            for token in pipeline.stream_answer(
                query=request_body.query,
                session_id=request_body.session_id,
                selected_document=request_body.selected_document,
                llm_provider=request_body.llm_provider,
                top_k=request_body.top_k,
                top_n=request_body.top_n,
                pre_retrieval_strategy=request_body.pre_retrieval_strategy,
                retrieval_strategy=request_body.retrieval_strategy,
                post_retrieval_strategy=request_body.post_retrieval_strategy,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: [ERROR]: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")