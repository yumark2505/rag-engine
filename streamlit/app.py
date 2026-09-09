import os
import time
import uuid
import requests
import streamlit as st

st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("RAG Strategy Sandbox")

API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_auth_headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}

# ── LẤY DANH SÁCH TỆP TRONG DB VÀ THƯ MỤC ──────────────────────────────
indexed_docs = []
try:
    doc_resp = requests.get(f"{API_BASE_URL}/documents/list", headers=get_auth_headers(), timeout=4)
    if doc_resp.status_code == 200:
        indexed_docs = doc_resp.json().get("documents", [])
except Exception:
    indexed_docs = []

folder_files = []
try:
    folder_resp = requests.get(f"{API_BASE_URL}/documents/folder", headers=get_auth_headers(), timeout=4)
    if folder_resp.status_code == 200:
        folder_files = folder_resp.json().get("files", [])
except Exception:
    folder_files = []

# ── THANH ĐIỀU KHIỂN SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")

    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.button("Làm mới đoạn hội thoại", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("1. Nạp Tài Liệu Nền (Ingestion)")

    # Hiển thị danh sách file sẵn có trong folder
    with st.expander(f"📁 Tệp có sẵn trong thư mục ({len(folder_files)})", expanded=True):
        if folder_files:
            for idx, fname in enumerate(folder_files, 1):
                is_indexed = fname in indexed_docs
                status_text = "Đã nạp" if is_indexed else "Chưa index"
                st.caption(f"**{idx}.** `{fname}` — *{status_text}*")
        else:
            st.caption("Thư mục `data/Document/` đang trống.")

    ingest_chunker = st.selectbox(
        "Thuật toán Chunker",
        options=["recursive", "tiktoken", "semantic"],
        format_func=lambda x: {
            "recursive": "Recursive Splitter (Theo ký tự/đoạn)",
            "tiktoken": "Tiktoken Splitter (Theo token)",
            "semantic": "Dynamic Semantic Splitter (Ngữ nghĩa)",
        }[x],
        index=0,
    )

    ingest_embedder = st.selectbox(
        "Mô hình Embedding",
        options=["ollama", "openai", "google"],
        format_func=lambda x: {
            "ollama": "Ollama (nomic-embed-text)",
            "openai": "OpenAI (text-embedding-3-small)",
            "google": "Google AI (text-embedding-004)",
        }[x],
        index=0,
    )

    # Re-index toàn bộ kho dữ liệu
    if st.button("Áp dụng & Re-index kho dữ liệu", use_container_width=True):
        with st.spinner("Đang gửi yêu cầu re-index tới máy chủ..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/documents/reindex",
                    params={"chunker_name": ingest_chunker, "embedder_name": ingest_embedder},
                    headers=get_auth_headers(),
                )
                if res.status_code in (200, 202):
                    task_id = res.json().get("task_id")
                    status_placeholder = st.empty()
                    status_placeholder.info(
                        f"Đang băm lại ({ingest_chunker}) và vector hóa ({ingest_embedder})..."
                    )

                    is_completed = False
                    for _ in range(60):
                        time.sleep(2)
                        st_res = requests.get(
                            f"{API_BASE_URL}/documents/status/{task_id}",
                            headers=get_auth_headers(),
                        ).json()
                        curr_status = st_res.get("status")

                        if curr_status == "COMPLETED":
                            status_placeholder.success("Đã hoàn tất Re-index vào PGVector!")
                            is_completed = True
                            time.sleep(1)
                            st.rerun()
                            break
                        elif curr_status and "FAILED" in curr_status:
                            status_placeholder.error(f"Xử lý thất bại: {curr_status}")
                            is_completed = True
                            break

                    if not is_completed:
                        status_placeholder.warning("Quá trình re-index vẫn đang tiếp tục ngầm...")
                else:
                    st.error(f"Lỗi: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Lỗi kết nối tới server: {e}")

    # Tải tệp PDF lên server
    uploaded_file = st.file_uploader("Hoặc tải lên tệp PDF mới", type=["pdf"])
    if uploaded_file and st.button("Tải lên & Xử lý nền", use_container_width=True):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }
        form_data = {
            "chunker_name": ingest_chunker,
            "embedder_name": ingest_embedder,
        }

        with st.spinner("Đang gửi tệp và cấu hình lên server..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files=files,
                    data=form_data,
                    headers=get_auth_headers(),
                )
                if res.status_code in (200, 202):
                    task_id = res.json().get("task_id")
                    status_placeholder = st.empty()
                    status_placeholder.info(
                        f"Đang xử lý nền (Chunker: {ingest_chunker}, Embedder: {ingest_embedder})..."
                    )

                    is_completed = False
                    for _ in range(60):
                        time.sleep(2)
                        status_res = requests.get(
                            f"{API_BASE_URL}/documents/status/{task_id}",
                            headers=get_auth_headers(),
                        ).json()
                        curr_status = status_res.get("status")

                        if curr_status == "COMPLETED":
                            status_placeholder.success("Đã hoàn tất nạp dữ liệu vào Vector Store!")
                            is_completed = True
                            time.sleep(1)
                            st.rerun()
                            break
                        elif curr_status and "FAILED" in curr_status:
                            status_placeholder.error(f"Xử lý thất bại: {curr_status}")
                            is_completed = True
                            break

                    if not is_completed:
                        status_placeholder.warning("Quá trình xử lý vẫn đang tiếp tục ngầm...")
                else:
                    st.error(f"Lỗi tải tệp: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến server: {e}")

    # ── QUẢN LÝ & LỌC TÀI LIỆU TRUY VẤN ──────────────────────────────
    st.markdown("---")
    st.subheader("2. Phạm Vi Tìm Kiếm Tài Liệu")

    available_docs = ["Tất cả"] + indexed_docs
    selected_doc = st.selectbox(
        "Chọn tài liệu tra cứu",
        options=available_docs,
        index=0,
        help="Chọn 'Tất cả' để tìm kiếm toàn bộ kho, hoặc chọn chính xác 1 tệp để cô lập thông tin.",
    )

    if selected_doc != "Tất cả":
        if st.button(f"Xóa tệp '{selected_doc}'", use_container_width=True):
            with st.spinner("Đang xóa tài liệu khỏi database..."):
                try:
                    del_res = requests.delete(
                        f"{API_BASE_URL}/documents/{selected_doc}",
                        headers=get_auth_headers(),
                    )
                    if del_res.status_code == 200:
                        st.success(f"Đã xóa hoàn toàn tệp '{selected_doc}'!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Không thể xóa: {del_res.text}")
                except Exception as e:
                    st.error(f"Lỗi kết nối khi xóa: {e}")

    st.markdown("---")
    st.subheader("3. Chiến Lược Truy Vấn (Serving)")

    llm_choice = st.selectbox(
        "Mô hình LLM",
        options=["ollama", "openai", "google", "vllm"],
        format_func=lambda x: {
            "ollama": "Ollama (Qwen local)",
            "openai": "OpenAI (GPT-4o mini)",
            "google": "Google AI (Gemini 2.5 Flash)",
            "vllm": "vLLM (Private GPU)",
        }[x],
        index=0,
    )

    pre_choice = st.selectbox(
        "Pre-retrieval",
        options=["identity", "query_transform", "hyde"],
        format_func=lambda x: {
            "identity": "Identity (Nguyên bản - Tối ưu Latency)",
            "query_transform": "Query Transform (Decompose câu hỏi)",
            "hyde": "HyDE (Văn bản giả định)",
        }[x],
    )

    retrieval_choice = st.selectbox(
        "Retrieval Engine",
        options=["vector", "bm25", "hybrid"],
        format_func=lambda x: {
            "vector": "Dense Vector (pgvector)",
            "bm25": "Sparse Keyword (BM25)",
            "hybrid": "Hybrid Search (Vector + BM25)",
        }[x],
    )

    post_choice = st.selectbox(
        "Post-retrieval",
        options=["rerank", "contextual_compression"],
        format_func=lambda x: {
            "rerank": "FlashRank Reranker",
            "contextual_compression": "Contextual Compression",
        }[x],
    )

    st.markdown("---")
    st.subheader("4. Tham Số Retrieval & Output")
    top_k = st.slider("Top K (Số chunk tìm kiếm)", min_value=1, max_value=15, value=8)
    top_n = st.slider("Top N (Sau rerank/nén)", min_value=1, max_value=10, value=3)
    use_streaming = st.toggle("Bật chế độ Streaming", value=False)

# ── KHUNG HIỂN THỊ TIN NHẮN ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Trích dẫn tài liệu tham khảo"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    score = meta.get("rerank_score", "N/A")
                    filename = meta.get("filename", meta.get("source", "Tài liệu"))
                    st.markdown(
                        f"**{idx}. {filename}** (Score: `{score}`):\n```text\n{src.get('content', '')}\n```"
                    )

# ── GỬI TRUY VẤN VÀ HIỂN THỊ KẾT QUẢ ────────────────────────────────
if query := st.chat_input("Nhập câu hỏi nghiên cứu..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "selected_document": selected_doc if selected_doc != "Tất cả" else None,
        "llm_provider": llm_choice,
        "top_k": top_k,
        "top_n": top_n,
        "pre_retrieval_strategy": pre_choice,
        "retrieval_strategy": retrieval_choice,
        "post_retrieval_strategy": post_choice,
    }

    with st.chat_message("assistant"):
        if use_streaming:
            message_placeholder = st.empty()
            full_response = ""
            try:
                with requests.post(
                    f"{API_BASE_URL}/rag/stream",
                    json=payload,
                    headers=get_auth_headers(),
                    stream=True,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            text_chunk = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                            full_response += text_chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as e:
                st.error(f"Lỗi kết nối streaming: {e}")
        else:
            with st.spinner("Đang truy xuất ngữ cảnh và suy luận..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/rag/query",
                        json=payload,
                        headers=get_auth_headers(),
                        timeout=90,
                    )
                    res.raise_for_status()
                    data = res.json()

                    answer = data.get("answer", "")
                    sub_queries = data.get("sub_queries", [])
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if len(sub_queries) > 1:
                        with st.expander("Các truy vấn con đã mở rộng"):
                            for sq in sub_queries:
                                st.write(f"- {sq}")

                    if sources:
                        with st.expander("Trích dẫn tài liệu tham khảo"):
                            for idx, src in enumerate(sources, 1):
                                meta = src.get("metadata", {})
                                score = meta.get("rerank_score", "N/A")
                                filename = meta.get("filename", meta.get("source", "Tài liệu"))
                                st.markdown(
                                    f"**{idx}. {filename}** (Score: `{score}`):\n```text\n{src.get('content', '')}\n```"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"Không thể kết nối tới FastAPI RAG: {e}")