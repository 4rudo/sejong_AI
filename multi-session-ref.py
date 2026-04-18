"""멀티세션 RAG 챗봇 — Supabase 세션·벡터 저장 (7.MultiService/prompts/멀티세션 ref.txt)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _REPO_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

_LOG_DIR = _REPO_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"multi_session_rag_{datetime.now().strftime('%Y%m%d')}.log"

_LOGGER = logging.getLogger("multi_session_rag")
_LOGGER.setLevel(logging.WARNING)
_LOGGER.propagate = False
if not _LOGGER.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.WARNING)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    _LOGGER.addHandler(_fh)
    _LOGGER.addHandler(_ch)

for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.ERROR)
    logging.getLogger(_name).propagate = False

st.set_page_config(
    page_title="멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide",
)


def _merge_streamlit_secrets_into_environ() -> None:
    """Streamlit Cloud Secrets(또는 로컬 .streamlit/secrets.toml)를 os.environ에 반영해 os.getenv()와 동작을 맞춘다."""
    try:
        root = st.secrets
    except Exception:
        return

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            if isinstance(val, dict):
                walk(val)
            elif isinstance(val, (list, tuple, type(None))):
                continue
            else:
                os.environ[str(key)] = str(val)

    try:
        data = dict(root)
    except Exception:
        try:
            data = {k: root[k] for k in root}
        except Exception:
            return
    walk(data)


_merge_streamlit_secrets_into_environ()

LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_BATCH = 10
RAG_TOP_K = 8

ANSWER_SYSTEM = (
    "너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. "
    "그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘.\n"
    "참고 문서가 있으면 그 내용을 바탕으로 답하고, 문서와 무관한 질문이면 일반 지식으로 답해줘."
)

TITLE_SYSTEM = (
    "첫 사용자 질문과 첫 어시스턴트 답변을 바탕으로, 이 대화를 대표하는 짧은 세션 제목을 한 줄로 지어라. "
    "20자 이내, 따옴표·불릿·번호 없이 제목만 출력해라."
)

FOLLOWUP_SYSTEM = (
    "방금까지의 질의응답을 바탕으로, 사용자가 이어서 물어볼 만한 질문을 한국어로 정확히 3개만 제시하세요. "
    "각 줄은 '1. ', '2. ', '3. '으로 시작해야 합니다. 다른 설명은 하지 마세요."
)


def _env_ok() -> tuple[str | None, str | None, str | None]:
    oai = (os.getenv("OPENAI_API_KEY") or "").strip()
    url = (os.getenv("SUPABASE_URL") or "").strip()
    anon = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    return oai or None, url or None, anon or None


@st.cache_resource(show_spinner=False)
def _supabase_client(url: str, anon: str) -> Client:
    return create_client(url, anon)


def get_supabase() -> Client | None:
    _, url, anon = _env_ok()
    if not url or not anon:
        return None
    return _supabase_client(url, anon)


def remove_separators(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"~~[^~]*~~", "", text)
    out = re.sub(r"(?m)^\s*(---+|===+|___+)\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip()


def _stream_delta_text(llm: ChatOpenAI, messages: list[BaseMessage]) -> Iterator[str]:
    acc = ""
    prev_clean = ""
    for chunk in llm.stream(messages):
        piece = (
            chunk.content
            if isinstance(chunk.content, str)
            else "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in (chunk.content or [])
            )
        )
        if not piece:
            continue
        acc += piece
        clean = remove_separators(acc)
        delta = clean[len(prev_clean) :]
        if delta:
            yield delta
            prev_clean = clean


def _join_stream(result: list[Any] | str) -> str:
    if isinstance(result, list):
        return "".join(str(x) for x in result)
    return str(result or "")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_query(embeddings: OpenAIEmbeddings, text: str) -> list[float]:
    v = embeddings.embed_query(text)
    return [float(x) for x in v]


def _embed_batch(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in row] for row in embeddings.embed_documents(texts)]


def fetch_sessions(sb: Client) -> list[dict[str, Any]]:
    res = (
        sb.table("chat_sessions")
        .select("id,title,created_at,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return list(res.data or [])


def insert_session_row(sb: Client, title: str) -> str:
    sid = str(uuid.uuid4())
    sb.table("chat_sessions").insert({"id": sid, "title": title}).execute()
    return sid


def replace_messages(sb: Client, session_id: str, history: list[dict[str, str]]) -> None:
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows: list[dict[str, Any]] = []
    for i, turn in enumerate(history):
        rows.append(
            {
                "session_id": session_id,
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
                "sort_order": i,
            }
        )
    if rows:
        sb.table("chat_messages").insert(rows).execute()


def update_session_title(sb: Client, session_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def load_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    res = (
        sb.table("chat_messages")
        .select("role,content,sort_order")
        .eq("session_id", session_id)
        .order("sort_order")
        .execute()
    )
    rows = list(res.data or [])
    out: list[dict[str, str]] = []
    for r in rows:
        out.append({"role": r.get("role", "user"), "content": r.get("content", "")})
    return out


def delete_session(sb: Client, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).execute()


def copy_vectors_to_session(sb: Client, from_sid: str, to_sid: str) -> None:
    res = sb.table("vector_documents").select("*").eq("session_id", from_sid).execute()
    rows = list(res.data or [])
    if not rows:
        return
    batch: list[dict[str, Any]] = []
    for r in rows:
        emb = r.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue
        batch.append(
            {
                "session_id": to_sid,
                "content": r.get("content", ""),
                "metadata": r.get("metadata") or {},
                "embedding": emb,
                "file_name": r.get("file_name") or "unknown.pdf",
                "chunk_index": int(r.get("chunk_index") or 0),
            }
        )
        if len(batch) >= VECTOR_BATCH:
            sb.table("vector_documents").insert(batch).execute()
            batch = []
    if batch:
        sb.table("vector_documents").insert(batch).execute()


def ingest_pdfs_to_supabase(
    sb: Client,
    session_id: str,
    uploads: list[Any],
    embeddings: OpenAIEmbeddings,
) -> list[str]:
    names: list[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    sb.table("vector_documents").delete().eq("session_id", session_id).execute()

    for f in uploads:
        fname = getattr(f, "name", "unknown.pdf") or "unknown.pdf"
        names.append(fname)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
        finally:
            os.unlink(path)

        splits = splitter.split_documents(docs)
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for i, d in enumerate(splits):
            meta = dict(d.metadata or {})
            meta["file_name"] = fname
            meta["chunk_index"] = i
            texts.append(d.page_content)
            metas.append(meta)

        for i in range(0, len(texts), VECTOR_BATCH):
            batch_t = texts[i : i + VECTOR_BATCH]
            batch_m = metas[i : i + VECTOR_BATCH]
            embs = _embed_batch(embeddings, batch_t)
            rows = []
            for j, (t, m, e) in enumerate(zip(batch_t, batch_m, embs, strict=False)):
                rows.append(
                    {
                        "session_id": session_id,
                        "content": t,
                        "metadata": m,
                        "embedding": e,
                        "file_name": fname,
                        "chunk_index": int(m.get("chunk_index", i + j)),
                    }
                )
            sb.table("vector_documents").insert(rows).execute()

    return names


def retrieve_by_rpc(
    sb: Client,
    session_id: str,
    query_embedding: list[float],
    k: int,
) -> list[Document]:
    try:
        res = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in res.data or []:
            docs.append(
                Document(
                    page_content=row.get("content") or "",
                    metadata={
                        "file_name": row.get("file_name"),
                        "metadata": row.get("metadata"),
                        "similarity": row.get("similarity"),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("RPC 검색 실패, 로컬 유사도로 대체: %s", exc)
        return []


def retrieve_fallback(
    sb: Client,
    session_id: str,
    query_embedding: list[float],
    k: int,
) -> list[Document]:
    res = sb.table("vector_documents").select("content,metadata,file_name,embedding").eq("session_id", session_id).execute()
    rows = list(res.data or [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        emb = r.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue
        if not isinstance(emb, list):
            continue
        sim = _cosine_sim(query_embedding, [float(x) for x in emb])
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[Document] = []
    for sim, r in scored[:k]:
        out.append(
            Document(
                page_content=r.get("content") or "",
                metadata={"file_name": r.get("file_name"), "similarity": sim, "metadata": r.get("metadata")},
            )
        )
    return out


def generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    msgs = [
        SystemMessage(content=TITLE_SYSTEM),
        HumanMessage(content=f"첫 질문:\n{first_q}\n\n첫 답변:\n{first_a}\n"),
    ]
    try:
        out = llm.invoke(msgs)
        raw = out.content if isinstance(out.content, str) else str(out.content)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("제목 생성 실패: %s", exc)
        return "새 대화"
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r'^["\']|["\']$', "", line).strip()
    return line[:80] if line else "새 대화"


def generate_followup_block(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    msgs = [
        SystemMessage(content=FOLLOWUP_SYSTEM),
        HumanMessage(content=f"사용자 질문:\n{user_q}\n\n어시스턴트 답변:\n{answer}\n"),
    ]
    try:
        out = llm.invoke(msgs)
        raw = out.content if isinstance(out.content, str) else str(out.content)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("다음 질문 생성 실패: %s", exc)
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    picked: list[str] = []
    for ln in lines:
        if re.match(r"^[123]\.\s", ln):
            picked.append(ln)
        if len(picked) >= 3:
            break
    if len(picked) < 3:
        return ""
    return "\n\n### 💡 다음에 물어볼 수 있는 질문들\n" + "\n".join(picked[:3])


def auto_save_session(
    sb: Client,
    session_id: str,
    history: list[dict[str, str]],
    llm: ChatOpenAI,
) -> str | None:
    """메시지를 DB에 반영하고, 첫 질문·첫 답이 모두 있을 때만 제목을 LLM으로 갱신합니다."""
    replace_messages(sb, session_id, history)
    if (
        len(history) == 2
        and history[0].get("role") == "user"
        and history[1].get("role") == "assistant"
    ):
        u0 = history[0].get("content", "")
        a0 = history[1].get("content", "")
        if u0 and a0:
            title = generate_session_title(llm, u0, a0)
            update_session_title(sb, session_id, title)
            return title
    return None


def _inject_styles() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div[data-testid="stChatMessage"] { border-radius: 12px; padding: 0.5rem 0.75rem; margin-bottom: 0.35rem; }
.stButton > button {
  background-color: #ff69b4 !important;
  color: #ffffff !important;
  border: none !important;
}
.stButton > button:hover { filter: brightness(0.95); }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo_path = _REPO_ROOT / "logo.png"
    cols = st.columns([1, 3, 1])
    with cols[0]:
        if logo_path.is_file():
            st.image(str(logo_path), width=180)
        else:
            st.markdown('<p style="font-size:4rem;margin:0;">📚</p>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            """
<div style="text-align:center;width:100%;">
  <div style="font-size:4rem !important;line-height:1.1;font-weight:700;">
    <span style="color:#1f77b4 !important;">멀티세션</span>
    <span style="color:#ffd700 !important;">RAG 챗봇</span>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.empty()


def _init_state() -> None:
    if "ms_chat_history" not in st.session_state:
        st.session_state.ms_chat_history = []
    if "ms_session_id" not in st.session_state:
        st.session_state.ms_session_id = None
    if "ms_processed_files" not in st.session_state:
        st.session_state.ms_processed_files = []
    if "ms_rag_enabled" not in st.session_state:
        st.session_state.ms_rag_enabled = "RAG 사용"


def _label_for_session(row: dict[str, Any]) -> str:
    title = (row.get("title") or "제목 없음").strip()
    ts = row.get("updated_at") or row.get("created_at") or ""
    return f"{title}  |  {ts}"


def load_session_into_ui(sb: Client, session_id: str) -> None:
    st.session_state.ms_session_id = session_id
    st.session_state.ms_chat_history = load_messages(sb, session_id)
    res = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    rows = list(res.data or [])
    names = sorted({str(r.get("file_name")) for r in rows if r.get("file_name")})
    st.session_state.ms_processed_files = names


def main() -> None:
    _init_state()
    _inject_styles()
    _render_header()

    oai_key, sup_url, sup_anon = _env_ok()
    sb = get_supabase()

    if not oai_key:
        st.error(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "로컬: 프로젝트 루트 `.env` / 배포: Streamlit Cloud **Secrets**에 동일 이름으로 추가해 주세요."
        )
    if not sup_url or not sup_anon:
        st.error(
            "SUPABASE_URL 또는 SUPABASE_ANON_KEY가 설정되지 않았습니다. "
            "로컬 `.env` 또는 Cloud **Secrets**를 확인해 주세요."
        )
    if not sb and (sup_url and sup_anon):
        st.error("Supabase 클라이언트를 만들 수 없습니다. URL·키를 확인해 주세요.")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, api_key=oai_key) if oai_key else None
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMS, api_key=oai_key) if oai_key else None

    if sb and st.session_state.ms_session_id is None:
        sid = insert_session_row(sb, "새 대화")
        st.session_state.ms_session_id = sid

    sessions: list[dict[str, Any]] = fetch_sessions(sb) if sb else []
    session_ids: list[str] = [str(r.get("id")) for r in sessions if r.get("id")]
    id_label: dict[str, str] = {str(r.get("id")): _label_for_session(r) for r in sessions if r.get("id")}

    def _on_session_pick() -> None:
        client = get_supabase()
        sid = st.session_state.get("ms_sel_sid")
        if client and sid:
            load_session_into_ui(client, str(sid))

    with st.sidebar:
        st.markdown("### 세션 관리")
        st.caption(f"LLM: **{LLM_MODEL}** (고정)")
        rag_choice = st.radio(
            "RAG (PDF 검색)",
            ("사용 안 함", "RAG 사용"),
            index=1 if st.session_state.ms_rag_enabled == "RAG 사용" else 0,
            key="ms_rag_radio",
        )
        st.session_state.ms_rag_enabled = rag_choice

        if session_ids and st.session_state.ms_session_id in session_ids:
            st.session_state.ms_sel_sid = st.session_state.ms_session_id

        if session_ids:
            st.selectbox(
                "저장된 세션 선택 (선택 시 자동 로드)",
                options=session_ids,
                format_func=lambda sid: id_label.get(str(sid), str(sid)),
                key="ms_sel_sid",
                on_change=_on_session_pick,
            )

        if st.button("세션로드", disabled=not sb or not session_ids):
            pick = st.session_state.get("ms_sel_sid")
            if sb and pick:
                load_session_into_ui(sb, str(pick))
                st.success("세션을 불러왔습니다.")
                st.rerun()

        if st.button("세션저장", disabled=not sb or not llm):
            if not sb or not llm:
                st.warning("Supabase·OpenAI 설정을 확인해 주세요.")
            else:
                hist = list(st.session_state.ms_chat_history)
                if len(hist) < 2:
                    st.warning("저장할 대화(첫 질문·답)가 없습니다.")
                else:
                    u0 = ""
                    a0 = ""
                    for t in hist:
                        if t.get("role") == "user" and not u0:
                            u0 = t.get("content", "")
                        elif t.get("role") == "assistant" and u0 and not a0:
                            a0 = t.get("content", "")
                            break
                    if not u0 or not a0:
                        st.warning("첫 질문과 첫 답변이 있어야 세션을 저장할 수 있습니다.")
                    else:
                        title = generate_session_title(llm, u0, a0)
                        new_id = str(uuid.uuid4())
                        sb.table("chat_sessions").insert({"id": new_id, "title": title}).execute()
                        replace_messages(sb, new_id, hist)
                        cur_sid = st.session_state.ms_session_id or ""
                        if cur_sid:
                            copy_vectors_to_session(sb, cur_sid, new_id)
                        st.success(f"새 세션이 저장되었습니다: {title}")
                        st.rerun()

        if st.button("세션삭제", disabled=not sb):
            sid = st.session_state.ms_session_id
            if sb and sid:
                delete_session(sb, sid)
                st.session_state.ms_chat_history = []
                st.session_state.ms_processed_files = []
                st.session_state.ms_session_id = insert_session_row(sb, "새 대화")
                st.success("선택된 세션이 삭제되었고 새 세션이 시작되었습니다.")
                st.rerun()

        if st.button("화면초기화"):
            st.session_state.ms_chat_history = []
            st.session_state.ms_processed_files = []
            if sb:
                st.session_state.ms_session_id = insert_session_row(sb, "새 대화")
            st.rerun()

        if st.button("vectordb"):
            sid = st.session_state.ms_session_id
            if not sb or not sid:
                st.info("Supabase 세션이 없습니다.")
            else:
                res = (
                    sb.table("vector_documents")
                    .select("file_name")
                    .eq("session_id", sid)
                    .execute()
                )
                rows = list(res.data or [])
                names = sorted({str(r.get("file_name")) for r in rows if r.get("file_name")})
                if not names:
                    st.info("현재 세션에 인덱싱된 파일이 없습니다.")
                else:
                    st.markdown("**현재 세션 vectordb 파일명**")
                    for n in names:
                        st.text(f"- {n}")

        st.markdown("---")
        uploads = st.file_uploader("PDF 업로드", type=["pdf"], accept_multiple_files=True)
        if st.button("파일 처리하기", disabled=not sb or not embeddings):
            if not sb or not embeddings or not oai_key:
                st.error("OPENAI·Supabase 설정과 임베딩에 필요한 키를 확인해 주세요.")
            elif not uploads:
                st.warning("PDF를 선택해 주세요.")
            else:
                sid = st.session_state.ms_session_id
                if not sid:
                    st.error("세션 ID가 없습니다.")
                else:
                    try:
                        names = ingest_pdfs_to_supabase(sb, sid, list(uploads), embeddings)
                        st.session_state.ms_processed_files = names
                        if llm:
                            auto_save_session(sb, sid, st.session_state.ms_chat_history, llm)
                        st.success(f"처리 완료: {', '.join(names)}")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.error("PDF 처리 오류: %s", exc, exc_info=True)
                        st.error("PDF 처리 중 오류가 발생했습니다.")

        if st.session_state.ms_processed_files:
            st.caption("처리된 파일")
            for name in st.session_state.ms_processed_files:
                st.text(f"- {name}")

        st.text(
            "현재 상태\n"
            f"- 세션 ID: {st.session_state.ms_session_id or '(없음)'}\n"
            f"- RAG: {st.session_state.ms_rag_enabled}\n"
            f"- 처리된 파일 수: {len(st.session_state.ms_processed_files)}\n"
            f"- 메시지 수: {len(st.session_state.ms_chat_history)}"
        )

    for turn in st.session_state.ms_chat_history:
        role = turn.get("role", "user")
        with st.chat_message(role):
            st.markdown(turn.get("content", ""))

    user_msg = st.chat_input("질문을 입력하세요")
    if not user_msg:
        return
    if not oai_key or not llm or not embeddings:
        st.error("OPENAI_API_KEY가 필요합니다.")
        return
    if st.session_state.ms_rag_enabled == "RAG 사용" and not st.session_state.ms_processed_files:
        st.warning("RAG를 쓰려면 PDF를 업로드한 뒤 파일 처리하기를 눌러 주세요.")
        return

    sid = st.session_state.ms_session_id
    if not sid or not sb:
        st.error("세션을 초기화할 수 없습니다. Supabase 설정을 확인해 주세요.")
        return

    st.session_state.ms_chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    assistant_core = ""
    try:
        with st.chat_message("assistant"):
            if st.session_state.ms_rag_enabled == "RAG 사용":
                qemb = _embed_query(embeddings, user_msg)
                docs = retrieve_by_rpc(sb, sid, qemb, RAG_TOP_K)
                if not docs:
                    docs = retrieve_fallback(sb, sid, qemb, RAG_TOP_K)
                context = "\n\n".join(d.page_content for d in docs)
                mem_parts: list[str] = []
                for t in st.session_state.ms_chat_history[:-1][-10:]:
                    tag = "사용자" if t.get("role") == "user" else "어시스턴트"
                    mem_parts.append(f"{tag}: {t.get('content', '')}")
                memory_snip = "\n".join(mem_parts) if mem_parts else "(이전 맥락 없음)"
                human = (
                    f"이전 대화:\n{memory_snip}\n\n참고 문서:\n{context}\n\n질문:\n{user_msg}"
                )
                messages = [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=human)]
                assistant_core = _join_stream(st.write_stream(_stream_delta_text(llm, messages)))
            else:
                messages = [SystemMessage(content=ANSWER_SYSTEM)]
                for t in st.session_state.ms_chat_history[:-1]:
                    if t.get("role") == "user":
                        messages.append(HumanMessage(content=t.get("content", "")))
                    else:
                        messages.append(AIMessage(content=t.get("content", "")))
                messages.append(HumanMessage(content=user_msg))
                assistant_core = _join_stream(st.write_stream(_stream_delta_text(llm, messages)))

            follow = generate_followup_block(llm, user_msg, assistant_core)
            if follow:
                st.markdown(follow)
                assistant_full = assistant_core + follow
            else:
                assistant_full = assistant_core

    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("응답 생성 오류: %s", exc, exc_info=True)
        st.error("응답을 만드는 중 문제가 발생했습니다.")
        return

    st.session_state.ms_chat_history.append({"role": "assistant", "content": assistant_full})

    if sb and llm:
        auto_save_session(sb, sid, st.session_state.ms_chat_history, llm)


if __name__ == "__main__":
    main()
