from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4
import hashlib
import json
import os
import re
import sqlite3
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from fastembed import TextEmbedding


load_dotenv()

app = FastAPI(title="SourceMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CONVERSATIONS_DB = DATA_DIR / "conversations.db"
COLLECTION_NAME = "sourcemind_knowledge"

qdrant_client = QdrantClient(url=QDRANT_URL)
embedding_model = TextEmbedding()
VECTOR_SIZE = 384


def make_openai_client(api_key: Optional[str]) -> OpenAI:
    """Build an OpenAI client from a per-request key, falling back to env for dev."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "OpenAI API key required. Open the OpenAI Settings panel in the UI "
                "and paste your key, or set OPENAI_API_KEY in the backend environment."
            ),
        )
    return OpenAI(api_key=key)


def init_conversations_db():
    with sqlite3.connect(CONVERSATIONS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                question TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                feedback TEXT,
                correction TEXT
            )
            """
        )
        conn.commit()


def log_conversation(
    conversation_id: str,
    question: str,
    provider: str,
    model: str,
    citations: list,
    answer: str,
):
    """Persist a completed RAG turn so it can later be exported as training data."""
    try:
        with sqlite3.connect(CONVERSATIONS_DB) as conn:
            conn.execute(
                """
                INSERT INTO conversations
                    (id, ts, question, provider, model, citations_json, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    provider,
                    model,
                    json.dumps(citations, default=str),
                    answer,
                ),
            )
            conn.commit()
    except Exception:
        # Logging is best-effort; never let it break the user-facing response.
        pass


def ensure_collection():
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


ensure_collection()
init_conversations_db()


def file_hash(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def duplicate_filter(sha256: str):
    return Filter(
        must=[
            FieldCondition(
                key="file_sha256",
                match=MatchValue(value=sha256),
            )
        ]
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\d+\n", "\n", text)
    return text.strip()


def retrieve_sources(question: str, limit: int = 5):
    query_vector = list(embedding_model.embed([question]))[0]
    query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector

    return qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    ).points


def build_source_context(results):
    source_blocks = []
    citations = []

    for i, result in enumerate(results):
        payload = result.payload or {}

        source_blocks.append(
            f"Source {i + 1}:\n"
            f"File: {payload.get('source_file')}\n"
            f"Page: {payload.get('page_number')}\n"
            f"Chunk ID: {payload.get('chunk_id')}\n"
            f"Text:\n{payload.get('text')}"
        )

        citations.append({
            "source_number": i + 1,
            "source_file": payload.get("source_file"),
            "page_number": payload.get("page_number"),
            "chunk_id": payload.get("chunk_id"),
            "score": result.score,
        })

    return "\n\n---\n\n".join(source_blocks), citations


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "SourceMind running with Qdrant + duplicate prevention",
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    contents = await file.read()
    sha256 = file_hash(contents)

    existing = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=duplicate_filter(sha256),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )[0]

    if existing:
        payload = existing[0].payload or {}
        return {
            "status": "duplicate",
            "saved_filename": payload.get("source_file"),
            "file_sha256": sha256,
            "message": "This PDF already exists in the knowledge library.",
        }

    document_id = str(uuid4())
    safe_filename = file.filename.replace(" ", "_")
    saved_filename = f"{document_id}_{safe_filename}"
    file_path = UPLOAD_DIR / saved_filename

    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "status": "uploaded",
        "saved_filename": saved_filename,
        "file_sha256": sha256,
        "message": "PDF uploaded successfully.",
    }


@app.get("/extract/{filename}")
def extract_text(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    extracted = []

    for i, page in enumerate(pages):
        extracted.append({"page": i + 1, "text": page.page_content[:500]})

    return {"filename": filename, "pages": len(pages), "preview": extracted}


@app.get("/ingest/{filename}")
def ingest_pdf(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    contents = file_path.read_bytes()
    sha256 = file_hash(contents)

    existing = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=duplicate_filter(sha256),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )[0]

    if existing:
        return {
            "status": "duplicate_skipped",
            "chunks": 0,
            "message": "This PDF has already been processed into Qdrant.",
        }

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    points = []

    for page_index, page in enumerate(pages):
        page_number = page_index + 1
        cleaned = clean_text(page.page_content)

        if not cleaned:
            continue

        chunks = splitter.split_text(cleaned)
        vectors = list(embedding_model.embed(chunks))

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid4())
            vector = vectors[i].tolist() if hasattr(vectors[i], "tolist") else vectors[i]

            payload = {
                "source_file": filename,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "file_sha256": sha256,
                "text": chunk,
            }

            points.append(PointStruct(id=chunk_id, vector=vector, payload=payload))

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {
        "status": "stored_in_qdrant",
        "chunks": len(points),
        "file_sha256": sha256,
    }


@app.post("/reset-knowledge-library")
def reset_knowledge_library():
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    return {"status": "reset_complete", "message": "Qdrant knowledge library cleared."}


def strip_uuid_prefix(filename: str) -> str:
    """Saved filenames look like '<uuid4>_<original>'. Return just the original part."""
    if not filename:
        return filename
    parts = filename.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 36 and parts[0].count("-") == 4:
        return parts[1]
    return filename


@app.get("/documents")
def list_documents():
    documents = {}
    next_offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            filename = payload.get("source_file")
            if not filename:
                continue

            doc = documents.setdefault(filename, {
                "filename": filename,
                "file_sha256": payload.get("file_sha256"),
                "pages": set(),
                "chunks": 0,
            })
            page_number = payload.get("page_number")
            if page_number is not None:
                doc["pages"].add(page_number)
            doc["chunks"] += 1

        if not next_offset:
            break

    result = []
    for doc in documents.values():
        file_path = UPLOAD_DIR / doc["filename"]
        result.append({
            "filename": doc["filename"],
            "original_filename": strip_uuid_prefix(doc["filename"]),
            "file_sha256": doc["file_sha256"],
            "pages": len(doc["pages"]),
            "chunks": doc["chunks"],
            "file_on_disk": file_path.exists(),
        })

    result.sort(key=lambda d: d["original_filename"].lower())

    return {"documents": result, "count": len(result)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove all Qdrant chunks for a document. The PDF on disk is intentionally
    kept so the document can be re-ingested without re-uploading."""

    selector = FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filename),
                )
            ]
        )
    )

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=selector,
    )

    file_path = UPLOAD_DIR / filename

    return {
        "status": "deleted",
        "filename": filename,
        "file_on_disk": file_path.exists(),
        "message": (
            f"Removed all chunks for {filename} from the knowledge library. "
            f"PDF kept on disk so it can be re-ingested."
        ),
    }


@app.get("/search")
def search_knowledge_library(question: str):
    results = retrieve_sources(question)

    matches = []

    for i, result in enumerate(results):
        payload = result.payload or {}
        matches.append({
            "rank": i + 1,
            "text": payload.get("text"),
            "metadata": {
                "source_file": payload.get("source_file"),
                "page_number": payload.get("page_number"),
                "chunk_id": payload.get("chunk_id"),
            },
            "score": result.score,
        })

    return {"question": question, "matches": matches}


@app.get("/ask-local")
def ask_local(question: str, model: str = "qwen3.5:4b"):
    results = retrieve_sources(question)

    if not results:
        return {
            "question": question,
            "provider": "ollama",
            "model_used": model,
            "answer": "I cannot verify that from the uploaded sources.",
            "citations": [],
        }

    context, citations = build_source_context(results)

    prompt = f"""
You are a source-grounded research assistant.

Rules:
- Answer only using the provided source chunks.
- If the chunks do not support the answer, say: I cannot verify that from the uploaded sources.
- Cite every important claim using [Source 1], [Source 2], etc.
- Do not invent sources.

Question:
{question}

Source chunks:
{context}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

    return {
        "question": question,
        "provider": "ollama",
        "model_used": model,
        "answer": response.json().get("response", ""),
        "citations": citations,
    }


def stream_header(citations: list, conversation_id: str) -> str:
    """First line of every streaming response: JSON metadata, terminated by \\n.

    The frontend buffers up to the first newline, parses it as JSON, then
    treats the remainder of the stream as the answer body. The conversation_id
    lets the UI submit feedback against this specific turn.
    """
    return json.dumps({"citations": citations, "conversation_id": conversation_id}) + "\n"


@app.get("/ask-local-stream")
def ask_local_stream(question: str, model: str = "qwen3.5:4b"):
    results = retrieve_sources(question)
    conversation_id = str(uuid4())

    if not results:
        def no_sources():
            answer = "I cannot verify that from the uploaded sources."
            yield stream_header([], conversation_id)
            yield answer
            log_conversation(
                conversation_id=conversation_id,
                question=question,
                provider="ollama",
                model=model,
                citations=[],
                answer=answer,
            )
        return StreamingResponse(no_sources(), media_type="text/plain")

    context, citations = build_source_context(results)

    prompt = f"""
You are a source-grounded research assistant.

Rules:
- Answer only using the provided source chunks.
- If the chunks do not support the answer, say: I cannot verify that from the uploaded sources.
- Cite every important claim using [Source 1], [Source 2], etc.
- Do not invent sources.
- Be clear, concise, and structured.

Question:
{question}

Source chunks:
{context}
"""

    def generate():
        yield stream_header(citations, conversation_id)
        accumulated: list[str] = []

        try:
            with requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                stream=True,
                timeout=300,
            ) as response:
                if response.status_code != 200:
                    error_message = f"\n[Ollama error: {response.text}]"
                    accumulated.append(error_message)
                    yield error_message
                    return

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = data.get("response", "")
                    if token:
                        accumulated.append(token)
                        yield token

                    if data.get("done"):
                        break
        finally:
            log_conversation(
                conversation_id=conversation_id,
                question=question,
                provider="ollama",
                model=model,
                citations=citations,
                answer="".join(accumulated),
            )

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/ask-openai-stream")
def ask_openai_stream(
    question: str,
    model: str = "gpt-4o-mini",
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    client = make_openai_client(x_openai_key)
    results = retrieve_sources(question)
    conversation_id = str(uuid4())

    if not results:
        def no_sources():
            answer = "I cannot verify that from the uploaded sources."
            yield stream_header([], conversation_id)
            yield answer
            log_conversation(
                conversation_id=conversation_id,
                question=question,
                provider="openai",
                model=model,
                citations=[],
                answer=answer,
            )
        return StreamingResponse(no_sources(), media_type="text/plain")

    context, citations = build_source_context(results)

    system_prompt = """
You are a source-grounded research assistant.

Rules:
1. Answer only using the provided source chunks.
2. If the chunks do not support the answer, say: "I cannot verify that from the uploaded sources."
3. Cite every important claim using [Source 1], [Source 2], etc.
4. Do not invent sources.
5. Be clear, concise, and structured.
"""

    user_prompt = f"""
Question:
{question}

Source chunks:
{context}
"""

    def generate():
        yield stream_header(citations, conversation_id)
        accumulated: list[str] = []

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated.append(delta)
                    yield delta
        finally:
            log_conversation(
                conversation_id=conversation_id,
                question=question,
                provider="openai",
                model=model,
                citations=citations,
                answer="".join(accumulated),
            )

    return StreamingResponse(generate(), media_type="text/plain")


class FeedbackPayload(BaseModel):
    feedback: str
    correction: Optional[str] = None


@app.post("/feedback/{conversation_id}")
def save_feedback(conversation_id: str, payload: FeedbackPayload):
    if payload.feedback not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="feedback must be 'up' or 'down'.")

    with sqlite3.connect(CONVERSATIONS_DB) as conn:
        cursor = conn.execute(
            """
            UPDATE conversations
               SET feedback = ?, correction = ?
             WHERE id = ?
            """,
            (payload.feedback, payload.correction, conversation_id),
        )
        conn.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="conversation_id not found")

    return {"status": "saved", "conversation_id": conversation_id}


@app.get("/conversations")
def list_conversations(limit: int = 100):
    with sqlite3.connect(CONVERSATIONS_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, ts, question, provider, model, citations_json, answer,
                   feedback, correction
              FROM conversations
             ORDER BY ts DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()

    items = []
    for row in rows:
        items.append({
            "id": row["id"],
            "ts": row["ts"],
            "question": row["question"],
            "provider": row["provider"],
            "model": row["model"],
            "citations": json.loads(row["citations_json"] or "[]"),
            "answer": row["answer"],
            "feedback": row["feedback"],
            "correction": row["correction"],
        })

    return {"conversations": items, "count": len(items)}


@app.get("/export-training-data")
def export_training_data(format: str = "messages", only_thumbs_up: bool = False):
    """Emit logged conversations as JSONL suitable for fine-tuning.

    format=messages -> {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
    format=text     -> {"text": "<question>\\n\\n<answer>"}
    """
    if format not in {"messages", "text"}:
        raise HTTPException(status_code=400, detail="format must be 'messages' or 'text'.")

    with sqlite3.connect(CONVERSATIONS_DB) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT question, answer, citations_json, correction, feedback
              FROM conversations
             WHERE answer IS NOT NULL AND answer != ''
        """
        params: tuple = ()
        if only_thumbs_up:
            query += " AND feedback = 'up'"
        rows = conn.execute(query, params).fetchall()

    def lines():
        for row in rows:
            answer = row["correction"] or row["answer"]
            citations = json.loads(row["citations_json"] or "[]")

            if format == "messages":
                context_block = "\n\n---\n\n".join(
                    f"Source {c.get('source_number')}: {c.get('source_file')} p.{c.get('page_number')}"
                    for c in citations
                ) or "(no retrieved sources)"

                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a source-grounded research assistant. "
                                "Answer only from the provided sources and cite them."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question:\n{row['question']}\n\n"
                                f"Sources:\n{context_block}"
                            ),
                        },
                        {"role": "assistant", "content": answer},
                    ]
                }
            else:
                record = {"text": f"{row['question']}\n\n{answer}"}

            yield json.dumps(record) + "\n"

    headers = {"Content-Disposition": 'attachment; filename="training_data.jsonl"'}
    return StreamingResponse(lines(), media_type="application/x-ndjson", headers=headers)