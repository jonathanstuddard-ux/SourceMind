# SourceMind

A private, source-grounded RAG (Retrieval-Augmented Generation) knowledge
system. Upload PDFs, ingest them into a local Qdrant vector database, and ask
questions answered with citations using either a **local GGUF** (llama.cpp) or OpenAI
cloud models.

- **Local-first.** PDFs and embeddings live on your machine in Qdrant.
- **Bring-your-own-key OpenAI.** Your OpenAI key stays in your browser's
  localStorage. The backend never persists it.
- **Streaming answers** with structured citations.
- **Conversation logging** built in — feedback (`Yes` / `No` + corrections)
  is stored locally and exportable as fine-tuning data.
- **One-command Docker deploy.** Local chat uses **llama.cpp** (`llama-server`) in Docker with your **GGUF** from `./models` (OpenAI-compatible `/v1` on host **9080** by default).

---

## Architecture

```
┌──────────────┐    fetch (stream)       ┌──────────────┐    embed + search   ┌──────────┐
│  Next.js UI  │ ──────────────────────▶ │   FastAPI    │ ──────────────────▶ │  Qdrant  │
│ (browser)    │  X-OpenAI-Key header    │   backend    │                     │  (vec DB) │
└──────────────┘                         │              │                     └──────────┘
                                         │              │  POST /v1/chat/completions
                                         │              │ ──────────────────▶ ┌────────────┐
                                         │              │                     │ llama.cpp  │
                                         │              │                     │ (GGUF Svc) │
                                         │              │     OpenAI HTTPS  └────────────┘
                                         │              │ ──────────────────▶ OpenAI cloud
                                         │              │
                                         │              │ ──▶ SQLite (conversations.db)
                                         └──────────────┘
```

- **Frontend:** Next.js 16 (React 19, Tailwind v4). Talks to the backend over
  HTTP, parses NDJSON-style streams (first line = JSON metadata header,
  remainder = answer text).
- **Backend:** FastAPI + uvicorn. Streams answers from either the local llama.cpp
  server or OpenAI, attaches citations, logs every turn to SQLite for later
  fine-tuning.
- **Vector store:** Qdrant. Embeddings are computed locally with FastEmbed
  (`BAAI/bge-small-en-v1.5`, 384-dim cosine).
- **LLM providers:** Local **llama.cpp** server (GGUF in `./models`, OpenAI-compatible `/v1`) and OpenAI (per-request key).

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | 4.x+ | Runs Qdrant, local LLM (llama.cpp), backend, frontend |
| Git | any | Cloning |
| OpenAI API key | optional | Only if you want OpenAI models |

> **Local LLM (llama.cpp):** Compose mounts `./models` read-only and starts
> `ghcr.io/ggml-org/llama.cpp:server` with **`Qwen3-1.7B-Q8_0.gguf`** by default
> (see `LLAMAEDGE_MODEL_GGUF`). Download that file into `./models/` first, or
> point `LLAMAEDGE_MODEL_GGUF` at another GGUF path **inside** the container. For
> NVIDIA on Linux/Windows, switch the image to `ghcr.io/ggml-org/llama.cpp:server-cuda`
> and add GPU device reservations per [llama.cpp Docker docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#docker).

---

## Quick start (Docker, recommended)

### 1. Clone and configure

```bash
git clone <this-repo> sourcemind
cd sourcemind
cp .env.example .env       # optional, defaults are fine for most users
```

### 1b. Put the default GGUF in `./models/`

The stack expects **`models/Qwen3-1.7B-Q8_0.gguf`** (unless you override `LLAMAEDGE_MODEL_GGUF`). Example:

```bash
mkdir -p models
curl -L -o models/Qwen3-1.7B-Q8_0.gguf \
  "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf"
```

### 2. Bring up the stack

```bash
docker compose up --build -d
```

That builds and starts four containers:

| Container | Host port(s) | Role |
|---|---|---|
| `sourcemind-qdrant`   | **6335** (HTTP), **6336** (gRPC) | Vector database (mapped off **6333** if another Qdrant already uses it) |
| `sourcemind-llamaedge` | **9080** → 8080 | **llama.cpp** OpenAI-compatible chat (`llama-server`, GGUF from `./models`) |
| `sourcemind-backend`  | **8000** | FastAPI + RAG logic |
| `sourcemind-frontend` | **3000** | Next.js UI |

Open <http://localhost:3000>.

### 3. Use it

1. **Upload PDF** — choose a searchable PDF. Image-only scans won't work;
   OCR them first.
2. **Process PDF** — extracts text, splits into ~1k-character chunks,
   embeds, stores in Qdrant. Duplicate uploads are detected by SHA-256.
3. **Document Library** — see what's indexed; re-ingest or delete chunks
   without losing the original PDF.
4. **OpenAI Settings** — paste your `sk-...` key (only if you plan to use
   OpenAI). It is saved in your browser only.
5. **Ask SourceMind** — pick a provider and model, ask away. Citations show
   the exact source file, page, and chunk ID. Use `Yes` / `No` to log
   feedback.

---

## Configuration

All configuration is environment-driven. See [`.env.example`](./.env.example).

| Variable | Default | Where it lives | Purpose |
|---|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | backend | Vector DB endpoint |
| `LLAMAEDGE_BASE_URL` | `http://localhost:9080` | backend | Local OpenAI-compatible **API root only** — do **not** include `/v1`. In Compose the backend uses `http://llamaedge:8080`. From the **host** (venv backend + stack in Docker), use `http://localhost:9080` for the default publish. |
| `LLAMAEDGE_MODEL_GGUF` | `/models/Qwen3-1.7B-Q8_0.gguf` | compose | Path **inside** the llamaedge container to the GGUF file (left side of `./models:/models`). |
| `LLAMAEDGE_API_KEY` | `not-needed` | backend | Placeholder for the OpenAI SDK; local llama.cpp ignores it. |
| `LLAMAEDGE_DEFAULT_CHAT_MODEL` | `Qwen3-1.7B-Q8_0.gguf` | backend | Default `model` query param and `/local-models` fallback; must match your GGUF id from `GET /v1/models`. Set in Compose for Docker. |
| `OPENAI_API_KEY` | unset | backend (dev only) | Optional fallback. Production users should leave this empty and supply their key from the UI. |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | frontend (build-time) | URL the browser uses to reach the backend |

### How the OpenAI key flows

1. User pastes key into **OpenAI Settings** in the UI → saved to
   `localStorage` under `sourcemind.openaiKey`.
2. On every `/ask-openai-stream` request the frontend attaches the key as
   the `X-OpenAI-Key` HTTP header.
3. The backend builds a per-request `OpenAI` client from that header and
   never writes it to disk.

---

## Common operations

### View logs

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f qdrant
docker compose logs -f llamaedge   # llama.cpp server
```

### Stop / restart

```bash
docker compose stop
docker compose start
docker compose restart backend
```

### Wipe the knowledge library (keeps PDFs on disk)

```bash
curl -X POST http://localhost:8000/reset-knowledge-library
```

### Wipe everything (PDFs + vectors + conversation log)

```bash
docker compose down -v             # removes named volumes
rm -rf backend/data/uploads/*       # if running outside Docker
```

### Inspect logged conversations

```bash
curl http://localhost:8000/conversations | jq .
```

### Export training data (JSONL for fine-tuning)

```bash
# OpenAI fine-tuning chat format
curl 'http://localhost:8000/export-training-data?format=messages&only_thumbs_up=true' \
  -o training_data.jsonl

# Plain text (for base-model continued-pretraining)
curl 'http://localhost:8000/export-training-data?format=text' \
  -o training_data_text.jsonl
```

### Swap the local GGUF model

1. Place the new `.gguf` under **`./models/`** (or another host folder you mount at `/models`).
2. Set **`LLAMAEDGE_MODEL_GGUF`** in `.env` to the **in-container** path, e.g. `/models/MyModel-Q4_K_M.gguf`.
3. Optionally set **`LLAMAEDGE_DEFAULT_CHAT_MODEL`** to the id returned by `GET http://localhost:9080/v1/models` (often the GGUF filename).
4. `docker compose up -d --force-recreate llamaedge backend` so the server reloads the weights.

---

## Development mode (no Docker)

Useful when iterating on Python or React without a rebuild loop.

### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: drop a backend/.env with QDRANT_URL / LLAMAEDGE_BASE_URL overrides.
# DO NOT commit your .env. The .gitignore already excludes it.

# Make sure Qdrant is running. Easiest way:
docker run -d --name sourcemind-qdrant -p 6333:6333 \
  -v sourcemind_qdrant:/qdrant/storage qdrant/qdrant:latest

uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
# Tell Next.js where the backend lives during development:
echo 'NEXT_PUBLIC_BACKEND_URL=http://localhost:8000' > .env.local
npm run dev    # http://localhost:3000
```

---

## API reference (cheat sheet)

| Method & path | Description |
|---|---|
| `GET  /health` | Ping |
| `POST /upload` | Upload a PDF (multipart `file`) |
| `GET  /ingest/{filename}` | Ingest a previously uploaded PDF into Qdrant |
| `POST /reset-knowledge-library` | Drop and recreate the Qdrant collection |
| `GET  /documents` | List indexed documents |
| `DELETE /documents/{filename}` | Remove a document's chunks (keeps PDF) |
| `GET  /search?question=` | Top-k retrieval, no LLM |
| `GET  /local-models` | Chat model ids from the local server (`GET /v1/models`) for the UI |
| `GET  /ask-local-stream?question=&model=` | Streamed answer from local llama.cpp (OpenAI chat completions) |
| `GET  /ask-openai-stream?question=&model=` | Streamed answer from OpenAI. Requires `X-OpenAI-Key` header (or `OPENAI_API_KEY` env) |
| `POST /feedback/{conversation_id}` | `{"feedback":"up"\|"down","correction":"..."}` |
| `GET  /conversations?limit=` | Recent logged turns |
| `GET  /export-training-data?format=messages\|text&only_thumbs_up=` | JSONL export |

Streaming responses use a 1-line JSON header, then the answer body:

```
{"citations":[{...},...],"conversation_id":"<uuid>"}\n
First token of the answer Second token ...
```

---

## Troubleshooting

### "command not found: uvicorn"

You haven't activated the virtualenv. Run:

```bash
cd backend && source .venv/bin/activate
```

…or invoke uvicorn through the venv directly:

```bash
backend/.venv/bin/uvicorn app.main:app --reload
```

### "Address already in use" on port 8000 / 3000 / 6333 / 9080

Something else is bound to that port. Find and kill it:

```bash
lsof -ti:8000 | xargs kill -9
```

### Local (GGUF) answers "I cannot verify that from the uploaded sources"

That's the safety rail kicking in. Either:
- The PDF wasn't ingested (check the Document Library section in the UI), or
- The retrieval found nothing relevant — try rephrasing, or upload more
  documents.

### Backend can't reach the local LLM from inside Docker

Confirm the llamaedge container is healthy and the API responds:

```bash
curl -s http://localhost:9080/v1/models
```

From inside the backend container (same Compose network):

```bash
docker exec sourcemind-backend python -c "import urllib.request; print(urllib.request.urlopen('http://llamaedge:8080/v1/models').read()[:200])"
```

If you point `LLAMAEDGE_BASE_URL` at a **host**-bound llama.cpp server instead, use
`http://host.docker.internal:<port>` — the `extra_hosts:
host.docker.internal:host-gateway` entry is already in `docker-compose.yml` for
Linux.

### "OpenAI API key required"

Open the **OpenAI Settings** panel in the UI and paste your `sk-...` key.
It is stored only in your browser. For local development you can also set
`OPENAI_API_KEY` in `backend/.env`.

### PDF upload returns "duplicate"

That exact byte-for-byte file is already indexed (SHA-256 match). Either
delete it from the Document Library and re-upload, or use a different file.

### Embeddings download is slow on first run

FastEmbed downloads `BAAI/bge-small-en-v1.5` (~100MB) the first time. It
caches under `~/.cache/fastembed/` (or in the container's filesystem inside
the named volume).

---

## Privacy & security

- **No telemetry.** Nothing is sent to a third party except the OpenAI **cloud**
  calls that you explicitly trigger. Local llama.cpp traffic stays on your
  machine / Docker network.
- **OpenAI keys are never persisted server-side.** They live in
  `localStorage` and are sent only as a per-request header.
- **Conversation logs are local.** They live in
  `backend/data/conversations.db` (SQLite). Delete the file to wipe history.
- **Uploaded PDFs are local.** They live in `backend/data/uploads/`.
- **Do not commit `backend/.env`.** The top-level `.gitignore` excludes it,
  but if you've already committed a key to a remote, **rotate it
  immediately** at <https://platform.openai.com/api-keys>.

---

## Roadmap

- Image-PDF support via OCR (Tesseract or Apple Vision).
- Per-user feedback / multi-tenant mode.
- Hybrid search (BM25 + dense).
- LoRA / QLoRA fine-tuning loop driven directly off
  `/export-training-data?only_thumbs_up=true`.
- Optional pgvector backend for production deployments.

---

## License

MIT (or your choice — add a `LICENSE` file).
