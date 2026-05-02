# SourceMind

A private, source-grounded RAG (Retrieval-Augmented Generation) knowledge
system. Upload PDFs, ingest them into a local Qdrant vector database, and ask
questions answered with citations using either local Ollama models or OpenAI
cloud models.

- **Local-first.** PDFs and embeddings live on your machine in Qdrant.
- **Bring-your-own-key OpenAI.** Your OpenAI key stays in your browser's
  localStorage. The backend never persists it.
- **Streaming answers** with structured citations.
- **Conversation logging** built in — feedback (`Yes` / `No` + corrections)
  is stored locally and exportable as fine-tuning data.
- **One-command Docker deploy.** Ollama runs natively on the host so it can
  use Apple Silicon Metal / native CUDA at full speed.

---

## Architecture

```
┌──────────────┐    fetch (stream)       ┌──────────────┐    embed + search   ┌──────────┐
│  Next.js UI  │ ──────────────────────▶ │   FastAPI    │ ──────────────────▶ │  Qdrant  │
│ (browser)    │  X-OpenAI-Key header    │   backend    │                     │  (vec DB) │
└──────────────┘                         │              │                     └──────────┘
                                         │              │     POST /generate
                                         │              │ ──────────────────▶ ┌──────────┐
                                         │              │                     │  Ollama  │
                                         │              │                     │ (host)   │
                                         │              │     OpenAI HTTPS    └──────────┘
                                         │              │ ──────────────────▶ OpenAI cloud
                                         │              │
                                         │              │ ──▶ SQLite (conversations.db)
                                         └──────────────┘
```

- **Frontend:** Next.js 16 (React 19, Tailwind v4). Talks to the backend over
  HTTP, parses NDJSON-style streams (first line = JSON metadata header,
  remainder = answer text).
- **Backend:** FastAPI + uvicorn. Streams answers from either Ollama or
  OpenAI, attaches citations, logs every turn to SQLite for later
  fine-tuning.
- **Vector store:** Qdrant. Embeddings are computed locally with FastEmbed
  (`BAAI/bge-small-en-v1.5`, 384-dim cosine).
- **LLM providers:** Ollama (host process) and OpenAI (per-request key).

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | 4.x+ | Runs Qdrant, backend, frontend |
| Ollama | 0.3+ | Local LLM runtime (host install) |
| Git | any | Cloning |
| OpenAI API key | optional | Only if you want OpenAI models |

> **Why Ollama on the host instead of in a container?** On macOS, Docker
> containers cannot access Apple's Metal GPU. Running Ollama natively gives
> you full GPU acceleration, and it works the same on Linux with NVIDIA. The
> backend container reaches the host via `host.docker.internal`.

---

## Quick start (Docker, recommended)

### 1. Install and start Ollama on the host

```bash
# macOS
brew install ollama
ollama serve &      # runs on http://localhost:11434

# Linux
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

Pull at least one model (the smallest one is enough to start):

```bash
ollama pull qwen2.5:3b
# Optional, larger / better:
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

Verify it's reachable:

```bash
curl http://localhost:11434/api/tags
```

### 2. Clone and configure

```bash
git clone <this-repo> sourcemind
cd sourcemind
cp .env.example .env       # optional, defaults are fine for most users
```

### 3. Bring up the stack

```bash
docker compose up --build -d
```

That builds and starts three containers:

| Container | Port | Role |
|---|---|---|
| `sourcemind-qdrant`   | 6333  | Vector database |
| `sourcemind-backend`  | 8000  | FastAPI + RAG logic |
| `sourcemind-frontend` | 3000  | Next.js UI |

Open <http://localhost:3000>.

### 4. Use it

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
| `OLLAMA_URL` | `http://localhost:11434` | backend | Ollama endpoint. In Docker compose this is overridden to `http://host.docker.internal:11434`. |
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

### Update an Ollama model

```bash
ollama pull qwen2.5:7b
```

The backend picks up new models automatically — they appear in the model
dropdown when you set `provider = ollama` (the dropdown is a static list
right now; you can edit `frontend/src/app/page.tsx` to add more).

---

## Development mode (no Docker)

Useful when iterating on Python or React without a rebuild loop.

### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: drop a backend/.env with QDRANT_URL / OLLAMA_URL overrides.
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
| `GET  /ask-local-stream?question=&model=` | Streamed answer from Ollama |
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

### "Address already in use" on port 8000 / 3000 / 6333

Something else is bound to that port. Find and kill it:

```bash
lsof -ti:8000 | xargs kill -9
```

### Ollama answers "I cannot verify that from the uploaded sources"

That's the safety rail kicking in. Either:
- The PDF wasn't ingested (check the Document Library section in the UI), or
- The retrieval found nothing relevant — try rephrasing, or upload more
  documents.

### Backend can't reach Ollama from inside Docker

Confirm it's running on the **host**:

```bash
curl http://localhost:11434/api/tags
```

…and that the backend container can reach it:

```bash
docker exec sourcemind-backend curl -s http://host.docker.internal:11434/api/tags
```

On Linux this requires the `extra_hosts: host.docker.internal:host-gateway`
entry (already in `docker-compose.yml`).

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

- **No telemetry.** Nothing is sent to a third party except the OpenAI calls
  that you explicitly trigger.
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
