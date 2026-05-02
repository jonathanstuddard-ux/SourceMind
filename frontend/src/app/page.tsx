"use client";

import { useCallback, useEffect, useState } from "react";

type Provider = "ollama" | "openai";

type Citation = {
  source_number?: number;
  source_file?: string;
  page_number?: number;
  chunk_id?: string;
  score?: number;
};

type Document = {
  filename: string;
  original_filename: string;
  file_sha256?: string;
  pages: number;
  chunks: number;
  file_on_disk: boolean;
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const OPENAI_KEY_STORAGE = "sourcemind.openaiKey";

const ollamaModels = [
  { value: "qwen3.5:4b", label: "qwen3.5:4b - Fast local model (installed)" },
  { value: "gemma4:31b", label: "gemma4:31b - Large local model (installed)" },
  { value: "qwen2.5:7b", label: "qwen2.5:7b - Strong local model" },
  { value: "llama3.1:8b", label: "llama3.1:8b - Reliable local model" },
  { value: "mistral:7b", label: "mistral:7b - Fast local model" },
  { value: "gemma2:9b", label: "gemma2:9b - Balanced local model" },
  { value: "phi3:medium", label: "phi3:medium - Lightweight local model" },
];

const openaiModels = [
  { value: "gpt-4o-mini", label: "gpt-4o-mini - Fast OpenAI model" },
  { value: "gpt-4o", label: "gpt-4o - Stronger OpenAI model" },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [question, setQuestion] = useState("");
  const [provider, setProvider] = useState<Provider>("ollama");
  const [model, setModel] = useState("qwen3.5:4b");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [busyDocument, setBusyDocument] = useState<string | null>(null);

  const [openaiKey, setOpenaiKey] = useState("");
  const [openaiKeyStored, setOpenaiKeyStored] = useState(false);
  const [showKey, setShowKey] = useState(false);

  const [conversationId, setConversationId] = useState<string | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<"up" | "down" | null>(null);
  const [correction, setCorrection] = useState("");
  const [correctionSaved, setCorrectionSaved] = useState(false);
  const [showCorrection, setShowCorrection] = useState(false);
  const [savingFeedback, setSavingFeedback] = useState(false);
  const [feedbackPulse, setFeedbackPulse] = useState(false);

  const modelOptions = provider === "ollama" ? ollamaModels : openaiModels;
  const needsOpenaiKey = provider === "openai" && !openaiKeyStored;

  const fetchDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/documents`);
      const data = await res.json();
      if (!res.ok) {
        setStatus(`Could not load documents: ${data.detail || "Unknown error"}`);
        return;
      }
      setDocuments(Array.isArray(data.documents) ? data.documents : []);
    } catch {
      setStatus("Could not load documents. Make sure the backend is running.");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  useEffect(() => {
    // Fetch the document library once on mount. This is an intentional
    // external-data-fetch effect; the setState happens inside fetchDocuments.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(OPENAI_KEY_STORAGE);
    if (stored) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setOpenaiKey(stored);
      setOpenaiKeyStored(true);
    }
  }, []);

  function saveOpenaiKey() {
    const trimmed = openaiKey.trim();
    if (!trimmed) {
      setStatus("Please paste a non-empty OpenAI API key first.");
      return;
    }
    window.localStorage.setItem(OPENAI_KEY_STORAGE, trimmed);
    setOpenaiKey(trimmed);
    setOpenaiKeyStored(true);
    setStatus("OpenAI API key saved in this browser.");
  }

  function clearOpenaiKey() {
    window.localStorage.removeItem(OPENAI_KEY_STORAGE);
    setOpenaiKey("");
    setOpenaiKeyStored(false);
    setStatus("OpenAI API key cleared from this browser.");
  }

  async function deleteDocument(doc: Document) {
    const confirmed = window.confirm(
      `Delete all chunks for "${doc.original_filename}" from the knowledge library?\n\n` +
        `The PDF will stay on disk so you can re-ingest it later.`
    );
    if (!confirmed) return;

    setBusyDocument(doc.filename);
    setStatus(`Deleting ${doc.original_filename} from knowledge library...`);

    try {
      const res = await fetch(
        `${BACKEND_URL}/documents/${encodeURIComponent(doc.filename)}`,
        { method: "DELETE" }
      );
      const data = await res.json();
      if (!res.ok) {
        setStatus(`Delete failed: ${data.detail || "Unknown error"}`);
        return;
      }
      setStatus(data.message || `Deleted ${doc.original_filename}.`);
      await fetchDocuments();
    } catch {
      setStatus("Delete failed. Make sure the backend is running.");
    } finally {
      setBusyDocument(null);
    }
  }

  async function reingestDocument(doc: Document) {
    setBusyDocument(doc.filename);
    setStatus(`Re-ingesting ${doc.original_filename}...`);

    try {
      const res = await fetch(
        `${BACKEND_URL}/ingest/${encodeURIComponent(doc.filename)}`
      );
      const data = await res.json();
      if (!res.ok) {
        setStatus(`Re-ingest failed: ${data.detail || "Unknown error"}`);
        return;
      }
      if (data.status === "duplicate_skipped") {
        setStatus(data.message || "Already ingested.");
      } else {
        setStatus(`Re-ingested ${data.chunks ?? 0} chunks for ${doc.original_filename}.`);
      }
      await fetchDocuments();
    } catch {
      setStatus("Re-ingest failed. Make sure Qdrant and the backend are running.");
    } finally {
      setBusyDocument(null);
    }
  }

  function handleProviderChange(nextProvider: Provider) {
    setProvider(nextProvider);
    setModel(nextProvider === "ollama" ? "qwen3.5:4b" : "gpt-4o-mini");
    setAnswer("");
    setCitations([]);
    setConversationId(null);
    setFeedbackStatus(null);
    setShowCorrection(false);
    setCorrectionSaved(false);
    setStatus(
      nextProvider === "ollama"
        ? "Using local Ollama models."
        : openaiKeyStored
          ? "Using OpenAI cloud models."
          : "Add your OpenAI API key in the OpenAI Settings panel below to ask questions."
    );
  }

  async function uploadPdf() {
    if (!file) {
      setStatus("Please choose a PDF first.");
      return;
    }

    setLoading(true);
    setStatus("Uploading PDF...");
    setAnswer("");
    setCitations([]);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus(`Upload failed: ${data.detail || "Unknown error"}`);
        return;
      }

      setUploadedFilename(data.saved_filename);

      if (data.status === "duplicate") {
        setStatus(data.message || "This PDF already exists in the knowledge library.");
      } else {
        setStatus(`Uploaded: ${data.saved_filename}`);
      }
      await fetchDocuments();
    } catch {
      setStatus("Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  async function ingestPdf() {
    if (!uploadedFilename) {
      setStatus("Upload a PDF first.");
      return;
    }

    setLoading(true);
    setStatus("Processing PDF into SourceMind knowledge library...");
    setAnswer("");
    setCitations([]);

    try {
      const res = await fetch(
        `${BACKEND_URL}/ingest/${encodeURIComponent(uploadedFilename)}`
      );

      const data = await res.json();

      if (!res.ok) {
        setStatus(`Processing failed: ${data.detail || "Unknown error"}`);
        return;
      }

      if (data.status === "duplicate_skipped") {
        setStatus(data.message || "This PDF has already been processed.");
      } else {
        setStatus(`Processed ${data.chunks ?? 0} chunks into Qdrant.`);
      }
      await fetchDocuments();
    } catch {
      setStatus("Processing failed. Make sure Qdrant and the backend are running.");
    } finally {
      setLoading(false);
    }
  }

  async function askQuestion() {
    if (!question.trim()) {
      setStatus("Please enter a question.");
      return;
    }
    if (provider === "openai" && !openaiKeyStored) {
      setStatus(
        "Add your OpenAI API key in the OpenAI Settings panel below to ask questions."
      );
      return;
    }

    setLoading(true);
    setStatus(
      provider === "ollama"
        ? `SourceMind is streaming locally with ${model}...`
        : `SourceMind is streaming from OpenAI ${model}...`
    );
    setAnswer("");
    setCitations([]);
    setConversationId(null);
    setFeedbackStatus(null);
    setShowCorrection(false);
    setCorrection("");
    setCorrectionSaved(false);

    try {
      const endpoint =
        provider === "openai" ? "ask-openai-stream" : "ask-local-stream";

      const headers: Record<string, string> = {};
      if (provider === "openai") {
        headers["X-OpenAI-Key"] = openaiKey;
      }

      const res = await fetch(
        `${BACKEND_URL}/${endpoint}?question=${encodeURIComponent(
          question
        )}&model=${encodeURIComponent(model)}`,
        { headers }
      );

      if (!res.ok) {
        const text = await res.text();
        setStatus(`Question failed: ${text || "Unknown error"}`);
        return;
      }

      if (!res.body) {
        setStatus("Streaming failed: no response body.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let headerParsed = false;
      let streamedAnswer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        if (!headerParsed) {
          buffer += decoder.decode(value, { stream: true });
          const newlineIndex = buffer.indexOf("\n");
          if (newlineIndex === -1) continue;

          const headerLine = buffer.slice(0, newlineIndex);
          const remainder = buffer.slice(newlineIndex + 1);
          buffer = "";
          headerParsed = true;

          try {
            const header = JSON.parse(headerLine);
            if (Array.isArray(header.citations)) {
              setCitations(header.citations);
            }
            if (typeof header.conversation_id === "string") {
              setConversationId(header.conversation_id);
            }
          } catch {
            streamedAnswer += headerLine;
          }

          if (remainder) {
            streamedAnswer += remainder;
            setAnswer(streamedAnswer);
          }
          continue;
        }

        streamedAnswer += decoder.decode(value, { stream: true });
        setAnswer(streamedAnswer);
      }

      setStatus(
        `Answer streamed using ${
          provider === "openai" ? "OpenAI" : "Ollama"
        } / ${model}.`
      );
    } catch {
      setStatus(
        "Question failed. Make sure Ollama/OpenAI, Qdrant, and backend are running."
      );
    } finally {
      setLoading(false);
    }
  }

  async function submitFeedback(verdict: "up" | "down", correctionText?: string) {
    if (!conversationId) return;
    setSavingFeedback(true);
    const sentCorrection = !!(correctionText && correctionText.trim());
    try {
      const body: { feedback: "up" | "down"; correction?: string } = {
        feedback: verdict,
      };
      if (sentCorrection) {
        body.correction = correctionText!.trim();
      }
      const res = await fetch(`${BACKEND_URL}/feedback/${conversationId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        setStatus(`Feedback save failed: ${text || "Unknown error"}`);
        return;
      }
      setFeedbackStatus(verdict);
      if (sentCorrection) {
        setCorrectionSaved(true);
      }
      setFeedbackPulse(true);
      window.setTimeout(() => setFeedbackPulse(false), 1200);
      setStatus(
        sentCorrection
          ? "Thanks — your correction was saved."
          : verdict === "up"
            ? "Thanks — saved as a positive example."
            : "Thanks — saved for review and future tuning."
      );
      if (verdict === "up") {
        setShowCorrection(false);
      }
    } catch {
      setStatus("Could not save feedback. Make sure the backend is running.");
    } finally {
      setSavingFeedback(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <section className="rounded-2xl bg-slate-900 p-8 shadow-xl border border-slate-800">
          <p className="text-sm uppercase tracking-widest text-blue-400 font-semibold">
            Private RAG Knowledge System
          </p>
          <h1 className="text-4xl md:text-5xl font-bold mt-3">SourceMind</h1>
          <p className="text-slate-300 text-lg max-w-3xl mt-3">
            Upload searchable PDFs, process them into a Qdrant-powered knowledge
            library, and ask questions using either local Ollama models or OpenAI
            cloud models with source citations.
          </p>
        </section>

        <section className="grid md:grid-cols-2 gap-6">
          <div className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-4">1. Upload PDF</h2>

            <label className="mb-4 flex cursor-pointer items-center justify-center rounded-xl border border-dashed border-slate-600 bg-slate-800 px-4 py-8 text-center hover:bg-slate-700 transition">
              <span className="text-slate-200 break-all">
                {file ? file.name : "Click here to choose a searchable PDF"}
              </span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
            </label>

            <button
              onClick={uploadPdf}
              disabled={loading}
              className={`px-4 py-2 rounded-xl font-semibold ${
                loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              Upload PDF
            </button>

            {uploadedFilename && (
              <p className="mt-4 text-sm text-slate-400 break-all">
                Uploaded file: {uploadedFilename}
              </p>
            )}
          </div>

          <div className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-4">2. Process PDF</h2>
            <p className="text-slate-300 mb-4">
              SourceMind extracts text, cleans it, chunks it, creates embeddings
              with FastEmbed, and stores vectors in Qdrant.
            </p>

            <button
              onClick={ingestPdf}
              disabled={loading}
              className={`px-4 py-2 rounded-xl font-semibold ${
                loading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-emerald-600 hover:bg-emerald-700"
              }`}
            >
              Process into Knowledge Library
            </button>
          </div>
        </section>

        <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
          <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
            <h2 className="text-2xl font-semibold">3. Document Library</h2>
            <button
              onClick={fetchDocuments}
              disabled={documentsLoading}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                documentsLoading
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-slate-700 hover:bg-slate-600"
              }`}
            >
              {documentsLoading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          <p className="text-slate-400 text-sm mb-4">
            Documents currently indexed in Qdrant. Deleting only removes the
            chunks &mdash; the PDF stays on disk so you can re-ingest later.
          </p>

          {documents.length === 0 ? (
            <p className="text-slate-400">
              {documentsLoading
                ? "Loading documents..."
                : "No documents in the knowledge library yet."}
            </p>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => {
                const isBusy = busyDocument === doc.filename;
                return (
                  <div
                    key={doc.filename}
                    className="rounded-xl bg-slate-800 p-4 border border-slate-700"
                  >
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                      <div className="min-w-0">
                        <p className="font-semibold break-all">
                          {doc.original_filename}
                        </p>
                        <p className="text-xs text-slate-400 break-all mt-1">
                          {doc.filename}
                        </p>
                        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-sm text-slate-300">
                          <span>{doc.pages} {doc.pages === 1 ? "page" : "pages"}</span>
                          <span>{doc.chunks} {doc.chunks === 1 ? "chunk" : "chunks"}</span>
                          {doc.file_sha256 && (
                            <span className="text-slate-400">
                              sha256: {doc.file_sha256.slice(0, 12)}…
                            </span>
                          )}
                          {!doc.file_on_disk && (
                            <span className="text-amber-400">
                              PDF missing on disk
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex gap-2 shrink-0">
                        <button
                          onClick={() => reingestDocument(doc)}
                          disabled={isBusy || !doc.file_on_disk}
                          title={
                            doc.file_on_disk
                              ? "Re-process this PDF into Qdrant"
                              : "Original PDF is missing from disk"
                          }
                          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                            isBusy || !doc.file_on_disk
                              ? "bg-gray-600 cursor-not-allowed"
                              : "bg-emerald-600 hover:bg-emerald-700"
                          }`}
                        >
                          Re-ingest
                        </button>
                        <button
                          onClick={() => deleteDocument(doc)}
                          disabled={isBusy}
                          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                            isBusy
                              ? "bg-gray-600 cursor-not-allowed"
                              : "bg-red-600 hover:bg-red-700"
                          }`}
                        >
                          {isBusy ? "Working..." : "Delete"}
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
          <h2 className="text-2xl font-semibold mb-2">4. OpenAI Settings</h2>
          <p className="text-slate-400 text-sm mb-4">
            Your OpenAI API key is stored only in this browser&apos;s
            localStorage and sent to your backend as a request header. It is
            never persisted on the server. Local Ollama models do not need a
            key.
          </p>

          <div className="flex flex-col sm:flex-row gap-2 mb-3">
            <input
              type={showKey ? "text" : "password"}
              value={openaiKey}
              onChange={(e) => setOpenaiKey(e.target.value)}
              placeholder="sk-..."
              autoComplete="off"
              spellCheck={false}
              className="flex-1 rounded-xl bg-slate-800 border border-slate-700 p-3 text-white font-mono text-sm"
            />
            <button
              onClick={() => setShowKey((s) => !s)}
              className="px-3 py-2 rounded-xl text-sm font-medium bg-slate-700 hover:bg-slate-600"
              type="button"
            >
              {showKey ? "Hide" : "Show"}
            </button>
            <button
              onClick={saveOpenaiKey}
              className="px-3 py-2 rounded-xl text-sm font-semibold bg-blue-600 hover:bg-blue-700"
              type="button"
            >
              Save
            </button>
            <button
              onClick={clearOpenaiKey}
              disabled={!openaiKeyStored && !openaiKey}
              className={`px-3 py-2 rounded-xl text-sm font-medium ${
                !openaiKeyStored && !openaiKey
                  ? "bg-gray-700 cursor-not-allowed"
                  : "bg-red-600 hover:bg-red-700"
              }`}
              type="button"
            >
              Clear
            </button>
          </div>

          <p className="text-sm">
            Status:{" "}
            {openaiKeyStored ? (
              <span className="text-emerald-400 font-medium">
                Key saved in this browser
              </span>
            ) : (
              <span className="text-amber-400 font-medium">
                No key saved &mdash; OpenAI provider is disabled
              </span>
            )}
          </p>
        </section>

        <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
          <h2 className="text-2xl font-semibold mb-4">5. Ask SourceMind</h2>

          <label className="block text-sm text-slate-300 mb-2">
            Choose AI Provider
          </label>
          <select
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value as Provider)}
            className="mb-4 w-full rounded-xl bg-slate-800 border border-slate-700 p-3 text-white"
          >
            <option value="ollama">Local Ollama</option>
            <option value="openai">OpenAI Cloud</option>
          </select>

          <label className="block text-sm text-slate-300 mb-2">
            Choose Model
          </label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="mb-4 w-full rounded-xl bg-slate-800 border border-slate-700 p-3 text-white"
          >
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your uploaded documents..."
            className="w-full h-32 rounded-xl bg-slate-800 border border-slate-700 p-4 text-white mb-4"
          />

          <button
            onClick={askQuestion}
            disabled={loading || needsOpenaiKey}
            className={`px-5 py-3 rounded-xl font-semibold ${
              loading || needsOpenaiKey
                ? "bg-gray-600 cursor-not-allowed"
                : "bg-purple-600 hover:bg-purple-700"
            }`}
          >
            {loading
              ? "Thinking..."
              : needsOpenaiKey
                ? "Add OpenAI key above to ask"
                : "Ask Question"}
          </button>

          {loading && (
            <p className="mt-3 text-sm text-slate-400">
              SourceMind is working. Local models may take longer than OpenAI,
              especially on CPU.
            </p>
          )}
        </section>

        {status && (
          <section className="rounded-xl bg-slate-800 p-4 border border-slate-700">
            <p className="text-slate-200">{status}</p>
          </section>
        )}

        {answer && (
          <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-4">Answer</h2>
            <div className="whitespace-pre-wrap text-slate-200 leading-7">
              {answer}
            </div>

            {conversationId && !loading && (
              <div className="mt-5 rounded-xl bg-slate-800 border border-slate-700 p-4">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-sm text-slate-300">
                    Was this answer helpful?
                  </span>
                  <button
                    onClick={() => submitFeedback("up")}
                    disabled={savingFeedback}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                      feedbackStatus === "up"
                        ? "bg-emerald-600"
                        : "bg-slate-700 hover:bg-slate-600"
                    } ${savingFeedback ? "opacity-50 cursor-not-allowed" : ""}`}
                    type="button"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => {
                      setShowCorrection(true);
                      if (feedbackStatus !== "down") {
                        submitFeedback("down");
                      }
                    }}
                    disabled={savingFeedback}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                      feedbackStatus === "down"
                        ? "bg-red-600"
                        : "bg-slate-700 hover:bg-slate-600"
                    } ${savingFeedback ? "opacity-50 cursor-not-allowed" : ""}`}
                    type="button"
                  >
                    No
                  </button>
                  {feedbackStatus && (
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold transition-all duration-300 ${
                        feedbackPulse ? "ring-2 ring-white/40 scale-105" : ""
                      } ${
                        correctionSaved
                          ? "bg-emerald-700/40 text-emerald-200 border border-emerald-500/40"
                          : feedbackStatus === "up"
                            ? "bg-emerald-700/40 text-emerald-200 border border-emerald-500/40"
                            : "bg-amber-700/40 text-amber-200 border border-amber-500/40"
                      }`}
                      role="status"
                      aria-live="polite"
                    >
                      <span aria-hidden="true">✓</span>
                      {correctionSaved
                        ? "Correction saved"
                        : feedbackStatus === "up"
                          ? "Saved as a positive example"
                          : "Saved as needing review"}
                    </span>
                  )}
                </div>

                {showCorrection && (
                  <div className="mt-4">
                    <label className="block text-sm text-slate-300 mb-2">
                      Optional: write the answer you would have preferred.
                      It will be saved alongside this conversation.
                    </label>
                    <textarea
                      value={correction}
                      onChange={(e) => {
                        setCorrection(e.target.value);
                        if (correctionSaved) setCorrectionSaved(false);
                      }}
                      placeholder="A better answer would be..."
                      className="w-full h-24 rounded-xl bg-slate-900 border border-slate-700 p-3 text-white text-sm"
                    />
                    <button
                      onClick={() => submitFeedback("down", correction)}
                      disabled={savingFeedback || !correction.trim()}
                      className={`mt-2 px-3 py-1.5 rounded-lg text-sm font-medium ${
                        savingFeedback || !correction.trim()
                          ? "bg-gray-600 cursor-not-allowed"
                          : "bg-blue-600 hover:bg-blue-700"
                      }`}
                      type="button"
                    >
                      {savingFeedback
                        ? "Saving..."
                        : correctionSaved
                          ? "Update correction"
                          : "Save correction"}
                    </button>
                  </div>
                )}
              </div>
            )}

            <h3 className="text-xl font-semibold mt-6 mb-3">Citations</h3>

            {citations.length === 0 ? (
              <p className="text-slate-400">No citations returned.</p>
            ) : (
              <div className="space-y-3">
                {citations.map((citation, index) => (
                  <div
                    key={index}
                    className="rounded-xl bg-slate-800 p-4 border border-slate-700"
                  >
                    <p className="font-semibold">
                      Source {citation.source_number ?? index + 1}
                    </p>
                    <p className="text-sm text-slate-300 break-all">
                      File: {citation.source_file || "Unknown"}
                    </p>
                    <p className="text-sm text-slate-300">
                      Page: {citation.page_number ?? "Unknown"}
                    </p>
                    <p className="text-sm text-slate-300 break-all">
                      Chunk ID: {citation.chunk_id || "Unknown"}
                    </p>
                    {citation.score !== undefined && (
                      <p className="text-sm text-slate-400">
                        Relevance score: {citation.score.toFixed(4)}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}