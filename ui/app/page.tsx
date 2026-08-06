"use client";

import { useRef, useState } from "react";
import { parseSSE } from "./lib/sse";
import type { ChatMessage, Citation } from "./lib/types";

// Must be called directly from the browser, never proxied through a Vercel
// serverless function -- Vercel Hobby functions time out at 10s, but a real
// (non-cached) /query answer legitimately takes 12-24s (PLAN.md P3.8).
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Real questions this corpus answers well, verified throughout development --
// same list as scripts/prewarm_cache.py's DEMO_QUERIES, kept in sync manually
// (small, stable list; not worth importing across the Python/TS boundary).
const SAMPLE_QUERIES = [
  "what is speculative decoding?",
  "how does continuous batching work in LLM serving?",
  "what is a KV cache and how does it affect memory usage in transformers?",
  "what are the tradeoffs of tensor parallelism?",
];

function updateLast(prev: ChatMessage[], patch: Partial<ChatMessage>): ChatMessage[] {
  const next = [...prev];
  const last = next[next.length - 1];
  if (last) next[next.length - 1] = { ...last, ...patch };
  return next;
}

function CitationCard({ citation, index }: { citation: Citation; index: number }) {
  return (
    <a
      href={citation.abs_url}
      target="_blank"
      rel="noreferrer"
      className="block rounded-lg border border-zinc-200 px-3 py-2 text-sm transition hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600"
    >
      <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">[{index + 1}]</span>{" "}
      <span className="font-medium">{citation.title}</span>
      {citation.section && (
        <span className="text-zinc-500 dark:text-zinc-400"> &middot; {citation.section}</span>
      )}
    </a>
  );
}

function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-2xl ${isUser ? "w-fit" : "w-full"}`}>
        <div
          className={`whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-zinc-900 text-zinc-50 dark:bg-zinc-100 dark:text-zinc-900"
              : "bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100"
          }`}
        >
          {message.content}
          {message.pending && !message.content && (
            <span className="inline-block animate-pulse text-zinc-400">thinking&hellip;</span>
          )}
          {message.error && <span className="text-red-500">{message.error}</span>}
        </div>
        {message.citations && message.citations.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {message.citations.map((c, i) => (
              <CitationCard key={c.chunk_id} citation={c} index={i} />
            ))}
          </div>
        )}
        {message.latencyMs !== undefined && (
          <div className="mt-1.5 text-xs text-zinc-400 dark:text-zinc-500">
            {message.latencyMs < 500
              ? `${message.latencyMs}ms (cache hit)`
              : `${(message.latencyMs / 1000).toFixed(1)}s`}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function runQuery(query: string) {
    if (!query || isStreaming) return;

    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", pending: true },
    ]);
    setIsStreaming(true);
    const start = performance.now();

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const retryAfter = res.headers.get("Retry-After");
        const message =
          res.status === 429
            ? `Rate limited -- try again in ${retryAfter ?? "a bit"}s.`
            : `Request failed (${res.status}).`;
        setMessages((prev) => updateLast(prev, { pending: false, error: message }));
        return;
      }

      let content = "";
      let citations: Citation[] = [];
      for await (const evt of parseSSE(res)) {
        if (evt.event === "token") {
          content += (evt.data as { content: string }).content;
          setMessages((prev) => updateLast(prev, { content, pending: true }));
        } else if (evt.event === "sources") {
          citations = (evt.data as { citations: Citation[] }).citations;
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }

      const latencyMs = Math.round(performance.now() - start);
      setMessages((prev) => updateLast(prev, { content, citations, latencyMs, pending: false }));
    } catch (err) {
      setMessages((prev) =>
        updateLast(prev, {
          pending: false,
          error: err instanceof Error ? err.message : "Request failed.",
        }),
      );
    } finally {
      setIsStreaming(false);
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void runQuery(input.trim());
  }

  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-black">
      <header className="border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-lg font-semibold">CoreRAG</h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Agentic RAG over AI-systems arXiv literature -- hybrid retrieval, cross-encoder
          reranking, a reflection loop, and a semantic cache.
        </p>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-sm text-zinc-400 dark:text-zinc-500">
              Ask something about LLM inference serving, KV cache compression, speculative
              decoding, distributed training, or hardware accelerators -- or try one of these:
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => runQuery(q)}
                  className="rounded-full border border-zinc-300 px-3 py-1.5 text-xs text-zinc-600 transition hover:border-zinc-500 hover:text-zinc-900 dark:border-zinc-700 dark:text-zinc-400 dark:hover:border-zinc-500 dark:hover:text-zinc-100"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} message={m} />
        ))}
        <div ref={bottomRef} />
      </main>

      <form
        onSubmit={handleSubmit}
        className="mx-auto flex w-full max-w-3xl gap-2 border-t border-zinc-200 px-6 py-4 dark:border-zinc-800"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isStreaming}
          className="flex-1 rounded-full border border-zinc-300 px-4 py-2 text-sm outline-none focus:border-zinc-500 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-950 dark:focus:border-zinc-500"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="rounded-full bg-zinc-900 px-5 py-2 text-sm font-medium text-zinc-50 transition disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {isStreaming ? "..." : "Ask"}
        </button>
      </form>
    </div>
  );
}
