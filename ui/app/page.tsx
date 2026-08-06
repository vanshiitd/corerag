"use client";

import { useCallback, useRef, useState } from "react";
import { parseSSE } from "./lib/sse";
import type { ChatMessage, Citation, TraceInfo } from "./lib/types";

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

function CitationCard({
  citation,
  index,
  highlighted,
  cardRef,
}: {
  citation: Citation;
  index: number;
  highlighted: boolean;
  cardRef: (el: HTMLAnchorElement | null) => void;
}) {
  return (
    <a
      ref={cardRef}
      href={citation.abs_url}
      target="_blank"
      rel="noreferrer"
      className={`block rounded-lg border px-3 py-2 text-sm transition ${
        highlighted
          ? "border-zinc-400 bg-zinc-50 dark:border-zinc-500 dark:bg-zinc-900"
          : "border-zinc-200 hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600"
      }`}
    >
      <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">[{index + 1}]</span>{" "}
      <span className="font-medium">{citation.title}</span>
      {citation.section && (
        <span className="text-zinc-500 dark:text-zinc-400"> &middot; {citation.section}</span>
      )}
    </a>
  );
}

/** Splits answer text on "[n]" citation markers. Each piece is either plain text
 * or a 0-based citation index -- pure and ref-free so it's safe to call during
 * render; the caller decides how to turn a citation-index piece into a click
 * target (needs a ref to the matching card, which must stay out of this helper). */
function splitAnswerText(content: string): (string | number)[] {
  return content.split(/(\[\d+\])/g).map((part) => {
    const match = /^\[(\d+)\]$/.exec(part);
    return match ? Number(match[1]) - 1 : part;
  });
}

function TracePanel({ trace }: { trace: TraceInfo }) {
  const badges: string[] = [];
  badges.push(trace.cached ? "cache hit" : "live run");
  if (trace.route) badges.push(trace.route === "multi_hop" ? "multi-hop route" : "simple route");
  if (trace.retries) badges.push(`${trace.retries} reflection ${trace.retries === 1 ? "retry" : "retries"}`);
  if (trace.low_confidence) badges.push("low confidence");

  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {badges.map((b) => (
        <span
          key={b}
          className={`rounded-full border px-2 py-0.5 text-[11px] ${
            b === "low confidence"
              ? "border-amber-300 text-amber-600 dark:border-amber-800 dark:text-amber-500"
              : "border-zinc-200 text-zinc-400 dark:border-zinc-800 dark:text-zinc-500"
          }`}
        >
          {b}
        </span>
      ))}
    </div>
  );
}

function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const citationRefs = useRef<Record<number, HTMLAnchorElement | null>>({});
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const jumpToCitation = useCallback((index: number) => {
    citationRefs.current[index]?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlighted(index);
    window.setTimeout(() => setHighlighted((h) => (h === index ? null : h)), 1500);
  }, []);

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
          {isUser
            ? message.content
            : splitAnswerText(message.content).map((part, i) => {
                if (
                  typeof part === "number" &&
                  part >= 0 &&
                  part < (message.citations?.length ?? 0)
                ) {
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => jumpToCitation(part)}
                      className="border-0 bg-transparent p-0 font-mono text-current underline decoration-dotted underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
                    >
                      [{part + 1}]
                    </button>
                  );
                }
                return <span key={i}>{typeof part === "number" ? `[${part + 1}]` : part}</span>;
              })}
          {message.pending && !message.content && (
            <span className="inline-block animate-pulse text-zinc-400">thinking&hellip;</span>
          )}
          {message.error && <span className="text-red-500">{message.error}</span>}
        </div>
        {message.citations && message.citations.length > 0 && (
          <div className="mt-2 space-y-1.5">
            {message.citations.map((c, i) => (
              <CitationCard
                key={c.chunk_id}
                citation={c}
                index={i}
                highlighted={highlighted === i}
                cardRef={(el) => {
                  citationRefs.current[i] = el;
                }}
              />
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
        {message.trace && <TracePanel trace={message.trace} />}
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
      let trace: TraceInfo | undefined;
      for await (const evt of parseSSE(res)) {
        if (evt.event === "token") {
          content += (evt.data as { content: string }).content;
          setMessages((prev) => updateLast(prev, { content, pending: true }));
        } else if (evt.event === "sources") {
          citations = (evt.data as { citations: Citation[] }).citations;
        } else if (evt.event === "trace") {
          trace = evt.data as TraceInfo;
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }

      const latencyMs = Math.round(performance.now() - start);
      setMessages((prev) =>
        updateLast(prev, { content, citations, trace, latencyMs, pending: false }),
      );
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
