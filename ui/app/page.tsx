"use client";

import { useCallback, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
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

function updateAt(prev: ChatMessage[], index: number, patch: Partial<ChatMessage>): ChatMessage[] {
  const next = [...prev];
  const target = next[index];
  if (target) next[index] = { ...target, ...patch };
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

/** Rewrites "[n]" citation markers into markdown links to a "#cite-{index}"
 * fragment so react-markdown parses them as real link nodes we can intercept --
 * markers whose n falls outside the actual citation count (e.g. mid-stream,
 * before the "sources" event has arrived) are left as plain "[n]" text. Pure
 * and ref-free, safe to call during render. */
function citationMarkdownSource(content: string, citationCount: number): string {
  return content.replace(/\[(\d+)\]/g, (marker, numStr: string) => {
    const n = Number(numStr);
    return n >= 1 && n <= citationCount ? `[${marker}](#cite-${n - 1})` : marker;
  });
}

const MARKDOWN_COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 list-disc pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="mb-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => (
    <code className="rounded bg-zinc-200 px-1 py-0.5 text-xs dark:bg-zinc-800">{children}</code>
  ),
  // Citation markers become real <a href="#cite-N"> nodes (via
  // citationMarkdownSource); clicks are handled by delegation on the wrapping
  // div (see Message's handleAnswerClick) rather than a closure here, since
  // that closure would need to read a ref and this component object is passed
  // into ReactMarkdown during render -- a pattern eslint's react-hooks/refs
  // rule flags as unsafe (can't statically prove the ref isn't read synchronously).
  a: ({ href, children }) => {
    const isCite = href?.startsWith("#cite-");
    return (
      <a
        href={href}
        target={isCite ? undefined : "_blank"}
        rel={isCite ? undefined : "noreferrer"}
        className={
          isCite
            ? "border-0 bg-transparent p-0 font-mono text-current underline decoration-dotted underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
            : "underline decoration-dotted underline-offset-2 hover:text-zinc-900 dark:hover:text-zinc-100"
        }
      >
        {children}
      </a>
    );
  },
};

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

function Message({ message, onRetry }: { message: ChatMessage; onRetry: () => void }) {
  const isUser = message.role === "user";
  const citationRefs = useRef<Record<number, HTMLAnchorElement | null>>({});
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const jumpToCitation = useCallback((index: number) => {
    citationRefs.current[index]?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlighted(index);
    window.setTimeout(() => setHighlighted((h) => (h === index ? null : h)), 1500);
  }, []);

  // Event delegation, not a per-link onClick, so MARKDOWN_COMPONENTS' `a`
  // override can stay a plain, ref-free component (see the comment there).
  function handleAnswerClick(e: React.MouseEvent<HTMLDivElement>) {
    const link = (e.target as HTMLElement).closest("a[href^='#cite-']");
    if (!(link instanceof HTMLAnchorElement)) return;
    const match = /^#cite-(\d+)$/.exec(link.getAttribute("href") ?? "");
    if (match) {
      e.preventDefault();
      jumpToCitation(Number(match[1]));
    }
  }

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
          {isUser ? (
            message.content
          ) : (
            <div onClick={handleAnswerClick}>
              <ReactMarkdown components={MARKDOWN_COMPONENTS}>
                {citationMarkdownSource(message.content, message.citations?.length ?? 0)}
              </ReactMarkdown>
            </div>
          )}
          {message.pending && !message.content && (
            <span className="inline-block animate-pulse text-zinc-400">thinking&hellip;</span>
          )}
          {message.error && (
            <div className="text-red-500">
              {message.error}{" "}
              <button
                type="button"
                onClick={onRetry}
                className="underline decoration-dotted underline-offset-2 hover:text-red-600"
              >
                Retry
              </button>
            </div>
          )}
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

  async function streamInto(query: string, index: number) {
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
        setMessages((prev) => updateAt(prev, index, { pending: false, error: message }));
        return;
      }

      let content = "";
      let citations: Citation[] = [];
      let trace: TraceInfo | undefined;
      for await (const evt of parseSSE(res)) {
        if (evt.event === "token") {
          content += (evt.data as { content: string }).content;
          setMessages((prev) => updateAt(prev, index, { content, pending: true }));
        } else if (evt.event === "sources") {
          citations = (evt.data as { citations: Citation[] }).citations;
        } else if (evt.event === "trace") {
          trace = evt.data as TraceInfo;
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }

      const latencyMs = Math.round(performance.now() - start);
      setMessages((prev) =>
        updateAt(prev, index, { content, citations, trace, latencyMs, pending: false }),
      );
    } catch (err) {
      setMessages((prev) =>
        updateAt(prev, index, {
          pending: false,
          error: err instanceof Error ? err.message : "Request failed.",
        }),
      );
    } finally {
      setIsStreaming(false);
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }

  async function runQuery(query: string) {
    if (!query || isStreaming) return;
    setInput("");
    const index = messages.length + 1; // user msg lands at messages.length, assistant right after
    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", pending: true },
    ]);
    await streamInto(query, index);
  }

  async function retryMessage(index: number) {
    if (isStreaming) return;
    const query = messages[index - 1]?.content;
    if (!query) return;
    setMessages((prev) =>
      updateAt(prev, index, {
        content: "",
        citations: undefined,
        trace: undefined,
        latencyMs: undefined,
        error: undefined,
        pending: true,
      }),
    );
    await streamInto(query, index);
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
          <Message key={i} message={m} onRetry={() => void retryMessage(i)} />
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
