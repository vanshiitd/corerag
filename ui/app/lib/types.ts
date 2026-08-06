/** Mirrors core/retrieval.py's ScoredChunk (the shape the /query "sources"
 * SSE event serializes via Pydantic's .model_dump()). */
export interface Citation {
  chunk_id: string;
  arxiv_id: string;
  title: string;
  authors: string[];
  abs_url: string;
  section: string | null;
  index: number;
  text: string;
  context: string | null;
  score: number;
}

/** Mirrors the /query "trace" SSE event: which path the agent graph took to
 * produce this answer (cache hit vs. a real run, the router's decision,
 * how many reflection retries it took, and whether it gave up low-confidence). */
export interface TraceInfo {
  cached: boolean;
  route: "simple" | "multi_hop" | null;
  retries: number | null;
  low_confidence: boolean | null;
  elapsed_ms: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  /** Wall-clock ms from submit to the final `sources` event -- a real,
   * per-answer trace/latency readout, not a simulated number. */
  latencyMs?: number;
  /** Server-reported routing/reflection path for this answer. */
  trace?: TraceInfo;
  /** True while a response is still streaming in. */
  pending?: boolean;
  error?: string;
}
