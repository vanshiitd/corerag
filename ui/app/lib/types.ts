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

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  /** Wall-clock ms from submit to the final `sources` event -- a real,
   * per-answer trace/latency readout, not a simulated number. */
  latencyMs?: number;
  /** True while a response is still streaming in. */
  pending?: boolean;
  error?: string;
}
