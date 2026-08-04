/**
 * Minimal SSE parser for a fetch() POST response body.
 *
 * The browser's native EventSource can't send a POST body, so /query's
 * `event: <name>\ndata: <json>\n\n` stream (api/routes.py's `_sse()`) has to
 * be parsed by hand from the raw ReadableStream instead.
 */

export interface SSEEvent {
  event: string;
  data: unknown;
}

export async function* parseSSE(response: Response): AsyncGenerator<SSEEvent> {
  if (!response.body) {
    throw new Error("Response has no body to stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const eventLine = block.split("\n").find((l) => l.startsWith("event: "));
        const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
        if (eventLine && dataLine) {
          yield {
            event: eventLine.slice("event: ".length),
            data: JSON.parse(dataLine.slice("data: ".length)),
          };
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}
