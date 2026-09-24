export type StreamEvent = {
  type:
    | "status" // stage: guardrails | guard_model | cache | rewrite | filters | retrieve
    | "meta" // model_used surfaced before tokens
    | "token"
    | "faithfulness"
    | "override"
    | "output_warning"
    | "done";
  [key: string]: any;
};

export async function streamQuery(
  token: string,
  query: string,
  onEvent: (ev: StreamEvent) => void
): Promise<void> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002"}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload) continue;
      onEvent(JSON.parse(payload));
    }
  }
}
