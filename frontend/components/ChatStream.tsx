"use client";
import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { streamQuery, StreamEvent } from "@/lib/stream";
import Message from "@/components/Message";

type Msg = { role: "user" | "assistant"; content: string; docIds?: string[] };

export default function ChatStream() {
  const { auth } = useAuth();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");

  const send = async () => {
    if (!auth || !input.trim() || busy) return;
    const q = input.trim();
    setMessages((m) => [...m, { role: "user", content: q }]);
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    setInput("");
    setStage("");
    setBusy(true);
    let acc = "";
    setMessages((m) => {
      const next = [...m];
      const last = next[next.length - 1];
      last.content = "…";
      return next;
    });
    const onEvent = (ev: StreamEvent) => {
      if (ev.type === "token") {
        acc += ev.text;
        setMessages((m) => {
          const next = [...m];
          const last = next[next.length - 1];
          last.content = acc;
          return next;
        });
      } else if (ev.type === "status") {
        setStage(String(ev.stage ?? ""));
      } else if (ev.type === "done") {
        setMessages((m) => {
          const next = [...m];
          const last = next[next.length - 1];
          last.content = ev.answer ?? acc;
          last.docIds = ev.doc_ids ?? [];
          return next;
        });
      }
    };
    try {
      await streamQuery(auth.token, q, onEvent);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        last.content = `Error: ${msg}`;
        return next;
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col p-4">
      <div className="flex-1 space-y-3 overflow-y-auto">
        {messages.map((m, i) => (
          <Message key={i} role={m.role} content={m.content} docIds={m.docIds} />
        ))}
      </div>
      {busy && stage ? (
        <div className="pb-1 text-center text-xs uppercase tracking-wide text-slate-400">
          {stage}…
        </div>
      ) : null}
      <div className="flex gap-2 pt-4">
        <input
          className="flex-1 rounded border p-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask a question about your documents…"
        />
        <button className="rounded bg-blue-600 p-2 px-4 text-white" onClick={send} disabled={busy}>
          Send
        </button>
      </div>
    </div>
  );
}
