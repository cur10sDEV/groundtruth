"use client";
import { useState } from "react";
import { apiFetch } from "@/lib/api";

type Citation = {
  chunk_id: string;
  doc_id: string;
  text: string;
  start_offset: number;
  end_offset: number;
};

export default function Message({
  role,
  content,
  docIds = [],
  queryId,
  token,
}: {
  role: "user" | "assistant";
  content: string;
  docIds?: string[];
  queryId?: string;
  token?: string;
}) {
  const [cites, setCites] = useState<Citation[] | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const clickable = role === "assistant" && !!queryId && !!token;

  const toggle = async () => {
    if (!clickable || busy) return;
    if (open) {
      setOpen(false);
      return;
    }
    if (!cites) {
      setBusy(true);
      setErr("");
      try {
        const data = await apiFetch<Citation[]>(`/query/${queryId}/citations`, { token });
        setCites(data);
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Citations unavailable");
        return;
      } finally {
        setBusy(false);
      }
    }
    setOpen(true);
  };

  const parts = content.split(/(\[\d+\])/g);
  return (
    <div className={role === "user" ? "text-right" : ""}>
      <div
        className={
          "inline-block max-w-[80%] whitespace-pre-wrap rounded p-3 " +
          (role === "user" ? "bg-blue-100" : "bg-white border")
        }
      >
        {parts.map((part, i) => {
          if (!/^\[\d+\]$/.test(part)) return part;
          const label = part.slice(1, -1);
          if (!clickable) {
            return (
              <sup
                key={i}
                className="rounded bg-slate-200 px-1 text-[10px] font-semibold text-slate-700"
              >
                {label}
              </sup>
            );
          }
          return (
            <button
              key={i}
              onClick={toggle}
              disabled={busy}
              title="View cited chunks"
              className="rounded bg-slate-200 px-1 text-[10px] font-semibold text-slate-700 hover:bg-slate-300 disabled:opacity-50"
            >
              {label}
            </button>
          );
        })}
      </div>
      {err && <div className="mt-1 max-w-[80%] text-xs text-red-600">{err}</div>}
      {open && cites && (
        <div className="mt-1 max-w-[80%] space-y-2 rounded border bg-slate-50 p-2 text-left text-xs text-slate-600">
          {cites.length === 0 && <div>No citations stored for this answer.</div>}
          {cites.map((c, i) => (
            <div key={c.chunk_id}>
              <span className="font-semibold">
                [{i + 1}] {c.doc_id}
              </span>
              <p className="mt-0.5 whitespace-pre-wrap">{c.text}</p>
            </div>
          ))}
        </div>
      )}
      {role === "assistant" && docIds.length > 0 && (
        <div className="mt-1 max-w-[80%] truncate text-xs text-slate-500">
          Sources: {docIds.join(", ")}
        </div>
      )}
    </div>
  );
}
