"use client";
import { useEffect, useRef, useState } from "react";
import { useAuth } from "@/lib/auth";
import { apiFetch } from "@/lib/api";

type Doc = { id: string; filename: string; status: string; version: number };

const TERMINAL = ["EMBEDDED", "FAILED"];

export default function DocumentList({ refresh }: { refresh: number }) {
  const { auth } = useAuth();
  const [docs, setDocs] = useState<Doc[]>([]);
  const [err, setErr] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!auth) return;
    const load = async () => {
      try {
        const data = await apiFetch<Doc[]>("/documents", { token: auth.token });
        setDocs(data);
        if (data.length > 0 && data.every((d) => TERMINAL.includes(d.status))) {
          if (timer.current) {
            clearInterval(timer.current);
            timer.current = null;
          }
        }
      } catch {
        // transient fetch failure; next poll or refresh retries
      }
    };
    load();
    timer.current = setInterval(load, 5000);
    return () => {
      if (timer.current) clearInterval(timer.current);
      timer.current = null;
    };
  }, [auth, refresh, reload]);

  const remove = async (id: string) => {
    if (!auth || deleting === id) return;
    setDeleting(id);
    setErr("");
    try {
      await apiFetch(`/documents/${id}`, { method: "DELETE", token: auth.token });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(null);
      setReload((r) => r + 1);
    }
  };

  return (
    <div className="space-y-2">
      {err && <p className="text-red-600">{err}</p>}
      <ul className="space-y-2">
        {docs.map((d) => (
          <li key={d.id} className="flex items-center justify-between rounded border p-2">
            <span>{d.filename}</span>
            <span>{d.status}</span>
            <span>v{d.version}</span>
            <button
              className="rounded border border-red-300 px-2 text-sm text-red-600 disabled:opacity-50"
              onClick={() => remove(d.id)}
              disabled={deleting === d.id}
            >
              {deleting === d.id ? "Deleting…" : "Delete"}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
