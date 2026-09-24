"use client";
import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { API_URL } from "@/lib/api";

export default function UploadButton({ onUploaded }: { onUploaded: () => void }) {
  const { auth } = useAuth();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const upload = async (file: File) => {
    if (!auth) return;
    setBusy(true);
    setErr("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${auth.token}` },
        body: form,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setErr(String(data.error ?? "Upload failed"));
        return;
      }
      onUploaded();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf,.docx,.md,.txt"
        disabled={busy}
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) upload(f);
        }}
      />
      {err && <p className="text-red-600">{err}</p>}
    </div>
  );
}
