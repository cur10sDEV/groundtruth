"use client";
import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { API_URL } from "@/lib/api";

type SignResponse = {
  doc_id: string;
  status: string;
  new_version?: number;
  upload: { url: string; fields: Record<string, string> };
};

export default function UploadButton({ onUploaded }: { onUploaded: () => void }) {
  const { auth } = useAuth();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const upload = async (file: File) => {
    if (!auth) return;
    setBusy(true);
    setErr("");
    try {
      const signRes = await fetch(`${API_URL}/documents/sign`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${auth.token}` },
        body: JSON.stringify({ filename: file.name }),
      });
      if (!signRes.ok) {
        const data = await signRes.json().catch(() => ({}));
        setErr(String(data.error ?? "Upload request failed"));
        return;
      }
      const { upload: target } = (await signRes.json()) as SignResponse;
      const form = new FormData();
      Object.entries(target.fields).forEach(([k, v]) => form.append(k, v));
      form.append("file", file);
      const res = await fetch(target.url, { method: "POST", body: form });
      if (!res.ok) {
        setErr(`Upload failed (HTTP ${res.status})`);
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
