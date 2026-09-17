# Phase 7 — Frontend (Next.js)

**Goal:** Build the Next.js app: auth (login/signup), chat UI consuming the SSE query stream,
document upload + status polling, and citation display. Wire `@flagsmith/react` for live UI flags.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (sections 3, 6, UI/UX)

## Dependencies

Phase 4 (API routes), Phase 5 (Flagsmith wiring note). The backend API runs at
`http://localhost:8002`.

---

### Task 7.1: Scaffold Next.js app + API client + auth pages

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.mjs`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/page.tsx` (redirect to `/login` if no token)
- Create: `frontend/app/login/page.tsx`
- Create: `frontend/app/signup/page.tsx`
- Create: `frontend/lib/api.ts`
- Create: `frontend/lib/auth.tsx`
- Create: `frontend/.env.example`

**Interfaces:**
- Produces:
  - `lib/api.ts`: `apiFetch(path, {token, method, body})` returning parsed JSON; `API_URL` from
    `NEXT_PUBLIC_API_URL` (default `http://localhost:8002`).
  - `lib/auth.tsx`: `AuthProvider` (React context) storing `token`, `user_id`, `org_id` in
    localStorage; `useAuth()` hook; `login(email, password)`, `signup(email, password, org_name)`.
  - Login/signup pages that call the backend `/auth/login` and `/auth/signup` then store the token.

- [ ] **Step 1: Write package.json and configs**

`frontend/package.json`:
```json
{
  "name": "rag-prod-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.15",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "tailwindcss": "^3.4.13",
    "postcss": "^8.4.47",
    "autoprefixer": "^10.4.20",
    "@flagsmith/react": "^1.0.3",
    "flagsmith": "^3.8.0"
  },
  "devDependencies": {
    "typescript": "^5.6.3",
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.0"
  }
}
```

`frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`frontend/next.config.mjs`:
```js
/** @type {import('next').NextConfig} */
const nextConfig = { reactStrictMode: true };
export default nextConfig;
```

`frontend/.env.example`:
```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8002
NEXT_PUBLIC_FLAGSMITH_KEY=
```

- [ ] **Step 2: Write api client and auth context**

`frontend/lib/api.ts`:
```ts
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

export async function apiFetch<T>(
  path: string,
  opts: { token?: string; method?: string; body?: unknown } = {}
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: opts.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
      ...(opts.token ? { Authorization: `Bearer ${opts.token}` } : {}),
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}
```

`frontend/lib/auth.tsx`:
```tsx
"use client";
import { createContext, useContext, useState, ReactNode } from "react";
import { apiFetch } from "./api";

type AuthState = { token: string; user_id: string; org_id: string } | null;

const AuthContext = createContext<{
  auth: AuthState;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, org_name: string) => Promise<void>;
  logout: () => void;
}>({ auth: null, login: async () => {}, signup: async () => {}, logout: () => {} });

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<AuthState>(() => {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem("rag-auth");
    return raw ? JSON.parse(raw) : null;
  });

  const store = (a: AuthState) => {
    setAuth(a);
    localStorage.setItem("rag-auth", JSON.stringify(a));
  };

  const login = async (email: string, password: string) => {
    const data = await apiFetch<{ token: string; user_id: string; org_id: string }>(
      "/auth/login",
      { method: "POST", body: { email, password } }
    );
    store(data);
  };

  const signup = async (email: string, password: string, org_name: string) => {
    const data = await apiFetch<{ token: string; user_id: string; org_id: string }>(
      "/auth/signup",
      { method: "POST", body: { email, password, org_name } }
    );
    store(data);
  };

  const logout = () => {
    setAuth(null);
    localStorage.removeItem("rag-auth");
  };

  return (
    <AuthContext.Provider value={{ auth, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
```

- [ ] **Step 3: Write layout, login, signup pages**

`frontend/app/layout.tsx`:
```tsx
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { FlagsmithProvider } from "@/lib/flags";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
```

`frontend/app/login/page.tsx` (client):
```tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login(email, password);
      router.push("/chat");
    } catch (error: any) {
      setErr(error.message);
    }
  };

  return (
    <form onSubmit={submit} className="mx-auto mt-24 max-w-sm space-y-4 rounded border p-6">
      <h1 className="text-xl font-semibold">Sign in</h1>
      {err && <p className="text-red-600">{err}</p>}
      <input className="w-full border p-2" placeholder="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input className="w-full border p-2" type="password" placeholder="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <button className="w-full rounded bg-blue-600 p-2 text-white">Login</button>
    </form>
  );
}
```

`frontend/app/signup/page.tsx`: mirror login, call `signup(email, password, org_name)` with an extra
org-name field, then `router.push("/chat")`.

- [ ] **Step 4: Install and build**

Run: `cd frontend && npm install && npm run build`
Expected: build succeeds without type errors.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat(frontend): scaffold Next.js app with auth pages and api client"
```

---

### Task 7.2: Chat UI with SSE streaming

**Files:**
- Create: `frontend/app/chat/page.tsx`
- Create: `frontend/components/ChatStream.tsx`
- Create: `frontend/components/Message.tsx`
- Create: `frontend/lib/stream.ts`

**Interfaces:**
- Produces:
  - `lib/stream.ts`: `streamQuery(token, query, onEvent)` — POSTs `/query` and reads the SSE body
    (via `fetch` + `ReadableStream`), parses `data:` lines into events, calls `onEvent(ev)`.
  - `ChatStream` component: input box + message list; on send, calls `streamQuery` and appends
    `status`/`token`/`faithfulness`/`done` events; renders markdown-ish answer + citations.
  - `Message` component: renders one message (user or assistant) with citation markers `[n]`.

- [ ] **Step 1: Write the SSE reader**

`frontend/lib/stream.ts`:
```ts
export type StreamEvent = {
  type: "status" | "token" | "faithfulness" | "override" | "done";
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
```

- [ ] **Step 2: Write the chat components**

`frontend/components/ChatStream.tsx`:
```tsx
"use client";
import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { streamQuery, StreamEvent } from "@/lib/stream";

type Msg = { role: "user" | "assistant"; content: string };

export default function ChatStream() {
  const { auth } = useAuth();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const send = async () => {
    if (!auth || !input.trim() || busy) return;
    const q = input.trim();
    setMessages((m) => [...m, { role: "user", content: q }]);
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    setInput("");
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
      } else if (ev.type === "done") {
        setMessages((m) => {
          const next = [...m];
          const last = next[next.length - 1];
          last.content = ev.answer ?? acc;
          return next;
        });
      }
    };
    await streamQuery(auth.token, q, onEvent);
    setBusy(false);
  };

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col p-4">
      <div className="flex-1 space-y-3 overflow-y-auto">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : ""}>
            <div
              className={
                "inline-block max-w-[80%] whitespace-pre-wrap rounded p-3 " +
                (m.role === "user" ? "bg-blue-100" : "bg-white border")
              }
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>
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
```

`frontend/app/chat/page.tsx`:
```tsx
"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import ChatStream from "@/components/ChatStream";

export default function ChatPage() {
  const { auth } = useAuth();
  const router = useRouter();
  useEffect(() => {
    if (!auth) router.replace("/login");
  }, [auth, router]);
  if (!auth) return null;
  return <ChatStream />;
}
```

- [ ] **Step 3: Add globals.css**

`frontend/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 4: Build to verify**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/app frontend/components frontend/lib/stream.ts
git commit -m "feat(frontend): add chat UI with SSE streaming"
```

---

### Task 7.3: Document upload + status + citations

**Files:**
- Create: `frontend/app/documents/page.tsx`
- Create: `frontend/components/UploadButton.tsx`
- Create: `frontend/components/DocumentList.tsx`

**Interfaces:**
- Produces:
  - `UploadButton`: file input → `POST /documents/upload` with `multipart/form-data` using the token.
  - `DocumentList`: `GET /documents`, polls every 5s until status is `EMBEDDED`/`FAILED`; shows
    filename, status, version; `DELETE` button.
  - Documents page linking to chat.

- [ ] **Step 1: Write UploadButton**

`frontend/components/UploadButton.tsx`:
```tsx
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
    const res = await fetch(`${API_URL}/documents/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${auth.token}` },
      body: form,
    });
    if (!res.ok) {
      setErr("Upload failed");
      setBusy(false);
      return;
    }
    setBusy(false);
    onUploaded();
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf,.docx,.md,.txt"
        disabled={busy}
        onChange={(e) => e.target.files && upload(e.target.files[0])}
      />
      {err && <p className="text-red-600">{err}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Write DocumentList + page**

`frontend/components/DocumentList.tsx`:
```tsx
"use client";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { apiFetch } from "@/lib/api";

type Doc = { id: string; filename: string; status: string; version: number };

export default function DocumentList({ refresh }: { refresh: number }) {
  const { auth } = useAuth();
  const [docs, setDocs] = useState<Doc[]>([]);

  useEffect(() => {
    if (!auth) return;
    const load = async () => {
      const data = await apiFetch<Doc[]>("/documents", { token: auth.token });
      setDocs(data);
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [auth, refresh]);

  return (
    <ul className="space-y-2">
      {docs.map((d) => (
        <li key={d.id} className="flex items-center justify-between rounded border p-2">
          <span>{d.filename}</span>
          <span>{d.status}</span>
          <span>v{d.version}</span>
        </li>
      ))}
    </ul>
  );
}
```

`frontend/app/documents/page.tsx`:
```tsx
"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import UploadButton from "@/components/UploadButton";
import DocumentList from "@/components/DocumentList";

export default function DocumentsPage() {
  const { auth } = useAuth();
  const router = useRouter();
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    if (!auth) router.replace("/login");
  }, [auth, router]);
  if (!auth) return null;
  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Documents</h1>
      <UploadButton onUploaded={() => setRefresh((r) => r + 1)} />
      <DocumentList refresh={refresh} />
    </div>
  );
}
```

- [ ] **Step 3: Build to verify**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/documents frontend/components/UploadButton.tsx frontend/components/DocumentList.tsx
git commit -m "feat(frontend): add document upload and status list with polling"
```

---

### Task 7.4: Flagsmith provider for the frontend

**Files:**
- Create: `frontend/lib/flags.tsx`
- Modify: `frontend/app/layout.tsx`

**Interfaces:**
- Produces `lib/flags.tsx` exporting `FlagsmithProvider` (wrapping `@flagsmith/react`'s provider,
  initialized with `NEXT_PUBLIC_FLAGSMITH_KEY`), so the UI can toggle flag-dependent behavior live.

- [ ] **Step 1: Write the flags provider**

`frontend/lib/flags.tsx`:
```tsx
"use client";
import { FlagsmithProvider as FSProvider } from "@flagsmith/react";
import flagsmith from "flagsmith";
import { ReactNode } from "react";

export function FlagsmithProvider({ children }: { children: ReactNode }) {
  const key = process.env.NEXT_PUBLIC_FLAGSMITH_KEY;
  if (!key) return <>{children}</>;
  return (
    <FSProvider flagsmith={flagsmith} options={{ environmentKey: key }}>
      {children}
    </FSProvider>
  );
}
```

- [ ] **Step 2: Wire into layout**

Wrap `{children}` inside `AuthProvider` with `<FlagsmithProvider>{children}</FlagsmithProvider>` in
`frontend/app/layout.tsx`.

- [ ] **Step 3: Build to verify**

Run: `cd frontend && npm run build`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/flags.tsx frontend/app/layout.tsx
git commit -m "feat(frontend): add Flagsmith provider for live UI flags"
```

---

**Phase 7 exit check:** with backend running on `:8002`, a user can sign up, log in, upload a
document, see it ingest to `EMBEDDED`, and ask a question in chat that streams an answer with
citations. `cd frontend && npm run build` succeeds.