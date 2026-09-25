# Next.js 16 Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the frontend to next@16.3.6 + React 19 + Tailwind v4, adopting the React 19 idioms that honestly fit, with one isolated commit per axis.

**Architecture:** Small client-component App Router app (5 pages, 4 components, 4 lib files) talking directly to FastAPI with localStorage JWT auth. Platform jump first, idiom refactors second, Tailwind v4 last — ordered so any build failure has exactly one suspect. No test harness exists; verification is tsc + `npm run build` + dev-boot smoke per task.

**Tech Stack:** Next.js 16.3.6 (Turbopack), React 19, Tailwind CSS 4, TypeScript 5.6 (unchanged).

**Spec:** `docs/superpowers/specs/2026-09-25-nextjs16-upgrade-design.md`

## Global Constraints

- Frontend only — do not touch `backend/`, `infra/`, or anything outside `frontend/` (plus one README line).
- Node floor: 20.9+ (host has 24.15.0). Do not change Node tooling.
- Version pins: `next@16.3.6` exact; `react@19`, `react-dom@19`, `@types/react@19`, `@types/react-dom@19`. TypeScript stays 5.6.x; `@types/node` stays 26.x; `@flagsmith/flagsmith` stays 12.4.0.
- Do NOT add ESLint, a test harness, or any package beyond the ones listed.
- ChatStream.tsx and UploadButton.tsx must NOT be refactored (spec decision — no shoehorning React 19 hooks into the streaming/upload flows).
- Per-task gates, all from `frontend/`: `npx tsc --noEmit` → clean; `npm run build` → success; dev-boot smoke → HTTP 200.
- If a gate fails, fix within the task before committing; if Turbopack build fails inexplicably, try `next build --webpack` to isolate, but do not make webpack the permanent config without recording it in the task report.
- Three commits, one per task, in order. Each must leave the tree green.
- RabbitMQ may be running in docker — leave it. Do not start long-lived services without killing them before task end.
- Don't overthink. When a step says "expected: clean," run it and trust the output.

---

### Task 1: Platform jump — next@16.3.6 + React 19

**Files:**
- Modify: `frontend/package.json` (dependency bumps + remove `lint` script)
- Modify: `frontend/package-lock.json` (via npm)
- Modify: `frontend/README`-facing file: repo root `README.md` (one line, node floor)
- Possibly auto-modified by Next on first build: `frontend/tsconfig.json`, `frontend/next-env.d.ts` — commit whatever Next writes.

**Interfaces:**
- Consumes: nothing new.
- Produces: the app running on next@16.3.6 / react@19 with identical behavior; gates still `npx tsc --noEmit` + `npm run build`.

- [ ] **Step 1: Install the new versions**

```bash
npm install next@16.3.6 react@19 react-dom@19 && npm install -D @types/react@19 @types/react-dom@19
```

Expected: clean install, no peer-dep errors (`@flagsmith/flagsmith` is vanilla JS — no React peer).

- [ ] **Step 2: Remove the dead lint script**

In `frontend/package.json`, delete this line:

```json
    "lint": "next lint",
```

(The command no longer exists in Next 16 and never worked here — no ESLint is installed. Do not add a replacement.)

- [ ] **Step 3: Run tsc**

```bash
npx tsc --noEmit
```

Expected: clean. Known-compatible facts (verified at plan time): the only `useRef` in the app (`DocumentList.tsx:38`) already passes `null` as its initial value; no `React.FC`, no `forwardRef`, no `defaultProps`. If anything still fails, fix it mechanically (type annotations only — no refactoring) and re-run until clean. Record what (if anything) needed fixing in the task report.

- [ ] **Step 4: Build (first Turbopack build)**

```bash
npm run build
```

Expected: success. Note the build duration Next prints for the report (this is the platform win the upgrade is buying). If Next rewrites `tsconfig.json`/`next-env.d.ts`, keep the changes.

- [ ] **Step 5: Dev-boot smoke**

```bash
npm run dev &
sleep 8
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login
kill %1
```

Expected: `200` and `200`. Then confirm no stray process: `pgrep -f "next dev" || echo none`.

- [ ] **Step 6: README node floor**

In repo root `README.md`, find the frontend runbook line (currently reads `# 4. frontend (node 18+)`) and change it to:

```markdown
# 4. frontend (node 20.9+)
```

- [ ] **Step 7: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/tsconfig.json frontend/next-env.d.ts README.md
git commit -m "feat(frontend): next 16 + react 19 platform jump"
```

---

### Task 2: React 19 idioms — useActionState forms + context-as-provider

**Files:**
- Rewrite: `frontend/app/login/page.tsx`
- Rewrite: `frontend/app/signup/page.tsx`
- Modify: `frontend/lib/auth.tsx` (provider render, lines 47-51)
- Modify: `frontend/lib/flags.tsx` (provider render, line ~75)

**Interfaces:**
- Consumes: `useAuth()` context shape from `lib/auth.tsx` — unchanged: `{ auth, login, signup, logout }`. `login(email, password) => Promise<void>`; `signup(email, password, org_name) => Promise<void>` (unchanged signatures).
- Produces: same routes and post-login navigation (`router.push("/chat")`); uncontrolled forms posting via React actions. `AuthContext` value shape unchanged — no consumer updates needed. Backend API paths unchanged: `POST /auth/login` `{email, password}`, `POST /auth/signup` `{email, password, org_name}`.

- [ ] **Step 1: Rewrite the login page**

Replace the entire contents of `frontend/app/login/page.tsx` with:

```tsx
"use client";
import { useActionState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [err, formAction, isPending] = useActionState(
    async (_prev: string | null, formData: FormData) => {
      try {
        await login(String(formData.get("email")), String(formData.get("password")));
        router.push("/chat");
        return null;
      } catch (error) {
        return error instanceof Error ? error.message : "Login failed";
      }
    },
    null
  );

  return (
    <form action={formAction} className="mx-auto mt-24 max-w-sm space-y-4 rounded border border-slate-300 p-6">
      <h1 className="text-xl font-semibold">Sign in</h1>
      {err && <p className="text-red-600">{err}</p>}
      <input className="w-full border border-slate-300 p-2" name="email" placeholder="email" />
      <input className="w-full border border-slate-300 p-2" name="password" type="password" placeholder="password" />
      <button className="w-full rounded bg-blue-600 p-2 text-white" disabled={isPending}>
        {isPending ? "Signing in…" : "Login"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Rewrite the signup page**

Replace the entire contents of `frontend/app/signup/page.tsx` with:

```tsx
"use client";
import { useActionState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function SignupPage() {
  const { signup } = useAuth();
  const router = useRouter();
  const [err, formAction, isPending] = useActionState(
    async (_prev: string | null, formData: FormData) => {
      try {
        await signup(
          String(formData.get("email")),
          String(formData.get("password")),
          String(formData.get("org_name"))
        );
        router.push("/chat");
        return null;
      } catch (error) {
        return error instanceof Error ? error.message : "Signup failed";
      }
    },
    null
  );

  return (
    <form action={formAction} className="mx-auto mt-24 max-w-sm space-y-4 rounded border border-slate-300 p-6">
      <h1 className="text-xl font-semibold">Create account</h1>
      {err && <p className="text-red-600">{err}</p>}
      <input className="w-full border border-slate-300 p-2" name="email" placeholder="email" />
      <input className="w-full border border-slate-300 p-2" name="password" type="password" placeholder="password" />
      <input className="w-full border border-slate-300 p-2" name="org_name" placeholder="organization name" />
      <button className="w-full rounded bg-blue-600 p-2 text-white" disabled={isPending}>
        {isPending ? "Creating…" : "Sign up"}
      </button>
    </form>
  );
}
```

(Note: these two forms already use explicit `border-slate-300` — valid in Tailwind 3.4, and it front-loads part of Task 3's border fix for the files Task 2 rewrites anyway.)

- [ ] **Step 3: Context-as-provider in auth.tsx**

In `frontend/lib/auth.tsx`, replace the return block:

```tsx
  return (
    <AuthContext.Provider value={{ auth, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
```

with:

```tsx
  return <AuthContext value={{ auth, login, signup, logout }}>{children}</AuthContext>;
```

- [ ] **Step 4: Context-as-provider in flags.tsx**

In `frontend/lib/flags.tsx`, replace:

```tsx
  return <FlagsContext.Provider value={flags}>{children}</FlagsContext.Provider>;
```

with:

```tsx
  return <FlagsContext value={flags}>{children}</FlagsContext>;
```

- [ ] **Step 5: Gates — tsc + build**

```bash
npx tsc --noEmit && npm run build
```

Expected: both clean. (If tsc flags the `value` prop on context — that would mean React 19 types did not install correctly in Task 1; verify `@types/react` is 19.x before proceeding.)

- [ ] **Step 6: Live smoke of the auth path (API level)**

The forms' browser click-through cannot be automated without adding e2e tooling (out of scope), so verify the exact backend path the actions call, with the dev server up:

```bash
docker compose --project-name rag -f infra/docker-compose.yml up -d postgres redis
(cd backend && ./.venv/bin/python -m uvicorn app.main:app --port 8002) &
(cd frontend && npm run dev) &
sleep 10
curl -s -X POST http://localhost:8002/auth/signup -H "Content-Type: application/json" -d '{"email":"smoke-task2@example.com","password":"pw123456","org_name":"Smoke Co"}'
curl -s -X POST http://localhost:8002/auth/login -H "Content-Type: application/json" -d '{"email":"smoke-task2@example.com","password":"pw123456"}'
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login
kill %1 %2
docker compose --project-name rag -f infra/docker-compose.yml stop postgres redis
```

Expected: signup returns a token JSON; login returns a token JSON; the page returns `200`. Leave no processes running (`pgrep -f "uvicorn|next dev" || echo none`).

- [ ] **Step 7: Commit**

```bash
git add frontend/app/login/page.tsx frontend/app/signup/page.tsx frontend/lib/auth.tsx frontend/lib/flags.tsx
git commit -m "refactor(frontend): react 19 idioms — useActionState auth forms, context providers"
```

---

### Task 3: Tailwind 3.4 → 4

**Files:**
- Modify: `frontend/package.json` + `frontend/package-lock.json` (via npm)
- Rewrite: `frontend/postcss.config.mjs`
- Modify: `frontend/app/globals.css`
- Delete: `frontend/tailwind.config.ts`
- Modify: `frontend/components/Message.tsx` (2 bare borders), `frontend/components/DocumentList.tsx` (1), `frontend/components/ChatStream.tsx` (1)

**Interfaces:**
- Consumes: Task 1/2 state (app on React 19; login/signup already carry explicit `border-slate-300`).
- Produces: Tailwind v4 build with visually identical pages.

- [ ] **Step 1: Swap packages**

```bash
npm install tailwindcss@4 @tailwindcss/postcss && npm uninstall autoprefixer
```

Expected: clean install.

- [ ] **Step 2: Rewrite postcss.config.mjs**

Replace the entire file with:

```js
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
```

- [ ] **Step 3: Rewrite globals.css**

Replace the three directive lines (`@tailwind base;`, `@tailwind components;`, `@tailwind utilities;`) with the single line:

```css
@import "tailwindcss";
```

(Keep anything else in the file, if present.)

- [ ] **Step 4: Delete tailwind.config.ts**

```bash
git rm frontend/tailwind.config.ts
```

(v4 auto-detects content; the config was minimal.)

- [ ] **Step 5: Fix the remaining bare borders**

v4 changes bare `border` from gray to `currentColor` — every bare usage must get an explicit color or borders turn near-black. Four remain (the auth forms were fixed in Task 2):

`frontend/components/Message.tsx` — the assistant bubble, currently:

```tsx
          (role === "user" ? "bg-blue-100" : "bg-white border")
```

becomes:

```tsx
          (role === "user" ? "bg-blue-100" : "bg-white border border-slate-300")
```

and the citations panel, currently `className="mt-1 max-w-[80%] space-y-2 rounded border bg-slate-50 p-2 text-left text-xs text-slate-600"` becomes the same string with `rounded border border-slate-300 bg-slate-50`.

`frontend/components/DocumentList.tsx` — the list item, currently `className="flex items-center justify-between rounded border p-2"` becomes `"flex items-center justify-between rounded border border-slate-300 p-2"`. (The delete button's `border-red-300` is already explicit — leave it.)

`frontend/components/ChatStream.tsx` — the chat input, currently `className="flex-1 rounded border p-2"` becomes `"flex-1 rounded border border-slate-300 p-2"`.

- [ ] **Step 6: Sweep — confirm zero bare borders remain**

```bash
grep -rn 'border' frontend/app frontend/components --include='*.tsx' | grep -v 'border-' | grep -v 'border:' || echo "no bare borders"
```

Expected: `no bare borders`. (The grep intentionally lists every line containing the word `border` without a modifier — the only acceptable survivors are `border-red-300`-style tokens, which the `border-` filter excludes.)

- [ ] **Step 7: Gates — tsc + build**

```bash
npx tsc --noEmit && npm run build
```

Expected: clean. Then confirm the built CSS actually contains the fixed class:

```bash
grep -l "border-slate-300" .next/static/css/*.css && echo "border color compiled"
```

Expected: a file path + `border color compiled`.

- [ ] **Step 8: Dev-boot spot-check**

```bash
npm run dev &
sleep 8
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login
kill %1
pgrep -f "next dev" || echo none
```

Expected: `200`, then `none`.

- [ ] **Step 9: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/postcss.config.mjs frontend/app/globals.css frontend/components/Message.tsx frontend/components/DocumentList.tsx frontend/components/ChatStream.tsx
git commit -m "feat(frontend): tailwind v4"
```

---

## Post-plan note for the controller

After Task 3 commits, remind the user that the two auth forms have not been human-clicked (no browser automation exists) — a 60-second manual pass over login + signup + chat in the dev server is the final acceptance, and the runbook's `npm run dev` is all it takes. Parked items (server-first restructure, ESLint) live in the spec's Decisions section and the SDD ledger.
