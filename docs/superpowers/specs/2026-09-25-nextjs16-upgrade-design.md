# Next.js 16 Upgrade — Design Spec

**Date:** 2026-09-25
**Status:** Approved design → implementation planning
**Scope:** Frontend only (`frontend/`)

## Context

The frontend is a small client-component App Router app (5 pages, 4 components, 4 lib
files) on `next@14.2.35` / `react@18.3.1` / `tailwindcss@3.4.13`. It talks directly to
the FastAPI backend with a localStorage JWT; all pages are client components. It runs
on the host (`npm run dev`), not in compose. Node on the machine is v24 (Next 16
requires >= 20.9). `@flagsmith/flagsmith@12.4.0` is already the latest published
version and is vanilla JS (no React peer-dep risk).

Next.js latest stable is 16.3.6. Verified against the v16 upgrade guide, the major
breaking changes that do **not** touch this app: async request APIs (no dynamic
routes, no route handlers, no `cookies()`/`headers()` usage), `middleware`→`proxy`
rename (no middleware), fetch caching default changes (all fetching is client-side).
What does touch us: React 19 is required, Turbopack becomes the default bundler for
dev and build, and the `next lint` command is removed.

The owner chose: **platform jump + React 19 idioms where they honestly fit + Tailwind
v4 as a final isolated task.** The full server-first restructure (cookie sessions,
BFF proxy, server components) was reviewed and deliberately **parked** — to be
spec'd together with the future org-level functionality work (see Decisions).

## Tasks

### Task 1 — Platform jump (next@16.3.6, react@19)

- Install: `next@16.3.6`, `react@19`, `react-dom@19`, `@types/react@19`,
  `@types/react-dom@19`. TypeScript stays at 5.6.x.
- Remove the `"lint": "next lint"` script from `package.json` (never functional — no
  ESLint is installed — and the command no longer exists in Next 16). Do not add
  ESLint in this project; that is a separate future decision.
- `next.config.mjs` and `tsconfig.json` unchanged; Next 16 adjusts what it needs on
  first build.
- README runbook: update the frontend prerequisite from "node 18+" to "node 20.9+".
- React 19 types are stricter; expect a small number of mechanical fixes surfaced by
  `tsc` (known example: `useRef` now requires an initial value — one usage in
  `DocumentList.tsx`). Fix what surfaces; no proactive refactoring beyond it.
- Verify: `npx tsc --noEmit` + `npm run build` (Turbopack) + `next dev` boot with
  HTTP 200 checks on `/` and `/login`.

### Task 2 — React 19 idioms (only where they honestly fit)

- **`useActionState` for the login and signup forms** (`app/login/page.tsx`,
  `app/signup/page.tsx`): rebuild as uncontrolled `<form action={formAction}>` with
  the action signature `(prevState, formData) => state`; built-in `isPending`
  replaces manual submit state; error text flows through the returned state.
  Navigation after success (`router.push`) stays inside the action.
- **Context-as-provider** in `lib/auth.tsx` and `lib/flags.tsx`: render
  `<Context value={...}>` directly (React 19) instead of `<Context.Provider>`.
- **ChatStream and UploadButton are deliberately unchanged.** The chat is a custom
  streaming protocol whose append-then-mutate flow already behaves as
  `useOptimistic` would; the upload is a multi-step signed-fetch flow that
  `useActionState` does not fit. Shoehorning the hooks there adds complexity and
  bug surface with no behavioral gain.
- Verify: same gates as Task 1. Manual UI sanity of both auth pages is expected as
  part of the task (form submit path is a live-only flow; the implementer boots the
  dev server and exercises the two forms against the real backend).

### Task 3 — Tailwind 3.4 → 4 (isolated, last)

- Install `tailwindcss@4` + `@tailwindcss/postcss`; drop `autoprefixer` (built in).
- `postcss.config.mjs` becomes `plugins: { "@tailwindcss/postcss": {} }`.
- `globals.css`: replace the three `@tailwind` directives with `@import "tailwindcss";`.
- Delete `tailwind.config.ts` (v4 auto-detects content; the config was minimal).
- **Border-color fix:** v4 changes bare `border` default color from gray to
  `currentColor`. At spec time grep finds 8 bare usages (Message.tsx:92,
  DocumentList.tsx:94, ChatStream.tsx:99, signup/page.tsx:25/28/29/30,
  login/page.tsx:28). Each gets an explicit `border-slate-300` (matching today's
  rendered color). Re-grep during implementation — the list is the checkpoint, not
  the source of truth. `DocumentList.tsx:108` already sets `border-red-300`.
- Verify: same gates; confirm the built CSS still contains the expected utilities
  and boot the dev server to spot-check the pages (borders visibly gray, not black —
  that is the regression this task exists to prevent).

## Commit structure and verification

Three separate commits (one per task), each independently revertible. Per-task
gates: `npx tsc --noEmit`, `npm run build`, dev-boot smoke. Task 2 additionally
exercises login + signup live against the running backend. No frontend test harness
exists; unchanged.

## Out of scope

- Server-first restructure (cookie sessions via FastAPI, BFF proxy, server
  components) — **parked** for the future org-level project.
- ESLint adoption.
- ChatStream / UploadButton refactor.
- TypeScript major bump; `@types/node` bump beyond what install requires.

## Risks

- React 19 type strictness → mechanical fixes, surfaced by tsc, bounded.
- Turbopack vs webpack behavior differences → caught empirically by the build + dev
  boot gates; webpack remains available as a fallback flag if Turbopack build fails.
- Tailwind v4 subtle class changes beyond the known border default → visual
  comparison during Task 3; the app uses only plain utilities.
- Flagsmith — none: latest version, vanilla JS client.

## Decisions (owner-approved)

1. **Approach 2 of 3.** Platform jump + honest React 19 idioms now; server-first
   restructure reviewed and parked for the org-level era (it requires FastAPI
   session/cookie endpoints — backend work, full spec).
2. **Tailwind v4 included, as the final isolated task**, so any build failure has
   exactly one suspect.
3. **No shoehorning React 19 hooks into streaming/upload flows** — modernization
   must buy something real (less code, standard pattern), not just novelty.
4. **No ESLint** — the removed `next lint` script is not replaced; linting can be
   adopted deliberately later.
