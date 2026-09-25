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
