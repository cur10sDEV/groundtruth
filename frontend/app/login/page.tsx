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
