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

  return <AuthContext value={{ auth, login, signup, logout }}>{children}</AuthContext>;
}

export const useAuth = () => useContext(AuthContext);
