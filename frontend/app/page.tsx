"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function HomePage() {
  const { auth } = useAuth();
  const router = useRouter();
  useEffect(() => {
    router.replace(auth ? "/chat" : "/login");
  }, [auth, router]);
  return null;
}
