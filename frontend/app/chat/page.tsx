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
