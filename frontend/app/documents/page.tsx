"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Documents</h1>
        <Link href="/chat" className="text-blue-600 underline">
          Chat
        </Link>
      </div>
      <UploadButton onUploaded={() => setRefresh((r) => r + 1)} />
      <DocumentList refresh={refresh} />
    </div>
  );
}
