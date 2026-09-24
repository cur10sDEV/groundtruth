import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { FlagsmithProvider } from "@/lib/flags";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <AuthProvider>
          <FlagsmithProvider>{children}</FlagsmithProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
