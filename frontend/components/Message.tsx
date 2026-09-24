"use client";

export default function Message({
  role,
  content,
  docIds = [],
}: {
  role: "user" | "assistant";
  content: string;
  docIds?: string[];
}) {
  const parts = content.split(/(\[\d+\])/g);
  return (
    <div className={role === "user" ? "text-right" : ""}>
      <div
        className={
          "inline-block max-w-[80%] whitespace-pre-wrap rounded p-3 " +
          (role === "user" ? "bg-blue-100" : "bg-white border")
        }
      >
        {parts.map((part, i) =>
          /^\[\d+\]$/.test(part) ? (
            <sup
              key={i}
              className="rounded bg-slate-200 px-1 text-[10px] font-semibold text-slate-700"
            >
              {part.slice(1, -1)}
            </sup>
          ) : (
            part
          )
        )}
      </div>
      {role === "assistant" && docIds.length > 0 && (
        <div className="mt-1 max-w-[80%] truncate text-xs text-slate-500">
          Sources: {docIds.join(", ")}
        </div>
      )}
    </div>
  );
}
