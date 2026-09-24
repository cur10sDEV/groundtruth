"use client";

import flagsmith from "@flagsmith/flagsmith";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export const DEFAULT_FLAGS = {
  "reranker.enabled": false,
  "cache.enabled": true,
  "multi_query.enabled": true,
  "filter_extraction.enabled": true,
  "faithfulness.enabled": true,
  "guard_model.enabled": false,
} as const;

export type FlagName = keyof typeof DEFAULT_FLAGS;
export type Flags = Record<FlagName, boolean>;
export const FLAG_NAMES = Object.keys(DEFAULT_FLAGS) as FlagName[];

const FlagsContext = createContext<Flags>(DEFAULT_FLAGS);

function snapshot(): Flags {
  const out: Flags = { ...DEFAULT_FLAGS };
  for (const name of FLAG_NAMES) {
    try {
      out[name] = flagsmith.hasFeature(name, { fallback: DEFAULT_FLAGS[name] });
    } catch {
      out[name] = DEFAULT_FLAGS[name];
    }
  }
  return out;
}

export function FlagsmithProvider({ children }: { children: ReactNode }) {
  const [flags, setFlags] = useState<Flags>(DEFAULT_FLAGS);

  useEffect(() => {
    const environmentKey = process.env.NEXT_PUBLIC_FLAGSMITH_KEY;
    if (!environmentKey || typeof window === "undefined") return;

    let cancelled = false;
    const update = () => {
      if (cancelled) return;
      try {
        setFlags(snapshot());
      } catch {
        // A flag provider must never take the app down; keep current values.
      }
    };

    try {
      flagsmith
        .init({
          environmentID: environmentKey,
          api: process.env.NEXT_PUBLIC_FLAGSMITH_API_URL || undefined,
          // realtime only when an explicit SSE endpoint is configured
          realtime: Boolean(process.env.NEXT_PUBLIC_FLAGSMITH_REALTIME_URL),
          eventSourceUrl: process.env.NEXT_PUBLIC_FLAGSMITH_REALTIME_URL || undefined,
          onChange: update,
        })
        .then(update)
        .catch(update);
    } catch {
      // Unreachable or misconfigured Flagsmith: defaults stay in effect.
    }

    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo(() => flags, [flags]);

  return <FlagsContext.Provider value={value}>{children}</FlagsContext.Provider>;
}

export function useFlags(): Flags {
  return useContext(FlagsContext);
}
