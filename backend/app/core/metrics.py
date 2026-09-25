from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter("rag_requests_total", "Total requests")
ERRORS_TOTAL = Counter("rag_errors_total", "Total errors")
LATENCY = Histogram("rag_latency_seconds", "Request latency", buckets=[0.1, 0.25, 0.5, 1, 2, 5])
TOKENS_IN = Counter("rag_tokens_in_total", "Input tokens")
TOKENS_OUT = Counter("rag_tokens_out_total", "Output tokens")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses")
INGESTION_PROCESSED = Counter("rag_ingestion_processed_total", "Docs ingested")
INGESTION_FAILED = Counter("rag_ingestion_failed_total", "Docs failed")
EVENTS_DROPPED = Counter(
    "rag_events_dropped_total", "Ingestion events dropped by the translator", ["reason"]
)


def record_request(duration_seconds: float, status: str) -> None:
    REQUESTS_TOTAL.inc()
    LATENCY.observe(duration_seconds)
    if status != "ok":
        ERRORS_TOTAL.inc()


def incr_cache(hit: bool) -> None:
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()


def record_tokens(tokens_in: int, tokens_out: int) -> None:
    TOKENS_IN.inc(tokens_in)
    TOKENS_OUT.inc(tokens_out)
