import json
import sys

from app.rag.retrieval.retriever import get_retriever


def mrr_at_k(retrieved: list[str], expected: list[str], k: int = 10) -> float:
    for i, rid in enumerate(retrieved[:k]):
        if rid in expected:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for e in expected if e in retrieved[:k])
    return hits / len(expected)


async def main(path: str = "eval/seed_qa.json") -> None:
    with open(path) as f:
        qas = json.load(f)
    retriever = get_retriever()
    recalls, mrrs = [], []
    for qa in qas:
        retrieved = await retriever.retrieve(
            qa["question"], org_id="eval-org", user_ids=["eval-user"], filters={}, limit=10
        )
        ids = [c.chunk_id for c in retrieved]
        expected = qa["expected_chunk_ids"]
        recalls.append(recall_at_k(ids, expected))
        mrrs.append(mrr_at_k(ids, expected))
        print(f"Q: {qa['question']}")
        print(f"  recall@10={recalls[-1]:.2f} mrr={mrrs[-1]:.2f}")
    print(f"\nMEAN recall@10={sum(recalls)/len(recalls):.3f}")
    print(f"MEAN mrr={sum(mrrs)/len(mrrs):.3f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval/seed_qa.json"))
