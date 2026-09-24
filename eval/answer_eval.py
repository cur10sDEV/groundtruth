import json
import sys

import litellm

from app.core.config import get_settings
from app.core.logging import get_logger, new_correlation_id
from app.rag.retrieval.faithfulness import check_faithfulness
from app.rag.retrieval.orchestrator import resolve_text_for_chunk_ids, run_query

logger = get_logger(__name__)


def parse_relevance(raw: str) -> float:
    try:
        data = json.loads(raw)
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


async def judge_relevance(query: str, answer: str) -> float:
    s = get_settings()
    prompt = (
        "On a scale 0.0 to 1.0, how well does the following answer address the "
        'question? Return JSON {"score": <float>}.\n\n'
        f"Question: {query}\nAnswer: {answer}"
    )
    try:
        resp = await litellm.acompletion(
            model=s.llm_primary_model,
            api_key=s.llm_api_key_primary or None,
            messages=[
                {"role": "system", "content": "You are a strict relevance judge."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return parse_relevance(resp.choices[0].message.content)
    except Exception as exc:
        logger.warning("relevance judge failed, scoring 0.0", extra={"exc": str(exc)})
        return 0.0


async def evaluate_answer(query: str, answer: str, contexts: list[dict]) -> dict:
    f = await check_faithfulness(query, answer, contexts)
    relevance = await judge_relevance(query, answer)
    return {"faithfulness": f.score, "faithful": f.faithful, "relevance": relevance}


async def main(path: str = "eval/seed_qa.json") -> None:
    with open(path) as f:
        qas = json.load(f)
    faithfulness_scores, relevance_scores = [], []
    for qa in qas:
        answer, chunk_ids = "", []
        async for ev in run_query(
            qa["question"],
            org_id="eval-org",
            user_ids=["eval-user"],
            feature_flags={"cache.enabled": False},
            trace_id=new_correlation_id(),
        ):
            if ev["type"] == "done":
                answer = ev["answer"]
                chunk_ids = ev["chunk_ids"]
        contexts = await resolve_text_for_chunk_ids(chunk_ids, org_id="eval-org")
        result = await evaluate_answer(qa["question"], answer, contexts)
        faithfulness_scores.append(result["faithfulness"])
        relevance_scores.append(result["relevance"])
        print(f"Q: {qa['question']}")
        print(
            f"  faithfulness={result['faithfulness']:.2f} "
            f"faithful={result['faithful']} relevance={result['relevance']:.2f}"
        )
    print(f"\nMEAN faithfulness={sum(faithfulness_scores) / len(faithfulness_scores):.3f}")
    print(f"MEAN relevance={sum(relevance_scores) / len(relevance_scores):.3f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval/seed_qa.json"))
