from retrieval_eval import mrr_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["c"]) == 1.0
    assert recall_at_k(["a", "b"], ["c", "d"]) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["x", "b", "c"], ["b"]) == 0.5
    assert mrr_at_k(["a", "b"], ["z"]) == 0.0
