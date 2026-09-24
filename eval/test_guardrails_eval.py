from guardrails_eval import classify, compute_metrics


def test_classify_matrix():
    assert classify(True, True) == "TP"
    assert classify(True, False) == "FN"
    assert classify(False, False) == "TN"
    assert classify(False, True) == "FP"


def test_compute_metrics():
    results = [
        {"result": "TP"}, {"result": "TP"}, {"result": "TN"},
        {"result": "FP"}, {"result": "FN"},
    ]
    m = compute_metrics(results)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert m["accuracy"] == 3 / 5
