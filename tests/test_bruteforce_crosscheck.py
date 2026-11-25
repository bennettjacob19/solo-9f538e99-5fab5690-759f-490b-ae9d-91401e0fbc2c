"""独立暴力复算交叉验证：不依赖引擎聚合，直接逐样本逐档重算区间 MSE。

复算逻辑与 :mod:`app.calibration` 完全独立：只从引擎取出已受理的原始
样本（测试侧核对用途）和检查点库权重，自行过滤负载闭区间、逐档计算
MSE，验证选档、并列容差、预测值与实读条数全部吻合。
"""

from __future__ import annotations

import math

from app import ml


def _brute_force_select(store, load_start, load_end, features):
    group = store.active_group
    gs = store.groups[group]
    samples = [s for b in gs.batches.values() for s in b.records
               if load_start <= s.load <= load_end]
    results = []
    for no in range(1, ml.CHECKPOINT_COUNT + 1):
        weights = store.registry[group][no - 1]["weights"]
        if not samples:
            results.append((no, None))
            continue
        sse = sum(
            (ml.predict(weights, s.features) - s.measured) ** 2
            for s in samples
        )
        results.append((no, sse / len(samples)))
    calibrated = [(no, m) for no, m in results if m is not None]
    if not calibrated:
        return None
    lo = min(m for _, m in calibrated)
    winners = [(no, m) for no, m in calibrated
               if m - lo <= ml.MSE_TIE_TOL]
    no, mse = winners[0]
    weights = store.registry[group][no - 1]["weights"]
    return {
        "checkpoint_no": no,
        "mse": mse,
        "prediction": ml.predict(weights, features),
        "session_id": store.registry[group][no - 1]["session_id"],
        "sample_count": len(samples),
    }


def _sample(no, load, feats, measured):
    return {"sample_no": no, "load": load, "features": feats,
            "measured": measured}


def test_brute_force_matches_engine_across_intervals(store):
    store.ingest_batch("b1", [
        _sample(1, 100, [1, 2, 3, 4], 7),
        _sample(2, 250, [-1, -2, -3, -4], -7),
        _sample(3, 400, [10, -10, 0, 5], 3),
        _sample(4, 900, [7, 7, -7, -7], 40),
    ])
    store.ingest_batch("b2", [
        _sample(1, 150, [0, 1, 2, 3], -3),
        _sample(2, 600, [50, -50, 50, -50], 99),
    ])

    for start, end in [(0, 65535), (100, 400), (200, 300), (500, 1000),
                       (100, 100), (899, 901)]:
        feats = [3, -2, 1, 0]
        expected = _brute_force_select(store, start, end, feats)
        actual = store.infer(start, end, feats)
        if expected is None:
            assert actual["status"] == "uncalibrated"
            assert actual["summaries_read"] == 16
            continue
        assert actual["status"] == "ok"
        assert actual["checkpoint_no"] == expected["checkpoint_no"]
        assert math.isclose(actual["mse"], expected["mse"],
                            rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(actual["prediction"], expected["prediction"],
                            rel_tol=1e-12, abs_tol=1e-12)
        assert actual["session_id"] == expected["session_id"]
        assert actual["summaries_read"] == 16
        assert actual["candidate_count"] >= 1


def test_brute_force_matches_after_cross_segment_correction(store):
    store.ingest_batch("b1", [_sample(1, 100, [1, 1, 1, 1], 5)])
    store.correct_sample("b1", _sample(1, 800, [2, 2, 2, 2], -5))

    # 旧区间：暴力视角也无样本。
    assert _brute_force_select(store, 50, 150, [1, 1, 1, 1]) is None
    # 新区间：两侧算法一致。
    expected = _brute_force_select(store, 700, 900, [2, 2, 2, 2])
    actual = store.infer(700, 900, [2, 2, 2, 2])
    assert actual["checkpoint_no"] == expected["checkpoint_no"]
    assert math.isclose(actual["mse"], expected["mse"],
                        rel_tol=1e-12, abs_tol=1e-12)


def test_brute_force_matches_after_group_swap(store):
    store.ingest_batch("b1", [_sample(1, 300, [1, -1, 1, -1], 9)])
    store.begin_swap("group_b")
    store.commit_swap()
    # 新组未校准。
    assert _brute_force_select(store, 0, 65535, [1, 1, 1, 1]) is None
    store.ingest_batch("b2", [_sample(2, 300, [4, 3, 2, 1], -8)])
    expected = _brute_force_select(store, 0, 65535, [4, 3, 2, 1])
    actual = store.infer(0, 65535, [4, 3, 2, 1])
    assert actual["group"] == "group_b"
    assert actual["checkpoint_no"] == expected["checkpoint_no"]
    assert math.isclose(actual["mse"], expected["mse"],
                        rel_tol=1e-12, abs_tol=1e-12)
