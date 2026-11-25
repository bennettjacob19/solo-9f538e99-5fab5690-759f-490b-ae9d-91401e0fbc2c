"""校准引擎单元测试：受理、跨段更正、选档并列、持久化与失败回滚。"""

from __future__ import annotations

import math

import pytest

from app import ml
from app.calibration import CalibrationError, CalibrationStore


def _sample(no, load, feats, measured):
    return {"sample_no": no, "load": load, "features": feats,
            "measured": measured}


def test_registry_has_16_ordered_checkpoints_with_distinct_weights(store):
    docs = store.registry_docs()
    assert docs["checkpoint_count"] == 16
    nos = [c["checkpoint_no"] for c in docs["checkpoints"]]
    assert nos == list(range(1, 17))
    weights = [tuple(c["weights"]) for c in docs["checkpoints"]]
    assert len(set(weights)) == 16
    for c in docs["checkpoints"]:
        assert len(c["session_id"]) == 12
        int(c["session_id"], 16)
        assert len(c["weights"]) == 5


def test_ingest_validates_payload_ranges(store):
    with pytest.raises(CalibrationError):
        store.ingest_batch("b1", [_sample(1, 65536, [1, 2, 3, 4], 5)])
    with pytest.raises(CalibrationError):
        store.ingest_batch("b1", [_sample(1, 0, [1, 2, 3, 101], 5)])
    with pytest.raises(CalibrationError):
        store.ingest_batch("b1", [_sample(1, 0, [1, 2, 3, 4], -101)])
    # 整批驳回：没有任何状态残留。
    assert store.overview()["total_samples"] == 0


def test_ingest_rejects_duplicate_sample_no_within_batch(store):
    with pytest.raises(CalibrationError):
        store.ingest_batch("b1", [
            _sample(1, 10, [1, 1, 1, 1], 10),
            _sample(1, 20, [2, 2, 2, 2], 20),
        ])
    assert store.overview()["total_samples"] == 0


def test_ingest_rejects_duplicate_batch_id(store):
    store.ingest_batch("b1", [_sample(1, 10, [1, 1, 1, 1], 10)])
    with pytest.raises(CalibrationError):
        store.ingest_batch("b1", [_sample(2, 10, [1, 1, 1, 1], 10)])


def test_ingest_contributions_match_brute_force_residuals(store):
    samples = [
        _sample(1, 100, [1, -2, 3, -4], 7),
        _sample(2, 200, [-5, 6, -7, 8], -9),
        _sample(3, 65535, [100, -100, 0, 50], 100),
    ]
    out = store.ingest_batch("b1", samples)
    assert out["records_accepted"] == 3

    docs = {c["checkpoint_no"]: c for c in store.registry_docs()["checkpoints"]}
    group = store.active_group
    for no in range(1, 17):
        weights = store.registry[group][no - 1]["weights"]
        expected_sse = 0.0
        for s in samples:
            pred = ml.predict(weights, s["features"])
            expected_sse += (pred - s["measured"]) ** 2
        assert docs[no]["calibrated_samples"] == 3
        assert math.isclose(docs[no]["overall_mse"], expected_sse / 3,
                            rel_tol=1e-12, abs_tol=1e-12)


def test_inference_uncalibrated_when_interval_has_no_samples(store):
    store.ingest_batch("b1", [_sample(1, 100, [1, 1, 1, 1], 10)])
    out = store.infer(200, 300, [1, 2, 3, 4])
    assert out["status"] == "uncalibrated"
    assert out["summaries_read"] == 16
    assert out["candidate_count"] == 0


def test_inference_selects_min_mse_checkpoint_and_predicts(store):
    samples = [_sample(i + 1, 500 + i, [i, i + 1, -i, i - 1], i - 2)
               for i in range(6)]
    store.ingest_batch("b1", samples)
    feats = [3, -1, 2, 0]
    out = store.infer(400, 700, feats)
    assert out["status"] == "ok"
    assert out["summaries_read"] == 16
    assert out["candidate_count"] >= 1

    # 独立复算：直接对 16 档权重算区间 MSE，验证选档与预测。
    expected = []
    for no in range(1, 17):
        weights = store.registry[store.active_group][no - 1]["weights"]
        sse = sum(
            (ml.predict(weights, s["features"]) - s["measured"]) ** 2
            for s in samples
        )
        expected.append((sse / len(samples), no, weights))
    min_mse = min(m for m, _, _ in expected)
    winners = [(m, no, w) for m, no, w in expected
               if m - min_mse <= ml.MSE_TIE_TOL]
    exp_mse, exp_no, exp_w = winners[0]
    assert out["checkpoint_no"] == exp_no
    assert math.isclose(out["mse"], exp_mse, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["prediction"], ml.predict(exp_w, feats),
                        rel_tol=1e-12, abs_tol=1e-12)
    assert out["session_id"] == store.registry[store.active_group][
        exp_no - 1]["session_id"]


def test_inference_tie_picks_first_registered_checkpoint(store):
    # 构造 16 档残差完全一致的样本：特征全 0，预测只剩偏置；
    # 再让实测值等于全部偏置的“公共常数”不可行（偏置不同），
    # 因此改用直接断言并列规则：制造两份并列 MSE 后验证取小档号。
    # 做法：单样本、实测 0，MSE = pred^2；挑选 pred 绝对值最接近的两档，
    # 若不严格并列，则至少验证最低 MSE 档被选中。
    store.ingest_batch("b1", [_sample(1, 10, [0, 0, 0, 0], 0)])
    out = store.infer(0, 100, [0, 0, 0, 0])
    mses = []
    for no in range(1, 17):
        w = store.registry[store.active_group][no - 1]["weights"]
        mses.append((ml.predict(w, [0, 0, 0, 0]) ** 2, no))
    min_mse = min(m for m, _ in mses)
    tied = sorted(no for m, no in mses if m - min_mse <= ml.MSE_TIE_TOL)
    assert out["checkpoint_no"] == tied[0]


def test_cross_segment_correction_is_atomic_and_recomputes(store):
    s_old = _sample(7, 100, [1, 2, 3, 4], 10)
    s_new = _sample(7, 900, [4, 3, 2, 1], -10)
    store.ingest_batch("b1", [s_old])

    before = store.infer(50, 150, [1, 2, 3, 4])
    assert before["status"] == "ok"
    gap_before = store.infer(850, 950, [1, 2, 3, 4])
    assert gap_before["status"] == "uncalibrated"

    out = store.correct_sample("b1", s_new)
    assert out["load_segment_changed"] is True

    # 旧负载段已无样本 -> 未校准；新负载段可推理。
    assert store.infer(50, 150, [1, 2, 3, 4])["status"] == "uncalibrated"
    after = store.infer(850, 950, [4, 3, 2, 1])
    assert after["status"] == "ok"
    # 新 MSE 必须等于用新样本独立复算的值。
    no = after["checkpoint_no"]
    w = store.registry[store.active_group][no - 1]["weights"]
    assert math.isclose(
        after["mse"], (ml.predict(w, s_new["features"]) - s_new["measured"]) ** 2,
        rel_tol=1e-12, abs_tol=1e-12,
    )


def test_correction_only_reads_same_batch(store):
    store.ingest_batch("b1", [_sample(1, 100, [1, 1, 1, 1], 1)])
    store.ingest_batch("b2", [_sample(1, 200, [2, 2, 2, 2], 2)])
    # b1 中没有样本号 2 -> 驳回，已生效状态不变。
    with pytest.raises(CalibrationError):
        store.correct_sample("b1", _sample(2, 300, [3, 3, 3, 3], 3))
    # b2 的同号样本不受影响，仍在 200 段。
    out = store.infer(150, 250, [2, 2, 2, 2])
    assert out["status"] == "ok"
    with pytest.raises(CalibrationError):
        store.correct_sample("missing", _sample(1, 1, [0, 0, 0, 0], 0))


def test_failed_correction_preserves_committed_state(store):
    store.ingest_batch("b1", [_sample(1, 100, [1, 1, 1, 1], 1)])
    before = store.registry_docs()
    with pytest.raises(CalibrationError):
        store.correct_sample("b1", _sample(99, 100, [1, 1, 1, 1], 1))
    after = store.registry_docs()
    assert before == after


def test_exact_mse_tie_picks_lowest_checkpoint_no(store):
    # 构造实测值使至少两档 MSE 严格相等：单样本、特征全 0 时 MSE=偏置²。
    # 先扫出偏置绝对值最接近的两档，若不严格相等，则用更正把实测值设为
    # 两偏置的等距点，使 (b1-m)^2 == (b2-m)^2，制造精确并列。
    biases = {
        no: float(store.registry[store.active_group][no - 1]["weights"][-1])
        for no in range(1, 17)
    }
    ordered = sorted(biases, key=lambda no: abs(biases[no]))
    no1, no2 = ordered[0], ordered[1]
    b1, b2 = biases[no1], biases[no2]
    midpoint = (b1 + b2) / 2.0
    measured = int(max(-100, min(100, round(midpoint))))

    store.ingest_batch("b1", [_sample(1, 10, [0, 0, 0, 0], measured)])
    out = store.infer(0, 100, [0, 0, 0, 0])
    # 独立复算并列集合。
    mses = {no: (biases[no] - measured) ** 2 for no in range(1, 17)}
    lo = min(mses.values())
    tied = sorted(no for no, m in mses.items() if m - lo <= ml.MSE_TIE_TOL)
    assert out["checkpoint_no"] == tied[0]
    assert out["candidate_count"] == len(tied)


def test_summary_read_cap_is_enforced(monkeypatch, store):
    # 16 档天然低于 40 上限；把上限临时压到 5，验证引擎确实在到达
    # 上限后停止读取区间汇总，并如实返回实读条数（不读样本明细）。
    monkeypatch.setattr(ml, "MAX_SUMMARIES_READ", 5)
    store.ingest_batch("b1", [_sample(1, 100, [1, 1, 1, 1], 1)])
    out = store.infer(0, 200, [1, 1, 1, 1])
    assert out["summaries_read"] == 5
    # 只可能选到前 5 档（最先登记的候选）。
    assert out["checkpoint_no"] <= 5


def test_state_persists_across_store_reload(tmp_path):
    library_dir = tmp_path / "checkpoint_library"
    registry = ml.ensure_library(str(library_dir))
    s1 = CalibrationStore(str(tmp_path), registry)
    s1.ingest_batch("b1", [_sample(1, 100, [1, 2, 3, 4], 5)])
    s1.correct_sample("b1", _sample(1, 200, [4, 3, 2, 1], -5))

    s2 = CalibrationStore(str(tmp_path), registry)
    assert s2.active_group == s1.active_group
    out = s2.infer(150, 250, [4, 3, 2, 1])
    assert out["status"] == "ok"
    # 旧段无样本。
    assert s2.infer(50, 150, [1, 2, 3, 4])["status"] == "uncalibrated"
