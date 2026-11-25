"""候选组整组更换测试：准备期可用性、提交切换、撤除与失败保留。"""

from __future__ import annotations

import pytest

from app import ml
from app.calibration import CalibrationError, CalibrationStore


def _sample(no, load, feats, measured):
    return {"sample_no": no, "load": load, "features": feats,
            "measured": measured}


def test_swap_lifecycle_and_group_isolation(store):
    # group_a 录入校准并更正。
    store.ingest_batch("a1", [_sample(1, 100, [1, 2, 3, 4], 7)])
    store.correct_sample("a1", _sample(1, 300, [2, 2, 2, 2], 8))
    assert store.infer(250, 350, [1, 1, 1, 1])["status"] == "ok"

    # 准备换到 group_b：活动组不变，准备期仍可更正与推理。
    prep = store.begin_swap("group_b")
    assert prep["active_group"] == "group_a"
    assert store.active_group == "group_a"
    store.correct_sample("a1", _sample(1, 400, [3, 3, 3, 3], 9))
    interim = store.infer(350, 450, [1, 1, 1, 1])
    assert interim["status"] == "ok" and interim["group"] == "group_a"

    # 重复准备被驳回，状态不变。
    with pytest.raises(CalibrationError):
        store.begin_swap("group_b")
    assert store.pending_swap is not None

    # 提交后整组切换：新组无校准样本 -> 未校准。
    committed = store.commit_swap()
    assert committed["previous_group"] == "group_a"
    assert store.active_group == "group_b"
    assert store.pending_swap is None
    out = store.infer(0, 65535, [1, 1, 1, 1])
    assert out["status"] == "uncalibrated"
    docs = store.registry_docs()
    assert all(c["group"] == "group_b" for c in docs["checkpoints"])

    # 新组独立录入，不影响旧组状态；切回后旧样本仍在。
    store.ingest_batch("b1", [_sample(1, 50, [0, 0, 0, 0], 1)])
    store.begin_swap("group_a")
    store.commit_swap()
    back = store.infer(350, 450, [3, 3, 3, 3])
    assert back["status"] == "ok" and back["group"] == "group_a"
    # group_b 的数据仍然保留。
    assert store.groups["group_b"].batches["b1"].records[0].load == 50


def test_abort_swap_keeps_active_group(store):
    store.ingest_batch("a1", [_sample(1, 100, [1, 1, 1, 1], 1)])
    store.begin_swap("group_b")
    aborted = store.abort_swap()
    assert aborted["active_group"] == "group_a"
    assert store.pending_swap is None
    assert store.infer(0, 200, [1, 1, 1, 1])["status"] == "ok"
    # 无待撤除准备时报错。
    with pytest.raises(CalibrationError):
        store.abort_swap()


def test_swap_failures_preserve_committed_state(store):
    snapshot = store.registry_docs()
    with pytest.raises(CalibrationError):
        store.begin_swap("group_ghost")
    with pytest.raises(CalibrationError):
        store.begin_swap("group_a")  # 与当前组相同
    assert store.pending_swap is None
    assert store.registry_docs() == snapshot

    # 无准备直接提交失败。
    with pytest.raises(CalibrationError):
        store.commit_swap()
    assert store.active_group == "group_a"


def test_pending_swap_persists_across_reload(tmp_path):
    library_dir = tmp_path / "checkpoint_library"
    registry = ml.ensure_library(str(library_dir))
    s1 = CalibrationStore(str(tmp_path), registry)
    s1.ingest_batch("a1", [_sample(1, 100, [1, 1, 1, 1], 1)])
    s1.begin_swap("group_b")
    assert s1.pending_swap == {"group": "group_b"}

    # 重启后 pending 仍在，活动组未变；提交后完成切换。
    s2 = CalibrationStore(str(tmp_path), registry)
    assert s2.active_group == "group_a"
    assert s2.pending_swap == {"group": "group_b"}
    assert s2.infer(0, 200, [1, 1, 1, 1])["status"] == "ok"
    s2.commit_swap()
    assert s2.active_group == "group_b"

    s3 = CalibrationStore(str(tmp_path), registry)
    assert s3.active_group == "group_b" and s3.pending_swap is None


def test_two_groups_have_distinct_checkpoint_weights(store):
    a = [tuple(e["weights"]) for e in store.registry["group_a"]]
    b = [tuple(e["weights"]) for e in store.registry["group_b"]]
    assert set(a).isdisjoint(set(b))
    assert len(a) == len(b) == ml.CHECKPOINT_COUNT
