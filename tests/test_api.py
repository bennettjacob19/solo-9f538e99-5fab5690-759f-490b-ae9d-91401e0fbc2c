"""FastAPI 端到端测试：校验码、未校准语义、更正、换组与选档读取上限。"""

from __future__ import annotations


def _sample(no, load, feats, measured):
    return {"sample_no": no, "load": load, "features": feats,
            "measured": measured}


def test_health_and_docs_registry(client):
    r = client.get("/health")
    assert r.status_code == 200
    r = client.get("/docs")
    assert r.status_code == 200
    body = r.json()
    assert body["checkpoint_count"] == 16
    assert [c["order"] for c in body["checkpoints"]] == list(range(1, 17))
    assert all(len(c["session_id"]) == 12 for c in body["checkpoints"])
    # openapi 仍可机器读取，Swagger UI 已让位给业务登记册。
    assert client.get("/openapi.json").status_code == 200


def test_validation_errors_return_422(client):
    # 负载越界。
    r = client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [_sample(1, 70000, [1, 2, 3, 4], 5)],
    })
    assert r.status_code == 422
    # 特征越界 / 数量不对。
    r = client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [_sample(1, 10, [1, 2, 3], 5)],
    })
    assert r.status_code == 422
    r = client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [_sample(1, 10, [1, 2, 3, 200], 5)],
    })
    assert r.status_code == 422
    # 实测值越界。
    r = client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [_sample(1, 10, [1, 2, 3, 4], -101)],
    })
    assert r.status_code == 422


def test_batch_conflicts_return_409_without_state_change(client):
    payload = {"batch_id": "b1",
               "samples": [_sample(1, 100, [1, 1, 1, 1], 1),
                           _sample(1, 200, [2, 2, 2, 2], 2)]}
    r = client.post("/calibration/batches", json=payload)
    assert r.status_code == 409
    assert client.get("/state").json()["total_samples"] == 0

    ok = {"batch_id": "b1",
          "samples": [_sample(1, 100, [1, 1, 1, 1], 1)]}
    assert client.post("/calibration/batches", json=ok).status_code == 201
    r = client.post("/calibration/batches", json=ok)
    assert r.status_code == 409


def test_full_lifecycle_ingest_correct_infer(client):
    r = client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [
            _sample(1, 100, [1, 2, 3, 4], 7),
            _sample(2, 200, [-1, -2, -3, -4], -7),
            _sample(3, 300, [10, -10, 0, 5], 3),
        ],
    })
    assert r.status_code == 201
    assert r.json()["records_accepted"] == 3

    # 区间无样本：200 + 未校准。
    r = client.post("/inference", json={
        "load_start": 500, "load_end": 600,
        "features": [1, 2, 3, 4],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "uncalibrated"
    assert body["accepted"] is False
    assert body["summaries_read"] == 16

    # 正常选档推理。
    r = client.post("/inference", json={
        "load_start": 100, "load_end": 300,
        "features": [1, 2, 3, 4],
    })
    body = r.json()
    assert body["status"] == "ok"
    assert 1 <= body["checkpoint_no"] <= 16
    assert body["summaries_read"] == 16
    assert body["candidate_count"] >= 1
    assert len(body["session_id"]) == 12
    assert isinstance(body["mse"], float)
    assert isinstance(body["prediction"], float)
    assert len(body["weights"]) == 5

    # 跨段更正：100 -> 700。
    r = client.post("/calibration/corrections", json={
        "batch_id": "b1",
        "sample": _sample(1, 700, [9, 8, 7, 6], -2),
    })
    assert r.status_code == 200
    assert r.json()["load_segment_changed"] is True

    # 旧样本所在区间现在只剩 200/300 两个样本，700 段新增一个样本。
    r = client.post("/inference", json={
        "load_start": 50, "load_end": 150, "features": [1, 2, 3, 4]})
    assert r.json()["status"] == "uncalibrated"
    r = client.post("/inference", json={
        "load_start": 650, "load_end": 750, "features": [9, 8, 7, 6]})
    assert r.json()["status"] == "ok"

    # 更正不存在的样本号 -> 409，状态保留。
    r = client.post("/calibration/corrections", json={
        "batch_id": "b1",
        "sample": _sample(99, 700, [1, 1, 1, 1], 1),
    })
    assert r.status_code == 409


def test_inference_input_validation(client):
    r = client.post("/inference", json={
        "load_start": 300, "load_end": 100, "features": [1, 2, 3, 4]})
    assert r.status_code == 422
    r = client.post("/inference", json={
        "load_start": -1, "load_end": 100, "features": [1, 2, 3, 4]})
    assert r.status_code == 422
    r = client.post("/inference", json={
        "load_start": 0, "load_end": 100, "features": [1, 2, 3, 101]})
    assert r.status_code == 422


def test_group_swap_flow_over_http(client):
    # 先给 group_a 一些校准数据。
    client.post("/calibration/batches", json={
        "batch_id": "a1",
        "samples": [_sample(1, 100, [1, 1, 1, 1], 5)]})

    r = client.post("/candidate-groups/swap", json={"target_group": "group_b"})
    assert r.status_code == 200
    assert r.json()["pending_group"] == "group_b"

    # 准备期仍可推理（活动组未变）。
    r = client.post("/inference", json={
        "load_start": 0, "load_end": 200, "features": [1, 1, 1, 1]})
    assert r.json()["group"] == "group_a"

    # 非法目标组 -> 409 且无 pending。
    r = client.post("/candidate-groups/swap", json={"target_group": "nope"})
    assert r.status_code == 409

    r = client.post("/candidate-groups/swap/commit")
    assert r.status_code == 200
    assert r.json()["active_group"] == "group_b"

    r = client.post("/inference", json={
        "load_start": 0, "load_end": 200, "features": [1, 1, 1, 1]})
    assert r.json()["status"] == "uncalibrated"

    # 无准备时提交 -> 409。
    assert client.post("/candidate-groups/swap/commit").status_code == 409

    # 再准备并撤除。
    client.post("/candidate-groups/swap", json={"target_group": "group_a"})
    r = client.post("/candidate-groups/swap/abort")
    assert r.status_code == 200
    assert r.json()["active_group"] == "group_b"


def test_summaries_read_never_exceeds_cap(client):
    client.post("/calibration/batches", json={
        "batch_id": "b1",
        "samples": [_sample(1, 100, [1, 2, 3, 4], 5)]})
    body = client.post("/inference", json={
        "load_start": 0, "load_end": 65535,
        "features": [1, 2, 3, 4]}).json()
    assert body["summaries_read"] <= 40
    assert body["summaries_read"] == 16
