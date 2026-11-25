#!/usr/bin/env python3
"""端到端演示：检查点登记、分批校准、跨段更正、区间选档推理与整组换组。

用法：
    python demo/run_demo.py            # 默认对接 http://127.0.0.1:8000
    BASE_URL=http://host:port python demo/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def sample(no: int, load: int, feats: list[int], measured: int) -> dict:
    return {"sample_no": no, "load": load, "features": feats,
            "measured": measured}


def show(title: str, status: int, body: dict) -> None:
    print(f"-- {title} [{status}]")
    print(json.dumps(body, ensure_ascii=False, indent=2))


def main() -> int:
    print(f"==> 对接服务 {BASE_URL}")

    # 1. /docs 登记册：1～16 号检查点按序登记。
    status, docs = call("GET", "/docs")
    assert status == 200, docs
    print(f"==> 登记册：活动组 {docs['active_group']}，"
          f"{docs['checkpoint_count']} 档检查点")
    for c in docs["checkpoints"][:3]:
        print(f"    档{c['checkpoint_no']:>2} 会话 {c['session_id']} "
              f"权重 {[round(w, 4) for w in c['weights']]}")
    print("    ...")

    # 2. 分批录入校准记录。
    status, body = call("POST", "/calibration/batches", {
        "batch_id": "demo-batch-1",
        "samples": [
            sample(1, 100, [1, 2, 3, 4], 7),
            sample(2, 200, [-1, -2, -3, -4], -7),
            sample(3, 300, [10, -10, 0, 5], 3),
        ],
    })
    show("录入第一批", status, body)
    assert status == 201

    status, body = call("POST", "/calibration/batches", {
        "batch_id": "demo-batch-2",
        "samples": [
            sample(1, 500, [5, 5, 5, 5], 10),
            sample(2, 600, [0, 0, 0, 1], -50),
        ],
    })
    show("录入第二批", status, body)
    assert status == 201

    # 3. 区间无样本 -> 未校准。
    status, body = call("POST", "/inference", {
        "load_start": 40000, "load_end": 50000,
        "features": [1, 2, 3, 4],
    })
    show("空区间推理（应未校准）", status, body)
    assert body["status"] == "uncalibrated"

    # 4. 正常区间选档推理。
    status, body = call("POST", "/inference", {
        "load_start": 100, "load_end": 300,
        "features": [1, 2, 3, 4],
    })
    show("区间选档推理", status, body)
    assert status == 200 and body["status"] == "ok"
    print(f"==> 选中 会话={body['session_id']} 档号={body['checkpoint_no']} "
          f"MSE={body['mse']:.6f} 预测={body['prediction']:.6f} "
          f"实读汇总={body['summaries_read']} 条")

    # 5. 跨负载段更正：样本 1 从负载 100 整组改到负载 700。
    status, body = call("POST", "/calibration/corrections", {
        "batch_id": "demo-batch-1",
        "sample": sample(1, 700, [9, 8, 7, 6], -2),
    })
    show("跨段更正", status, body)
    assert status == 200 and body["load_segment_changed"] is True

    status, old_seg = call("POST", "/inference", {
        "load_start": 50, "load_end": 150, "features": [1, 2, 3, 4]})
    status, new_seg = call("POST", "/inference", {
        "load_start": 650, "load_end": 750, "features": [9, 8, 7, 6]})
    print(f"==> 更正后旧段={old_seg['status']}，新段={new_seg['status']} "
          f"(档 {new_seg.get('checkpoint_no')})")
    assert old_seg["status"] == "uncalibrated"
    assert new_seg["status"] == "ok"

    # 6. 更正失败保留已生效状态。
    status, body = call("POST", "/calibration/corrections", {
        "batch_id": "demo-batch-1",
        "sample": sample(99, 700, [1, 1, 1, 1], 1),
    })
    show("更正不存在的样本号（应 409 且状态不变）", status, body)
    assert status == 409

    # 7. 候选组整组更换。
    status, body = call("POST", "/candidate-groups/swap",
                        {"target_group": "group_b"})
    show("换组准备（准备期仍可推理）", status, body)
    assert status == 200
    status, body = call("POST", "/candidate-groups/swap/commit")
    show("提交换组", status, body)
    assert status == 200 and body["active_group"] == "group_b"

    status, body = call("POST", "/inference", {
        "load_start": 0, "load_end": 65535, "features": [1, 2, 3, 4]})
    show("换组后推理（新组未校准）", status, body)
    assert body["status"] == "uncalibrated"

    print("==> 演示完成 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
