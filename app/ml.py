"""检查点权重的落盘、加载与检查点库生成。

每个检查点是一个 ``.npz`` 文件，内含：

* ``weights``：5 个 float64 权重，依次为四项特征系数与偏置；
* ``checkpoint_id``：档号（1～16）；
* ``group``：所属候选组标识；
* ``session_id``：训练会话号（12 位十六进制）。

检查点库按 ``<data_dir>/checkpoint_library/<group>/<session_id>/checkpoint_xxx.npz``
组织，组内恰好 16 档，按档号顺序登记。
"""

from __future__ import annotations

import os

import numpy as np

INPUT_DIM = 4
# 权重向量 = 四项特征系数 + 偏置。
WEIGHT_DIM = INPUT_DIM + 1
CHECKPOINT_COUNT = 16
WEIGHT_DTYPE = np.float64
# 负载的合法范围。
LOAD_MIN = 0
LOAD_MAX = 65535
# 特征与实测值的合法范围。
VALUE_MIN = -100
VALUE_MAX = 100
# MSE 并列容差：与最低 MSE 相差不超过该值即视为并列。
MSE_TIE_TOL = 1e-9
# 选档最多读取的区间汇总条数（16 档 < 40，天然满足）。
MAX_SUMMARIES_READ = 40

# 固定库种子，保证任何环境下生成的两组检查点逐元素一致。
LIBRARY_SEED = 20261125


def checkpoint_path(library_dir: str, group: str, session_id: str,
                    checkpoint_no: int) -> str:
    return os.path.join(
        library_dir, group, session_id, f"checkpoint_{checkpoint_no:03d}.npz"
    )


def save_checkpoint(path: str, weights: np.ndarray, checkpoint_id: int,
                    group: str, session_id: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        weights=np.asarray(weights, dtype=WEIGHT_DTYPE),
        checkpoint_id=np.int64(checkpoint_id),
        group=np.asarray(group),
        session_id=np.asarray(session_id),
    )


def load_checkpoint(path: str) -> dict:
    """加载检查点，返回权重与登记信息。"""
    with np.load(path, allow_pickle=False) as data:
        return {
            "weights": data["weights"].astype(WEIGHT_DTYPE, copy=True),
            "checkpoint_id": int(data["checkpoint_id"]),
            "group": str(data["group"]),
            "session_id": str(data["session_id"]),
        }


def predict(weights: np.ndarray, features: list[int]) -> float:
    """四项整数特征 + 常数偏置项的线性预测。"""
    if len(features) != INPUT_DIM:
        raise ValueError("特征必须恰好四项")
    phi = np.asarray(list(features) + [1.0], dtype=WEIGHT_DTYPE)
    return float(phi @ np.asarray(weights, dtype=WEIGHT_DTYPE))


def _session_id_for(group_index: int, checkpoint_no: int) -> str:
    """由组序号与档号确定性派生 12 位十六进制会话号。"""
    seed = LIBRARY_SEED + group_index * 1009 + checkpoint_no * 37
    rng = np.random.default_rng(seed)
    token = rng.integers(0, 16**12, dtype=np.int64)
    return f"{int(token):012x}"


def _weights_for(group_index: int, checkpoint_no: int) -> np.ndarray:
    """确定性生成某组某档的权重；不同组、不同档权重互不相同。"""
    rng = np.random.default_rng(
        LIBRARY_SEED ^ (group_index * 7919 + checkpoint_no * 104729)
    )
    # 系数落在量级合理的区间，偏置单独一个尺度，便于人工核对。
    coef = rng.uniform(-2.0, 2.0, size=INPUT_DIM)
    bias = rng.uniform(-5.0, 5.0)
    return np.concatenate([coef, [bias]]).astype(WEIGHT_DTYPE)


def ensure_library(library_dir: str,
                   groups: tuple[str, ...] = ("group_a", "group_b")) -> dict:
    """确保检查点库存在；缺失时按固定种子确定性生成。

    返回登记册：``{group: [{checkpoint_no, session_id, path, weights}, ...]}``，
    每组严格按 1～16 顺序登记。
    """
    registry: dict[str, list[dict]] = {}
    for gi, group in enumerate(groups):
        entries: list[dict] = []
        for no in range(1, CHECKPOINT_COUNT + 1):
            session_id = _session_id_for(gi, no)
            path = checkpoint_path(library_dir, group, session_id, no)
            if not os.path.exists(path):
                save_checkpoint(
                    path, _weights_for(gi, no), no, group, session_id
                )
            info = load_checkpoint(path)
            if info["checkpoint_id"] != no or info["group"] != group \
                    or info["session_id"] != session_id:
                raise RuntimeError(f"检查点库文件内容与登记不符: {path}")
            entries.append({
                "checkpoint_no": no,
                "session_id": session_id,
                "group": group,
                "path": path,
                "weights": info["weights"],
            })
        registry[group] = entries
    return registry
