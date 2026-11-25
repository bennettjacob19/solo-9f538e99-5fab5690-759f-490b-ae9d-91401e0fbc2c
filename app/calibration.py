"""校准记录受理、跨段更正、候选整组更换与区间选档推理的核心引擎。

设计要点（与需求逐条对应）：

* 检查点按档号 1～16 顺序登记；每个候选组一套独立校准状态。
* 校准样本按批次受理，批次内样本号唯一；同号更正整条（含负载），
  更正只在**本批**内查找旧样本，旧贡献撤除与新贡献计入在同一把锁、
  同一次持久化内原子生效。
* 每个检查点维护“按负载聚合”的区间汇总行（count / SSE），选档推理
  只读这些汇总行，绝不读取样本明细；读取条数受
  :data:`~app.ml.MAX_SUMMARIES_READ` 上限约束并在响应中如实返回。
* 区间 MSE 取最低值，与其相差不超过 1e-9 的全部并列候选中取最先登记
  （档号最小）的检查点权重做预测。
* 候选组可整组更换：准备期（pending）内当前已提交组仍可受理更正与
  推理；提交时整体切换，失败则保留已生效状态。
* 每次推理使用的 MSE 与权重均取自受理时刻最新已提交状态：所有变更
  在锁内完成内存修改与原子落盘后才对后续请求可见。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from copy import deepcopy

from . import ml


class CalibrationError(ValueError):
    """校验或状态冲突；抛出前不会改动任何已生效状态。"""


@dataclass
class Sample:
    sample_no: int
    load: int
    features: list[int]
    measured: int

    def as_dict(self) -> dict:
        return {
            "sample_no": self.sample_no,
            "load": self.load,
            "features": list(self.features),
            "measured": self.measured,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            sample_no=int(d["sample_no"]),
            load=int(d["load"]),
            features=[int(v) for v in d["features"]],
            measured=int(d["measured"]),
        )


@dataclass
class Batch:
    batch_id: str
    records: list[Sample] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"batch_id": self.batch_id,
                "records": [s.as_dict() for s in self.records]}


@dataclass
class CheckpointAggregate:
    """单档检查点的区间汇总状态：按负载键聚合的 count / SSE。"""

    checkpoint_no: int
    session_id: str
    # load -> {"count": int, "sse": float}
    loads: dict[int, dict] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "checkpoint_no": self.checkpoint_no,
            "session_id": self.session_id,
            "loads": {str(k): {"count": v["count"], "sse": v["sse"]}
                      for k, v in sorted(self.loads.items())},
        }

    @classmethod
    def from_json(cls, d: dict) -> "CheckpointAggregate":
        return cls(
            checkpoint_no=int(d["checkpoint_no"]),
            session_id=str(d["session_id"]),
            loads={int(k): {"count": int(v["count"]), "sse": float(v["sse"])}
                   for k, v in d.get("loads", {}).items()},
        )

    def interval_summary(self, load_start: int, load_end: int) -> dict:
        """汇总负载闭区间 [load_start, load_end] 内的贡献（不碰样本明细）。"""
        count = 0
        sse = 0.0
        for load, row in self.loads.items():
            if load_start <= load <= load_end:
                count += row["count"]
                sse += row["sse"]
        return {
            "checkpoint_no": self.checkpoint_no,
            "session_id": self.session_id,
            "sample_count": count,
            "sse": sse,
            "mse": (sse / count) if count else None,
        }

    # -- 贡献计入 / 撤除 ------------------------------------------------
    def _row(self, load: int) -> dict:
        row = self.loads.get(load)
        if row is None:
            row = {"count": 0, "sse": 0.0}
            self.loads[load] = row
        return row

    def add_contribution(self, load: int, squared_error: float) -> None:
        row = self._row(load)
        row["count"] += 1
        row["sse"] += squared_error

    def remove_contribution(self, load: int, squared_error: float) -> None:
        row = self.loads[load]
        row["count"] -= 1
        row["sse"] -= squared_error
        if row["count"] == 0:
            # 清空的负载行直接删除，保证区间汇总稳定可复算。
            del self.loads[load]


@dataclass
class GroupState:
    """一个候选组的全部已提交校准状态。"""

    group: str
    batches: dict[str, Batch] = field(default_factory=dict)
    checkpoints: dict[int, CheckpointAggregate] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "group": self.group,
            "batches": {bid: b.as_dict() for bid, b in sorted(self.batches.items())},
            "checkpoints": [self.checkpoints[k].to_json()
                            for k in sorted(self.checkpoints)],
        }

    @classmethod
    def from_json(cls, d: dict) -> "GroupState":
        gs = cls(group=str(d["group"]))
        for bid, bd in d.get("batches", {}).items():
            gs.batches[bid] = Batch(
                batch_id=bid,
                records=[Sample.from_dict(r) for r in bd.get("records", [])],
            )
        for cd in d.get("checkpoints", []):
            agg = CheckpointAggregate.from_json(cd)
            gs.checkpoints[agg.checkpoint_no] = agg
        return gs


def validate_sample_payload(sample_no: int, load: int,
                            features: list[int], measured: int) -> None:
    if not isinstance(sample_no, int) or sample_no <= 0:
        raise CalibrationError("样本号必须为正整数")
    if not isinstance(load, int) or not (ml.LOAD_MIN <= load <= ml.LOAD_MAX):
        raise CalibrationError(
            f"负载必须为 {ml.LOAD_MIN}～{ml.LOAD_MAX} 的整数"
        )
    if len(features) != ml.INPUT_DIM or not all(
        isinstance(v, int) and ml.VALUE_MIN <= v <= ml.VALUE_MAX
        for v in features
    ):
        raise CalibrationError(
            f"四项特征必须均为 {ml.VALUE_MIN}～{ml.VALUE_MAX} 的整数"
        )
    if not isinstance(measured, int) or not (
        ml.VALUE_MIN <= measured <= ml.VALUE_MAX
    ):
        raise CalibrationError(
            f"实测值必须为 {ml.VALUE_MIN}～{ml.VALUE_MAX} 的整数"
        )


class CalibrationStore:
    """线程安全、带原子持久化的校准状态存储。"""

    STATE_FILENAME = "calibration_state.json"

    def __init__(self, data_dir: str, registry: dict):
        self.data_dir = data_dir
        self.state_path = os.path.join(data_dir, self.STATE_FILENAME)
        # registry: group -> 16 条登记项（含权重），档号顺序。
        self.registry = registry
        self._lock = threading.RLock()
        self.groups: dict[str, GroupState] = {}
        self.active_group: str = next(iter(registry))
        self.pending_swap: dict | None = None
        self._load_or_init()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load_or_init(self) -> None:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.active_group = raw["active_group"]
            self.pending_swap = raw.get("pending_swap")
            self.groups = {
                g: GroupState.from_json(d) for g, d in raw["groups"].items()
            }
        for group in self.registry:
            self.groups.setdefault(group, self._fresh_group(group))
        if self.active_group not in self.registry:
            raise RuntimeError("持久化状态中的活动组不在检查点库中")

    def _fresh_group(self, group: str) -> GroupState:
        gs = GroupState(group=group)
        for entry in self.registry[group]:
            gs.checkpoints[entry["checkpoint_no"]] = CheckpointAggregate(
                checkpoint_no=entry["checkpoint_no"],
                session_id=entry["session_id"],
            )
        return gs

    def _persist_locked(self) -> None:
        payload = {
            "active_group": self.active_group,
            "pending_swap": self.pending_swap,
            "groups": {g: gs.to_json() for g, gs in self.groups.items()},
        }
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, self.state_path)

    def _mutate(self, fn, *args, **kwargs):
        """在锁内执行变更并原子落盘；任一步失败都回滚到已生效状态。"""
        with self._lock:
            snapshot = deepcopy(self._snapshot_locked())
            try:
                result = fn(*args, **kwargs)
                self._persist_locked()
                return result
            except Exception:
                self._restore_snapshot_locked(snapshot)
                raise

    def _snapshot_locked(self) -> dict:
        return {
            "groups": self.groups,
            "active_group": self.active_group,
            "pending_swap": self.pending_swap,
        }

    def _restore_snapshot_locked(self, snapshot: dict) -> None:
        self.groups = snapshot["groups"]
        self.active_group = snapshot["active_group"]
        self.pending_swap = snapshot["pending_swap"]

    # ------------------------------------------------------------------
    # 登记册（/docs）
    # ------------------------------------------------------------------
    def registry_docs(self) -> dict:
        with self._lock:
            entries = []
            for entry in self.registry[self.active_group]:
                agg = self.groups[self.active_group].checkpoints[
                    entry["checkpoint_no"]
                ]
                total_count = sum(r["count"] for r in agg.loads.values())
                total_sse = sum(r["sse"] for r in agg.loads.values())
                entries.append({
                    "order": entry["checkpoint_no"],
                    "checkpoint_no": entry["checkpoint_no"],
                    "session_id": entry["session_id"],
                    "group": entry["group"],
                    "path": entry["path"],
                    "weights": [float(v) for v in entry["weights"]],
                    "calibrated_samples": total_count,
                    "overall_mse": (total_sse / total_count)
                    if total_count else None,
                })
            return {
                "active_group": self.active_group,
                "pending_swap": deepcopy(self.pending_swap),
                "checkpoint_count": len(entries),
                "checkpoints": entries,
            }

    # ------------------------------------------------------------------
    # 校准批次受理
    # ------------------------------------------------------------------
    def ingest_batch(self, batch_id: str, samples: list[dict]) -> dict:
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise CalibrationError("batch_id 必须为非空字符串")
        if not samples:
            raise CalibrationError("校准批次至少包含一条样本")
        parsed: list[Sample] = []
        seen: set[int] = set()
        for raw in samples:
            sample = Sample(
                sample_no=int(raw["sample_no"]),
                load=int(raw["load"]),
                features=[int(v) for v in raw["features"]],
                measured=int(raw["measured"]),
            )
            validate_sample_payload(
                sample.sample_no, sample.load, sample.features, sample.measured
            )
            if sample.sample_no in seen:
                raise CalibrationError(
                    f"批次内样本号 {sample.sample_no} 重复，整批驳回"
                )
            seen.add(sample.sample_no)
            parsed.append(sample)
        return self._mutate(self._ingest_locked, batch_id, parsed)

    def _ingest_locked(self, batch_id: str, parsed: list[Sample]) -> dict:
        gs = self.groups[self.active_group]
        if batch_id in gs.batches:
            raise CalibrationError(f"批次 {batch_id} 已受理，不可重复登记")
        gs.batches[batch_id] = Batch(batch_id=batch_id, records=parsed)
        for sample in parsed:
            self._apply_sample_locked(gs, sample, sign=1)
        return {
            "accepted": True,
            "group": self.active_group,
            "batch_id": batch_id,
            "records_accepted": len(parsed),
            "sample_nos": [s.sample_no for s in parsed],
        }

    # ------------------------------------------------------------------
    # 同号更正（跨负载段）
    # ------------------------------------------------------------------
    def correct_sample(self, batch_id: str, sample_payload: dict) -> dict:
        sample = Sample(
            sample_no=int(sample_payload["sample_no"]),
            load=int(sample_payload["load"]),
            features=[int(v) for v in sample_payload["features"]],
            measured=int(sample_payload["measured"]),
        )
        validate_sample_payload(
            sample.sample_no, sample.load, sample.features, sample.measured
        )
        return self._mutate(self._correct_locked, batch_id, sample)

    def _correct_locked(self, batch_id: str, new_sample: Sample) -> dict:
        gs = self.groups[self.active_group]
        batch = gs.batches.get(batch_id)
        # 只读本批同号旧样本；跨批同号不在查找范围内。
        if batch is None:
            raise CalibrationError(f"批次 {batch_id} 不存在，更正被驳回")
        old_index = next(
            (i for i, s in enumerate(batch.records)
             if s.sample_no == new_sample.sample_no),
            None,
        )
        if old_index is None:
            raise CalibrationError(
                f"批次 {batch_id} 中不存在样本号 {new_sample.sample_no}，"
                "更正被驳回且已生效状态不变"
            )
        old_sample = batch.records[old_index]
        # 先撤旧贡献，再计新贡献，连同样本替换一起在本次提交内生效。
        self._apply_sample_locked(gs, old_sample, sign=-1)
        self._apply_sample_locked(gs, new_sample, sign=1)
        batch.records[old_index] = new_sample
        return {
            "corrected": True,
            "group": self.active_group,
            "batch_id": batch_id,
            "sample_no": new_sample.sample_no,
            "old": old_sample.as_dict(),
            "new": new_sample.as_dict(),
            "load_segment_changed": old_sample.load != new_sample.load,
        }

    # ------------------------------------------------------------------
    # 贡献计算
    # ------------------------------------------------------------------
    def _weights_locked(self, group: str, checkpoint_no: int):
        entry = self.registry[group][checkpoint_no - 1]
        return entry["weights"]

    def _apply_sample_locked(self, gs: GroupState, sample: Sample,
                             sign: int) -> None:
        for no, agg in gs.checkpoints.items():
            weights = self._weights_locked(gs.group, no)
            pred = ml.predict(weights, sample.features)
            sq_err = (pred - sample.measured) ** 2
            if sign > 0:
                agg.add_contribution(sample.load, sq_err)
            else:
                agg.remove_contribution(sample.load, sq_err)

    # ------------------------------------------------------------------
    # 区间选档推理
    # ------------------------------------------------------------------
    def infer(self, load_start: int, load_end: int,
              features: list[int]) -> dict:
        # HTTP 入口由 Pydantic 做范围校验；引擎层保留防御性检查，
        # 便于不经 API 的直接调用同样安全。
        if type(load_start) is not int or type(load_end) is not int:
            raise CalibrationError("负载起止值必须为整数")
        if not (ml.LOAD_MIN <= load_start <= ml.LOAD_MAX) or not (
            ml.LOAD_MIN <= load_end <= ml.LOAD_MAX
        ):
            raise CalibrationError(
                f"负载起止值必须落在 {ml.LOAD_MIN}～{ml.LOAD_MAX}"
            )
        if load_start > load_end:
            raise CalibrationError("负载起始值不得大于终止值（闭区间）")
        if len(features) != ml.INPUT_DIM or not all(
            type(v) is int and ml.VALUE_MIN <= v <= ml.VALUE_MAX
            for v in features
        ):
            raise CalibrationError(
                f"四项特征必须均为 {ml.VALUE_MIN}～{ml.VALUE_MAX} 的整数"
            )
        with self._lock:
            # 快照引用：保证本次推理只看到受理时最新的已提交状态。
            gs = self.groups[self.active_group]
            group = self.active_group
            ordered = [self.registry[group][i]["checkpoint_no"]
                       for i in range(ml.CHECKPOINT_COUNT)]

            summaries: list[dict] = []
            read = 0
            for no in ordered:
                if read >= ml.MAX_SUMMARIES_READ:
                    break
                agg = gs.checkpoints[no]
                # 只读取区间汇总，不触碰 batches 中的样本明细。
                summaries.append(agg.interval_summary(load_start, load_end))
                read += 1

            calibrated = [s for s in summaries if s["sample_count"] > 0]
            if not calibrated:
                return {
                    "status": "uncalibrated",
                    "message": "请求负载区间内无校准样本，未校准",
                    "group": group,
                    "load_range": [load_start, load_end],
                    "features": list(features),
                    "summaries_read": read,
                    "candidate_count": 0,
                }

            min_mse = min(s["mse"] for s in calibrated)
            winners = [
                s for s in calibrated
                if s["mse"] - min_mse <= ml.MSE_TIE_TOL
            ]
            # 档号按登记顺序遍历，winners[0] 即并列中最先登记的候选。
            chosen = winners[0]
            entry = self.registry[group][chosen["checkpoint_no"] - 1]
            weights = entry["weights"]
            prediction = ml.predict(weights, features)
            return {
                "status": "ok",
                "group": group,
                "load_range": [load_start, load_end],
                "features": list(features),
                "summaries_read": read,
                "candidate_count": len(winners),
                "selection_tolerance": ml.MSE_TIE_TOL,
                "session_id": entry["session_id"],
                "checkpoint_no": chosen["checkpoint_no"],
                "mse": float(chosen["mse"]),
                "min_mse": float(min_mse),
                "prediction": prediction,
                "weights": [float(v) for v in weights],
            }

    # ------------------------------------------------------------------
    # 候选组整组更换
    # ------------------------------------------------------------------
    def begin_swap(self, target_group: str) -> dict:
        if not isinstance(target_group, str) or not target_group.strip():
            raise CalibrationError("目标组标识必须为非空字符串")
        return self._mutate(self._begin_swap_locked, target_group)

    def _begin_swap_locked(self, target_group: str) -> dict:
        if target_group not in self.registry:
            raise CalibrationError(
                f"候选组 {target_group} 不在检查点库中，换组准备失败，"
                "已生效状态不变"
            )
        if target_group == self.active_group:
            raise CalibrationError("目标组与当前活动组相同，无需换组")
        if self.pending_swap is not None:
            raise CalibrationError(
                "已存在待提交的换组准备，请先提交或撤除"
            )
        entries = self.registry[target_group]
        if len(entries) != ml.CHECKPOINT_COUNT or any(
            e["checkpoint_no"] != i + 1 for i, e in enumerate(entries)
        ):
            raise CalibrationError("目标组检查点登记不完整（需 1～16 档）")
        self.pending_swap = {"group": target_group}
        return {
            "swap_prepared": True,
            "active_group": self.active_group,
            "pending_group": target_group,
            "message": "换组准备中：当前组仍可更正与推理",
        }

    def commit_swap(self) -> dict:
        return self._mutate(self._commit_swap_locked)

    def _commit_swap_locked(self) -> dict:
        if self.pending_swap is None:
            raise CalibrationError("没有待提交的换组准备，提交失败")
        target = self.pending_swap["group"]
        if target not in self.registry:
            raise CalibrationError("目标组已失效，换组提交失败，状态保留")
        previous = self.active_group
        self.active_group = target
        self.groups.setdefault(target, self._fresh_group(target))
        self.pending_swap = None
        return {
            "swap_committed": True,
            "previous_group": previous,
            "active_group": target,
        }

    def abort_swap(self) -> dict:
        return self._mutate(self._abort_swap_locked)

    def _abort_swap_locked(self) -> dict:
        if self.pending_swap is None:
            raise CalibrationError("没有待撤除的换组准备")
        target = self.pending_swap["group"]
        self.pending_swap = None
        return {"swap_aborted": True, "active_group": self.active_group,
                "aborted_group": target}

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def overview(self) -> dict:
        with self._lock:
            gs = self.groups[self.active_group]
            batches = [
                {"batch_id": bid, "record_count": len(b.records)}
                for bid, b in sorted(gs.batches.items())
            ]
            return {
                "active_group": self.active_group,
                "pending_swap": deepcopy(self.pending_swap),
                "available_groups": list(self.registry.keys()),
                "total_samples": sum(len(b.records) for b in gs.batches.values()),
                "batches": batches,
                "checkpoint_summaries": [
                    {
                        "checkpoint_no": no,
                        "session_id": agg.session_id,
                        "sample_count": sum(r["count"]
                                           for r in agg.loads.values()),
                        "distinct_loads": len(agg.loads),
                    }
                    for no, agg in sorted(gs.checkpoints.items())
                ],
            }
