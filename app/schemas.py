"""FastAPI 请求模型与入参校验（范围错误统一 422，状态冲突 409）。"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from . import ml

# 带范围约束的整数类型：Pydantic 层拦截即返回 422，不进入业务引擎。
LoadInt = Annotated[int, Field(ge=ml.LOAD_MIN, le=ml.LOAD_MAX)]
BoundedInt = Annotated[int, Field(ge=ml.VALUE_MIN, le=ml.VALUE_MAX)]
FeatureList = Annotated[
    list[BoundedInt],
    Field(min_length=ml.INPUT_DIM, max_length=ml.INPUT_DIM),
]


class SampleInput(BaseModel):
    sample_no: int = Field(..., gt=0, description="样本号（正整数，批次内唯一）")
    load: LoadInt = Field(..., description="整数负载（0～65535）")
    features: FeatureList = Field(..., description="四项整数特征（每项 ±100）")
    measured: BoundedInt = Field(..., description="整数实测功耗偏差（±100）")


class BatchRequest(BaseModel):
    batch_id: str = Field(..., min_length=1, description="校准批次标识")
    samples: list[SampleInput] = Field(..., min_length=1, description="本批样本")


class CorrectionRequest(BaseModel):
    batch_id: str = Field(..., min_length=1, description="原受理批次标识")
    sample: SampleInput = Field(..., description="整条更正后的样本（含负载）")


class InferenceRequest(BaseModel):
    load_start: LoadInt = Field(..., description="负载起始值（含）")
    load_end: LoadInt = Field(..., description="负载终止值（含）")
    features: FeatureList = Field(..., description="四项整数特征（每项 ±100）")

    @field_validator("load_end")
    @classmethod
    def _check_closed_interval(cls, value: int, info) -> int:
        start = info.data.get("load_start")
        if start is not None and value < start:
            raise ValueError("负载终止值不得小于起始值（闭区间，含两端）")
        return value


class SwapBeginRequest(BaseModel):
    target_group: str = Field(..., min_length=1, description="目标候选组标识")
