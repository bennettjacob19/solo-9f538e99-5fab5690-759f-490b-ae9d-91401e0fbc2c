"""FastAPI 入口：训练检查点功耗偏差校准与区间选档推理服务。

端点：
* ``POST /calibration/batches``           分批录入校准记录
* ``POST /calibration/corrections``       同批同号整条更正（含跨负载段）
* ``POST /inference``                     区间均方误差选档并预测
* ``POST /candidate-groups/swap``         候选组整组更换：准备
* ``POST /candidate-groups/swap/commit``  提交换组
* ``POST /candidate-groups/swap/abort``   撤除换组准备
* ``GET  /docs``                          1～16 号检查点登记册（业务语义）
* ``GET  /state``                         当前校准状态总览
* ``GET  /health``                        健康检查

``/openapi.json`` 保留机器可读的接口描述；Swagger UI 让位给业务登记册。
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import ml
from .calibration import CalibrationError, CalibrationStore
from .schemas import (
    BatchRequest,
    CorrectionRequest,
    InferenceRequest,
    SwapBeginRequest,
)

DATA_DIR = os.environ.get("APP_DATA_DIR", os.path.join(os.getcwd(), "sessions_data"))
LIBRARY_DIR = os.path.join(DATA_DIR, "checkpoint_library")
os.makedirs(DATA_DIR, exist_ok=True)

# 启动时确保两组各 16 份检查点就位（确定性生成，幂等）。
REGISTRY = ml.ensure_library(LIBRARY_DIR)
STORE = CalibrationStore(DATA_DIR, REGISTRY)

app = FastAPI(
    title="训练检查点功耗偏差校准与选档推理",
    version="2.0.0",
    description=(
        "16 份训练检查点按档号登记；分批录入整数校准样本，按当前负载段"
        "实测均方误差选权重预测设备功耗偏差。支持同批同号跨段更正与"
        "候选组整组更换，更正/换组失败时保留已生效状态。"
    ),
    docs_url=None,
    redoc_url=None,
)


@app.exception_handler(CalibrationError)
async def _calibration_error_handler(_request, exc: CalibrationError):
    # 业务状态冲突/校验失败：409，且引擎保证抛出前未改动已生效状态。
    return JSONResponse(status_code=409,
                        content={"accepted": False, "message": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "checkpoint_groups": list(REGISTRY.keys())}


@app.get("/docs")
def docs_registry() -> dict:
    """1～16 号检查点登记册（按登记顺序），含会话号、权重与校准概况。"""
    return STORE.registry_docs()


@app.get("/state")
def state_overview() -> dict:
    return STORE.overview()


@app.post("/calibration/batches", status_code=201)
def ingest_batch(req: BatchRequest) -> dict:
    payload = [s.model_dump() for s in req.samples]
    try:
        return STORE.ingest_batch(req.batch_id, payload)
    except CalibrationError as exc:
        raise HTTPException(status_code=409, detail={
            "accepted": False, "message": str(exc),
        })


@app.post("/calibration/corrections")
def correct_sample(req: CorrectionRequest) -> dict:
    try:
        return STORE.correct_sample(req.batch_id, req.sample.model_dump())
    except CalibrationError as exc:
        raise HTTPException(status_code=409, detail={
            "accepted": False, "message": str(exc),
        })


@app.post("/inference")
def inference(req: InferenceRequest) -> dict:
    result = STORE.infer(req.load_start, req.load_end, list(req.features))
    if result.get("status") == "uncalibrated":
        # 区间无样本：业务上明确报“未校准”，HTTP 层仍为 200 并给出原因。
        result["accepted"] = False
    return result


@app.post("/candidate-groups/swap")
def begin_swap(req: SwapBeginRequest) -> dict:
    try:
        return STORE.begin_swap(req.target_group)
    except CalibrationError as exc:
        raise HTTPException(status_code=409, detail={
            "accepted": False, "message": str(exc),
        })


@app.post("/candidate-groups/swap/commit")
def commit_swap() -> dict:
    try:
        return STORE.commit_swap()
    except CalibrationError as exc:
        raise HTTPException(status_code=409, detail={
            "accepted": False, "message": str(exc),
        })


@app.post("/candidate-groups/swap/abort")
def abort_swap() -> dict:
    try:
        return STORE.abort_swap()
    except CalibrationError as exc:
        raise HTTPException(status_code=409, detail={
            "accepted": False, "message": str(exc),
        })
