"""测试夹具：每个测试使用独立临时数据目录与确定性检查点库。"""

from __future__ import annotations

import os
import sys

import pytest

# 让测试不依赖安装方式，直接从仓库根目录导入 app。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import ml  # noqa: E402
from app.calibration import CalibrationStore  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    library_dir = tmp_path / "checkpoint_library"
    registry = ml.ensure_library(str(library_dir))
    return CalibrationStore(str(tmp_path), registry)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """隔离数据目录的 TestClient（在导入 app.main 前重定向数据目录）。"""
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    # app.main 在导入时按环境变量初始化，先清掉可能的缓存模块。
    for mod in [m for m in list(sys.modules) if m.startswith("app.")]:
        if mod != "app":
            del sys.modules[mod]
    from fastapi.testclient import TestClient
    from app.main import app, STORE

    with TestClient(app) as c:
        c._test_store = STORE
        yield c
