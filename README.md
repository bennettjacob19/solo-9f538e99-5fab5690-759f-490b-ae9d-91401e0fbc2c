# 训练检查点功耗偏差校准与选档推理（FastAPI + NumPy）

用训练检查点预测设备功耗偏差时，按**当前负载段的实测均方误差**选择权重。
服务内置两组、每组 **16 份**训练检查点（`.npz`，含四特征系数 + 偏置共 5
个权重），在 `/docs` 按 1～16 顺序登记；校准样本分批录入，支持同批同号
整条更正（含跨负载段）与候选组整组更换，全部状态原子落盘、进程重启可恢复。

## 规则与实现对照

| 规则 | 实现 |
| --- | --- |
| `/docs` 按顺序登记 1～16 份检查点 | `GET /docs` 返回活动组 16 档登记册：档号、会话号、权重、样本数、整体 MSE |
| 分批录入校准记录 | `POST /calibration/batches`，批次号唯一、批内样本号唯一，整批校验失败全批驳回 |
| 样本字段：样本号、0～65535 整数负载、四项整数特征、整数实测值；特征/实测值 ±100 | Pydantic 范围校验（422）+ 引擎二次校验（409） |
| 同号可更正整条，包括负载 | `POST /calibration/corrections`，只读本批同号旧样本；撤旧贡献与计新贡献在同一把锁、同一次落盘内原子生效 |
| 推理填负载起止（含两端）与四项同格式特征 | `POST /inference`，闭区间 `[load_start, load_end]` |
| 从区间 MSE 与最低值相差 ≤ 1e-9 的候选中取最先登记者 | 按档号顺序遍历区间汇总，容差 `MSE_TIE_TOL=1e-9`，并列取最小档号 |
| 用选中检查点的权重预测，返回会话号、档号、误差、预测值 | 响应含 `session_id / checkpoint_no / mse / prediction / weights` |
| 区间无样本报未校准 | 返回 `status=uncalibrated`（HTTP 200，`accepted=false`） |
| 选档最多读 40 条区间汇总，返回实读条数，不读样本明细 | `MAX_SUMMARIES_READ=40`；推理路径只访问按负载聚合的 count/SSE 汇总，样本明细仅更正时按批读取；响应含 `summaries_read`（16 档时实读 16 条） |
| 候选可整组更换，准备期间仍可更正和推理 | `POST /candidate-groups/swap` 进入 pending（活动组不变），`/commit` 整体切换，`/abort` 撤除 |
| 每次推理的误差与权重均取自受理时最新已提交状态 | 所有变更在锁内完成内存修改 + 原子 `os.replace` 落盘后才可见；推理在锁内对当前已提交状态快照计算 |
| 更正或换组失败时保留已生效状态 | 变更前深拷贝快照，异常即回滚；业务冲突返回 409 且状态不变 |

## 目录结构

```
app/
  ml.py            检查点 .npz 读写、确定性双组检查点库（各 16 档）、线性预测
  calibration.py   校准引擎：批次受理、跨段更正、区间汇总选档、换组、JSON 持久化
  schemas.py       Pydantic 入参校验（负载 0～65535、特征/实测 ±100）
  main.py          FastAPI 路由（/docs 为业务登记册，/openapi.json 为机器接口描述）
tests/             pytest 测试 25 个（引擎、换组、HTTP 端到端、暴力复算交叉验证）
demo/run_demo.py   纯标准库端到端演示脚本
Dockerfile, docker-compose.yml
```

数据目录（默认 `./sessions_data`，容器内 `/data`）：

```
sessions_data/
  calibration_state.json                      原子持久化的校准状态
  checkpoint_library/
    group_a/<12位会话号>/checkpoint_001..016.npz
    group_b/<12位会话号>/checkpoint_001..016.npz
```

检查点库按固定种子确定性生成，缺失自动补齐、幂等；两组权重互不相同，
用于演示整组更换。

## 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务地址 http://localhost:8000 ，业务登记册 http://localhost:8000/docs
```

### 本地 Python

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 数据目录可用环境变量 APP_DATA_DIR 覆盖
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/docs` | 1～16 号检查点登记册（活动组，按登记顺序） |
| GET | `/state` | 活动组、pending 换组、批次与各档汇总概况 |
| POST | `/calibration/batches` | 分批录入校准记录（201 受理 / 409 冲突驳回） |
| POST | `/calibration/corrections` | 同批同号整条更正（含负载，原子生效） |
| POST | `/inference` | 区间 MSE 选档推理；无样本返回未校准 |
| POST | `/candidate-groups/swap` | 换组准备（pending，活动组不变） |
| POST | `/candidate-groups/swap/commit` | 提交换组（整组切换） |
| POST | `/candidate-groups/swap/abort` | 撤除换组准备 |
| GET | `/health` | 健康检查 |
| GET | `/openapi.json` | 机器可读接口描述 |

### 请求示例

录入校准批次：

```json
{
  "batch_id": "batch-001",
  "samples": [
    {"sample_no": 1, "load": 100, "features": [1, 2, 3, 4], "measured": 7},
    {"sample_no": 2, "load": 200, "features": [-1, -2, -3, -4], "measured": -7}
  ]
}
```

区间选档推理（负载闭区间，含两端）：

```json
{"load_start": 100, "load_end": 300, "features": [1, 2, 3, 4]}
```

响应（节选）：

```json
{
  "status": "ok",
  "group": "group_a",
  "summaries_read": 16,
  "candidate_count": 1,
  "session_id": "d826262749d1",
  "checkpoint_no": 7,
  "mse": 14.704007,
  "prediction": 6.914268,
  "weights": [0.12, -1.03, 0.44, 1.87, -2.20]
}
```

区间无样本：

```json
{"status": "uncalibrated", "accepted": false,
 "message": "请求负载区间内无校准样本，未校准", "summaries_read": 16}
```

## 演示与测试

```bash
python demo/run_demo.py           # 需先启动服务（可用 BASE_URL 改地址）
python -m pytest tests/ -q        # 25 个测试
```

## 选档与并发语义

* 每档维护按负载键聚合的 `count / SSE` 汇总行；推理时把落在请求闭区间内
  的负载行求和得到区间 MSE，全程不读取任何样本明细。
* 并列判定 `mse - min_mse <= 1e-9`（绝对容差），候选按档号 1→16 顺序扫描，
  故并列时必然返回最先登记的一档。
* 更正只在指定批次内查找同号样本；旧贡献按旧负载/旧特征撤除、新贡献按新
  值计入，连同明细替换在一次原子提交内完成。
* 换组准备期（pending）所有更正与推理仍作用于当前活动组；提交成功后新组
  立即生效且自带独立校准状态，旧组状态保留，可随时换回。
