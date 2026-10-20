# stock_forecasting 本機回測 / Local Backtests

[繁體中文](#中文) · [English](#english)

## 中文

### 用途

在本機回測已匯出的模型訊號（逐樣本預測），評估扣除台股交易成本後的表現。此目錄與
RunPod 使用的 Poetry 環境完全分離：

- 不安裝主專案套件，也不修改主專案的 Poetry 設定或 `poetry.lock`。
- 不在 RunPod 同步清單內，也不在雲端程式碼核對的掃描範圍內。
- 只讀取主專案 `artifacts/` 下已下載的評估陣列與預測，這些檔案不受 Git 追蹤。

### 環境

環境由 uv 管理，使用 Python 3.12。虛擬環境位於 `backtest/.venv`，套件版本鎖定於
`backtest/uv.lock`。

建立或更新環境：

```bash
cd backtest
uv sync
```

執行測試：

```bash
uv run pytest
```

執行 lint：

```bash
uv run ruff check .
```

### 第一階段回測

第一階段依照 `docs/experiment_protocol.md` §13.4 的事先登記規則，以
`config/stage1.json` 的固定設定回測台股（TWSE、TPEX 與兩者合併）排序訊號：收盤後產生
訊號、次一交易日開盤買進、持有 h 個交易日後收盤賣出，每日選前 K 檔等權持有，並扣除手續費、
證交稅、當沖稅與滑價。主要設定只用 validation 選出，test 只用於最後檢驗。

```bash
uv run python -m twbacktest.stage1
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--config` | 回測設定檔 | `config/stage1.json` |
| `--artifacts-root` | 主專案 `artifacts/` 目錄 | `../artifacts` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/stage1-<UTC 時間>-<隨機碼>` |
| `--workers` | 平行 process 數，1 至 16 | 依 CPU 核心數與實體記憶體計算 |

也可以用環境變數 `BACKTEST_WORKERS` 指定 worker 數。輸出目錄包含：

- `report.json`：選出的主要設定、門檻判定、成本敏感度與每個訊號的描述性結果。
- `grid.csv`：所有訊號、市場、持有天數、K、版本與成本情境的指標。
- `summary.md`：中英雙語摘要表。

### 第一階段 1b 回測：持有緩衝

第一階段 1b 依照 `docs/experiment_protocol.md` §13.5 的事先登記規則，以
`config/stage1b.json` 的固定設定，檢驗持有緩衝能否以較低換手把排序優勢轉成扣除成本後的報酬：
資金分成 K 個等額位置，排名仍在前 b × K 的持股續抱，空出的位置補入排名最高的未持有股票；
所有買賣在下一個交易日開盤成交，並以開盤到開盤的報酬逐部位記帳。開盤跳空達漲跌停附近時
視為無法成交。執行前會檢查記帳恆等式，不符時停止。

```bash
uv run python -m twbacktest.stage1b
```

選項與第一階段相同（`--config` 預設為 `config/stage1b.json`，輸出目錄預設為
`artifacts/diagnostics/backtest/stage1b-<UTC 時間>-<隨機碼>`）。輸出目錄包含：

- `report.json`：記帳恆等式檢查、選出的主要設定、門檻判定、成本敏感度與每個訊號的描述性結果。
- `grid.csv`：所有訊號、市場、預測天數 s、K、緩衝倍數 b、版本與成本情境的指標。
- `daily.csv`：各市場主要設定在主要成本情境下的每日報酬序列。
- `summary.md`：中英雙語摘要表。

### 與持有方式對齊的評估

依照 `docs/experiment_protocol.md` §13.6 的事先登記規則，以 `config/holding_eval.json` 的固定
設定，用「次一交易日開盤買進、h 日後開盤賣出」的開盤到開盤報酬重評排序訊號：每日 Rank IC、
前後十分位相對等權的超額、前十分位在交易時段與隔夜的拆解，以及固定參數的持有組合
（沿用第一階段 1b 的引擎）。只作描述，不設門檻。

```bash
uv run python -m twbacktest.holding_eval
```

選項與第一階段相同（`--config` 預設為 `config/holding_eval.json`，輸出目錄預設為
`artifacts/diagnostics/backtest/holding-eval-<UTC 時間>-<隨機碼>`）。輸出目錄包含：

- `report.json`：記帳恆等式檢查與可留意清單。
- `ranking.csv`：每個分割、訊號、市場與天期的 Rank IC、十分位與交易時段／隔夜拆解。
- `portfolio.csv`：固定參數持有組合的指標。
- `summary.md`：中英雙語摘要表。

### 本機推論一致性比對

依照 `docs/experiment_protocol.md` §13.7，比較 `inference/` 的本機預測與 RunPod 匯出的參照預測
（設定見 `config/local_parity.json`）。本機預測以股票與日期對應到評估陣列，必須涵蓋全部台股列。

```bash
uv run python -m twbacktest.parity --local "<LOCAL_PREDICTION_DIR>"
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--local` | 已完成的本機預測目錄 | 必填 |
| `--config` | 比對設定檔 | `config/local_parity.json` |
| `--artifacts-root` | 主專案 `artifacts/` 目錄 | `../artifacts` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/parity-<UTC 時間>-<隨機碼>` |

輸出目錄包含 `report.json`（各判準、預測差距與本機執行紀錄）與中英雙語 `summary.md`。

依 §13.7 修訂，推論改以 RunPod GPU 匯出流程執行，這個比對只保留作診斷。

### checkpoint 選點

依照 `docs/experiment_protocol.md` §13.7 及其修訂，以 `config/checkpoint_selection.json` 的候選
checkpoint 與規則選點：每個 run 選 validation 上市 10 日前十分位開盤到開盤超額最高者（同分
選步數較晚者）。預測來自 RunPod 匯出（`artifacts/diagnostics/signal-predictions/`）。計分前會
核對每份匯出的列順序，以及 validation 選模分數與訓練紀錄 `metrics.jsonl` 是否一致；test 只計算
選出的 checkpoint。

先列出每個候選的匯出與核對狀態：

```bash
uv run python -m twbacktest.selection --check --training-records "<RUNPOD_ARTIFACTS_DIR>"
```

全部通過後執行選點：

```bash
uv run python -m twbacktest.selection --training-records "<RUNPOD_ARTIFACTS_DIR>"
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--training-records` | 含 `<RUN_ID>/metrics.jsonl` 的目錄，例如主專案的 `artifacts/runpod` | 必填 |
| `--check` | 只列出各候選的匯出與核對狀態 | 關閉 |
| `--config` | 選點設定檔 | `config/checkpoint_selection.json` |
| `--artifacts-root` | 主專案 `artifacts/` 目錄 | `../artifacts` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/selection-<UTC 時間>-<隨機碼>` |
| `--workers` | 平行 process 數，1 至 16 | 依 CPU 核心數與實體記憶體計算 |

輸出目錄包含 `report.json`（匯出核對、各候選的分數與選出結果）、`ranking.csv`、`portfolio.csv`
與中英雙語 `summary.md`。

### 第三階段前瞻檢驗

依照 `docs/experiment_protocol.md` §13.8、其修訂與修訂補充，以 `config/stage3.json` 評分前瞻母體。
執行前須先完成主專案 README「前瞻推論管線」的各步驟：`labels`、`compare-inputs`、上傳輸入包、
GPU 推論，以及加 `--forward` 的下載。

```bash
uv run python -m twbacktest.stage3
```

程式先執行全部檢查並記錄結果：輸入包雜湊（資料檢查 1）、前瞻台股列的記帳恆等式（2）、
`overlap` 標籤與共用評估快照（3）、`compare-inputs` 的結果與門檻（4），以及 R22-IR `overlap`
預測與 2026-10-19 test 匯出的管線一致性檢查。任一項未通過就不計算任何前瞻指標，exit code 為 1。
全部通過後，以 §13.6 的評估程式評分登記的訊號與選點結果，並只依主要假說（R22-IR、上市、10 日）
判定：前十分位超額的 95% 區間下界大於 0，且固定組合年化淨超額大於 0。後十分位與前後差只作描述。
某一組前瞻匯出評分過之後，再用另一組匯出評分會被拒絕。

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--config` | 第三階段設定檔 | `config/stage3.json` |
| `--artifacts-root` | 主專案 `artifacts/` 目錄 | `../artifacts` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/stage3-<UTC 時間>-<隨機碼>` |
| `--workers` | 平行 process 數，1 至 16 | 依 CPU 核心數與實體記憶體計算 |

輸出目錄包含 `report.json`（各項檢查、使用的匯出與判定）、通過檢查時的 `ranking.csv` 與
`portfolio.csv`，以及中英雙語 `summary.md`。

### Kronos 排序模型試行（§13.11）

依照 `docs/experiment_protocol.md` §13.11 與 `config/rank_pilot.json`，比較 `a-rank64` seed 42 與
§13.10 的 rank baseline。執行前在主專案完成三件事：訓練、在診斷 Pod 以
`export-signal-predictions --checkpoint "<RUN_ID>" --splits validation test --confirm-test` 匯出
validation 最佳 checkpoint，然後下載（`<RUN_ID>` 是訓練時印出的 run ID）：

```bash
bash scripts/runpod_workflow.sh download-signal-predictions "<RUN_ID>"
bash scripts/runpod_workflow.sh download-signal-predictions --rank-baseline-arrays
```

```bash
uv run python -m twbacktest.rank_pilot --kronos-run "<RUN_ID>"
```

程式先確認 Kronos 匯出與 baseline 陣列的列順序相同，再計算每個（市場、horizon、訊號日）的
Spearman Rank IC。每個模型在各市場、各 horizon 的平均值，必須重現訓練或匯出時記錄的值（差距
≤ 10⁻⁹），否則不輸出任何判定，exit code 為 1。通過後依設定計算各格（台股／US × 短／中／長
天期）的每日序列與 block bootstrap 區間、Kronos 與每個 baseline 的配對差，並在 test 上判定 H1–H3。

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--kronos-run` | `a-rank64` seed 42 的 run ID（必填） | — |
| `--config` | 試行設定檔 | `config/rank_pilot.json` |
| `--artifacts-root` | 主專案 `artifacts/` 目錄 | `../artifacts` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/rank-pilot-<UTC 時間>-<隨機碼>` |
| `--workers` | 平行 process 數，1 至 16 | 依 CPU 核心數與實體記憶體計算 |

輸出目錄包含 `report.json`（一致性檢查、使用的匯出與判定）、`daily.csv`（每日 Rank IC）、
`cells.csv`、`pairs.csv`、`per_horizon.csv`，以及中英雙語 `summary.md`。

## English

### Purpose

This directory backtests exported model signals (per-sample predictions) on the local machine
and measures performance net of Taiwan trading costs. It is fully separate from the Poetry
environment used on RunPod:

- It does not install the main package or change the main Poetry settings or `poetry.lock`.
- It is outside the RunPod synchronization list and the cloud code-readiness scan.
- It only reads evaluation arrays and predictions already downloaded under the main project's
  `artifacts/` directory, which Git does not track.

### Environment

uv manages the environment with Python 3.12. The virtual environment is `backtest/.venv`, and
package versions are locked in `backtest/uv.lock`.

Create or update the environment:

```bash
cd backtest
uv sync
```

Run the tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

### Stage 1 backtest

Stage 1 follows the pre-registered rules in `docs/experiment_protocol.md` §13.4 and uses the
fixed settings in `config/stage1.json` to backtest Taiwan ranking signals (TWSE, TPEX and both
combined). Signals are formed after the close; positions are bought at the next session's open
and sold at the close h sessions later. Each day the top K stocks are held with equal weights,
net of commission, transaction tax, day-trade tax and slippage. Primary configurations are
chosen on validation only; test is used once for the final check.

```bash
uv run python -m twbacktest.stage1
```

| Option | Description | Default |
| --- | --- | --- |
| `--config` | Backtest configuration file | `config/stage1.json` |
| `--artifacts-root` | Main project `artifacts/` directory | `../artifacts` |
| `--output-root` | Output directory for this run; it must not exist yet | `artifacts/diagnostics/backtest/stage1-<UTC time>-<random>` |
| `--workers` | Parallel processes, 1 to 16 | Derived from CPU cores and physical memory |

The `BACKTEST_WORKERS` environment variable also sets the worker count. The output directory
contains:

- `report.json`: the selected primary configurations, gate results, cost sensitivities and
  descriptive results for every signal.
- `grid.csv`: metrics for every signal, market, holding period, K, version and cost scenario.
- `summary.md`: a bilingual summary table.

### Stage 1b backtest: hold buffer

Stage 1b follows the pre-registered rules in `docs/experiment_protocol.md` §13.5 and uses the
fixed settings in `config/stage1b.json` to test whether a hold buffer turns the ranking edge into
returns after costs at lower turnover. Capital is split into K equal slots; held stocks still
ranked within the top b × K are kept, and empty slots take the best-ranked stocks not yet held.
All trades execute at the next session's open, and positions are accounted for with open-to-open
returns. Opening gaps near the price limits count as unfillable. The run checks the accounting
identity first and stops if it fails.

```bash
uv run python -m twbacktest.stage1b
```

The options match stage 1 (`--config` defaults to `config/stage1b.json`, and the output directory
defaults to `artifacts/diagnostics/backtest/stage1b-<UTC time>-<random>`). The output directory
contains:

- `report.json`: the accounting-identity checks, selected primary configurations, gate results,
  cost sensitivities and descriptive results for every signal.
- `grid.csv`: metrics for every signal, market, score horizon s, K, buffer multiple b, version
  and cost scenario.
- `daily.csv`: the daily return series of each market's primary configuration under the primary
  costs.
- `summary.md`: a bilingual summary table.

### Holding-aligned evaluation

Following the pre-registered rules in `docs/experiment_protocol.md` §13.6 and the fixed settings in
`config/holding_eval.json`, this re-evaluates ranking signals on open-to-open returns (buy at the
next session's open, sell at the open h sessions later): daily Rank IC, top and bottom decile
excess over equal weight, the top decile's split into sessions and overnight, and a
fixed-parameter holding portfolio (the stage 1b engine). It is descriptive only, with no gate.

```bash
uv run python -m twbacktest.holding_eval
```

The options match stage 1 (`--config` defaults to `config/holding_eval.json`, and the output
directory defaults to `artifacts/diagnostics/backtest/holding-eval-<UTC time>-<random>`). The
output directory contains:

- `report.json`: the accounting-identity checks and the watch list.
- `ranking.csv`: Rank IC, deciles and the session/overnight split for every split, signal, market
  and horizon.
- `portfolio.csv`: the fixed-parameter holding portfolio metrics.
- `summary.md`: a bilingual summary table.

### Local inference parity check

Following `docs/experiment_protocol.md` §13.7, this compares local predictions from `inference/`
with the RunPod reference export (settings in `config/local_parity.json`). Local predictions are
mapped onto the evaluation arrays by stock and date and must cover every Taiwan row.

```bash
uv run python -m twbacktest.parity --local "<LOCAL_PREDICTION_DIR>"
```

| Option | Description | Default |
| --- | --- | --- |
| `--local` | Completed local prediction directory | Required |
| `--config` | Parity configuration file | `config/local_parity.json` |
| `--artifacts-root` | Main project `artifacts/` directory | `../artifacts` |
| `--output-root` | Output directory for this check; it must not exist yet | `artifacts/diagnostics/backtest/parity-<UTC time>-<random>` |

The output directory contains `report.json` (each criterion, prediction differences and the local
run record) and a bilingual `summary.md`.

Under the §13.7 amendment, inference uses the RunPod GPU export workflow, and this check remains
for diagnosis only.

### Checkpoint selection

Following `docs/experiment_protocol.md` §13.7 and its amendment, this selects among the candidate
checkpoints of `config/checkpoint_selection.json`: each run keeps the checkpoint with the highest
validation TWSE 10-session top-decile open-to-open excess (ties go to the later step). Predictions
come from RunPod exports (`artifacts/diagnostics/signal-predictions/`). Before scoring, each
export's row order is checked, and its validation selection score is compared with the training
record in `metrics.jsonl`; test is computed for the selected checkpoints only.

First list each candidate's export and check status:

```bash
uv run python -m twbacktest.selection --check --training-records "<RUNPOD_ARTIFACTS_DIR>"
```

When every candidate passes, run the selection:

```bash
uv run python -m twbacktest.selection --training-records "<RUNPOD_ARTIFACTS_DIR>"
```

| Option | Description | Default |
| --- | --- | --- |
| `--training-records` | Directory holding `<RUN_ID>/metrics.jsonl`, such as the main project's `artifacts/runpod` | Required |
| `--check` | Only list each candidate's export and check status | Off |
| `--config` | Selection configuration file | `config/checkpoint_selection.json` |
| `--artifacts-root` | Main project `artifacts/` directory | `../artifacts` |
| `--output-root` | Output directory for this selection; it must not exist yet | `artifacts/diagnostics/backtest/selection-<UTC time>-<random>` |
| `--workers` | Parallel processes, 1 to 16 | Derived from CPU cores and physical memory |

The output directory contains `report.json` (export checks, every candidate's score and the
selection), `ranking.csv`, `portfolio.csv` and a bilingual `summary.md`.

### Stage 3 forward test

Scores the forward population with `config/stage3.json`, following `docs/experiment_protocol.md`
§13.8, its amendment and its supplement. First complete the steps of "Forward-inference
pipeline" in the main project README: `labels`, `compare-inputs`, the package upload, GPU
inference and the download with `--forward`.

```bash
uv run python -m twbacktest.stage3
```

Every check runs first and is recorded: the input package hashes (data check 1), the
open-to-open identity on forward Taiwan rows (2), `overlap` labels against the shared evaluation
snapshot (3), the `compare-inputs` result and thresholds (4), and the pipeline consistency check
of R22-IR `overlap` predictions against the 2026-10-19 test export. If any fails, no forward
metric is computed and the exit code is 1. When all pass, the §13.6 evaluation scores the
registered signals and the selection picks, and only the primary hypothesis (R22-IR, TWSE,
10 sessions) decides: the 95% interval lower bound of the top-decile excess is above 0 and the
fixed portfolio's annual net excess is above 0. The bottom decile and the spread are descriptive.
Once one set of forward exports has been scored, scoring a different set is refused.

| Option | Description | Default |
| --- | --- | --- |
| `--config` | Stage 3 configuration file | `config/stage3.json` |
| `--artifacts-root` | Main project `artifacts/` directory | `../artifacts` |
| `--output-root` | Output directory for this run; it must not exist yet | `artifacts/diagnostics/backtest/stage3-<UTC time>-<random>` |
| `--workers` | Parallel processes, 1 to 16 | Derived from CPU cores and physical memory |

The output directory contains `report.json` (every check, the exports used and the decision),
`ranking.csv` and `portfolio.csv` when the checks pass, and a bilingual `summary.md`.

### Kronos rank model pilot (§13.11)

Compares `a-rank64` seed 42 with the §13.10 rank baselines, following
`docs/experiment_protocol.md` §13.11 and `config/rank_pilot.json`. Before running it, the main project
trains the model, exports the validation-best checkpoint on a diagnostic Pod with
`export-signal-predictions --checkpoint "<RUN_ID>" --splits validation test --confirm-test`, and
downloads (`<RUN_ID>` is the run ID printed at training time):

```bash
bash scripts/runpod_workflow.sh download-signal-predictions "<RUN_ID>"
bash scripts/runpod_workflow.sh download-signal-predictions --rank-baseline-arrays
```

```bash
uv run python -m twbacktest.rank_pilot --kronos-run "<RUN_ID>"
```

The program first confirms that the Kronos export and the baseline arrays share one row order, then
computes the Spearman Rank IC per (market, horizon, signal date). Each model's mean per market and
horizon must reproduce the value recorded at training or export time (difference ≤ 10⁻⁹); otherwise
no decision is written and the exit code is 1. After that it computes each cell's daily series (Taiwan
or US × short, medium or long horizons) with block-bootstrap intervals, the paired differences between
Kronos and every baseline, and the H1–H3 decisions on test.

| Option | Description | Default |
| --- | --- | --- |
| `--kronos-run` | Run ID of `a-rank64` seed 42 (required) | — |
| `--config` | Pilot configuration file | `config/rank_pilot.json` |
| `--artifacts-root` | Main project `artifacts/` directory | `../artifacts` |
| `--output-root` | Output directory for this run; it must not exist yet | `artifacts/diagnostics/backtest/rank-pilot-<UTC time>-<random>` |
| `--workers` | Parallel processes, 1 to 16 | Computed from CPU cores and physical memory |

The output directory contains `report.json` (consistency check, exports used and decisions),
`daily.csv` (daily Rank IC), `cells.csv`, `pairs.csv`, `per_horizon.csv` and a bilingual
`summary.md`.
