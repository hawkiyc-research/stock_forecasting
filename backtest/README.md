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
