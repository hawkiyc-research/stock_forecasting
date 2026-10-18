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
