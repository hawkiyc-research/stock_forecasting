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
