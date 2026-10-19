# stock_forecasting 本機推論 / Local Inference

[繁體中文](#中文) · [English](#english)

## 中文

### 用途

在本機（Apple silicon 的 MPS 或 CPU）以已訓練的 Kronos 訊號 checkpoint 推論，產生回測需要的
逐樣本預測。訓練一律在 RunPod；這裡只做唯讀推論，不修改模型、checkpoint、RunPod 產物或
lifecycle 紀錄。規則與驗收標準見 `docs/experiment_protocol.md` §13.7。

此目錄與 RunPod 使用的 Poetry 環境及 `backtest/` 完全分離：

- 主套件以路徑依賴安裝；本目錄有自己的 `uv.lock`，不修改主專案的 Poetry 設定或 `poetry.lock`。
- 主套件需要 numpy < 2，`backtest/` 使用 numpy ≥ 2，因此兩個環境分開，以檔案交換結果。
- 只推論上市、上櫃列，每列記錄股票、日期與市場；`backtest/` 依股票與日期對應到評估陣列。
- 非 CUDA 裝置一律使用 float32；RunPod 的匯出是 bf16，兩者差異由一致性比對量測。

### 環境

環境由 uv 管理，使用 Python 3.12 與 torch 2.9.1。虛擬環境位於 `inference/.venv`。

```bash
cd inference
uv sync
```

執行測試：

```bash
uv run pytest
```

### 準備模型權重與評估資料

下載 checkpoint 設定中固定版本的 Kronos 基礎模型與 tokenizer（存入 Hugging Face 快取，可用
`HF_HOME` 指定位置）：

```bash
uv run python -m localinfer.fetch_models --checkpoint "<CHECKPOINT_DIR>"
```

在專案根目錄先列出評估資料的大小，再下載 validation／test 共用的 K 線庫與清理後的樣本索引：

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --evaluation-data --list
```

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --evaluation-data
```

資料存於被 Git 排除的 `artifacts/diagnostics/evaluation-data/<DATASET_ID>/prepared/`，每個檔案都以
SHA-256 核對；重新執行時保留已核對的檔案。

### 推論

```bash
uv run python -m localinfer.predict \
  --checkpoint "<CHECKPOINT_DIR>" \
  --bar-store "<DATASET_DIR>/prepared/bar-store" \
  --sample-universe "<DATASET_DIR>/prepared/sample-universes/<SAMPLE_UNIVERSE_ID>" \
  --split validation
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--checkpoint` | 本機 checkpoint 目錄，例如 `<RUN_ID>/checkpoint-018399` | 必填 |
| `--bar-store` | 本機評估 K 線庫 `prepared/bar-store` | 必填 |
| `--sample-universe` | 本機樣本索引 `prepared/sample-universes/<ID>` | 必填 |
| `--split` | `validation` 或 `test` | 必填 |
| `--device` | `auto`、`mps` 或 `cpu`；`auto` 有 MPS 時用 MPS | `auto` |
| `--batch-size` | 每批樣本數 | 128 |
| `--workers` | 資料讀取 worker 數，1 至 8；也可用 `LOCAL_INFERENCE_WORKERS` | 依 CPU 與實體記憶體計算 |
| `--output-root` | 輸出根目錄 | `artifacts/diagnostics/local-predictions` |

輸出目錄 `<RUN_ID>/<CHECKPOINT>/<SPLIT>-<UTC 時間>-<隨機碼>/` 包含：

- `predictions.npy`：float32 預測（alpha 為 `[筆數, 14]`，方向模型為 logits `[筆數, 14, 3]`）。
- `keys.parquet`：每列的股票、日期與市場。
- `targets_alpha.npy`：每列的 alpha 標籤，供核對。
- `run.json`：checkpoint 檔案雜湊、資料識別、裝置、速度與記憶體峰值。
- `status.json`：`complete` 才代表結果完整。

MPS 若不支援某個運算，改用 `--device cpu`；實際使用的裝置都必須通過一致性比對（見
`backtest/README.md`）。

## English

### Purpose

This directory runs trained Kronos signal checkpoints locally (MPS on Apple silicon, or CPU) to
produce the per-sample predictions that backtests need. Training always happens on RunPod; this is
read-only inference that changes no model, checkpoint, RunPod artifact or lifecycle record. Rules
and acceptance criteria are in `docs/experiment_protocol.md` §13.7.

It is fully separate from the Poetry environment used on RunPod and from `backtest/`:

- The main package is installed as a path dependency; this directory has its own `uv.lock` and does
  not change the main Poetry settings or `poetry.lock`.
- The main package needs numpy < 2 while `backtest/` uses numpy ≥ 2, so the environments stay apart
  and exchange results through files.
- Only TWSE and TPEX rows are inferred; each row records its stock, date and market, and
  `backtest/` maps rows onto the evaluation arrays by stock and date.
- Non-CUDA devices always use float32; RunPod exports use bf16, and the parity check measures the
  difference.

### Environment

uv manages the environment with Python 3.12 and torch 2.9.1. The virtual environment is
`inference/.venv`.

```bash
cd inference
uv sync
```

Run the tests:

```bash
uv run pytest
```

### Model weights and evaluation data

Download the Kronos base model and tokenizer at the revisions pinned by the checkpoint config into
the Hugging Face cache (`HF_HOME` sets its location):

```bash
uv run python -m localinfer.fetch_models --checkpoint "<CHECKPOINT_DIR>"
```

From the project root, list the evaluation data size first, then download the shared
validation/test bar store and its cleaned sample index:

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --evaluation-data --list
```

```bash
bash scripts/runpod_workflow.sh download-signal-predictions --evaluation-data
```

The data lands in Git-ignored `artifacts/diagnostics/evaluation-data/<DATASET_ID>/prepared/`; every
file is verified by SHA-256, and verified files are kept on a rerun.

### Inference

```bash
uv run python -m localinfer.predict \
  --checkpoint "<CHECKPOINT_DIR>" \
  --bar-store "<DATASET_DIR>/prepared/bar-store" \
  --sample-universe "<DATASET_DIR>/prepared/sample-universes/<SAMPLE_UNIVERSE_ID>" \
  --split validation
```

| Option | Description | Default |
| --- | --- | --- |
| `--checkpoint` | Local checkpoint directory, such as `<RUN_ID>/checkpoint-018399` | Required |
| `--bar-store` | Local evaluation bar store `prepared/bar-store` | Required |
| `--sample-universe` | Local sample index `prepared/sample-universes/<ID>` | Required |
| `--split` | `validation` or `test` | Required |
| `--device` | `auto`, `mps` or `cpu`; `auto` uses MPS when available | `auto` |
| `--batch-size` | Samples per batch | 128 |
| `--workers` | Data-loading workers, 1 to 8; `LOCAL_INFERENCE_WORKERS` also sets it | Derived from CPUs and physical memory |
| `--output-root` | Output root | `artifacts/diagnostics/local-predictions` |

The output directory `<RUN_ID>/<CHECKPOINT>/<SPLIT>-<UTC time>-<random>/` contains:

- `predictions.npy`: float32 predictions (`[rows, 14]` for alpha, logits `[rows, 14, 3]` for
  direction models).
- `keys.parquet`: each row's stock, date and market.
- `targets_alpha.npy`: each row's alpha label, for checks.
- `run.json`: checkpoint file digests, data identity, device, speed and peak memory.
- `status.json`: only `complete` marks a complete result.

If MPS lacks an operation, use `--device cpu`; whichever device is used must pass the parity check
(see `backtest/README.md`).
