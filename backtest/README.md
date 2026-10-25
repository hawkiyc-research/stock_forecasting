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

### Top-k dropout 回測（§13.13）

依照 `docs/experiment_protocol.md` §13.13 第 3 項的事先登記規則，以 `config/topk_dropout.json`
回測 walk-forward 排序分數（B0 LightGBM 或 D0／D1 cross-stock 模型，由 `download-lgbm`／
`download-xs` 下載的 `predictions.parquet`）：每個交易日依前一收盤的排序，賣出持股中排名最低的
n 檔、買進未持有中排名最高的股票，下一交易日開盤成交，資金分成 K 個位置並逐部位以開盤到開盤
記帳（沿用第一階段 1b 引擎）。TWSE K = 50、每日換 5 檔；TPEX K = 30、每日換 3 檔。score
horizon 在 {1, 5, 10} 中選 TWSE P1 淨超額 IR 最高者，之後固定用於 P2 與 P1+P2；成本情境與
第一階段相同（主情境折扣 0.6、滑價 ×1），區間為月份 block（21 個交易日）的 circular block
bootstrap 1,000 次（seed 11）。價格來自資料集的 raw `market.parquet`（以 S3 wrapper 下載到
`artifacts/datasets/<dataset>/raw/`）。

```bash
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--predictions` | walk-forward 預測檔 | 必填 |
| `--model` | 預測檔內的模型名稱（例如 `t-lgbm-alpha`、`t-xs-scratch`） | 必填 |
| `--prices` | raw `market.parquet` | 必填 |
| `--compare-predictions`、`--compare-model` | 第二個模型（例如 B0），H-B 的「IR 不低於 baseline」用 | 無 |
| `--config` | 回測設定檔 | `config/topk_dropout.json` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/topk-dropout-<UTC 時間>-<隨機碼>` |

輸出目錄包含 `report.json`（選定 horizon、H-B 判定）、`grid.csv`（所有市場、horizon、期間、成本情境與
版本的指標）、`daily_series.csv`（主情境下每個 horizon 的每日淨報酬與等權基準）與 `summary.md`
（中英雙語摘要）。H-B：TWSE P1+P2 主情境年化淨超額的 95% 區間下界大於 0，且（有比較對象時）
IR 不低於 baseline。

### 持有對齊的 walk-forward 回測（次要分析）

`twbacktest.holding_walkforward` 以 `config/holding_walkforward.json` 把 walk-forward 排序分數依標籤的
持有方式回測：前 K 名（10／20／50）在次一交易日開盤買進、h（1／2／3／5／10）個交易日後收盤賣出，
h 個起始 sleeve 輪流進場，沿用第一階段引擎與成本情境（h = 1 課當沖稅；主情境折扣 0.6、滑價 ×1、
保守換手）；市場分 TWSE、TPEX 與 TW（兩市場合併）。這不是 §13.13 事先登記的判定，只回答「排序優勢
在完全對齊標籤的持有方式下是否大於成本」；h、K 的選擇（TWSE P1 淨超額 IR 最高）屬事後分析。

```bash
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

選項與 Top-k dropout 相同，但沒有 `--compare-*`（`--config` 預設 `config/holding_walkforward.json`，
輸出目錄預設 `artifacts/diagnostics/backtest/holding-wf-<UTC 時間>-<隨機碼>`）。輸出目錄包含
`report.json`（選定的 h、K 及其 P2、P1+P2 指標）、`grid.csv`（所有市場、期間、h、K、成本情境與
版本）與 `summary.md`（中英雙語摘要）。B0 的結果見 `docs/lgbm_baseline_results.md`。

### 台股策略層：band 組合與台指期疊加（§13.14）

`twbacktest.tw_strategy` 依照 `docs/experiment_protocol.md` §13.14 的事先登記規則，以
`config/tw_strategy.json` 把 walk-forward 排序分數建成成本感知的多頭組合：流動性宇宙（20 日成交金額
中位數門檻、收盤價 ≥ 10 元、排除 ETF）、市場內百分位名次做每檔 EMA 平滑（半衰期 3 日）、買進／持有門檻
（進前 K 名買、掉出前 m × K 名賣）加每日換入上限 ⌈K／10⌉，沿用第一階段 1b 的逐部位開盤到開盤引擎；
另以資本 1,000 萬與 3,000 萬元做整張可行性敏感度。台指期疊加用期交所官方每日行情的 TX 近月連續序列
（`twbacktest.taifex download-contract`，第三個星期三開盤換月、每日以結算價 mark）：對沖版（放空
TX、15% 保證金準備、追繳與回補）、現金替代版（閒置現金持有 TX 多頭）與 TX 多頭參考。參數格子
h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3} 以 TWSE P1 多頭版淨超額 IR 選定；H-D 與 H-D-hedged 在
P1+P2 判定。報告欄位固定同時列出淨超額與絕對報酬（年化淨報酬、CAGR、Sharpe、最大回撤、對報酬指數超額）。

```bash
uv run python -m twbacktest.taifex download-contract --commodity TX --start 2022-12-01 --end-exclusive 2026-07-01 --root ../artifacts/diagnostics/taifex-tx
uv run python -m twbacktest.tw_strategy --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 選項 | 說明 | 預設值 |
| --- | --- | --- |
| `--predictions`、`--model` | 成對重複：預測檔與其中的模型名稱，每個模型分別選參數與判定 | 必填 |
| `--prices` | raw `market.parquet`（需含 `volume`） | 必填 |
| `--artifacts-root` | 台指期資料根目錄的上層（`futures.root` 相對於此） | `../artifacts` |
| `--workers` | spawn process pool 大小（`BACKTEST_WORKERS` 亦可） | 依 CPU 與記憶體 |
| `--config` | 回測設定檔 | `config/tw_strategy.json` |
| `--output-root` | 本次輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/tw-strategy-<UTC 時間>-<隨機碼>` |

輸出目錄包含 `report.json`（各模型選定參數、H-D／H-D-hedged 判定、前瞻候選）、`grid.csv`（所有模型、
市場、h、K、m、資本、版本、成本情境與期間）、`daily_series.csv`（選定參數在主情境下的每日序列）與
`summary.md`（中英雙語摘要）。B0 與 D0 的結果見 `docs/strategy_layer_results.md`。

### 股票期貨空方當沖（§13.15）與股票期貨宇宙診斷

`twbacktest.futures_daytrade` 依 `docs/experiment_protocol.md` §13.15 的事先登記規則，以 `config/futures_daytrade.json`
把 h = 1 分數變成股票期貨空方當沖：前一收盤在股票期貨宇宙（近月契約一般時段有成交、20 日平均成交量 ≥ 100 口）
內排名，放空後 q 分位，次日以期交所官方的期貨開盤價放空、當日結算價平倉；另有多空版與台指期對沖版。
成本為期交稅、每口手續費與依價位級距的 tick 滑價；資本情境取整口數、5% 成交量參與率上限、最高級距保證金。
需要先用 `twbacktest.taifex download` 抓全契約日資料（`config/capturability.json` 的 `taifex` 區段決定日期範圍；
本專案另以 2022-12 至 2026-06 建了 `artifacts/diagnostics/taifex-all/`）。

```bash
uv run python -m twbacktest.futures_daytrade --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

`--prices` 只用來計算期貨對現股的時點噪音統計；`--artifacts-root` 下需有 `diagnostics/taifex-all` 與
`diagnostics/taifex-tx`。輸出目錄包含 `report.json`（選定 q、H-E 判定、不可行的資本情境、時點噪音）、
`grid.csv`、`daily.csv`（主情境每日序列）與 `summary.md`。結果見 `docs/futures_daytrade_results.md`。

`twbacktest.futures_universe` 是描述性診斷：在全部、有股票期貨、期貨流動三個區段計算 h = 1 與 h = 10 的
每日 Rank IC 與前後十分位毛報酬（標籤定義的持有，未扣成本）。

```bash
uv run python -m twbacktest.futures_universe --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

`twbacktest.exposure_diagnostics` 是 `docs/experiment_protocol.md` §13.17 的步驟 0（描述性，含事前停止規則）：
以殘差化標籤模型的 h = 5／10 分數，在 §13.14 流動性宇宙與 §13.15 股票期貨流動宇宙內，計算對殘差目標與
原始報酬的 Rank IC、前後四分位價差，以及 plain／產業中性／規模中性三種金額中性多空組合的毛年化，並以
股票期貨成本（期交稅、每口手續費、1 tick、每日換 1/h 加每月換月）估年化成本拖累；停止規則為股票期貨宇宙
h = 10、P1 ∪ P2 的產業中性毛年化須達成本拖累的兩倍。需要同一 run 的 `targets.parquet`（`residual_hNN`
欄）與 `configs/reference/` 的公司參考表。輸出 `grid.csv`、`daily.csv`、`report.json`（含判定）與
`summary.md`。結果見 `docs/exposure_diagnostics_results.md`。

```bash
uv run python -m twbacktest.exposure_diagnostics --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --targets ../artifacts/lgbm/<launch-id>/targets.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --targets ../artifacts/xs/<launch-id>/targets.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

### 機構成本下的六情境回測（§13.19）

`twbacktest.scenario_suite` 是 `docs/experiment_protocol.md` §13.19 的資訊性分析：同一組 walk-forward 分數在
每個市場的流動性宇宙內建六個多空情境，每個情境都在 `config/cost_profiles.json` 的兩個成本檔（散戶、機構）
下計分並並列。S1 多日多空：多頭 band 組合（§13.14 規則）加上反向排序的空頭 band 組合，空頭位置每日等權
重設、機構另扣借券年費；S2 = S1 加滾動 β（組合自身毛報酬對指數，嚴格用前 60 個區間）的台指期疊加；
S3 當沖：t 收盤 h = 1 分數，t+1 開盤買前 q、放空後 q，收盤平倉，當沖稅 0.15%；S4 = S3 加台指期日內疊加；
S5／S6 = 多日與當沖各半（不含／含 TX）。參數（S1 的 h、K、m 與 S3 的 q）以 TWSE、P1、機構主要成本、
分數位置的淨 Sharpe 選定後固定；另報 TPEX、兩市場各半的 TW、滑價 0／1／2 倍，以及機構 3 億與 10 億
（多日：每檔名目超過 20 日成交金額中位數 5% 的候選跳過；當沖：名目封頂、超出留現金）。事前寫死的
「吸引力」門檻：機構主要成本、分數位置、P1 ∪ P2 淨 Sharpe ≥ 1.0、淨報酬區間下界 > 0、P2 淨報酬 > 0。
輸出 `grid.csv`（含 P1 選擇格子與全部最終列）、`daily.csv`（主要成本的每日序列）、`report.json`（選定
參數與門檻判定）與 `summary.md`。結果見 `docs/institution_cost_scenarios_results.md`。

```bash
uv run python -m twbacktest.scenario_suite --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 選項 | 說明 | 預設 |
| --- | --- | --- |
| `--config` | 情境設定檔（宇宙、格子、TX、區間、選擇與門檻規則；成本檔路徑在其中） | `config/scenario_suite.json` |
| `--predictions` / `--model` | 成對、可重複：walk-forward `predictions.parquet` 與其中的模型名 | 必填 |
| `--prices` | 資料集 raw `market.parquet` | 必填 |
| `--artifacts-root` | 主專案 `artifacts/`（讀 `diagnostics/taifex-tx`） | `../artifacts` |
| `--output-root` | 輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/scenario-suite-<UTC 時間>-<隨機碼>` |

設定檔可選的 `construction` 區塊（§13.19 補充二，`config/scenario_suite_v1.json`）：`score_variants` 取 `v0`
（平滑百分位分數）與 `v1`（每日在宇宙內對前 20／60 日報酬與 60 日 beta 取 OLS 殘差，特徵由
`exposure_features.trailing_features` 直接從 raw `market.parquet` 計算）；`leg_matching` 取 `none` 與 `momentum`
（依前 20 日報酬分 `momentum_quintiles` 個五分位，每個五分位各跑一對多空 band、K 平均分配、等權合併）；
`scenarios` 限制只算部分情境；`multi_day.fixed` 以模型名給定 (h, K, m)，略過 P1 選擇。每一列與門檻判定都帶
`score_variant`、`leg_matching` 欄；非基線組合的門檻鍵為 `市場|變體/配平`。`config/cost_profiles.json` 的 `participation_mode`
決定資本情境碰到參與率上限（位置金額 > 5% × 20 日成交金額中位數）時的處理：`scale`（2026-10-26 起的預設）把部位
封頂在上限、超出部分留現金並報告 `deployed_share` 與 `capped_share`；`skip` 是舊規則，整檔跳過換下一名（§13.19
原表格用的規則，會把組合推向排序後段的大型股）。`tw_strategy` 的 `capital.participation_mode` 同義，預設 `skip`
以維持 §13.14 的結果可重現。`construction.grid_score_variant`
讓 P1 選擇格子改用 v1 分數計算，`multi_day.grid_markets` 讓格子在多個市場各算一次（只報告，不改變選擇；
`config/scenario_suite_v1_grid.json` 即 §13.19 補充三的 V1 全格子）。每次執行另寫 `yearly.csv`：每條主要成本
日序列依曆年的複利淨報酬、毛報酬、年化 Sharpe、最大回撤、勝率與交易日數。

```bash
uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1.json --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

### 反轉暴露診斷（§13.19 補充）

`twbacktest.reversal_exposure` 把 S1 多空組合的結果拆成「對動能與 beta 的結構性暴露」與「扣掉暴露後剩下的
排序能力」：分數變體 v0 為 §13.14 的平滑百分位分數，v1 為每日在宇宙內把 v0 對前 20 日與前 60 日報酬和 60 日
beta 做 OLS 取殘差；建構為每日成批、持有 10 日的前後四分位金額中性多空（兩腿各占資本 100%，同 S1）：
`plain`、`momentum`（在前 20 日報酬的五分位內排序與配對）與 `turnover_low/mid/high`（依週轉率跳升三等分，
各自報 plain 價差）；目標為原始遠期報酬、原始 alpha 與 `targets.parquet` 的殘差目標。每列另報分數對三個特徵
的平均橫斷面 Spearman 相關、兩個成本檔（`config/cost_profiles.json`）下每批兩腿進出一次的年化成本拖累與淨
年化。`report.json` 的 `reading` 依協定第 6 項的事前規則給出 `exposure_is_main_cause`、
`model_is_reversal_only` 與 `conditioning_helps`。

```bash
uv run python -m twbacktest.reversal_exposure --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --targets ../artifacts/lgbm/<launch-id>/targets.parquet --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --targets ../artifacts/xs/<launch-id>/targets.parquet --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| 選項 | 說明 | 預設 |
| --- | --- | --- |
| `--predictions` / `--model` | 成對、可重複 | 必填 |
| `--targets` | 同一 run 的 `targets.parquet`（`residual_h10`），每個模型一個；省略時殘差目標欄為缺值 | 無 |
| `--prices` | 資料集 raw `market.parquet`（直接讀取成交量以算週轉率跳升） | 必填 |
| `--cost-profiles` | 成本檔 | `config/cost_profiles.json` |
| `--artifacts-root` | 只用來決定預設輸出目錄 | `../artifacts` |
| `--output-root` | 輸出目錄，必須尚不存在 | `artifacts/diagnostics/backtest/reversal-exposure-<UTC 時間>-<隨機碼>` |

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

### Top-k dropout backtest (§13.13)

Following the pre-registered rules of `docs/experiment_protocol.md` §13.13 item 3,
`config/topk_dropout.json` backtests walk-forward ranking scores (the B0 LightGBM or a D0/D1
cross-stock model, i.e. the `predictions.parquet` downloaded by `download-lgbm` / `download-xs`):
every session, by the previous close's ranking, the n lowest-ranked holdings are sold and the
highest-ranked non-holdings bought at the next open; capital sits in K slots with per-position
open-to-open accounting (the stage 1b engine). TWSE K = 50 with 5 replaced per day; TPEX K = 30
with 3. The score horizon is the one in {1, 5, 10} with the highest TWSE P1 net excess IR and is
then fixed for P2 and P1+P2; cost scenarios match stage 1 (primary: discount 0.6, slippage ×1);
intervals come from a monthly-block (21 sessions) circular block bootstrap with 1,000 replicates
(seed 11). Prices are the dataset's raw `market.parquet` (downloaded through the S3 wrapper to
`artifacts/datasets/<dataset>/raw/`).

```bash
uv run python -m twbacktest.topk_dropout --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Option | Description | Default |
| --- | --- | --- |
| `--predictions` | Walk-forward prediction file | required |
| `--model` | Model name inside the file (e.g. `t-lgbm-alpha`, `t-xs-scratch`) | required |
| `--prices` | Raw `market.parquet` | required |
| `--compare-predictions`, `--compare-model` | A second model (e.g. B0) for H-B's "IR not below the baseline" clause | none |
| `--config` | Backtest configuration | `config/topk_dropout.json` |
| `--output-root` | Output directory for this run; must not exist | `artifacts/diagnostics/backtest/topk-dropout-<UTC time>-<random>` |

The output directory holds `report.json` (selected horizon and the H-B verdict), `grid.csv`
(every market, horizon, period, cost scenario and version), `daily_series.csv` (daily net returns
and the equal-weight benchmark per horizon under the primary costs) and `summary.md` (bilingual).
H-B: the 95% interval lower bound of the TWSE P1+P2 annualized net excess under the primary
scenario is above 0 and, when a comparison model is given, the IR is not below the baseline's.

### Holding-aligned walk-forward backtest (secondary analysis)

`twbacktest.holding_walkforward` backtests walk-forward ranking scores the way the label is
defined, using `config/holding_walkforward.json`: the top K (10/20/50) is bought at the next
session's open and sold at the close h (1/2/3/5/10) sessions later, with h start sleeves taking
turns, the stage 1 engine and cost scenarios (h = 1 pays the day-trade tax; primary scenario
discount 0.6, slippage ×1, conservative turnover); markets are TWSE, TPEX and TW (both
combined). It is not a pre-registered decision of §13.13: it only answers whether the ranking
edge exceeds costs when the holding exactly matches the label, and the choice of h and K
(highest TWSE P1 net excess IR) is post hoc.

```bash
uv run python -m twbacktest.holding_walkforward --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

The options match top-k dropout without `--compare-*` (`--config` defaults to
`config/holding_walkforward.json`; the output directory defaults to
`artifacts/diagnostics/backtest/holding-wf-<UTC time>-<random>`). The output directory holds
`report.json` (the selected h and K with their P2 and P1+P2 metrics), `grid.csv` (every market,
period, h, K, cost scenario and version) and `summary.md` (bilingual). B0's result is in
`docs/lgbm_baseline_results.md`.

### Taiwan strategy layer: band portfolio with a TAIEX futures overlay (§13.14)

Following the pre-registered rules of `docs/experiment_protocol.md` §13.14, `twbacktest.tw_strategy`
turns walk-forward ranking scores into a cost-aware long-only portfolio with
`config/tw_strategy.json`: a liquidity universe (20-session median dollar-volume threshold,
close ≥ NT$10, no ETFs), within-market percentile ranks smoothed per symbol by an EMA (3-session
half-life), a buy/hold band (buy into the top K, sell when out of the top m × K) with a daily
entry cap of ⌈K/10⌉, on the stage 1b per-position open-to-open engine; board-lot feasibility is
reported at NT$10 million and NT$30 million of capital. The TAIEX futures overlay uses the
continuous TX front month built from TAIFEX's official daily data
(`twbacktest.taifex download-contract`, rolled at the third Wednesday's open and marked at the
daily settlement): a hedged version (short TX with a 15% margin reserve, calls and
reinvestment), a cash-equitized version (idle cash long TX) and a TX-long reference. The grid
h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3} is selected on the TWSE P1 long-only net excess IR; H-D
and H-D-hedged are judged on P1+P2. The report columns always pair the net excess with absolute
returns (annualized net return, CAGR, Sharpe, maximum drawdown, excess over the total-return
index).

```bash
uv run python -m twbacktest.taifex download-contract --commodity TX --start 2022-12-01 --end-exclusive 2026-07-01 --root ../artifacts/diagnostics/taifex-tx
uv run python -m twbacktest.tw_strategy --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Option | Description | Default |
| --- | --- | --- |
| `--predictions`, `--model` | Repeated pairs: a prediction file and the model name inside it; each model is selected and judged separately | required |
| `--prices` | Raw `market.parquet` (with `volume`) | required |
| `--artifacts-root` | Parent of the futures data root (`futures.root` is relative to it) | `../artifacts` |
| `--workers` | Spawn process pool size (`BACKTEST_WORKERS` also works) | sized from CPUs and memory |
| `--config` | Backtest configuration | `config/tw_strategy.json` |
| `--output-root` | Output directory for this run; must not exist | `artifacts/diagnostics/backtest/tw-strategy-<UTC time>-<random>` |

The output directory holds `report.json` (each model's selected parameters, the H-D / H-D-hedged
verdicts and the forward candidate), `grid.csv` (every model, market, h, K, m, capital, version,
cost scenario and period), `daily_series.csv` (the selected parameters' daily series under the
primary costs) and `summary.md` (bilingual). The B0 and D0 results are in
`docs/strategy_layer_results.md`.

### Stock-futures short day trade (§13.15) and the stock-futures universe diagnostic

Following the pre-registered rules of `docs/experiment_protocol.md` §13.15, `twbacktest.futures_daytrade`
turns h = 1 scores into a stock-futures short day trade with `config/futures_daytrade.json`: the
previous close ranks the stock-futures universe (near-month contract traded in the regular session,
20-session mean volume ≥ 100 contracts), the bottom q is shorted at the next session's official
TAIFEX futures open and covered at its daily settlement; long-short and TX-hedged versions are also
reported. Costs are the futures tax, a per-contract commission and tick slippage by price band;
capital scenarios use whole contracts, a 5% volume participation cap and highest-tier margin. Fetch
the all-contract daily data first with `twbacktest.taifex download` (the `taifex` section of
`config/capturability.json` sets the date range; this project built
`artifacts/diagnostics/taifex-all/` for 2022-12 to 2026-06).

```bash
uv run python -m twbacktest.futures_daytrade --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

`--prices` only feeds the futures-versus-stock timing statistic; `--artifacts-root` must hold
`diagnostics/taifex-all` and `diagnostics/taifex-tx`. The output directory holds `report.json`
(selected q, the H-E verdict, infeasible capital scenarios, timing noise), `grid.csv`, `daily.csv`
(primary-scenario daily series) and `summary.md`. Results are in `docs/futures_daytrade_results.md`.

`twbacktest.futures_universe` is a descriptive diagnostic: daily Rank IC and top/bottom decile gross
returns for h = 1 and h = 10 in three segments (all, with stock futures, liquid futures), on the
label's holding before costs.

```bash
uv run python -m twbacktest.futures_universe --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

`twbacktest.exposure_diagnostics` is step 0 of `docs/experiment_protocol.md` §13.17 (descriptive, with a
pre-registered stop rule): with the residual-label models' h = 5 / 10 scores, inside the §13.14 liquidity
universe and the §13.15 liquid stock-futures universe, it computes Rank IC and top-minus-bottom quartile
spreads against the residual target and the raw return, the gross annual return of plain / industry-neutral /
size-neutral dollar-neutral long-short books, and the annual cost drag under stock-futures costs (tax,
per-contract commission, one tick, 1/h of the book replaced daily plus a monthly roll); the stop rule requires
the industry-neutral gross in the stock-futures universe at h = 10 on P1 ∪ P2 to reach twice the cost drag.
It needs the same run's `targets.parquet` (`residual_hNN` columns) and the company reference table under
`configs/reference/`. Outputs: `grid.csv`, `daily.csv`, `report.json` (with the verdicts) and `summary.md`.
Results in `docs/exposure_diagnostics_results.md`.

```bash
uv run python -m twbacktest.exposure_diagnostics --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --targets ../artifacts/lgbm/<launch-id>/targets.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --targets ../artifacts/xs/<launch-id>/targets.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

### Six scenarios under institution costs (§13.19)

`twbacktest.scenario_suite` is the informational analysis of `docs/experiment_protocol.md` §13.19: the same
walk-forward scores build six long-short scenarios inside each market's liquidity universe, and every
scenario is scored under both cost profiles of `config/cost_profiles.json` (retail and institution), side
by side. S1 multi-day long-short: the long band portfolio (§13.14 rule) plus a short band portfolio on the
reversed ranking, each short slot reset to its value every session, with the institution borrow fee; S2 =
S1 plus a TX overlay sized by the book's rolling β (its own gross return on the index, strictly the previous
60 intervals); S3 day trade: h = 1 scores at the close of t, buy the top q and sell short the bottom q at the
open of t+1, close at the close, day-trade tax 0.15%; S4 = S3 plus an intraday TX overlay; S5 / S6 = half
multi-day and half day trade without / with the TX overlays. The parameters (h, K, m of S1 and q of S3) are
selected by net Sharpe on TWSE, P1, institution primary costs and fractional positions, then fixed; TPEX,
the half-and-half TW combination, slippage at 0 / 1 / 2× and the institution capital scenarios NT$300M and
NT$1B (multi-day: candidates whose slot exceeds 5% of the 20-session median dollar volume are skipped; day
trade: slots capped, the excess stays in cash) are reported as well. The pre-registered "attractive to the
industry" bar: institution primary costs, fractional positions, P1 ∪ P2 net Sharpe ≥ 1.0, net-return lower
bound > 0 and a positive P2 net return. Outputs: `grid.csv` (the P1 selection grid and every final row),
`daily.csv` (daily series under primary costs), `report.json` (selected parameters and the bar verdicts) and
`summary.md`. Results in `docs/institution_cost_scenarios_results.md`.

```bash
uv run python -m twbacktest.scenario_suite --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--config` | Scenario configuration (universe, grids, TX, intervals, selection and bar rules; the cost-profile path is inside it) | `config/scenario_suite.json` |
| `--predictions` / `--model` | In pairs, repeatable: a walk-forward `predictions.parquet` and the model name inside it | required |
| `--prices` | The dataset's raw `market.parquet` | required |
| `--artifacts-root` | The main project's `artifacts/` (reads `diagnostics/taifex-tx`) | `../artifacts` |
| `--output-root` | Output directory, must not exist yet | `artifacts/diagnostics/backtest/scenario-suite-<UTC time>-<random>` |

The optional `construction` block of the configuration (§13.19 supplement 2, `config/scenario_suite_v1.json`):
`score_variants` takes `v0` (the smoothed percentile score) and `v1` (its daily OLS residual inside the
universe on the trailing 20 / 60-session returns and the 60-session beta, features computed by
`exposure_features.trailing_features` straight from the raw `market.parquet`); `leg_matching` takes `none`
and `momentum` (`momentum_quintiles` quintiles of the trailing 20-session return, one long-short band pair
per quintile with K split equally, combined with equal weights); `scenarios` restricts the run to some
scenarios; `multi_day.fixed` gives (h, K, m) per model name and skips the P1 selection. Every row and every
bar verdict carries `score_variant` and `leg_matching`; non-baseline verdicts are keyed `market|variant/matching`.
`participation_mode` in `config/cost_profiles.json` sets what a capital scenario does when a slot exceeds the
participation cap (5% of the 20-session median dollar volume): `scale` (the default since 2026-10-26) caps the
position at the limit, leaves the rest of the slot in cash and reports `deployed_share` and `capped_share`;
`skip` is the old rule that skipped the name for the next candidate (used by the original §13.19 tables; it
pushes the book towards large names deep in the ranking). `capital.participation_mode` in `tw_strategy` is
the same switch, default `skip` so the §13.14 results stay reproducible. `construction.grid_score_variant`
computes the P1 selection grid on the v1 score and `multi_day.grid_markets`
repeats the grid on further markets (reported only, the selection is unchanged; `config/scenario_suite_v1_grid.json`
is the §13.19 supplement 3 v1 grid). Every run also writes `yearly.csv`: per calendar year of every primary-cost
daily series, the compounded net and gross return, annualized Sharpe, maximum drawdown, hit rate and interval count.

```bash
uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1.json --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

### Reversal-exposure diagnostic (§13.19 supplement)

`twbacktest.reversal_exposure` separates the S1 long-short result into the structural exposure to
momentum and beta and the ranking skill left after removing it. Score variants: `v0` is the §13.14
smoothed percentile score, `v1` its daily OLS residual inside the universe on the trailing 20- and
60-session returns and a 60-session beta. Constructions are daily cohorts held 10 sessions, top-minus-
bottom quartile, dollar-neutral with each leg at full capital like S1: `plain`, `momentum` (ranked and
paired inside quintiles of the trailing 20-session return) and `turnover_low/mid/high` (the plain
spread inside terciles of the turnover spike). Targets: the raw forward return, the raw alpha and the
residual target of `targets.parquet`. Every row also reports the mean cross-sectional Spearman
correlation of the score with the three features and, under both cost profiles of
`config/cost_profiles.json`, the annual cost drag of one entry and exit per leg per cohort and the net.
`report.json` carries the pre-registered reading of protocol item 6 (`exposure_is_main_cause`,
`model_is_reversal_only`, `conditioning_helps`).

```bash
uv run python -m twbacktest.reversal_exposure --predictions ../artifacts/lgbm/<launch-id>/predictions.parquet --model t-lgbm-alpha-residual_v1a --targets ../artifacts/lgbm/<launch-id>/targets.parquet --predictions ../artifacts/xs/<launch-id>/predictions.parquet --model t-xs-scratch --targets ../artifacts/xs/<launch-id>/targets.parquet --prices ../artifacts/datasets/<dataset>/raw/market.parquet
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--predictions` / `--model` | In pairs, repeatable | required |
| `--targets` | The same run's `targets.parquet` (`residual_h10`), one per model; without it the residual-target rows are NaN | none |
| `--prices` | The dataset's raw `market.parquet` (volume is read directly for the turnover spike) | required |
| `--cost-profiles` | Cost profiles | `config/cost_profiles.json` |
| `--artifacts-root` | Only used for the default output directory | `../artifacts` |
| `--output-root` | Output directory, must not exist yet | `artifacts/diagnostics/backtest/reversal-exposure-<UTC time>-<random>` |
