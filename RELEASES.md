# 版本紀錄

## 中文

### 未發布 2026-10-22

- 台灣行情取得：除權息調整改在有界 spawn process pool 執行（每個 provider 一個
  pool，worker 數受 `--workers`、可見 CPU、可用記憶體與交易日數限制，
  `FIN_TS_MATERIALIZATION_WORKERS` 可設上限），抓取、快取重播與時間預算仍留在
  執行緒；產出與原本逐日路徑 byte-identical，資料內容識別碼（content identity）
  不變，既有資料集、provider checkpoint 與 bar store 全部沿用。
- 架構快照檢查改為比對 `configs/data_content_identity_snapshot.json` 記錄的
  語意 digest，而不是凍結整個模組的位元組；凍結的 CPU prepare／finalize 腳本
  檢查不變。
- 新增協定 §13.13 的 B0 `t-lgbm-alpha`：`stock-forecasting-walk-forward-lgbm` 在
  季度 walk-forward 的每個 fold、horizon、seed 訓練 LightGBM（以 fold validation
  的每日 Rank IC 早停），輸出逐筆預測、原始 alpha 目標、每日 Rank IC 與 cell
  摘要（block bootstrap 區間）；`runpod_workflow.sh lgbm`／`cpu-lgbm`／
  `download-lgbm` 為對應的 Pod、tmux 與下載入口，lifecycle 標記
  `lifecycle/research/lgbm.json`。
- 新增 §13.13 的 cross-stock transformer：`models/cross_stock.py`（patch 編碼器、inter-stock
  attention、market-state gating、MSE＋橫斷面相關 loss）、`cross_stock_training.py`
  （同日同市場 batch 的 walk-forward 訓練、B0 同格式輸出、D − B0 配對 bootstrap 與 H-A 判定）、
  `data/xs_cache.py`（每個 split 的視窗／特徵／標籤 memmap 快取）、`encoders.py`（凍結
  Kronos-small 與 Chronos-2-small 轉接器與 embedding 快取）；`runpod_workflow.sh xs-train`／
  `xs-train` tmux／`download-xs` 為 GPU Pod 入口，lifecycle 標記
  `lifecycle/research/xs-train.json`。
- 新增 §13.13 第 3 項的 Top-k dropout 回測 `twbacktest.topk_dropout`
  （`backtest/config/topk_dropout.json`：TWSE K = 50 每日換 5 檔、TPEX 30／3，horizon 依
  TWSE P1 淨超額 IR 選定，H-B 判定）與次要分析 `twbacktest.holding_walkforward`
  （`backtest/config/holding_walkforward.json`：依標籤方式持有的前 K 名回測）；兩者都讀
  `download-lgbm`／`download-xs` 的 `predictions.parquet` 與資料集的 raw `market.parquet`。
- B0 結果與回測判定記錄於 `docs/lgbm_baseline_results.md`：P1 ∪ P2 短格 Rank IC 0.118
  [0.109, 0.127]；Top-k dropout H-B 未通過（TWSE 年化淨超額 −6.3% [−19.0%, +7.1%]）。
- 新增 §13.14 的台股策略層 `twbacktest.tw_strategy`（`backtest/config/tw_strategy.json`：流動性宇宙、
  EMA 平滑、買進／持有門檻加每日換入上限、整張可行性敏感度、台指期對沖／現金替代／多頭參考與
  保證金會計）；`twbacktest.taifex download-contract` 從期交所官方下載單一契約（如 TX）並建連續近月
  序列（第三個星期三開盤換月、結算價 mark）。結果記錄於 `docs/strategy_layer_results.md`：換手降到
  每日 2%，但 H-D 與 H-D-hedged 都未通過（P2 訊號失效）。資本情境（1,000 萬整張／零股、3,000 萬、
  1 億、3 億、10 億）含零股成本、最低手續費與 5% 成交金額參與率上限；另有 `twbacktest.futures_universe`
  描述性診斷（股票期貨宇宙內的 Rank IC 與前後十分位毛報酬）。
- 新增 §13.15 的股票期貨空方當沖回測 `twbacktest.futures_daytrade`（`backtest/config/futures_daytrade.json`：
  以期交所官方期貨開盤價與結算價、tick 滑價、每口手續費、參與率上限、保證金與資本情境）與
  `taifex.stock_futures_sessions`（每個交易日的近月股票期貨契約表）；結果記錄於
  `docs/futures_daytrade_results.md`：毛報酬 +61–65%／年但 tick 滑價吃掉 125%，H-E 未通過。
- 新增 `scripts/fetch_tw_company_reference.py` 與 `configs/reference/tw_company_reference_20261022.csv`
  （證交所、櫃買 OpenAPI 公司基本資料快照：產業別、已發行股數），供 §13.16 殘差化標籤使用。
- 本機 guard 現在也會為 `research-xs-train` 的 GPU Pod 啟動 GPU 時脈監測。

### v0.2.2 2026-09-21

- Baseline 直接供應連續 tensor，保留動態 training sampling 與完整 lazy evaluation；
  以真實資料端到端吞吐共同調整 batch／workers／prefetch，每個 job 僅保留一個
  活躍 worker pool，提前關閉時有界排空 prefetch。
- Full GBDT／GPU jobs 共同 RAM 准入、待啟動 GPU jobs 的 CPU 預留，以及保守的
  clean-file-cache 折抵；經精確 hash 審核的 execution-only 修正保留既有 baseline ID。
- 完整雲端回歸 590 passed、97 subtests passed，pytest／Ruff exit 0；唯一
  Git-dependent skip 已在本機通過。加入逐測試／子工作持久化診斷及 tmux 驗證入口。
- 離線 Kronos CUDA 梯度、44,784 products 的完整 cutoff 稽核、百萬筆磁碟 backing
  指標聚合補驗通過。既有 inputs／GRU／DLinear checkpoint 及 prepared manifest 未改變。
- 限定真實輸入測速為 GRU 1.34 倍、DLinear 1.44 倍；GPU 平均使用率仍約 19%，
  不宣稱已滿載、完整訓練等比例加速或模型準確度改善。
  詳見 [tensor 資料管線驗收](docs/baseline_tensor_pipeline_validation.md)。

### v0.2.1 2026-09-21

- Baseline 向量化動態 windows／數值特徵，新增有界 Parquet metadata cache 與必要欄位
  解碼；同機 input 路徑對照由 52.51 秒降至 9.47 秒，不代表全量訓練加速倍數。
- cgroup CPU quota 與 host／GPU／shared-memory 准入；每個模型自動選擇 train／eval
  batch 與 prefetch，維持同 GPU／CPU 工作有界並行。
- 神經模型 epoch 內 sample-cursor 接續、tabular cache durable row cursor，以及
  GBDT horizon／quantile 中途保存；沿用既有 baseline／tmux CLI。
- 最終完整雲端回歸 573 passed、97 subtests passed；1 項 Git-dependent 檢查於
  本機通過。CPU prepare 與資料身份不變，不需新行情 API call。
- 已驗證數值等價與吞吐改善；仍有 input wait，未宣稱 GPU 完全滿載或正式訓練已完成。
  詳見 [baseline 執行效率驗收](docs/baseline_runtime_validation.md)。

### v0.2.0 2026-09-21

- Training 保留動態 sampling；validation／test 動態完整遍歷精確 cutoff ranges，
  不抽樣、不丟尾 batch、不預先展開巨大 input-window 資料集。
- 獨立 full-data baseline 建置、權重／指標快取及本機付費 Pod 啟動前的重用檢查；
  同 GPU 多模型與 CPU 工作有界並行。主模型訓練需先完成對應 baseline。
- Validation-driven plateau、市場感知、20 維尺度特徵與明確輸出尺度、同日同市場
  ranking loss、LoRA rank 32／alpha 64。沒有新增 ensemble。
- Inference-only batch tuning、多 worker／prefetch／pinned-memory 預算、輕量 spawn
  狀態及 baseline persistent pools；CPU prepare 的資料語意與既有資料集保持不變。
- 工程驗收：完整 suite 與 fixture 修正重測合併後，562 項雲端通過；1 項 Git-dependent
  檢查在本機通過。真實 Kronos CUDA 梯度、全 cutoff 稽核及百萬筆指標容量測試通過。
- 這是已完成限定工程驗收的架構快照，不是新模型準確度改善或實盤可用性的保證；
  尚未執行新架構的正式 A／B 全量訓練。詳見 [驗收紀錄](docs/performance_workflow_validation.md)。

### v0.1.0 2026-09-20

此 tag 保存 full-size baseline 改版前的模型架構與已完成的 A／B 評估證據。

- Kronos-base、LoRA rank 8／alpha 16、8 維歷史尺度分支與 benchmark 直接連接。
- 模型輸出 schema 5.0：1–14 個持有交易日的 q10／q50／q90 alpha。
- Train 截止 2025-06-01 exclusive；validation 為 2025-06-01 至
  2025-12-01 exclusive；holdout 為 2025-12-01 至 2026-06-01 exclusive。
- A：`run-20260915T091153Z-918316277`，2021 年起始，最佳 `checkpoint-043773`。
- B：`run-20260917T110811Z-1986920595`，2016 年起始，最佳 `checkpoint-094452`。
- 兩份報告位於 `reports/`。此版本的 benchmark 仍使用抽樣集合，不能解讀為完整
  validation／test 評估，也不能把兩個 run 當成相同股票—日期成員的配對比較。
- 自有程式碼與明示發布的新增模型成果採個人非商業授權；上游 Kronos 保留 MIT。

版本 tag 表示可追溯的程式與報告快照，不表示已通過實盤部署驗收。下載的權重與行情
資料不納入 Git，須以 run ID、checkpoint 與各自 manifest 追溯。

## English

### Unreleased 2026-10-22

- Taiwan acquisition applies corporate-action adjustments in a bounded spawn process
  pool (one pool per provider; workers limited by `--workers`, visible CPUs, available
  memory and the session count, capped by `FIN_TS_MATERIALIZATION_WORKERS`), while
  fetching, cache replay and the time budget stay in threads. Output is byte-identical
  to the former per-session path and the data content identity is unchanged, so every
  existing dataset, provider checkpoint and bar store remains valid.
- The architecture-snapshot check now compares the semantic digests recorded in
  `configs/data_content_identity_snapshot.json` instead of freezing whole module
  bytes; the frozen CPU prepare/finalize script checks are unchanged.
- Protocol §13.13 B0 `t-lgbm-alpha`: `stock-forecasting-walk-forward-lgbm` trains one
  LightGBM per quarterly walk-forward fold, horizon and seed (early stopping on the fold
  validation's daily Rank IC) and writes per-sample predictions, raw alpha targets, daily
  Rank IC and the cell summary with block-bootstrap intervals; `runpod_workflow.sh lgbm`,
  `cpu-lgbm` and `download-lgbm` are the Pod, tmux and download entries, with the
  lifecycle marker `lifecycle/research/lgbm.json`.
- Protocol §13.13 cross-stock transformer: `models/cross_stock.py` (patch encoder, inter-stock
  attention, market-state gating, MSE + cross-sectional correlation loss),
  `cross_stock_training.py` (walk-forward training on date/market batches, B0-compatible
  outputs, the D − B0 paired bootstrap and the H-A decision), `data/xs_cache.py` (per-split
  window/feature/label memory maps) and `encoders.py` (frozen Kronos-small and Chronos-2-small
  adapters with embedding caches); `runpod_workflow.sh xs-train`, the `xs-train` tmux workflow
  and `download-xs` are the GPU Pod entries, with the lifecycle marker
  `lifecycle/research/xs-train.json`.
- Protocol §13.13 item 3 top-k dropout backtest `twbacktest.topk_dropout`
  (`backtest/config/topk_dropout.json`: TWSE K = 50 with 5 replaced per day, TPEX 30/3, the
  horizon chosen by TWSE P1 net excess IR, the H-B decision) and the secondary analysis
  `twbacktest.holding_walkforward` (`backtest/config/holding_walkforward.json`: top-K held
  the way the label is defined); both read the `predictions.parquet` of `download-lgbm` /
  `download-xs` and the dataset's raw `market.parquet`.
- B0 results and backtest verdicts are recorded in `docs/lgbm_baseline_results.md`: P1 ∪ P2
  short-cell Rank IC 0.118 [0.109, 0.127]; the top-k dropout H-B fails (TWSE annualized net
  excess −6.3% [−19.0%, +7.1%]).
- Protocol §13.14 Taiwan strategy layer `twbacktest.tw_strategy` (`backtest/config/tw_strategy.json`:
  liquidity universe, EMA smoothing, buy/hold band with a daily entry cap, board-lot feasibility
  sensitivity, TAIEX futures hedged / cash-equitized / long-reference legs with margin
  accounting); `twbacktest.taifex download-contract` fetches one contract (e.g. TX) from the
  official TAIFEX download and builds the continuous front month (third-Wednesday open roll,
  settlement mark). Results in `docs/strategy_layer_results.md`: turnover falls to 2% a day,
  but H-D and H-D-hedged both fail (the P2 signal failure). Capital scenarios (NT$10M lots / odd
  lots, NT$30M, NT$100M, NT$300M, NT$1B) carry odd-lot costs, minimum commissions and a 5% dollar-
  volume participation cap; `twbacktest.futures_universe` is a descriptive diagnostic (Rank IC and
  top/bottom decile gross returns inside the stock-futures universe).
- Protocol §13.15 stock-futures short day trade `twbacktest.futures_daytrade`
  (`backtest/config/futures_daytrade.json`: official TAIFEX futures opens and settlements, tick
  slippage, per-contract commission, participation cap, margin and capital scenarios) and
  `taifex.stock_futures_sessions` (per-session near-month stock-futures table); results in
  `docs/futures_daytrade_results.md`: +61–65% gross a year, but tick slippage costs 125%, so
  H-E fails.
- `scripts/fetch_tw_company_reference.py` and `configs/reference/tw_company_reference_20261022.csv`
  (TWSE/TPEX OpenAPI company basic-data snapshot: industry code, issued shares) for the §13.16
  residualized labels.
- The local guard now starts the GPU clock monitor for `research-xs-train` GPU Pods too.

### v0.2.2 2026-09-21

- Direct contiguous baseline tensors with dynamic training sampling and exhaustive
  lazy evaluation; jointly tune batch/workers/prefetch on real end-to-end throughput.
  Keep one active worker pool per job and drain only bounded submitted prefetch.
- Joint full-GBDT/GPU RAM admission, CPU reservations for pending GPU jobs and
  conservative clean-file-cache credit. Exact hash-reviewed execution-only repairs
  retain the existing baseline identity.
- Full cloud regression: 590 tests and 97 subtests passed; pytest/Ruff exit 0. The
  sole Git-dependent skip passed locally. Add durable test/child-job diagnostics
  and the tmux regression entry point.
- Offline Kronos CUDA gradients, exhaustive cutoff auditing of 44,784 products and
  million-row disk-backed metric aggregation passed. Existing inputs, GRU/DLinear
  checkpoints and the prepared manifest are unchanged.
- Bounded real-input measurements: GRU 1.34x and DLinear 1.44x; average GPU utilization
  remains approximately 19%. These are not claims of saturation, proportional
  whole-training speedups or improved accuracy.
  See [tensor-pipeline acceptance](docs/baseline_tensor_pipeline_validation.md).

### v0.2.1 2026-09-21

- Vectorized lazy baseline windows/features, bounded Parquet metadata caching and
  numerical-column decoding. The same-machine input-path comparison improved from
  52.51 to 9.47 seconds; this is not a whole-training speedup measurement.
- cgroup CPU quotas and host/GPU/shared-memory admission; per-model train/eval
  batch and prefetch tuning, with bounded same-GPU/CPU concurrency.
- Mid-epoch neural sample-cursor resume, durable tabular-cache row cursors and
  GBDT horizon/quantile checkpointing, using the existing baseline/tmux CLI.
- Final full cloud regression: 573 passed and 97 subtests passed; one Git-dependent
  check passed locally. Prepared-data identity is unchanged; no new market calls.
- Numerical equivalence and throughput improvements are verified. Input wait remains;
  this is not evidence of full GPU saturation or completed production training.
  See [baseline runtime acceptance](docs/baseline_runtime_validation.md).

### v0.2.0 2026-09-21

- Preserve dynamic training sampling; exhaust exact validation/test cutoff ranges
  lazily, without sampling, dropping the final batch, or materializing input windows.
- Independent full-data baseline building, reusable weights/metrics and a local
  cache gate before paid Pod creation; bounded same-GPU experiments and CPU work.
  Main-model training requires a matching completed baseline.
- Validation-driven LR plateaus, market awareness, 20 scale features with explicit
  output scaling, same-date/market ranking loss, and LoRA rank 32/alpha 64; no ensemble.
- Inference-only batch tuning, memory-budgeted workers/prefetch/pinning, lightweight
  spawn state and persistent baseline pools. Prepared-data semantics remain unchanged.
- Bounded acceptance reconciles the full suite and repaired-fixture reruns: 562
  cloud passes and one Git-dependent check passed locally. Real Kronos CUDA gradients,
  exhaustive cutoff auditing and million-row metric aggregation passed.
- This is an engineering-validated architecture snapshot, not evidence of improved
  accuracy or live-trading readiness. Production A/B training remains outstanding;
  see the [acceptance record](docs/performance_workflow_validation.md).

### v0.1.0 2026-09-20

This tag preserves the architecture and completed A/B evaluation evidence before
the full-size baseline workflow revision.

- Kronos-base, LoRA rank 8 / alpha 16, eight historical-scale features, and a
  direct benchmark connection.
- Model output schema 5.0: q10/q50/q90 alpha for holding horizons 1–14.
- Train ends before 2025-06-01; validation is [2025-06-01, 2025-12-01);
  holdout is [2025-12-01, 2026-06-01).
- A: `run-20260915T091153Z-918316277`, history from 2021, best `checkpoint-043773`.
- B: `run-20260917T110811Z-1986920595`, history from 2016, best `checkpoint-094452`.
- Both reports are in `reports/`. This version still benchmarks sampled sets;
  these are not full validation/test evaluations or paired A/B evaluations of
  identical symbol-date members.
- Project-owned code and expressly released model additions use the personal
  non-commercial license. Upstream Kronos retains MIT licensing.

A version tag identifies a reproducible code/report snapshot, not production
trading approval. Downloaded weights and market data are excluded from Git;
identify them by run ID, checkpoint, and their respective manifests.
