# 版本紀錄

## 中文

### 未發布（2026-10-29 之後）

- 台股還原價：`TWSEProvider`／`TPExProvider` 的 `fetch_actions` 在除權日「開盤競價基準／開始交易基準價」與「除權息參考價」相差超過前收 0.5% 時改用基準價作價格因子（有償現金增資的基準價停在前收，參考價會製造不存在的跳動；2016 年起 861 列、誤差中位數 1.9%、最大 32%），`source` 加 `_opening_basis`、metadata 加 `opening_basis_rows`；`twse_official`／`tpex_official` 內容識別碼與 `configs/data_content_identity_snapshot.json` 更新，既有資料集不變，下一次資料準備產生新的資料集身分。發現、量化與對 §13.20 數字的敏感度見協定 §13.20 補充三。

### v0.4.0 2026-10-25

- 台股橫斷面排序主線（協定 §13.12–§13.17）完成並結案：B0 LightGBM 基線、D0 跨股票 transformer 與消融 A1、
  殘差化標籤變體 A／B、D1 凍結編碼器（Kronos-small、Chronos-2-small）、策略層 H-B／H-D／H-E／H-F 判定與
  §13.17 步驟 0 診斷。結論：模型有 h = 1 與殘差多日排序能力，但在台灣工具與成本下沒有勝過 0050 買進持有的
  可部署策略。結果見 `docs/residual_labels_results.md`、`docs/exposure_diagnostics_results.md` 與
  `reports/taiwan_rank_pipeline_results_20261025.docx`（`scripts/build_results_report.py` 產生）。
- `runpod_workflow.sh xs-train --labelVariant`：D0／D1 在殘差化標籤上重訓；變體的 rank 與殘差目標寫在跨股票快取旁
  的 `label-variants/<digest>/`，快取與 embedding 快取不重建；GPU 工作以 `label_variant` 配對同變體的 B0
  （`RUNPOD_XS_BASELINE_LAUNCH` 可指定）；`results.json` 新增 `cells_residual`、`paired_vs_baseline_residual`
  與 `seed_stability`（事前的單 seed 規則）。
- `scripts/compare_xs_runs.py` 在相同訊號日配對兩個 xs 輸出（D1 對 D0）；`scripts/score_label_variant.py` 計 H-F。
- 跨股票 trainer 移除每步的主機同步（epoch loss 留在 GPU、loss 細節回傳 tensor、群組索引常駐 GPU、驗證一次回傳）
  並改用 fused AdamW；模型與數字不變，每 epoch 13.4 秒降為 9.2 秒，訓練階段 GPU 利用率 25–33% 升到 44–48%。
- 修正：研究型 GPU Pod 沒有 W&B run ID 時 GPU 時脈監測無法啟動；tmux 包裝器對研究工作重寫失敗 marker 時誤帶
  run ID；Kronos-small adapter 把字串路徑交給原始碼驗證器。
- 新增 §13.17 步驟 0 診斷 `twbacktest.exposure_diagnostics`（殘差 alpha 幅度對股票期貨成本，含停止規則）。
- README 重組：目前主線（台股橫斷面排序 pipeline、操作手冊、完整 CLI 參數）移到前面，Kronos alpha／方向模型
  主線整節移入歷史實驗附錄；`MODEL_DEVELOPMENT_ROADMAP.md` 第 1、9 節補上結論與研究帳本。

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
- 新增 §13.16 的殘差化標籤 `data/residual_labels.py`（每個訊號日、市場內把原始 alpha 目標對截距、
  市值代理與產業虛擬變數回歸，殘差再 gauss-rank；shard 多出 `residual_hNN`，identity 含規格、參考表
  與實作，原標籤庫不變）；`stock-forecasting-build-features`／`walk-forward-lgbm` 新增 `--label-variant`，
  `runpod_workflow.sh features`／`lgbm` 新增 `--labelVariant`（Pod 環境 `RUNPOD_LABEL_VARIANT`）；
  LightGBM 在變體上命名為 `t-lgbm-alpha-<variant>`，對殘差目標與原始 alpha 各算 Rank IC，續跑只接受
  同一變體的折。規格 `configs/labels/residual_v1a.json`（成交金額代理）、`residual_v1b.json`（市值快照）。
- 本機 guard 現在也會為 `research-xs-train` 的 GPU Pod 啟動 GPU 時脈監測。

### v0.3.0 2026-10-16

- 一組實驗、兩個獨立模型：`a-signal64`／`b-signal64` 以同一 Pod 依序訓練 alpha 穩健點估計（normalized Smooth L1）
  與固定 ±1% 報酬帶的個股漲／平／跌分類（CE），各有獨立 Kronos／LoRA-64、optimizer、head 與 checkpoint；
  `-datemix` 對照只改訓練順序。父 run 管理成對子模型，`runs`／`status`／`download`／`resume` 支援父 ID。
- direction64 V1／V2／V3 多任務系列（成本感知雙方向、市場 residual V2 與 PCGrad、有序方向 V3 與市場狀態 gate）、
  adaptive64／adaptive128 與 LoRA 容量實驗，連同完整 CUDA 驗收與模型效能報告（`reports/`，舊版歸檔於
  `reports/deprecated/`）。
- RunPod 控制面：隔離的平行實驗與 run-scoped lifecycle、本機 GPU 降頻監控與通知、以 Pod ID 選擇或連接全部 Pod 的
  TTL SSH socket、run inventory 與完成狀態查詢、baseline readiness 對當前資料規則的核對、GPU telemetry 重連、
  過大 CUDA probe 的安全拒絕。
- 文件：完整雙語 README 稽核與重排、RunPod CLI 選項與機房 GPU 查詢、特徵路徑與保留架構的 CLI 流程、
  以證據連結的模型開發 roadmap 與交接文件（`MODEL_DEVELOPMENT_ROADMAP.md`）。

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

### Unreleased (after 2026-10-29)

- Taiwan adjusted prices: `fetch_actions` of `TWSEProvider` / `TPExProvider` uses the exchange's opening basis (開盤競價基準 / 開始交易基準價) as the ex-day price factor whenever it differs from the reference price by more than 0.5% of the prior close (for paid rights issues the basis stays at the prior close and the reference price invented a jump; 861 rows since 2016, median error 1.9%, largest 32%); `source` gains `_opening_basis`, metadata gains `opening_basis_rows`; the `twse_official` / `tpex_official` content identities and `configs/data_content_identity_snapshot.json` are updated, existing datasets are unchanged and the next data preparation gets a new dataset identity. Finding, quantification and the sensitivity of the §13.20 numbers: protocol §13.20 supplement 3.

### v0.4.0 2026-10-25

- The Taiwan cross-sectional rank line (protocol §13.12–§13.17) is complete and closed: the B0 LightGBM
  baseline, the D0 cross-stock transformer with ablation A1, residualized label variants A / B, the D1
  frozen encoders (Kronos-small, Chronos-2-small), the strategy-layer decisions H-B / H-D / H-E / H-F and the
  §13.17 step 0 diagnostic. Conclusion: the models rank at h = 1 and, on residual targets, over multiple
  days, but under Taiwan's instruments and costs no deployable strategy beats 0050 buy-and-hold. Results in
  `docs/residual_labels_results.md`, `docs/exposure_diagnostics_results.md` and
  `reports/taiwan_rank_pipeline_results_20261025.docx` (built by `scripts/build_results_report.py`).
- `runpod_workflow.sh xs-train --labelVariant`: D0 / D1 retrain on residualized labels; the variant's ranks
  and residual targets live beside the cross-stock cache under `label-variants/<digest>/`, so neither the
  cache nor the embedding caches are rebuilt; the GPU job pairs against the same-variant B0 by
  `label_variant` (`RUNPOD_XS_BASELINE_LAUNCH` overrides); `results.json` adds `cells_residual`,
  `paired_vs_baseline_residual` and `seed_stability` (the pre-registered single-seed rule).
- `scripts/compare_xs_runs.py` pairs two xs outputs on common signal dates (D1 vs D0);
  `scripts/score_label_variant.py` scores H-F.
- The cross-stock trainer drops its per-step host syncs (epoch loss on the GPU, loss details as tensors,
  group index resident on the GPU, one validation copy) and uses fused AdamW; model and numbers unchanged,
  13.4 → 9.2 s per epoch, training-phase GPU utilization 25–33% → 44–48%.
- Fixes: the GPU clock monitor could not arm for research GPU Pods without a W&B run ID; the tmux wrapper
  attached a run ID when rewriting a research failure marker; the Kronos-small adapter handed a string path
  to the source verifier.
- New §13.17 step 0 diagnostic `twbacktest.exposure_diagnostics` (residual alpha size against stock-futures
  costs, with the stop rule).
- README restructure: the current main line (the Taiwan rank pipeline, its operations guide and the complete
  CLI parameters) moves to the front and the Kronos alpha / direction line moves whole into the historical
  appendix; `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9 record the conclusions and the research ledger.

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
- Protocol §13.16 residualized labels `data/residual_labels.py` (within each signal date and
  market the raw alpha targets are regressed on an intercept, a size proxy and industry dummies
  and the residuals gauss-ranked; shards add `residual_hNN`, the identity includes the spec, the
  reference table and the implementation, and the existing stores are untouched);
  `stock-forecasting-build-features` / `walk-forward-lgbm` gain `--label-variant` and
  `runpod_workflow.sh features` / `lgbm` gain `--labelVariant` (Pod env `RUNPOD_LABEL_VARIANT`);
  LightGBM on a variant is named `t-lgbm-alpha-<variant>`, scores Rank IC against both the residual
  and the raw alpha targets, and resumes partial folds only from the same variant. Specs
  `configs/labels/residual_v1a.json` (dollar-volume proxy) and `residual_v1b.json` (market-cap
  snapshot).
- The local guard now starts the GPU clock monitor for `research-xs-train` GPU Pods too.

### v0.3.0 2026-10-16

- One experiment, two independent models: `a-signal64` / `b-signal64` train the robust alpha point estimate
  (normalized Smooth L1) and the fixed ±1% band stock down/neutral/up classifier (CE) in sequence on one Pod,
  each with its own Kronos / LoRA-64, optimizer, head and checkpoints; the `-datemix` controls change only
  the training order. A parent run manages the paired children; `runs` / `status` / `download` / `resume`
  accept the parent ID.
- The direction64 V1 / V2 / V3 multitask series (cost-aware dual direction, market-residual V2 with PCGrad,
  ordinal V3 with market-state gates), the adaptive64 / adaptive128 and LoRA capacity experiments, with full
  CUDA acceptance and the model performance reports (`reports/`; earlier versions archived under
  `reports/deprecated/`).
- RunPod control plane: isolated parallel experiments with run-scoped lifecycles, local GPU throttling
  alerts, TTL SSH sockets selected by Pod ID or connecting every Pod, run inventory and completion queries,
  baseline readiness checked against the current data rules, GPU telemetry reconnection, safe rejection of
  oversized CUDA probes.
- Documentation: the audited bilingual README reorganization, RunPod CLI options and regional GPU queries,
  feature paths and retained-architecture CLI workflows, and the evidence-linked model development roadmap
  and handoff (`MODEL_DEVELOPMENT_ROADMAP.md`).

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
