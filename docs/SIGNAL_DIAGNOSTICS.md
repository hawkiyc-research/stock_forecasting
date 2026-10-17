# 預測訊號診斷 / Predictive signal diagnostics

## 繁體中文

本工具以獨立的診斷產物比較固定 train eval、完整 validation 與分類校準。它不更新模型權重、訓練採樣、原始行情、標籤、checkpoint、baseline 或正式選模結果。兩個原任務仍是 alpha 點估計與固定 ±1% 的個股三分類，1–14 日 loss 等權。Chronos-2 沿用既有的獨立樣本群組與原 head。

### 雲端匯出與固定 train eval

模型推論只在已有的 CUDA RunPod 執行，重用 `probe-scales` 的掛載檢查、GPU lease、tmux、逾時與診斷完成訊號。它不建立 Pod，也不執行 sync。仍需遵循原有 guard 的生命週期：診斷結束可觸發本機 guard 終止 Pod，不能把它當成與訓練共用 GPU 的背景工作。

在雲端專案目錄執行：

```bash
bash scripts/runpod_tmux_launch.sh export-signal-predictions \
  --checkpoint "<CHILD_RUN_ID>" \
  --splits train validation \
  --train-samples 16384 --sample-seed 59 \
  --audit-samples 24 --audit-workers 2 --max-export-gib 4
```

`--checkpoint` 可重複 1–16 次，逐個處理，省略 checkpoint 名稱即解析該 run 的 validation 最佳點；`RUN_ID/checkpoint-NNNNNN` 可匯出 leaderboard 仍保留的診斷點。獨立 signal 模型支援 Kronos／Chronos-2；其他架構需在其原分支匯出相同合約。`--splits` 預設僅 `validation`；`test` 必須另加 `--confirm-test`，本校準分析器不讀取 test。

Train 使用固定 seed 的無放回均勻樣本，預設 16,384，範圍 16–50,000；不是把全 train 展開，也不是年度 recency sampler 的分布。樣本原始 ordinal 與 membership 雜湊會保存。各模型使用同一 seed／母體，可對照 eval mode 下相同 loss；原訓練的年度 decay 與歷史 date-mixing metadata 分開記錄，不重新啟用混洗。原 train batch log 與此均勻 train eval 仍代表不同分布。

完整 validation／test 逐批推論，使用既有 DataLoader 的 CPU／記憶體／prefetch 規劃，多 worker、CUDA pinning 與既有非同步傳輸。只有 prediction arrays 保存完整 split；標籤與 metadata 重用既有共用 baseline cache。固定 train 樣本會保存小型標籤、metadata、drawdown 與 ordinals，不保存完整訓練視窗資料集。預設全 job 產物上限 4 GiB，`--max-export-gib` 可設 0.1–64；磁碟與 scratch 不足會拒絕當次 split。

`--audit-samples` 為每個 split 的案例上限，0–32，預設 24；0 可停用。先按市場挑選高 loss 且過去回撤至少 40% 的案例，再以穩定 hash 選同日同市場、過去回撤低於 20% 的一般對照，不依對照的未來報酬挑選。一般對照不足時記錄未配對，不偷偷改用其他日期。原始價格、vendor adjusted close、PIT 輸入、下一日 adjusted open、各天期 label 與內部重算誤差寫入 `audit.json`。這能驗證內部計算一致性，不能證明行情商 corporate actions 正確；不因稽核自動刪除極端樣本。

結果在 `/runpod-volume/diagnostics/signal-predictions/<run>/<checkpoint>/export-*/`。各 split 的 `split.json` 保存 shape、SHA-256、membership、原始 loss、分市場 Rank IC／IR（alpha）、讀取設定與產物清單。全 job 只有 `status.json=complete` 且雜湊正確才視為可分析結果。中斷會留 failed／partial 產物，不當作完成。

### 下載與共享資料

本機控制面可下載已存在的產物；以下指令不建立 Pod 或同步程式碼：

```bash
bash scripts/runpod_workflow.sh download-signal-predictions "<CHILD_RUN_ID>"
bash scripts/runpod_workflow.sh download-signal-predictions --baseline-arrays
```

第一個指令下載指定 run／checkpoint 的最新完成 export；多 checkpoint 時須指定完整 selector。第二個預設只下載目前 shared baselines 的 validation arrays 及 train 類別稽核；`--include-test` 才下載額外 test。檔案與大小依 manifest 核對，status 最後發布。傳輸 threads 預設 2，可用 `RUNPOD_SIGNAL_PREDICTION_DOWNLOAD_WORKERS=1..4` 調整，仍受 CPU／可用記憶體限制；逾時用 `RUNPOD_SIGNAL_PREDICTION_TRANSFER_TIMEOUT`，預設 1,800 秒。

### 時間隔離的分類校準比較

這個分析器只使用標準函式庫、CPU 與有限檔案映射，不匯入 NumPy／PyTorch，也不建立 Python 環境。可在保存產物的控制面或雲端執行；不得把它當成深度模型訓練。先固定日期與來源，再執行：

```bash
python3 -I scripts/analyze_signal_calibration.py \
  --baseline-root "<DIRECTION_BASELINE_CACHE_DIRECTORY>" \
  --model "candidate=<EXACT_COMPLETED_EXPORT_DIRECTORY>" \
  --calibration-end 2025-08-29 --score-start 2025-10-01 \
  --fit-rows 2048 --seed 59 --workers 2 --memory-budget-mb 512 \
  --max-iterations 80 --bias-penalty 0.1 --job-timeout 3600 \
  --output "<NEW_DIAGNOSTIC_DIRECTORY>"
```

`--baseline-root` 必須是完成的 direction baseline cache，包含 `complete.json` 及其 validation 檔案；下載後位於 `artifacts/diagnostics/signal-decomposition/baselines/<DIRECTION_BASELINE_ID>/`。`--model LABEL=DIR` 可重複最多 16 次，只接受明確指定、完成且 validation-selected best 的方向 export；GRU-42／GBDT-42 自動加入同一比較。所有輸入先核對 byte size／SHA-256／membership；只讀 `inputs/validation`，不靠目錄名稱猜最新結果。

`--calibration-end` 是最後可用的擬合 cutoff；`--score-start` 是較晚評分的第一天。每個市場兩者間須至少有 14 個已觀察到的交易日期，且兩段都要有樣本。缺失日期會讓防護更保守，不會縮短 label 成熟間隔。這不改動原 train／validation／test 切分。

固定比較四種結果：原始 logits、完整 train 的市場×天期固定先驗、一個全域 temperature、全域 temperature＋三類零和 bias（共三個自由參數）。後兩者只使用較早 validation 的同一份有界均勻樣本；`--fit-rows` 為 16–16,384 個 cutoff、預設 2,048，十四天期等權。Bias 的 L2 penalty 固定於 `--bias-penalty`，預設 0.1，必須正值；不根據較晚評分區段選 penalty。`--max-iterations` 限 1–500、預設 80；回溯最多 24 次，未收斂會明確記錄狀態。正值 temperature 不改 argmax，bias 才可能改變類別選擇。

每個模型都套用相同程序。完整較晚 validation 逐筆評分，報告等權天期 CE／Brier／BA／F1、最高類 ECE、逐類可靠度、分市場×天期與月份×天期結果；逐類 one-vs-rest AUC 使用 15 機率 bins 近似，清楚標註，不能當精確 AUROC。極端 logits 的 CE 用穩定 log-softmax，避免機率 underflow 隱藏過度自信。

每個檔案最多映射約 4 MiB 的視窗；不把完整 arrays 載入 RAM。`--workers` 預設 2、限 1–8，也可用 `SIGNAL_DIAGNOSTIC_WORKERS` 設定；依可見 CPU、cgroup／可用記憶體、`--memory-budget-mb`（64–4096，預設 512）及有界擬合點估算下調。CPU 工作使用 spawn processes，pending jobs 不超過 workers；錯誤、逾時（`--job-timeout` 1–21,600 秒）與取消會終止並回收 workers，不無限重試。

為限制分組報告記憶體，validation 最多 12 個月份；每個模型詳細結果完成即單獨寫檔，不在 parent 累積全部大型報告。新 `--output` 目錄保存 immutable `protocol.json`、`comparison.json`、`model-<label>.json` 與 `status.json`，記錄來源雜湊、相同擬合 ordinals、時間界線、校準參數與資源規劃，不自動挑選贏家、不改推論預設、不覆寫既有結果。既有 checkpoint 已看過完整 validation 做選模，所以這仍是事後時間診斷，不是新的獨立確認集；正式改善需在預先封存或後續時間段確認。不得用 test 擬合先驗、校準或改選 checkpoint。

## English

This independent diagnostic compares fixed train evaluation, complete validation and classification calibration. It leaves model weights, samplers, bars, targets, checkpoints, baselines and official selection unchanged. Raw alpha point prediction and fixed ±1% stock direction retain equally weighted 1–14-day losses. Chronos-2 retains its existing independent groups and head.

### Cloud export and fixed train evaluation

Model inference runs only inside an existing CUDA RunPod, reusing `probe-scales` mount checks, exclusive GPU lease, tmux, timeouts and diagnostic signals. It creates no Pod and performs no sync. The original local guard may terminate the Pod when the job finishes; this is not a background job to share with active training.

```bash
bash scripts/runpod_tmux_launch.sh export-signal-predictions \
  --checkpoint "<CHILD_RUN_ID>" --splits train validation \
  --train-samples 16384 --sample-seed 59 \
  --audit-samples 24 --audit-workers 2 --max-export-gib 4
```

Repeat `--checkpoint` 1–16 times for sequential processing. A child run resolves its validation best; `RUN_ID/checkpoint-NNNNNN` exports a retained leaderboard checkpoint. Independent Kronos/Chronos-2 signal checkpoints are supported; other architectures should export the same contract from their own branch. `--splits` defaults to validation only. Test requires `--confirm-test` and is never read by the calibration analyzer.

Train uses a fixed uniform sample without replacement, default 16,384, range 16–50,000, independent of candidate seeds. It is neither a materialized full train dataset nor the annual-recency sampling distribution. Original ordinals and membership digest are saved. Train eval uses eval mode and the same losses; original yearly decay and historical mixing metadata remain distinct and do not enable mixing.

Complete validation/test inference remains batched, using existing CPU/memory/prefetch planning, multiple loader workers, CUDA pinning and asynchronous transfers. Full splits persist predictions while labels/metadata are reused from the shared baseline cache. Only bounded train labels, metadata, drawdown and ordinals are copied. Total persisted artifacts default to a 4-GiB cap; `--max-export-gib` accepts 0.1–64. Insufficient disk/scratch refuses that split.

`--audit-samples` caps cases per split at 0–32, default24; zero disables it. High-loss cases with past drawdown≥40% receive stable-hash controls from the same date/market with past drawdown<20%, without using controls' future returns. Missing controls remain explicit. `audit.json` stores a bounded set of raw/adjusted prices, PIT inputs, next adjusted opens, horizon labels and independent reconstruction errors. Internal consistency does not independently verify vendor corporate actions or authorize dropping extreme returns.

Outputs live under `/runpod-volume/diagnostics/signal-predictions/<run>/<checkpoint>/export-*/`. Split records include shapes, SHA-256, membership, original loss, alpha Rank IC/IR, loader settings and artifact lists. Only complete status plus valid hashes authorizes analysis; failed/partial outputs do not count as completed results.

### Download and shared inputs

```bash
bash scripts/runpod_workflow.sh download-signal-predictions "<CHILD_RUN_ID>"
bash scripts/runpod_workflow.sh download-signal-predictions --baseline-arrays
```

Downloads use existing artifacts without creating Pods or syncing source. A bare run with multiple exported checkpoints is rejected as ambiguous. Baseline download defaults to validation arrays and embedded train class audit; `--include-test` explicitly adds test. Sizes/hashes are checked before publishing local completion. Transfer threads default to2, configurable with `RUNPOD_SIGNAL_PREDICTION_DOWNLOAD_WORKERS=1..4`, bounded by CPU/memory. `RUNPOD_SIGNAL_PREDICTION_TRANSFER_TIMEOUT` defaults to1,800 seconds.

### Temporally separated classification calibration

The CPU analyzer uses the standard library and bounded mapped windows without NumPy, PyTorch or environment creation. It may run beside saved artifacts locally or in the cloud; it is not deep-model training. Fix dates and sources before running:

```bash
python3 -I scripts/analyze_signal_calibration.py \
  --baseline-root "<DIRECTION_BASELINE_CACHE_DIRECTORY>" \
  --model "candidate=<EXACT_COMPLETED_EXPORT_DIRECTORY>" \
  --calibration-end 2025-08-29 --score-start 2025-10-01 \
  --fit-rows 2048 --seed 59 --workers 2 --memory-budget-mb 512 \
  --max-iterations 80 --bias-penalty 0.1 --job-timeout 3600 \
  --output "<NEW_DIAGNOSTIC_DIRECTORY>"
```

`--baseline-root` names the completed direction cache with `complete.json` and validation arrays, downloaded to `artifacts/diagnostics/signal-decomposition/baselines/<DIRECTION_BASELINE_ID>/`. Repeat `--model LABEL=DIR` up to16 times for exact completed validation-best direction exports. GRU-42/GBDT-42 receive the same procedure automatically. Sizes, hashes and ordered membership are checked; only `inputs/validation` is read.

`--calibration-end` bounds fitting cutoffs; `--score-start` begins later scoring. Each market must have at least14 observed trading dates strictly between them and support in both segments. Missing dates only lengthen the conservative maturity guard. Original dataset splits remain unchanged.

Methods are raw logits, fixed full-train market/horizon priors, one global temperature, and global temperature plus zero-sum three-class bias (three free parameters total). Fitting uses the same bounded uniform early-validation cutoff sample for every model: `--fit-rows`16–16,384, default2,048, with equal horizon weights. `--bias-penalty` is a fixed positive L2 value, default0.1, not selected on the later scoring segment. `--max-iterations`1–500 defaults to80; backtracking is capped at24 trials and nonconvergence is reported. Positive temperature preserves argmax; class bias can change it.

The complete later validation segment reports equal-horizon CE/Brier/BA/F1, maximum-confidence ECE, classwise reliability, market×horizon and month×horizon results. Classwise one-vs-rest AUC uses15 probability bins and is explicitly approximate. Stable log-softmax preserves exact extreme-logit CE rather than hiding confidence through probability underflow.

Each array maps approximately4 MiB at a time. `--workers` defaults to2, accepts1–8 or `SIGNAL_DIAGNOSTIC_WORKERS`, and is reduced by visible CPU, available/cgroup memory, `--memory-budget-mb`64–4096 (default512), and bounded fitting-point estimates. CPU jobs use spawn processes with at most one pending job per worker. Errors, cancellation and `--job-timeout`1–21,600 seconds terminate/reap workers without infinite retries.

Validation is bounded to twelve months; each model detail is written immediately rather than accumulated in parent memory. The new `--output` directory stores immutable protocol, comparison, per-model detail and status JSON with source hashes, identical fitting ordinals, temporal boundaries, parameters and resource plan. It automatically selects no winner and changes no inference defaults. Existing checkpoints were already selected using full validation, so this remains a post-hoc temporal diagnostic, not an independent confirmation set. Confirm predictive improvement on preregistered or subsequent periods; never fit priors/calibration or reselect checkpoints using test.
