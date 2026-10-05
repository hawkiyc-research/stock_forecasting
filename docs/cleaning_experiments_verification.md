# 資料清理與受控容量實驗驗收 / Data Cleaning and Controlled Adaptation Verification

## 中文

### 驗收範圍與環境

日期：2026-10-06（Asia/Taipei）。本次驗收涵蓋完整市場交易日序列、最低流動性、
共同評估來源、逐年 sampling 權重、分離的預測位置／區間寬度／ranking、validation-only
區間校準、六個 A/B 容量設定，以及持久化 loss 與成果下載。

雲端使用一個隔離的 RTX 5090 Pod，Python 3.12.3、PyTorch 2.9.1+cu128。
程式、專案 `.venv`、測試輸出與清理索引均放在 Pod-local `/tmp/stockqa`；
既有 network volume 僅供讀取行情和已快取的 Kronos 權重。
沒有啟動其他正式工作、下載行情、執行 CPU prepare 或全量 baseline 訓練。

Pod 於約 2026-10-05 22:19 UTC 建立，23:04 UTC 由本機刪除；REST 回傳
`deleted: true`，隨後帳號 Pod 清單為空。運行約 45 分鐘，以 US$0.99／小時計算，
GPU 租用費估算約 US$0.75，低於一小時／US$1.50 的驗證上限；此數字不是帳單結算。

### 既有資料全量掃描

兩份 snapshot 的所有候選 cutoff 均經過 vectorized 檢查；以六個受記憶體限制的
worker 分 bucket 執行，只產生 compact cutoff ranges，不展開或複製全部 windows。

| 正式實驗使用的集合 | A | B |
|---|---:|---:|
| Training 有效 windows | 4,710,117 | 10,616,448 |
| Validation 有效 windows | 528,315 | 528,315 |
| Test 有效 windows | 534,327 | 534,327 |

A training 起點為 2021-01-01，B 為 2016-01-01；training label 必須在 2025-06-01
前結束。Validation 為 `[2025-06-01, 2025-12-01)`，test 為
`[2025-12-01, 2026-06-01)`，每筆 label 均須落在自己的 split 內。

表中的 validation/test 均為同一份 B snapshot 的同一組清理 ranges，並非各自掃描
後取相同筆數或交集。A 自有 snapshot 的舊評估來源仍存在，但不作為正式共同評估來源。
缺少 run selection 或共同來源時，正式評估明確報錯，不回退到 training snapshot。

完整掃描亦記錄每 split／market 的排除原因。B training 的 39,198,475 個候選
windows 中，排除 6,600,044 個含市場交易日缺漏的 windows、13,674,494 個含無效
OHLCV／零成交量的 windows、7,781,157 個歷史流動性不足的 windows、460,821 個含
非市場交易日 bar 的 windows，以及 65,511 個 label 跨 split 邊界的 windows。
這些計數依第一個不通過的規則歸類，不重複加總；不是排除的股票檔數。

真實極端報酬未按幅度刪除：A/B training 分別保留 147,745／381,865 個包含
`abs(adjusted log return) > 0.5` 的 windows，共同 validation/test 分別保留
17,921／15,938 個。這只是診斷門檻，不是對所有異常來源的人工認證。

除全量 cutoff 規則檢查外，每份 snapshot、每個 split 另取 128 個分層位置，
實際經過 lazy dataset 讀取，驗證 128 個 input 日期、下一市場交易日進場及
第 14 日 label 結束日期。原 bar-store manifest／success marker 在掃描前後雜湊相同。
原 CPU preparation identity 所涵蓋的 10 個 source paths 均未修改；A/B dataset request
identity 不因三種模型容量設定而變動。

### 測試與 CUDA 驗收

| 檢查 | 結果 |
|---|---|
| 雲端完整 pytest，兩個 worker | 793 passed、234 subtests passed、3 skipped、0 failed |
| 全專案 Ruff | 通過 |
| Python／Bash 靜態語法 | 52 個 Python、9 個 Bash 變更檔通過 |
| macOS Bash 3.2 同步控制回歸 | 5 個測試通過 |
| 本機歷史 baseline／報告與 preparation identity 檢查 | 補驗雲端三個 skipped 項目，均通過 |
| 新／舊結果下載 | 新 run 必需日誌及校準不可漏抓；舊 run 不需不存在的新產物；403 不被當成 404 |
| Sampling／接續訓練 | 年度配額、seed、位置還原及中斷後重探 batch 的權重一致性通過 |
| Full evaluation | synthetic 全遍歷、無重複、跨 worker／batch 一致及 raw/calibrated 結果保留通過 |

三個 skipped 項目是雲端 source-only 部署沒有 `.git`，以及未上傳本機已下載的
歷史模型報告與 A/B baseline 完成紀錄；對應測試均另在本機以 stdlib 執行。
本機未建立或檢查 ML Python 環境，亦未執行 Poetry lock/install。
完整測試仍包含 PyTorch 現有 `pin_memory(device)` deprecation warnings，未隱藏警告。

另以真實 Kronos-base／tokenizer 權重、八筆真實清理後 windows 和雙 worker
DataLoader，逐一檢查三種架構的 CUDA forward/backward、finite gradients、
optimizer step、trainable state 與 optimizer 儲存／還原：

| 配置 | 全模型可訓練參數總數 | 結果 |
|---|---:|---|
| LoRA rank 32 | 16,076,633 | 通過；還原後預測逐值相同 |
| LoRA rank 64 | 21,950,297 | 通過；還原後預測逐值相同 |
| 前 10 層 LoRA-32，最後 2 層與 final norm 解凍 | 30,869,913 | 通過；解凍層有 gradient，還原後預測逐值相同 |

上述參數總數包含 task heads 等可訓練組件，不是單指 LoRA 參數。
Tokenizer 仍凍結，解凍 block 不重複套 LoRA。三種模型依序測試，避免同時持有多份
Kronos 和 optimizer state；這不是把正式訓練改成多 GPU。

### 使用與結論邊界

- Training 保留動態 sampling，每個 window 的相對抽樣權重為
  `0.8 ** (最新訓練年份 - cutoff 年份)`；validation/test 不使用此 sampling。
- 六個 YAML 對應 `a-lora32`、`a-lora64`、`a-partial`、`b-lora32`、`b-lora64`、
  `b-partial`。一次 sync 後以 `train --experiment` 切換，只發布 selection，不重傳原始碼。
- 本次清理改變 baseline 的實際 training 樣本，因此 A/B 各需建立一份新 baseline；
  只切換容量不重建 baseline。原結果和權重不刪除、不改寫，也不冒充新集合的公平基準。
- 新架構實驗從 pretrained Kronos-base 開始；不能接續數值契約不同的舊 checkpoint。
- 完整實際資料掃描不等於對全部實際 windows 做模型 forward。此次未跑六組正式訓練
  或全量實際 holdout 推論，不能據此宣稱方向一致率、correlation 或 coverage 已改善。
- 校準僅使用 validation，raw 與 calibrated test 指標同時保留；不保證未來 coverage
  必定 80%。訓練 total loss 含 ranking，分析 generalization gap 應比較
  `train/pinball_loss` 與 `validation/loss`。

原始驗證證據存於本機被 Git 排除的 `.runpod/verification/cleaning-20261006/`。
驗收包 SHA-256 為 `de9d04a1c4dce92554adb0c13bab36c74e5898783cd88b40138f9dd1bc9c0067`；
包含完整 pytest XML／log、Ruff、全量清理稽核與真實 Kronos 測試紀錄。

## English

### Scope and environment

Verification date: 2026-10-06, Asia/Taipei. Scope includes complete exchange-session
windows, causal liquidity screening, a common evaluation source, annual sampling
decay, separate location/width/ranking outputs, validation-only interval calibration,
six controlled A/B adaptation presets, durable loss logs, and artifact downloads.

One isolated RTX 5090 Pod ran Python 3.12.3 and PyTorch 2.9.1+cu128. Code, the
project-owned `.venv`, tests, and cleaning indexes stayed on Pod-local `/tmp/stockqa`.
The existing network volume supplied read-only market-data and cached-model inputs.
No market downloads, CPU preparation, full baseline training, or production runs
were performed. The Pod was created at approximately 22:19 UTC and deleted from
the local controller at 23:04 UTC on October 5. REST confirmed deletion and an empty
Pod list. About 45 minutes at US$0.99/hour gives an estimated GPU charge of US$0.75,
within the one-hour/US$1.50 limit; this is not a settled invoice.

### Full existing-data audit

Every candidate cutoff in both snapshots was checked with vectorized rules and
six resource-bounded bucket workers. Only compact ranges were materialized.

| Population used by production experiments | A | B |
|---|---:|---:|
| Eligible training windows | 4,710,117 | 10,616,448 |
| Shared validation windows | 528,315 | 528,315 |
| Shared test windows | 534,327 | 534,327 |

A/B training starts in 2021/2016, respectively. Training labels end before June 1,
2025; validation uses June–November 2025; test uses December 2025–May 2026. Labels
cannot cross their split's exclusive upper boundary. Both evaluation populations
come from the exact same B snapshot and ranges, not separately sampled or intersected
populations. Missing production selection/shared data fails instead of falling back
to the training snapshot.

Of B's 39,198,475 training candidates, first-failure categories exclude 6,600,044
session-gap windows, 13,674,494 invalid-OHLCV/zero-volume windows, 7,781,157
insufficient-liquidity windows, 460,821 off-calendar windows, and 65,511 windows
whose labels cross the split boundary. These mutually exclusive counts describe
windows, not stocks.

Large returns are not removed by magnitude. A/B retain 147,745/381,865 training
windows with an absolute adjusted log return above 0.5; shared validation/test
retain 17,921/15,938. This is an audit threshold, not manual certification of every
large move. Beyond the full vectorized audit, 128 stratified positions per snapshot
and split were loaded through the lazy dataset to verify every input date, next-session
entry, and horizon-14 exit. Source bar-store manifests/success markers were unchanged.
All 10 CPU-preparation identity source paths remain unchanged, and capacity changes
preserve each A/B dataset request identity.

### Regression and CUDA evidence

- Full two-worker cloud pytest: **793 passed, 234 subtests passed, 3 skipped, 0 failed**.
- Full-project Ruff passed; static parsing passed for 52 changed Python and 9 Bash files.
- Five macOS Bash 3.2 sync-control tests passed.
- The three cloud skips require local Git/history artifacts; all passed separately
  using local stdlib tests. No local ML environment was inspected or created.
- Annual quotas, deterministic sampling/resume, interrupted training with batch
  reprobe, complete synthetic evaluation, storage contracts, and new/old download
  behavior passed. Permission failures are not treated as absent historical artifacts.
- Existing PyTorch `pin_memory(device)` deprecation warnings remain visible.

Real Kronos-base/tokenizer weights and eight real cleaned windows were also used
for CUDA forward/backward, finite-gradient, optimizer-step, trainable-weight save/load,
and optimizer-state restore checks through a two-worker DataLoader:

| Variant | Total trainable parameters | Result |
|---|---:|---|
| LoRA rank 32 | 16,076,633 | Passed; bit-exact predictions after restoration |
| LoRA rank 64 | 21,950,297 | Passed; bit-exact predictions after restoration |
| LoRA-32 on first 10 blocks; last 2 blocks/final norm unfrozen | 30,869,913 | Passed; gradients reach unfrozen blocks; bit-exact restored predictions |

Counts include task heads, not just adapters. The tokenizer stays frozen; unfrozen
blocks receive no redundant LoRA adapters. Variants run sequentially to bound memory.

### Operational and scientific limits

Training remains dynamically sampled, with per-window annual weight
`0.8 ** (latest_training_year - cutoff_year)`; validation/test remain exhaustive.
After one code sync, `train --experiment` chooses among the six A/B YAML presets
and publishes only the selection. The new cleaning population requires one new
baseline build per history group; capacity changes do not invalidate that baseline.
Historical results remain intact. Numerically different old checkpoints cannot be
resumed into the new architecture.

This verifies data and execution correctness, not better financial prediction.
No full six-experiment training or full real-data model holdout pass was run.
Direction accuracy, correlation, and out-of-sample coverage improvements remain
empirical questions. Calibration fits validation only and preserves raw/calibrated
test metrics; future 80% coverage is not guaranteed. Compare `train/pinball_loss`
with `validation/loss`, not the ranking-inclusive training total.

Raw evidence is retained in the Git-ignored local directory
`.runpod/verification/cleaning-20261006/`. The verification archive SHA-256 is
`de9d04a1c4dce92554adb0c13bab36c74e5898783cd88b40138f9dd1bc9c0067`.
