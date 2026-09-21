# Baseline tensor 資料管線驗收 / Baseline tensor-pipeline acceptance

## 中文

**狀態：部分驗證通過，完整回歸尚未完成；未發布新的驗收版本 tag。**

### 範圍與環境

- 日期：2026-09-21；受測功能程式碼：`c16f4ae`，包含 `0083c8e`、`49e68b4`、
  `bb215f1` 與 `08bfb8f` 的修正。本紀錄不覆蓋 v0.2.1 的歷史驗收。
- 驗證 run：`baseline-run-20260921T073833Z-10520`；RTX 4090 24 GiB，
  CPU quota 13.6 cores，規劃使用 13 cores。這不是舊 RTX 5090 Pod 的跨硬體測速比較。
- 原低效率 Pod `8p25bj0iq33kqd` 已停止；network volume、prepared data、inputs cache
  與已提交的 checkpoint 保留。
- 驗證 Pod `sh2a2fuore4sqw` 於 07:38:56 UTC 建立，55 分鐘本機 hard guard
  已啟用。內層 workflow 到期後，本機 guard 於 08:31:21 UTC 確認刪除，
  再查 Pod 清單為空；約 52 分 25 秒，沒有超過一小時授權。Guard 與控制程序已退出。
- 未執行 CPU prepare／finalize、未下載行情、未在本機建立或安裝 Python 訓練環境。
  本次只做有上限的雲端驗證，不是正式全量 baseline／A／B 訓練。

### 問題與修正

上一版已向量化讀取，但 neural baseline 仍經過主模型的逐筆資料／metadata 組裝，
再將 list 中的小 tensor 合併、複製至 GPU。單純 GPU 合成輸入測速亦不能代表真實
資料供應速度；同時保留 train／validation worker pools 則壓縮了可用 worker 預算。

本次修正：

1. Neural baseline 直接取得連續的 `[batch, 2, context, 5]` tensor，合併後才 pin
   memory；training 不建立不需要的時間字串與 Kronos metadata。Evaluation 保留
   完整的股票、日期、市場等成員資訊，不丟尾 batch。
2. GPU 合成 probe 只作記憶體准入；真正的 batch／workers／prefetch 選擇，以真實
   train／validation windows 的端到端吞吐為準。探測用模型副本與隔離 RNG，不修改
   正式權重。GRU 副本重新整理 cuDNN weights，避免用非正式訓練路徑誤導選擇。
3. 每個 neural job 同時只保留一個 worker pool；切換 validation 時先提交 sample
   cursor，關閉舊 pool，完成完整 validation 後再接續。提前關閉時只排空已送出的
   有界 prefetch，不能把未訓練的預取樣本算進 cursor，也不能靜默忽略 worker error。
4. 預留 full GBDT 與 GPU jobs 的共同 RAM；inputs 完成後優先安排 GBDT。尚待啟動
   的 GPU jobs 也參與 CPU 預留，避免第一個 native fit 搶走全部 CPU threads。
   GPU jobs 結束後，下一次 GBDT fit 可使用釋出的 CPU。
5. Baseline 專用記憶體准入最多折抵 50% 乾淨、未映射的 cgroup file cache，仍保留
   20% host 安全空間；dirty／mapped／shared／anonymous 頁面不列入。這是保守估計，
   不是 OS 回收保證，也不修改 CPU prepare 的資源規劃。
6. 經完整 source-hash 白名單審核的 execution-only 修正沿用既有 baseline ID，另存
   實際 execution contract。未知程式修改仍使快取失效；資料、標籤、數值參數、校準
   契約沒有被排除。不同 Python 版本的 AST 序列化也固定為一致格式。

Training 仍動態抽樣；validation／test 仍按每個產品的所有合法 cutoff 動態、固定順序
遍歷，沒有建立巨大的常駐 window 資料集。更換 batch／workers 保證接續樣本次序與
已提交位置，不保證不同 batch 下 optimizer 軌跡逐 bit 相同。

提前關閉 pool 的實作依賴受測 PyTorch 2.9.1 的 iterator 內部狀態，並非穩定公開 API；
升級 PyTorch 必須重跑有界 prefetch 排空、worker error 傳遞及中途接續測試。
相關行為可追溯至 [PyTorch 2.9.1 DataLoader 原始碼](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/data/dataloader.py)。

### 真實資料吞吐與 GPU 使用率

同一 Pod、相同模型初始化、相同樣本順序；每個模型先 warm up 16,384 windows，
再量測 **1,048,576 windows**。時間含 input wait、GPU transfer、forward、loss、
backward、gradient clipping 與 optimizer step；不含初始化、調校、validation 或保存。

| 模型 | 舊資料路徑 | 新資料路徑 | 吞吐提升 | 新 train batch／workers／prefetch |
| --- | ---: | ---: | ---: | --- |
| GRU | 22,496 windows/s；46.61 s | 30,182 windows/s；34.74 s | 1.34 倍 | 512／2／2 |
| DLinear | 20,945 windows/s；50.06 s | 30,142 windows/s；34.79 s | 1.44 倍 | 4,353／2／1 |

舊路徑是 v0.2.1 的 collator 與觀測到的 batch 設定，但使用目前共用讀取層，不是
整個舊 commit 的獨立部署。兩輪皆同時執行 GRU＋DLinear；**新路徑額外同時執行
full-size GBDT fit，舊路徑沒有這個 CPU 工作**，因此不是完全相同負載的純單因素實驗。
這是限定範圍的工程證據，不應外推為完整 epoch、全部模型完訓時間或準確度改善。

兩個 neural jobs 同時處於量測區間時，每秒取樣 GPU utilization：舊路徑平均
15.18%（45 點），新路徑 19.00%（34 點，最高 37%）。**GPU 仍未充分利用。**
GRU 的 input wait 為 27.73／34.74 秒；DLinear 為 33.93／34.79 秒。
Host 等待時間可能與 GPU 工作重疊，不能直接當作 GPU idle 百分比。剩餘限制包括
動態資料供應與 CPU／GBDT 資源競爭；小模型也不是靠增加 batch 就能保證滿載。

PatchTST 在上述並行工作結束後獨立量測 1,048,576 windows：16.99 秒、61,723
windows/s；train 與 evaluation 均選擇 batch 4,353、2 workers、prefetch 1。
它沒有同負載的舊版本對照，不能把這個數值解讀為 PatchTST 的加速倍數。

初輪較短的 262,144-window 測試曾得到 GRU 3.27 倍、DLinear 2.16 倍；那時每個
job 有 3 workers，且程式尚未加入最後的 cuDNN／shutdown／cache-credit 修正。
應以前述最終百萬筆結果為準，不能挑選初輪較大的倍數作結論。

### 完整性、接續與記憶體

- 真實 train／validation／test 各 128 個索引：新舊序列 tensor 與 target
  `rtol=0, atol=0` 相等，evaluation metadata 相等；另有不同 horizon 起點、亂序、
  重複索引與完整尾 batch 等 fixture 回歸。
- 原 GRU／DLinear checkpoint 的副本從 epoch 0、cursor 18,134,537 接續到
  18,135,561，scheduler 隨之推進；正式 `model.pt`、`resume.pt` SHA-256 不變。
  改變 batch 32→47、workers 1→2 的接續，以及提前關閉 pool 的 3 項 smoke tests
  全部通過。正式檔案沒有被副本驗證覆寫。
- Full GBDT admission 估計 34,547,854,992 bytes；本機可見 CPU quota 不是 host
  core 數。最終計畫同時准入 2 個 GPU jobs、各 2 workers、GBDT 5 threads，保留
  20% host headroom。實測 GBDT 用完整 **30,224,227 rows** 完成一次 horizon／
  quantile／iteration fit，27.96 秒，程序 peak RSS 約 14.42 GiB；它不是完整 GBDT
  收斂訓練，也不能代表所有 quantiles 與最佳模型副本同時存在的最高記憶體。
- CPU prepare semantic contract 與 prepared manifest 不變。全部 44,784 個 eligible
  products 的 cutoff 稽核確認 validation **1,232,972 windows／13,538 products**，
  test **1,264,861 windows／16,084 products**，未物化 input windows。
- 真實離線 Kronos CUDA forward／backward／optimizer step 通過，gradient 有限且
  非零；另跑 4,096 筆 holdout input profile，不是完整 holdout 預測。
- 1,264,861 筆**合成預測**的磁碟 backing／指標聚合：29.74 秒，peak RSS 約
  1.48 GiB。這不是完整真實 testing 的模型分數。

最終 `tensor-final-status.json` 是 `pytest=124`、`baseline_throughput=0`、`ruff=1`，
**不能標記完整驗收通過**。Pytest 的 740 秒單次預算到期時，堆疊位於
`test_complete_baseline_builder_and_cache_reuse`／`build_baselines` 的子工作等待迴圈，
沒有產生最終 JUnit XML。僅憑此堆疊，不能斷言只是測試太慢，也不能判定正式訓練死鎖；
需在下一個獲授權的驗證中優先重跑 builder 與完整 suite，保留逐測試及子工作進度。

雲端 Ruff 的兩項 E501 是控制測試中的長字串。下載的測試檔仍為未換行版本，已提交的
本機檔案已換行；兩者 AST 完全相同。受測的 11 個 baseline implementation 檔案
SHA-256 全部與本機相同，因此這不是 baseline 功能程式的版本差異。本機完整 Ruff
重跑通過，另有 13 項控制平面、7 項 runtime 控制測試通過；這不替代未完成的雲端
suite。下一次驗證須先確認所有受測來源檔案與部署 manifest 一致。

整個 Pod 觀測到 cgroup 記憶體高水位 27,139,686,400 bytes（約 25.28 GiB），
`memory.failcnt=0`、`oom_kill=0`；包含多輪驗證及 file cache，不是正式長時間訓練
峰值的保證。

第一輪 pytest 為保留最終版重跑時間而主動中斷（164 passed、1 skipped，exit 2），
不列為完整通過；第一次 checkpoint-copy 提前退出的 worker shutdown 異常有保留，
已修正並重測，不能將初輪 log 說成無錯誤。

### 追溯與限制

Network volume 證據：
`/runpod-volume/diagnostics/full-workflow/baseline-run-20260921T073833Z-10520/`。
關鍵檔案為 `tensor-final-status.json`、`pytest-final.log`、`ruff-final.log`、
`cleanup-status.json`、`baseline-throughput-final/summary.json`、各 job runtime JSON、
`before/gpu-samples.json`／`after/gpu-samples.json`、`lazy-evaluation-audit.json`、
`kronos-smoke.json` 及 `capacity/capacity.json`。

沿用的 baseline ID：
`baseline-259f4346bb21d61c98cb9f756859767654b5c8c62a2b888b0b8dea91ecb647c6`。
資料 request SHA-256：
`aa49bbbdb9061c74a37eb659c8f8908c03014fc49b4a57f341b61061999eda0a`。
Validation membership SHA-256：
`b5187abd95bee5223841ca21733b3b4904ea449ef20b54f1a06d37ac4b53febb`；test：
`67adda03221adf6c6929e84b7043ce55f62ede5f871dc23fcd8c391d6ebb529a`。

續訓沿用 README 的 `runpod_workflow.sh baseline` 與遠端
`runpod_tmux_launch.sh baseline`，不需指定 checkpoint 路徑或重跑 CPU prepare。
GBDT 仍可能比 neural jobs 花更長時間；目前沒有自動遷移 CPU-only 尾段至 CPU Pod。
完整 baseline 是否收斂、最終模型分數、長時間 RAM 穩定性仍需正式訓練結果判定，
不能由本次有限時間驗證保證。

## English

**Status: partial validation only; full regression is incomplete. No new accepted-release tag.**

### Scope and environment

- Date: 2026-09-21. Tested implementation: `c16f4ae`, including fixes `0083c8e`,
  `49e68b4`, `bb215f1` and `08bfb8f`. The historical v0.2.1 record is retained.
- Run: `baseline-run-20260921T073833Z-10520`; RTX 4090 24 GiB, 13.6-core container
  CPU quota, 13 cores admitted. This is not a cross-hardware comparison against
  the old RTX 5090 production Pod.
- Inefficient Pod `8p25bj0iq33kqd` was stopped; the network volume, prepared data,
  completed input cache and committed checkpoints were preserved.
- Verification Pod `sh2a2fuore4sqw` was created at 07:38:56 UTC with a 55-minute
  local hard guard. After the inner workflow timed out, the control-host guard
  confirmed deletion at 08:31:21 UTC, followed by an empty Pod list. Elapsed time
  was approximately 52m25s, within the authorized hour. Guard/controller exited.
- No CPU prepare/finalize, market-data downloads or local training-environment
  setup. Only bounded cloud checks were run, not production baseline/A/B training.

### Diagnosis and repairs

The previous implementation vectorized reads but still built per-record model
metadata and small tensor lists before joining/transferring neural inputs. Synthetic
GPU throughput did not reflect data delivery, while simultaneous train/validation
worker pools consumed the worker budget.

1. Supply contiguous `[batch, 2, context, 5]` tensors, assembled before pinning.
   Skip unused timestamps/Kronos metadata in training; retain complete evaluation
   membership fields and every final partial batch.
2. Use synthetic GPU probes only for memory admission. Jointly tune batch, workers
   and prefetch on real train/validation windows using end-to-end throughput.
   Isolate model copies and RNG; restore the GRU copy's cuDNN weight layout.
3. Keep only one active worker pool per neural job. Commit the training sample
   cursor before full validation and resume afterward. Early shutdown drains only
   bounded already-submitted prefetch; it never advances the committed training
   cursor or suppresses genuine worker failures.
4. Reserve RAM jointly for full GBDT and GPU jobs; prioritize GBDT once inputs are
   ready. Pending GPU jobs also reserve CPU capacity before native fitting starts.
   Later GBDT fits can use CPU threads released by finished neural jobs.
5. Baseline-only admission credits at most half of clean, unmapped cgroup file
   cache, retaining 20% host headroom. Dirty/mapped/shared/anonymous memory receives
   no credit. This is an estimate, not an OS guarantee; CPU prepare is unchanged.
6. Preserve the baseline ID only for reviewed execution-only implementations in
   an exact source-hash allowlist, recording the actual execution contract. Unknown
   changes still invalidate identity; data, labels, numerical parameters and
   calibration remain covered. AST serialization is stable across Python versions.

Training still samples dynamically. Validation/test exhaust every eligible cutoff
in deterministic order without materializing windows. Rebatching/worker changes
preserve committed sample order and position, not bitwise optimizer trajectories.

Early pool shutdown relies on iterator internals of tested PyTorch 2.9.1, not a
stable public API. PyTorch upgrades must rerun bounded-drain, worker-error and
mid-epoch resume tests; see the [PyTorch 2.9.1 DataLoader source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/data/dataloader.py).

### Real-window throughput and GPU utilization

Same Pod, model initialization and sample order; 16,384 warmup windows followed by
**1,048,576 measured windows per model**. Timings include data wait, transfers,
forward/loss/backward, clipping and optimizer steps, but exclude initialization,
tuning, validation and checkpoint saving.

| Model | Previous input path | New input path | Speedup | New train batch/workers/prefetch |
| --- | ---: | ---: | ---: | --- |
| GRU | 22,496 windows/s; 46.61 s | 30,182 windows/s; 34.74 s | 1.34x | 512/2/2 |
| DLinear | 20,945 windows/s; 50.06 s | 30,142 windows/s; 34.79 s | 1.44x | 4,353/2/1 |

The old path uses the v0.2.1 collator and observed batches with the current shared
reader, not an independent deployment of the entire historical commit. Both waves
run GRU and DLinear concurrently; **only the new wave additionally runs a full-size
GBDT fit**, so this is not an identical-load, single-factor experiment. Do not
extrapolate to whole epochs, total completion time or predictive accuracy.

One-second GPU samples within the interval when both neural jobs were measured
averaged 15.18% before (45 points) and 19.00% after (34 points, maximum 37%).
**GPU utilization remains low.** Input wait was 27.73/34.74 seconds for GRU and
33.93/34.79 seconds for DLinear. Host waits may overlap GPU execution and are not
GPU-idle percentages. Dynamic data delivery and CPU/GBDT contention remain limits;
larger batches alone cannot guarantee saturation for these small models.

PatchTST was measured separately after the concurrent wave: 1,048,576 windows in
16.99 seconds, or 61,723 windows/s. Both train and evaluation selected batch 4,353,
two workers and prefetch one. Without a matched old-path measurement, this is not
a PatchTST speedup estimate.

The initial 262,144-window check reported 3.27x/2.16x, with three workers per job
and before final cuDNN/shutdown/cache-credit fixes. Use the final million-window
measurements above; do not select the larger preliminary numbers as the conclusion.

### Integrity, resume and memory

- Exact `rtol=0, atol=0` sequence/target equality on 128 real indices from each
  train/validation/test split; identical evaluation metadata, plus fixture checks
  for horizon offsets, shuffled/duplicate indexes and complete final batches.
- Copies of production GRU/DLinear checkpoints resumed epoch 0 from cursor
  18,134,537 to 18,135,561 with scheduler advancement. Original `model.pt`/`resume.pt`
  hashes were unchanged. All three shutdown and batch 32-to-47/worker 1-to-2 resume
  smoke cases passed; no production checkpoint was overwritten.
- Estimated full GBDT admission: 34,547,854,992 bytes. Final plan: two GPU jobs,
  two workers each and five GBDT threads, retaining 20% host headroom. A native fit
  used all **30,224,227 training rows**, one horizon, quantile and iteration:
  27.96 seconds, approximately 14.42 GiB process peak RSS. This is not converged
  GBDT training or a peak-memory guarantee for every fitted/best-model copy.
- CPU-prepare semantics and prepared manifest were unchanged. Exhaustive cutoff
  auditing covered 44,784 eligible products: validation **1,232,972 windows/13,538
  products**, test **1,264,861 windows/16,084 products**, without expanding windows.
- Real offline Kronos CUDA forward/backward/optimizer check passed with finite,
  nonzero gradients. A separate 4,096-row holdout input profile is not full scoring.
- Disk-backed aggregation of **1,264,861 synthetic predictions** took 29.74 seconds
  with approximately 1.48 GiB peak RSS; these are not real test accuracy metrics.

Final `tensor-final-status.json`: `pytest=124`, `baseline_throughput=0`, `ruff=1`.
**Full acceptance has not passed.** The 740-second pytest budget expired in the
child-job wait loop of `build_baselines`, inside
`test_complete_baseline_builder_and_cache_reuse`; no final JUnit XML was produced.
This stack alone establishes neither harmless slowness nor a production deadlock.
The next authorized check must prioritize the builder and full suite with durable
per-test and child-job progress.

The two cloud Ruff E501 findings concern long strings in a control test. The
downloaded deployed copy was unwrapped, whereas the committed local copy was already
wrapped; their ASTs are identical. All 11 tested baseline implementation hashes
match the local files, so this is not a baseline implementation-version mismatch.
The full local Ruff rerun passed, as did 13 control-plane and seven runtime-control
tests. These do not replace the unfinished cloud suite. Before the next run,
verify every deployed test/source file against its manifest.

Observed Pod-wide cgroup memory high water was 27,139,686,400 bytes (approximately
25.28 GiB), with `memory.failcnt=0` and `oom_kill=0`. This includes repeated checks
and file cache, not a long-running production peak-memory guarantee.

The initial pytest run was deliberately interrupted to leave time for final-code
retesting (164 passed, one skipped, exit 2); it is not a complete pass. The initial
checkpoint-copy shutdown failure remains in logs and was repaired/retested.

### Evidence and limitations

Evidence root:
`/runpod-volume/diagnostics/full-workflow/baseline-run-20260921T073833Z-10520/`.
Primary files: `tensor-final-status.json`, `pytest-final.log`, `ruff-final.log`,
`cleanup-status.json`, `baseline-throughput-final/summary.json`, per-job runtime
JSON, `before/gpu-samples.json`/`after/gpu-samples.json`, `lazy-evaluation-audit.json`,
`kronos-smoke.json` and `capacity/capacity.json`.

Retained baseline ID:
`baseline-259f4346bb21d61c98cb9f756859767654b5c8c62a2b888b0b8dea91ecb647c6`.
Dataset request SHA-256:
`aa49bbbdb9061c74a37eb659c8f8908c03014fc49b4a57f341b61061999eda0a`.
Validation membership SHA-256:
`b5187abd95bee5223841ca21733b3b4904ea449ef20b54f1a06d37ac4b53febb`; test:
`67adda03221adf6c6929e84b7043ce55f62ede5f871dc23fcd8c391d6ebb529a`.

Resume using the existing README `runpod_workflow.sh baseline` and remote
`runpod_tmux_launch.sh baseline` commands, without checkpoint paths or CPU prepare.
GBDT can still outlast neural jobs; automatic CPU-only-tail migration to a CPU Pod
is not implemented. Full convergence, final scores and long-duration memory
stability require production results and are not guaranteed by bounded acceptance.
