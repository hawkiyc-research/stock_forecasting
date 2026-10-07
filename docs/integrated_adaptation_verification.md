# 整合市場適應版本驗收 / Integrated Market Adaptation Verification

## 中文

日期：2026-10-08（Asia/Taipei）。基底提交：`5909b4e`。本紀錄區分工程回歸與模型
預測品質；小型訓練或合成容量測試不證明正式 holdout 的指標已改善。

### 實作與比較範圍

- A/B 各新增 `adaptive64`、`adaptive128`。兩種容量固定相同市場 residual、數值／
  benchmark ranking 路徑、校準、正則化及排程，只改 LoRA rank／alpha。
- 原六份設定逐位元組保持不變，部分解凍仍可使用，列為次要研究線。
- Training pinball 使用固定市場權重，另保留未加權 loss；checkpoint 與 early stopping
  仍只讀未校準的完整 validation normalized pinball。
- 樣本數排程保存實際呈現 windows、plateau 與低 LR 間隔；更換 batch plan 不歸零。
  途中才到達最低 LR 的間隔不算完整低 LR 間隔；只有重跑 validation 而沒有訓練也不計數。
- 凍結狀態校準與逐日延遲回饋分開報告。只有依市場交易日曆已到期、且出場日嚴格
  早於當次預測日的標籤可更新 online 寬度；q50、ranking、模型及凍結係數不變。

### 工程驗證

驗證使用 RTX 4090、Python 3.12.3、PyTorch 2.9.1+cu128，程式與輸出隔離於
Pod-local `/tmp/stock-forecasting-integrated-qa`。沿用既有遠端環境和已清理資料，
沒有建立／檢查本機 ML 環境、安裝套件、修改 lockfile、下載行情或執行正式訓練。

| 檢查 | 結果與界線 |
| --- | --- |
| 本機 stdlib 契約與控制測試 | 98 passed；涵蓋新設定、baseline 完成／清理檢查、A/B 切換、多 Pod、README CLI、歷史結果與資料管線 |
| 新增整合測試 | 13 passed；包含 CPU/CUDA head 路徑、校準、假日與 `h_start=3` 延遲、單筆推論、排程與 baseline identity |
| CUDA 資料管線回歸 | 53 passed；包含熱路徑等價、梯度／亂數狀態、batch probe、失敗回復與 worker 清理 |
| 中斷／重新調校／接續訓練 | 2 passed；原 cosine 與新 sample-plateau 各比較不中斷與中斷後的最終權重、步數、樣本數及 validation 次數 |
| 最終完整 CPU suite | 894 passed、38 skipped、0 failed，301.04 秒；略過項目與補驗範圍見下方 |
| 534,327 筆 × 14 horizons 校準容量 | 合成 memmap，4 個有界 workers、每塊 8,192 筆；擬合 5.97 秒、總計 8.73 秒，峰值 RSS 868,110,336 bytes（約 828 MiB），未實體化 input windows |
| Ruff、Python 語法與 Git diff whitespace | 通過 |

CPU suite 刻意隱藏 CUDA：30 項本次相關 CUDA 案例已由上表的 GPU 回歸另行覆蓋；
3 項需要 Git 或歷史下載 artifacts 的檢查已在本機 stdlib 測試補驗。其餘 5 項是
baseline CUDA 整體訓練／並行測試，本輪沒有重跑；baseline numerical code 保持不變，
因此不能把本輪結果說成已重新執行全部 baseline GPU 訓練驗收。

測試並非零警告：CPU suite 有 11 個 warnings；接續訓練測試的 48,204 個 warnings
主要是 PyTorch 內部 `pin_memory(device)`／`is_pinned(device)` 的重複棄用警告，
以及舊測試設定 `evaluation_max_samples` 已忽略的提示。沒有為消除警告而修改
DataLoader 或恢復 evaluation 抽樣，原始 warnings 保留在 log 中。

真實 Kronos 分別以八筆既有 clean train windows 執行三個 optimizer steps。LoRA64
可訓練參數為 **21,989,702**，LoRA128 為 **33,737,030**。兩者梯度／loss finite，
tokenizer 保持 frozen；保存並還原可訓練權重、optimizer 與 scheduler 後，quantile 和
ranking 輸出逐值完全一致。這不是吞吐量 benchmark，也不是新模型效能報告。

LoRA64 另從共用評估來源取 validation/test 各 256 個合法 windows，經正式
`evaluate_loader` 路徑完成 validation 擬合、凍結狀態校準、原靜態對照及延遲 online
診斷；每個小型集合都完整遍歷，但不冒充正式 528,315／534,327 筆全量推論。
正式流程仍對全部合格樣本循序評估，沒有加入取樣上限。

較早測試曾因新增診斷輸出未列入測試的 expected keys、隔離封裝漏帶 relay fixtures、
以及 smoke harness 未呼叫既有 batch 搬移函式而失敗。這些項目已修正重測；未修改
正式 DataLoader 或以放寬正式完整性檢查消除失敗。原始失敗紀錄保留在驗證 artifacts。

### 不變的資料與 baseline 邊界

比對基底提交確認 13 個 baseline numerical source、`baseline.json`、
`data_cleaning.json` 與原 compatibility registry 均逐位元組相同。主模型 `training.py`
中 baseline 使用的六個 calibration/sampling 定義及 `ROBUST_SCALE_*` 常數亦 AST 相同。
四份新設定與相同資料組別的原設定具有相同 dataset request 和 baseline identity。
沒有新增 SHA 例外、重寫 manifest、重建切分或修改已有結果。

新增校準／排程僅加入主模型 numerical contract；它們屬於新實驗的訓練語意，不把舊
架構 checkpoint 冒充新架構來接續。新實驗從相同 pretrained Kronos-base 初始化；
各自的中斷接續沿用自己的設定與權重。

### 證據、費用與限制

原始證據位於本機 ignored `.runpod/verification/integrated-20261008/`，包含測試
JUnit、完整 log、原始碼清單與雜湊、兩種容量 smoke 結果、容量測試與 Pod 終止紀錄。
只下載小型證據；合成 memmap、QA 模型權重及 optimizer state 不加入 repository，
隨隔離 Pod 的暫存磁碟一起移除，不寫入正式 baseline/checkpoint namespace。

隔離 Pod `n0b45ebcen25in` 使用 EU-RO-1 的 RTX 4090，建立於
2026-10-08 03:11:29、終止並確認移除於 04:04:38（Asia/Taipei），約 53 分 8 秒；
沒有延長原定 55 分鐘期限。按 US$0.74／小時估算 GPU 費用約 **US$0.66**，
此為執行時間乘費率的估算，不是 RunPod 最終帳單或既有 network volume 費用。

尚未執行新版正式 A/B 全量訓練、全量真實 holdout 推論、多 seed 或後續回測。
不得由上述工程測試推論 correlation、方向一致率或 coverage 已改善，也不保證未來
coverage 達 80%。比較整合版與原版是整套方法比較；單一元件的因果效果需另作消融。

## English

Date: 2026-10-08 (Asia/Taipei). Base commit: `5909b4e`. This record distinguishes
engineering acceptance from predictive quality. Small training runs and synthetic
capacity checks do not establish improved production holdout metrics.

### Implementation and comparison scope

- Groups A/B each add `adaptive64` and `adaptive128`. Market residuals, numerical/
  benchmark ranking paths, calibration, regularization and scheduling are identical;
  only LoRA rank/alpha differ between capacities.
- The original six presets are byte-identical and available, with partial unfreezing
  retained as a secondary research line.
- Training weights pinball by market while preserving unweighted loss logs. Checkpoint
  selection and early stopping still use raw full-validation normalized pinball only.
- The sample-clock schedule persists window presentations, plateau and minimum-LR
  intervals across batch-plan changes. An interval that only reaches the LR floor
  partway through does not count as a full floor interval; repeated evaluation without
  additional training cannot satisfy this safeguard.
- Frozen regime calibration and daily delayed feedback are reported separately. Online
  width updates use only labels whose market-calendar exit strictly precedes the
  current prediction. q50, ranking, model weights and frozen factors remain unchanged.

### Engineering verification

An RTX 4090 ran Python 3.12.3 and PyTorch 2.9.1+cu128 with isolated code/output under
Pod-local `/tmp/stock-forecasting-integrated-qa`. Existing remote dependencies and
cleaned data were reused. No local ML environment inspection/creation, package install,
lockfile changes, market downloads or production training occurred.

| Check | Result and boundary |
| --- | --- |
| Local stdlib control/contract checks | 98 passed, including presets, baseline completion/cleaning, A/B switching, multi-Pod control, README CLI, historical results and pipeline contracts |
| Added integration checks | 13 passed, including CPU/CUDA head paths, calibration, holidays, `h_start=3` delays, single-point inference, scheduling and baseline identity |
| CUDA pipeline regression | 53 passed, including hot-path equivalence, gradients/RNG, batch probes, failure recovery and worker cleanup |
| Interrupted/reprobed/resumed training | 2 passed; cosine and sample-plateau each compare final weights, steps, sample counts and validation counts against uninterrupted training |
| Final complete CPU suite | 894 passed, 38 skipped, 0 failed in 301.04 seconds; skip boundaries and supplementary checks are explained below |
| Calibration capacity: 534,327 × 14 horizons | Synthetic memmaps, 4 bounded workers and 8,192 rows per chunk; fitting 5.97 seconds, total 8.73 seconds, peak RSS 868,110,336 bytes (about 828 MiB), no input-window materialization |
| Ruff, Python syntax and Git whitespace | Passed |

The CPU suite deliberately hid CUDA. The 30 relevant CUDA cases were covered separately
by the GPU regression checks above. Three checks requiring Git or historical downloaded
artifacts passed in the local stdlib suite. The remaining five baseline CUDA end-to-end/
parallel-training cases were not rerun; baseline numerical code is unchanged. These results
must not be described as a fresh execution of every baseline GPU acceptance test.

The runs are not warning-free. The CPU suite emitted 11 warnings; the resume suite's
48,204 warnings mainly repeat PyTorch-internal `pin_memory(device)`/`is_pinned(device)`
deprecations and notices that the legacy fixture's `evaluation_max_samples` is ignored.
No DataLoader changes or evaluation sampling were introduced to suppress them; logs retain
the original warnings.

Each real Kronos capacity trained for three optimizer steps on eight existing cleaned
train windows. LoRA64 has **21,989,702** trainable parameters; LoRA128 has **33,737,030**.
Losses/gradients were finite and the tokenizer remained frozen. Restoring trainable weights,
optimizer and scheduler reproduced quantile and ranking outputs exactly. These are not
throughput benchmarks or new model-quality results.

LoRA64 additionally evaluated 256 valid windows from each shared validation/test source
through production `evaluate_loader`, covering validation fitting, frozen regime calibration,
the original static reference and delayed online diagnostics. Each small set was fully
traversed; this is not the production 528,315/534,327-window inference run. Production
continues to evaluate every eligible window sequentially without a sampling cap.

Earlier checks found an expected-output-key fixture missing additive diagnostics, relay
fixtures omitted from the isolated package, and a smoke harness that omitted the existing
batch device transfer. Those checks were corrected and rerun without changing the production
DataLoader or relaxing integrity gates. Original failed evidence is retained.

### Preserved data and baseline boundaries

The 13 baseline numerical sources, `baseline.json`, `data_cleaning.json` and existing
compatibility registry are byte-identical to the base commit. The six shared baseline
calibration/sampling definitions and `ROBUST_SCALE_*` constants in `training.py` are
AST-identical. New presets share dataset-request and baseline identities with existing
presets in the same data group. No SHA exceptions, manifest rewrites, split rebuilding or
existing-result mutations were introduced.

New calibration/scheduling enter only the main-model numerical contract. New experiments
initialize from the same pretrained Kronos-base, not by treating an old architecture's
checkpoint as a new model. Each experiment resumes its own configuration and weights.

### Evidence, costs and limitations

Raw evidence is local and ignored under `.runpod/verification/integrated-20261008/`, including
JUnit/logs, source hashes, capacity-specific smoke results, the calibration capacity check
and Pod termination evidence. Only small evidence is downloaded. Synthetic memmaps, QA model
weights and optimizer state are not committed or written to production result namespaces;
they disappear with the isolated Pod's temporary disk.

Isolated Pod `n0b45ebcen25in` used an RTX 4090 in EU-RO-1. It was created at
2026-10-08 03:11:29 and terminated/confirmed absent at 04:04:38 (Asia/Taipei), about
53 minutes 8 seconds, without extending its original 55-minute deadline. At US$0.74/hour,
estimated GPU cost is **US$0.66**. This is elapsed time multiplied by the quoted rate,
not the final RunPod bill or existing network-volume charges.

No new full A/B training, complete real holdout inference, multi-seed evaluation or later
backtest has been performed. These checks do not establish better correlation, direction
accuracy or coverage, and provide no future 80% coverage guarantee. Comparing the integrated
bundle with original presets does not isolate any individual component's causal effect.
