# 多 Pod 控制與文件驗收 / Multi-Pod Control and Documentation Verification

## 中文

### 範圍

驗收日期：2026-10-06。控制流程實作版本為 `5e18f88`，完整 README 整理版本為
`9cd2ff1`。本次驗收的是不同容量實驗的並行控制、跨機器共用儲存鎖、各 run
的設定與產物隔離，以及本機 guard 的終止隔離；不是正式全量模型訓練。

兩台獨立 RTX 4090 Pod 位於 EU-RO-1，使用 PyTorch `2.9.1+cu128`。測試程式與
所有寫入限定在既有 volume 的 `verification/parallel-20261006T075149Z/`。
沒有覆寫正式專案、Python 環境、行情、baseline 或主模型結果；沒有下載行情、
執行 CPU prepare、全量 baseline 或正式 A/B 訓練。建立前帳號 Pod 清單為空。

### 雲端實測

| 項目 | 實際觀察 |
| --- | --- |
| 兩個容量設定 | 讀取 `a_lora32.yaml` 與 `a_lora64.yaml`，逐一核對 selection 內的設定檔雜湊；dataset identity 相同、selection identity 不同 |
| 兩機並行 | 兩個獨立 tmux 工作同時執行 CUDA forward、backward 與 optimizer step，皆持續增加步數 |
| 儲存與還原 | 使用專案 `LoRALinear` 的小型合成工作，各自保存模型及 optimizer；重新載入模型後，輸出逐值相同 |
| 不同 run 的 volume 租約 | 兩機可同時持有 shared lease；第三個不同 run 的短暫取鎖檢查成功 |
| 同一 run 的跨機器互斥 | 從第二台嘗試取得第一台 run 的 training 或 validation 租約，均以 exit code `75` 拒絕 |
| 獨占流程隔離 | 兩台持有 shared lease 時，baseline 的 exclusive lease 被拒絕；本機使用真實 API inventory 的 sync／exclusive admission 也被拒絕 |
| 共用 cache 初始化 | 執行未修改的 cache coordinator，以合成 builder 代替昂貴計算；兩機 critical section 不重疊，第二台等待第一台釋放鎖後才進入 |
| Guard 身分檢查 | 各 guard 接受自己的真實 lifecycle，拒絕另一台的 Pod／run 身分 |
| 停止隔離 | 第一台完成並刪除後，第二台仍為 running，步數由先前的 1,230 增至 2,215；第一台的 run lease 可以重新取得 |
| 最終清理 | 兩台各由本機 guard 依自己的 `lifecycle-ready` 終止，REST 均回覆 `deleted: true`；再次查詢帳號 Pod 清單為 `[]` |

合成工作使用單個 256→256 的 LoRA linear layer；rank 32／64 的可訓練參數分別為
16,384／32,768，**不是完整 Kronos 模型的參數量**。它驗證 CUDA 工作與檔案隔離，
不代表完整 Kronos、真實 DataLoader 或全量 evaluation 已在這兩台 Pod 上重跑。
完整模型與資料清理的另一份驗收見
[資料清理與容量實驗驗證](cleaning_experiments_verification.md)。

共用 cache 測試只驗證實際 coordinator 與 network filesystem 上的鎖語意，
沒有重新生成正式 training cache。完整 launcher 的 preflight、selection 發布、
兩台 guard 綁定與部分失敗情境另外由控制測試覆蓋；隔離雲端工作不會冒充正式
CPU／baseline readiness 已完成。

### 執行異常與處理

- 第二台第一次建立回傳 HTTP 500。先重新查詢 inventory，確認只有第一台、沒有
  隱藏的成功建立結果，再重試第二台；沒有額外建立第三台。
- 解開測試副本時，volume 拒絕還原 macOS UID／GID。沒有修改 volume 權限或
  執行 chown；改以唯讀方式在兩台逐檔核對 252 個檔案，SHA-256 全部相符。
- 初次啟動的兩個背景 guard 曾顯示 armed，但後續程序檢查發現均已退出，未取得
  退出原因。沒有把 armed 檔案當作持續存活的證據；改由保持存活的本機控制工作
  重新啟動 guard，**沿用原始建立時間的期限**。兩次實際 lifecycle-triggered
  deletion 與最終空 inventory 均已確認。此紀錄不宣稱已證明最初退出的原因。
  後續以獨立的 `nohup sleep 30` 重現：建立它的工具命令返回後，同一秒查詢程序已
  消失。這支持工具工作階段會清理背景程序，因此雲端驗收須維持本機控制工作存活，
  不能只依賴脫離後的背景程序；未將此觀察外推為一般 Terminal 的行為。
- Cache 測試 harness 初版的字串代換及 SSH 環境傳遞各有一處錯誤，修正後才取得
  上表中的成功結果；正式 cache coordinator 未因這些 harness 錯誤修改。

### 時間、費用與證據

| 實驗 | 建立請求時間（UTC） | 刪除成功時間（UTC） | 經過時間 |
| --- | --- | --- | --- |
| `a-lora32` | 07:57:40.758 | 08:09:30 | 約 11 分 49 秒 |
| `a-lora64` | 07:59:07.990 | 08:10:40 | 約 11 分 32 秒 |

API 回報兩台費率均為 US$0.74／小時，按請求至刪除的時間估算 GPU 租用費合計
約 **US$0.29**。兩台皆低於各 30 分鐘的上限；這是 GPU 時間估算，不是最終帳單，
也不把既有 network volume 的持續儲存費算成此次新建 GPU 的費用。

原始證據保存在本機被 Git 排除的 `artifacts/verification/runpod-parallel-20261006/`，
包含來源快照、harness、兩份 guard log、API 清單、終態 lifecycle、CUDA 工作紀錄、
鎖測試及費用估算，共約 3.6 MB。`SHA256.json` 的 SHA-256 為
`07f4b7fad870e1ae47c025ebc0064ebd89976b494f51eae0b857e3c256c09147`。
正式訓練或部署憑證不在驗收文件中。

### README 全文檢查

README 的中文及英文部分均完整重整，不只追加多 Pod 段落。閱讀順序涵蓋模型、
資料清理、固定切分、離線流程、RunPod 操作、產物、診斷、CLI 參考及驗收；
容量實驗放在訓練操作內，細節收於對應章節，並維持中文完整在前、英文完整在後。

核對包含 GPU 機房查詢、configure、六個容量設定、重複 `--experiment`、
`--launchWorkers`、首次 prepare 與既有資料重用、baseline、resume、validation、
下載、probe 及其參數／預設值／執行位置。`--data-center` 明確只篩選 GPU 目錄，
不會修改 `.env` 或部署位置。

- 88 個 stdlib 控制測試通過。
- 另外 35 個 probe tmux／guard、baseline tmux 及 checkpoint guard 測試通過。
- 6 個 README CLI reference 測試通過。
- 全專案 Ruff、變更 Bash 語法及 `git diff --check` 通過。
- README 的 92 個 shell 範例通過 `bash -n`；100 個 code fences、34 張表格、
  中英文命令對稱性、內部連結，以及 79 個 Python parser options／36 個 shell
  workflow options 的文件覆蓋檢查通過。
- 同步 dry-run 與 secret scan 通過；dry-run 沒有上傳或改寫正式部署。

### 本機控制流程再驗收

2026-10-06 再驗收時，發現結果下載測試尚未帶入新增的 `runpod_runs.py`、
run-scoped lifecycle 與 object listing 模擬；診斷文件測試則仍要求歷史 run ID，
未跟隨 README 改用通用 `<RUN_ID>` 範例。兩者皆已修正測試，沒有更改正式腳本。
下載測試現在執行真實的 run discovery helper，涵蓋指定 run 的身分隔離、依完成時間
選取 run、歷史結果、必要 loss／校準／completion artifacts，以及 403 不被當成 404。

獨立的 `docs/experiment_protocol.md` 亦已同步現行 early stopping、validation-only
校準及自動 full-holdout 行為，新增文件回歸檢查；README 本體不需改動。

- 合併執行 **179 項 stdlib 控制／文件／下載／tmux／guard 測試**，0 failures、
  0 errors、0 skipped，耗時約 104 秒。
- 全專案 Ruff、50 個受版本控制 shell scripts 的 `bash -n` 及 `git diff --check` 通過。
- README 全文重新檢查：92 段 shell 範例、74 個本機連結、雙語命令一致性及 CLI
  options 的說明覆蓋通過。
- 初次合併測試有一項日誌斷言失敗。追查到本機 sandbox 拒絕 Bash 的
  `/dev/fd/62` process substitution，並非等待不足；經核准在 sandbox 外執行同一組
  測試後全數通過。沒有新增 sleep、跳過斷言或修改正式 logging 行為。

本次只修改測試與文件，沒有修改 `src/`、`scripts/`、`configs/` 或依賴設定，
不改資料準備、baseline／checkpoint 數值契約。沒有啟動 Pod、同步正式部署、
執行行情 API 或建立本機 ML 環境；也沒有重跑全量模型訓練。前述雲端模型與
雙 Pod 證據仍各自受其驗收範圍限制，不把分項測試誤稱為六組正式訓練已完成。

這些證據界定工程流程的驗收範圍，不證明任何容量實驗的預測指標已改善，
亦不替代後續正式訓練的持續監控。

## English

### Scope

Verification date: October 6, 2026. Multi-Pod control implementation: `5e18f88`;
complete README restructuring: `9cd2ff1`. This acceptance covers concurrent
capacity-experiment control, cross-host storage locks, per-run configuration and
artifact isolation, and local-guard termination isolation—not production training.

Two independent RTX 4090 Pods in EU-RO-1 used PyTorch `2.9.1+cu128`. All test code
and writes were confined to `verification/parallel-20261006T075149Z/` on the existing
volume. Production source, Python environments, market data, baseline artifacts and
main-model results were not overwritten. No market downloads, CPU preparation,
full baseline training or formal A/B training ran. The initial Pod inventory was empty.

### Live evidence

| Check | Observation |
| --- | --- |
| Capacity configurations | Loaded `a_lora32.yaml` and `a_lora64.yaml`, verified each selection's config hash; equal dataset identities and different selection identities |
| Concurrent execution | Independent tmux CUDA forward/backward/optimizer workloads both advanced |
| Persistence | Small workloads using the project's `LoRALinear` saved model/optimizer state independently; reloaded model outputs were bit-exact |
| Shared volume leases | Both Pods held shared leases; a short third, distinct-run lease check succeeded |
| Same-run exclusion | Training and validation attempts from the second host against the first run both returned exit code `75` |
| Exclusive workflow exclusion | Baseline's exclusive lease was rejected; local sync/exclusive admission against real API inventory was also rejected |
| Shared cache initialization | The unchanged coordinator with synthetic builders serialized critical sections across hosts; the second waited for the first to release its lock |
| Guard ownership | Each guard validator accepted its own real lifecycle and rejected the other Pod/run identity |
| Stop isolation | After the first Pod completed and was deleted, the second remained running and advanced from the earlier 1,230 steps to 2,215; the first run's lease was released |
| Final cleanup | Both local guards independently deleted their Pods on `lifecycle-ready`; REST returned `deleted: true` twice, and a subsequent inventory was `[]` |

The synthetic CUDA workload used one 256→256 LoRA linear layer with 16,384/32,768
trainable parameters for ranks 32/64. These are **not full Kronos model counts**.
This establishes concurrent CUDA execution and artifact isolation, not a new full
Kronos, real DataLoader or complete evaluation run on these two Pods. Separate
full-model/data-cleaning evidence is in
[the cleaning and capacity verification](cleaning_experiments_verification.md).

The cache test exercises the actual coordinator and network-filesystem lock
semantics without rebuilding production caches. Full-launcher preflight, selection
publication, two guard bindings and partial-failure behavior are separately covered
by control tests. The isolated workloads do not assert that production CPU/baseline
readiness has already been satisfied.

### Execution issues and handling

- The first attempt to create the second Pod returned HTTP 500. Inventory was
  reconciled before retrying only that Pod; no third Pod was created.
- Archive extraction could not restore macOS UID/GID ownership on the volume.
  No permissions or ownership were changed. Read-only SHA-256 verification of
  all 252 files passed independently on both Pods.
- Both initial background guards reported armed but were later found absent;
  their exit causes were not captured. Armed files were not treated as proof of
  liveness. Guards were rearmed under a persistent local controller with the
  **original creation-based deadlines**, then both lifecycle-triggered deletions
  and the empty final inventory were verified. This does not establish the
  cause of the initial exits.
  A separate `nohup sleep 30` probe was subsequently absent in the same UTC
  second after its launching tool command returned. This supports background
  process cleanup by the tool session, so cloud verification keeps a foreground
  local controller alive. It is not generalized to ordinary Terminal behavior.
- The cache-test harness initially had a string-substitution error and an SSH
  environment-propagation omission. Both were corrected before the successful
  result above; production cache-coordinator code was not changed for these errors.

### Duration, estimated cost and evidence

| Experiment | Create request, UTC | Successful deletion, UTC | Elapsed |
| --- | --- | --- | --- |
| `a-lora32` | 07:57:40.758 | 08:09:30 | Approximately 11m 49s |
| `a-lora64` | 07:59:07.990 | 08:10:40 | Approximately 11m 32s |

Both API-reported rates were US$0.74/hour. Request-to-deletion GPU time totals
approximately **US$0.29**, with each Pod below its 30-minute limit. This is a GPU-time
estimate, not a settled invoice; ongoing storage for the pre-existing network
volume is not counted as a newly created GPU cost.

Raw evidence is retained in the Git-ignored local directory
`artifacts/verification/runpod-parallel-20261006/`: source snapshot, harness, guard
logs, API inventory, terminal lifecycles, CUDA records, lock checks and estimates,
approximately 3.6 MB. The SHA-256 of its `SHA256.json` is
`07f4b7fad870e1ae47c025ebc0064ebd89976b494f51eae0b857e3c256c09147`.
Deployment credentials are not included in this verification document.

### Complete README audit

Both language sections were restructured as a complete document, not just extended
with a multi-Pod addendum. The reading order covers the model, cleaning, fixed
splits, offline workflow, RunPod operations, artifacts, probes, CLI reference and
acceptance. Capacity experiments belong within training operations; detailed
material stays in the relevant sections. Complete Traditional Chinese precedes English.

The audit includes GPU data-center queries, configure, six capacity presets,
repeatable `--experiment`, `--launchWorkers`, first-time preparation versus reuse,
baseline, resume, validation, downloads and probes, with options, defaults and
execution location. `--data-center` is explicitly a catalog filter, not a change
to `.env` or deployment location.

- 88 stdlib control tests passed.
- 35 additional probe tmux/guard, baseline tmux and checkpoint-guard tests passed.
- Six README CLI-reference tests passed.
- Full-project Ruff, changed-shell syntax and `git diff --check` passed.
- All 92 shell examples passed `bash -n`; 100 fences, 34 tables, bilingual command
  parity, internal links, and documentation coverage of 79 Python-parser options
  and 36 shell-workflow options passed structural checks.
- Sync dry-run and secret scanning passed, without uploading or modifying the
  production deployment.

### Local control-workflow re-verification

Re-verification on October 6 found that download fixtures had not included the
new `runpod_runs.py`, run-scoped lifecycle records, or object listing. The probe
documentation test still expected a historical run ID instead of the README's
generic `<RUN_ID>` example. Both fixtures were corrected without changing the
production scripts. Download tests now execute the real run-discovery helper and
cover explicit-run isolation, completion-time selection, historical results,
required loss/calibration/completion artifacts, and distinguishing 403 from 404.

The separate `docs/experiment_protocol.md` now matches current early stopping,
validation-only calibration, and automatic full-holdout behavior, with a new
documentation regression check. The README itself did not need another edit.

- **179 combined stdlib control/documentation/download/tmux/guard tests passed**:
  zero failures, errors, or skips, in approximately 104 seconds.
- Full-project Ruff, `bash -n` for all 50 version-controlled shell scripts, and
  `git diff --check` passed.
- The complete README was rechecked: 92 shell examples, 74 local links, bilingual
  command parity, and CLI option-description coverage passed.
- One log assertion initially failed because the local sandbox rejected Bash's
  `/dev/fd/62` process substitution, not because the test needed a longer wait.
  The same combined suite passed outside the sandbox after approval. No sleeps,
  skipped assertions, or production logging changes were introduced.

Only tests and documentation changed; `src/`, `scripts/`, `configs/`, dependencies,
data preparation, and baseline/checkpoint numerical contracts were untouched.
No Pods, deployment sync, market API calls, local ML environments, or full model
training were involved. Earlier cloud-model and dual-Pod evidence retains its
stated scope; component verification does not mean six production experiments
have already completed.

This evidence defines engineering acceptance scope. It neither demonstrates better
forecast metrics for a capacity variant nor replaces monitoring of production runs.
