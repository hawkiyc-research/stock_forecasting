# 訊號診斷流程實作與驗證 / Signal diagnostic implementation and verification

## 繁體中文

日期：2026-10-18。分支：`codex/model-performance-20261016`；修改基點：`938018826fb98d78cfd331b3b3046d536ed6335d`。程式與驗證均在 Codex 的獨立 worktree 進行。這次完成診斷與受控校準工具，尚未產生新的模型效能結果；沒有據此宣稱預測品質改善。

### 已完成內容

1. 移植並擴充 Claude `27fb9daaad485d1e22310c8d7d275b528b68c83a` 的唯讀 checkpoint prediction export 與下載合約。保留 validation-best／leaderboard named selectors、逐批推論、原始評分、雜湊、GPU lease 與既有 tmux guard。新增有界且跨模型固定的 train eval 樣本，無需建立完整訓練視窗資料集。
2. 新增最多 32 個高 loss／大回撤案例及同日同市場的一般對照，保存原始與 adjusted 價格、PIT 輸入及 next-open→horizon-close label 重算。使用受 CPU／記憶體規劃的 DataLoader workers；保存實際 worker plan。稽核驗證內部一致性，行情商 corporate actions 仍需外部證據。
3. 新增標準函式庫離線分析器，以完整 train 先驗、原始 logits、全域 temperature、全域 temperature＋收縮 class bias 作相同程序比較。只在較早 validation 擬合，較晚 validation 評分，每市場保留至少 14 個觀察交易日期的成熟間隔，不讀 test、不自動挑選贏家或修改推論設定。
4. 新增 immutable protocol／逐模型報告／comparison／status、輸入與程式雜湊、固定 ordinals、月份與市場分組。每個 array 使用約 4 MiB 的映射視窗；spawn workers、pending jobs、擬合樣本、月份與磁碟產物均有界。錯誤／取消／逾時會終止並回收 workers。
5. 提供完整繁中後接英文的 [使用說明](../docs/SIGNAL_DIAGNOSTICS.md)，並更新 README 與現有 CLI 對照檢查。

### 已執行的本機驗證

| 檢查 | 結果 |
| --- | --- |
| 新增 stdlib 數值／流程／例外／逾時測試 | 22 通過 |
| 專案命名與 CLI 控制 | 13 通過 |
| README CLI 對照 | 10 通過 |
| tmux 回歸，遠端操作全部 mocked | 16 通過 |
| 合計 | 61 通過 |
| Ruff 0.15.21 lint／format | 13 個 Python 檔通過 |
| AST／Bash 語法 | 13 個 Python／4 個 shell 檔通過 |
| 原有核心程式逐 byte 比較 | 23 個檔案與修改基點完全一致 |

數值測試包含校準梯度的有限差分驗證、極端 logits 的穩定 CE、temperature 的 argmax 不變性、bias 收縮、classwise reliability／近似 AUC、資料身份／checksum、未來標籤不影響擬合、時間隔離、子程序例外及真實 blocked-reader timeout 回收。沒有匯入 NumPy／PyTorch 或執行模型。

合成壓力測試以兩個有界 writer threads 建立並讀取 534,327 筆陣列，總量 448,835,000 bytes（約 428 MiB），行程峰值 RSS 38,567,936 bytes（約 36.8 MiB），約 1.21 秒。這只驗證產物讀取器的有界映射，不代表 CUDA、真實 DataLoader 或模型的記憶體／速度。詳細數據、測試命令及核心檔雜湊保存在同名 [JSON](signal_diagnostics_implementation_20261018.json)。

### 尚待雲端驗證的範圍

`tests/test_signal_prediction_export.py` 與 `tests/test_signal_diagnostic_cloud.py` 需要專案 ML 依賴，本機未執行。未驗證真實 CUDA checkpoint 載入、推論、行情 label 重算或新 calibration 效果。之後可在雲端現有專案環境執行：

```bash
"${PROJECT_ROOT}/.venv/bin/python" -m pytest   tests/test_signal_prediction_export.py tests/test_signal_diagnostic_cloud.py
```

本次未執行任何 RunPod sync、下載、啟動、訓練或推論；未檢查／建立本機 ML 環境、未安裝依賴、未產生 lockfile、未 push。模型、backbone、head、訓練 loss、標籤、年度舊採樣、資料切分及惰性讀取程式均未變更。歷史 date-mixing 欄位只在唯讀載入時解析，不加入目前訓練設定，也不放寬 resume 合約。

現有 checkpoint 已用完整 validation 選模，時間隔離校準仍屬事後診斷；未來改善須在事先保留或後續時間段確認。使用說明清楚保留原 diagnostic lifecycle guard 的 Pod 終止行為；診斷工作不得與既有訓練共用活動 GPU lease。

## English

Date: 2026-10-18. Branch: `codex/model-performance-20261016`; implementation base: `938018826fb98d78cfd331b3b3046d536ed6335d`. Implementation and checks used the isolated Codex worktree. This delivers diagnostic and controlled calibration tooling, with no new model performance result or demonstrated predictive improvement.

### Delivered behavior

1. Ported and extended Claude's read-only prediction export and download contract from `27fb9daaad485d1e22310c8d7d275b528b68c83a`. Validation-best and retained named checkpoints, batched inference, original metrics, hashes, GPU lease and existing tmux guard remain available. A fixed bounded train evaluation sample avoids materializing all training windows.
2. Added at most 32 high-loss/drawdown cases and same-date/market ordinary controls, including raw/adjusted prices, PIT inputs and independent next-open-to-horizon-close labels. CPU/memory-budgeted DataLoader workers report their actual plan. Internal consistency does not independently verify vendor corporate actions.
3. Added a standard-library offline analyzer comparing full-train priors, raw logits, global temperature and global temperature plus shrunk class bias under identical procedures. Fitting uses earlier validation; scoring uses later validation with at least fourteen observed market dates between them. Test is never opened and no winner or production configuration is applied automatically.
4. Added immutable protocols, streamed per-model detail, comparison and status records, source/input hashes, fixed ordinals and market/month groups. Each array maps approximately 4 MiB. Spawn workers, pending jobs, fitting rows, report months and persisted artifacts are bounded. Failure, cancellation and timeout terminate and reap workers.
5. Added a complete Chinese-then-English [guide](../docs/SIGNAL_DIAGNOSTICS.md), README entries and existing CLI-reference coverage.

### Local verification

| Check | Result |
| --- | --- |
| New stdlib numerical/control/error/timeout tests | 22 passed |
| Project naming and CLI controls | 13 passed |
| README CLI references | 10 passed |
| Mocked tmux regression | 16 passed |
| Total | 61 passed |
| Ruff 0.15.21 lint/format | 13 Python files passed |
| AST/Bash syntax | 13 Python/4 shell files passed |
| Protected existing core sources | 23 files byte-identical to the base commit |

Numerical coverage includes finite-difference gradients, extreme-logit log loss, temperature argmax invariance, bias shrinkage, classwise reliability and approximate AUC. Control coverage includes identities/checksums, future-label isolation, temporal separation, propagated worker failures and a real blocked-reader timeout with child-process cleanup. These checks import no NumPy/PyTorch and execute no model.

The bounded two-thread synthetic artifact stress test used 534,327 rows, 448,835,000 bytes (approximately 428 MiB), peak process RSS 38,567,936 bytes (approximately 36.8 MiB), and approximately 1.21 seconds. This measures the artifact reader alone, not CUDA, real DataLoader or model memory/throughput. The matching [JSON](signal_diagnostics_implementation_20261018.json) records commands, exact measurements and protected-source hashes.

### Pending cloud validation

`tests/test_signal_prediction_export.py` and `tests/test_signal_diagnostic_cloud.py` require project ML dependencies and were not run locally. Real CUDA checkpoint loading, inference, market-label reconstruction and calibration effects remain unverified. The existing cloud project environment can later run:

```bash
"${PROJECT_ROOT}/.venv/bin/python" -m pytest   tests/test_signal_prediction_export.py tests/test_signal_diagnostic_cloud.py
```

No RunPod sync, download, launch, training or inference was executed. No local ML environment was inspected/created, no dependencies installed, no lockfile generated, and no push performed. Model backbones/heads, losses, labels, legacy yearly sampler, split semantics and lazy training readers remain unchanged. Historical date-mixing metadata is normalized only for read-only loading; current training config and resume contracts remain strict.

Existing checkpoints were selected with full validation, so this is a post-hoc temporal diagnostic rather than independent confirmation. Predictive improvement requires preregistered or subsequent periods. The guide documents the inherited diagnostic guard's Pod-termination behavior; diagnostics must not share an active training GPU lease.
