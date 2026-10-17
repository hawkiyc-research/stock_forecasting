# Chronos-2 與原有動態採樣

## 繁體中文

### 基底與目前方案

本分支為 `codex/model-performance-20261016`。先以反向提交還原 `0555f8b`，
使程式樹與 `c7b555068c13668e08c28cd13a3fc8862a64df9e` 完全一致，
再單獨加入 Chronos-2。舊提交保留在 Git 歷史中，不重寫其他分支。

目前比較為 **原 Kronos＋原採樣** 對 **Chronos-2＋相同原採樣**。
Alpha 點估計與個股漲／平／跌維持兩個獨立模型，兩個任務同等重要。
更換 backbone 所需的表徵、原生 normalization 與 LoRA targets；
標籤、loss、數值特徵、benchmark conditioning、切分與評估沿用原版本。
同日去均值 loss、日期成分模型及新的正則化設計不在這次變更內。

### 撤回混洗的依據

2026-10-17 使用者提供 Claude 的 checkpoint 摘要：每次更新的日期×市場組合
中位數從 2 增至 166，混洗 alpha 的最佳 validation loss 為 0.4636，
差於 R21 的 0.4548；US 14 日 validation Rank IC 約 0.007 對 0.035。
方向模型當時仍在訓練，部分校準與市場平均指標也較差。
這些是使用者提供的階段性 validation 證據；本輪沒有連線遠端重新核對，
也沒有將其當成新的正式 test 結果。

依使用者決策撤回混洗，改用原採樣比較 Chronos-2。
同日 batch 是否形成隱性正則化仍是假說；本輪不把它視為已證實機制，
也不由單次結果推論所有隨機採樣無效。
歷史資料與品質限制仍見 [開發 roadmap](../MODEL_DEVELOPMENT_ROADMAP.md)。

### 原採樣與動態讀取契約

`training.py`、`date_market_sampler.py`、`data/dataset.py`，以及 labels、
scale calibration 和 baseline 的數值實作，保持與 `c7b5550` 相同。
靜態契約記錄核對來源 SHA-256 與新舊 preset 的資料／訓練／評估內容。

- `yearly_sampling_decay: 0.8` 保留；原 `_training_sampler` 因而使用
  `DateMarketSampler`，即使兩個 signal 任務的 ranking loss 為零。
- `DATALOADER_SELECTION_BLOCK_SIZE=128`；依日期×市場排序的索引仍以連續
  128 筆區段輸出，區段內保留原本的正向／反向走訪。年度配額、區段排列、
  seed、epoch、固定 batch 尾端補齊與 resume 不變。
- 本分支新增的逐樣本 Fisher–Yates 路徑、設定欄位、採樣 audit、四份 random
  presets 與實驗名稱已從目前程式移除，仍可由 `0555f8b` 的歷史查閱。
- `LazyFinancialWindowDataset` 按 batch 從既有 symbol bar-store 動態讀取行情，
  計算 windows 與 labels。多 worker、bounded caches、prefetch、pinned memory、
  non-blocking transfer 與 window 內時間順序沿用原版本。
- 原有精簡 date-market metadata index 依舊建立／重用；不建立完整 windows、
  完整 label matrix 或預計算 embeddings 資料集。
- Validation／test 完整動態遍歷、readiness、完整性檢查、固定切分、train-only
  calibration、baseline 身分及 checkpoint／run 身分規則保留。

### Chronos-2 接入

使用 `chronos-forecasting==2.3.2`、`amazon/chronos-2`，固定權重 revision 為
`29ec3766d36d6f73f0696f85560a422f50e8498c`；參見
[官方模型](https://huggingface.co/amazon/chronos-2) 與
[固定版本程式](https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2)。
通用時序 benchmark 不代表本專案金融標籤上的效果。

- 原生歷史 encoder 只接收截至 cutoff 的 OHLCV，保留 instance normalization
  與 arcsinh。Padding 改成原生左側對齊並以 NaN 排除。
- 每筆 window 的五個 channels 共用 group；不同樣本、asset 與 benchmark
  使用獨立 group。同日 batch 不代表允許 encoder 在樣本之間讀取資訊。
- 128 bars 產生 8 個歷史 patches × 5 channels，共 40 個 tokens，再接原有
  resampler。Encoder 沒有未來 labels／covariates，輸出排除 REG／future tokens。
- 使用既有 LoRA，targets 為 encoder 的 `q/k/v/o/wi/wo`。Rank64 保留，
  實際 trainable parameter count 仍需分 backbone 記錄。
- 每個任務有自己的 backbone、LoRA、optimizer、head 與 checkpoint。
  尚未驗證的 gradient checkpointing 會拒絕，context 維持 128。
- 評估的既有主模型 key `kronos_full` 保留；辨識 backbone 必須讀取保存的
  `config.model.time_series_backend`、model ID 與 revision。

### 實驗設定與執行邊界

| 資料組／任務 | 單一實驗名稱 | YAML |
| --- | --- | --- |
| A alpha | `a-chronos2-alpha-point64` | `configs/experiments/a_chronos2_alpha_point64.yaml` |
| A 方向 | `a-chronos2-absolute-direction64` | `configs/experiments/a_chronos2_absolute_direction64.yaml` |
| B alpha | `b-chronos2-alpha-point64` | `configs/experiments/b_chronos2_alpha_point64.yaml` |
| B 方向 | `b-chronos2-absolute-direction64` | `configs/experiments/b_chronos2_absolute_direction64.yaml` |

四份設定的 `data`、`training`、`validation` 區段與原 Kronos preset 相同。
既有 `a-signal64`／`b-signal64` 成對入口保留原 Kronos；本輪沒有新增 paired aliases。
新 YAML 也可透過 `stock-forecasting-train --config PATH` 使用。

套件沿用 Poetry；Chronos-2 是選用 extra，雲端 setup 使用
`RUNPOD_POETRY_EXTRAS=chronos2`，保留雲端 lock 與映像提供的 PyTorch／CUDA。
本機不建立／檢查 ML 環境或產生 lock，也不覆寫其他訓練正在使用的雲端 checkout。
本輪只修改與提交自己的 worktree，沒有 RunPod sync、Pod 操作、訓練或 push。

### 驗證範圍

本機只執行 stdlib／控制流程／靜態檢查，涵蓋原採樣 selector、真實 sampler 的
128 筆區段行為、四份 preset、baseline／資料身分及既有功能回歸。
本輪通過 46 項：新增控制測試 6 項、既有 training pipeline 13 項、signal runtime 6 項、
README 契約 17 項、套件權責 fixture／靜態測試 4 項。另已檢查 Python 3.12 與 shell 語法。
[來源與設定契約](../reports/chronos2_legacy_sampling_contract_20261017.json)
僅證明程式與設定一致性，不能作為訓練完成或模型品質證據。

已準備雲端測試，涵蓋原年度採樣、真實 lazy bar-store＋兩個 spawn workers、
固定 batch／resume、原生 encoder 等價性、padding、樣本隔離、LoRA 梯度與兩任務 loss。
ML／CUDA 測試與兩項需要雲端 `poetry.lock` 的 repository 檢查仍待雲端執行。
下列命令僅供之後的隔離雲端驗收，本輪未執行：

```bash
poetry run pytest tests/test_chronos2_legacy_sampling.py tests/test_chronos2_backbone.py \
  tests/test_signal_model.py tests/test_config_contract.py tests/test_poetry_runtime_ownership.py
```

後續比較須列出實際 microbatch／accumulation、參數量、吞吐與記憶體；
有效 batch256 或相同 GPU 名稱不足以代表所有執行條件相同。
使用固定 validation 指標選模，兩個任務分別評估，保留市場／持有期與多 seed 診斷。
目前尚無 Chronos-2 新訓練結果。

## English

### Baseline and current scope

Work remains on `codex/model-performance-20261016`. A reverse commit restores the
tree exactly to `c7b555068c13668e08c28cd13a3fc8862a64df9e`, then Chronos-2 is added
separately. Commit `0555f8b` remains in history; other branches are not rewritten.

The comparison is **original Kronos/original sampling** versus
**Chronos-2/the same original sampling**. Alpha point regression and absolute stock
movement classification remain separate, equally important models. The backbone
representation, native normalization and LoRA targets change; labels, losses,
numerical features, benchmark conditioning, splits and evaluation rules remain.
Cross-sectional demeaning, a date-component model and new regularization objectives
are outside this change.

### Why mixing was reverted

On 2026-10-17, the user supplied Claude's checkpoint summary: median date-market
groups per update increased from 2 to 166, while mixed alpha's best validation loss
was 0.4636 versus R21's 0.4548. US 14-day validation Rank IC was approximately 0.007
versus 0.035. Classification was still running, with some worse calibration and
market-macro metrics. These are user-supplied interim validation observations;
this work did not independently inspect remote artifacts or treat them as new final
test evidence.

The user chose to compare Chronos-2 with the original sampler. Implicit regularization
from same-date batches remains a hypothesis; neither that mechanism nor universal
failure of random sampling is established here. Historical quality/data limitations
remain in the [roadmap](../MODEL_DEVELOPMENT_ROADMAP.md).

### Exact sampling and lazy loading contract

`training.py`, `date_market_sampler.py` and `data/dataset.py` remain identical to
`c7b5550`, along with numerical label, scale and baseline implementations. The static
artifact checks source hashes and preset data/training/evaluation sections.

- `yearly_sampling_decay: 0.8` selects the original `DateMarketSampler`, including
  signal tasks with zero ranking loss. `DATALOADER_SELECTION_BLOCK_SIZE=128` retains
  contiguous date-market index blocks and original forward/reverse traversal.
  Annual quotas, block ordering, seed/epoch behavior, fixed tail padding and resume remain.
- The newly added sample-level Fisher–Yates code, configuration fields, sampling
  audit, four random presets and experiment names are removed from the live tree.
  They remain inspectable in commit `0555f8b`.
- `LazyFinancialWindowDataset` reads batches from the symbol bar-store and calculates
  windows/labels dynamically. Existing bounded workers/caches, prefetch, pinned memory,
  non-blocking transfers and temporal order within windows remain. Compact date-market
  metadata is built/reused as before, without complete window, label-matrix or embedding datasets.
- Validation/test traverse all eligible windows dynamically. Readiness, integrity,
  fixed-time splits, train-only calibration, baseline identity and checkpoint/run
  identity contracts remain mandatory.

### Chronos-2 integration

The adapter pins `chronos-forecasting==2.3.2` and `amazon/chronos-2` at
`29ec3766d36d6f73f0696f85560a422f50e8498c`; see the
[official model](https://huggingface.co/amazon/chronos-2) and
[pinned implementation](https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2).
General forecasting benchmarks do not establish performance on these financial labels.

The native encoder receives observed OHLCV only, retaining instance normalization/arcsinh.
Padding becomes native left alignment and NaN. Five channels within each window share
a group; other samples and asset/benchmark histories have separate groups. Same-date
batching does not enable cross-sample encoder information flow. Eight historical
patches across five channels produce 40 tokens for 128 bars, feeding the existing
resampler without REG/future output tokens or future labels/covariates.

Existing LoRA targets encoder `q/k/v/o/wi/wo`, retaining rank64. Actual trainable counts
still differ across backbones. Each task owns its backbone, LoRA, optimizer, head and
checkpoints. Unverified gradient checkpointing is rejected; context remains128.
The legacy main-model key `kronos_full` stays; identify the model using the saved
backend, model ID and revision.

### Presets and execution boundaries

| Group/task | Individual experiment | YAML |
| --- | --- | --- |
| A alpha | `a-chronos2-alpha-point64` | `configs/experiments/a_chronos2_alpha_point64.yaml` |
| A direction | `a-chronos2-absolute-direction64` | `configs/experiments/a_chronos2_absolute_direction64.yaml` |
| B alpha | `b-chronos2-alpha-point64` | `configs/experiments/b_chronos2_alpha_point64.yaml` |
| B direction | `b-chronos2-absolute-direction64` | `configs/experiments/b_chronos2_absolute_direction64.yaml` |

Data, training and validation sections match original Kronos presets. Existing
`a-signal64`/`b-signal64` pairs retain Kronos; no new paired aliases are added. Presets
also work through `stock-forecasting-train --config PATH`.

Poetry remains the package manager. `RUNPOD_POETRY_EXTRAS=chronos2` enables the optional
extra during cloud setup, preserving cloud-only lock generation and image-owned
PyTorch/CUDA. No local ML environment inspection/creation or lock generation is
performed. Do not overwrite a cloud checkout used by another training run. This work
only modifies/commits its own worktree, with no RunPod sync, Pod operation, training or push.

### Verification limits

Local standard-library/control/static checks cover the original selector, actual
sampler block behavior, four presets, data/baseline identities and existing functionality.
Forty-six checks passed: six new controls, thirteen training-pipeline controls, six
signal-runtime controls, seventeen README contracts and four dependency-ownership
fixture/static checks. Python 3.12 and shell syntax were also checked.
The [source/configuration artifact](../reports/chronos2_legacy_sampling_contract_20261017.json)
records static consistency, not training completion or prediction quality.

Cloud tests cover native annual sampling, real lazy bar-store loading with two spawn
workers, fixed batches/resume, encoder equivalence, padding, sample isolation, LoRA
gradients and both task losses. ML/CUDA tests and two repository checks requiring
the cloud `poetry.lock` remain pending. The Poetry pytest command above is for later
isolated cloud validation and was not executed here.

Later comparisons must report actual microbatch/accumulation, parameter counts,
throughput and memory; effective batch256 or matching GPU names alone do not control
all execution conditions. Select on fixed validation metrics and assess both tasks
separately, with market/horizon and multiple-seed diagnostics. No new Chronos-2
training result exists.
