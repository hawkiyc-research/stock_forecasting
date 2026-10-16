# 動態隨機採樣與 Chronos-2 受控比較

## 繁體中文

### 範圍與判斷

本輪從 `c7b555068c13668e08c28cd13a3fc8862a64df9e` 分出
`codex/model-performance-20261016`。Alpha 點估計與個股漲／平／跌同等重要，
可依預測證據比較其他時序 backbone。模型仍提供預測訊號；標籤不加入費用、部位或交易規則。
較早 roadmap 中保留 Kronos 主線的限制，已由這次允許比較其他 backbone 的需求更新。
歷史 preset 與結果繼續保留。

建議先修復已確認的 batch 集中，再比較 backbone。兩者可在同一分支完成工程，
實驗必須分開：Kronos＋舊採樣、Kronos＋新採樣、Chronos-2＋相同新採樣。
前兩者衡量採樣影響；後兩者衡量 backbone 影響。新增程式尚無雲端訓練品質結果。
本輪沒有 RunPod sync、Pod 啟動、遠端訓練、本機模型訓練或 push。

### 已有實驗告訴我們什麼

現有證據詳見 [開發 roadmap](../MODEL_DEVELOPMENT_ROADMAP.md) 第 3、5、6 節，
以及三份正式報告：

- [Seed 42](../reports/training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx)。
- [Seed 347](../reports/training_result_analysis_run-20261015T064559Z-624f212a27894308.docx)。
- [Seed 5067](../reports/training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx)。

| 已觀察項目 | 數值與實際意義 |
| --- | --- |
| Alpha normalized Smooth L1，越低越好 | Seeds 42／347／5067：0.528707／0.540340／0.536003；GRU 0.529025。Seed42 的日配對區間跨零，尚無穩健跨 seed 優勢。 |
| 個股方向 CE，越低越好 | 1.039206／1.032702／1.027663；GBDT 0.999028。Accuracy 的改善沒有轉成較好的整體機率品質。 |
| 共同 test | 534,327 windows、112 個相依 cutoff dates；membership SHA-256 `93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`。 |
| 市場與持有期差異 | US 占 82.30%；TPEX 長持有期排序與 neutral recall 仍弱，不能只依樣本整體分數選主模型。 |
| 訓練條件 | Seed42 使用 32×8；347／5067 使用 64×4，有效 batch 同為 256。它們沒有控制所有 microbatch 條件。 |

歷史容量擴大、partial unfreezing、adaptive 與方向頭改版未建立一致優勢。
這支持先處理可直接確認的資料呈現方式，再增加模型候選。

### 已確認的採樣問題

現行 `yearly_sampling_decay=0.8` 即使 ranking loss 為零，也會使用
`DateMarketSampler`。它的索引依日期、市場排序，再以連續 128 筆作為混合單位；
block 內仍按正序或反序讀取。`FixedSizeBatchSampler` 只切 batch，沒有再洗牌。
因此有效 batch 256 常集中在約兩個日期×市場區段；群組邊界會影響精確數量。
本機缺少該次實際 date-market index，不能把「每步恰好兩組」寫成實測統計。
這個機制具有隨機成分，但不符合這輪要求的逐樣本充分混合。

新設定使用 `training.sampling_strategy: sample_random`：

1. 保留原年度質量 `count(year) × 0.8 ** age`，以 largest remainder 配出精確 epoch 額度。
2. 年度內以 Fisher–Yates 逐筆選樣；每個選樣 cycle 不重複。額度超過該年筆數時，
   選取完整 cycles 加上一個部分 cycle。這保留年度加權的重複抽樣語意。
3. 對所有年度選出的單筆索引再做一次完整 Fisher–Yates shuffle，然後切 microbatch。
   最後的順序不保留年度、日期、市場或 cycle 區塊；重複索引可能被隨機排到相近位置。
4. Seed、epoch 決定結果。縮小的 Stage 1 subset 保持 membership 固定而改變順序；
   Stage 2 的年度部分選樣隨 epoch 更新。接續訓練重新生成同一順序並跳過已完成 batches。
5. 保留固定 batch 的尾端補齊，補齊筆數繼續記錄；validation／test 的完整遍歷不改動。
6. `sample_random` 與要求同日期 pairs 的 ranking objective 同時開啟會明確失敗。
   歷史 ranking preset 繼續使用 `legacy_blocks`。

`legacy_blocks` 仍是未指定新欄位時的相容預設。八份新實驗 YAML 明確指定
`sample_random`；執行舊的 `a-signal64`／`b-signal64` 入口不會自動切到新版本。

### 動態讀取與資源限制

`LazyFinancialWindowDataset`、按 symbol 儲存的 bar-store、`__getitems__` batch 讀取、
有上限的 symbol／Parquet cache、persistent workers、prefetch、pinned memory 與
non-blocking transfer 沿用既有管線。每個 worker 透過 compact metadata 重開唯讀索引，
不序列化整份 dataset 或 cache。每個 128 日 window 內仍維持時間順序。

新增暫存只包含 `uint64` ordinal。使用 mmap 的 order／cycle 檔案，其上限為
`8 × (epoch quota + largest selected stratum)` bytes；未加年度權重的 4,710,117 筆案例
為 75,361,872 bytes，約 71.87 MiB。預設 `sampling_scratch_max_gib=1.0`，
超過額度或磁碟不足會直接失敗。每個 iterator 使用獨立暫存目錄，正常完成或取消會釋放。
年度權重需要的 date-market index 是既有精簡 metadata cache；不包含行情 windows。
不產生完整的訓練 windows、完整 label matrix 或預計算 embeddings。

Fisher–Yates 的 swap 具有循序相依性，索引置換在主程序執行；磁碟資料讀取、window
組裝與既有可離線處理的工作維持有資源預算的平行處理。新正式設定的 DataLoader
若沒有安全可用的子 worker，會明確失敗。雲端仍須觀察吞吐、batch wait、RAM、GPU 與
`/dev/shm`；索引測試不能代表實際 Parquet 隨機 I/O 的速度。

每個 session 在 run 目錄保存 `sampling-audit-*.json`，預設涵蓋前 128 次真正完成的
optimizer updates，而不是 DataLoader 預取／被跳過的 batches。內容包括每步筆數、
unique dates、unique date×market、unique markets、最大單一日期市場占比及有效群組數。
重新啟動使用新的 session 檔名，保留先前 audit。年度 quotas 與採樣版本也會保存在摘要。

### Chronos-2 的接法與限制

官方 [模型](https://huggingface.co/amazon/chronos-2)、
[技術報告](https://arxiv.org/abs/2510.15821) 與
[v2.3.2 原始碼](https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2)
支持多變量／covariate 時序建模，適合列為候選；公開通用時序 benchmark 不能證明本專案
金融標籤上的優勢。

- Backend：`chronos2`；套件：`chronos-forecasting==2.3.2`；權重：`amazon/chronos-2`。
  固定 revision：`29ec3766d36d6f73f0696f85560a422f50e8498c`。
- 使用官方 `Chronos2Model.encode`，傳入截至 cutoff 的 OHLCV；保留原生 instance
  normalization 與 arcsinh。右側 padding 轉為原生左側對齊後設成 NaN，使其不污染
  normalization 或讓最後一個有效觀測偏離預測 cutoff。
- 每筆 window 的五個 OHLCV channels 共用一個 group；不同樣本、asset 與 benchmark
  各有獨立 group。禁止其他日期／樣本透過 group attention 改變該筆輸出。
- 128 bars 產生 8 個歷史 patches × 5 channels＝40 個 tokens，依 patch 時間順序
  接到現有 resampler。輸出排除原生 REG／future placeholder tokens；encoder 只收到
  history，沒有 future labels 或 future covariates。
- 保留 train-only scales、數值特徵、benchmark conditioner、兩個獨立 head、loss、選模
  與評估；各任務仍有自己的 backbone、LoRA、optimizer 和 checkpoint。
- 使用既有 LoRA 實作，僅更新 encoder 的 `q/k/v/o/wi/wo`；不新增 PEFT 依賴。
  LoRA rank64 不代表與 Kronos 有相同可訓練參數量，報告應列出實際參數量及成本。
- 本輪只支援兩個獨立 signal 任務；Chronos-2 gradient checkpointing 尚未驗證，
  設為 true 會失敗。最大 context 沿用 128，不一併調長或變更 label。
- 評估產物中的 `kronos_full` 保留為既有主模型欄位 key；辨識 backbone 必須讀取保存的
  `config.model.time_series_backend`、model ID／revision，不能只看欄位名稱。

雲端套件沿用 Poetry；`chronos2` 是選用 extra。既有遠端 setup 可用
`RUNPOD_POETRY_EXTRAS=chronos2` 啟用，仍沿用雲端 lock 與映像所提供的 PyTorch／CUDA，
不允許 Poetry 取代映像 GPU 套件。本機未解析 lock、安裝套件或檢查／建立 ML 環境。
新 backbone 的實際權重載入、數值／梯度與 CUDA 記憶體驗收仍待隔離的雲端執行。

### 設定與驗收順序

新 YAML 可透過既有 `stock-forecasting-train --config PATH` 使用，也已註冊為單一模型
experiment，例如 `a-random-alpha-point64`、`a-chronos2-absolute-direction64`。
CLI 名稱將下表檔名的底線換成連字號並去掉 `.yaml`。既有 baseline readiness、完成檢查
及 run 身分規則照常適用。成對入口 `a-signal64`／`b-signal64` 保留原版本；
這輪不增加新的 paired aliases，也沒有執行任何 RunPod 工作流程。

| 資料組 | Kronos＋新採樣 | Chronos-2＋新採樣 |
| --- | --- | --- |
| A alpha | `configs/experiments/a_random_alpha_point64.yaml` | `configs/experiments/a_chronos2_alpha_point64.yaml` |
| A 方向 | `configs/experiments/a_random_absolute_direction64.yaml` | `configs/experiments/a_chronos2_absolute_direction64.yaml` |
| B alpha | `configs/experiments/b_random_alpha_point64.yaml` | `configs/experiments/b_chronos2_alpha_point64.yaml` |
| B 方向 | `configs/experiments/b_random_absolute_direction64.yaml` | `configs/experiments/b_chronos2_absolute_direction64.yaml` |

之後的驗收順序：

1. 在獨立雲端 checkout、獨立環境與輸出目錄，完成以下數值測試及 pinned cache smoke。
   先確認 baseline identity、實際 split membership 與 train-only calibration 符合，
   不能因新增 backbone 而跳過 readiness 或無理由重做資料／baseline。
2. 同一資料組先做 Kronos＋新採樣。固定 seed、microbatch／accumulation 及 optimizer
   設定比較；自動 batch 的實際結果若不同，必須列出。檢視真實 sampling audit 與 I/O。
3. 完成兩個任務後，以相同設定比較 Chronos-2。先做 validation，使用相同已固定的
   selection 規則；研究方向收斂後才完成受控 final test。歷史 test 已被反覆檢視，
   不能當作全新的未見資料；後續 walk-forward 留出新時間區段。
4. 通過第一 seed 再確認至少三 seeds。Alpha 同時檢視 normalized Smooth L1、MAE、
   符號一致率及各市場／持有期的日 Rank IC；分類檢視 CE、Brier、balanced accuracy、
   macro-F1、三類 recall 與類別塌縮。兩任務分別達標，不以其中一項掩蓋另一項退化。
5. 對逐日差異使用考慮時間相依的配對區間，另列市場等權、月份與波動／drawdown
   subgroup。若只剩極端區段失敗，再針對原始行情與 corporate actions 做資料稽核。

前處理尚待釐清：極端報酬／drawdown 是真實事件或調整資料錯誤，需要雲端原始行情才能
核對。本機未見足以確認「重複 normalization」或 label off-by-one 的證據。
固定 split 實際採 label-end guard；文件中的額外 purge／embargo 敘述不應被當成已執行。
本輪保留相同 membership。若另做更嚴格 purge，須另立 split 版本與配對 baseline。

### 本機已完成與尚未完成

- 已完成 12 項新 stdlib 採樣、3 項唯讀 selection、13 項既有 training pipeline、6 項
  signal runtime、17 項 README 契約、4 項套件權責 fixture／靜態測試，共 55 項。
  另已檢查 Python／shell 語法，以及八份 preset 的資料、評估與 optimizer 設定一致性。
  兩項依賴雲端產生之 `poetry.lock` 的 repository 檢查尚待雲端完成。
- [規模測試記錄](../reports/sampling_scale_check_20261017.json)：兩個子程序，seeds42／347，
  各 4,710,117 筆合成索引，完整覆蓋、沒有重複。單輪約 2.32–2.34 秒，程序峰值 RSS 約
  106.7–106.8 MiB。每組連續 128 筆的合成案例中，前 128 次有效 batch256 更新平均約
  255 個不同群組、最少 252。這些群組是合成資料，並非實際交易日期／市場。
- `python3 scripts/check_sampling_scale.py --output /tmp/sampling-scale-check.json`
  可重跑規模測試，worker／seed／樣本數／暫存上限可調整。
- 已新增雲端測試：年度 ordinal 映射、固定 batch／resume、真實 lazy bar-store＋spawn
  workers、八份 YAML、原生 Chronos-2 等價性、padding、不受其他樣本影響、LoRA 梯度
  及兩個任務的 factory／loss。**這些 ML 測試本輪未執行。**
- 雲端訓練品質、實際資料吞吐、完整模型權重載入及 CUDA 峰值記憶體尚未驗證。
  程式與索引測試通過，不能改寫為模型預測能力已改善。

隔離雲端環境中的數值測試命令（本輪沒有執行）：

```bash
poetry run pytest tests/test_random_training_sampler.py tests/test_chronos2_backbone.py \
  tests/test_signal_model.py tests/test_config_contract.py tests/test_poetry_runtime_ownership.py
```

## English

### Scope and decision

This branch, `codex/model-performance-20261016`, starts at
`c7b555068c13668e08c28cd13a3fc8862a64df9e`. Alpha point prediction and absolute stock
movement classification have equal priority. Alternative time-series backbones may
be compared on predictive evidence. Targets remain statistical signals without
fees, position sizing or trading rules. This updates the older roadmap restriction
that required Kronos to remain the research backbone; historical functionality is retained.

Repair the confirmed batch concentration before interpreting another backbone.
Implementation can coexist in one branch, with separate experiments: Kronos with
legacy sampling, Kronos with sample-random sampling, and Chronos-2 with the same
sample-random sampling. The first comparison measures sampling; the second measures
backbone effects. No new cloud quality result exists. This work performed no RunPod
sync, Pod launch, remote training, local model training or push.

### Evidence and diagnosis

The [development roadmap](../MODEL_DEVELOPMENT_ROADMAP.md), sections 3, 5 and 6, and
the three linked seed reports above provide the historical evidence. Alpha
normalized Smooth L1 is 0.528707/0.540340/0.536003 for seeds42/347/5067, compared with
GRU 0.529025. Seed42's date-paired interval crosses zero. Classification CE is
1.039206/1.032702/1.027663 versus GBDT 0.999028. A robust advantage is unproven.

All six signal children share 534,327 test windows and 112 dependent cutoff dates,
with membership SHA-256
`93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`.
US represents 82.30% of windows; market/horizon failures remain material. Seed42
used 32×8 microbatches/accumulation, while 347/5067 used 64×4. Effective batch256
alone does not make them identical execution conditions. Historical capacity,
partial unfreezing, adaptive schedules and direction heads did not establish a
consistent improvement.

With `yearly_sampling_decay=0.8`, training selects `DateMarketSampler` even when
ranking loss is zero. Its index is sorted by date and market, and contiguous
128-sample blocks remain forward or reversed internally. The fixed batch sampler
only cuts that stream. Updates of 256 samples can therefore concentrate in roughly
two date/market regions. Boundaries affect the exact number; the original runtime
index is not available locally, so this is a verified mechanism, not a measured
claim that every historical update contained exactly two groups.

### Sample-random semantics and dynamic loading

New presets explicitly set `training.sampling_strategy: sample_random`.
Annual mass remains `count(year) × 0.8 ** age`; largest-remainder quotas retain the
exact epoch budget. Each annual selection cycle is sampled without replacement
using Fisher–Yates. Oversampling selects complete cycles plus a partial cycle.
A final Fisher–Yates permutation mixes all selected individual ordinals across
years, dates, markets and cycle boundaries. Repeated selected indices may appear
close together in the resulting random presentation order.

Seed and epoch determine the sequence. Reduced Stage1 subsets retain membership
while changing presentation order; Stage2 partial annual selections vary by epoch.
Resume regenerates the same order and skips completed batches. Fixed-size training
tail padding remains explicit. Validation/test traversal is unchanged. Grouped
ranking objectives cannot be combined with sample-random training; historical
ranking presets retain `legacy_blocks`, the compatibility default.

The data path remains `LazyFinancialWindowDataset` with symbol-oriented bar-store
and batched `__getitems__`. Windows retain chronological order internally. Existing
bounded symbol/Parquet caches, multiprocessing workers, prefetch, pinned memory and
non-blocking transfers remain in place. Workers reopen compact read-only metadata
rather than receiving complete datasets/caches. New production sample-random
loaders fail if no safe child worker fits their resource budget.

Only uint64 ordinals are staged in disk-backed mmap files. The bound is
`8 × (epoch quota + largest selected stratum)` bytes: 75,361,872 bytes, or 71.87 MiB,
for an unweighted epoch of 4,710,117 samples. The default scratch cap is 1 GiB;
insufficient storage or budget fails explicitly. Each iterator owns a unique
scratch directory and releases it on completion/cancellation. The existing annual
date-market cache contains metadata. No complete windows, complete label matrix or
precomputed embedding dataset is created.

Fisher–Yates swaps are sequentially dependent, so index permutation runs in the
parent process. Data reads and window construction retain bounded parallelism.
Cloud throughput, batch wait, RAM, GPU memory and shared-memory usage still need
measurement; index-only performance does not establish random Parquet I/O speed.

Each training session writes a distinct `sampling-audit-*.json` in its run directory.
By default it records the first 128 completed optimizer updates using actually
consumed microbatches, excluding prefetch and skipped resume batches. Records contain
sample counts, unique dates/date-market groups/markets, largest group share and
effective group count. Quotas and the sampling version are persisted in run summaries.

### Chronos-2 integration

The [official model](https://huggingface.co/amazon/chronos-2),
[technical report](https://arxiv.org/abs/2510.15821), and
[pinned implementation](https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2)
justify a multivariate/covariate-capable candidate, without establishing an advantage
on these financial targets. The backend is `chronos2`, package
`chronos-forecasting==2.3.2`, model `amazon/chronos-2`, revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

The adapter calls native `Chronos2Model.encode` on observed OHLCV history and keeps
native instance normalization/arcsinh. Right padding is converted to native left
alignment, then masked values become NaN before normalization; the final observation
stays adjacent to the forecast cutoff.
Five channels within one window share a group; different samples and the asset and
benchmark histories receive separate groups. No future labels or covariates enter
encoding. Eight context patches across five channels become 40 tokens for 128 bars,
ordered by patch time. Native REG/future placeholder tokens are omitted from the
adapter output.

Existing train-only calibration, numerical features, benchmark conditioning,
independent heads/losses, selection and evaluation remain. Each task owns its own
backbone, LoRA, optimizer and checkpoints. Existing LoRA targets only native encoder
`q/k/v/o/wi/wo` layers, with no PEFT dependency. Rank64 does not equate trainable
parameter counts across architectures; report actual counts and costs. Context stays
128, and the backend supports the two independent signal tasks. Unverified Chronos-2
gradient checkpointing is rejected. The legacy `kronos_full` benchmark key is retained;
use the persisted backend, model ID and revision to identify the evaluated model.

Poetry remains the dependency manager. The optional `chronos2` extra is enabled by
`RUNPOD_POETRY_EXTRAS=chronos2` during the existing cloud-only setup. Cloud lock
resolution and the image-owned PyTorch/CUDA protection remain. No local dependency
resolution, installation or ML environment inspection/creation was performed.
Actual snapshot loading, numerical/gradient behavior and CUDA memory remain pending.

### Presets and acceptance order

The presets cover A/B data groups and both independent tasks:

| Group/task | Kronos with random sampling | Chronos-2 with random sampling |
| --- | --- | --- |
| A alpha | `configs/experiments/a_random_alpha_point64.yaml` | `configs/experiments/a_chronos2_alpha_point64.yaml` |
| A direction | `configs/experiments/a_random_absolute_direction64.yaml` | `configs/experiments/a_chronos2_absolute_direction64.yaml` |
| B alpha | `configs/experiments/b_random_alpha_point64.yaml` | `configs/experiments/b_chronos2_alpha_point64.yaml` |
| B direction | `configs/experiments/b_random_absolute_direction64.yaml` | `configs/experiments/b_chronos2_absolute_direction64.yaml` |

They work through `stock-forecasting-train --config PATH` and are also registered as
individual experiments, such as `a-random-alpha-point64` and
`a-chronos2-absolute-direction64`. Replace underscores in the filename with hyphens
and remove `.yaml` for the experiment name. Old `a-signal64`/`b-signal64` paired
entrypoints retain their presets; no new paired aliases were added. Baseline
readiness, completion gates and run identity contracts remain mandatory.
No RunPod workflow was executed in this round.

In an isolated cloud checkout/environment with distinct output directories:

1. Run numerical tests and the pinned cache smoke. Verify baseline identities,
   split membership and train-only calibration; avoid unnecessary data/baseline rebuilds.
2. Compare Kronos/random against Kronos/legacy, fixing the data group, seed,
   microbatch/accumulation and optimizer settings. Report automatic batch differences
   and inspect actual sampling audit and I/O behavior.
3. Complete both tasks, then compare Chronos-2 under the same new sampling. Select
   on validation using fixed rules before controlled final testing. Historical test
   data has been inspected repeatedly; reserve new periods for walk-forward evidence.
4. Confirm at least three seeds after the first promising result. For alpha, report
   Smooth L1, MAE, sign agreement and daily Rank IC by market/horizon. For classification,
   report CE, Brier, balanced accuracy, macro-F1, each class recall and collapse.
   Each task must improve independently.
5. Use paired intervals that respect temporal dependence and report equal-market,
   month, volatility and drawdown subgroups. Inspect raw corporate actions/adjustments
   if extreme groups remain problematic.

Raw cloud bars are needed to distinguish real extreme moves from adjustment errors.
Local evidence did not confirm double normalization or an off-by-one label error.
The fixed split actually uses label-end guards; extra purge/embargo prose must not
be mistaken for executed behavior. Membership stays fixed here. A stricter purge
experiment needs a distinct split version and matched baselines.

### Verification status

Twelve new standard-library sampling tests, three read-only selection tests, thirteen existing
training-pipeline control tests, six signal-runtime control tests, seventeen README
contract tests and four dependency-ownership fixture/static tests passed: 55 checks.
Python syntax, shell syntax and preset data/optimizer equality were also checked.
Two repository checks require the cloud-generated `poetry.lock` and remain pending. The
[scale artifact](../reports/sampling_scale_check_20261017.json) uses two processes,
seeds42/347 and 4,710,117 synthetic ordinals per process: exact coverage, no repeats,
approximately 2.32–2.34 seconds per run and 106.7–106.8 MiB peak process RSS. For synthetic
groups of 128 consecutive samples, the first 128 updates of 256 samples averaged
about 255 distinct groups, minimum252. These are not real market/date observations.
Reproduce with `python3 scripts/check_sampling_scale.py --output /tmp/sampling-scale-check.json`;
workers, seeds, sample count and scratch cap are configurable.

Cloud tests were added for annual ordinal mapping, fixed batching/resume, the real
lazy bar-store with spawn workers, all eight presets, native Chronos-2 equivalence,
padding, sample isolation, LoRA gradients and both factory/loss paths.
**Those ML tests were not executed in this round.** Cloud training quality, actual
market-data throughput, full snapshot loading and CUDA memory remain unverified.
The Poetry pytest command in the Chinese section is for that later isolated cloud
validation. Source implementation and index tests do not prove better predictions.
