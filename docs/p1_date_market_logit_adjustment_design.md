# P1 設計：分類器訓練時的日期×市場 logit adjustment / P1 Design: Date×Market Logit Adjustment for Classifier Training

## 繁體中文

狀態：**設計草稿，待審閱；尚未實作，也未授權任何 GPU 訓練。** 本文件只定義方法、實驗與事先固定的
validation 判準。是否啟動取決於 P0（逐樣本匯出與誤差拆解）的結果，見第 4.1 節。

**2026-10-18 更新：** P0 已完成，S0 成立：移除擇時讓 R21–R23 validation CE 平均改善 0.0146，門檻是 0.005。參照組數值見[誤差拆解報告 §7.6](error_decomposition_audit.md)。第 7 節的待確認事項仍待審閱。

### 1. 要解決的問題

固定 ±1% 三分類的獨立分類器（`a-absolute-direction64`，R21–R23）在 validation 上只勉強追平
baseline，在 test 上甚至輸給固定的 train 先驗。誤差拆解指出原因集中在日期層級：

| 證據 | 數值 | 來源 |
| --- | --- | --- |
| 已知當日類別比例可降低的 CE（oracle） | validation：train 先驗 1.03305 → 0.97724 | [誤差拆解報告](error_decomposition_audit.md) |
| Baseline 的擇時能力 | GRU／GBDT 皆為負 | 同上 |
| 可學的部分 | 同日鑑別力：GBDT 1.03305 − 1.00469 = 0.0284；GRU 0.0237 | 同上（train 先驗正規化） |
| Kronos 分類 validation CE（loss 選中點） | seed 42／347／5067：1.02844／1.01311／1.01663（平均 1.01939） | R21–R23 `metrics.jsonl` |
| Kronos 逐日落後 GBDT 與當日上漲比例的相關（test，描述性） | R21–R24：0.49／0.40／0.11／0.81 | [roadmap](../MODEL_DEVELOPMENT_ROADMAP.md) P0 列 |
| 打亂日期順序（R24）後 | 擇時偏誤更嚴重，test 10–14 日約 91% 預測為下跌 | roadmap §5.6 |

解讀：同一日期、同一市場的股票共享大盤走勢，因此一天之內的類別比例隨大盤大幅擺動。模型可以從
benchmark 等輸入去猜「這一天整體會漲還是跌」，而這種擇時在 train 期間可擬合，換到新的期間卻不
可轉移。P1 的目標是在訓練時就把日期層級的類別比例當成已知的固定效果，讓模型只學「在同一天裡，
哪些股票比較可能漲／跌／持平」。

### 2. 方法

#### 2.1 定義

- 樣本 i 屬於日期 d、市場 m；天期 h ∈ {1,…,14}；類別 k ∈ {down, neutral, up}，標籤 y 沿用現行
  固定 ±1% 定義（`log1p(±0.01)`，不改）。
- `n[d,m,h,k]`：**完整 train 母體**中，日期 d、市場 m、天期 h 的類別 k 樣本數（不受每 epoch 取樣
  影響）；`N[d,m,h]` 為其總和，只計有限標籤。
- `p̄[m,h,k]`：完整 train 的市場×天期先驗，等於 `class-label-audit.json` 的計數比例
  （例：US 1 日 neutral 42.8%、14 日 10.4%）。
- 模型 head 的輸出 `s(x) ∈ R^{14×3}` 解讀為「同日相對 log-odds」，架構與現行完全相同。

#### 2.2 訓練時的留一平滑 offset

```text
π_{-i}[k] = ( n[d,m,h,k] − 1{y_i,h = k} + α · p̄[m,h,k] ) / ( N[d,m,h] − 1 + α )
z_train   = s(x_i) + log π_{-i}           # offset 為常數，不反傳梯度
loss      = CE(z_train, y_i)              # 其餘 masking、天期等權、市場權重皆不變
```

- **留一（leave-one-out）**：扣掉樣本自己的標籤，避免 offset 含有自身答案。每個日期×市場有
  數百到數千檔股票（validation 平均每日 US 約 3,848、TWSE 約 585、TPEX 約 284），所以留一的
  影響很小，但成本為零，因此採用。
- **平滑**：向該市場×天期的 train 先驗收縮，強度 α = 10（虛擬樣本數，事先固定、不調參）。
  只在樣本數很少的日期有實質作用，避免某類別為 0 時 log 發散。
- 計數只用 train 標籤；validation／test 的標籤從不進入 offset。
- 與 batch 組成無關：offset 依每個樣本的（日期, 市場, 天期）查表取得，不假設 micro-batch
  只含單一日期×市場。

#### 2.3 推論與評估

```text
z_infer = s(x) + log p̄[m,h]               # 固定的 train 市場×天期先驗，不使用任何未來資訊
```

- Validation、test、匯出與實際推論一律輸出 `z_infer`；validation 選模、early stopping 與評估
  store 全部沿用現行流程，只是 logits 已含 train 先驗。
- 預期效果：每日平均預測分布會貼近 train 先驗，不再隨模型對大盤的猜測擺動。CE 的理論位置約為
  「固定 train 先驗 CE − 模型同日鑑別力」。若 Kronos 的同日鑑別力與 GRU／GBDT 相當（0.024–0.028），
  validation CE 約落在 1.005–1.009；這是預估，不是判準。
- 「以近期已到期標籤估計先驗」的版本需要等標籤到期（label-end guard），會引入時變假設，
  不在本次範圍。

#### 2.4 考慮過的其他做法

| 做法 | 不採用為主方案的原因 |
| --- | --- |
| 只做推論期的先驗正規化（P0 正在評估） | 不需訓練，是本方案的對照組；但模型仍在訓練時花容量擬合擇時。P1 的價值正是要超越它（判準 C1） |
| 以 micro-batch 內標籤計算 π | 每個 micro-batch 只有 32 筆，14 日 neutral 約 3 筆，雜訊大，而且包含樣本自身標籤 |
| 每個日期一個可學的 bias 參數 | 推論時新日期沒有參數可用，效果等同本方案但估計更差 |
| 精確的條件概似（conditional logit，消去固定效果） | 三類別、每日數百到數千樣本時計算量過大；留一 offset 是其可行近似 |
| 改用依波動度縮放的標籤帶 | 改變產品定義，屬 P2，需另行授權 |
| 隨機混洗日期（R24） | 已證實使擇時更嚴重 |

### 3. 實驗設計

- **單一因子**：新 preset `a-absolute-direction64-dateadj`，只加入 2.2–2.3 的 offset（α = 10），
  資料組別、架構、取樣（同日區塊、年度衰減 0.8）、optimizer、選模指標、seed 等其餘數值全部與
  `a-absolute-direction64` 相同。這是新的 run identity，不接續舊 checkpoint。
- **只訓練分類器**：alpha 模型不變，以單模型實驗執行（`train --experiment a-absolute-direction64-dateadj
  --seed …`），GPU 時間約為一次成對訓練的一半。
- **Seeds 與配對比較**：沿用 42、347、5067，與 R21／R22／R23 逐 seed 配對。
- **順序**：先跑 seed 42 作為 pilot（類似 R24）。pilot 通過第 4.4 節的繼續條件後，才補跑另外兩個 seed。
- **評估**：每個 P1 checkpoint 都用 `export-signal-predictions` 匯出逐樣本預測，並以
  `scripts/analyze_signal_decomposition.py` 和 baseline、R21–R23 放在同一份拆解中比較。

### 4. 事先固定的 validation 判準

以下全部只用完整 validation。參照組 R_s 是 R21–R23 中同一 seed 的 loss 選中 checkpoint，數值來自
P0 匯出後的分析。這些參照數值會在 P1 開始訓練之前寫入實驗協定 §13.3 並凍結；P1 的任何結果都不得
回頭改變判準。

#### 4.1 啟動門檻 S0（依 P0 結果）

- **S0**：P0 中，R21–R23 經無標籤先驗正規化後的 validation CE 改善量，三 seed 平均至少 **0.005**。
- 理由：GBDT 的改善為 0.0092、GRU 為 0.0045。若 Kronos 的改善小於 0.005，代表擇時不是它的主要
  誤差來源，P1 的預期效益就小，不值得投入訓練。
- S0 不成立時不實作 P1，改回到 P0 結果重新討論。

#### 4.2 主要判準 C1：訓練期調整必須勝過免費的推論期調整

- 三 seed 平均：P1 的 validation aggregate CE 低於 R_s 經先驗正規化後的 CE。
- 逐 seed：三個 seed 中至少兩個，P1_s 低於 R_s 正規化後的 CE。

#### 4.3 護欄（全部須成立）

- **G1 市場**：P1 的 market-macro CE（三 seed 平均）不高於 R 正規化後的對應值；US、TWSE、TPEX
  各自的 CE 都不得比 R 正規化後高出 0.005 以上。
- **G2 同日鑑別力不得犧牲**：P1 的 `oracle_date_adjusted` CE（以當日實際類別比例取代模型自身
  日平均分布）三 seed 平均，不高於 R 的對應值加 0.002。
- **G3 校準**：P1 在 10–14 日每個天期的 ECE（三 seed 平均）不高於 R21–R23 選中點的紀錄平均：
  10 日 0.0753、11 日 0.0780、12 日 0.0784、13 日 0.0823、14 日 0.0846。

#### 4.4 機制檢查 M1 與 pilot 繼續條件

- **M1**：P1 的 `date_only` CE（只用模型每日平均預測分布）與 `train_prior` CE 的差距（取絕對值、
  三 seed 平均），必須小於 R（未正規化）的同一差距。這是方法確實移除擇時的直接證據。
  M1 不成立時，視為實作未達成預期機制，應先檢查實作，不得調參。
- **Pilot（seed 42）繼續條件**：同時滿足
  - P1_42 低於 R21 正規化後的 CE；
  - 三個市場都沒有比 R21 正規化後高出 0.005 以上；
  - M1 方向正確。

#### 4.5 「可用」等級 U

- 在 C1、G1–G3、M1 都成立的前提下，另外要求 P1 的三 seed 平均 validation CE 不高於 **1.00469**
  （GBDT 經同樣先驗處理）且不高於 **1.00932**（GRU 經同樣先驗處理），market-macro CE 不高於
  **1.00717**。
- 比較對象是經過相同推論期先驗處理的 baseline，所以比較公平；不和未處理的 baseline 比，也不和
  訓練期調整混為一談。

#### 4.6 決策表

| 結果 | 決定 |
| --- | --- |
| S0 不成立 | 不實作 P1，回到 P0 結果討論 |
| Pilot 未達繼續條件 | 停止，記錄為負結果，不補跑其他 seed |
| C1、G1–G3、M1 成立，U 不成立 | 方法機制成立，保留為分類訓練選項，但還不能宣稱優於 baseline |
| C1、G1–G3、M1、U 全部成立 | 成為分類主線候選，才讀取 test 並做逐日配對區間 |
| M1 不成立 | 先查實作，不調參 |
| 任一判準不成立 | 保留結果，同一輪不加入其他改動，以免無法歸因 |

#### 4.7 Test 政策

- Test 只記錄，不用來選 seed、checkpoint 或改判準；test 期間已多次使用，不是全新資料。
- 只在 U 成立時，才以逐日配對區間和 GBDT 比較。
- 描述性報告：每月預測類別比例與 train 先驗的差距（參考線 15 個百分點）、2025-12 的表現、
  14 日 neutral recall。

### 5. 實作範圍（審閱核准後才做）

1. **train 類別計數表**：以和訓練標籤相同的計算方式，逐 symbol 平行掃描完整 train 母體（沿用
   `date_market_index` 的有界 spawn 工作池與快取模式），輸出 `n[d,m,h,k]`，大小約 1–2 MB。快取
   identity 包含 bar-store manifest、sample universe、split、horizons 與門檻。各市場×天期的加總
   必須與 `class-label-audit.json` 逐項相等（4,710,117 筆），不相等就拒絕訓練。
2. **設定**：新增 `model.direction_date_adjustment: none | date_market_loo`（預設 `none`）與
   `model.direction_date_prior_strength`（預設 10）。預設值不得改變任何既有 preset 的 architecture
   digest、baseline identity、resume contract 或唯讀載入；唯讀載入的正規化要擴及 ModelConfig 的新增欄位。
3. **訓練／模型**：訓練步驟依 batch 既有的 `cutoff_at`、`markets` 與 `target_absolute` 查表產生
   offset；`classification_loss` 用 `s + offset`；forward 回傳的 logits 一律是 `s + log p̄`。train 先驗
   以 buffer 存入 checkpoint，並納入 run contract。
4. **Preset 與 registry**：新增 `a_absolute_direction64_dateadj.yaml`，註冊為單模型實驗並沿用 task
   baseline。
5. **測試**：
   - 本機 stdlib：registry 註冊，以及 preset 和原版只差這一個因子。
   - 雲端：留一計算、平滑、offset 不反傳梯度、推論 logits 等於 `s + log p̄`、計數表與 audit 一致、
     預設值不改變既有 digest。
6. **文件**：實驗協定 §13.3（含凍結後的 R_s 參照數值）、README preset 表與指令、roadmap。

GPU 成本：每個 seed 為一次單模型分類訓練，先跑 1 個 seed，通過後再跑 2 個。

### 6. 風險與限制

- **加法 offset 的假設**：offset 假設日期效果讓所有股票的類別 odds 同幅移動。高 beta 股票在
  大漲日更容易被判為上漲，這種「股票敏感度×當日方向」的交互作用無法由加法 offset 表達，會殘留
  部分日期依賴。M1 會揭露殘留程度。
- **放棄擇時**：方法會刻意捨棄模型可能具備的擇時能力。現有證據顯示擇時能力為負，但如果產品
  需要大盤方向判斷，應另建專門模型。
- **推論先驗固定**：推論時使用固定的 train 先驗。新期間的類別比例若明顯偏移（例如 2025-12 上漲
  52%），CE 下限仍會升高。P1 只移除模型自己造成的擇時錯誤，無法修正真實的 regime 變化。
- **14 日 neutral recall**：train 中 14 日 neutral 只有約 10%，以 argmax 決策時 recall 仍可能偏低。
  這屬於決策規則問題，不列為判準。
- **分類範圍**：本設計只處理分類器；alpha 的日期層級成分很小（拆解報告），不在範圍內。

### 7. 待你確認的事項

1. S0 門檻 0.005 是否合適，或改成以 P0 的同日鑑別力差距為門檻。
2. α = 10，以及 G1（0.005）、G2（0.002）的容許度。
3. 只跑 A 組（`a-*`），與 R21–R24 一致；B 組暫不跑。
4. 採單模型分類訓練，不重跑 alpha。

### 8. 參考文獻

- Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021, arXiv 2007.07314：訓練時在
  logits 加上 log 先驗。本方案把它套用在每個日期×市場的先驗上。
- Saerens, Latinne, Decaestecker, *Adjusting the Outputs of a Classifier to New a Priori Probabilities*,
  Neural Computation 14(1), 2002：推論期的先驗位移，即 P0 正規化的理論依據。
- Chamberlain, *Analysis of Covariance with Qualitative Data*, Review of Economic Studies 47(1), 1980：
  以條件概似消去群組固定效果；本方案是其在多類別、大群組下的可行近似。

---

## English

Status: **design draft for review; not implemented and no GPU training authorized.** This document
only defines the method, the experiment and pre-registered validation criteria. Whether it starts
depends on P0 (per-sample export and error decomposition); see section 4.1.

**2026-10-18 update:** P0 is complete and S0 passes: removing timing lowers R21–R23 validation CE by 0.0146 on average, against a 0.005 gate. Reference values are in [error decomposition report §7.6](error_decomposition_audit.md). The section 7 confirmation items still await review.

### 1. Problem

The independent fixed-±1% three-class classifier (`a-absolute-direction64`, R21–R23) only roughly
matches the baselines on validation and loses to a fixed train prior on test. The error
decomposition concentrates the cause at the date level:

| Evidence | Value | Source |
| --- | --- | --- |
| CE removable by knowing the day's class shares (oracle) | validation: train prior 1.03305 → 0.97724 | [error decomposition report](error_decomposition_audit.md) |
| Baseline timing skill | negative for GRU and GBDT | same |
| Learnable part | within-date discrimination: GBDT 1.03305 − 1.00469 = 0.0284; GRU 0.0237 | same (train-prior normalization) |
| Kronos classifier validation CE (loss-selected) | seeds 42/347/5067: 1.02844/1.01311/1.01663 (mean 1.01939) | R21–R23 `metrics.jsonl` |
| Correlation of Kronos daily shortfall against GBDT with the day's up share (test, descriptive) | R21–R24: 0.49/0.40/0.11/0.81 | [roadmap](../MODEL_DEVELOPMENT_ROADMAP.md) P0 row |
| After mixing dates (R24) | stronger timing bias; about 91% "down" predictions at 10–14 days on test | roadmap §5.6 |

Interpretation: stocks on one date and market share the market move, so class shares within a day
swing widely with the market. The model can guess from benchmark inputs whether the whole day will
rise or fall; that timing is fittable during train but does not transfer to a new period. P1
treats the date-level class shares as a known fixed effect during training, so the model only
learns which stocks on the same day are more likely to rise, fall or stay flat.

### 2. Method

#### 2.1 Definitions

- Sample i belongs to date d and market m; horizon h ∈ {1,…,14}; class k ∈ {down, neutral, up}; the
  label keeps the current fixed ±1% definition (`log1p(±0.01)`, unchanged).
- `n[d,m,h,k]`: number of class-k samples on date d, market m, horizon h in the **complete train
  population** (independent of per-epoch sampling); `N[d,m,h]` is their sum over finite labels.
- `p̄[m,h,k]`: the complete-train market×horizon prior, equal to the count shares in
  `class-label-audit.json` (for example US neutral 42.8% at 1 day and 10.4% at 14 days).
- The head output `s(x) ∈ R^{14×3}` is read as within-date relative log-odds; the architecture is
  unchanged.

#### 2.2 Leave-one-out smoothed offset during training

```text
π_{-i}[k] = ( n[d,m,h,k] − 1{y_i,h = k} + α · p̄[m,h,k] ) / ( N[d,m,h] − 1 + α )
z_train   = s(x_i) + log π_{-i}           # constant offset, no gradient
loss      = CE(z_train, y_i)              # masking, equal horizon weights and market weights unchanged
```

- **Leave-one-out** removes the sample's own label, so the offset never contains its answer. Each
  date and market has hundreds to thousands of stocks (validation averages about 3,848 US, 585 TWSE
  and 284 TPEX per day), so the effect is small, but it costs nothing.
- **Smoothing** shrinks toward the market×horizon train prior with strength α = 10 pseudo-samples,
  fixed in advance and never tuned. It matters only on dates with few samples and prevents log(0)
  for an empty class.
- Counts use train labels only; validation and test labels never enter an offset.
- Batch composition does not matter: each sample looks up its own (date, market, horizon) row, so
  no micro-batch is assumed to hold a single date and market.

#### 2.3 Inference and evaluation

```text
z_infer = s(x) + log p̄[m,h]               # fixed train market×horizon prior; no future information
```

- Validation, test, exports and real inference always emit `z_infer`. Validation selection, early
  stopping and the evaluation store stay unchanged; the logits simply include the train prior.
- Expected effect: daily mean predicted distributions stay near the train prior instead of
  following the model's guess about the market. CE should sit near "fixed train prior CE minus the
  model's within-date discrimination". If Kronos discriminates within dates as well as GRU/GBDT
  (0.024–0.028), validation CE would be about 1.005–1.009; this is an estimate, not a criterion.
- A prior estimated from recent matured labels needs a label-end guard and a time-varying
  assumption, and is out of scope.

#### 2.4 Alternatives considered

| Alternative | Why it is not the main design |
| --- | --- |
| Inference-time prior normalization only (being evaluated in P0) | Needs no training and is this design's control, but the model still spends training capacity on timing. P1 must beat it (criterion C1) |
| Shares computed from the micro-batch's own labels | Only 32 samples per micro-batch (about 3 neutral at 14 days), noisy, and includes the sample's own label |
| One learnable bias per date | No parameter exists for a new date at inference; the same idea with a worse estimate |
| Exact conditional likelihood (conditional logit eliminating fixed effects) | Intractable for three classes and hundreds to thousands of samples per day; the leave-one-out offset is its practical approximation |
| Volatility-scaled label bands | Changes the product definition; P2 with separate authorization |
| Shuffling dates (R24) | Already shown to worsen timing |

### 3. Experiment design

- **Single factor:** new preset `a-absolute-direction64-dateadj` adds only the offset of sections
  2.2–2.3 (α = 10). Data group, architecture, sampling (same-date blocks, yearly decay 0.8),
  optimizer, selection metric, seeds and every other numerical setting equal
  `a-absolute-direction64`. It is a new run identity and resumes no old checkpoint.
- **Classifier only:** the alpha model is unchanged, so P1 runs as a single-model experiment
  (`train --experiment a-absolute-direction64-dateadj --seed …`), about half the GPU time of a
  paired run.
- **Seeds and pairing:** reuse 42, 347 and 5067, paired seed by seed with R21/R22/R23.
- **Order:** seed 42 runs first as a pilot (as R24 did); the other two seeds run only when the pilot
  meets the continuation rule in section 4.4.
- **Evaluation:** every P1 checkpoint is exported with `export-signal-predictions` and compared with
  the baselines and R21–R23 in one decomposition through `scripts/analyze_signal_decomposition.py`.

### 4. Pre-registered validation criteria

Everything uses the complete validation set only. Reference R_s is the loss-selected R21–R23
checkpoint with the same seed, measured by the P0 analysis. The reference values are written into
experiment protocol §13.3 and frozen before P1 training starts; no P1 result may change the criteria.

#### 4.1 Start gate S0 (from P0)

- **S0:** in P0, label-free prior normalization lowers R21–R23 validation CE by at least **0.005** on
  the three-seed mean.
- Rationale: normalization gains 0.0092 for GBDT and 0.0045 for GRU. Below 0.005, timing is not
  Kronos' main error source and P1's expected benefit is small.
- If S0 fails, P1 is not implemented and the P0 results are discussed instead.

#### 4.2 Primary criterion C1: training-time adjustment must beat free inference-time adjustment

- Three-seed mean: P1 validation aggregate CE is below R_s after prior normalization.
- Per seed: P1_s is below R_s after normalization for at least two of three seeds.

#### 4.3 Guards (all must hold)

- **G1 markets:** P1 market-macro CE (three-seed mean) is not above the normalized R value, and no
  single market (US, TWSE, TPEX) is more than 0.005 above its normalized R CE.
- **G2 no loss of within-date discrimination:** P1 `oracle_date_adjusted` CE (the day's realized class
  shares replacing the model's own daily mean), three-seed mean, is at most R's value plus 0.002.
- **G3 calibration:** P1 ECE at each of 10–14 days (three-seed mean) is not above the recorded R21–R23
  mean at their selected checkpoints: 0.0753, 0.0780, 0.0784, 0.0823 and 0.0846 for 10–14 days.

#### 4.4 Mechanism check M1 and pilot continuation

- **M1:** the absolute gap between P1 `date_only` CE (the model's daily mean distribution only) and
  `train_prior` CE, three-seed mean, must be smaller than the same gap for unnormalized R. This is
  direct evidence that timing was removed. If M1 fails, the implementation is examined first; no
  tuning.
- **Pilot (seed 42) continues only when all hold:**
  - P1_42 is below normalized R21;
  - no market is more than 0.005 above normalized R21;
  - M1 points the right way.

#### 4.5 "Usable" level U

- With C1, G1–G3 and M1 holding, P1's three-seed mean validation CE must also be at most **1.00469**
  (GBDT with the same prior handling) and at most **1.00932** (GRU with the same handling), with
  market-macro CE at most **1.00717**.
- The comparison is against baselines given identical inference-time prior handling, so it is fair:
  not against unadjusted baselines, and not conflated with training-time adjustment.

#### 4.6 Decision table

| Outcome | Decision |
| --- | --- |
| S0 fails | Do not implement P1; discuss the P0 results |
| Pilot misses the continuation rule | Stop, record a negative result, run no further seeds |
| C1, G1–G3 and M1 hold, U fails | The mechanism works and stays a classifier training option, without a claim of beating the baselines |
| C1, G1–G3, M1 and U all hold | Mainline classifier candidate; only then read test with date-paired intervals |
| M1 fails | Examine the implementation; no tuning |
| Any criterion fails | Keep the result and add no other change in the same round, so effects stay attributable |

#### 4.7 Test policy

- Test is recorded only and never chooses seeds, checkpoints or criteria; it has been used
  repeatedly and is not fresh data.
- Only when U holds is P1 compared with GBDT using date-paired intervals.
- Descriptive reporting: monthly predicted class shares against the train prior (15-point reference
  line), December 2025 behavior, and 14-day neutral recall.

### 5. Implementation scope (only after approval)

1. **Train class-count table:** scan the complete train population per symbol with the same label
   computation as training, reusing the bounded spawn pool and cache pattern of `date_market_index`.
   The output `n[d,m,h,k]` takes about 1–2 MB. The cache identity covers the bar-store manifest,
   sample universe, split, horizons and threshold. Market×horizon sums must equal
   `class-label-audit.json` exactly (4,710,117 samples), or training refuses to start.
2. **Configuration:** add `model.direction_date_adjustment: none | date_market_loo` (default `none`)
   and `model.direction_date_prior_strength` (default 10). Defaults must not change any existing
   preset's architecture digest, baseline identity, resume contract or read-only loading; the
   read-only normalization extends to the new ModelConfig fields.
3. **Training/model:** the training step builds offsets from the batch's existing `cutoff_at`,
   `markets` and `target_absolute`; `classification_loss` uses `s + offset`; forward always returns
   logits `s + log p̄`. The train prior is a checkpoint buffer included in the run contract.
4. **Preset and registry:** add `a_absolute_direction64_dateadj.yaml`, registered as a single-model
   experiment reusing the task baseline.
5. **Tests:**
   - local stdlib: registry registration, and the preset differs from the original by this one
     factor only;
   - cloud: leave-one-out arithmetic, smoothing, no gradient through the offset, inference logits
     equal `s + log p̄`, the count table matches the audit, and defaults leave existing digests
     unchanged.
6. **Docs:** experiment protocol §13.3 (with the frozen R_s reference values), README preset table
   and commands, roadmap.

GPU cost: one single-model classifier run per seed, starting with one seed and adding two more only
after the pilot passes.

### 6. Risks and limits

- **Additive offset assumption:** the offset assumes a date shifts every stock's class odds equally.
  High-beta stocks are more likely labelled "up" on strong up days; such stock-sensitivity×day-
  direction interactions cannot be expressed by an additive offset, so some date dependence
  remains. M1 shows how much.
- **Timing is given up:** the method deliberately gives up any timing skill the model might have.
  Evidence says that skill is negative; if the product needs market-direction calls, that needs a
  dedicated model.
- **Fixed inference prior:** inference uses the fixed train prior. When a new period's class
  balance shifts clearly (for example 52% up in December 2025), the CE floor still rises. P1 removes
  only the model's own timing errors, not real regime change.
- **14-day neutral recall:** train has only about 10% neutral at 14 days, so recall can stay low
  under argmax decisions. That is a decision-rule question and not a criterion.
- **Classifier only:** the design covers only the classifier; the alpha date-level component is
  small (decomposition report) and out of scope.

### 7. Items for your confirmation

1. Whether the S0 threshold of 0.005 is right, or a gate on P0's within-date discrimination gap
   instead.
2. α = 10 and the G1 (0.005) and G2 (0.002) tolerances.
3. Run only group A (`a-*`), consistent with R21–R24; group B is not run for now.
4. Classifier-only single-model training, without rerunning alpha.

### 8. References

- Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021, arXiv 2007.07314: add the log
  prior to logits during training. This design applies it to per-date, per-market priors.
- Saerens, Latinne, Decaestecker, *Adjusting the Outputs of a Classifier to New a Priori
  Probabilities*, Neural Computation 14(1), 2002: inference-time prior shift, the basis of the P0
  normalization.
- Chamberlain, *Analysis of Covariance with Qualitative Data*, Review of Economic Studies 47(1),
  1980: conditional likelihood eliminating group fixed effects; this design is a practical
  approximation for many classes and large groups.
