# 日期層級與同日橫斷面誤差拆解 / Date-Level versus Within-Date Error Decomposition

## 中文

### 1. 目的與範圍

本診斷回答「目前模型訓練不起來的誤差來自哪裡」：alpha loss 與個股三分類 CE 中，有多少屬於同一日期、同一市場所有股票共享的共同成分（日期層級），有多少屬於同日內股票之間的差異（橫斷面）。分析只讀取既有檔案，不訓練、不重抓行情、不重建 baseline，也不使用 GPU。

結論以 validation 為準；test 以相同方式計算，只作記錄。主模型 Kronos 的逐樣本預測在評估後即刪除（寫在 `TemporaryDirectory`），因此本診斷能完整拆解 GRU／GBDT，主模型只能用最終 benchmark 保存的逐日 loss 做輔助分析。

### 2. 資料與方法

| 項目 | 內容 |
| --- | --- |
| 評估集合 | 現行共用評估：validation 528,315 筆、336 個日期×市場群組；test 534,327 筆、316 個群組；test membership `93e2e6a0…` 與 R21–R24 相同 |
| 標籤 | alpha log 報酬；個股絕對 log 報酬依 `log1p(±1%)` 切成跌／平／漲 |
| Baseline 預測 | 現行 baseline cache 的 GRU、GBDT（seed 42）validation／test 預測；分類為 logits |
| 尺度與先驗 | train 擬合的 robust scales `s_h`；train 類別比例取自 `class-label-audit.json`，依市場×天期 |
| 指標口徑 | 與專案相同：Smooth L1（beta=1）作用於 `(預測−標籤)/s_h`，各天期等權平均；CE 讀 logits |

令 `g` 為日期×市場群組，`ȳ_g`、`p̄_g` 為群組內標籤與預測的平均。

- **Alpha 的日期層級上限**：以 `ȳ_g`（已知當日同市場平均 alpha）作為預測的 oracle loss，與 zero-return 比較；另以標準化 MSE 的群組間／群組內分量計算日期層級占比（MSE 的拆解是精確可加的）。
- **Alpha 模型拆解**：只保留模型自己的日期平均 `p̄_g`（date-only），以及「oracle 日期平均＋模型同日差異 `p−p̄_g`」兩種變體；並計算同日內與跨日期的相關係數。
- **分類的日期層級上限**：以群組實際類別比例（Jeffreys 平滑、樣本內）作為 oracle 先驗，與 train 先驗比較。這是樂觀上限，不是可部署模型。
- **分類模型拆解**：
  - date-only：只用模型在該群組的平均預測分布 `q̄_g`。
  - 移除自身擇時：以 Bayes prior shift 把 `q̄_g` 換成 train 先驗，`p' ∝ p × π / q̄_g`。只用同日所有股票的預測與 train 先驗，不用標籤，在推論時可行。
  - oracle 調整：把 `q̄_g` 換成當日實際類別比例，衡量同日內的鑑別力。
- **主模型逐日分析**：R21–R24 的 test 逐日 CE 減去 GBDT，與當日 14 日上漲比例做相關。

正確性核對：重算的 GRU／GBDT validation 與 test 指標，與 baseline 及 benchmark 記錄值的差距都在 5×10⁻⁷ 以內；GBDT test 逐日 CE 的 112 天最大差 3×10⁻¹⁵。

### 3. 結果

#### 3.1 Alpha

| 項目 | Validation | Test |
| --- | ---: | ---: |
| Zero-return loss | 0.46006 | 0.53633 |
| 已知當日同市場平均 alpha 的 oracle loss | 0.44346（−3.61%） | 0.50959（−4.99%） |
| 日期層級占標準化 MSE 的比例 | 2.63% | 3.84% |
| L1 區間（`abs(y/s_h) ≥ 1`）樣本比例，1／5／14 日 | 23.7%／23.2%／24.3% | 29.3%／29.2%／30.8% |
| 上述樣本占 zero-return loss 的比例 | 81.1%／80.8%／80.9% | 84.0%／83.2%／83.5% |

依市場的 oracle loss 降幅：validation 美股 2.2%、上市 15.1%、上櫃 6.1%；test 美股 2.3%、上市 19.5%、上櫃 11.6%。

| Baseline | GRU val | GBDT val | GRU test | GBDT test |
| --- | ---: | ---: | ---: | ---: |
| 相對 zero-return 的總改善 | +0.00483 | +0.00410 | +0.00730 | +0.00449 |
| 只用自身日期平均的改善 | +0.00292 | +0.00341 | +0.00410 | +0.00379 |
| 同日差異在 oracle 之上的改善 | +0.00127 | +0.00073 | +0.00233 | +0.00069 |
| 同日內相關係數 | 0.082 | 0.044 | 0.095 | 0.044 |
| 跨日期相關係數（日期平均） | −0.046 | +0.021 | +0.118 | +0.063 |

對照：R21–R23 的 Kronos 最佳 validation loss 為 0.45311–0.45476，相對 zero-return 改善 0.0053–0.0070，在 validation 上優於 GRU（0.00483）。

#### 3.2 個股三分類

| 項目 | Validation | Test |
| --- | ---: | ---: |
| Train 先驗 CE | 1.03305 | 1.01712 |
| 已知當日類別比例的 oracle CE | 0.97724（−5.40%） | 0.95714（−5.90%） |
| GRU CE／只用自身日期分布／移除自身擇時／oracle 調整 | 1.01377／1.03746／1.00932／0.95295 | 0.99958／1.02185／0.99474／0.93416 |
| GBDT CE／只用自身日期分布／移除自身擇時／oracle 調整 | 1.01393／1.04222／1.00469／0.94848 | 0.99903／1.02605／0.99002／0.92960 |

由此得到：

- **擇時能力（train 先驗 CE − 只用自身日期分布）**：GRU −0.0044／−0.0047，GBDT −0.0092／−0.0089。兩者的日期層級預測都比 train 先驗差。
- **同日鑑別力（oracle CE − oracle 調整）**：GRU +0.0243／+0.0230，GBDT +0.0288／+0.0275。
- **自身擇時的代價（CE − 移除自身擇時）**：GRU +0.0044／+0.0048，GBDT +0.0092／+0.0090。

#### 3.3 主模型 test 逐日分析（只記錄）

| Run | 分類逐日 CE 減 GBDT 平均 | 與當日 14 日上漲比例的相關 | Alpha 逐日 loss 減 GBDT 平均 |
| --- | ---: | ---: | ---: |
| R21 seed 42 | +0.0388 | 0.49 | −0.0031 |
| R22 seed 347 | +0.0325 | 0.40 | +0.0084 |
| R23 seed 5067 | +0.0279 | 0.11 | +0.0044 |
| R24 datemix 42 | +0.1168 | 0.81 | +0.0038 |

逐日平均為日期等權，不等於樣本加權的整體差距。

### 4. 解讀

**已確認（由資料直接算出）：**

1. **Alpha 的日期層級成分很小**：只占標準化 MSE 的 2.6%–3.8%。例外是上市市場，其 oracle 可降低 15%–20% 的 loss；原因可能是加權指數的權值集中（推論，未驗證）。整體而言，alpha 訓練困難不是日期層級成分造成的。先前「主要誤差在日期層級」的推論對 alpha 不成立，在此更正。
2. **Alpha 的 loss 由尾端樣本主導**：約 24%–31% 的樣本落在 Smooth L1 的線性區間，卻貢獻 81%–84% 的 loss。這些樣本的梯度只剩方向（±1/`s_h`），所以 loss 可下降的幅度天生很小：所有模型相對 zero-return 只改善約 1%–1.5%。這是標籤分布與指標的性質，不是計算錯誤；alpha 標籤的算法與股利口徑已核對無誤，梯度有界且連續。
3. **分類的日期層級資訊大，但沒有模型能預測它**：已知當日類別比例可把 CE 降低 5.4%–5.9%，約是 learned baseline 總改善（約 0.019）的三倍。但 GRU／GBDT 的日期層級預測都比 train 先驗差，移除自身擇時後 CE 反而下降。
4. **可學到的部分是同日鑑別力，兩個任務都有**：alpha 同日相關 0.04–0.10；分類在給定正確日期分布時改善 0.023–0.029。這支持「好的 backbone 應該在兩個任務都表現不錯」：共同的可學部分是同日內的股票差異。
5. **主模型在大盤上漲日表現最差**：R21–R24 分類逐日落後 GBDT 的幅度，與當日上漲比例正相關（0.11–0.81），R24 最強。這與長天期偏向預測「跌」一致。

**推論（待驗證）：** Kronos 分類器與 GBDT 的 test CE 差距（0.03–0.04），有相當部分來自日期層級的擇時偏差，而不是同日鑑別力不足。要精確量化，需要主模型的逐樣本 logits。

### 5. 對下一步的影響

1. **最優先，只需 GPU 推論、不需訓練**：讓評估流程保存逐樣本預測，對 R21–R24 的最佳 checkpoint 重跑 validation／test。這樣可以：
   - 用同一方法拆解 Kronos，精確量化它的擇時代價與同日鑑別力。
   - 驗證一個不使用標籤、在推論時可行的後處理，也就是同日橫斷面先驗正規化：把每個日期×市場的平均預測分布換成 train 先驗。它對 GBDT 可把 validation CE 從 1.01393 降到 1.00469，比較時應同時套用於所有模型。
   - 評估多個 seed 集成的效果。
2. **視第 1 項結果而定**：若 Kronos 的擇時代價明顯大於 baseline，再設計訓練時的日期固定效果。分類以當日實際類別比例做 logit adjustment，推論時改用收縮先驗。
3. **降低優先順序**：alpha 的同日去均值訓練。日期層級成分很小，預期改善有限；上市市場是例外，可作次要研究。

### 6. 限制

- Oracle 使用群組自身的標籤，是樂觀上限，不是可部署模型。
- Baseline 只有 seed 42；主模型只有逐日彙總資料，不能做逐樣本拆解。
- Test 期間已多次用於研究比較；本報告的 test 數值不用於選擇方案。
- 本報告的數值由一次性程式產生，輸出保存在本分支 worktree 的 `artifacts/audit/`（不受 Git 追蹤）。同一方法已移入 [scripts/analyze_signal_decomposition.py](../scripts/analyze_signal_decomposition.py)，在相同輸入上重算 48 項指標與逐日值的最大差距為 0；重現步驟見 [README 逐樣本預測匯出與誤差拆解](../README.md#probes-zh)。下載的 14 個 baseline 陣列已依 baseline manifest 的 SHA-256 與大小驗證。

## English

### 1. Purpose and scope

This diagnosis answers where the current models' errors come from: how much of alpha loss and stock-classification CE belongs to a component shared by every stock on the same date and market (date level), and how much belongs to differences between stocks on the same date (cross-sectional). It only reads existing files: no training, market redownload, baseline rebuild or GPU.

Conclusions rely on validation; test is computed identically for the record. Main-model Kronos per-sample predictions are deleted after evaluation (they are written to a `TemporaryDirectory`), so GRU/GBDT are fully decomposed while the main model is analyzed only through the per-date losses saved in the final benchmarks.

### 2. Data and method

| Item | Content |
| --- | --- |
| Evaluation set | Current shared evaluation: validation 528,315 samples in 336 date×market groups; test 534,327 samples in 316 groups; test membership `93e2e6a0…` equals R21–R24 |
| Labels | Alpha log returns; absolute stock log returns split into down/neutral/up at `log1p(±1%)` |
| Baseline predictions | Current baseline-cache GRU and GBDT (seed 42) validation/test predictions; classification stores logits |
| Scales and priors | Train-fitted robust scales `s_h`; train class shares per market×horizon from `class-label-audit.json` |
| Metric definitions | As in the project: Smooth L1 (beta=1) on `(prediction−label)/s_h`, horizons equally weighted; CE reads logits |

Let `g` be a date×market group and `ȳ_g`, `p̄_g` the group means of labels and predictions.

- **Date-level alpha bound:** the oracle loss of predicting `ȳ_g` (the known same-day, same-market mean alpha) versus zero-return, plus the date-level share from the between/within-group parts of normalized MSE (an exact additive decomposition).
- **Alpha model decomposition:** variants keeping only the model's own date mean `p̄_g` (date-only) and "oracle date mean plus the model's within-date part `p−p̄_g`", with within-date and between-date correlations.
- **Date-level classification bound:** realized group class shares (Jeffreys-smoothed, in sample) as an oracle prior versus the train prior. This is an optimistic bound, not a deployable model.
- **Classification model decomposition:**
  - Date-only: the model's own mean predicted distribution `q̄_g` for the group.
  - Own timing removed: a Bayes prior shift replacing `q̄_g` with the train prior, `p' ∝ p × π / q̄_g`. It uses only same-day predictions across stocks and the train prior, never labels, so it is feasible at inference time.
  - Oracle-adjusted: `q̄_g` replaced with the realized shares, measuring within-date discrimination.
- **Main-model daily analysis:** R21–R24 test daily CE minus GBDT, correlated with the day's realized 14-day up share.

Correctness checks: recomputed GRU/GBDT validation and test metrics match the recorded baseline and benchmark values within 5×10⁻⁷; GBDT test daily CE differs by at most 3×10⁻¹⁵ over 112 days.

### 3. Results

#### 3.1 Alpha

| Item | Validation | Test |
| --- | ---: | ---: |
| Zero-return loss | 0.46006 | 0.53633 |
| Oracle loss with the known same-day, same-market mean alpha | 0.44346 (−3.61%) | 0.50959 (−4.99%) |
| Date-level share of normalized MSE | 2.63% | 3.84% |
| Linear-regime (`abs(y/s_h) ≥ 1`) sample share, 1/5/14 days | 23.7%/23.2%/24.3% | 29.3%/29.2%/30.8% |
| Those samples' share of zero-return loss | 81.1%/80.8%/80.9% | 84.0%/83.2%/83.5% |

Oracle loss reduction by market: validation US 2.2%, TWSE 15.1%, TPEX 6.1%; test US 2.3%, TWSE 19.5%, TPEX 11.6%.

| Baseline | GRU val | GBDT val | GRU test | GBDT test |
| --- | ---: | ---: | ---: | ---: |
| Total improvement over zero-return | +0.00483 | +0.00410 | +0.00730 | +0.00449 |
| Improvement from its own date means only | +0.00292 | +0.00341 | +0.00410 | +0.00379 |
| Within-date improvement on top of the oracle | +0.00127 | +0.00073 | +0.00233 | +0.00069 |
| Within-date correlation | 0.082 | 0.044 | 0.095 | 0.044 |
| Between-date correlation (date means) | −0.046 | +0.021 | +0.118 | +0.063 |

For reference, R21–R23 Kronos best validation losses are 0.45311–0.45476, improvements of 0.0053–0.0070 over zero-return, ahead of GRU (0.00483) on validation.

#### 3.2 Stock three-class classification

| Item | Validation | Test |
| --- | ---: | ---: |
| Train-prior CE | 1.03305 | 1.01712 |
| Oracle CE with the known same-day class shares | 0.97724 (−5.40%) | 0.95714 (−5.90%) |
| GRU CE/own date distribution only/own timing removed/oracle-adjusted | 1.01377/1.03746/1.00932/0.95295 | 0.99958/1.02185/0.99474/0.93416 |
| GBDT CE/own date distribution only/own timing removed/oracle-adjusted | 1.01393/1.04222/1.00469/0.94848 | 0.99903/1.02605/0.99002/0.92960 |

Derived quantities:

- **Timing skill (train-prior CE − own date distribution only):** GRU −0.0044/−0.0047, GBDT −0.0092/−0.0089. Both date-level forecasts are worse than the train prior.
- **Within-date discrimination (oracle CE − oracle-adjusted):** GRU +0.0243/+0.0230, GBDT +0.0288/+0.0275.
- **Cost of own timing (CE − own timing removed):** GRU +0.0044/+0.0048, GBDT +0.0092/+0.0090.

#### 3.3 Main-model test daily analysis (recorded only)

| Run | Mean daily classification CE minus GBDT | Correlation with the day's 14-day up share | Mean daily alpha loss minus GBDT |
| --- | ---: | ---: | ---: |
| R21 seed 42 | +0.0388 | 0.49 | −0.0031 |
| R22 seed 347 | +0.0325 | 0.40 | +0.0084 |
| R23 seed 5067 | +0.0279 | 0.11 | +0.0044 |
| R24 datemix 42 | +0.1168 | 0.81 | +0.0038 |

Daily means weight dates equally and need not equal the sample-weighted aggregate gap.

### 4. Interpretation

**Confirmed (computed directly from the data):**

1. **Alpha's date-level component is small:** only 2.6%–3.8% of normalized MSE. TWSE is the exception, where the oracle reduces loss by 15%–20%; concentrated weights in the cap-weighted index may explain this (inference, not verified). Overall, alpha training difficulty is not caused by the date-level component. The earlier inference that errors are mainly date-level does not hold for alpha and is corrected here.
2. **Alpha loss is dominated by tail samples:** about 24%–31% of samples lie in the linear Smooth L1 regime yet contribute 81%–84% of the loss. Their gradients carry only a sign (±1/`s_h`), so attainable loss reductions are inherently small: every model improves on zero-return by only about 1%–1.5%. This is a property of the label distribution and metric, not a computational error; the alpha label construction and dividend treatment were checked, and its gradients are bounded and continuous.
3. **Classification has large date-level information that no model forecasts:** the known same-day class shares reduce CE by 5.4%–5.9%, about three times the learned baselines' total improvement (about 0.019). Yet GRU/GBDT date-level forecasts are worse than the train prior, and removing their own timing lowers CE.
4. **The learnable part is within-date discrimination, present in both tasks:** within-date alpha correlation is 0.04–0.10; classification improves by 0.023–0.029 given the correct date distribution. This supports the view that a good backbone should do well on both tasks: their shared learnable part is the difference between stocks on the same date.
5. **The main model does worst on rising-market days:** R21–R24 daily classification shortfalls against GBDT correlate positively with the day's up share (0.11–0.81), most strongly for R24. This matches the long-horizon bias toward "down".

**Inference (not yet verified):** a substantial part of the 0.03–0.04 test CE gap between the Kronos classifier and GBDT comes from date-level timing bias rather than weak within-date discrimination. Quantifying it exactly requires the main model's per-sample logits.

### 5. Implications for the next step

1. **First priority, GPU inference only, no training:** let evaluation persist per-sample predictions and rerun validation/test for the best R21–R24 checkpoints. This enables:
   - Decomposing Kronos with the same method, quantifying its timing cost and within-date discrimination exactly.
   - Testing a label-free post-processing that is feasible at inference time, same-day cross-sectional prior normalization: replace each date×market's mean predicted distribution with the train prior. It lowers GBDT validation CE from 1.01393 to 1.00469, and any comparison should apply it to every model.
   - Evaluating multi-seed ensembles.
2. **Conditional on item 1:** if Kronos's timing cost clearly exceeds the baselines', design training-time date fixed effects. Classification would use logit adjustment by the day's realized class shares in training and a shrunk prior at inference.
3. **Lower priority:** same-date demeaned alpha training. The date-level component is small and the expected improvement limited; TWSE is the exception and a secondary study.

### 6. Limitations

- The oracles use each group's own labels and are optimistic bounds, not deployable models.
- Baselines exist only for seed 42; the main model has only per-date aggregates, not per-sample decomposition.
- The test period has been used repeatedly in research; test values here select nothing.
- This report's values came from a one-off program whose outputs live in this branch worktree's untracked `artifacts/audit/`. The same method now lives in [scripts/analyze_signal_decomposition.py](../scripts/analyze_signal_decomposition.py); on identical inputs it reproduces all 48 metrics and their daily values with a maximum difference of 0. See [README per-sample prediction export and error decomposition](../README.md#probes-en) for reproduction steps. The 14 downloaded baseline arrays were verified against the baseline manifests' SHA-256 values and sizes.
