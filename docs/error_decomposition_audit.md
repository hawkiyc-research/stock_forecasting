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

**推論（待驗證）：** Kronos 分類器與 GBDT 的 test CE 差距（0.03–0.04），有相當部分來自日期層級的擇時偏差，而不是同日鑑別力不足。要精確量化，需要主模型的逐樣本 logits。（**已於第 7 節以逐樣本預測驗證成立。**）

### 5. 對下一步的影響

1. **最優先，只需 GPU 推論、不需訓練（已完成，結果見第 7 節）**：讓評估流程保存逐樣本預測，對 R21–R24 的最佳 checkpoint 重跑 validation／test。這樣可以：
   - 用同一方法拆解 Kronos，精確量化它的擇時代價與同日鑑別力。
   - 驗證一個不使用標籤、在推論時可行的後處理，也就是同日橫斷面先驗正規化：把每個日期×市場的平均預測分布換成 train 先驗。它對 GBDT 可把 validation CE 從 1.01393 降到 1.00469，比較時應同時套用於所有模型。
   - 評估多個 seed 集成的效果。
2. **視第 1 項結果而定**：若 Kronos 的擇時代價明顯大於 baseline，再設計訓練時的日期固定效果。分類以當日實際類別比例做 logit adjustment，推論時改用收縮先驗。
3. **降低優先順序**：alpha 的同日去均值訓練。日期層級成分很小，預期改善有限；上市市場是例外，可作次要研究。

### 6. 限制

- Oracle 使用群組自身的標籤，是樂觀上限，不是可部署模型。
- Baseline 只有 seed 42；第 1–6 節撰寫時，主模型只有逐日彙總資料，不能做逐樣本拆解（第 7 節已補上）。
- Test 期間已多次用於研究比較；本報告的 test 數值不用於選擇方案。
- 本報告的數值由一次性程式產生，輸出保存在本分支 worktree 的 `artifacts/audit/`（不受 Git 追蹤）。同一方法已移入 [scripts/analyze_signal_decomposition.py](../scripts/analyze_signal_decomposition.py)，在相同輸入上重算 48 項指標與逐日值的最大差距為 0；重現步驟見 [README 逐樣本預測匯出與誤差拆解](../README.md#probes-zh)。下載的 14 個 baseline 陣列已依 baseline manifest 的 SHA-256 與大小驗證。

### 7. P0 後續：Kronos 逐樣本拆解（2026-10-18）

#### 7.1 資料與核對

以 `export-signal-predictions` 在一台 RTX 5090 診斷 Pod 上，對下列 10 個 checkpoint 重跑完整 validation／test 並保存逐樣本預測（約 3.5 小時，每秒約 1,180 筆）：

- R21–R24 的 alpha 與分類器各自依 validation loss 選中的 checkpoint（8 個）。
- 依[實驗協定 §13.2](experiment_protocol.md) 事先固定的市場等權 Rank IC-IR 規則重選的 R21 alpha `checkpoint-022079` 與 R22 alpha `checkpoint-018399`；R23 依此規則與 loss 選中相同。

核對結果：

- **R21–R23**：匯出重算的 validation loss、84 個分市場 Rank IC／IC-IR，以及選中 checkpoint 的 test loss，與訓練紀錄和最終 benchmark 完全相同（差距 0）。
- **R24**：差距很小（validation loss 最多 6×10⁻⁵、Rank IC 9×10⁻⁴、test loss 3×10⁻⁴）。R24 訓練於 RTX 4090、匯出於 RTX 5090，推論差異來自不同 GPU 的數值路徑（未逐項驗證）。
- **本機 Rank IC**：每個匯出模型的分市場 Rank IC／IC-IR 與雲端紀錄逐項相同（每個 split 126 項，最大差距 4×10⁻¹⁶）；GRU／GBDT 與 baseline 紀錄的最大差距 3×10⁻¹⁶。

三 seed 集成的定義：alpha 平均三個點估計；分類平均三個機率分布。方法與第 2 節相同，所有後處理同時套用於 baseline 與 Kronos。

#### 7.2 個股三分類

Validation（train 先驗 CE 1.03305；已知當日類別比例的 oracle CE 0.97724）：

| 模型 | CE | 擇時能力 | 自身擇時的代價 | 移除自身擇時後 CE | 同上，market-macro | 同日鑑別力 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 1.01377 | −0.0044 | +0.0044 | 1.00932 | 1.00865 | +0.0243 |
| GBDT | 1.01393 | −0.0092 | +0.0092 | 1.00469 | 1.00717 | +0.0288 |
| Kronos seed 42 | 1.02844 | −0.0236 | +0.0239 | 1.00456 | 1.00569 | +0.0294 |
| Kronos seed 347 | 1.01311 | −0.0094 | +0.0097 | 1.00337 | 1.00479 | +0.0303 |
| Kronos seed 5067 | 1.01663 | −0.0095 | +0.0101 | 1.00648 | 1.00658 | +0.0274 |
| 三 seed 集成 | 1.01155 | −0.0081 | +0.0083 | **1.00320** | **1.00463** | +0.0306 |
| R24 datemix | 1.02722 | −0.0188 | +0.0192 | 1.00804 | 1.00944 | +0.0261 |

Test（只記錄；train 先驗 CE 1.01712；oracle 0.95714）：

| 模型 | CE | 擇時能力 | 自身擇時的代價 | 移除自身擇時後 CE | 同上，market-macro | 同日鑑別力 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.99958 | −0.0047 | +0.0048 | 0.99474 | 0.99841 | +0.0230 |
| GBDT | 0.99903 | −0.0089 | +0.0090 | 0.99002 | 0.99660 | +0.0275 |
| Kronos seed 42 | 1.03921 | −0.0525 | +0.0530 | 0.98626 | 0.99114 | +0.0319 |
| Kronos seed 347 | 1.03270 | −0.0466 | +0.0470 | 0.98572 | 0.99027 | +0.0323 |
| Kronos seed 5067 | 1.02766 | −0.0412 | +0.0417 | 0.98594 | 0.99115 | +0.0321 |
| 三 seed 集成 | 1.02647 | −0.0413 | +0.0416 | **0.98486** | 0.99008 | +0.0331 |
| R24 datemix | 1.11794 | −0.1270 | +0.1286 | 0.98938 | 0.99372 | +0.0289 |

欄位定義同第 3.2 節：擇時能力 = train 先驗 CE − 只用自身日期分布的 CE；自身擇時的代價 = CE − 移除自身擇時後 CE；同日鑑別力 = oracle CE − oracle 調整後 CE。移除自身擇時後的 validation 分市場 CE：Kronos 三 seed 在 TWSE（1.0196–1.0208）與 TPEX（0.9931–0.9942）都優於 GBDT（1.0220、0.9969）；US（1.0017–1.0053）與 GBDT（1.0026）互有勝負。

#### 7.3 Alpha loss

| 模型 | Val loss | Val：日均改為整段水準 | Val 同日相關 | Test loss | Test：日均改為整段水準 | Test 同日相關 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.45523 | −0.00072 | 0.082 | 0.52903 | +0.00074 | 0.095 |
| GBDT | 0.45596 | −0.00040 | 0.044 | 0.53184 | −0.00018 | 0.044 |
| Kronos 42（loss） | 0.45476 | −0.00017 | 0.099 | 0.52871 | +0.00017 | 0.096 |
| Kronos 347（loss） | 0.45447 | −0.00085 | 0.107 | 0.54034 | −0.01056 | 0.091 |
| Kronos 5067 | 0.45311 | 0.00000 | 0.119 | 0.53600 | −0.01041 | 0.124 |
| 集成（loss 選中） | **0.45214** | −0.00006 | 0.115 | 0.53112 | −0.00590 | 0.112 |
| Kronos 42（IC-IR） | 0.45880 | −0.00092 | 0.119 | 0.54143 | −0.01272 | 0.115 |
| Kronos 347（IC-IR） | 0.45755 | −0.00080 | 0.097 | 0.53947 | −0.01004 | 0.107 |
| 集成（IC-IR 選中） | 0.45333 | −0.00008 | 0.117 | 0.53591 | −0.01049 | 0.122 |
| R24 datemix | 0.46363 | −0.00253 | 0.081 | 0.53532 | −0.00776 | 0.106 |

Zero-return loss：validation 0.46006、test 0.53633。「日均改為整段水準」是把模型每個日期×市場的平均預測，換成它自己在整個 split 的同市場平均（不使用標籤），表中為 loss 的變化量。

#### 7.4 分市場 Rank IC

各市場為 1–14 日平均 Rank IC（`spearman_average_ties_v1`）；IC-IR 為三市場等權（協定 §13.2 的 eqIR）。

| 模型 | Val US | Val TWSE | Val TPEX | Val eqIR | Test US | Test TWSE | Test TPEX | Test eqIR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | +0.0340 | +0.0765 | +0.0702 | 0.412 | +0.0551 | +0.0247 | +0.0176 | 0.200 |
| GBDT | +0.0167 | +0.0464 | +0.0386 | 0.214 | +0.0355 | +0.0001 | +0.0232 | 0.102 |
| Kronos 42（loss） | +0.0171 | +0.0628 | +0.0629 | 0.328 | +0.0378 | −0.0060 | −0.0374 | −0.016 |
| Kronos 347（loss） | +0.0276 | +0.0830 | +0.0806 | 0.432 | +0.0650 | +0.0137 | −0.0309 | 0.077 |
| Kronos 5067 | +0.0467 | +0.0680 | +0.0814 | 0.519 | +0.0783 | +0.0575 | +0.0184 | 0.316 |
| 集成（loss 選中） | +0.0286 | +0.0815 | +0.0812 | 0.436 | +0.0660 | +0.0200 | −0.0209 | 0.110 |
| Kronos 42（IC-IR） | +0.0416 | +0.0822 | +0.0665 | 0.533 | +0.0650 | +0.0364 | −0.0045 | 0.198 |
| Kronos 347（IC-IR） | +0.0213 | +0.0590 | +0.0772 | 0.490 | +0.0693 | +0.0684 | +0.0160 | 0.356 |
| 集成（IC-IR 選中） | +0.0349 | +0.0786 | +0.0873 | 0.590 | +0.0709 | +0.0549 | +0.0099 | 0.277 |
| R24 datemix | +0.0257 | +0.0353 | +0.0549 | 0.275 | +0.0712 | +0.0110 | −0.0151 | 0.106 |

依協定 §13.2，IC-IR 選中組與 loss 選中組的 test 差異只作描述性對照，不作為選模規則優劣的證據。

#### 7.5 解讀

**已確認（由資料直接算出）：**

1. **第 4 節的推論成立**：Kronos 分類器輸給 baseline，原因是日期層級的擇時，而不是同日鑑別力不足。它的擇時能力在 validation 為 −0.009～−0.024，在 test 惡化到 −0.041～−0.053；GBDT 只有 −0.009。R24 datemix 在 test 達 −0.127，失敗幾乎全來自擇時。
2. **Kronos 的同日鑑別力是所有模型中最好的，兩個任務、兩個期間都成立**：分類在 validation 為 +0.027～+0.030（GBDT +0.029、GRU +0.024），在 test 為 +0.032（GBDT +0.028、GRU +0.023）；alpha 的同日相關在 validation 為 0.099–0.119（GRU 0.082、GBDT 0.044），在 test 為 0.091–0.124（GRU 0.095）。
3. **移除自身擇時後，分類器追平或超越 baseline**：這一步不使用標籤，只需同日、同市場所有股票的預測與 train 先驗。Validation CE 三 seed 平均 1.00480，與 GBDT 的 1.00469 相當，market-macro 三個 seed 都優於 GBDT（1.0048–1.0066 對 1.0072）；test 為 0.9857–0.9863，優於 GBDT 同樣處理後的 0.9900，也優於原始 GBDT 0.9990 與固定 train 先驗 1.0171。
4. **三 seed 集成在 loss 類指標最好**：分類移除擇時後 validation 1.00320、test 0.98486；alpha validation loss 0.45214，是所有來源中最低。但 alpha 的 validation US Rank IC（0.029）沒有超過最好的單一模型（0.047）。
5. **Alpha 的 US 排序訊號可以延續，台股不行**：Kronos 的 US Rank IC 由 validation 0.017–0.047 延續到 test 0.038–0.078（GRU 0.034→0.055）。台股在 validation 很高，到了 test 所有模型都崩落：TPEX 多個 Kronos 為負，GRU／GBDT 只剩 0.018／0.023，GBDT 的 TWSE 為 0.000。這是期間或市場環境的性質，不是 Kronos 特有。
6. **Alpha 的日期層級預測在 test 也有代價**：把每日平均預測換成整段水準後，validation loss 變化在 0 到 −0.0025 之間，test 則讓 8 個 Kronos 來源中的 7 個改善 0.006–0.013；loss 選中的 seed 42 是例外（+0.0002）。
7. **P1 的啟動門檻 S0 成立**：移除擇時讓 R21–R23 的 validation CE 分別改善 0.0239／0.0097／0.0101，平均 0.0146，高於門檻 0.005。

**推論（未驗證）：** Kronos 比 baseline 更會「擇時」，可能是因為它透過 gated benchmark conditioner 與尺度分支直接看到大盤狀態；台股的 validation 訊號無法延續，可能反映台股橫斷面因子隨 regime 改變。兩者都需要另外的消融或跨年診斷才能確認。

#### 7.6 對下一步的影響

1. **分類器推論後處理（最高效益、不需 GPU 訓練）**：把「無標籤的同日先驗正規化」做成正式的推論與評估選項，對所有模型一致套用，並在協定中先寫好 validation 驗收判準。它要求同一日期、同一市場的股票一起推論，批次推論本來就是如此。
2. **P1 訓練時的日期固定效果**：S0 已成立，可在設計審閱後實作。依 [P1 設計](p1_date_market_logit_adjustment_design.md)，參照組 R_s 的 validation 數值為：
   - 移除擇時後 CE 1.00456／1.00337／1.00648（平均 1.00480）；
   - market-macro 1.00569／1.00479／1.00658（平均 1.00569）；
   - oracle 調整後 CE 0.94787／0.94694／0.94986（平均 0.94822）；
   - 日期分布 CE 與 train 先驗的差距 0.0236／0.0094／0.0095（平均 0.0142）。
   這些數值會在 P1 開始前寫入協定 §13.3 並凍結。P1 的價值必須超過免費的推論期正規化（設計判準 C1）。
3. **Alpha**：
   - loss 選模在 seed 42 選到 US 排序很弱的點，支持日後正式改用 Rank IC 類指標選模（需程式變更與新 preset）；
   - 由於台股 validation 訊號無法延續，市場等權分數會被台股高估，正式改選模前應先做各年、各市場的 train Rank IC 診斷；
   - 「日均改為整段水準」在 validation 幾乎沒有改善，只能列為次要候選。
4. **不建議**：再跑 datemix、加大容量、延長訓練。這些都不處理擇時問題。

#### 7.7 限制

- Test 已多次使用；本節 test 數值只作描述，不用於選擇方案。
- Baseline 只有 seed 42；Kronos 為三個 seed，集成只用固定的三個 checkpoint。
- IC-IR 選中組本身就是依 validation 挑出，其 validation 數值偏樂觀。
- 同日先驗正規化需要同一日期、同一市場的預測一起取得，不適用於單一股票的即時查詢。
- 輸出保存在本分支 worktree 的 `artifacts/diagnostics/signal-predictions/`（逐樣本預測，約 1.1 GB）與 `artifacts/diagnostics/signal-decomposition/`（`results-p0.json`、`rank-ic-p0.json`），不受 Git 追蹤；重現步驟見 [README 逐樣本預測匯出與誤差拆解](../README.md#probes-zh)。

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

**Inference (not yet verified):** a substantial part of the 0.03–0.04 test CE gap between the Kronos classifier and GBDT comes from date-level timing bias rather than weak within-date discrimination. Quantifying it exactly requires the main model's per-sample logits. (**Verified with per-sample predictions in section 7.**)

### 5. Implications for the next step

1. **First priority, GPU inference only, no training (done; results in section 7):** let evaluation persist per-sample predictions and rerun validation/test for the best R21–R24 checkpoints. This enables:
   - Decomposing Kronos with the same method, quantifying its timing cost and within-date discrimination exactly.
   - Testing a label-free post-processing that is feasible at inference time, same-day cross-sectional prior normalization: replace each date×market's mean predicted distribution with the train prior. It lowers GBDT validation CE from 1.01393 to 1.00469, and any comparison should apply it to every model.
   - Evaluating multi-seed ensembles.
2. **Conditional on item 1:** if Kronos's timing cost clearly exceeds the baselines', design training-time date fixed effects. Classification would use logit adjustment by the day's realized class shares in training and a shrunk prior at inference.
3. **Lower priority:** same-date demeaned alpha training. The date-level component is small and the expected improvement limited; TWSE is the exception and a secondary study.

### 6. Limitations

- The oracles use each group's own labels and are optimistic bounds, not deployable models.
- Baselines exist only for seed 42; when sections 1–6 were written, the main model had only per-date aggregates, not per-sample decomposition (section 7 adds it).
- The test period has been used repeatedly in research; test values here select nothing.
- This report's values came from a one-off program whose outputs live in this branch worktree's untracked `artifacts/audit/`. The same method now lives in [scripts/analyze_signal_decomposition.py](../scripts/analyze_signal_decomposition.py); on identical inputs it reproduces all 48 metrics and their daily values with a maximum difference of 0. See [README per-sample prediction export and error decomposition](../README.md#probes-en) for reproduction steps. The 14 downloaded baseline arrays were verified against the baseline manifests' SHA-256 values and sizes.

### 7. P0 follow-up: Kronos per-sample decomposition (2026-10-18)

#### 7.1 Data and checks

`export-signal-predictions` reran complete validation/test and saved per-sample predictions for 10 checkpoints on one RTX 5090 diagnostic Pod (about 3.5 hours, about 1,180 samples per second):

- the loss-selected alpha and classifier checkpoints of R21–R24 (8);
- R21 alpha `checkpoint-022079` and R22 alpha `checkpoint-018399`, re-selected by the market-equal Rank IC-IR rule pre-registered in [experiment protocol §13.2](experiment_protocol.md); for R23 that rule picks the loss-selected checkpoint.

Checks:

- **R21–R23:** the exported validation loss, the 84 per-market Rank IC/IC-IR values and the selected checkpoints' test loss equal the training records and final benchmarks exactly (difference 0).
- **R24:** small differences (validation loss at most 6×10⁻⁵, Rank IC 9×10⁻⁴, test loss 3×10⁻⁴). R24 trained on an RTX 4090 and was exported on an RTX 5090; the differences are inferred to come from the two GPUs' numerical paths (not verified item by item).
- **Local Rank IC:** every exported model's per-market Rank IC/IC-IR matches the cloud record item by item (126 values per split, at most 4×10⁻¹⁶ apart); GRU/GBDT match the baseline records within 3×10⁻¹⁶.

Three-seed ensembles average the three point estimates for alpha and the three probability distributions for classification. Methods follow section 2, and every post-processing applies to baselines and Kronos alike.

#### 7.2 Stock three-class classification

Validation (train-prior CE 1.03305; oracle CE with the known same-day class shares 0.97724):

| Model | CE | Timing skill | Cost of own timing | CE with own timing removed | Same, market-macro | Within-date discrimination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 1.01377 | −0.0044 | +0.0044 | 1.00932 | 1.00865 | +0.0243 |
| GBDT | 1.01393 | −0.0092 | +0.0092 | 1.00469 | 1.00717 | +0.0288 |
| Kronos seed 42 | 1.02844 | −0.0236 | +0.0239 | 1.00456 | 1.00569 | +0.0294 |
| Kronos seed 347 | 1.01311 | −0.0094 | +0.0097 | 1.00337 | 1.00479 | +0.0303 |
| Kronos seed 5067 | 1.01663 | −0.0095 | +0.0101 | 1.00648 | 1.00658 | +0.0274 |
| Three-seed ensemble | 1.01155 | −0.0081 | +0.0083 | **1.00320** | **1.00463** | +0.0306 |
| R24 datemix | 1.02722 | −0.0188 | +0.0192 | 1.00804 | 1.00944 | +0.0261 |

Test (recorded only; train-prior CE 1.01712; oracle 0.95714):

| Model | CE | Timing skill | Cost of own timing | CE with own timing removed | Same, market-macro | Within-date discrimination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.99958 | −0.0047 | +0.0048 | 0.99474 | 0.99841 | +0.0230 |
| GBDT | 0.99903 | −0.0089 | +0.0090 | 0.99002 | 0.99660 | +0.0275 |
| Kronos seed 42 | 1.03921 | −0.0525 | +0.0530 | 0.98626 | 0.99114 | +0.0319 |
| Kronos seed 347 | 1.03270 | −0.0466 | +0.0470 | 0.98572 | 0.99027 | +0.0323 |
| Kronos seed 5067 | 1.02766 | −0.0412 | +0.0417 | 0.98594 | 0.99115 | +0.0321 |
| Three-seed ensemble | 1.02647 | −0.0413 | +0.0416 | **0.98486** | 0.99008 | +0.0331 |
| R24 datemix | 1.11794 | −0.1270 | +0.1286 | 0.98938 | 0.99372 | +0.0289 |

Columns follow section 3.2: timing skill = train-prior CE − CE of the own date distribution only; cost of own timing = CE − CE with own timing removed; within-date discrimination = oracle CE − oracle-adjusted CE. With own timing removed, validation per-market CE for all three Kronos seeds beats GBDT in TWSE (1.0196–1.0208 against 1.0220) and TPEX (0.9931–0.9942 against 0.9969); US (1.0017–1.0053 against 1.0026) is mixed.

#### 7.3 Alpha loss

| Model | Val loss | Val: daily mean set to split level | Val within-date correlation | Test loss | Test: daily mean set to split level | Test within-date correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.45523 | −0.00072 | 0.082 | 0.52903 | +0.00074 | 0.095 |
| GBDT | 0.45596 | −0.00040 | 0.044 | 0.53184 | −0.00018 | 0.044 |
| Kronos 42 (loss) | 0.45476 | −0.00017 | 0.099 | 0.52871 | +0.00017 | 0.096 |
| Kronos 347 (loss) | 0.45447 | −0.00085 | 0.107 | 0.54034 | −0.01056 | 0.091 |
| Kronos 5067 | 0.45311 | 0.00000 | 0.119 | 0.53600 | −0.01041 | 0.124 |
| Ensemble (loss-selected) | **0.45214** | −0.00006 | 0.115 | 0.53112 | −0.00590 | 0.112 |
| Kronos 42 (IC-IR) | 0.45880 | −0.00092 | 0.119 | 0.54143 | −0.01272 | 0.115 |
| Kronos 347 (IC-IR) | 0.45755 | −0.00080 | 0.097 | 0.53947 | −0.01004 | 0.107 |
| Ensemble (IC-IR-selected) | 0.45333 | −0.00008 | 0.117 | 0.53591 | −0.01049 | 0.122 |
| R24 datemix | 0.46363 | −0.00253 | 0.081 | 0.53532 | −0.00776 | 0.106 |

Zero-return loss: validation 0.46006, test 0.53633. "Daily mean set to split level" replaces the model's mean prediction for each date×market with its own split-wide mean for that market (no labels); the table shows the resulting change in loss.

#### 7.4 Per-market Rank IC

Each market shows the 1–14 day mean Rank IC (`spearman_average_ties_v1`); IC-IR is the equal-market eqIR of protocol §13.2.

| Model | Val US | Val TWSE | Val TPEX | Val eqIR | Test US | Test TWSE | Test TPEX | Test eqIR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | +0.0340 | +0.0765 | +0.0702 | 0.412 | +0.0551 | +0.0247 | +0.0176 | 0.200 |
| GBDT | +0.0167 | +0.0464 | +0.0386 | 0.214 | +0.0355 | +0.0001 | +0.0232 | 0.102 |
| Kronos 42 (loss) | +0.0171 | +0.0628 | +0.0629 | 0.328 | +0.0378 | −0.0060 | −0.0374 | −0.016 |
| Kronos 347 (loss) | +0.0276 | +0.0830 | +0.0806 | 0.432 | +0.0650 | +0.0137 | −0.0309 | 0.077 |
| Kronos 5067 | +0.0467 | +0.0680 | +0.0814 | 0.519 | +0.0783 | +0.0575 | +0.0184 | 0.316 |
| Ensemble (loss-selected) | +0.0286 | +0.0815 | +0.0812 | 0.436 | +0.0660 | +0.0200 | −0.0209 | 0.110 |
| Kronos 42 (IC-IR) | +0.0416 | +0.0822 | +0.0665 | 0.533 | +0.0650 | +0.0364 | −0.0045 | 0.198 |
| Kronos 347 (IC-IR) | +0.0213 | +0.0590 | +0.0772 | 0.490 | +0.0693 | +0.0684 | +0.0160 | 0.356 |
| Ensemble (IC-IR-selected) | +0.0349 | +0.0786 | +0.0873 | 0.590 | +0.0709 | +0.0549 | +0.0099 | 0.277 |
| R24 datemix | +0.0257 | +0.0353 | +0.0549 | 0.275 | +0.0712 | +0.0110 | −0.0151 | 0.106 |

Per protocol §13.2, test differences between the IC-IR-selected and loss-selected groups are descriptive only and are not evidence about which selection rule is better.

#### 7.5 Interpretation

**Confirmed (computed directly from the data):**

1. **The section 4 inference holds:** the Kronos classifier trails the baselines because of date-level timing, not weak within-date discrimination. Its timing skill is −0.009 to −0.024 on validation and worsens to −0.041 to −0.053 on test, against −0.009 for GBDT. R24 datemix reaches −0.127 on test; its failure is almost entirely timing.
2. **Kronos has the best within-date discrimination of all models, in both tasks and both periods:** classification +0.027 to +0.030 on validation (GBDT +0.029, GRU +0.024) and +0.032 on test (GBDT +0.028, GRU +0.023); alpha within-date correlation 0.099–0.119 on validation (GRU 0.082, GBDT 0.044) and 0.091–0.124 on test (GRU 0.095).
3. **With own timing removed, the classifier matches or beats the baselines:** this step uses no labels, only the predictions for every stock of the same date and market plus the train prior. Validation CE averages 1.00480 over the three seeds, matching GBDT's 1.00469, and market-macro CE beats GBDT for all three seeds (1.0048–1.0066 against 1.0072). Test CE is 0.9857–0.9863, better than GBDT's 0.9900 with the same treatment, its raw 0.9990 and the fixed train prior's 1.0171.
4. **Three-seed ensembles are best on loss-type metrics:** classification 1.00320 on validation and 0.98486 on test with timing removed; alpha validation loss 0.45214, the lowest of all sources. Yet the alpha ensemble's validation US Rank IC (0.029) does not exceed the best single model (0.047).
5. **The alpha US ranking signal carries forward; Taiwan's does not:** Kronos US Rank IC moves from 0.017–0.047 on validation to 0.038–0.078 on test (GRU 0.034→0.055). Taiwan is strong on validation but collapses on test for every model: several Kronos TPEX values turn negative, GRU/GBDT fall to 0.018/0.023, and GBDT TWSE is 0.000. This is a property of the period or market, not specific to Kronos.
6. **Alpha's date-level predictions also cost on test:** replacing daily mean predictions with the split level changes validation loss by 0 to −0.0025, while on test 7 of 8 Kronos sources improve by 0.006–0.013; the loss-selected seed 42 is the exception (+0.0002).
7. **P1 start gate S0 passes:** removing timing lowers R21–R23 validation CE by 0.0239/0.0097/0.0101, a mean of 0.0146, above the 0.005 gate.

**Inference (not verified):** Kronos may time the market more than the baselines because the gated benchmark conditioner and scale branch expose the market state directly; Taiwan's validation signal may fail to carry forward because its cross-sectional factors change with the regime. Both need separate ablations or year-by-year diagnostics.

#### 7.6 Implications for the next step

1. **Classifier inference post-processing (highest value, no GPU training):** make label-free same-day prior normalization a formal inference and evaluation option, applied identically to every model, with validation acceptance criteria written into the protocol first. It requires predicting each date and market's stocks together, which batch inference already does.
2. **P1 training-time date fixed effects:** S0 passes, so implementation can follow design review. Per the [P1 design](p1_date_market_logit_adjustment_design.md), the reference R_s validation values are:
   - CE with timing removed 1.00456/1.00337/1.00648 (mean 1.00480);
   - market-macro 1.00569/1.00479/1.00658 (mean 1.00569);
   - oracle-adjusted CE 0.94787/0.94694/0.94986 (mean 0.94822);
   - gap between date-distribution CE and the train prior 0.0236/0.0094/0.0095 (mean 0.0142).
   They are written into protocol §13.3 and frozen before P1 starts. P1 must beat the free inference-time normalization (design criterion C1).
3. **Alpha:**
   - loss-based selection picked a checkpoint with weak US ranking for seed 42, which supports a later formal move to Rank IC-type selection (code change and new preset);
   - because Taiwan's validation signal does not carry forward, equal-market scores overstate it, so a year-by-year, per-market train Rank IC diagnosis should precede any formal selection change;
   - setting daily means to the split level barely changes validation and is only a secondary candidate.
4. **Not recommended:** more datemix runs, more capacity or longer training; none addresses timing.

#### 7.7 Limitations

- The test period has been used repeatedly; test values here are descriptive and select nothing.
- Baselines exist only for seed 42; Kronos has three seeds, and ensembles use three fixed checkpoints.
- The IC-IR-selected group was chosen on validation, so its validation values are optimistic.
- Same-day prior normalization needs the predictions of one date and market together and does not apply to a single-stock real-time query.
- Outputs live in this branch worktree's untracked `artifacts/diagnostics/signal-predictions/` (per-sample predictions, about 1.1 GB) and `artifacts/diagnostics/signal-decomposition/` (`results-p0.json`, `rank-ic-p0.json`); see [README per-sample prediction export and error decomposition](../README.md#probes-en) for reproduction.
