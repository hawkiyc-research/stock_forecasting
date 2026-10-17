# 歷次模型效能診斷 / Historical model performance diagnosis

分析日期：2026-10-17。程式碼參照：`d2da8b965bddb1417afe3afc777770b80294ce44`。本報告與小型證據 JSON 存放於 Codex 的獨立分支 `codex/model-performance-20261016`；沒有修改模型、採樣器、前處理、資料切分、依賴或訓練設定。

## 繁體中文

這次重新讀取本機 24 份 `validation-benchmark.json`，並分析已有逐步紀錄的驗證曲線。主要診斷使用 R21–R23 六個獨立 signal64 模型、各自選定 checkpoint 的 `trainer-state.json`，以及 `summary.json`；本次補充另核驗 Claude worktree 的 R24 兩份正式 test 與完整 train 類別計數。沒有執行模型、重建 Python 環境、啟動訓練、同步 RunPod 或操作其他工具的訓練。本報告不包含尚未完成的 Antigravity 結果；日期混洗的早期回報與後來正式 test 分開記錄；第 10 節補充已重新核驗的最終結果。

本次判斷：**目前證據較支持「跨期間／跨市場的泛化不穩定、點估計與排序的選模目標不一致，以及分類機率失準」；還沒有證據足以將根因定為前處理程式錯誤、模型容量不足或訓練時間不足。** 這些是需要分開驗證的問題，不能用一個「過擬合」標籤取代診斷。

**1. 比較範圍與證據限制**

R21–R23 六個模型及其 GRU、GBDT 基準使用完全相同的 test membership：534,327 個股票×日期視窗、112 個 cutoff dates，實際 cutoff 為 2025-12-01 至 2026-05-11，SHA-256 為 `93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`。這些視窗高度重疊，共用 benchmark，不能當成 534,327 個獨立市場情境。GRU／GBDT 等 learned baselines 只有 seed 42，三個主模型 seed 都沿用同一份 baseline，沒有三 seed 的 baseline 不確定性估計。

主 checkout 的 24 份 JSON 包含早期 validation-only 版本、不同 membership 的 sampled/full test、以及清理後共用 test。2026-10-14 的舊 `a-direction64` seed 42 是錯誤 preset 的重複結果，不算新增獨立實驗。舊 quantile normalized pinball、目前 alpha Smooth L1、分類 CE 是不同目標，不能把 loss 數字直接連成「改善／退步」曲線。V1／V2 的方向選模公式也不同；其 alpha 曲線最低點未必是正式選定 checkpoint。V3 沒有本次可重算的完整 final test。

**2. 點估計 loss 最低的 checkpoint，並非排序較好的 checkpoint**

直接核對 R21 alpha seed 42 的 checkpoint metadata 與 `metrics.jsonl`：

| Validation 位置 | Alpha loss，1–14 日平均 ↓ | US 14 日 Rank IC ↑ | TWSE 14 日 Rank IC ↑ |
| --- | ---: | ---: | ---: |
| 0.2 epoch，正式選定 `checkpoint-003680` | 0.454761 | 0.004800 | 0.052634 |
| 1.0 epoch，後續診斷 | 0.455741 | 0.035509 | 0.090852 |
| 1.4 epoch，最後一次 validation | 0.458827 | 0.034605 | 0.087834 |

最低 loss 的選模確實按照既定合約執行；`primary_5d/selection_score` 在目前兩任務代表全 1–14 日平均 loss，並非只看 5 日。問題是這個目標與同日股票排序能力不等價。這個差異出現在 validation 內，不需要讀 test 才能發現。

因此先前「混洗前 US 14 日 validation Rank IC 約 0.035」需註明 checkpoint：它接近後期紀錄，正式選定最佳 alpha checkpoint 的數值是 0.004800。若混洗實驗的 0.007 來自其最佳 loss checkpoint，就不能直接把 0.035→0.007 當成最佳 checkpoint 的排序退步幅度。使用者提供的混洗最佳 loss 0.4636 仍差於原三 seed 的 0.4531–0.4548，未通過既定 alpha loss 門檻的判定不受這項釐清影響。

這支持在未來實驗預先並列「點估計品質」與「排序品質」，而不是事後更換選模公式。Alpha 點估計仍是原任務；本報告沒有擅自將目標改成截面中心化、排名或交易報酬。

**3. 分類學到部分訊號，但機率品質與長天期輸出偏向有問題**

下表為同一 test、全 1–14 日平均；百分比為對應比例乘 100：

| 模型 | CE ↓ | Balanced accuracy ↑ | Macro-F1 ↑ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Signal64 seed 42 | 1.039206 | 36.39% | 0.338556 | 0.098948 |
| Signal64 seed 347 | 1.032702 | 37.67% | 0.355016 | 0.095237 |
| Signal64 seed 5067 | 1.027663 | 38.05% | 0.364459 | 0.081302 |
| GRU seed 42 | 0.999579 | 35.97% | 0.329188 | 0.030020 |
| GBDT seed 42 | 0.999028 | 36.63% | 0.334105 | 0.050451 |

三個主模型的 balanced accuracy 與 macro-F1 都高於這份 GRU 基準，CE／ECE 卻都較差。這符合「部分分類訊號存在，但機率失準」的診斷，無法從 CE 單獨推論完全沒有可辨識訊號。相對 GBDT，seed 42 的 balanced accuracy 較低，另外兩 seed 較高。

US 14 日真實 down／neutral／up 比例為 42.71%／10.27%／47.02%，三 seed 預測為 down 的比例卻達 76.49%–81.45%，neutral 僅約 1.15%–1.30%。US 1 日 neutral 真實比例約 42.4%，14 日約 10.3%，因此固定 ±1% 標籤會隨持有天數產生很不同的類別分布；這是既定標籤的性質，尚非標籤錯算證據。

長天期 neutral 弱勢也存在於基準：GRU 的 US 14 日 neutral 預測比例為 0%，GBDT 約 1.02%。因此不能把 neutral 問題全歸給 Kronos。主模型更突出的問題是機率校準差、長天期 down 偏向及市場差異。

若未來測試校準，temperature scaling 只縮放 logits，不改變 argmax 類別；它即使改善 CE／ECE，也不能單獨修正 down 偏向或 neutral recall。若要檢驗帶收縮的 bias／prior 修正，必須以預先指定的 validation 校準程序估計，不能用本次 test 的真實比例回填。

**4. 分類的 CE 劣勢集中在特定月份，並非整段時間均勻失效**

2025 年 12 月有 106,394 個視窗，占 test 的 19.91%。

| 主模型 seed | 12 月 CE | 同月 GRU CE | 12 月占全 test「相對 GRU CE 淨劣勢」 |
| --- | ---: | ---: | ---: |
| 42 | 1.155546 | 1.023723 | 66.24% |
| 347 | 1.152694 | 1.023723 | 77.53% |
| 5067 | 1.147582 | 1.023723 | 87.82% |

計算先以各月樣本數加權 CE 差，再除以全體 CE 差。這是有正負的淨差分解；其他月份的改善會抵銷劣勢，因此不能解讀為錯誤樣本的占比。三個 seed 在 2026 年 2 月 CE 都比 GRU 好，seed 5067 在 4 月也較好。

這提供明確的後續查驗方向：先在 train／validation 檢查每月各市場的標籤比例、logit bias、機率信心及 feature 分布，區分 regime／prior drift 與模型過度自信。不能由這張表直接斷定 12 月行情原因，也不應為改善既有 test 的 12 月而挑選新超參數。

**5. Alpha 的跨市場遷移與 validation 代表性有問題**

| Alpha seed | US：selected val → test 14 日 Rank IC | TWSE：selected val → test | TPEX：selected val → test |
| --- | ---: | ---: | ---: |
| 42 | +0.0048 → +0.0455 | +0.0526 → −0.0631 | +0.0680 → −0.0832 |
| 347 | +0.0175 → +0.0729 | +0.0799 → −0.0377 | +0.0828 → −0.0905 |
| 5067 | +0.0413 → +0.0831 | +0.0639 → +0.0320 | +0.0780 → −0.0165 |

這不是所有市場一起變差：US 保持正值，TPEX 三 seed 從正轉負，TWSE 兩 seed 轉負。現行 validation 有 81.57% US 視窗，test 有 82.30%，未加市場權重的全體 loss 主要受 US 影響。這支持「共同模型／共同選模不足以保證各市場泛化」的假說，但尚未證明必須拆成三個 backbone，也不支持直接翻轉台股訊號。

Alpha test Smooth L1 分別為 0.528707／0.540340／0.536003，GRU 為 0.529025，zero-alpha 為 0.536328。只有 seed 42 微幅勝 GRU。產物內保存的 seed 42 逐日等權配對差為 −0.000328，14 日期 circular moving-block bootstrap 的 95% 區間為 [−0.004133, +0.003755]，跨過零；不是已建立的穩健優勢，也不是等效性證明。該逐日差不等於樣本加權 aggregate loss 相減。

**6. 前處理仍值得查，但目前沒有抓到可確認的錯誤**

已核對程式碼與保存 metadata 的項目：

- 主模型 lazy reader 預設輸出 raw 的 cutoff-causal adjusted OHLCV；Kronos backbone 才做視窗內標準化。數值尺度分支在 backbone 正規化前取值。未發現主線先套 relative preprocessing、再送 Kronos 重複標準化的證據。
- Alpha 與 absolute direction 的進場位置均為 cutoff 後一個交易日的 open；第 h 天 close 為退出位置，h=1 是同日開到收。Main model 與 baseline 的程式採同一執行時點；分類使用 log(1−0.01)／log(1+0.01) 邊界。程式閱讀未發現簡單的 off-by-one 或把 log return 直接拿去和 ±0.01 比較的錯誤。
- 六個目前模型保存的 14 個 robust scales 完全一致，20 維 scale-feature 統計 SHA-256 也一致；校準紀錄為 train-only、50,000 samples、seed 59。這降低「seed 不同導致標準化漂移」的嫌疑，但相同統計不等於統計方法最適合所有市場。
- 清理規則檢查完整 128+14 個交易日、價格與成交量完整性，流動性門檻只用 cutoff 前 60 日。大型但有限的經濟報酬只記錄，不自動排除。資料 split 使用 label-end boundary guard；保存 metadata 記錄 violations=0。這是 metadata／程式層的證據，不是對原始行情逐筆重算的證明。

仍缺少本機原始 bar-store／corporate actions，無法確認個別復權、分割、下市、價格單位或真實異常行情是否正確。不能把這些可能性排除。

一個值得做定點核驗的群組是輸入視窗內回撤至少 40% 的樣本：98,233 個，占 test 18.38%，卻占三 seed alpha Smooth L1 總和約 35.09%–35.42%。這個回撤只使用已觀察到的輸入 close。它表明困難尾部對 loss 的影響較大，不代表這些資料必然錯誤。優先抽查少量 ticker／cutoff 的原始、復權價格與標籤，保留真實極端報酬；不應先刪掉這群資料來美化指標。

另外，manifest 的舊 preparation counts 為 train 14,007,228／test 1,261,524，不能當成目前實際訓練母體。`summary.json` 的 `sample_universes` 才記錄目前清理後的 train 4,710,117／validation 528,315／test 534,327。未來診斷應同時記錄 source identity 與實際 clean universe。

**7. 「不到一個 epoch 就最佳」不代表訓練時間不足，也不是完整過擬合定論**

Alpha 最佳 validation 點在 0.2／0.6／1.0 epoch，分類在 0.4／0.4／0.6 epoch；繼續訓練後 validation 通常未改善。另一方面，訓練 loss 是近期隨機訓練區段、模型處於 train mode 的值，validation 是完整未加年度採樣權重的母體；兩者不能直接相減當作精確泛化缺口。

實際年度 quota 對 2021–2025 的每個來源視窗抽取倍率約為 0.6375／0.7969／0.9961／1.2451／1.5564。計畫的一個 epoch 有 4,710,117 次抽取，但不計 batch padding，按 quota 最多只有 4,190,958 個不同視窗：最近年份會重複，早年份只看部分。這是既定 recency sampling 行為，並非本次改動；不能把一個 epoch 當成完整母體恰好看過一次。

要確認過擬合程度，應在 eval mode 對固定且有容量上限的 train 診斷樣本和 validation 使用同一 loss 定義，另列市場／月份／天期分組；訓練樣本需同時保留母體分布與採樣分布的說明。這個缺口未補齊前，「更低 train loss」不能單獨作為改善證據。

**8. 歷代容量與複雜化的結果，不支持先加大模型或拉長訓練**

以下僅在清理後共用 G3 test 內比較舊 quantile loss；不能與上述 Smooth L1 比數值：

| 歷史版本 | Test normalized pinball ↓ | 可支持的觀察 |
| --- | ---: | --- |
| LoRA32 | 0.293694 | 受控容量參考 |
| LoRA64 | 0.291608 | 比這份 LoRA32 結果較好 |
| Partial unfreezing | 0.293823 | 未因解凍更多參數勝過 LoRA64 |
| Adaptive64，三 seed | 0.296555–0.296831 | 加強／增加路徑後 raw alpha 未勝 LoRA64 |
| Direction V1，三 seed | 0.294654–0.297410 | 多任務／方向架構沒有帶來 alpha 突破 |
| Direction V2 | 0.293857 | 修正若干路徑與選模問題，仍未勝該 LoRA64 參考 |

這些版本同時改了多項設定，不是所有設計的單因素消融，不能說某個新增模組已被單獨證偽。可得到的務實判斷是：繼續無差別增加參數、分支或 epoch，缺乏目前實驗支持。

歷史固定權重 scale probe 曾在舊 checkpoint 顯示：conditioned readout 對 benchmark 20／60 日歷史波動的 validation R² 約 −0.322／−0.548，個股約 +0.552／+0.605。它支持曾經存在 benchmark 尺度泛化的讀取困難；不證明資訊完全消失，也不證明現在已有 explicit scale branch 的 signal64 仍存在同一問題。現行 latent／scale／benchmark 路徑的容量匹配消融尚缺，不能把主模型表現直接歸因於 Kronos 本身。

**9. 建議的下一步順序**

1. 等目前 Antigravity 結果完成，先按同一 membership、選定 checkpoint、任務 loss、market／horizon／month 分組對齊，並查是否真的改變了預測偏向和機率品質。不能只比較一個 aggregate 數字。
2. 若可取得既有 validation 輸出，先做不更新模型權重的校準／prior 診斷；若輸出沒有保存，後續僅做經授權的雲端串流 forward，輸出有上限的逐日統計，不建立巨大資料集。用 train／validation 分布檢驗假說，test 保留為本次事後診斷。
3. 下一次必要的資料稽核，以少量高損失／高回撤 ticker 和匹配的一般樣本核驗 input、復權、label、calendar、scale clipping；同時補固定 train eval。這應先於全面清理或重抓行情。
4. 再做一次只改一個因子的路徑／backbone 比較：保留使用者指定的舊採樣與動態讀取，Chronos-2 對現行 Kronos 之比較需固定 head、資料母體、label、scales 和選模規則；若要改 head 或 calibration，分開成實驗。對 latent、benchmark、數值尺度路徑做容量匹配對照，才能判斷 backbone 是否提供額外訊號。

以上是分析建議，本次沒有啟動實作或付費運算。使用者已指定的 Chronos-2 舊採樣版本維持原狀。本報告沒有建議恢復日期混洗，也沒有將「同日 batch 的隱性正則化」當作已證實機制：相同 benchmark 並不自動讓共同梯度消失；需要額外受控對照才能驗證該解釋。

**10. 研究交接後的補充核驗：R24、固定 train 先驗與解讀邊界**

本節在初版 `b637335` 之後補入。只讀取 Claude worktree 的保存產物與現有程式；沒有修改該 worktree、匯出新預測或操作 Pod。原有 24 份 benchmark／42 個來源雜湊的證據快照保留，新來源另列於 JSON 的 `follow_up_20261017`。

R24 的兩個任務均有 `state=ready`、`evaluation_split=test` 的正式結果，summary 記錄正常 early stopping，test membership 與 R21–R23 完全相同：

| R24 任務 | 正式 checkpoint | 最佳 validation loss | Test loss | 其他 test 診斷 |
| --- | --- | ---: | ---: | --- |
| Alpha | `checkpoint-007360` | 0.463628 | 0.535295 | 14d Rank IC：US +0.071839、TWSE −0.038456、TPEX −0.073688 |
| Absolute direction | `checkpoint-003680` | 1.027278 | 1.118239 | Brier 0.698024；ECE 0.180672；macro-F1 0.300239；US14d 預測 down 95.29% |

這補齊了原先仍在訓練的方向模型紀錄。R24 alpha 的 validation 未通過原門檻，方向的最終機率品質與偏向也沒有支持恢復混洗。Test 僅作保存結果的事後診斷，不用它重選方案或 checkpoint；「同日 batch 隱性正則化」仍未被證實為機制。

**固定完整 train 機率先驗，比原始主模型 CE 更好。** 從完整清理後 train 的市場×天期類別計數估計 `pi[k]=train_count[k]/sum(train_counts)`，對每個市場／天期的所有股票輸出相同機率，再用現有 test 的 class support 計算 CE 與三類平方誤差和的 Brier。十四個天期等權；test 標籤只用於評分，沒有參與機率估計。重算得到 CE **1.0171197373714438**、Brier **0.6204644412066332**。

該 CE 低於 R21–R23 原始分類器的 1.039206／1.032702／1.027663，仍高於 GRU 0.999579 與 GBDT 0.999028。這個常數機率參考不處理個股訊號；它揭露機率品質的基本缺口，不能據此否定主模型已觀察到的 BA／F1 優勢。來源 audit 的 SHA-256 是 `6552c61714fa6cbd3738ff8ddcb61c7f359971cb58ecd0a962bbd352afadb26a`。它代表完整 train 母體的先驗，不等於年度 recency sampler 的加權先驗；若要用先驗差解釋訓練偏差，還需核對實際採樣權重。

**排序必須與 learned baseline 比，而非只要求正 Rank IC。** 同一 test 上，alpha GRU 的 US14d Rank IC 為 +0.065927，落在三個 Kronos seed 的 +0.045486／+0.072947／+0.083138 之間；GBDT 的 TPEX14d 為 +0.047586，三個主模型均為負值。Alpha seed42 的 aggregate loss 相對 GRU 僅改善約 0.0601%，另兩 seed 分別落後約 2.1387%／1.3190%。這些是保存結果的描述；112 個相依日期與一份 baseline seed，仍不足以宣告新候選的獨立確認性勝出。

**以下拆解主張需要收斂，避免將描述當成根因或可改善上限。** 這些數學反例已用標準函式庫重算；沒有改動其他分支的拆解報告：

- Huber 線性區的梯度有界，不代表 loss 天生只能改善 1%–1.5%。完美預測的 loss 仍為零。尾部貢獻高可支持查驗資料與條件訊號，推不出可學性上限。
- 群組平均值是平方誤差下的最佳常數，未必是 Huber 最佳常數。以 targets `[0,0,100]`、beta=1 為例，預測群組平均 33.333 的平均 Huber loss 為 43.944444；預測常數 0.5 只要 33.083333。MSE 的 between／within 精確可加分解及其 2.6%–3.8% 比例，不能直接當作 Huber 的分量或上限；相關逐筆比例本輪沒有重新計算。
- 對每筆預測套 `p_new ∝ p*pi/mean(p)` 並個別正規化後，群組平均一般不等於 `pi`。原預測 `[.8,.1,.1]`／`[.2,.4,.4]`，目標先驗各 1/3，調整後平均為 `[.388889,.305556,.305556]`。這可以是受控比較的 heuristic，不能稱精確替換日期先驗，也不能自動當成可加的擇時／鑑別分解或純 label-shift 校正。
- Neutral 的 argmax 預測比例很低，不足以單獨證明 neutral 機率未校準。現有 ECE 使用 15 個 bins 的最高類信心；它不是逐類校準指標。後續應一起看 neutral one-vs-rest 辨識力、逐類可靠度、CE／Brier 與混淆矩陣，不能只追 neutral recall。
- `alpha=stock log return−benchmark log return` 不保證股票池的截面平均為零，因股票池與指數權重可能不同。截面去均值或把群組 mean 向零收縮，是額外模型假設；不能當成保留原始 alpha 任務的無損等價變換。

**Chronos-2 實驗的結論範圍也要固定。** 現行 `models/chronos2.py` 將單一樣本的五個 OHLCV 通道編成同組，取歷史 context patch hidden states，排除原生 forecast／placeholder 位置，再接既有 resampler／head。Asset 與 benchmark 各自編碼，再由既有 conditioner 結合。這是預訓練 encoder transfer 的受控比較，沒有測試完整 native forecasting／covariate readout；結果若負面，只能否證這個接法。不同 cutoff 樣本維持獨立群組，可避免跨樣本混入時間上不可見的 context；本次沒有為跨股票 ICL 改動分組。

現有 prediction-export 與 decomposition CLI 位於 Claude 的分支，本 Codex 分支沒有自動移植該功能。未來如需其輸出，應重用既有能力，保持串流、資料量上限與多 worker；本次沒有執行匯出或建立新資料集。

建議仍優先做時間隔離的 validation 校準診斷、固定 train eval 與少量原始標籤稽核，再分軸比較 backbone／head／sampling。GRU／GBDT 應接受相同校準程序。R21 seed42 採 32×8，seed347／5067 採 64×4，雖有效 batch 都是 256，這三次執行並非只差 seed；後續共同多 seed 比較需固定執行條件。所有線上更新需等使用的 horizon 標籤到期，保留 label-end guard；既有 test 不用於估計校準參數、回填先驗或挑選新 head。


**證據與重算方式**

小型證據：[historical_model_diagnosis_20261017.json](historical_model_diagnosis_20261017.json)。它保留 24 個歷史 run 的摘要、六個目前模型的選定 checkpoint、驗證曲線、分市場／月份統計、相同 membership 核對和來源 SHA-256。計算只讀本機 JSON，以兩個有上限的 I/O threads 分批解析，不載入完整行情、視窗資料或模型。

原始檔案位於本機主 checkout 的 `/Users/hawkiyc/Documents/stock_forecasting/artifacts/runpod/`；未拷貝到本 worktree，通常也不會隨 Git clone 交付。目前三組 parent run 是 `run-20261015T064558Z-f04e155d4299416f`、`run-20261015T064559Z-624f212a27894308`、`run-20261015T064600Z-13b8b4376ea5413b`，各有 `-alpha`／`-direction` 子目錄。核驗 checkpoint 要讀各 run 的 `validation-benchmark.json` 之 `checkpoint`，再對應同 checkpoint 的 `trainer-state.json`，不要用 completion 的最後一次 validation 取代選定 checkpoint。

程式路徑：`models/signal.py`（loss／固定標籤）、`signal_evaluation.py`（評估）、`data/dataset.py`、`direction_data.py`、`baseline_signal.py`（標籤／讀取）、`data/sample_universe.py`（清理）、`models/backbones.py`（Kronos normalization）、`date_market_sampler.py`（年度 quota）。以上均在 `src/stock_forecasting/` 下。歷史 scale probe 位於 `artifacts/diagnostics/representation-scales/run-20260905T203327Z-270337978/checkpoint-018145/probe-20260914T101527Z-105a4d66/summary.md`。

## English

Analysis date: 2026-10-17. Code reference: `d2da8b965bddb1417afe3afc777770b80294ce44`. This report and its compact evidence JSON belong to the isolated Codex branch `codex/model-performance-20261016`. No model, sampler, preprocessing, split, dependency or training configuration was changed.

The analysis reread 24 locally saved benchmark JSON files, available validation histories, and the six R21–R23 signal64 models' selected-checkpoint metadata and training summaries. This follow-up additionally verifies the two formal R24 tests in the Claude worktree and full-train class counts. It did not execute a model, inspect or create a Python environment, train, synchronize RunPod or operate another tool's training. Pending Antigravity results are outside scope. Early user-supplied date-mixing findings remain distinct from the newly verified final R24 artifacts in section 10.

**The evidence supports unstable temporal/market generalization, a distinction between point-loss selection and ranking quality, and miscalibrated classification probabilities. It does not yet establish a preprocessing bug, insufficient capacity or insufficient training time as the root cause.** These hypotheses require separate checks.

**1. Comparison contract**

All six R21–R23 models and their saved GRU/GBDT baselines share 534,327 test windows and 112 cutoff dates, from 2025-12-01 through 2026-05-11, with membership SHA-256 `93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`. Overlapping windows and shared benchmarks are dependent observations. Learned baselines contain only seed 42 and are reused for all three candidate seeds; baseline variation across seeds is unknown.

Older artifacts include validation-only evaluations and sampled/full tests with different memberships. The 2026-10-14 old a-direction64 seed-42 result repeats a wrong preset and is not an independent replication. Quantile pinball, point Smooth L1 and classification CE are different objectives. V1/V2 used different direction-selection formulas; their minimum alpha-validation metric need not identify the selected checkpoint. There is no fully recomputable final V3 test in this analysis.

**2. Lowest point loss does not identify the best ranking checkpoint**

For R21 alpha seed 42, metadata confirms selection of `checkpoint-003680` at 0.2 epoch: validation Smooth L1=0.454761, US 14-day Rank IC=0.004800 and TWSE IC=0.052634. At 1.0 epoch, loss=0.455741 while US/TWSE IC=0.035509/0.090852. The final 1.4-epoch evaluation reports loss=0.458827 and US IC=0.034605.

Selection followed its existing contract. For current tasks, the compatibility key `primary_5d/selection_score` represents the full 1–14-horizon mean loss. Point-error minimization and within-date ranking are distinct criteria.

Consequently, the previously quoted historical validation IC≈0.035 must be tied to its later checkpoint. If date-mixing IC≈0.007 is from its best-loss checkpoint, 0.035→0.007 is not a comparison between best-loss checkpoints. The user-supplied mixing best loss 0.4636 still fails the historical 0.4531–0.4548 alpha-loss range; this clarification does not reverse that rejection.

Future experiments should preregister point quality and ranking diagnostics together. This report does not change alpha into a centered, ranking-only or trading-return target.

**3. Classification contains some signal but produces poor probabilities and long-horizon bias**

Across all 14 horizons, candidate seeds 42/347/5067 have CE=1.039206/1.032702/1.027663, balanced accuracy=36.39%/37.67%/38.05%, macro-F1=0.338556/0.355016/0.364459 and ECE=0.098948/0.095237/0.081302. The saved GRU has CE=0.999579, balanced accuracy=35.97%, F1=0.329188 and ECE=0.030020. GBDT has CE=0.999028, balanced accuracy=36.63%, F1=0.334105 and ECE=0.050451.

Every candidate exceeds this GRU's balanced accuracy and F1 while trailing its CE/ECE. That supports partial discrimination with poor probability quality, rather than no learned information. Seed 42 trails GBDT balanced accuracy; the other two exceed it.

Actual US 14-day down/neutral/up shares are 42.71%/10.27%/47.02%. Candidates predict down on 76.49%–81.45% and neutral on approximately 1.15%–1.30%. True US neutral support falls from about 42.4% at one day to 10.3% at 14 days under the fixed ±1% definition. This horizon-dependent imbalance follows the specified labels and is not itself evidence of incorrect labels.

Baselines also struggle with neutral predictions: US 14-day neutral prediction shares are 0% for GRU and about 1.02% for GBDT. Kronos alone cannot explain that failure. Candidates show additional calibration and down-bias problems. Temperature scaling leaves argmax predictions unchanged; improved CE/ECE would not by itself repair class bias or neutral recall. Any shrinkage-based bias/prior correction must be fitted using a preregistered validation calibration procedure, never test label proportions.

**4. Classification deficits are concentrated in time**

December 2025 contains 106,394 windows, or 19.91% of test. Its candidate CE is 1.155546/1.152694/1.147582 versus GRU 1.023723. This month contributes 66.24%/77.53%/87.82% of each candidate's net population-weighted CE deficit relative to GRU. All candidates beat GRU CE in February; seed 5067 also beats it in April.

Each monthly contribution is `(candidate_month_loss − GRU_month_loss) × month_samples / total_samples`. Shares divide contributions by the overall loss difference and are signed; improvements elsewhere offset deficits. They are not fractions of erroneous samples.

This motivates train/validation checks of monthly, market-specific class support, logit bias, confidence and feature distributions. It does not identify a particular December market event or justify tuning parameters to this already examined test month.

**5. Alpha generalization differs across markets**

At the selected checkpoints, US 14-day validation→test IC is +0.0048→+0.0455, +0.0175→+0.0729 and +0.0413→+0.0831. TWSE is +0.0526→−0.0631, +0.0799→−0.0377 and +0.0639→+0.0320. TPEX is +0.0680→−0.0832, +0.0828→−0.0905 and +0.0780→−0.0165.

US remains positive while TPEX changes sign for all three seeds and TWSE for two. US accounts for 81.57% of validation windows and 82.30% of test windows. Population-weighted loss primarily reflects US performance. A common model/selection criterion does not establish per-market robustness; these observations do not yet require separate backbones or justify flipping Taiwan signals.

Alpha test losses are 0.528707/0.540340/0.536003 versus GRU 0.529025 and zero-alpha 0.536328. Only seed 42 narrowly beats GRU. Its saved date-paired difference is −0.000328 with a 14-date circular moving-block bootstrap 95% interval of [−0.004133,+0.003755], crossing zero. This is neither established robust superiority nor an equivalence result. Date-equal paired differences differ from population-weighted aggregate differences.

**6. Preprocessing remains an audit target, without a confirmed bug**

The code reads raw cutoff-causal adjusted OHLCV, applies Kronos window normalization in the backbone, and computes numerical scale features before backbone normalization. There is no identified double-relative-normalization path in the mainline. Both main and baseline labels use the next-session open and horizon-day close; horizon one is that same day's open-to-close return. Direction thresholds are log(1−0.01)/log(1+0.01), with no simple indexing or log/simple-return boundary mismatch found in source review.

All six current models have identical 14 robust scales and identical scale-feature statistics SHA-256, with train-only calibration on 50,000 samples and seed 59. This weakens a seed-dependent scaler-drift explanation, without proving the scaler is optimal across markets. Cleaning checks complete 128+14 sessions and valid prices/volumes; liquidity uses only the previous 60 sessions. Large finite economic returns are diagnostic rather than automatic exclusions. Saved split metadata records an exact label-end guard and zero violations. These are code/metadata findings rather than independent raw-bar verification.

Raw bar-store and corporate actions are not locally available for direct verification of splits, delistings, adjustment factors, price units or real extreme events. Windows with observed input drawdown ≥40% number 98,233, or 18.38% of test, and account for approximately 35.09%–35.42% of candidate alpha Smooth L1 totals. This is a targeted audit group, not proof of corrupted records. Inspect a small matched set of tickers/cutoffs before changing filters; preserve real extreme returns.

Old preparation counts in the manifest are not the effective training population. The current `summary.json/sample_universes` records 4,710,117 train, 528,315 validation and 534,327 test windows; the preparation manifest still describes 14,007,228 train and 1,261,524 test. Record both source and cleaned-universe identities in diagnostics.

**7. Early peaks are not proof that more training is needed**

Best validation points occur at alpha epochs 0.2/0.6/1.0 and classifier epochs 0.4/0.4/0.6. Later validation generally fails to improve. Logged training loss describes recent stochastic windows in train mode, while validation evaluates the full unweighted population. Their difference is not a matched generalization-gap estimate.

Annual sampling draws each 2021–2025 source window approximately 0.6375/0.7969/0.9961/1.2451/1.5564 times per planned epoch. Excluding batch padding, 4,710,117 draws cover at most 4,190,958 distinct windows by these quotas. Recent years repeat while older years are subsampled. This existing recency policy means an epoch is not one exhaustive pass through every window.

A bounded, fixed train diagnostic sample evaluated in eval mode using the same loss and market/month/horizon groups would provide a cleaner overfitting check. Its population and sampling distributions must be recorded. Lower stochastic train loss alone is insufficient evidence of improvement.

**8. Historical capacity and complexity results do not favor indiscriminate expansion**

Within the cleaned common G3 test, old quantile normalized pinball is 0.293694 for LoRA32, 0.291608 for LoRA64, 0.293823 for partial unfreezing, 0.296555–0.296831 for three Adaptive64 seeds, 0.294654–0.297410 for Direction V1, and 0.293857 for V2. Additional unfrozen parameters and more complex designs did not surpass the historical LoRA64 reference. Several settings changed together, so these are not single-factor module ablations and are not comparable numerically to current Smooth L1.

An older fixed-checkpoint scale probe reported conditioned-readout validation R² around −0.322/−0.548 for benchmark 20/60-day historical volatility, versus +0.552/+0.605 for asset volatility. It indicates difficulty reading transferable benchmark scales through that old readout, not total information loss or proof that today's explicit-scale signal model retains the same defect. Capacity-matched latent/scale/benchmark ablations remain missing, preventing attribution of current failures to Kronos alone.

**9. Proposed next steps**

First align completed Antigravity results by membership, selected checkpoint, task loss and market/horizon/month diagnostics. Then examine validation calibration/prior behavior without updating model weights, using saved outputs where available. If outputs are absent, a future authorized cloud forward-only pass can stream bounded daily statistics without materializing a large dataset.

Next, audit a small matched group of difficult and ordinary raw windows and add fixed train evaluation before broad data cleanup. Finally, compare one factor at a time: preserve the user's legacy sampler and dynamic reading for Chronos-2 versus Kronos, holding heads, labels, populations, scales and selection fixed. Treat head/calibration changes as separate experiments. Capacity-matched input-path controls can test whether the backbone supplies incremental signal.

These are recommendations; no new implementation or paid computation was launched. The existing Chronos-2 legacy-sampling implementation remains unchanged. Date mixing is not reintroduced. The same-day-batch implicit-regularization explanation remains a hypothesis; identical benchmarks do not automatically cancel common gradients.

**10. Follow-up verification: R24, a fixed full-train prior and interpretation boundaries**

This section supplements the initial `b637335` report. It reads saved artifacts in the Claude worktree and existing source code without modifying that worktree, exporting predictions or operating a Pod. The initial 24-benchmark/42-source evidence snapshot is preserved; new sources are recorded separately under `follow_up_20261017` in the JSON.

Both R24 child benchmarks are `ready`, use formal `test` evaluation, have the same membership as R21–R23, and record normal early stopping. Alpha selects checkpoint 007360, with best validation loss 0.463628 and test loss 0.535295; test 14-day IC is US +0.071839, TWSE −0.038456, TPEX −0.073688. Direction selects checkpoint 003680, with validation CE 1.027278 and test CE 1.118239, Brier 0.698024, ECE 0.180672 and macro-F1 0.300239. It predicts down for 95.29% of US 14-day samples.

These artifacts complete the previously provisional direction record. They do not support restoring mixing; alpha already failed its validation criterion. Test remains a descriptive record and does not reselect an experiment or checkpoint. The implicit-regularization mechanism of same-day batches remains unproven.

**A deployable fixed full-train probability prior beats raw candidate CE.** Estimate class probabilities separately for each market/horizon using only full cleaned train class counts, then score constant probabilities using saved test class support. Averaging horizons equally yields CE **1.0171197373714438** and multiclass Brier **0.6204644412066332**. Test labels participate only in scoring, not probability estimation. The class audit SHA-256 is `6552c61714fa6cbd3738ff8ddcb61c7f359971cb58ecd0a962bbd352afadb26a`.

This prior beats raw R21–R23 CE 1.039206/1.032702/1.027663 but trails GRU 0.999579 and GBDT 0.999028. It establishes a probability-quality reference without using stock-specific signals; it does not erase the candidates' observed BA/F1 gains. Full-population train priors differ from annual-recency-weighted sampler priors, which must be checked before attributing training bias to prior mismatch.

Positive IC alone does not establish incremental ranking skill: GRU US14d IC is +0.065927, between the three Kronos seeds' +0.045486/+0.072947/+0.083138. GBDT TPEX14d IC is +0.047586, versus negative candidate values. Alpha seed42 improves aggregate GRU loss by only about 0.0601%; seeds347/5067 trail by about 2.1387%/1.3190%. A repeatedly examined test with 112 dependent dates and one baseline seed is descriptive evidence rather than an independent confirmation set.

**Mathematical interpretation needs limits.** The following counterexamples were checked using standard-library arithmetic; another branch's decomposition report was not edited:

- Bounded Huber gradients do not imply an inherent 1%–1.5% improvement ceiling. Perfect predictions still achieve zero loss. High tail contribution motivates data/signal checks, not an unlearnability conclusion.
- Group means minimize squared error, not necessarily Huber loss. For targets `[0,0,100]` and beta=1, predicting the group mean gives mean Huber 43.944444, versus 33.083333 at constant 0.5. Exact MSE between/within decompositions and reported 2.6%–3.8% fractions are not additive Huber components or improvement ceilings. Those per-sample fractions were not recomputed here.
- Per-sample renormalization of `p_new ∝ p*pi/mean(p)` does not generally make the new group mean equal `pi`. For predictions `[.8,.1,.1]` and `[.2,.4,.4]` with uniform target prior, the new mean is `[.388889,.305556,.305556]`. It is a testable heuristic, not exact date-prior replacement, an automatically additive timing/discrimination decomposition, or an unconditional label-shift correction.
- Few neutral argmax predictions alone do not establish miscalibrated neutral probabilities. Existing ECE uses 15 bins of maximum-class confidence, not classwise calibration. Check neutral one-vs-rest discrimination, classwise reliability, CE/Brier and confusion together rather than optimizing recall alone.
- Benchmark-excess alpha need not have zero stock-universe cross-sectional mean because universe and benchmark weights differ. Demeaning or shrinking a group mean toward zero adds an assumption rather than preserving raw alpha by a lossless equivalent transformation.

**Chronos-2 conclusions must match the implemented experiment.** The current adapter groups five OHLCV channels within each sample, retains historical context-patch states, excludes native forecast/placeholder positions, and feeds the existing resampler/head. Asset and benchmark are encoded separately before the existing conditioner. This controlled pretrained-encoder transfer does not evaluate the complete native forecasting/covariate readout. A negative result would reject this connection rather than Chronos-2 generally. Different-cutoff samples remain in separate groups; this follow-up does not enable cross-stock ICL.

The existing prediction-export/decomposition CLI lives on the Claude branch and was not copied into this branch or executed. Future use should reuse that capability with bounded streaming and multiple workers rather than materializing a large dataset.

Priorities remain time-separated validation calibration, fixed train evaluation and small raw-label audits, followed by separate backbone/head/sampling comparisons. Apply the same calibration protocol to GRU/GBDT. Candidate seed42 uses micro-batch32×accumulation8, while seeds347/5067 use64×4; effective batch256 does not make them seed-only replications. Future comparisons should fix execution conditions. Any online update must wait for the relevant horizon label to mature, preserve label-end guards, and avoid test-fitted calibration, priors or head selection.


**Evidence and reproduction**

The adjacent JSON preserves 24 historical summaries, six selected checkpoints and validation curves, subgroup calculations, membership checks and source hashes. Analysis uses two bounded I/O readers for local JSON, without loading models, raw bars or materialized windows.

Source artifacts remain in `/Users/hawkiyc/Documents/stock_forecasting/artifacts/runpod/`, outside this worktree, and are normally Git-ignored. The three current parent IDs end in `f04e155d4299416f`, `624f212a27894308` and `13b8b4376ea5413b`, each with alpha/direction child directories. Resolve the benchmark's `checkpoint` and inspect that checkpoint's `trainer-state.json`; completion metadata describes the last validation and must not replace selection metadata.

Relevant source modules under `src/stock_forecasting/` are `models/signal.py`, `signal_evaluation.py`, `data/dataset.py`, `direction_data.py`, `baseline_signal.py`, `data/sample_universe.py`, `models/backbones.py` and `date_market_sampler.py`. The historical scale probe is under `artifacts/diagnostics/representation-scales/run-20260905T203327Z-270337978/checkpoint-018145/probe-20260914T101527Z-105a4d66/summary.md`.
