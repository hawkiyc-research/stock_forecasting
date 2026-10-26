# Kronos Alpha 與方向模型實驗協定 / Kronos Alpha and Direction Experiment Protocol

## 中文

### 1. 研究問題

主要問題：在嚴格 point-in-time、next-open execution 與 chronological evaluation
下，以金融 K-line 預訓練的 Kronos-base 經 LoRA 微調後，能否比簡單規則、傳統 ML
與小型 causal DL baseline 更準確地預測個別股票／ETF 從可設定的第 1、2 或 3 個
持有交易日起至固定第 14 日、
相對市場 benchmark 的 alpha，以及個股絕對方向？`a-signal64`／`b-signal64`
每個組別與 seed 以一個實驗流程、單一 Pod 依序訓練 `alpha_point` 與
`absolute_direction`，分別測試單任務穩健回歸與固定報酬帶三分類；既有分位數與
成本敏感雙方向實驗保留作對照，不將其結果視為同一任務。

本協定不把 validation Sharpe 或單次回測視為「可交易 alpha」的充分證據，也不把模型
輸出直接當交易規則。技術指標 baseline 保留為對照；保留的 `direction64` V1／V2／V3
多任務 preset 另將十個有定義的歷史技術通道輸入 Kronos predictor，不用 baseline
規則取代 backbone。最新兩個獨立任務不啟用這組技術通道。

### 2. 固定假設與範圍

- 商品層級預測，不是投資組合預測。
- 日線 OHLCV only。
- 美國與台灣股票、ETF；期貨與選擇權不在本 PoC。
- 收盤 `t` 後產生訊號，下一市場交易日 raw open 進場，不跳過缺成交日。
- 進場日算第 1 個持有交易日。
- `h_start` 可設定為 1、2 或 3；horizons 是 `h_start` 到固定第 14 日。
- alpha label 為 benchmark-relative adjusted execution log return；獨立絕對方向則依個股 adjusted simple return 的固定 ±1% band 分類。
- CAPM abnormal return 只作未來 diagnostic/ablation。
- 單任務點估計用 normalized Smooth L1，獨立方向用 CE，不共用 LoRA 更新或混合任務 loss。舊 quantile 用 pinball，ranking head 用同日排序 loss；`direction64` V1／V2 保留成本方向 softmax CE，V3 使用條件有序三分類 NLL，三者均保留原五級分位數後處理。

### 3. 模型假說

單任務假說：在相同 Kronos-base／LoRA-64、資料及尺度路徑下，將穩健 alpha 點估計與
個股漲／平／跌分開訓練，能否得到可重現、超越任務對齊 baseline 的 holdout 改善？
每個任務有自己的 encoder 微調權重、head、optimizer 與 checkpoint；不預設分開必然較好。
兩個模型從相同 pretrained revision 各自初始化，沒有共享可訓練參數或合併 loss；
「一次實驗」指同一父 run 管理兩個子模型，不是兩個需要分別手動啟動的實驗。
同 Pod 依序執行使兩個完整模型不需同時占用 GPU 記憶體；多個 seed 可在不同 Pod 並行。
±1% 是初始候選，不宣稱最佳。完整 train 的市場／持有期類別稽核必須先通過；門檻不得由 test 挑選。

既有多任務假說：共享金融預訓練 encoder、歷史 benchmark conditioning 與可讀出的歷史尺度
能在相同資料及切分下提升成本敏感的個股絕對／alpha 方向辨識，且不以失控的分位數
誤差換取分類表面增益。V3 用人口機率品質、macro-F1 與決策錯誤護欄選模；保留 V2
等市場／任務／horizon 的 balanced-decision BA 選模作對照，
同時保留人口分布 raw accuracy、majority 基準、up precision／recall 及 normalized pinball。
原有分位數、correlation、ranking 與校準假說仍作對照；不以單一 horizon 的多數類別
accuracy 或提高 coverage 宣稱方向能力，也不預設本版會達到 60%。

必要反證條件：若 zero-return、past-only momentum、GBDT、GRU、DLinear 或 PatchTST
在同一 validation protocol 下持續優於完整模型，則不能以 foundation-model 名義宣稱
有模型增益。

### 4. Dataset identity

每次 run 必須綁定：

- dataset profile 與 `selected_datasets`
- raw Parquet、bar-store manifest、symbol index 與 cutoff ranges 的 SHA-256、size、row count
- provider、market、symbol、asset type 與 date range
- benchmark mapping SHA-256
- h_start-independent bar-store preparation spec 與 data-pipeline digest
- training selection 的 `h_start`、固定 `max_horizon=14` 與完整 horizon 序列
- train-only runtime calibration 樣本數、seed 與 checkpoint 內的 robust scales
- chronological split counts 與 cutoff exclusion audit

正式比較不得混用不同 profile、universe、資料版本或 benchmark mapping。EODHD 與
官方台股資料的來源差異必須在報告中揭露。

### 5. 標籤與因果性

對 cutoff `t` 與 horizon `h`：

```text
entry = next market session's raw regular-session open; never postpone a missing entry
exit  = h-th shared holding day's raw close
alpha_h = log(asset adjusted gross return) - log(benchmark adjusted gross return)
```

商品與 benchmark 使用相同 entry/exit timestamps。模型 input 只包含兩者截至
`cutoff_at` 的 point-in-time adjusted OHLCV；未來 entry、exit、benchmark return、
asset return、CAPM diagnostic 與任何 future label 都不得出現在 input tensor。

測試至少覆蓋：

1. 修改 cutoff 後 benchmark 只改 label、不改兩個 contexts。
2. 對 adjusted close 乘共同常數不改 input 或 label。
3. split 後 raw price 跳空被 point-in-time adjustment 消除，volume 方向正確。
4. context 的最大 timestamp 不超過 cutoff。
5. benchmark calendar gap fail closed。

### 6. Split 與 Stage 設計

從既有壓縮 bar-store 建立連續市場交易日與最低流動性檢查的 compact runtime ranges。
正式 train < 2025-06-01、validation [2025-06-01, 2025-12-01)、test
[2025-12-01, 2026-06-01)；最晚 label 日期嚴格早於該 split 右界。A/B 共用同一
validation/test 來源，完整循序評估。詳細清理門檻及年度衰減以 README 與
`configs/data_cleaning.json` 為準。Runtime 清理索引與主模型 DataLoader 不預先展開完整
windows／labels；baseline 另以有界批次建立磁碟 input／label cache。兩者都不刪除真實極端報酬。

| 項目 | Stage 1 | Stage 2 |
|---|---|---|
| 目的 | 驗證完整腳本、資料、GPU、loss、checkpoint 與 validation | 完整 train split 的 PoC 結果 |
| train 樣本 | 動態年度加權取樣；每 epoch 呈現有效數的 5%，上限 500,000 | 動態年度加權取樣；每 epoch 呈現次數等於有效 train 數 |
| epoch 上限 | 2 | 5 |
| validation cadence | 每個 epoch 的 20%／40%／60%／80%／100% | 同左 |
| early stopping | 設定的 validation monitor 連續 5 次未改善，且完成最低 LR 的兩個訓練間隔；第 1 epoch 起生效。獨立 point／direction 分別用全 horizon Smooth L1／CE；原分位數用 normalized pinball；V1 用固定加權 pinball＋雙 CE；V2 用等市場 BA 與 pinball 護欄；V3 用機率／決策品質與同護欄（第 7、9 節） | 同左 |
| retention | validation 最佳 5 個完整 checkpoints，加上 completion result | 同左 |
| validation/test | 完整有效 windows，不抽樣 | 同左 |
| 模型架構 | Kronos-base + LoRA + shared resampler + conditioner + alpha head | 完全相同 |
| 初始化 | 原始 pretrained base | 原始 pretrained base |
| 接續 Stage 1 checkpoint | 否 | 否 |

上表模型架構列適用於標準分位數 Stage 1／2；獨立任務的 head 與驗收另見第 7、12 節。
Stage 1 不是「最早 5% 時間」，也不是縮短 validation/test。兩個標準預設 config 的
`config.model_architecture_digest()` 必須相同；此 digest 包含 `h_start`／輸出維度，
最後不足一個 training batch 的 target set 會從同一集合開頭確定性補齊，且將補齊數量
寫入 training summary。Stage 2 重新從相同 pretrained revision 開始，以免把 Stage 1
script smoke 當成額外訓練資料。

### 7. Objective

#### 7.1 當前主線：兩個獨立任務

成對入口 `a-signal64`／`b-signal64` 內的兩個子模型，各自以下列一個 objective 訓練與選模：

```text
alpha_point: mean(SmoothL1((prediction - alpha) / train_scale[h], beta=1.0))
absolute_direction: mean(CrossEntropy(logits[h], fixed_return_class[h]))
fixed_return_class: down if return < -0.01; up if return > +0.01; otherwise neutral
```

方向 return 是 next-open 至 horizon close 的個股 adjusted simple return，非 alpha，
不扣交易成本；等於任一邊界屬 neutral。程式在 log-return label 空間用 `log1p(-0.01)`
與 `log1p(+0.01)` 作完全相同的判定。兩個任務都對有效樣本及持有期等權平均，
不加入分位數、排序、舊雙方向／表徵擾動 loss，也不擬合區間校準。
`alpha_point` 是穩健 location，不冒充 q50 或保證為條件期望。
完整 validation 對應 loss 決定 checkpoint／early stop；監控鍵仍為
`primary_5d/selection_score`、mode=`min`，分數涵蓋全部 horizons，不只第 5 日。
Training 明列 `train/normalized_smooth_l1` 或 `train/cross_entropy`，另存
`train/loss`、`validation/loss`；跨任務不得直接排名 selection score。

兩者保留 paired Kronos／LoRA、resampler、benchmark conditioning、歷史尺度與
benchmark 直接數值路徑、market embedding 及 horizon embedding。Head 對 latent
tokens 做 mean pooling，再依 horizon 讀取；不沿用 V3 的有序方向 head、horizon-query
attention、市場狀態 gating、train priors 或 PCGrad。Point 的輸出乘固定 train-fitted
robust scale；`explicit_output_scale` 在此保留擴充尺度特徵，不以每筆 volatility
放大預測位置。兩個任務都不產生區間寬度或交易規則。

#### 7.2 保留對照：分位數與 V1／V2 多任務

以下為保留的 `quantiles` 實驗目標。其主 objective 是 q10/q50/q90 pinball loss。每個 horizon 以 train-only robust scale
正規化後等權平均；training 另加獨立 ranking head 的同日排序 loss（預設權重 0.05），
不直接把 q50 當排序 score。原有實驗的 validation selection 使用未校準 normalized pinball：

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
selection_score = mean_h(normalized_pinball_h), h=h_start,...,14
```

為保留既有 RunPod checkpoint monitor 路徑，aggregate score 同時寫在
`primary_5d/selection_score`；它仍涵蓋全部 horizons，不是只有 5 日。checkpoint
selection mode 固定為 `min`。

`direction64` 以原 LoRA64 設定為基礎，加成本帶三分類、horizon query、十技術通道與
表徵擾動 regularization。台股 1,000 股、美股直接券商 50 股；成本只依 cutoff 已知價格
估計。完整公式、文獻與費率來源見 [README 模型架構](../README.md#model-zh)。
V1 先用最多 1,024 筆 train-only windows，經未訓練模型 forward 量測
`sQ,sAbs,sAlpha`，固定後訓練；即使樣本 calibration seed 固定，數值仍可能受到
model initialization seed 影響：
`pinball + (sQ/sAbs)*CE_abs + (sQ/sAlpha)*CE_alpha + 0.05*ranking + 0.05*sQ*adversarial`。
各 horizon 等權。Checkpoint monitor 改為 `direction/selection_loss`，只含前三項，
在完整 validation 上計算，不含擾動或 ranking regularizer。舊 pinball 監控欄位語意不變；
reference 存 checkpoint，resume 不重新估計。已完成的雙方向 baseline 保留其獨立目標；
v2 不更換 label、資料規則或成本情境，不需因此重訓 baseline 或重做 CPU prepare。

`direction64` 原版保留以供重現；`a-direction64-v2`／`b-direction64-v2` 使用獨立的 `a_direction64_v2.yaml`／
`b_direction64_v2.yaml`。方向 head 直接讀取既有 20 維 train-only 標準化尺度特徵，
另加入成本門檻除以個股／相對 20 日波動 × `sqrt(horizon)` 的四個有界輸入；比值
限於 `[-20,20]` 後作 `asinh`，不改精確 label 門檻或裁切真實報酬。
歷史訊號 `f` 與 train prior `π` 分開，人口 logits 為 `z=f+τ*log(π)`，預設 `τ=1`。
自然類別比例的 `CE(z)` 接受固定 train 市場權重，不另乘 inverse-class 權重。
市場權重依年度加權 train population 比例 `p_m` 用 capped water filling 計算：
`w_m=min(k/p_m, market_weight_max)`，求解 `k` 使 `sum(p_m*w_m)=1`。
因此 train 加權平均為 1，最終有效權重不超過 `market_weight_max`（預設 3），
不是先截斷再正規化。
人口分布機率為 `softmax(z)`；`argmax(f)` 的 `balanced_decision` 另列，不能替換
原始 accuracy 或當成交易勝率。方法動機見
[Menon et al.](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)。
推論方向 schema 2 以 `probability_source="population_logits"` 標示機率來源；
`balanced_decisions.absolute`／`.alpha` 為各持有期的類別索引（`0=down`、
`1=neutral`、`2=up`），來源 `history_residual_logits_without_train_prior`，不是機率。

最多 8,192 筆 train windows 以固定 calibration seed 59 作市場／年份分層抽樣，
年度衰減加權統計還原 training population，**不執行模型 forward**。
Reference 固定為 train 零中心 scaled null quantiles 的 pinball、`log(3)`、`log(3)`；
不因 training seed 或未訓練模型的初始化改變任務係數。先驗、市場權重、reference
及抽樣 provenance 均保存並於 resume 還原。各 horizon loss 仍等權。

[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
僅針對共享可訓練 LoRA 的三任務梯度，在實際 dot product 為負時投影；非 LoRA
參數和表徵正則化保留普通 total-loss gradient。無衝突時不改梯度；`monitor` 只量測。
正式訓練與 GPU batch probe 的 clipping 均設定 `error_if_nonfinite=True`，梯度管理不掩蓋
非有限值；偵測到非有限 gradient norm 時會在更新前報錯，不執行該步 `optimizer.step()`。
每 100 個 optimizer steps 彙總 norms／cosines／負衝突及 nonfinite 計數。
精確訓練步數位置記於 `gradient/runtime_optimizer_step`；tracking canonical step
可能只對應最近的 logging 邊界，不是當次 GPU 更新的確切步數或時間戳。
三任務額外 buffer 的保守峰值 `28 × LoRA numel` bytes 不可超過 `512 MiB`，
batch autotune 也使用正式 backward 路徑，不以普通 backward 低估 AMP／accumulation 成本。
文獻動機不是本資料已改善方向率或已存在梯度衝突的證據。

v2 的 `direction/selection_loss` 越低越好，定義為：

```text
selection_loss = 1 - mean_market(mean_task(mean_horizon(balanced_decision_BA)))
                 + max(pinball/first_full_validation_anchor - 1 - 0.01, 0)*1
```

BA 平均切片中有真實支持的類別 recall；support 逐項揭露。
第一輪**完整** validation 的 pinball 作凍結 anchor，容許相對退化 1%，超過後係數 1
加罰分。Anchor 只建立一次、隨 checkpoint 保存及 resume 還原；test 不參與 fitting。
Anchor 是各 run 獨立的首輪 validation 基準，不能以 `selection_loss` 跨 seed 排名；
跨 seed 應比較同一 holdout 的 raw pinball，以及分列的 raw／balanced direction 指標。
主要設定、範圍與 CLI 選擇見 [README 訓練目標](../README.md#training-zh)。

A/B 的 robust scales 各自由其 training labels 估計，數值可能不同；normalized
pinball 與 normalized Smooth L1 適合各 run 內的選模，不能只憑其跨組大小判定 A/B
優劣。Point 跨組比較共同 holdout 的 raw MAE、RMSE、correlation、符號一致率與
逐市場 RankIC；分位數對照另比較 raw pinball 與區間品質。固定標籤的獨立分類
比較同一 holdout 的 CE、accuracy、BA、macro-F1 及類別支持，不用 coverage。

### 8. Comparator suite

Baseline 依任務獨立保存，但使用單一 `baseline --experiment a-signal64`（B 組為
`b-signal64`）流程，先在本機核對兩個 task cache；皆完成直接跳過，不租用 Pod。
若尚未完成，單一 Pod 依序建置點估計與同 ±1% 標籤的純三分類 baseline，略過已完成部分。
`readiness --baseline --experiment a-signal64` 同時查詢兩者的資料篩選規則及完成結果。
成對入口不改 active selection；個別任務的進階入口仍保留，見 README 附錄。
不同任務結果不得混用；同資料與任務的 training seeds 共用一次
完成結果，不因主模型 LoRA 容量或 seed 重訓 baseline。新 baseline 不重做 CPU prepare。
點估計神經 baseline 使用 normalized Smooth L1；tree 訓練使用 absolute-error（MAE），
仍以完整 validation normalized Smooth L1 選模，此 training／selection objective
差異須在報告揭露。分類神經模型用 CE、tree 用 log loss；規則提供 hard class，
其機率分數只採明列數值 floor 的確定性預測，不代表經學習或校準機率。
完整 validation/test 仍使用共同有效樣本。

Baseline 神經模型每個未提前停止的 epoch 以隨機 permutation 遍歷完整 train 一次，
GBDT 也以完整 train 擬合，沒有年度 `sample_weight`。主模型 preset 使用
`yearly_sampling_decay=0.8` 的動態取樣；因此相同資料全集不代表相同年度訓練權重。
神經 baseline 使用 validation-plateau 與低 LR early-stop 條件；GBDT 用完整 validation、
patience 與最佳模型，但沒有神經 optimizer 的低 LR 排程；規則不執行梯度訓練。
報告不得將這些比較稱為所有訓練策略或計算預算完全相同。

任務、標籤／loss 參數及 baseline 數值實作屬於 baseline 身分；主模型 LoRA 容量與
training seed 不屬於。各任務保存獨立權重或規則參數、最佳 validation 與完整 test
結果；神經 resume 保留 optimizer／scheduler／RNG／sample cursor，GBDT 保留
warm-start 模型與 horizon／iteration cursor。舊分位數／成本方向結果不滿足新任務
的完成檢查。主模型 final test 只重用已核對任務、樣本集合及 scales 的 baseline
結果，不重訓或重新推論 baseline。

以下 comparator 描述適用於既有分位數／成本方向實驗；獨立任務保留相同模型家族，
但採上面的任務對齊輸出與 loss，不建立假分位數：

所有 baseline 只能讀相同 cutoff-inclusive asset/benchmark contexts，並用相同有效
train universe 與 validation split；年度訓練權重差異如上：

- constant distributions：always-buy、zero-return
- rules：relative momentum/reversal、MA crossover、RSI、MACD、volatility-scaled
- traditional ML：per-horizon quantile GBDT
- causal DL：paired GRU、DLinear、compact PatchTST
- full model：Kronos-base LoRA + dynamic benchmark conditioner

規則 baseline 的 residual quantiles、GBDT／neural 的參數擬合與 robust scales 只使用
train labels；validation 用來選 model/checkpoint 與決定 early stopping。
Baseline 的神經模型與 GBDT 另訓練個股絕對方向／alpha 方向三分類，成本、horizons
與完整評估集合須和方向主模型一致；固定規則只提供 hard classes，不捏造 CE。
它們以各自固定 train reference 加權的 pinball＋雙 CE 選模，保留原分位數指標。
跨架構比較使用各任務 accuracy、balanced accuracy、macro-F1、up precision／recall
與訊號比例，不用 reference 不同的加權 selection loss 排名。
主模型的凍結區間校準在 checkpoint 選定後才使用 validation 擬合；test 不參與其調參。
`adaptive64`／`adaptive128` 另提供市場 residual、數值／benchmark ranking 路徑、
market-weighted training pinball、sample-clock 衰減與較強 dropout／weight decay；
原六份容量設定保留。細節與預設值見 [README](../README.md#training-zh)。

### 9. Validation metrics

主要 selection metric：

- 獨立 `alpha_point`：完整 validation normalized Smooth L1；獨立 `absolute_direction`：完整 validation CE，皆對全部 horizons 等權且越低越好。點估計另報 MAE、RMSE、Pearson、符號一致率與有效 date-level RankIC；分類另報 accuracy、BA、macro-F1、類別分布與 confusion，不以 coverage 描述分類品質。
- 原有實驗：all-horizon mean normalized pinball；`direction64`：固定 reference 加權的 pinball＋雙 CE；`direction64-v2`：等市場方向 BA 與凍結 pinball 護欄，皆越低越好，不跨定義比較 selection loss。
- V3：population CE／Brier 對 train prior 的比率與 macro-F1，加相反方向／塌縮罰分及凍結 pinball 護欄；完整公式見第 13 節。

`direction64` 另報每個 horizon／市場／月份的兩個三分類 confusion matrix、accuracy、
balanced accuracy、macro-F1、majority accuracy、up precision／recall／訊號比例。
分類 precision 不等同回測獲利，三分類 accuracy 也不是舊 q50 二元方向一致率。
v2 另列 `balanced_decision` 的 confusion／accuracy／BA／macro-F1，而原 raw accuracy
與人口分布機率保持原義；不能改名或覆蓋。機率診斷包含 train-prior CE skill、20 個
confidence bins 與聯合 up 的門檻／precision／count；逐日 confusion 聚合硬上限
4,096 entries，超限報錯而非抽樣。這些機率未被假定為已校準。

既有分位數任務的每個 `h_start`–14 日 horizon 另報告：

- raw pinball 與 normalized pinball
- q50 MAE 與 normalized MAE
- q50 Pearson correlation
- q50 sign agreement
- q10–q90 interval coverage、width 與 coverage error

上述分位數對照另保留以下研究 diagnostics：

- date-level cross-sectional RankIC/IR
- equal-weight long/short net log return、Sharpe、max drawdown、turnover
- 五級 postprocess signal distribution
- market、asset type、provider、year slices

獨立 point 只從 cross-sectional 結果發布逐市場／horizon 的 RankIC／IR，不發布
long-short return、Sharpe、turnover 或五級訊號；獨立分類不發布報酬排序。
兩個獨立任務都有市場、月份等 subgroup 指標，並以 input context 的歷史回撤
20%／40% 作診斷切片；這不是刪除樣本的門檻。它們都沒有分位數 coverage 或 width。

cross-sectional diagnostics 不是投資組合模型輸出，也不是主要 checkpoint metric。
transaction-cost 假設必須明列；單一 Sharpe 不可解讀為已證明可交易。
新的 RankIC 使用 `spearman_average_ties_v1`；ties 取平均名次，constant prediction／target
日期不列有效 correlation。報告須同時讀 `rank_ic_status`、`rank_ic_evaluated_dates` 與
`rank_ic_ir_status`，不可把沒有有效日期的數值占位值當作結果；歷史無方法標記的數字未自動重算。

### 10. Test policy

正式 RunPod 訓練以完整 validation 選定最佳 checkpoint；訓練完成後，由同一 Pod
自動執行完整 test benchmark。若啟用區間校準，先固定該 checkpoint，僅用完整
validation 擬合凍結校準係數，再套用於 test；raw 與 calibrated 指標均保留。
保留的 adaptive 分位數整合實驗額外列出 `calibrated_static` 對照和 `calibrated_online` 逐日診斷：
online 只使用依既有交易日曆已於前一交易日或更早到期的報酬更新上下尾寬度，
不改 q50、不回寫凍結校準，也不調模型；不能把它冒充完全凍結的 test。
Baseline 的 test 指標直接讀取已完成的快取，不在這個階段重新訓練或推論。
同一 run 內所有模型的 test membership 與報酬尺度一致性檢查通過後，才發布
`test_unlocked=true`；test 不決定 checkpoint、early stopping、凍結係數或 online 超參數。
若使用 test 結果修改模型，該集合已成為研究迭代用的評估資料；後續獨立泛化主張
必須以未參與這些決策的新時間區間驗證，不能僅更換 dataset version 就宣稱獨立。

### 11. Checkpoint 與重現性

Production Stage 1/2 不允許 `max_steps` 或固定 step cadence。optimizer budget 由
target set、batch size、gradient accumulation 與 epoch 數推導，每個 epoch 固定做
5 次 validation。每個可接受 checkpoint 必須：

1. 由完整 validation 選出：獨立 point／direction 與原分位數實驗看各任務定義的 `primary_5d/selection_score`；保留的 `direction64` V1／V2／V3 看其版本定義的 `direction/selection_loss`。
2. 保存 trainable parameter union、optimizer、scheduler、RNG state 與 resolved config。
3. 綁定 run ID、dataset artifacts、model architecture、Kronos source/model/tokenizer
   revisions 與 bounded training-source digest。
4. 通過 artifact SHA-256/size 與 strict trainable-key restore。
5. 分位數／保留的多任務使用 `model_output_schema_version=5.0`；獨立任務使用 `signal-1.0` 並明列 `prediction_task`。不接受不相符的 task／schema／architecture，不能跨任務或方向版本 resume。

v2 另保存 direction schema 2 的 train priors、固定 loss references、市場權重、
首輪完整 validation anchor 及 PCGrad 亂數／步數狀態。新方向 head 必須以新 run
初始化，不能將原版 `direction64` 權重接續成 v2；舊 preset 仍可原樣重現。

Leaderboard 最多保留 validation 最佳 5 個完整可續傳 checkpoints。正常完成或
early stopping 都另外保存唯一的 `completion-result/`；它包含最後可訓練權重、
resolved config、停止原因、實際步數／樣本數及 validation metrics，但不重複 optimizer
與 scheduler state。

報告成功狀態必須以 terminal lifecycle、run manifest、best-checkpoint pointer、trainer
state 與 raw validation JSON 交叉確認，不能只看 W&B chart 或 README。

### 12. 最低驗收

Stage 1：

- 年度加權動態取樣，呈現次數為有效 train 數的 5%（上限 500,000）；seed／epoch
  決定可重現序列，最多 2 epochs。Early stopping 從第 1 epoch 起啟用，但仍須
  同時滿足連續 5 次未改善及最低 LR 下兩個訓練間隔的條件。
- 完成 remote ruff/pytest、forward/backward、validation-ranked checkpoint save/reload、
  inference schema smoke 與自動 lifecycle termination。
- 輸出 `[B,15-h_start,3]` ordered alpha quantiles，`h_start ∈ {1,2,3}`；
  沒有自然語言／fact 生成；標準 preset 無 classifier，方向 preset 另驗收雙三分類 heads。

Stage 2 與獨立任務：

- 標準 Stage 2 使用與標準 Stage 1 相同的 architecture digest；獨立 point／direction preset 各有自己的 task 與架構，不能互相接續。均使用完整 train population 的動態取樣及相同 pretrained revisions。
- 獨立點估計驗收 `[B,15-h_start]` alpha point；獨立方向驗收 `[B,15-h_start,3]` logits／機率與固定報酬帶，不要求分位數／ranking／成本分類輸出。
- 完成所有 baseline 與完整模型的同協定 validation。
- raw JSON 有 task-specific per-horizon、aggregate 與 subgroup metrics；point／quantile 另有有效 cross-sectional 指標，classifier 不捏造報酬排序。
- 資料 provenance 可追溯，且 training code 無 provider/API import。

研究主張：

- Stage 1 只證明腳本可運作。
- 單一 seed Stage 2 只算 PoC。
- 要主張模型優於 baseline，至少需多 seed、報告 dispersion，並完成預先定義的 ablation。

### 13. 優先 ablation

當前主線以 `a-signal64`／`b-signal64` 成對實驗測試兩個獨立模型，在同資料組別
使用至少三個 training seeds（42、43、44），對照各自已完成的 task-aligned baseline。
先比較同任務共同 holdout 的指標與 seed dispersion，再研究容量或其他單因子；
不可把 point 與分類 selection score 互相比大小。CLI 使用
`train --experiment a-signal64 --seed 42 --seed 43 --seed 44`
（B 組改用 `b-signal64`），建立三台 Pod、三個父 run、六個獨立模型；也可每個 seed
分別下命令以選擇不同 GPU。每台 runtime 是兩模型共用的整體上限，不是各自重新計時。
父 `status` 只有在兩個正確任務都完成訓練與完整 test 後才標記 COMPLETE；`resume` 略過已完成
模型、補齊未完成訓練或 test，`download` 一次取得兩份子模型結果而不複製權重。完整命令見
[README 訓練操作](../README.md#train-zh)。

保留的歷史多任務比較是同資料組別的 `direction64-v2` 與 `direction64-v3`，亦需
多 seed；V1 `direction64` 作重現對照。這是整套路徑／prior／梯度／選模方法比較，
不是單一元件的因果效果，也不是目前兩個 single-task head 的結構說明。
原 `adaptive64`／`adaptive128` 的容量對照仍保留：其餘設定固定，只改 LoRA rank／alpha；
與 `lora64` 比較也是整套方法比較。`partial` 保留為次要研究線，不取代 Kronos／LoRA。

V3 的 `a_direction64_v3.yaml`／`b_direction64_v3.yaml` 使用 `market_ordinal_v3`：
下界條件機率 `a`、上界條件機率 `b` 組成 `[1-a,a*(1-b),a*b]`；以三分類 NLL 訓練。
其等價梯度是所有樣本的下界 BCE，加上僅 non-down 樣本的上界條件 BCE，
不是兩個分開平均後等權相加的 BCE。此保留版本
保留成本、absolute／alpha、alpha 分位數、排序、LoRA、PCGrad 與 train-only references。
這是 [CORN](https://arxiv.org/abs/2111.08851) 的順序結構，不是承諾金融方向率。
借鑑 [MASTER](https://ojs.aaai.org/index.php/AAAI/article/view/27767) 的市場條件特徵選擇，
只用既有 benchmark 七個歷史狀態特徵，對三條 token stream 與 20 個數值特徵作有界 gating。
不改清洗、baseline 數值契約、DataLoader；相同資料／成本的 baseline 可重用。
V3 selection loss 等市場／任務／horizon 平均 `(CE/priorCE+Brier/priorBrier+1-population_macro_F1)/3`，
加相反方向誤判與塌縮罰分（各預設 0.25）及既有凍結 pinball 護欄。
塌縮指真實占比至少 5%、預測占比不足 1%；門檻及權重可設定，不移除樣本。
BA、population 與 balanced-decision accuracy、per-class precision／recall、confidence bins
均保留，另報 ordinal MAE、相反方向錯誤、Brier skill、塌縮類別與各選模分項。
Test 不擬合任何先驗或門檻；V2/V3 不跨架構 resume，selection loss 不跨 run 排名。
參數及完整公式見 [README 歷史 V3 訓練目標](../README.md#legacy-objectives-zh)。

後續可在不改 output contract 與資料切分下，另作以下單因子消融：

1. gated benchmark conditioner vs asset-only（benchmark gate 固定為 0）。
2. Kronos LoRA vs frozen Kronos + trainable downstream modules。
3. Kronos vs compact PatchTST/DLinear，在相同 histories、horizons 與任務 loss 下比較，並揭露年度抽樣及 optimizer 差異；GBDT point 的 MAE 訓練不算相同 loss 對照。
4. raw price-index benchmark input vs total-return-adjusted benchmark input。
5. CAPM abnormal-return diagnostic，只作分析，不取代主要 label。

一次只改一個因子，並產生新的 experiment/run identity；不得用同一 checkpoint 名稱
覆寫不同語意。

#### 13.1 訓練順序單因子對照：`*-signal64-datemix`

歷史 `DateMarketSampler` 依（日期, 市場）排序後輸出 128 筆連續區塊；主模型每個
micro-batch 來自同一日期與市場，256 筆有效 batch 約只涵蓋 2 個日期×市場。同日股票共享
市場與風格衝擊，因此每次更新都由少數日期的共同結果主導；baseline 神經模型的區塊則是單一
股票的連續日期，兩者的 batch 組成並不相同。

假說：日期集中的 batch 讓模型追逐日期層級的共同衝擊，表現為早期最佳點、seed 敏感、
長天期類別偏移與逐日 loss 波動。`a-signal64-datemix`／`b-signal64-datemix` 只加入
`training.date_mixing_buffer_samples: 16384` 的有界局部混洗，其餘數值設定與對應
`*-signal64` 相同：同一 epoch 的樣本集合與年度配額不變、模型架構 digest 不變、task
baselines 重用。這是新的 run identity，不接續舊 checkpoint。

```bash
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64-datemix
bash scripts/runpod_workflow.sh train --experiment a-signal64-datemix --seed 42
```

比較程序：

1. 先以 seed 42 與既有 `a-signal64` seed 42 的**完整 validation** 比較；不以 test 選擇方案。
2. 確認 `summary.json` 的 `training_batch_composition`：混洗順序每 optimizer step 的相異
   日期×市場數應遠高於 `unmixed_reference_order`。
3. 事先固定、只用 `metrics.jsonl` 已記錄 validation 指標的判準：
   - alpha：最佳 normalized Smooth L1 低於歷史 seed 42 最佳值 0.4548，且最佳點不是首次評估；
   - 方向：最佳 CE 低於 1.0284；該點 10–14 日 ECE 低於歷史 seed 42 最佳點的
     0.0864–0.1037；market-macro CE（1.0197）與 macro-F1（0.3592）不得變差。
4. 只有 validation 判準成立時，才補齊其他 seeds 並讀取最終 test，以逐日配對區間對照
   GRU／GBDT 等 learned baselines；test 期間已多次用於比較，不能稱為完全未見資料的確認。
   最終 benchmark 的 `daily_signal_loss` 與各 horizon 預測類別比例作為輔助診斷，
   檢查逐日 loss 波動與長天期類別偏移是否減少。
5. 任一判準不成立也保留結果；不在同一輪加入其他改動，以免無法歸因。

結果：`a-signal64-datemix` seed 42（R24）的 alpha 與方向判準皆未通過，因此不補跑其他 seeds。
完整數值、batch 組成與解讀見 [模型開發 roadmap 第 5.6 節](../MODEL_DEVELOPMENT_ROADMAP.md)。

#### 13.2 以橫斷面 Rank IC-IR 重選 alpha checkpoint（只推論，事先固定）

獨立 alpha 以完整 validation normalized Smooth L1 選 checkpoint。訊噪比低時，各次評估的
loss 只差 0.2%–2%，cross-sectional 排序品質的差距卻可能大得多。本節是推論期診斷：
不重訓、不改 `checkpoint_monitor`、不改寫 best-checkpoint pointer，也不取代已發布的結果。

事先固定的規則（2026-10-18 寫入，早於讀取任何非最佳 checkpoint 的 test 結果）：

1. 分數 eqIR：取該次 validation 評估的
   `validation/cross_sectional/by_market/<市場>/<h>d/rank_ic_ir`，先對 h=1–14 平均成市場
   分數，再對 US、TWSE、TPEX 等權平均。任一市場或天期沒有有效值時，該次評估不參選。
2. 候選只限該 run validation leaderboard 保留的 checkpoints；未保留的評估點沒有權重可匯出。
3. 選 eqIR 最高者；若與 loss 選出的最佳 checkpoint 相同，就不另外匯出。
4. 範圍是 `a-signal64` 三個 seed（R21–R23）的 alpha 子模型。分類器以 CE 選模，CE 本身是與
   任務一致的 proper scoring rule，不重選；R24（datemix）已記錄為負結果，不納入。

由 `metrics.jsonl` 計算的結果：

| Run | loss 選中 | 其 eqIR | eqIR 選中 | 其 eqIR | 是否另外匯出 |
| --- | --- | ---: | --- | ---: | --- |
| R21 alpha | `checkpoint-003680` | 0.328 | `checkpoint-022079` | 0.533 | 是 |
| R22 alpha | `checkpoint-011040` | 0.432 | `checkpoint-018399` | 0.490 | 是 |
| R23 alpha | `checkpoint-018399` | 0.519 | `checkpoint-018399` | 0.519 | 否（相同） |

執行：以 `export-signal-predictions --checkpoint RUN_ID/checkpoint-NNNNNN` 推論完整 validation
與 test；指定的 checkpoint 必須在 leaderboard 上，否則拒絕。`split.json` 重算的 validation
loss 與各市場 Rank IC-IR 必須與 `metrics.jsonl` 同一步的記錄一致。分析時，loss 選模與 eqIR 選模各自計算三 seed 集成與誤差拆解。
Test 只記錄，不得用來修改本規則、選 seed 或回頭換 checkpoint。

判讀限制：

- 從 5 個保留 checkpoint 中取最大值有選擇偏差，validation eqIR 會高估。
- 每個市場的 validation Rank IC 只有 112 個有效日期，候選間的 eqIR 差距（例如 R22 的
  0.432 對 0.490）可能仍在雜訊內。R22 的 eqIR 選擇也以平均 Rank IC 換取穩定度
  （eqIC 0.0637→0.0525）。
- 報告時並列 loss 選中與 eqIR 選中兩組，以及各自的三 seed 集成，並分市場列出 Rank IC。
  兩組的 test 差異只作描述性對照，不得解讀為選模規則的優劣。
- 市場等權分數主要來自台股：US 的 1–14 日平均 Rank IC 只有 0.017–0.047，而台股 validation IC
  曾在 test 反轉（TPEX）。因此 eqIR 不一定代表可轉移的訊號。
- 結果只作為日後是否正式改 `checkpoint_monitor` 的證據；正式變更需要程式修改、新 preset 與
  三 seed 訓練。

#### 13.3 第 1 階段：台股免費資料的方向資訊測試（只用 CPU，事先固定）

P0 逐樣本拆解把三分類 CE 分成兩部分：「報酬會不會超出 ±1%」與「超出時是漲還是跌」。
分類器相對 train 先驗的改善全部來自前者；後者在每個天期都不優於先驗，Kronos 與 GBDT 皆然。
例如 validation 台股 1 日，前者改善 5.8%，後者為 −0.3%（Kronos 三 seed 平均）。

本節檢驗：加入免費的台股籌碼、期貨與美股隔夜資料後，GBDT 的「漲還是跌」能否優於先驗。
這是決定是否以新特徵投入 GPU 訓練的前置測試。它不重訓 Kronos，也不改現有資料集、標籤或 baseline。
操作方式見 README 的「台股資訊測試」。

事先固定的設定（2026-10-18 寫入，早於任何輔助資料下載與模型擬合），全部記錄於
`configs/research/tw_information_test.json`：

1. **樣本**：`a-absolute-direction64` 資料組中 TWSE、TPEX 的清理後樣本，沿用既有切分
   （train 至 2025-06-01、validation 至 2025-12-01、test 至 2026-06-01）。
2. **標籤**：
   - 進出場與 `execution_total_return` 相同：第 t+1 日開盤進場，第 t+h 日收盤出場。
   - 主要天期為 1／2／3／5 日；10／14 日只作對照。
   - 主要門檻為 max(成本下限, k_h × σ20 × √h)。成本下限：股票 0.585%、ETF 0.385%。
     k_h 只用 train 校準，使中性比例為 1/3。
   - 固定 ±1% 只用於 `F0` 與 `all_pre_open` 的對照。
3. **特徵組**：`F0`、`F0_long`、`short_only`、`flows`、`market`、`us_evening`、`us_pre_open`、
   `all_evening`、`all_pre_open`。內容與資料公布時點見 README。
4. **模型**：
   - sklearn HistGradientBoosting：learning rate 0.05、max leaf nodes 31、min samples leaf 500、
     L2 1.0、最多 400 輪、seed 42。
   - 只在 train 擬合。boosting 輪數以每 10 輪檢查一次的方式，在 train 最後 20% 上選擇；
     該段與前段間隔 15 個交易日。
   - 不以 validation 擬合或選參。
5. **主要指標**：validation 台股整體的「漲還是跌」CE，只計算非中性標籤的樣本。
   比較基準是 train 各市場的類別比例（Jeffreys 平滑）。
6. **判準**：某個新增特徵組（`F0` 以外）在 1／2／3 日中至少兩個天期，同時滿足以下三項：
   1. 「漲還是跌」CE 相對先驗改善至少 1%，即 (先驗 − 模型)／先驗 ≥ 0.01。
   2. 相對先驗的差距，95% 區間下界大於 0。
   3. 相對 `F0` 的差距，95% 區間下界大於 0。

   區間的計算方式：以日期為單位做 circular block bootstrap，區塊 5 日、重抽 1,000 次、seed 7；
   統計量是逐日加總後的比值。報告的 `criteria.information_found` 為 true 即判定通過。
7. **結果的處置**：
   - 通過：通過的特徵組進入台股模型設計（新標籤、新天期）。設計需另行審閱，GPU 訓練需另外核准。
   - 未通過：判定這些免費資料在 1–3 日不含足以優於先驗的方向資訊。不投入 GPU 訓練，先檢討再決定下一步。
8. **次要問題**（只描述，不影響判準）：
   - 回看長度：`F0_long`、`short_only` 相對 `F0` 的「漲還是跌」差距與區間
     （`window_question_direction_gap_vs_F0`）。
   - 訊號時點：`all_evening` 對 `all_pre_open`。
   - 門檻：固定 ±1% 對主要門檻。
   - Alpha：各特徵組的 normalized Smooth L1 與分市場 Rank IC。
   - Test 指標與簡易回測：每日買進 signal 前 10% 且 signal 大於 0 的標的，扣除成本下限。
     多日持有期互相重疊，t 值會偏樂觀。

判讀限制：

- 有 8 個新增特徵組、3 個天期，存在多重比較。判準要求同一組在兩個天期通過，以降低偶然通過的機率。
- GBDT 未通過，不排除其他模型能利用這些資料；但代表在目前樣本量下，訊號弱到難以穩定估計。
- 某個輔助來源為 `failed` 時，依賴它的特徵組不參與判準，並列於 `skipped_feature_sets`；
  以其餘特徵組判定。
- Validation 只有約半年（2025-06 至 2025-12），區間會較寬，結論只適用於這段期間。

#### 13.4 第一階段回測：台股排序訊號扣除成本後的檢驗（本機，事先固定）

§13.3 顯示，免費資料無法讓絕對漲跌優於先驗；但多個模型的台股截面排序（Rank IC）在 test 仍有效。本節檢驗這些排序訊號扣除台股實際成本後，能否穩定勝過被動持有。這是研究用回測，不重訓模型，也不使用 RunPod。程式和設定放在 `backtest/`，使用獨立的 uv 環境；預先登記的設定記錄在 `backtest/config/stage1.json`。

事先固定的設定（2026-10-19 寫入，早於任何回測計算）：

1. **資料**：
   - 使用 P0 評估母體中的上市、上櫃列；列順序已用樣本雜湊與各匯出檔核對一致。
   - 訊號日：validation 為 2025-06-02 至 2025-11-10（112 日），test 為 2025-12-01 至 2026-05-11（103 日）。
   - 報酬：第 t+1 日開盤到第 t+h 日收盤的還原價 log 報酬（`absolute_targets`）。
   - 大盤報酬：絕對報酬減 alpha，在同市場、同一天皆相同（加權指數或櫃買指數）。
2. **訊號（13 個）**：每個訊號都使用和持有天數相同天期的預測。
   - Kronos alpha（8 個）：R21（checkpoint 003680）、R21-IR（022079）、R22（011040）、R22-IR（018399）、R23（018399）、R24 datemix（007360），以及三 seed 集成兩個——LOSS（R21、R22、R23）和 IR（R21-IR、R22-IR、R23），集成取點預測的平均。
   - Baseline alpha（2 個）：GRU、GBDT。
   - 方向模型對照（3 個）：R21–R23 方向模型的平均機率、GRU 方向、GBDT 方向，訊號為 P(漲) − P(跌)。
3. **交易規則**：
   - 第 t 日收盤後排序，第 t+1 日開盤等權買進前 K 檔，持有 h 個交易日，在第 t+h 日收盤賣出。
   - 下一批在同一天收盤後選股，下一個交易日開盤買進。
   - h 個起始日各成一條序列，等同把資金分成 h 份輪流進場。
   - 網格：h ∈ {1, 2, 3, 5, 10}；K ∈ {10, 20, 50}；市場 ∈ {TWSE、TPEX、TW（兩市場合併排序）}。
   - 版本：只做多為主要版本；避險版的持股報酬改為減去對應大盤報酬，期貨成本以 0 近似。
4. **成本**：
   - 買進：手續費 0.1425% × 折扣。
   - 賣出：手續費 0.1425% × 折扣，加證交稅（股票 0.3%、ETF 0.1%）。
   - h = 1 是現股當沖，股票證交稅為 0.15%。
   - 滑價：每邊上市 10 bps、上櫃 20 bps。
   - 主要情境：折扣 0.6、滑價 1 倍、保守換手（每批全額換股）。
   - 敏感度：折扣 {1.0, 0.6, 0.28} × 滑價 {0, 1, 2} 倍 × 換手 {保守, 樂觀}。樂觀換手是指續抱的股票不重複付費。
   - 不模擬：每筆最低手續費、漲跌停、流動性與整股限制。
5. **指標**：
   - 主要指標：每批淨報酬相對「同日母體等權、同一持有窗口、不計成本」的**淨超額報酬**。
   - 其他：相對指數的超額報酬、年化（× 252/h）、IR、Sharpe、最大回撤、每批勝率、換手率、成本拖累、毛超額報酬，以及損益兩平的每邊成本。
   - IR = 每批超額報酬平均 ÷ 標準差 × √(252/h)，先在每條起始序列內計算，再取平均。
   - 95% 區間：依訊號日排序的每日批次超額序列，做 circular moving-block bootstrap，區塊長度 max(5, 2h)，重抽 1,000 次，seed 11。
6. **選擇程序**：只用 validation。
   - 在每個市場（TWSE、TPEX、TW）的「alpha 訊號 × h × K」組合中，以只做多、主要成本情境計算，選出淨超額 IR 最高的一組，作為**主要設定**。
   - test 只跑一次。完整網格只作描述；標題結果固定為主要設定，不得改選。
7. **進入下一階段的條件**：主要設定在 validation 的淨超額報酬 95% 區間下界大於 0，**且**在 test 的淨超額報酬為正。第三階段只對通過的主要設定，用 2026-06-01 之後的新資料驗證。

判讀限制：

- 從每個市場 150 組設定中取最大值，會高估 validation 的結果；test 用來檢查這個偏差。
- validation 和 test 各約 5 個月，區間會較寬。
- 母體已通過資料清理規則，可能排除了之後停牌或下市的股票。這個偏誤對所有策略和等權基準相同，但讓絕對報酬偏樂觀。
- 沒有模擬漲停開盤買不到、成交量不足、每筆最低手續費等限制，結果偏樂觀，需要在第三階段補上。

#### 13.5 第一階段 1b：持有緩衝與逐部位記帳（本機，事先固定）

§13.4 的主要設定在三個市場都未通過門檻：validation 選出的 10 日持有設定，test 淨超額年化為 −37% 至 −44%。事後分析有兩項發現：

- 1 日排序在 validation 和 test 都有毛優勢。前 K 名比同日等權每天多約 25–30 bps（上市）、15–20 bps（上櫃），但主要成本情境每次來回約 52 bps，優勢被成本吃掉。
- 相鄰兩日的前 K 名約有 85% 重疊。§13.4 的樂觀換手雖然不重複收費，卻漏算續抱股票的隔夜跳空，且仍用當沖稅率，不能作為降低換手的證據。

本節用逐部位記帳檢驗：以持有緩衝降低換手，能否把排序的毛優勢轉成扣除成本後的淨超額報酬。程式與 §13.4 共用 `backtest/`；事先登記的設定記錄在 `backtest/config/stage1b.json`。

**資料污染聲明**：本節的構想來自 §13.4 對 test 的事後分析（1 日毛優勢、十分位報酬與低成本情境），test 因此已不是未見資料。本節的 test 結果只作探索性判讀；唯一的確認性檢驗是第三階段，使用 2026-06-01 之後的資料，模型與規則事先凍結。

事先固定的設定（2026-10-19 寫入，早於任何 1b 策略計算；寫入前只做過第 2 點的記帳恆等式檢查）：

1. **資料**：與 §13.4 相同的母體、分割、13 個訊號與列順序核對。
2. **記帳恆等式**：
   - 同一檔股票在相鄰訊號日 d、d+1 的資料列，對 h = 1…13 都滿足 `abs[d, h+1] − abs[d+1, h] = log(open(d+2) / open(d+1))`，因此可以算出開盤到開盤的持有報酬。
   - 寫入前的檢查：validation 96,166 組、test 93,402 組相鄰資料列全部成立（最大差異 ≤ 9×10⁻⁸）。這代表相鄰分割日期是連續交易日，且所有資料列使用同一套還原價格。
   - 執行時重新檢查股票與指數（絕對報酬減 alpha）的恆等式；誤差 ≤ 10⁻⁵ 的比例低於 99.9% 就停止。
3. **交易規則**（每個市場、訊號、參數組合各自模擬）：
   - 第 d 日收盤後，以預測天數 s 的分數將該市場所有股票排序（1 = 最好）。
   - 持有中且排名 ≤ b × K 的股票續抱；排名 > b × K 的股票賣出。
   - 空出的位置依排名由高到低補入尚未持有的股票，補滿 K 檔。b = 1 即每天換成前 K 名，作為無緩衝對照。
   - 所有買賣在第 d+1 日開盤集合競價成交；不設持有上限。
   - 漲跌停：開盤相對前一日收盤上漲 9.5% 以上的買單視為買不到，該位置保留現金到下一次成交；下跌 9.5% 以上的賣單視為賣不掉，續抱並在下一個訊號日重新判斷。缺前一日資料時視為可成交；期末清倉不套用跌停限制。
   - 資金分成 K 個等額位置。位置賣出後的淨額全數買入新股票，不做再平衡。
   - 持有中的股票在下一個訊號日沒有資料列時，改以第 d+2 日收盤出場（`abs[d, 2]`），該位置在下一次成交保持現金。
   - validation 與 test 各自獨立模擬：第一個訊號日收盤後建倉，最後一個訊號日的下一個開盤全部出清。
4. **成本**：
   - 買進：手續費 0.1425% × 折扣，加滑價。
   - 賣出：手續費 0.1425% × 折扣，加證交稅（股票 0.3%、ETF 0.1%）與滑價。所有部位至少持有一個隔夜，不適用當沖稅。
   - 滑價：每邊上市 10 bps、上櫃 20 bps。
   - 主要情境：折扣 0.6、滑價 1 倍。敏感度：折扣 {1.0, 0.6, 0.28} × 滑價 {0, 1, 2} 倍。
5. **網格**：s ∈ {1, 3, 5}；K ∈ {10, 20, 50}；b ∈ {1, 2, 4}；市場 ∈ {TWSE、TPEX、TW（兩市場合併排序）}；13 個訊號全部列入候選（方向訊號以 P(漲) − P(跌) 排序）。每個市場 351 組候選。
6. **指標**（每日、不重疊的開盤到開盤序列）：
   - 主要指標：組合每日淨報酬相對「同日、同市場全部股票等權、每日再平衡、不計成本」的**淨超額報酬**。
   - 其他：年化（× 252）淨超額、IR、CAGR、Sharpe、最大回撤、每日勝率、每日單邊換手率（新買入位置數 ÷ K）、持有天數（平均、中位數、p90）、年化成本拖累、損益兩平的每邊成本（年化毛超額 ÷ 年化雙邊換手）、漲跌停擋單次數、缺資料出場次數。
   - IR = 每日淨超額平均 ÷ 標準差 × √252。
   - 避險版：每日減去持股位置對應指數（加權指數或櫃買指數）的開盤到開盤報酬，期貨成本以 0 近似，只作描述。
   - 95% 區間：每日淨超額序列的 circular moving-block bootstrap，區塊長度 10 個交易日，重抽 1,000 次，seed 11。
7. **選擇程序**：只用 validation。
   - 每個市場在 351 組候選中，以只做多、主要成本情境的淨超額 IR 最高者為**主要設定**；同分時依訊號、s、K、b 排序。
   - test 只跑一次。完整網格只作描述；標題結果固定為主要設定，不得改選。
8. **門檻**：主要設定在 validation 的淨超額 95% 區間下界大於 0，**且** test 淨超額平均為正。
   - 通過只代表值得進入第三階段前瞻測試。
   - 三個市場都未通過時，停止「降低換手」路線，回到模型本身。

判讀限制：

- 從每個市場 351 組候選取最大值，會高估 validation 結果；test 已被事後分析看過，無法完全檢查這個偏差。
- 仍未模擬成交量與流動性、每筆最低手續費、整股交易單位、股利所得稅與補充保費；母體的存活者偏誤同 §13.4。
- 漲跌停以還原價的開盤跳空近似，除權息日可能誤判。
- 位置不再平衡，權重會隨價格漂移。

#### 13.6 與持有方式對齊的評估：現有訊號的開盤到開盤重評（本機，事先固定）

§13.4、§13.5 的結果見[回測結果報告](backtest_stage1_results.md)：前 K 名的優勢只出現在交易時段，隔夜就反轉。現行標籤（1 日 = 次一交易日開盤到收盤）與 Rank IC 評估，都和「收盤後產生訊號、隔日開盤買進、持有數天到兩週、開盤賣出」不一致。2026-10-19 決定讓標籤與評估對齊持有方式，分兩部分：

- **A（本節）**：建立與持有方式對齊的評估，對現有訊號做開盤到開盤重評。只在本機計算，不訓練模型。
- **B（之後另立一節事先登記）**：GBDT 改用開盤到開盤標籤訓練，檢查是否有可學的多日優勢。B 的判準使用本節的評估定義。

本節只作描述，不設門檻，也不據此改選模型；test 已在先前分析中多次使用，只作探索性判讀。程式在 `backtest/`，設定記錄在 `backtest/config/holding_eval.json`。

事先固定的設定（2026-10-19 寫入，早於任何開盤到開盤指標的計算）：

1. **資料**：與 §13.4 相同的母體、分割、13 個訊號與列順序核對；執行時重新檢查 §13.5 的記帳恆等式，不符就停止。
2. **開盤到開盤標籤**：
   - 股票在訊號日 d、持有 h 日：`R(d, h) = abs[d, h+1] − abs[d+h, 1] = log(open(d+1+h) / open(d+1))`，需要同一檔股票在第 d+h 個訊號日的資料列，否則為缺值。
   - 指數以同一方式由「絕對報酬減 alpha」推得；開盤到開盤 alpha = 股票減指數。
   - 持有天數：主要 h = 5、10；輔助 h = 3。
3. **訊號**：每個訊號使用自己對天期 h 的預測；方向訊號為 P(漲) − P(跌)。
4. **指標**（validation 與 test；上市、上櫃分開計算）：
   - **Rank IC**：每日截面 Spearman（平均名次、至少 3 筆、非常數），對開盤到開盤 alpha；另列對現行標籤的 Rank IC 作對照。報告平均與 IR（平均 ÷ 標準差）。
   - **十分位**：每日依分數排序，前 10%（至少 1 檔，四捨五入）與後 10% 的開盤到開盤簡單報酬，各減同日同市場有標籤股票的等權平均；另列前後差。
   - **交易時段與隔夜拆解**：前 10% 的超額 log 報酬拆成 h 個交易時段（各日開盤到收盤）與其餘的隔夜部分，均相對等權。
   - **固定參數的持有組合**：沿用 §13.5 的引擎與記帳，預測天數 s = h、K = 20、b = 2、主要成本情境（折扣 0.6、滑價 1 倍）、只做多；市場為上市、上櫃與合併。報告年化淨超額、IR、毛超額、成本拖累、每日換手與平均持有天數。
   - 區間：每日序列的 circular moving-block bootstrap，重抽 1,000 次、seed 11；h 日標籤的序列區塊長度 max(5, 2h)，持有組合的每日序列區塊長度 10。
5. **判讀**：
   - 只描述，不設門檻。
   - 主要天期中，若某訊號在某市場的 validation 前十分位超額 95% 區間下界大於 0，列為「可留意」，但不據此做決策。
   - 決策只依 B 或之後另行事先登記的檢驗。

判讀限制：

- 13 個訊號 × 2 市場 × 3 天期，多重比較會產生偶然的「可留意」項目。
- h 日標籤互相重疊，區塊 bootstrap 只部分處理自相關。
- 接近分割末端的訊號日沒有完整的 h 日標籤，validation 與 test 的有效日期隨 h 減少。
- 其餘限制同 §13.5（流動性、最低手續費、整股、股利稅、存活者偏誤）。

#### 13.7 本機 Kronos 推論：一致性比對與 checkpoint 選點（本機，事先固定）

§13.6 顯示：同一個 run 中，不同 checkpoint 的隔夜行為差異很大（R22 的 loss 選點隔夜拖累 −213.8 bps，Rank IC-IR 選點只有 −10.7 bps）；上市 10 日的 R22-IR 是唯一在 validation 與 test 都為正的訊號。2026-10-19 決定：訓練仍只在 RunPod；回測需要的推論在本機（M5 Pro）執行。本節固定兩件事：本機推論必須先通過的一致性比對，以及以持有指標選 checkpoint 的規則。設定記錄在 `backtest/config/local_parity.json` 與 `backtest/config/checkpoint_selection.json`。

事先固定的設定（2026-10-19 寫入，早於任何本機推論）：

1. **本機推論**：
   - 在獨立的 uv 子專案 `inference/` 執行，與 Poetry、RunPod 環境及 `backtest/` 分開。
   - 載入 checkpoint 自己的 `resolved-config.yaml` 與 `adapter.safetensors`；Kronos 原始碼與基礎權重使用以 sha256 與 revision 固定的版本。
   - 裝置：MPS fp32；MPS 不支援時改用 CPU fp32。實際採用的裝置必須自己通過一致性比對。
   - 範圍：只推論 P0 評估母體中的上市、上櫃列。每列記錄股票與日期，`backtest/` 依此一對一對應到評估陣列，對應必須完整。
   - 推論不改變模型、checkpoint 或任何 lifecycle 紀錄。
2. **一致性比對（P1）**：
   - 參照：R22 checkpoint-018399 在 RunPod RTX 5090、bf16 匯出的 validation 預測（`export-20261017T223254Z-ce4974f3`）。
   - 本機以同一個 checkpoint 推論 validation 的台股列。
   - 通過條件（全部符合）：
     1. 每個「日期 × 市場 × 天期」的截面 Spearman（本機對參照），至少 99% 的組合 ≥ 0.999。
     2. 依 §13.6 的定義，上市、上櫃各自 5 日與 10 日的開盤到開盤 Rank IC，兩者差距 ≤ 0.005。
     3. 同樣條件下，前十分位超額的差距 ≤ 5 bps。
   - 另記錄：每秒處理筆數、記憶體峰值、預測的最大絕對差。
   - 未通過：不進行 checkpoint 選點與 §13.8；回報原因後再討論。
3. **checkpoint 選點（P2）**：
   - 候選：三個 alpha run 各自保留的 5 個 checkpoint。
     - R21（`run-20261015T064558Z-f04e155d4299416f-alpha`）：003680、007360、011040、018399、022079。
     - R22（`run-20261015T064559Z-624f212a27894308-alpha`）：003680、007360、011040、014720、018399。
     - R23（`run-20261015T064600Z-13b8b4376ea5413b-alpha`）：003680、007360、018399、022079、025759。
   - 規則：每個 run 選 validation 上市 10 日前十分位開盤到開盤超額（§13.6 定義）最高的 checkpoint；同分時選步數較晚者。
   - 只用 validation 選點。選出的 3 個 checkpoint 另對 test 推論，只作描述。
   - 不設門檻；選出的 checkpoint 在 §13.8 以次要描述檢驗。

判讀限制：

- 選點與描述都用同一段 validation，選出的 checkpoint 在 validation 上的數值偏樂觀；確認只能靠 §13.8 的未見資料。
- 15 個候選、validation 112 日；選點規則只看上市 10 日，其他市場與天期不參與。
- 本機為 fp32、參照為 bf16；一致性比對量測的是兩者的差異，不保證逐列相同。

**§13.7 修訂（2026-10-19，於本機一致性比對未通過之後；上方原登記內容保留不改）**

本機推論的結果：依原設定以 MPS fp32 推論 R22 checkpoint-018399 的 validation 台股列（97,363 列，每秒 52 筆），一致性比對未通過。

| 判準 | 結果 | 通過 |
| --- | --- | --- |
| 截面 Spearman ≥ 0.999 的組合至少 99% | 0%（3,136 組；中位數 0.995，最低 0.908） | 否 |
| Rank IC 差距 ≤ 0.005 | 最大 0.0017 | 是 |
| 前十分位差距 ≤ 5 bps | 最大 11.1 bps（上櫃 5 日） | 否 |

診斷：差異來自數值精度，不是系統性偏差，也不是程式錯誤。

- 本機對參照的平均偏移 0.04–0.25 bps（預測標準差 83 bps），迴歸斜率 1.000–1.002，整體相關 0.997。
- 在 CPU 以 RunPod 相同的 bf16 計算，單日截面 Spearman 為 0.9996–0.9999；只把凍結權重捨入成 bf16 則沒有改善。差異來自計算過程中的 bf16 捨入。
- MPS bf16 不準（Spearman 0.65–0.93），CPU bf16 太慢（每秒約 2.4 筆），兩者都不可行。

修訂後的設定：

1. 推論改用既有的 RunPod GPU 匯出流程（`export-signal-predictions`，CUDA bf16），與參照匯出是同一套程式與數值。本機只做評分與回測；`inference/` 子專案只作診斷，不用於任何判定。
2. 一致性比對取消。完整性檢查沿用 P0 的做法：每份匯出的列順序雜湊必須與 baseline 一致；validation 重算的選模分數，必須與該 checkpoint 在 `metrics.jsonl` 同一步的 validation 紀錄一致（差距 ≤ 10⁻⁶）。
3. 選點規則不變。15 個候選中已有 5 個在 P0 匯出（R21 的 003680、022079；R22 的 011040、018399；R23 的 018399），其餘 10 個以一次匯出工作補齊，validation 與 test 一併匯出。
4. 選點程式只讀 validation；test 只計算選出的 3 個 checkpoint，由程式強制，只作描述。
5. §13.8 第 5 點「§13.7 的一致性比對通過後才執行」改為：推論以 RunPod GPU 匯出流程執行，匯出完整性檢查通過後才評分。前瞻資料的推論需要把匯出程式擴充到 `forward` 分割，實作前另行說明。

#### 13.8 第三階段：前瞻檢驗（2026-06-01 之後的未見資料，事先固定）

本節是上市 10 日 R22-IR 的第一個確認性檢驗。這個候選是 §13.6 事後找到的，因此判準在前瞻資料準備之前寫入，模型、市場、天期與參數全部凍結。設定記錄在 `backtest/config/stage3.json`。

事先固定的設定（2026-10-19 寫入，早於前瞻資料的下載與任何前瞻推論）：

1. **資料**：
   - 在 RunPod CPU 以既有資料準備流程建立新資料組：相同 profile（`us_tw_eodhd`）、universe 與清理規則；結束日為準備當時可取得的最新交易日。
   - 新增 `forward` 分割：訊號日 ≥ 2026-06-01，且 14 日標籤完整。train、validation、test 的定義不變。
   - 資料檢查（全部通過才評分）：
     1. 新資料組的 manifest 就緒。
     2. `forward` 分割的台股列，§13.5 的記帳恆等式至少 99.9% 成立（誤差 ≤ 10⁻⁵）。
     3. 與現有資料組重疊的 test 期間，台股列的 `absolute_targets`（h = 1…14）至少 99% 的列，最大差距 ≤ 10⁻⁵。
2. **標籤與指標**：沿用 §13.6 的開盤到開盤定義、十分位與固定組合；只在 `forward` 分割內計算。
3. **主要假說（唯一影響判定）**：R22-IR（R22 checkpoint-018399）、上市、h = 10。兩項都符合才算通過：
   1. 前十分位開盤到開盤超額平均的 95% 區間下界大於 0。區間以 circular moving-block bootstrap 計算：區塊 20、重抽 1,000 次、seed 11。
   2. 固定組合（s = 10、K = 20、b = 2、主要成本情境、只做多）的年化淨超額平均大於 0。

   區間跨 0 即判定未通過，不另作解讀。
4. **次要描述（不影響判定）**：
   - ENS-IR、R23（checkpoint-018399）、§13.7 選出的 3 個 checkpoint。
   - 對照：loss 選出的 R21（003680）與 R22（011040）。
   - R22-IR 的上櫃結果與 5 日結果。
5. **程序**：
   - §13.7 的一致性比對通過後才執行。
   - 每個凍結的模型只對 `forward` 分割推論一次；不重選模型、市場、天期或參數。
   - 結果寫成中英雙語報告。
6. **結果的處置**：
   - 通過：上市 10 日的 R22-IR 列為交易規則的候選訊號，之後持續前瞻監測。
   - 未通過：不採用，回到模型本身的改善。

判讀限制：

- 可用訊號日約 80 個，10 日窗口互相重疊，區間會很寬；未通過不一定代表沒有優勢，但不能採用。
- 只涵蓋 2026 年夏季到秋季這一段行情。
- 新資料組的還原價會因 6 月之後的除權息而改變歷史數值；評分只在新資料組內部進行。
- 其餘限制同 §13.5（流動性、最低手續費、整股、股利稅、存活者偏誤）。

**§13.8 修訂（2026-10-20，於前瞻資料下載與任何前瞻推論之前；上方原登記內容保留不改）**

修訂原因：

1. **只需要台股資料**：
   - 第三階段只評台股。台股樣本的輸入只有個股本身和台股基準（`TAIEX.TW`、`TPEX.TWO`），行情來自證交所與櫃買中心官方來源，完全不用美股資料。
   - 沿用 `us_tw_eodhd` 就得重新下載全部美股歷史，而這些列不參與評分。
2. **需要明確的例外範圍**：§13.7 修訂後，推論改用 RunPod GPU 匯出流程。現行 checkpoint 載入規則要求資料組與訓練時完全相同，前瞻推論需要明確寫出可放行的差異。
3. **需要後十分位才能解讀**：P2 發現 test 期間前、後十分位都勝過等權平均，前十分位要和後十分位一起看。

修訂後的設定：

1. **資料**（取代原第 1 點的前兩項）：
   - 在 RunPod CPU 以既有流程建立新資料組，profile 改為 `tw_only`，只含證交所與櫃買中心官方資料。
   - 起始日（2016-01-01）、universe 模式、清理規則、benchmark 對照、固定切分邊界，都與目前推論所用的共用評估快照相同。
   - 結束日（不含）為建立 CPU Pod 當天（UTC），開 Pod 前寫入 `backtest/config/stage3.json`。
   - 新增 `forward` 切分：訊號日 d 滿足 2026-06-01 ≤ d < 結束日，且第 14 個交易日的標籤日早於結束日。train、validation、test 的定義不變。
   - 休市日：2026-06-01 之後若有證交所公告的臨時休市（例如颱風）或調整交易日，只補這段期間的日期，每一筆都列在報告中。2026-06-01 之前的日曆不變。
2. **資料檢查**（取代原第 1 點的檢查清單；在本機進行，全部通過才開 GPU）：
   1. 新資料組的 manifest 就緒。
   2. `forward` 切分的台股列，§13.5 的記帳恆等式至少 99.9% 成立（誤差 ≤ 10⁻⁵）。
   3. 與共用評估快照的 test 期間比對，以代號與訊號日對齊：
      - 舊資料組的台股列，至少 99% 要在新資料組找得到。
      - 對到的列中，至少 99% 的 `absolute_targets`（h = 1…14）差距都 ≤ 10⁻⁵。
3. **推論範圍**（新增）：
   - 只放行這些差異：資料組身分與產物、profile（`us_tw_eodhd` → `tw_only`）、結束日、`forward` 邊界，以及第 1 點允許補的休市日。
   - 模型、輸入長度、天期、標籤定義、其餘清理規則、train／validation／test 邊界，都必須與 checkpoint 訓練時相同，由程式逐項檢查。
   - `export.json` 記錄新舊兩個資料組的身分。
   - 推論用的 GPU 型號（RTX 5090）、torch、Python 和 bf16 設定，都與 2026-10-19 的匯出相同。租不到 RTX 5090 時等待，不換 GPU。
   - 推論 6 個 checkpoint 的 `forward` 切分：R21 的 003680、018399、022079，R22 的 011040、018399，R23 的 018399。
4. **管線一致性檢查**（新增，評分前必須通過）：
   - 在同一個 GPU 工作中，以新資料組對 R22-IR 推論 test 切分。
   - 以代號與訊號日對齊 2026-10-19 匯出的 test 台股列，至少 99% 的舊列要對得到。
   - 預期兩次的預測逐位元相同。若不完全相同，以下兩項都符合才算通過：
     - (a) 對到的列，14 個天期合併計算的預測相關係數 ≥ 0.999。
     - (b) 依 §13.6 重算的上市 10 日前十分位超額，與選點報告的 +103.2 bps 相差 ≤ 5 bps。
   - 未通過：不評分；查明原因並記錄後才能重跑。
5. **程序**（補充原第 5 點）：
   - 順序：CPU 準備資料 → 本機資料檢查 → GPU 推論（含一致性檢查）→ 本機評分。
   - 資料檢查與一致性檢查全部通過前，不計算任何前瞻指標。檢查紀錄不完整時，評分程式拒絕執行。
   - 評分前若因技術故障重跑推論，可以，但要記錄原因；評分只使用一次推論的結果。
   - 開始評分後不再重跑，也不重選模型、市場、天期或參數。
6. **描述性補充**（不影響判定）：主要假說與每個次要訊號，都在前十分位旁同時列出後十分位超額，以及前後十分位差。

判讀限制（補充）：

- 新資料組只含台股，美股列不在範圍內，不影響本節任何判定。
- 台股標的清單以官方來源的下載內容為準，可能缺少期間內下市的標的（存活者偏誤，同 §13.4）。
- 一致性檢查只涵蓋 test 期間；只發生在前瞻期間的資料問題，由資料檢查第 2 項把關。

**§13.8 修訂補充（2026-10-20，實作前、取得前瞻資料之前）**

原因：訓練資料管線的切分邏輯被納入資料組指紋，新增切分會讓現有資料組全部失效；而且模型若證實有價值，之後本來就需要獨立的推論流程。

1. 前瞻資料改由獨立的前瞻推論管線產生，不建立新的訓練資料組。這一點取代修訂第 1 點的「在 RunPod CPU 以既有流程建立新資料組」：
   - 在本機（台灣網路）以既有的證交所與櫃買中心 provider，抓取 2025-03-01 起至結束日的台股資料。
   - 正規化、還原價、合格標的、基準對照與清理規則，一律呼叫訓練資料管線的同一批函式。
2. 樣本群有兩個，都沿用 validation／test 的樣本規則：
   - `overlap`：訊號日 2025-12-01 ≤ d < 2026-06-01，標籤早於 2026-06-01，與原 test 定義相同。
   - `forward`：定義不變。
3. 資料檢查：
   - 第 3 項改與 `overlap` 比對。
   - 新增第 4 項（GPU 前）：`overlap` 的台股列，新管線的模型輸入與共用評估快照逐列比對，至少 99% 的列最大相對差距 ≤ 10⁻⁶。
4. 推論不載入任何資料組，只檢查模型契約；匯出紀錄輸入包的雜湊。管線一致性檢查改用 `overlap`。
5. 其餘不變：GPU 型號、6 個 checkpoint、判準、程序、描述性補充。

**§13.8 結果（2026-10-20）**：主要假說未通過，依第 6 點不採用；見[前瞻檢驗結果報告](stage3_forward_results.md)。

#### 13.9 可捕捉性分析（validation，探索性，2026-10-20 於分析前寫入）

背景：§13.8 未通過，但上市的開盤到開盤 Rank IC 在 validation、test、前瞻三個期間都為正。只做多前十分位、持有 10 日，只用到排序資訊的一小部分，而且現股成本高。使用者可交易的工具：做多現股、台指期／小台／微台、個股期貨（可放空）。本節找出哪一種可實作的做法，最能把排序資訊轉成扣成本後的超額。設定記錄在 `backtest/config/capturability.json`。

1. **資料範圍**：
   - 只用 validation（訊號日 2025-06-02 至 2025-11-10，116 日）做比較與選擇。
   - test 與 §13.8 前瞻期間都已看過，只作探索性對照，不參與選擇。
   - 不訓練、不重新推論：使用 2026-10-19 匯出的 validation 預測。訊號為 §13.8 的 6 個登記與選點訊號。
2. **A：排序資訊在哪裡（不計成本）**：
   1. 分群：有無個股期貨（期交所標的清單，且該日有掛牌交易）、流動性三分位（訊號日前 20 個交易日的平均成交值，同日同市場內分組）、上市與上櫃。各群計算 Rank IC、前後十分位超額與前後差（h = 5、10）。
   2. 十分位曲線：h = 5、10 的十個十分位超額。
   3. 衰減：h = 1…13 的 Rank IC，以及前、後十分位的交易時段與隔夜分解。
3. **B：可實作組合（扣成本）**，候選固定為：
   - B0（基準）：只做多現股前 20 檔（§13.6 的固定組合）。
   - B1：B0 加上以台指期貨避險（避險比例 1 與滾動 beta 兩種）。
   - B2：個股期貨多空，在有期貨的標的中前段做多、後段放空，多空金額相等。
   - B3：個股期貨依排名分配多空權重，全部有期貨的標的都持有，多空中性。
   - 每個候選比較 5 日與 10 日換倉，以及分數平滑（前幾日分數平均）開或關。
   - 成本：現股沿用主要情境；期貨交易稅十萬分之二、每口手續費、依成交量分級的滑價情境、每月轉倉。手續費與資金規模（決定契約口數的顆粒度）由使用者提供，未提供前以參數列出敏感度。
4. **期貨資料**：期交所每日期貨行情（公開下載表單），涵蓋 validation 期間與其前 20 個交易日。契約代碼以期交所目前的標的清單對照股票代號；對不到的契約列出數量。個股期貨的報酬以標的股票的開盤到開盤報酬近似。
5. **判讀**：探索性，不設門檻。依 validation 選出一個做法（訊號、組合、換倉頻率、平滑），之後另寫事先登記，以 2026-10-20 之後的新資料確認。結果寫成中英雙語報告。

判讀限制：

- validation 只有 116 日，區間會很寬；組合之間的差異以配對方式比較。
- 期貨開盤（08:45）早於現股（09:00），另有基差、除權息調整與轉倉，近似報酬會有誤差。
- 期交所目前的清單可能缺少期間內已下市的契約。
- 保證金、追繳與期貨部位的容量限制，除口數顆粒度外不模擬。

#### 13.10 橫斷面排序任務：rank 標籤與 baseline（2026-10-21 於實作前寫入）

背景：數值 alpha 與絕對方向都沒有穩健勝過 baseline；只有同日同市場內的相對排序有正的 Rank IC（Kronos alpha 在 validation 0.099–0.119、test 0.091–0.124）。專案改為預測橫斷面名次。標籤區間不變：t0 收盤產生訊號，t1 開盤進場，第 h 日收盤（h = 1…14）。進場門檻與出場規則由使用者的 hermes agent 以逐日滾動預測回測決定，模型只提供各 horizon 的排序。舊的 alpha、方向模型與 baseline 不再保留相容。

1. **任務**：`cross_sectional_rank`，參數只有最小組大小 `rank_min_group_size`（預設 20）。
2. **標籤**：
   - 來源：既有的相對大盤調整後對數報酬（t+1 開盤到 t+h 收盤）。每個市場只有一個大盤（TWSE 為 TAIEX.TW、TPEX 為 TPEX.TWO、US 為 VTI.US），所以同日同市場內，相對報酬的名次與絕對報酬的名次相同。
   - 分組：同一 split、同一 cutoff 日、同一市場；只用該 split 的樣本範圍內的股票。
   - 轉換：同分取平均名次 r，計算 (r − 0.5) / n，再取標準常態分配的反函數。
   - 少於 `rank_min_group_size` 檔的分組不給標籤（NaN），訓練時遮掉。評估仍以原報酬計算 Rank IC，不受遮罩影響。
3. **標籤儲存（Parquet）**：
   - 每個 split 一份，放在該 split 的 bar store 旁：`rank-labels/<識別碼>/`。
   - `shards/bucket-XXXX/shard.parquet` 沿用 bar store 的分桶；每個 symbol 一個 row group，欄位為 cutoff 時間與 `rank_h01…rank_h14`（float32）。
   - `label-index.parquet` 記錄每個 symbol 的檔案、row group、列數與樣本序號起點；`rank-labels.json` 記錄識別內容、各檔 sha256 與各市場的分組統計。
   - 識別碼由 bar store manifest、樣本範圍、split、horizon、轉換規則、最小組大小與標籤程式的雜湊組成；任一項改變就重建。
   - dataset 依 (symbol, cutoff) 對位並比對時間戳後讀取，dataloader 管線不變。
4. **Baseline**（模型與 seed 沿用 `configs/baseline.json`）：
   - GBDT：每個 horizon 一個 squared-error 回歸樹模型；去掉無標籤的列。
   - GRU、DLinear、PatchTST：每個 horizon 一個輸出，損失為遮掉無標籤值後的均方誤差。
   - 6 個規則訊號：原始訊號值直接當排序分數。
   - always_buy、zero_return：常數 0，作為沒有排序能力的對照。
5. **評估與選點**：
   - 主要指標：每個（市場、horizon）的日 Rank IC 平均，以原報酬計算，與既有評估相同。
   - 選點分數：三個市場等權、14 個 horizon 平均的 Rank IC，取負號，越小越好。早停與 checkpoint 選擇都用它。
   - 輔助：rank 標籤上的均方誤差與相關係數、各市場與 horizon 的 Rank IC-IR、既有的前後五分位價差診斷。
6. **執行**：標籤在 baseline GPU Pod 開始時建立（有界的多程序：計算報酬、排名、寫 Parquet）。之後的 Kronos 訓練使用同一份標籤。
7. **判讀**：這組 baseline 只建立比較基準，不設門檻。Kronos `a-rank64` 另行登記並與這組 baseline 比較。日期混洗（`*-rank64-datemix`）在 `a-rank64` 之後再評估。

判讀限制：

- US 的樣本占多數；選點採市場等權，台股（TWSE、TPEX）合占三分之二。
- validation 約 116 個交易日，Rank IC 的區間仍寬。

**§13.10 結果（2026-10-21）**：baseline `baseline-5fb3473a…` 全部完成。市場等權 Rank IC（validation → test）：PatchTST 0.057 → 0.035、GRU 0.056 → 0.007、GBDT 0.044 → 0.020、DLinear 0.033 → 0.003、均線交叉 0.020 → 0.043。US 在 test 維持 0.04–0.056；台股除 1 日外在 test 接近 0（GBDT、GRU、PatchTST 的 TWSE 1 日約 0.08）。只有 seed 42；每個市場與 horizon 的日 Rank IC 標準誤約 0.01–0.018。

#### 13.11 Kronos 橫斷面排序模型 `a-rank64`（2026-10-21 於實作與訓練前寫入）

背景：§13.10 的 baseline 顯示，學習型模型在 validation 有排序能力，但台股在 test 只剩 1 日 horizon 維持。本節登記 Kronos 排序模型的設定、指標分組與判定方式。台股 1 日與長天期的分開報告，是看過 §13.10 的 test 結果後才決定的，因此 test 上的判定屬於試行；最終確認以 2026-10-20 之後的前瞻資料為準，屆時另行登記。

1. **模型與訓練**：
   - Preset `a-rank64`：架構與 `a-alpha-point64` 相同（Kronos-base、LoRA 64、combined 數值分支、市場 embedding）。輸出頭每個 horizon 一個排序分數，不乘報酬尺度。
   - 損失：§13.10 的 gauss-rank 標籤上的均方誤差；無標籤的值不計，各 horizon 等權。
   - 資料：A 組（2021-01-01 起），切分不變；rank 標籤與 §13.10 的 baseline 使用同一份 Parquet。
   - 最佳化設定沿用 `a-alpha-point64`（5 個 epoch、相同學習率與排程）。early stopping 與 checkpoint 選擇都用 §13.10 的選點分數（validation 上三個市場等權、14 個 horizon 平均的 Rank IC，取負號）。
   - 只跑 seed 42；是否加跑其他 seed 依第 6 點決定。
2. **評估樣本**：validation 只用於選點；test 用於本節判定；2026-10-20 之後的前瞻資料用於最終確認。
3. **指標與分組**：
   - 基本量：每個（市場、horizon、訊號日）的 Spearman Rank IC，以原報酬（benchmark-relative adjusted log return）計算，與 §13.10 相同。
   - horizon 分組：短（h = 1）、中（h = 2–5）、長（h = 6–14）。
   - 市場分組：台股（TWSE 與 TPEX 等權）、US。
   - 每一格（市場組 × horizon 組）的每日值：先對組內 horizon 平均，台股再對 TWSE、TPEX 平均。這一格的指標是每日值在全部訊號日上的平均。
   - 95% 區間：每日值序列的 circular moving-block bootstrap，重抽 1,000 次、seed 11；區塊長度短組 5、中組 10、長組 28 個交易日。
   - 配對比較：同一訊號日上，Kronos 與比較模型的每日值相減，再用相同方式計算區間。
   - 另報各市場、各 horizon 的值，只作描述。
4. **比較對象**（依 validation 事先固定）：
   - 主要：PatchTST。它在 §13.10 的 baseline 中 validation 選點分數最佳（市場等權 Rank IC 0.057）。
   - 次要描述：GBDT、GRU、DLinear；均線交叉在 test 最佳，但這是事後觀察，只作描述。
5. **判定**（test，seed 42）：
   - H1（台股短天期）：Kronos 台股短組的 95% 區間下界大於 0。
   - H2（台股短天期勝過 baseline）：Kronos 減 PatchTST 的台股短組配對差，95% 區間下界大於 0。
   - H3（台股長天期，獨立判定）：Kronos 台股長組的 95% 區間下界大於 0。
   - 中組與 US 只作描述。區間跨 0 即判定未通過，不另作解讀。
6. **處置**：
   - H1、H2 都通過：台股 1 日排序列為 hermes 的候選訊號；加跑 seed 43、44，並另行登記前瞻確認。
   - H1 通過、H2 未通過：台股 1 日有排序能力，但 Kronos 沒有勝過 PatchTST；候選訊號由兩者在前瞻確認中比較，不預設採用 Kronos。
   - H1 未通過：Kronos 在 test 沒有台股短天期排序能力，回到模型設計。
   - H3 只決定台股長天期能否作為候選，不影響 H1、H2 的處置。
   - 日期混洗（`a-rank64-datemix`）在本節結果之後另行登記。
7. **程序**：
   - 在 RunPod GPU 訓練；Pod 的可用 CPU 至少 12 顆才啟動。
   - 以 `export-signal-predictions` 匯出所選 checkpoint 的 validation、test 逐筆預測；baseline 的逐筆預測從 volume 下載。
   - 在本機 `backtest/` 計分，設定在計分前寫入 `backtest/config/rank_pilot.json`。結果寫成中英雙語報告。

判讀限制：

- test 已在 §13.10 看過；分組方式在看過結果後決定，所以 test 判定只是試行。
- 台股 test 約 103 個訊號日，h 日標籤互相重疊，區塊 bootstrap 只部分處理自相關。
- 只有一個 seed；§13.10 顯示模型間 0.01–0.02 的差距常在雜訊範圍內。
- test 期間（2025-12 到 2026-05）台股的長天期排序對所有學習型模型都失效，可能是期間特性，不一定代表模型缺陷。

**§13.11 結果（2026-10-21）**：H1 通過，台股短組 0.0642 [0.0271, 0.1015]；H2 未通過，Kronos 減 PatchTST −0.0022 [−0.0211, 0.0182]；H3 未通過，台股長組 0.0082 [−0.0440, 0.0637]。依第 6 點：台股 1 日有排序能力，但 Kronos 沒有勝過 PatchTST，前瞻確認時兩者一起比較；不加跑 seed 43、44；台股長天期不列為候選。選出的是第一次 validation 的 checkpoint-003680（約 0.2 epoch）。一致性檢查通過（504 項，最大差距 2.8×10⁻¹⁷）。見[排序模型試行結果報告](rank_pilot_results.md)。

#### 13.12 台股主線的訓練資料與特徵庫（2026-10-22 於資料準備與實作前改寫；取代 `78c257c` 的草稿）

背景：

- **§13.11 的結果**：Kronos `a-rank64` 在第一次 validation（約 0.2 epoch）就達到最佳，市場等權 Rank IC 0.060，之後沒有再改善。test 上台股 1 日 0.064 [0.027, 0.102] 通過，但沒有勝過 PatchTST（配對差 −0.002 [−0.021, 0.018]）；台股 6–14 日 0.008 [−0.044, 0.064] 未通過。Kronos 與 PatchTST 在六格的配對差都跨 0。台股長天期在 test 對所有學習型模型失效（validation 0.043–0.076 → test −0.037 到 0.012），均線交叉則從 0.030 升到 0.063。
- **文獻對「混合訓練 vs 只用本地市場」沒有一致結論，取決於市場大小與模型類型**：Jiang、Kelly、Xiu（JF 2023，26 國日頻 K 線圖 CNN）發現美國訓練的模型直接套用平均勝過本地重訓（Sharpe 0.69 vs 0.29），但增益集中在股票數少於約 300 檔的小市場，600–3,000 檔的市場增益小或為負；Choi、Jiang、Zhang（RAPS 2025，含台灣的月頻公司特徵）發現各市場分開訓練較好；Cakici、Zaremba（JBF 2026，45 國）發現兩者相當；Hellum、Pedersen、Rønn-Nielsen（2023）發現單一全球模型較好。台股清理後每日約 TWSE 600 檔、TPEX 290 檔，屬於「美股資料幫助有限」的那一端。
- **本專案的證據**：Kronos 的 loss 約 83% 來自美股，台股 validation Rank IC 仍與其他模型相當；混合不是目前的瓶頸，但也沒有證據顯示它有幫助。
- **決定（2026-10-22）**：主線改為只用台股、2016 年起，並以 walk-forward 重訓取代一次訓練。原草稿的四組對照（混合／只用台股 × 2021／2016）不再執行：它回答的是資料工程問題，而目標市場是台股，只用台股可把訓練資料縮到混合的約六分之一。美股移出主線。

1. **資料組**：
   - Profile `tw_only`（`twse_official` + `tpex_official`），日期契約 `[2016-01-01, 2026-06-01)`，以既有的 `configure` 與 `cpu-prepare` 建立；資料準備腳本維持凍結，不修改。
   - 清理規則與現有資料組相同。儲存仍依固定切分（train 至 2025-06-01、validation 至 2025-12-01、test 至 2026-06-01）分三個 bar store；本計畫的建模不再使用這個切分（見第 4 點），三個 store 都套用與 validation／test 相同的清理後樣本規則。
   - 控制面新增實驗群組字母 `t`（`tw_only`，2016-01-01 至 2026-06-01）；既有 `a`／`b` 代表美台混合 2021／2016 起，不混用。
   - 預期規模：台股樣本約 163 萬筆（2016 至 2025-05）；評估期每日約 TWSE 600 檔、TPEX 290 檔。
2. **標籤**：與 §13.10 相同：t0 收盤產生訊號、t1 開盤進場、第 h 日收盤（h = 1…14）的相對大盤調整後對數報酬；rank 標籤為同一訊號日、同一市場內的 gauss-rank，少於 20 檔的分組不給標籤。分組只依訊號日與市場，不依切分，因此三個 store 的標籤定義一致。
3. **特徵庫（Alpha158 型，手寫，不用 LLM 生成）**：
   - 清單在任何模型訓練前寫入 `configs/features/alpha_tw_v1.json`，之後不改；要改就是新版本、新識別碼。
   - 內容：沿用 Qlib Alpha158 的算子與視窗家族（K 線形態 9 個；開高低與 VWAP 相對收盤的滯後 0–4；成交量滯後；ROC、MA、STD、BETA、RSQR、RESI、MAX、MIN、QTLU、QTLD、RANK、RSV、IMAX、IMIN、IMXD、CORR、CORD、CNTP、CNTN、CNTD、SUMP、SUMN、SUMD、VMA、VSTD、WVMA、VSUMP、VSUMN、VSUMD × 視窗 5、10、20、30、60），加上台股補充：相對大盤的 5／20／60 日報酬、20 日平均成交值在同日同市場的百分位名次、市場指標（是否為 TPEX）。總數 183 個；資料沒有 VWAP，以典型價格（高＋低＋收）／3 代替。
   - 只用還原後 OHLCV 與大盤計算，只用訊號日（含）之前的資料。正規化在每個 walk-forward 折內以訓練視窗估計 robust z-score，裁剪至 ±3，缺值補 0；不得用折外資料。
   - 儲存方式與 rank 標籤相同：每個 store 一份 Parquet，放在 `features/<識別碼>/`，識別碼由 bar store manifest、特徵清單與特徵程式的雜湊組成。GBDT 與 DL 共用同一份。
4. **Walk-forward**：
   - 折：以日曆季為單位，第一折 2023-Q1（2023-01-02 起），最後一折 2026-Q2（至 2026-06-01 前有完整標籤的訊號日），共 14 折。
   - 第 q 折的訓練視窗為 `[2016-01-01, q 起日 − 15 個交易日)`（15 = 最長 horizon 14 加 1，避免標籤與預測期重疊）；視窗最後 12 個月為該折的 validation（早停與選點），其餘為 train。視窗起點固定（擴展視窗）。每折從頭訓練，不沿用前一折的權重。
   - 每折只用該折視窗內的資料做特徵正規化、早停與選點；該季的預測只來自該折。
   - 評估期分三段報告：P1 = 2023-01-02 至 2025-05-30（從未被任何模型評估過）、P2 = 2025-06-02 至 2026-05（§13.10、§13.11 已看過的 validation 與 test 期間）、P3 = 2026-06-01 之後（前瞻，另行登記）。
5. **程序**：
   1. 在 RunPod CPU 以 `configure --profile tw_only --start 2016-01-01 --end 2026-06-01` 建 selection，再 `cpu-prepare`（開 Pod 前先徵詢使用者）。
   2. 以新的 CPU 工作（經 `runpod_tmux_launch.sh` 的 research 家族）建立三個 store 的 rank 標籤與特徵庫，下載到本機 `artifacts/`。
   3. 之後的模型與回測依 §13.13。

判讀限制：

- 只用台股會失去跨市場的樣本量；若 §13.13 的 DL 在台股上不勝過 LightGBM，不能據此推論混合訓練也不會。
- 特徵庫是手工固定清單，未做因子篩選；篩選會引入另一層選擇偏誤，本計畫不做。
- 台股 2016–2020 的清理規則與現有資料相同，但官方來源可能缺少期間內下市的標的（存活者偏誤，同 §13.4）。

#### 13.13 台股橫斷面排序 pipeline：特徵 LightGBM 基線、跨股票 DL、消融與 Top-k dropout 回測（2026-10-22 於實作前寫入）

背景：目標是一個可作為交易策略基準的台股日頻排序模型，並以業界常用的方式驗證：同一組特徵的 GBDT 基線、walk-forward 重訓、Rank IC／ICIR、含成本的 Top-k dropout 組合。依據：Qlib Alpha360 公開基準上跨股票 DL（HIST 0.067、IGMTF 0.061、GATs 0.060）勝過 LightGBM 0.050，純 Transformer 最差 0.033；MASTER（AAAI 2024）報告 Rank IC 0.076 對 XGBoost 0.050，但第三方重現只有 0.030，可重現性風險要靠消融與多 seed 控制；QuantBench 發現 IC 類損失最好、3 個月滾動重訓最好、不重訓最差；Forecast Collapse（arXiv 2608.14106）指出時序基礎模型的目標都是逐序列，加入橫斷面相關項使 IC 大幅提高；Kronos 論文的台股附錄顯示通用時序基礎模型 zero-shot 的台股 Rank IC 約 0。LLM 迴圈挖 alpha 因成本排除；Kronos 不再是 backbone（§13.11）。

1. **模型**（資料、標籤、特徵、walk-forward 皆依 §13.12）：
   - **B0 `t-lgbm-alpha`（基線，必須先建）**：LightGBM，每個 horizon 一個回歸模型，目標為 gauss-rank 標籤，輸入為特徵庫。超參數在訓練前固定：`num_leaves` 31、`learning_rate` 0.05、`feature_fraction` 0.8、`bagging_fraction` 0.8、`min_data_in_leaf` 200、以該折 validation 的 horizon 等權 Rank IC 早停（50 輪）。seed 42、43、44，報個別與平均集成。若 RunPod CPU 無法安裝 `lightgbm`，改用 scikit-learn `HistGradientBoostingRegressor` 對應參數，並在報告註明。
   - **D0 `t-xs-scratch`（主模型）**：跨股票 transformer。股票內編碼器（128 個交易日的 OHLCV 視窗，patch transformer）與特徵庫向量側輸入 → 同一訊號日、同一市場的全部股票為一個 batch（沿用 `date_market_sampler`）→ 股票間 attention（1–2 層）→ 市場狀態門控（由該日該市場的橫斷面彙總特徵產生）→ 14 個 horizon 的排序分數。損失 = gauss-rank 標籤的均方誤差 + λ ×（1 − 當日橫斷面 Pearson 相關），λ = 1，訓練前固定。模型維度 128、層數 2 + 2，參數量約 1–3M。seed 42、43、44。
   - **D1 可抽換編碼器**（只在 D0 完成後做，每個是對 D0 的單因子更動）：契約為輸入 `(B, L, F)` 原始視窗、輸出 `(B, D_enc)`；正規化在 adapter 內部；凍結模式必須把 embedding 依 (symbol, date) 快取成 Parquet，walk-forward 只重訓跨股票模組。順序：`t-xs-kronos-small`（凍結）→ `t-xs-kronos-base-lora`（只在前者有效時）→ `t-xs-chronos2-small`（凍結，Apache-2.0）。TimesFM 2.5 只在預算剩餘時做。
   - **消融**（對 D0 的單因子更動，seed 42）：A1 關閉股票間 attention；A2 λ = 0（只用均方誤差）；A3 不用特徵庫側輸入；A4 關閉市場狀態門控。
   - 新的 DL 訓練入口不經過 §13.10 的 baseline 套件契約；它的前置檢查改為確認 B0 的 walk-forward manifest 存在。
2. **指標**：
   - 每個（市場、horizon、訊號日）的 Spearman Rank IC，以原報酬計算（同 §13.11）。
   - 格子：台股短（h = 1）、台股持有（h = 5–10，對應數天到兩週的持有方式）、台股長（h = 11–14）；台股 = TWSE 與 TPEX 等權。另報各市場、各 horizon。
   - 每格的每日值平均為 Rank IC；每日值的平均除以標準差為 Rank ICIR；另報每日值為正的比例。
   - 95% 區間：circular moving-block bootstrap，1,000 次、seed 11；區塊長度短 5、持有 20、長 28 個交易日。配對比較在同一訊號日相減後以相同方式計算。
   - 主要判定期間為 P1 ∪ P2（2023-01-02 至 2026-05，約 830 個訊號日）；P1、P2 分別報告。
3. **Top-k dropout 回測**（本機 `backtest/`，設定在計分前寫入 `backtest/config/topk_dropout.json`）：
   - 市場分開：TWSE k = 50、每日換出 5 檔；TPEX k = 30、每日換出 3 檔。每日依前一日收盤的排序分數，在 t+1 開盤賣出持有中排名最後的 n 檔、買入未持有中排名最前的 n 檔；逐部位開盤到開盤記帳（同 §13.5）。
   - 使用的 horizon 分數：h ∈ {1, 5, 10} 中，以 P1 的 TWSE 淨 IR 最高者；選定後固定用於 P2 與前瞻。
   - 成本沿用 `backtest/config/stage1.json` 的主要情境（手續費折扣 0.6、滑價 1 倍、保守週轉）；另報折扣 1.0／0.28 與滑價 0／2 倍。
   - 報告：年化淨超額（相對同市場等權全體）、IR、週轉率、最大回撤；區間為月區塊 bootstrap，1,000 次、seed 11。
4. **判定**（事先固定；B0、D0 均為三 seed 平均集成）：
   - **H-A（DL 勝過基線）**：D0 − B0 的配對 Rank IC 差，在台股短格與台股持有格的 95% 區間下界都大於 0，判定期間 P1 ∪ P2。
   - **H-B（可交易）**：TWSE Top-k dropout 的年化淨超額，主要成本情境下 95% 區間下界大於 0，且 D0 的 IR 不低於 B0。
   - **H-C（前瞻確認）**：2026-06-01 之後的資料，以既有前瞻管線另行登記判準。
   - **預訓練編碼器**：D1 各模型對 D0 的配對差，在台股短格或持有格的區間下界大於 0 才算有效；其餘為無效。
   - **消融**只作描述，用來判斷 D0 的增益來自哪個部件；不設門檻。
5. **處置**：
   - H-A 兩格都未通過：B0 成為策略基準（它本身仍須通過 H-B 才算可交易）；DL 只作研究，不投入更多 GPU。
   - H-A 至少一格通過：D0 為候選模型，進入 D1 與 H-B。
   - H-B 未通過：排序能力存在但不可交易，回到策略層（換倉頻率、持有天期、成本）而非模型層；H-C 不執行。
   - H-B 通過：登記 H-C，以 P3 確認。
6. **程序與順序**（GPU 放最後）：
   1. §13.12 的資料組、標籤與特徵庫（CPU）。
   2. B0 walk-forward（RunPod CPU 或本機，worker 數有上限）。
   3. D0 三 seed（RunPod GPU，RTX 4090 級；14 折 × 3 seed，估計 10–15 小時、約 5–10 美元；開 Pod 前徵詢使用者）。
   4. 消融 A1–A4（seed 42，估計 8–10 小時）。
   5. D1 編碼器（凍結模式先算 embedding 快取，之後每折只訓練跨股票模組）。
   6. Top-k dropout 回測（本機）。
   7. H-C 登記與前瞻確認。
   - 每個模型的逐筆 walk-forward 預測保存為 Parquet（P1 ∪ P2 全部訊號日），供本機計分與回測；結果寫成中英雙語報告。

7. **實作登記（2026-10-23，訓練前固定，未看任何 D0 結果）**：
   - D0 訓練參數：AdamW lr 5e-4、weight decay 1e-4、最多 20 epoch、以 fold validation 的
     每日 Rank IC（14 個 horizon 等權、兩市場等權）早停 patience 4、gradient clip 1.0、
     CUDA 上 bf16 autocast；每個 batch 為同一 signal date 與市場的全部股票；特徵側輸入在
     batch 內做橫斷面 z-score（裁剪 ±5）；視窗正規化為相對最後收盤的 log 價格與相對中位數的
     log 成交量。編碼器 patch 8、d 128、2 層；inter-stock attention 2 層、4 heads、ff 256、
     dropout 0.1。
   - D1 編碼器：`t-xs-kronos-small` 使用 `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`
     的最後一個位置 hidden state；`t-xs-chronos2-small` 使用 `autogluon/chronos-2-small`
     （28M，Apache-2.0）的 `embed` 輸出對 variates 與 patches 取平均，五個 OHLCV 通道作為同一
     序列的 variates。兩者都凍結，embedding 依 (symbol, cutoff) 快取一次。
   - 配對比較：D − B0 以共同 signal date 的 cell 序列相減，circular moving-block bootstrap
     （1,000 次、seed 11、block 5／20／28）；H-A 以 P1 ∪ P2 的 short 與 holding cell 判定。
   - 執行入口：`runpod_workflow.sh xs-train --experiment <name>`，每個實驗一台 GPU Pod。

判讀限制：

- P2 已在 §13.10、§13.11 看過，P1 是本計畫第一次評估，但資料準備規則與模型設計都是在看過 P2 的彙總結果後決定的；真正的確認只有 P3。
- 消融與 D1 只有 seed 42；§13.10 顯示模型間 0.01–0.02 的差距常在雜訊範圍內。
- 第三方重現 MASTER 只得到原文不到一半的 Rank IC；本計畫的 D0 是依該架構思路自行實作，不保證達到原文數字。
- TPEX 流動性低，台股 ML 選股的 alpha 常受可投資性限制（Chang、Chiu，SSRN 2026）；回測未模擬成交量上限、整股、最低手續費與股利稅。
- 若 `lightgbm` 無法安裝而改用 scikit-learn，基線名稱仍為 B0，但與 Qlib 公開基準的可比性下降。

**§13.13 B0 結果（2026-10-23）**：B0 `t-lgbm-alpha`（`launch-20261022T234922Z-22048038`，14 折 × 14 horizon × 3 seed = 588 個模型）在 P1 ∪ P2 的 Rank IC：短格 0.1178 [0.1091, 0.1266]、持有格 0.0859 [0.0650, 0.1061]、長格 0.0834 [0.0541, 0.1129]；P2 單獨看持有格 0.0343 [−0.0004, 0.0726]、長格 0.0107 [−0.0372, 0.0600]。第 3 項的 Top-k dropout 套用於 B0：選定 h = 10，TWSE P1 ∪ P2 主情境年化淨超額 −6.3% [−19.0%, +7.1%]、IR −0.58，H-B 未通過。依第 5 點：B0 為策略基準但不算可交易，H-C 對 B0 不執行，策略層的改動須另行登記。D0（H-A）照第 6 項第 3 步執行，結果另記。見 [B0 基線與回測結果報告](lgbm_baseline_results.md)。

**§13.13 D0 結果（2026-10-23）**：D0 `t-xs-scratch`（`launch-20261023T015746Z-224647797`，14 折 × 3 seed = 42 個 fit，RTX PRO 4500，4.8 小時）在 P1 ∪ P2 的 Rank IC：短格 0.1221 [0.1129, 0.1311]、持有格 0.0921 [0.0739, 0.1096]、長格 0.0893 [0.0664, 0.1121]；三個 seed 短格 0.1185／0.1181／0.1209。D0 − B0 配對差：短格 +0.0043 [+0.0015, +0.0071]（通過）、持有格 +0.0062 [−0.0012, +0.0136]（未通過），**H-A 一格通過**，依第 5 點 D0 為候選模型，進入消融與 D1。D0 的 Top-k dropout（第 3 項）：選定 h = 10，TWSE P1 ∪ P2 年化淨超額 −7.3% [−17.9%, +3.5%]、IR −0.79（低於 B0 的 −0.58），**H-B 未通過**。GPU 使用率僅 13–17%（同日分組取數為瓶頸）；該 Pod 沒有時脈監測紀錄。消融 A1–A4 與 D1 的結果另記。

**§13.13 消融 A1 結果（2026-10-23）**：`t-xs-a1-no-cross`（關閉股票間 attention，seed 42，`launch-20261023T075624Z-2596926448`，14 個 fit，70 分鐘）P1 ∪ P2 短格 0.1195 [0.1098, 0.1284]、持有格 0.0879、長格 0.0832；對 B0 的配對差短格 +0.0017 [−0.0014, +0.0047]（不顯著）。D0 單一 seed 42 的短格為 0.1185，與 A1 相當：D0 集成對 B0 的 +0.0043 來自三個 seed 的集成，不是股票間 attention。A2–A4 與 D1 依使用者 2026-10-23 的決定延後，待 §13.16 的標籤結果再定。

#### 13.14 台股策略層：成本感知的多頭 band 組合與台指期疊加（2026-10-23 於實作與計分前寫入）

背景：§13.13 的 B0 與 D0 都有排序能力（P1 ∪ P2 短格 Rank IC 0.118／0.122），但登記的 Top-k dropout（每日換 5 檔於 50 檔）在台股成本下 H-B 未通過：每日換手約 10% 的成本拖累約 17%／年，高於 P1 的毛超額；P2 的 h = 5–14 Rank IC 降到 0.02–0.05，h = 1 仍有 0.11。依 §13.13 第 5 點回到策略層。本節登記業界慣用的成本緩解做法在台股的版本：買進／持有門檻（Novy-Marx 與 Velikov 2016 的最有效單一方法）、訊號平滑、換手上限、流動性宇宙、成本感知（Jensen、Kelly、Malamud、Pedersen 2024），以及用台指期（期交稅十萬分之二）取代現股做 beta 對沖與閒置現金的市場曝險。台指期不進任何模型訓練，只在回測層使用；資料來自期交所每日行情下載（官方），不用第三方 API。

1. **輸入**：B0 `t-lgbm-alpha` 與 D0 `t-xs-scratch` 的 walk-forward 預測（三 seed 集成；D1 若有效則加入）；資料集 raw `market.parquet`（未調整開收盤價與成交量、調整後收盤價）；期交所 TX 每日行情 2022-12 至 2026-06（`twbacktest.taifex download-contract`），連續近月序列在到期月第三個星期三的開盤換月，開盤價以比率串接。
2. **宇宙**：只含股票（排除 ETF 與指數）；市場分開，TWSE 為判定市場，TPEX 只作描述。流動性門檻：訊號日前 20 個交易日的成交金額（收盤價 × 成交量）中位數 TWSE ≥ 新台幣 3,000 萬、TPEX ≥ 1,000 萬，且收盤價 ≥ 10 元；門檻在訓練前固定，不選。
3. **訊號**：使用單一 horizon 的集成分數（h 在第 6 項的格子中選）；每日先在市場內轉成百分位名次，再對每檔做 EMA 平滑，半衰期 3 個交易日（固定），中斷超過 5 個交易日即重設。
4. **組合規則（band）**：進前 K 名才買、掉出前 M 名才賣（M = m × K）；每日依前一收盤的平滑分數檢查，於次一開盤成交；每日新買不超過 ⌈K／10⌉ 檔；資金分成 K 個等權位置，逐部位開盤到開盤記帳（§13.5 的引擎，`positions.simulate` 的 buffer 規則加上每日換入上限）；漲停開盤不買、跌停開盤不賣（同 §13.13）。主要分析為分數位置（相當於大資金或可用零股）；另以資本 1,000 萬與 3,000 萬元做整張可行性敏感度：位置金額買不到一張（1,000 股）的股票留在現金並計數；最低手續費 20 元在這兩個資本下占比低於 0.01%，只報不模擬。
5. **成本**：現股沿用 `backtest/config/stage1.json` 的主要情境（手續費 0.1425% × 折扣 0.6、證交稅 0.3%、滑價 TWSE 0.1%／TPEX 0.2%），另報折扣 1.0／0.28 與滑價 0／2 倍。台指期每邊：期交稅 0.002% 契約金額、手續費每口 50 元、滑價 1 點（200 元），換月日兩邊各一次。
6. **台指期疊加**（都用 TX 近月連續序列，部位每日在開盤依前一日持股市值調整）：
   - **對沖版**：放空名目金額 = β × 持股市值，β = 1（固定）；保證金準備金 = 名目金額的 15%（原始保證金約 5% 加上 10% 的逐日結算緩衝），從總資本扣除，現股只用剩餘資本；準備金低於名目金額 10% 時，於次一開盤按比例賣出現股補足並計一次追繳。對沖版的基準為 0。
   - **現金替代版**：多頭組合中未投入股票的現金（空位、被擋的買單、整張不足）持有等額的 TX 多頭，保證金自該現金撥出。基準同多頭版。
   - **TX 多頭參考**：只持有近月 TX 多頭（含換月成本），對照 TAIEX 價格指數，用來報告基差與換月的拖累。
7. **選擇與判定**：
   - 格子：h ∈ {5, 10}、K ∈ {30, 50}、m ∈ {2, 3}，共 8 組；對每個模型以 TWSE P1 主要情境的多頭版淨超額（相對同市場流動性宇宙等權）IR 最高者為該模型的選定參數，選定後固定用於 P2、P1 ∪ P2 與前瞻。
   - **H-D（策略層可交易）**：選定參數的 TWSE 多頭版，P1 ∪ P2 主要情境年化淨超額（相對等權宇宙）的 95% 區間下界大於 0（月區塊 bootstrap，1,000 次、seed 11）。B0 與 D0 分別判定。
   - **H-D-hedged**：同參數的對沖版，P1 ∪ P2 年化淨報酬（絕對）的 95% 區間下界大於 0。
   - 前瞻候選模型：通過 H-D 者之中 TWSE P1 ∪ P2 淨超額 IR 較高的模型；兩者都未通過則沒有候選。
8. **報告（固定欄位）**：除了淨超額與區間、IR，一律同時列出年化淨報酬、CAGR、Sharpe、最大回撤、換手、成本拖累、對 TAIEX 價格指數的超額與對等權宇宙的超額，以及整張可行性下被擋的買單比例與追繳次數。
9. **處置**：H-D 通過 → 以候選模型登記 H-C（2026-06-01 之後的 P3）；H-D 未通過 → 多頭現股持有在台股成本下無法把排序能力變成報酬，下一個候選是股票期貨標的內的多空（§13.9 的可取性）或 h = 1 的當沖路線，兩者都要另行登記。
10. **程序**：設定檔 `backtest/config/tw_strategy.json` 在計分前寫入並 commit；先以合成資料的測試驗證 band、換手上限、整張可行性、對沖與保證金會計（含追繳）；再對 B0、D0 計分；結果寫成中英雙語報告。

判讀限制：

- 8 組參數在 P1 選一組，P1 本身帶有選擇偏誤；P1 ∪ P2 的判定含 P1，真正的確認仍是 P3。
- 流動性門檻與 β = 1 是事前固定的合理值，不是最佳化結果；單一 β 忽略個股 beta 的差異。
- 保證金模型是簡化版（固定比例的準備金與追繳規則），未用期交所逐日公告的保證金表；未模擬期貨的流動性與盤後時段。
- 未模擬成交量參與率對大資金的限制、股利稅與借券；整張可行性只在兩個資本水準下報告。

**§13.14 結果（2026-10-23）**：B0 與 D0 都選出 h = 10、K = 50、m = 3（TWSE P1 多頭版淨超額 IR 0.42／0.52）。每日換手 2.0%／2.5%、成本拖累 3.4%／4.8%／年；P1 淨超額 +5.4%／+5.5%（區間跨 0），P2 毛超額 −27%／−31%，P1 ∪ P2 年化淨超額 B0 −4.7% [−19.7%, +10.5%]、D0 −5.4% [−18.2%, +7.7%]，**H-D 未通過**；對沖版 P1 ∪ P2 年化淨報酬 −12.5%／−13.3%、28／27 次追繳，**H-D-hedged 未通過**。絕對報酬：多頭版年化淨報酬 +23.8%／+23.1%，Sharpe 1.72／1.50，但 TAIEX 報酬指數同期 +41%／年（本節所稱「TAIEX 指數」在資料集中為官方報酬指數，含息；第 8 項的「價格指數」應讀為報酬指數）。資本情境（分數位置、1,000 萬整張／零股、3,000 萬、1 億、3 億、10 億；零股加 0.1% 滑價、最低手續費、5% 參與率上限）：1,000 萬整張少 1.5–2.7 個百分點，零股補回；3 億元因參與率上限集中於大型流動股，P1 ∪ P2 淨超額縮到 −2.9%／−1.3% 但區間仍跨 0；沒有資本情境通過 H-D。前瞻候選模型：無。依第 9 點，下一步為另行登記股票期貨多空或 h = 1 當沖路線。見 [台股策略層結果報告](strategy_layer_results.md)。

#### 13.15 股票期貨空方當沖：h = 1 訊號的可交易性（2026-10-23 於實作與計分前寫入）

背景：§13.14 的多頭持有在所有資本情境下都未通過 H-D；股票期貨宇宙診斷（策略層報告第 7 節）顯示 h = 1 的排序能力在有股票期貨的標的中保留八到九成（流動段 Rank IC 0.10–0.11，P2 0.08–0.09），而 h = 1 的毛價差幾乎全部來自空方（後十分位次日盤中 −0.36% 到 −0.45%，前十分位只有 +0.04% 到 +0.07%，宇宙盤中平均為負）。現股做空受券源、平盤下限制與當沖稅限制；股票期貨期交稅十萬分之二、不需券源。本節登記以期交所官方期貨價格回測的空方當沖。

1. **資料**：期交所每日行情下載（全契約，`twbacktest.taifex download`，2022-12 至 2026-06）：每個股票期貨契約每日的開盤價、結算價、成交量與一般／盤後時段；標的對應依期交所股票期貨標的清單（契約代碼 → 股票代號、每口股數 2,000 或 100）。台指期近月序列同 §13.14。訊號為 B0、D0 的 h = 1 集成分數。
2. **宇宙**：訊號日 t 當天，近月股票期貨契約在一般時段有成交、且近月 20 日平均成交量 ≥ 100 口（標準口換算）的標的；優先用標準契約（2,000 股），只有小型契約（100 股）者以小型契約計價並計數。契約在到期日當天改用次月。訊號日沒有次一交易日期貨價格的標的跳過並計數。
3. **部位與價格**：依前一收盤的 h = 1 分數在宇宙內排名，放空後 q 分位（q ∈ {10%, 20%}，檔數 = ⌈n × q⌉）；次一交易日 t+1 在期貨開盤價（8:45）放空、以當日結算價平倉（結算價取自 13:30 前成交，為官方日結算 mark），不留倉。版本：**空方版**（判定用）；**多空版**（另放空等額名目的… 另做多前 q 分位等額名目）；**對沖版**（空方版加等額名目的台指期多頭，開盤到結算價）。
4. **資本與口數**：資本情境與 §13.14 相同金額（分數位置、1,000 萬、3,000 萬、1 億、3 億、10 億元；期貨無零股情境）。目標總名目 = L × 資本，L = 1（固定）；每檔名目 = 總名目／檔數；口數 = ⌊每檔名目／（開盤價 × 每口股數）⌋，不足一口者跳過並計數；每檔口數 ≤ 該契約當日成交量 × 5%。保證金以期交所最高級距原始保證金 20.25% 計（維持 15.53%），報告保證金使用率；L × 20.25% > 100% 視為不可行。
5. **成本**（每邊）：期交稅 0.002% × 名目；手續費每口 30 元（另報 60 元）；滑價 1 tick（另報 2 tick），tick 依標的價格級距（< 10 元 0.01、10–50 元 0.05、50–100 元 0.1、100–500 元 0.5、500–1,000 元 1、≥ 1,000 元 5）× 每口股數。主要情境：1 tick、30 元。台指期每邊：期交稅 0.002%、每口 50 元、1 點。
6. **指標與區間**：每日報酬 = 當日損益／資本（分數位置版為名目加權報酬 × L）；毛（無成本）與淨；年化報酬（× 252）、CAGR、Sharpe、最大回撤、勝率、成本拖累、平均檔數與口數、因口數或參與率被擋的比例、保證金使用率。區間：每日淨報酬的 circular block bootstrap，區塊 5、1,000 次、seed 11。期間 P1、P2、P1 ∪ P2（依訊號日）。
7. **選擇與判定**：
   - q 在 TWSE 加 TPEX 的合併宇宙（股票期貨宇宙不分市場）、P1、空方版、分數位置、主要成本下以淨 Sharpe 最高者選定，B0 與 D0 分別選。
   - **H-E（空方當沖可交易）**：選定 q 的空方版，P1 ∪ P2 年化淨報酬（絕對）的 95% 區間下界 > 0，且 P2 單獨的年化淨報酬 > 0。B0 與 D0 分別判定。
   - 多空版與對沖版只作描述。資本情境只作描述；但若 1,000 萬元情境因口數不足而被擋的比例超過 50%，在報告中標記該資本不可行。
   - 前瞻候選模型：通過 H-E 者之中 P1 ∪ P2 淨 Sharpe 較高者。
8. **報告（固定欄位）**：每個版本、資本情境、成本情境與期間都列出年化毛報酬、年化淨報酬與區間、CAGR、Sharpe、最大回撤、勝率、成本拖累、平均檔數、平均口數、被擋比例、保證金使用率；另報期貨開盤價對現股開盤價、結算價對現股收盤價的日差標準差（時點噪音）。
9. **處置**：H-E 通過 → 以候選模型登記 H-C（2026-06-01 之後的 P3 前瞻，含實際期貨價格）；H-E 未通過 → h = 1 訊號在現有工具與成本下也不可交易，模型與策略層都回到研究狀態，下一步改為訊號本身（殘差化標籤、多 horizon 融合）的登記實驗。
10. **程序**：設定檔 `backtest/config/futures_daytrade.json` 在計分前寫入並 commit；先以合成資料測試口數、參與率、成本與保證金；再對 B0、D0 計分；結果寫成中英雙語報告。

判讀限制：

- 契約代碼以 2026-10 的標的清單對應，期間內下市的股票期貨無法對應（存活者偏誤，約 38 個代碼）。
- 期貨 8:45 開盤價是可成交的價格，但 8:45–9:00 的流動性低於現股開盤後；滑價以 tick 計，未模擬開盤成交量不足。
- 結算價是官方 mark，實際平倉需在 13:30 前後成交，可能偏離結算價。
- 保證金以最高級距一律計算，實際逐檔級距較低；未模擬盤中追繳。
- q 在 P1 選擇，P1 ∪ P2 判定含 P1；P3 才是確認。

**§13.15 結果（2026-10-23）**：兩個模型都選 q = 10%（每日約 13 檔）。以期交所官方期貨開盤價與結算價計算，空方版 P1 ∪ P2 年化毛報酬 B0 +61.5%、D0 +64.5%（P1、P2 皆為正），但主要成本下年化淨報酬 B0 −63.4% [−91.5%, −36.0%]、D0 −60.5% [−86.9%, −33.7%]，P2 亦為負，**H-E 未通過**。成本拖累約 125%／年，幾乎全為 tick 滑價（每邊約 0.23%）；損益兩平的單邊成本約 0.12%。多空版與對沖版更差或未救回；六個資本情境皆為負。依第 9 點：h = 1 訊號在現有工具與成本下不可交易，模型與策略層回到研究狀態，下一步為 §13.16。見 [股票期貨空方當沖結果報告](futures_daytrade_results.md)。

#### 13.16 殘差化標籤與多 horizon 融合：多日排序在風格集中期的穩定性（2026-10-23 於實作前寫入）

背景：§13.13 的 B0 與 D0 在 P1 的 Rank IC 由 h = 1 的 0.13 緩降到 h = 10 的 0.11，但在 P2（2025-06 起，TAIEX 一年漲一倍、等權宇宙只漲三成的大型股集中行情）h = 5–14 掉到 0.02–0.05，h = 1 仍有 0.11；§13.14 的多頭策略與股票期貨宇宙內的 h = 10 都因此失效。文獻與業界的對應做法（策略層報告與本節登記前的討論）：因子半衰期與再平衡匹配（Flint、Vermaak）、快慢訊號分權（Gârleanu、Pedersen 2013）、多尺度訊號融合（Han、Zhou、Zhu 2016；Jiang、Kelly、Xiu 2023）、以及把報酬對市場、市值、產業取殘差當標籤讓模型只學相對表現（風險模型中性化的標準做法）。現行標籤只去掉市場方向，沒有去掉市值與產業暴露。本節登記：換標籤、不換模型，先用最便宜的 B0 驗證。

1. **資料**：訓練資料、特徵庫、walk-forward 切法與 §13.12、§13.13 完全相同。新增一個公司參考表 `configs/reference/tw_company_reference_20261022.csv`，來自證交所 OpenAPI `opendata/t187ap03_L`（上市）與櫃買 OpenAPI `mopsfin_t187ap03_O`（上櫃）2026-10-22 的快照：股票代號、市場、產業別兩碼、已發行普通股數。覆蓋 2026-05 仍在交易的標的 99.2%（上櫃）與 99.8%（上市），全部樣本（含已下市）90% 與 93%。
2. **殘差化標籤**（新的標籤庫 schema `cross-sectional-gauss-rank-residual-v1`，identity 含殘差化規格與參考表 digest；原標籤庫不變）：每個 (訊號日, 市場) 群組內，對每個 horizon 的原始 alpha 目標（已相對市場指數）做橫斷面最小平方回歸，解釋變數為截距、市值代理、產業虛擬變數；殘差再做 gauss-rank。群組內樣本不足 `min_group_size` 或某產業只有一檔時，該產業併入「其他」。
   - **變體 A（主要）**：市值代理 = 訊號日前 20 個交易日成交金額中位數的對數（逐日可得、無前視）；產業 = 參考表兩碼產業別，缺值為「未知」一類。
   - **變體 B（敏感度）**：市值代理 = log（訊號日原始收盤價 × 參考表已發行股數）；參考表缺值的列改用變體 A 的代理。股數為快照，歷史上有股票股利或減資的公司會有偏差，報告中註明。
3. **模型**：B0 `t-lgbm-alpha` 的設定、超參數、seed 不變，分別在變體 A、B 的標籤上重訓（RunPod CPU）；標籤為新的 `--label-variant`，特徵庫沿用。D0 與 D1 只在 B0 通過第 5 項後才在新標籤上重訓（D1 的 embedding 快取與標籤無關，不重算）。
4. **指標**：每個（市場、horizon、訊號日）的 Spearman Rank IC，同時對兩個目標計算：（a）**殘差目標**（模型所預測的量，衡量排序能力）；（b）**原始 alpha 目標**（多頭組合實際賺到的量，衡量可用性）。cell、期間、bootstrap 與 §13.13 相同。另報新舊 B0 在相同訊號日的配對差。
5. **判定（H-F，殘差化穩住多日排序）**：變體 A 的 B0，（i）P2 持有格（h = 5–10）對殘差目標的 Rank IC 95% 區間下界 > 0，且（ii）P1 ∪ P2 短格對原始 alpha 目標的 Rank IC 不低於舊 B0 減 0.01（配對差區間下界 > −0.01）。兩者都成立才通過。變體 B 只作描述，除非變體 A 未通過而變體 B 通過（則以變體 B 為主，並在報告中註明這是第二次機會）。
6. **多 horizon 融合**（策略層，描述性）：`tw_strategy` 的訊號改為 h = 1–14 分數的衰減加權平均（權重 ∝ exp(−h／τ)，τ ∈ {3, 7} 為新的格子維度，與單一 h 並列），其餘規則同 §13.14；對新標籤的 B0 與舊 B0 都跑。只作描述，不改 §13.14 的 H-D 判定規則。
7. **處置**：H-F 通過 → D0 在新標籤上重訓（三 seed），之後消融與 D1 在新標籤上一併決定；§13.14 與 §13.15 以新模型重跑，判定沿用各節規則。H-F 未通過 → 多日排序在風格集中期的失效不是標籤暴露造成的，多日 horizon 退出主線，模型只作為 h = 1 快訊號使用（§13.15）。
8. **程序**：參考表以腳本從官方 OpenAPI 抓取並 commit（含快照日期與 sha256）；標籤變體規格 `configs/labels/residual_v1.json` 在實作前寫入；先以合成資料測試回歸殘差、產業合併與 identity；再在 RunPod CPU 建標籤庫、訓 B0 變體 A、B；結果寫成中英雙語報告。

判讀限制：

- 產業與股數是 2026-10-22 的快照；已下市的 7–10% 樣本沒有產業（歸入未知）與股數（用變體 A 代理），有存活者偏誤。
- 殘差化移除的是線性的市值與產業暴露；非線性或其他風格（動能、波動）暴露仍在目標裡。
- 殘差目標上的 Rank IC 不等於可交易：多頭組合仍承受市值與產業因子報酬，§13.14 的判定才算可交易性。
- P2 已經看過；H-F 是在看過失效之後才設計的修補，真正的確認是 P3。

9. **D0 在新標籤上的重訓與 seed 穩定度（2026-10-23 於 D0 重訓前補登記）**：D0 `t-xs-scratch` 以 `--label-variant` 在 H-F 通過的變體標籤上重訓，設定、超參數與 seed（42、43、44）與 §13.13 相同；配對基線為同一變體的 B0；Rank IC 同時對殘差目標與原始 alpha 目標計算，判定沿用 §13.13 H-A（原始 alpha 目標的配對差）。另記 **seed 穩定度**：對每個 cell 與期間列出三個單 seed 的 Rank IC、極差、與三 seed 集成的增益（集成減單 seed 平均）。事前規則：P1 ∪ P2 的短格與持有格同時滿足「seed 極差 < 集成 95% 區間半寬的一半」且「集成增益 < 半寬的一半」時，單 seed 已落在集成自身的抽樣誤差內，D1 可只跑 seed 42；否則 D1 維持三 seed。舊標籤的 D0 已有的數字：短格三 seed 0.1185／0.1181／0.1209（極差 0.0028，集成 0.1221，增益 0.0029，半寬 0.0091）、持有格 0.0882／0.0880／0.0906（極差 0.0026，集成 0.0921，增益 0.0032，半寬 0.0179），兩格都符合；新標籤上以實際結果為準。

**§13.16 變體 A 結果（2026-10-23）**：B0 在 `residual_v1a` 標籤上重訓（`launch-20261023T103805Z-2637116736`，588 個模型，`t-lgbm-alpha-residual_v1a`）。對殘差目標：短格 P1 ∪ P2 0.1008 [0.0954, 0.1061]（ICIR 1.19）、持有格 P2 **0.0646 [0.0457, 0.0856]**（條件 (i) 通過）、長格 P2 0.0549 [0.0324, 0.0763]。對原始 alpha 目標：短格 P1 ∪ P2 0.1175，與舊 B0 的配對差 −0.0002 [−0.0013, +0.0009]（條件 (ii) 通過）；持有格 P2 0.0434 [0.0103, 0.0794]，對舊 B0 的配對差 +0.0090 [+0.0055, +0.0125]；長格 P2 0.0239 [−0.0216, 0.0692]（舊 B0 0.0107）。**H-F 通過。** 殘差化沒有傷到 h = 1，P2 的多日排序對殘差目標回到 0.05–0.06 且區間不跨 0；對原始 alpha 的改善較小（持有格 +0.009），因為多頭實現報酬仍含市值與產業因子。依第 7 點：D0 在新標籤上重訓，§13.14、§13.15 以新模型重跑。變體 B 另記。

**§13.16 變體 B 結果（2026-10-23，敏感度）**：B0 在 `residual_v1b` 標籤上重訓（`launch-20261023T130541Z-2860518787`，588 個模型，`t-lgbm-alpha-residual_v1b`）。對殘差目標：短格 P1 ∪ P2 0.0982 [0.0923, 0.1037]、持有格 P2 0.0483 [0.0290, 0.0710]、長格 P2 0.0346 [0.0111, 0.0580]。對原始 alpha 目標：短格 P1 ∪ P2 0.1136，與舊 B0 的配對差 −0.0042 [−0.0056, −0.0026]；持有格 P2 0.0277 [−0.0061, 0.0637]，配對差 **−0.0066 [−0.0102, −0.0029]**（比舊 B0 差）；長格 P2 0.0023。H-F 兩個條件形式上都成立（(i) 0.0290 > 0；(ii) −0.0056 > −0.01），但每一格都劣於變體 A：殘差目標持有格 P2 低 0.016，原始 alpha 短格低 0.004、持有格 P2 低 0.016。解讀：以快照股數 × 當日原始收盤價當市值代理，股數是 2026-10 的快照、價格卻是逐日的，等於把價格水準（與動能、拆股歷史糾纏）當成規模因子殘差化掉，反而移走部分可預測的部分；成交金額中位數才是逐日一致的規模代理。依第 5 點，變體 A 為主，變體 B 只作描述，不進入 D0／D1。

**§13.16 D0 結果（2026-10-23，`residual_v1a` 標籤，`launch-20261023T162203Z-877225215`，三 seed，42 個 fit，平均 8.5 epoch，配對基線 `t-lgbm-alpha-residual_v1a`）**。對原始 alpha 目標（P1 ∪ P2）：短格 0.1212 [0.1119, 0.1299]、持有格 0.0925 [0.0751, 0.1092]、長格 0.0914 [0.0694, 0.1138]；P2 持有格 0.0528 [0.0218, 0.0860]、長格 0.0396 [0.0037, 0.0737]（舊標籤 D0 分別為 0.0472 與 0.0360）。對殘差目標：短格 0.1035、持有格 0.0866、長格 0.0869；P2 持有格 **0.0676 [0.0465, 0.0899]**、長格 0.0620 [0.0377, 0.0858]。與同標籤 B0 的配對差（原始 alpha）：短格 **+0.0037 [+0.0011, +0.0065]**（通過）、持有格 +0.0037 [−0.0033, +0.0106]（未通過）、長格 +0.0038 [−0.0078, +0.0162]；對殘差目標：短格 +0.0027 [+0.0004, +0.0049]、持有格 +0.0002 [−0.0043, +0.0046]。**H-A 與舊標籤相同：只有短格通過**（舊標籤 +0.0043）；跨股票 DL 對特徵 GBDT 的增量仍只在 h = 1，且幅度約 0.003–0.004。P2 多日排序的改善來自標籤（B0 與 D0 在殘差目標上幾乎相同：持有格 0.0646 對 0.0676），不是來自模型。**Seed 穩定度**（P1 ∪ P2）：短格三 seed 0.1171／0.1168／0.1191，極差 0.0024，集成增益 0.0036，半寬 0.0090；持有格 0.0906／0.0846／0.0921，極差 0.0075，增益 0.0034，半寬 0.0170；長格極差 0.0083，增益 0.0035，半寬 0.0222。三格皆符合第 9 點規則 → **D1 只跑 seed 42**，D1 對 D0 的比較以 `scripts/compare_xs_runs.py` 在相同訊號日配對（D0 用三 seed 集成；單 seed D1 若落後約 0.003 以內屬集成增益範圍，報告時並列 D0 單 seed 平均）。執行註記：同一張 RTX PRO 4500 上，42 個 fit 合計 32,676 秒（舊標籤 17,264 秒），GPU 利用率 7–9%、時脈正常，為主機 CPU 開銷所致，不影響結果。

**§13.16 D1 結果（2026-10-24，`t-xs-kronos-small`，凍結 Kronos-small 編碼器，`residual_v1a` 標籤，`launch-20261024T074202Z-2401818372`，依第 9 點只跑 seed 42，14 個 fit，平均 8.0 epoch，跨股票模組 438,670 參數，embedding 維度 512；權重 `NeoQuasar/Kronos-small` 901c26c、`Kronos-Tokenizer-base` 0e01173）**。對原始 alpha（P1 ∪ P2）：短格 0.1172 [0.1079, 0.1260]、持有格 0.0948 [0.0767, 0.1121]、長格 0.0925 [0.0700, 0.1140]；P2 持有格 0.0549、長格 0.0444。對殘差目標：短格 0.1002、持有格 0.0893、長格 0.0887；P2 持有格 0.0710 [0.0480, 0.0950]、長格 0.0683 [0.0420, 0.0953]。**與同標籤 B0 配對（原始 alpha）**：短格 −0.0003 [−0.0037, +0.0030]、持有格 +0.0060 [−0.0017, +0.0137]、長格 +0.0049 [−0.0068, +0.0170] → **H-A 兩格皆未通過**；殘差目標上只有長格 P2 +0.0133 [+0.0022, +0.0287] 的區間不跨 0（九個格期組合中的一個，不作為證據）。**與 D0 三 seed 集成配對**（`scripts/compare_xs_runs.py`，原始 alpha）：短格 **−0.0040 [−0.0069, −0.0014]**、持有格 +0.0023 [−0.0018, +0.0065]、長格 +0.0011 [−0.0044, +0.0074]；殘差目標：短格 −0.0033 [−0.0055, −0.0013]、持有格 +0.0027 [−0.0003, +0.0057]。與 D0 **單 seed 42** 並列：短格 0.1172 對 0.1171、持有格 0.0948 對 0.0906、長格 0.0925 對 0.0894，短格差距即為 D0 的集成增益（0.0036）。**判讀**：凍結的 Kronos-small 編碼器沒有帶來 scratch patch encoder 以外的資訊；D0 對 B0 僅有的 h = 1 優勢（+0.0037）在 D1 消失，與 §13.11（Kronos ≈ PatchTST）、§13.13 A1（跨股票 attention 無增量，D0 的優勢是集成）一致。依 §13.13 D1 順序，`t-xs-kronos-base-lora` 以「前者有效」為前提，不執行。執行註記：RTX 5090，embedding 階段 GPU 利用率 92–100%、功率達 600 W 上限、時脈 2745–2820 MHz；訓練階段 25–33%；14 個 fit 合計 1,500 秒。

**§13.16 模型線結論（2026-10-24）**：三個模型在殘差目標上的多日排序能力相同（P2 持有格 B0 0.0646、D0 0.0676、D1 0.0710，區間重疊），h = 1 只有 D0 三 seed 集成對 B0 有 +0.004 的可信增量。依 H-A 處置，**B0 為策略基線**，D0 集成並列為候選訊號；策略層（§13.17）以 B0 與 D0 兩者執行，D1 不進入策略層。

**執行面變更（2026-10-24，Chronos-2 執行前）**：跨股票 trainer 的每步主機同步點移除（epoch loss 留在 GPU 累加、loss 細節改回傳 tensor、群組列索引常駐 GPU 以切片取 batch、驗證與預測一次回傳主機）並改用 fused AdamW。模型、損失、每步一個 (訊號日, 市場) 群組、超參數、seed 與資料全部不變，CPU 上的單元測試結果逐位元相同；此變更只縮短牆鐘時間，不影響任何數字，D0／D1 不重跑。GPU 利用率低的原因是每步一個群組、模型小（0.4–0.65M 參數）而被 kernel launch 與主機同步綁住；多群組打包成一個 batch 會改變有效 batch 與優化軌跡，屬訓練程序變更，不在此做。

**§13.16 D1 第二個編碼器**：`t-xs-chronos2-small`（凍結 Chronos-2-small，通用時序基礎模型，28M 參數，Apache-2.0），`residual_v1a` 標籤，依第 9 點只跑 seed 42，配對同標籤 B0，並以 `scripts/compare_xs_runs.py` 對 D0 集成與 D0 單 seed 配對。事前預期（Kronos 論文台股附錄、Rahimikia 2025）：通用預訓練不轉移到台股截面，結果應不優於 Kronos-small；若對 D0 單 seed 的配對差在短格或持有格的區間下界 > 0 才改變結論。

**§13.16 Chronos-2 結果（2026-10-24，`t-xs-chronos2-small`，`launch-20261024T104417Z-980120111`，seed 42，14 個 fit，平均 8.9 epoch，權重 `autogluon/chronos-2-small` ddec013，embedding 維度 512）**。對原始 alpha（P1 ∪ P2）：短格 0.1159 [0.1074, 0.1242]、持有格 0.0872 [0.0701, 0.1034]、長格 0.0857 [0.0644, 0.1058]；殘差目標 P2 持有格 0.0648、長格 0.0583。與同標籤 B0 配對：原始 alpha 短格 −0.0016 [−0.0044, +0.0012]、持有格 −0.0016 [−0.0089, +0.0065]（H-A 兩格未通過）；殘差目標短格 **−0.0026 [−0.0048, −0.0003]**、持有格 **−0.0047 [−0.0096, −0.0002]**（劣於 B0，區間不跨 0）。與 D0 三 seed 集成配對（原始 alpha）：短格 −0.0053 [−0.0073, −0.0034]、持有格 −0.0053 [−0.0091, −0.0013]、長格 −0.0057 [−0.0105, −0.0006]。與 Kronos-small D1（同為單 seed）配對：短格 −0.0013 [−0.0047, +0.0021]、持有格 **−0.0076 [−0.0128, −0.0020]**、長格 −0.0068 [−0.0135, +0.0001]；殘差目標持有格 −0.0076 [−0.0118, −0.0034]。**判讀**：與事前預期一致，通用時序基礎模型的凍結表徵在台股截面上不但沒有增量，多日格還略劣於金融預訓練的 Kronos-small 與從頭訓練的 patch encoder，也劣於特徵 GBDT；D1 第二個編碼器不改變模型線結論。執行註記：trainer 同步點移除後，相同 14 個 fit 合計 1,151 秒（D1 Kronos 為 1,500 秒），每 epoch 9.2 秒對 13.4 秒；訓練階段 GPU 利用率 44–48%（修正前 25–33%），embedding 階段 56–60%（adapter 先把視窗搬回 CPU 再交給 Chronos pipeline）。模型線至此收斂：**B0 為策略基線，D0 三 seed 集成並列候選**，D1 兩個編碼器皆不進入策略層。

#### 13.17 暴露中性策略：0050 核心加股票期貨多空疊加（草稿，2026-10-24 寫入，待使用者確認後才實作與計分）

背景：§13.16 證明三個模型在殘差目標（扣掉市場、市值、產業效應後的相對表現）上的多日排序能力在 P2 仍成立（持有格 Rank IC 0.065–0.071，區間不跨 0），但對原始 alpha 的 Rank IC 在 P2 只剩 P1 的四到五成；差距來自 P2 大型股集中行情的市值與產業因子報酬，不是模型。§13.14 的多頭 band 組合把這些暴露原封不動帶回組合，所以失敗；§13.15 的 h = 1 當沖敗在價差。本節登記的策略把報酬拆成兩塊分別取得：β 直接持有 0050，alpha 用股票期貨宇宙內產業與規模中性的多空疊加，持有 5–10 日把價差成本攤薄。**使用者設定的門檻（2026-10-24）：淨報酬不如無腦買進持有 0050，就判定沒有可部署的策略。**

0. **步驄 0：描述性診斷（先做，決定是否往下）**。用 B0 與 D0（三 seed 集成）在 `residual_v1a` 上的 walk-forward 分數，在（a）§13.14 的流動性宇宙與（b）§13.15 的股票期貨流動宇宙內，對 h ∈ {5, 10}、P1、P2 各算：前後四分位的殘差目標與原始遠期報酬毛價差；產業中性（同產業內多空名目相等，產業依 `configs/reference/tw_company_reference_20261022.csv`，不足者併「其他」）、金額中性的多空組合每日毛報酬及年化；每日換手（持有 h 日、無 band）。**停止規則**：若 (b) 中 h = 10 的產業中性多空 P1 ∪ P2 年化毛報酬，低於以 §13.15 成本（每邊期交稅 0.002%、手續費 30 元／口、滑價 1 tick）與該換手估得的年化成本拖累的兩倍，則不建策略，直接寫報告：殘差 alpha 的幅度不足以支付台灣的執行成本。
1. **資料與輸入**：同 §13.14 與 §13.15（raw `market.parquet`、期交所全契約與 TX 日行情、`0050.TW` 的開盤價與還原收盤價）。訊號為殘差標籤模型的 h ∈ {5, 10} 集成分數，先在宇宙內轉百分位，再以半衰期 3 日 EMA 平滑（同 §13.14）。多 horizon 融合（§13.16 第 6 項）不進本節格子，另作敏感度。
2. **基準**：**0050 買進持有**：P1 首個訊號日次一開盤買進、股息再投入（以還原收盤價計）、期末開盤賣出，計手續費 0.1425% × 0.6、ETF 證交稅 0.1%（賣出）、滑價 0.1%。另列 TAIEX 報酬指數供對照。所有判定對 0050 買進持有。
3. **策略 S1（主要）：0050 核心 + 股票期貨多空疊加**。
   - 資本配置：現金準備 c 用於期貨保證金的現金部分與逐日結算緩衝，其餘 (1 − c) 買 0050；0050 持股依期交所規定抵繳保證金（有價證券抵繳：市值折價 30%，抵繳額不得超過應有保證金的 50%）。總名目 = L × 資本，L = 1.0（主要）、0.5（敏感度）；原始保證金 20.25%、維持 15.53%（§13.15）。c = 應有保證金的 50% × 1.5（結算緩衝），L = 1 時約 15%，事前固定不選。現金準備低於應有保證金現金部分的 1.0 倍時，次一開盤賣出 0050 補足並計一次追繳。
   - 多空選股：§13.15 的股票期貨流動宇宙（近月 20 日平均成交量 ≥ 100 口、訊號日一般時段有成交）；依平滑分數做多前 K、放空後 K（每邊 K ∈ {20, 30}），buffer m ∈ {2, 3} 的 band 規則（進前 K 才進、掉出前 m × K 才出），每日每邊新進不超過 ⌈K／10⌉ 檔。
   - 中性化：金額中性（多空名目相等）；產業中性：每個產業的多空名目差不超過該產業名目的 20%，以縮減超額一邊達成；規模中性：依訊號日前 20 日成交金額中位數分上下兩半，各半內多空名目相等。無法同時滿足時以金額中性優先、產業其次、規模最後，並記錄每日未滿足的程度。
   - 交易：t 日收盤算訊號，t+1 期貨開盤進出；持有不設固定天數，由 band 決定；契約到期日前一日開盤換次月（兩邊成本）；口數 = ⌊每檔名目／（開盤價 × 每口股數）⌋，不足一口者跳過並計數，每檔口數 ≤ 當日成交量 × 5%；期貨損益以結算價逐日記帳。
   - 成本：期貨每邊期交稅 0.002%、手續費 30 元／口（另報 60）、滑價 1 tick（另報 2 tick），tick 級距同 §13.15；0050 同第 2 項。
4. **策略 S2（次要）：0050 核心 + 多頭衛星**。(1 − s) 買 0050，s ∈ {0.2, 0.3} 以 §13.14 的流動性宇宙、同一訊號、band 規則（K ∈ {30, 50}、m ∈ {2, 3}）買前 K 檔等權現股，成本同 §13.14 主要情境。不做空、不中性化；衛星扣成本後贏過 0050 即為 alpha。資本情境含整張與零股（§13.14）。
5. **策略 S3（描述性）：純多空疊加**。S1 的多空部位單獨計算、對照現金，用來報告 alpha 這一腿本身的年化報酬、Sharpe 與成本拖累。
6. **資本情境**：分數位置、1,000 萬、3,000 萬、1 億、3 億、10 億元；期貨口數取整、5% 參與率上限；1,000 萬元若因口數不足被擋超過 50% 的名目則標記不可行（§13.15）。
7. **選擇與判定**：
   - 格子在 P1 選：S1 的 (h, K, m) 以「策略每日淨報酬減 0050 每日淨報酬」的 P1 年化差 IR 最高者選定，S2 的 (h, K, m, s) 同法；B0 與 D0 分別選，選定後固定用於 P2、P1 ∪ P2 與前瞻。
   - **H-G（勝過 0050）**：選定參數的 S1，主要成本與分數位置下，P1 ∪ P2「策略減 0050」年化淨差的 95% 區間下界 > 0（每日差序列的 circular block bootstrap，區塊 20、1,000 次、seed 11），**且** P2 單獨的年化淨差 > 0。S2 以相同規則另判。B0 與 D0 分別判定。
   - 描述性但必列：最大回撤與 Sharpe 相對 0050 的差、追繳次數、保證金使用率、中性化未滿足天數比例、換手與成本拖累。
   - 前瞻候選：通過 H-G 的（模型, 策略）中 P1 ∪ P2 淨差 IR 最高者。
8. **報告（固定欄位）**：每個策略、模型、資本與成本情境、期間都列出年化淨報酬、CAGR、Sharpe、最大回撤、對 0050 的年化淨差與區間、對 TAIEX 報酬指數的差、換手、成本拖累、被擋比例、追繳次數；0050 買進持有與 TAIEX 報酬指數同表列出。中英雙語。
9. **處置**：H-G 通過 → 登記 H-C（P3 前瞻，含實際期貨價格與保證金）。H-G 未通過 → 結論為「模型有 h = 1 與殘差多日排序能力，但在台灣的工具與成本下沒有勝過 0050 買進持有的可部署策略」，報告照實寫，不再調參重跑。
10. **程序**：步驄 0 的診斷腳本與結果先出；設定檔 `backtest/config/exposure_neutral.json` 在計分前寫入並 commit；先以合成資料測試中性化、band、抵繳保證金會計（含折價、50% 上限、追繳）、換月；再對 B0、D0 計分。

判讀限制（事前）：

- P2 已經看過，而且 P2（TAIEX 一年漲一倍、0050 因台積電權重跑贏 TAIEX）是 0050 最強的一段；用它當基準是最難的考法，回測贏了也要 P3 才算確認。
- 中性化放棄市值與產業因子報酬是設計選擇；S1 的現金準備（約 15%）在上漲年份是確定的拖累，必須在 alpha 腿贏回來。
- 股票期貨宇宙約 110 檔流動標的，產業分布不均（電子為主），產業中性的 20% 容忍是權衡；契約代碼對應沿用 2026-10 清單（存活者偏誤約 38 檔）。
- 期貨開盤價可成交但 8:45–9:00 流動性較薄；滑價以 tick 計；結算價為官方 mark。保證金與抵繳規則取自期交所現行公告（折價 30%、抵繳上限 50%），未模擬規則變動、逐檔保證金級距與盤中追繳。
- 格子在 P1 選擇有選擇偏誤；P1 ∪ P2 判定含 P1。

**§13.17 步驟 0 結果（2026-10-24，`twbacktest.exposure_diagnostics`，輸出 `artifacts/diagnostics/backtest/exposure-step0-b0-d0-20261024T120000Z/`）**。B0（`t-lgbm-alpha-residual_v1a`）與 D0（三 seed 集成）在 `residual_v1a` 上的分數；前後四分位、金額中性；成本為每邊期交稅 0.002%、手續費每口 30 元、1 tick 滑價，每日換 1/h 部位加每月換月。

| 宇宙 | h | 期間 | 殘差 Rank IC（B0／D0） | 殘差四分位價差／批 | 產業中性多空毛年化（L = 1）B0 | D0 | 年化成本拖累 |
|---|---|---|---|---|---|---|---|
| 流動性宇宙（市場內） | 10 | P1 | 0.091／0.089 | 0.88%／0.96% | +8.7% [+4.5%, +13.3%] | +10.1% [+6.3%, +13.9%] | 18.0%（現股成本另計） |
| 流動性宇宙 | 10 | P2 | 0.053／0.057 | 0.06%／0.34% | −4.5% [−12.0%, +2.8%] | +0.9% [−7.2%, +9.5%] | |
| 流動性宇宙 | 10 | P1 ∪ P2 | 0.080／0.080 | 0.65%／0.79% | +5.0% [+0.5%, +9.4%] | +7.6% [+3.3%, +11.4%] | |
| 股票期貨流動宇宙（合併） | 10 | P1 | 0.065／0.052 | 0.52%／0.36% | +4.5% [−1.1%, +10.6%] | +3.6% [−0.9%, +8.9%] | 18.3% |
| 股票期貨流動宇宙 | 10 | P2 | **−0.007／+0.006** | −1.17%／−0.69% | −14.0% [−26.7%, −2.2%] | −7.9% [−21.9%, +5.6%] | 18.5% |
| 股票期貨流動宇宙 | 10 | P1 ∪ P2 | 0.045／0.039 | 0.05%／0.07% | **−0.6%** [−7.0%, +5.4%] | **+0.4%** [−5.6%, +6.2%] | 18.3% |
| 股票期貨流動宇宙 | 5 | P1 ∪ P2 | 0.062／0.056 | 0.27%／0.24% | +5.7% [−0.4%, +11.9%] | +5.2% [−0.2%, +11.2%] | 30.8% |

**停止規則觸發（兩個模型都停）**：股票期貨流動宇宙、h = 10、P1 ∪ P2 的產業中性多空毛年化 B0 −0.6%、D0 +0.4%，成本拖累 18.3%，門檻 36.7%；差距不是幾倍而是幾十倍。h = 5 的毛年化較高（+5–6%）但成本拖累 30.8%，同樣不可能。規模中性與無中性化（plain）的版本結論相同。

**判讀**：（1）殘差 alpha 的**幅度**太小：即使在全流動性宇宙、P1 這段最好的期間，前後四分位的殘差價差也只有每 10 日 0.9–1.0%，產業中性多空毛年化 9–10%；台灣股票期貨單邊成本 0.25%、持有 10 日的年化拖累 18%，股票現股的證交稅與滑價更高。Rank IC 0.08–0.10 對應的可交易價差就是這個量級，排序能力存在與可交易是兩件事。（2）**alpha 不在可放空的宇宙裡**：約 130 檔有流動股票期貨的大型股中，殘差 Rank IC 在 P2 掉到 0（B0 −0.007、D0 +0.006），P1 ∪ P2 只有 0.04–0.045；P2 的殘差排序能力全部來自沒有股票期貨、只能做多且成本更高的中小型股。（3）因此 S1（0050 加多空疊加）不可能勝過 0050；S2（多頭衛星）能用的 alpha 上限是流動性宇宙多頭這一側、毛年化約 3–5%，低於現股 h = 10 持有的成本拖累，也不可能。依第 0 項：**不建策略，直接寫報告**。結論：這個模型線有 h = 1 排序能力（執行成本不可交易）與殘差多日排序能力（幅度太小、且不在可放空宇宙內），在台灣現有工具與成本下沒有勝過 0050 買進持有的可部署策略；§13.17 的 S1／S2／S3、H-G 與 `exposure_neutral.json` 不實作。

**§13.17 補充：S1 重新登記為「0050 核心加正交化多空疊加」（2026-10-25 於實作與計分前寫入；待 §13.19 補充三第 3 項的可借券名單到位後實作，待 P3 計分）。** 停止 S1 的理由是原分數的多空在股票期貨宇宙沒有 alpha；§13.19 補充二顯示對 beta 與動能正交化後的現股多空（V1、無配平）在機構成本下 P1 ∪ P2 淨 +17% 到 +29%、四個曆年為正，而且 alpha 在流動性宇宙的現股，不在股票期貨宇宙。因此疊加的工具由股票期貨改為**現股多空**，β 仍由 0050 提供。本補充只登記設計與門檻，不改動補充二的任何結果。

1. **資料與輸入**：同 §13.19 補充二（B0、D0 的 h = 10 集成分數，V1 正交化，band 參數固定為 (10, 50, 3)／(10, 50, 2)，流動性宇宙）；`0050.TW` 的開盤價與還原收盤價；TX 近月連續序列（變體 B 用）；可借券名單（`tw_chip_data` 第 10.3 節的「平盤下得融（借）券賣出」每日名單，兩所皆可回溯至 2013-09-23）。
2. **基準**：0050 買進持有，P1 首個訊號日次一開盤買進、股息再投入、期末開盤賣出，計手續費、ETF 證交稅 0.1%、滑價（兩個成本檔各自的費率）。另列 TAIEX 報酬指數。
3. **變體 A（主要）：0050 現股核心 + 現股多空疊加**。資本 C：0050 持有 w × C，w ∈ {0.5, 0.7}；多空疊加每腿名目 L × C，L ∈ {0.5, 1.0}（機構）、{0.5}（散戶，受融券 90% 保證金與擔保品限制）。多頭腿先用剩餘現金 (1 − w) × C，不足的部分以 0050 質押借款支應，借款利率 r_fin ∈ {2.5%, 4.0%}／年按日計；空頭腿以借券（機構）或融券（散戶）執行，賣出價金留作擔保品不生息，借券費年率 2%（機構）或融券手續費 0.08%（散戶），股息由空方補償（以還原價格計算即已含）。每日報酬 = w × r_0050 + L × (r_long + r_short) − max(L − (1 − w), 0) × r_fin／252 − 成本；多空兩腿的 band、成本與借券費同補充二。**空頭腿只在可借券名單內選股**（名單缺值的日期沿用前一有效名單，並計缺值天數）。
4. **變體 B（次要）：β 由台指期提供**。資本 15% 作 TX 多頭保證金（名目 = C，依結算價逐日結算、換月兩邊成本同 §13.14），85% 作現股多空（每腿 0.85 × C）；不需質押借款。TX 多頭對報酬指數的隱含成本（§13.14 量得 −2.4%／年）照實計入。
5. **格子與選擇**：變體 A 的 (w, L) 與變體 B 各一組結果，在 P1 以「策略減 0050 每日淨差」的年化 IR 選一組；B0 與 D0 各選；選定後固定用於 P2、P1 ∪ P2 與 P3。多空 band 參數**不**重選。
6. **H-G（勝過 0050，重新啟用）**：選定組合在主要成本與分數位置下，P1 ∪ P2「策略減 0050」年化淨差的 95% 區間下界 > 0（區塊 21、1,000 次、seed 11）且 P2 單獨淨差 > 0。機構與散戶兩個成本檔分別判，表格並列。**但因 V1 設計在看過 P2 之後決定，P1 ∪ P2 的 H-G 只作資訊性判讀；正式判定在 P3（2026-06 之後，含實際成本與可借券名單）**。另列「策略減 0050」的 Sharpe 差、最大回撤差、質押借款與借券費總額、追繳或擔保品不足的天數、可借券名單過濾掉的候選比例。
7. **事前寫下的預期**：以補充二的數字粗估，變體 A（w = 0.5、L = 1.0、r_fin = 2.5%）在 P1 ∪ P2 約為 0.5 × 41% + 18% − 1.3% ≈ +37%，與 0050 的 +41% 相當，贏不贏取決於多空腿在可借券名單內還剩多少與融資成本；變體 B 約為 TX +38% + 0.85 × 18% ≈ +53%。若兩個變體都不能在 P1 ∪ P2 贏過 0050，結論維持「無勝過 0050 的策略」；若贏了，也只是 P3 的待驗證假設。
8. **資本情境**：分數位置、1,000 萬（散戶，含零股）、3,000 萬、1 億、3 億（機構）；3 億以上依補充二的容量結果不列。
9. **程序**：`backtest/twbacktest/core_overlay.py` 與設定檔 `backtest/config/core_overlay.json`；合成資料測試：質押借款與借券費會計、可借券名單過濾、TX 保證金與換月；可借券名單進 curated 層後先跑補充四（空頭腿限名單的 S1／S2 重算），再跑本補充。結果寫入本節與一份雙語報告。

判讀限制（事前）：P2 是 0050 最強的一段，用它當基準是最難的考法，也是看過答案後的設計；可借券名單只告訴我們「可以借」，借不到或借券費遠高於 2% 的中小型股不在模型內；質押借款利率與擔保品比率是假設值；散戶變體以融券執行，融券的額度與停券事件未模擬。

#### 13.18 免費籌碼與事件資料的排序增量、低換手策略與雙成本結構（草稿，2026-10-25 寫入，待使用者確認後才實作與計分）

背景：§13.13–§13.17 證明日線 OHLCV 特徵的橫斷面排序在台灣散戶成本下沒有可部署的策略；殘差 alpha 每 10 日不到 1%，而我們的設計每天對全宇宙重新評分、持有 5–10 日、每天換掉十分之一的部位，年化成本 12–18%。文獻與資料商研究（TEJ 20 個分點因子、週頻持有 5 日的十分位多空每週 0.47%；FinLab 傳統分點指標 IC ≈ 0.04；學術上法人買賣超多為順勢追價、落後一期預測力弱；處置宣告後三日顯著負超額）顯示：籌碼與事件資料的連續排序因子幅度與現有模型同量級，價值在於**反向類與事件型的用法**，以及把換手配到成本結構。使用者 2026-10-25 的決定：**先用免費官方資料做前期評估**，付費分點列為第二階段，是否購買依本節結果決定。資料由獨立專案 `tw_chip_data`（決策摘要 v1.0 第 10.1 節）提供，本節只登記 `stock_forecasting` 這一側的實驗。

2026-10-25 補充（使用者同意後寫入）：學界與業界沒有任何公開研究對日頻排序模型做過「有無集保、有無分點」的配對消融；最接近的替代證據一致指向同一形態——集保（1–5 張級距週變化）單獨的 IC 絕對值約 0.005–0.012、持有越久越強、年週轉近 30 倍，是月頻型的慢訊號（TEJ 2016–2023）；分點作為連續排序因子與現有模型同量級（TEJ 20 個因子，2020-12–2026-06），但作為篩選與條件變數有價值：TEJ 2021–2024 月頻研究中加入分點熱門度使純動能的 Sharpe 由 0.9 升至 1.575、1.243 升至 1.98、1.372 升至 2.468，淨報酬幾乎不變（85.57% → 90.30%），改善來自波動與回撤；韓國投資人類別資料讓短期反轉策略獲利提高 40–70%（Ülkü 與 Onishchenko 2019）；A 股以單筆金額代理「誰在買」的資金流向因子對換手率、動能、波動率取殘差後仍有增量，但與換手率高度重疊且衰減快（華泰 2018）。據此，**集保與分點的購買延後到下列兩道閘門之後**：G1（第 5 項）與 G2（第 6 項），任一通過才購買分點；集保只在 S-C 進入實作時購買。

1. **資料**：`tw_chip_data` 第一階段的 curated 層，12 個品項，鍵為 (date, symbol, market)，每列帶 `available_at`（Asia/Taipei）；下游在訊號日 t 只能用 `available_at` ≤ t 收盤後、且依各品項「T+1 開盤前」規則可用的列。品項與歷史起點：注意有價證券（TWSE 2001、TPEX 2012）、處置有價證券（TWSE 2005、TPEX 2012；處置起迄於公告時即已知）、三大法人買賣超（TWSE 合併表 2012-05-02、分表 2004-12；TPEX 全市場 2014-12-01，之前只有前 N 名排行、標 `coverage = top_n` 且**排除**）、融資融券餘額（TWSE 2001、TPEX 2007）、融券與借券賣出餘額（TWSE 2005-07、TPEX 2012 年內）、當沖統計（兩市場 2014-01-06）、外資及陸資持股（TWSE 2004-02）、集保股權分散（官方僅近一年，**只作描述、不進判定**；FinMind Backer 方案有 2010-01-29 起的週表，依第 10 項延後購買）、**得為融資融券與得為借券賣出有價證券每日名單（2026-10-25 補登記，§13.19 補充三第 3 項；供空頭腿的可借券宇宙，不進特徵）**、每月營收（2010-01；2013-01 起合併基礎，切換需標記）、開休市日曆、公司參考表。價量資料組與 walk-forward **不變**（`53e3915b…`，2016 起，`tw_quarterly_v1` 14 折，P1 ∪ P2 判定）；籌碼的更長歷史在本節只用於滾動視窗的暖機，是否把價量資料組回推到 2012 另案登記。
2. **特徵庫 `chip_tw_v1`**（規格 `configs/features/chip_tw_v1.json`，在 `alpha_tw_v1` 旁建、identity 另含 curated 層 manifest 的 digest；全部只用 `available_at` ≤ 訊號日收盤的列）：
   - 法人流量（每類 investor_type 分開：外資、外資自營商、投信、自營商自行、自營商避險）：淨買超股數除以當日成交量的 1／5／10／20 日累計、連續同向天數、20 日淨買超占流通股比、相對 60 日分布的 z 分數。
   - 融資融券：融資餘額 1／5／20 日變化率、融資使用率（餘額／限額）、融券餘額變化率、資券比、資券互抵占成交量比。
   - 借券：借券賣出餘額 1／5／20 日變化率、借券餘額占流通股比。
   - 當沖：當沖比（當沖成交股數／成交量）及其 20 日均值與 z 分數；當沖比的 1 日跳升幅度。
   - 外資持股：持股比率 5／20 日變化、距法令上限的空間。
   - 事件旗標與距離：注意股累計次數（30 日內）、是否處置中、處置宣告後第幾個交易日、處置結束後第幾個交易日、月營收公布後第幾個交易日、營收年增率與前三個月趨勢的驚奇、`basis` 切換旗標。
   - 預估 60–80 欄；每欄記錄來源品項、公布時刻假設與缺值處理（缺值保留 NaN，LightGBM 原生處理；D0 若啟用則以 0 填補並加缺值旗標）。
3. **模型與對照**：B0 LightGBM 設定、超參數、seed（42／43／44）與 §13.13 相同；三組特徵在同一 walk-forward 各訓一次：F0 = `alpha_tw_v1`（已有結果，`launch-20261023T103805Z-2637116736`，不重跑）、F1 = `chip_tw_v1`、F2 = `alpha_tw_v1` + `chip_tw_v1`。標籤以 `residual_v1a`（§13.16 通過的變體）為主，標準標籤只作描述。D0 與 D1 **不在本節**：只有 F2 通過第 5 項且殘差幅度明顯時，才另案決定是否重訓 D0。
4. **指標**：同 §13.13／§13.16（殘差目標與原始 alpha 目標的 Rank IC cell、block bootstrap、配對差），另加 §13.17 步驟 0 的幅度指標：h ∈ {1, 5, 10} 的前後四分位殘差價差（每批）與產業中性多空毛年化，在流動性宇宙與股票期貨流動宇宙各算。配對差 F2 − F0、F1 − F0 在同一訊號日相減。
5. **判定 H-H（籌碼有增量）**：F2 對 F0，(i) P1 ∪ P2 持有格對殘差目標的配對 Rank IC 差 95% 區間下界 > 0；且 (ii) P1 ∪ P2 持有格前後四分位殘差價差（每 10 日）至少為 F0 的 1.5 倍。兩者都成立才通過；F1 單獨的結果只作描述（回答「籌碼能不能取代價量」）。
   **閘門 G1（分點購買的第一道門，2026-10-25 補登記）**：同一組配對差 F2 − F0 在兩個宇宙分列——(a) 流動性宇宙全體；(b) 中小型股 = 流動性宇宙扣除 §13.17 的股票期貨流動宇宙（P2 殘差訊號所在的位置，也是散戶與隔日沖主導的區段）。G1 通過的條件是 (a) 或 (b) 任一宇宙的 P1 ∪ P2 持有格對殘差目標的配對 Rank IC 差 95% 區間下界 > 0，且該宇宙的前後四分位殘差價差（每 10 日）不低於 F0。G1 比 H-H 寬：H-H 問「籌碼能不能把排序推上一個檔次」，G1 只問「籌碼的量有沒有任何增量」；G1 都不通過，就沒有理由期待加上「誰在買」能翻盤。
   **G1 子分析（2026-10-25 補登記，描述性，不改變 G1 的通過條件）**：§13.19 的持倉診斷顯示價量模型的多日排序本質上是短期反轉訊號，P2 動能延續時兩腿同時逆向。文獻的機制（Ülkü 與 Onishchenko 2019；Barber、Lee、Liu、Odean 2009）是法人買進推升的強勢股傾向延續、散戶追價的強勢股傾向反轉，所以籌碼特徵若有價值，應集中在「近期強勢股裡分出延續與反轉」。因此 F1 − F0 與 F2 − F0 的配對差都要分期（P1、P2、P1 ∪ P2）與分組報告：(a) 近期強勢股 = 訊號日前 20 日報酬在宇宙內前五分之一；(b) 其餘。每組報對殘差目標與原始 alpha 的配對 Rank IC 差、前後四分位價差，以及**強勢股組的後四分位事後報酬對宇宙的差**（籌碼特徵是否讓被排到後段的強勢股不再是事後繼續漲的那一批）。預期（事前寫下、不作判定）：若籌碼機制成立，F2 在 P2 強勢股組的改善應大於其他組；若改善均勻分布在各組，籌碼的增量就不是條件化反轉，而是另一個獨立訊號。
6. **事件型描述性分析（策略前的步驟 0）**：在流動性宇宙內做事件研究，不含選擇：處置宣告日 t0 的 [t0−5, t0+10] 累積超額報酬與 [t0+1, t0+5] 的持有報酬；法人連續淨買超 ≥ 5 日的次日起 5／10 日報酬；融資餘額單日增幅前 5% 的後續 5／10 日報酬；當沖比單日跳升前 5% 的次日報酬（沒有分點時的隔日沖代理）。每個事件報樣本數、平均超額、勝率與 block bootstrap 區間，並與同期全宇宙比較。**停止規則**：若所有事件的 [t0+1, t0+5] 平均超額的絕對值都小於散戶一趟成本（0.67%），事件觸發式策略（第 7 項 S-B）不實作。
   **閘門 G2（分點購買的第二道門，2026-10-25 補登記）**：只看三類免費的反向型事件——處置宣告日、注意股首次公告日（30 日內首次列入）、當沖比單日跳升前 5%——各自的 [t0+1, t0+5] 平均超額報酬（相對同期流動性宇宙等權）。G2 通過的條件是任一事件類別同時滿足：樣本數 ≥ 200；平均超額絕對值 ≥ 2 × 0.67% = 1.34%；block bootstrap 95% 區間不含 0.67%（即扣掉一趟成本後仍有把握為同號）。這三類事件是沒有分點時「隔日沖與過熱籌碼反轉」的最佳代理；G2 通過代表反轉現象存在、值得用分點身分把觸發條件磨利；G2 不通過代表連粗代理都看不到可覆蓋成本的反轉，分點不太可能無中生有。
7. **策略層（只有 H-H 通過才實作，每條都用 F2 分數）**：
   - **S-A 低換手多頭**：週頻再平衡，週末訊號日依 F2 殘差分數選前 K（K ∈ {30, 50}），band m ∈ {2, 3}，單週換手上限 50%；對照 0050 買進持有與等權流動宇宙。
   - **S-B 事件觸發**：只在第 6 項通過停止規則的事件觸發時進場（多頭或反向），固定持有 h ∈ {5, 10} 或 band 出場；每日最多新進 ⌈K／10⌉ 檔。
   - **S-C 月頻多因子**：同一 F2 分數在月末訊號日選前 K，單月換手上限 50%；這是第三列玩法的對照，用來量低換手能保留多少 alpha。
   - **機構版**（同參數）：S-A／S-C 加入借券放空後 K（流動性宇宙內可借券標的，借券費率參數化）或股票期貨宇宙多空；散戶版只有多頭與股票期貨多空。
8. **雙成本結構**（`backtest/config/cost_profiles.json`，所有回測模組讀同一份，每張結果表都並列兩欄）：
   | 項目 | 散戶（判定用） | 機構（描述，暫定值待查證） |
   | --- | --- | --- |
   | 手續費 | 0.1425% × 0.6；敏感度 × 0.28 | 0.03% |
   | 證交稅 | 0.3%；當沖 0.15%（至 2027） | 0.3% |
   | 滑價（吃單） | TWSE 0.1%、TPEX 0.2%；敏感度 0 與 2 倍 | 0.05% |
   | 放空 | 只能用流動股票期貨（§13.15 成本） | 借券，年費 2%（參數），借券費按持有日計 |
   | 期貨保證金 | 現金 | 持股抵繳（折價 30%、上限 50%） |
   | 資本情境 | 分數位置、1,000 萬整張／零股、3,000 萬、1 億 | 3 億、10 億，5% 參與率上限 |
9. **判定 H-I（勝過 0050）**：散戶成本、分數位置，格子在 P1 以「策略減 0050 每日淨差」的年化 IR 選定（S-A、S-B、S-C 各選一組），P1 ∪ P2 年化淨差的 95% 區間下界 > 0（區塊 20、1,000 次、seed 11）且 P2 單獨 > 0；任一策略通過即通過。機構成本下以相同規則另報 H-I-inst，只作資訊。一律同時列出絕對報酬、Sharpe、最大回撤、換手、成本拖累、被擋買單比例與追繳次數。
10. **處置與付費資料的購買規則（2026-10-25 改寫）**：
    - H-I 通過 → 登記 P3 前瞻（2026-06 之後，含實際成本）。
    - **分點**：G1 或 G2 任一通過 → 購買 FinMind Sponsor Pro 一個月（NT$3,330，整日 parquet 回補 2021-06-30 起的分點與分點統計；Sponsor 月繳 NT$999 需約 230 萬次逐檔逐日請求，約 16 天連續抓取，只在使用者選擇時採用），另案登記第二階段（F3 = F2 + 分點特徵、S-B 的分點觸發條件）。G1 與 G2 都不通過 → 不購買，籌碼線結案，報告寫明替代證據與本節消融結果。H-H 通過但 H-I 未通過 → 問題在成本與幅度，分點不太可能補救，同樣不購買。
    - **集保**：只在 S-C 月頻策略進入實作時購買 FinMind Backer 一個月（NT$699，2010-01-29 起的週表）補齊歷史；在那之前只用官方近一年資料作描述。
    - **分點資料跨度（事前登記，購買前寫死）**：FinMind 分點自 2021-06-30 起，約 5.3 年。業界與學界的短線籌碼研究用 5–8 年（TEJ 分點因子 2020-12–2026-06；FinLab 分點特徵 2018 起；Barber 等台灣投資人研究 1995–1999），18 年以上的訓練窗屬月頻多因子定價研究，不是本線的對象；QuantBench 顯示日頻模型以 3 個月滾動重訓最佳，近期資料的價值高於深遠歷史。第二階段沿用 `tw_quarterly_v1`（2016 起擴張、折 2023Q1–2026Q2）以便與 F0／F1／F2 在相同日期配對，分點特徵在 2021-06-30 前為缺值（LightGBM 原生處理）；各折的分點歷史 2023Q1 約 1.5 年、2024Q1 約 2.5 年、2026Q2 約 4.9 年（含 12 個月驗證）；另報「折自 2024Q1 起」的敏感度結果。檢定力取決於測試區間（P1 ∪ P2 約 820 個交易日），不取決於訓練歷史，買更早的分點不會加長測試區間；2021 年以前的分點只有證交所資訊商店（每月約 NT$80,000）或 TEJ 有，不列入。
11. **程序**：`tw_chip_data` 先交付 curated 層與 manifest；本專案新增 `data/chip_features.py` 與規格檔、`stock-forecasting-build-features --spec` 支援多規格、合成資料測試（含 `available_at` 過濾與 top_n 排除）；RunPod CPU 建特徵庫與 B0（`features --spec`、`lgbm --spec`）；本機新增事件研究模組與 `cost_profiles.json`，既有回測模組加 `--cost-profile retail|institution|both`；結果寫成中英雙語報告。

判讀限制（事前）：

- P2 已經看過；本節是看到 §13.16–§13.17 結果後設計的，確認需要 P3。
- 公布時刻多數標「待實測」，第一版用保守規則（T+1 開盤前）；若實測發現某品項更晚公布，該欄位要延後一天並重跑。
- TPEX 2014-12 前的法人資料只有前 N 名，已排除；集保只有一年，不進判定。
- 事件樣本稀疏（處置每年數百筆），事件研究的區間會寬；沒有分點資料時「隔日沖」只能以當沖比跳升代理。
- 借券費率、機構手續費與滑價是暫定值，影響的只有機構欄，不影響判定。
- 若第二階段購買分點，事後分析只能引用以下三條資料限制，不得再以「歷史不夠長」解釋 H-H／H-I 的失敗：(1) 樣本外區間 2023-01–2026-05 含 2024-08 與 2025-04 的急跌，但不含 2022 年空頭（2022 只在訓練內），這是全專案所有模型共同的限制；(2) 分點行為不平穩（隔日沖分點輪替、當沖稅減半延至 2027），更早的分點資料價值遞減；(3) FinMind 分點 2022-10-31–11-03 缺漏，上櫃部分在該區間不完整，需在資料層標記並在回測中排除或補值。
- 事件型分析的檢定力受事件數而非年數限制：隔日沖代理事件每日數百筆不缺；處置與注意公告一年數百件，其免費歷史自 2015 年起，不受分點起始日影響；報告列出每類事件數。

#### 13.19 機構成本下的六情境回測：台股多空、台指期疊加與現股當沖（資訊性分析，2026-10-25 於實作與計分前寫入；使用者指示執行）

背景：§13.14–§13.17 的所有回測都以散戶成本（手續費 0.1425% × 0.6、證交稅 0.3%、滑價 0.1–0.2%、放空只能用股票期貨）判定，結論是排序能力被成本吃掉。使用者 2026-10-25 的問題：同一組模型（B0 `t-lgbm-alpha-residual_v1a`、D0 `t-xs-scratch` 三 seed 集成，皆為 `residual_v1a` 標籤）在**業界的成本條件**下是否有吸引力。本節是描述性分析，不設通過即前瞻的假設，但所有參數、選擇規則與「吸引力」門檻在計分前寫死；每張表依 §13.18 第 8 項並列散戶與機構兩欄。

1. **六個情境**（每個情境都是「多空」：做多前段、放空後段）：
   | 編號 | 名稱 | 部位 | 標的 |
   | --- | --- | --- | --- |
   | S1 | 短至中天期台股多空 | 多頭 band 組合 + 空頭 band 組合，各占資本 100%（總曝險 2 倍、金額中性） | 流動性宇宙內的現股；機構以借券放空，散戶以融券放空 |
   | S2 | S1 + 台指期 | S1 加上以滾動 beta 對沖殘餘市場曝險的 TX 部位 | 現股 + TX 近月 |
   | S3 | 第一日當沖台股多空 | t 日收盤 h = 1 分數，t+1 開盤買前 q／放空後 q，t+1 收盤全部平倉（現股當沖、先賣後買） | 流動性宇宙內的現股 |
   | S4 | S3 + 台指期 | S3 加上以滾動 beta 對沖的 TX 日內部位（開盤進、結算價出） | 現股 + TX |
   | S5 | 當沖 + 短至中天期 | 資本各半：0.5 × S1 + 0.5 × S3 | 現股 |
   | S6 | S5 + 台指期 | 0.5 × S2 + 0.5 × S4 | 現股 + TX |
2. **資料與宇宙**：同 §13.14（raw `market.parquet`、TX 官方日行情 2022-12 至 2026-06）；流動性宇宙 = 20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬、收盤 ≥ 10 元、排除 ETF；每個市場各自建組合，另報兩市場各半的「TW」合併。放空宇宙與多頭相同（假設流動性宇宙內皆可借券／融券／當沖先賣，實際可借量與當沖標的名單不模擬，列為限制）。
3. **多日組合（S1）**：訊號 = 單一 horizon 分數 → 市場內百分位 → 半衰期 3 日 EMA（同 §13.14）。多頭：進前 K 買、掉出前 m × K 賣，每日新進 ≤ ⌈K／10⌉，漲停開盤不買、跌停開盤不賣。空頭：以分數反向排序套同一 band 規則，跌停開盤不放空、漲停開盤不回補；空頭位置每日依位置值等權重設（報酬 = −現股開盤到開盤報酬）；機構另扣借券費年率 2%（按持有日計）。格子 h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3}。
4. **當沖（S3）**：t+1 開盤到收盤；q ∈ {5%, 10%}（每邊）；等權；漲停開盤的多頭候選與跌停開盤的空頭候選跳過並順延；當沖證交稅 0.15%（兩個成本檔相同）。
5. **TX 疊加（S2、S4）**：對沖名目 = −β_t × 資本，β_t 為該組合自身毛報酬對市場指數報酬的滾動回歸係數（嚴格使用 t 之前 60 個區間，少於 30 個區間時 β = 0）；S2 用開盤到開盤的指數與 TX 結算價對結算價，S4 用日內指數與 TX 開盤到結算價。TX 成本每邊：期交稅 0.002%、手續費每口 50 元（散戶）／30 元（機構）、滑價 1 點；名目變動與換月（兩邊）計費。
6. **雙成本檔**（`backtest/config/cost_profiles.json`，§13.18 第 8 項的實作；主要值，敏感度另列滑價 0 與 2 倍）：
   | 項目 | 散戶 | 機構 |
   | --- | --- | --- |
   | 手續費（每邊） | 0.1425% × 0.6 = 0.0855% | 0.03% |
   | 證交稅（賣出） | 0.3%；當沖 0.15% | 0.3%；當沖 0.15% |
   | 滑價（每邊） | TWSE 0.1%、TPEX 0.2% | 0.05%（兩市場） |
   | 多日放空 | 融券：放空時另付融券手續費 0.08%；不計保證金利息 | 借券：年費 2%，按持有日計 |
   | 當沖放空 | 現股當沖先賣後買，無借券費 | 同左 |
   | TX | 每口 50 元、1 點 | 每口 30 元、1 點 |
   | 資本情境 | 分數位置 | 分數位置；3 億、10 億（每檔名目 ≤ 20 日成交金額中位數的 5%，超過者跳過並計數） |
7. **選擇規則**：S1 的 (h, K, m) 與 S3 的 q 分別以 **TWSE、P1、機構主要成本、分數位置的淨 Sharpe 最高**選定（B0 與 D0 各選），選定後固定用於 P2、P1 ∪ P2、兩個成本檔、TPEX、TW 合併與 S2–S6。完整格子另列。
8. **指標**：年化毛報酬、年化淨報酬與 95% 區間（circular block bootstrap：多日與合併用區塊 21，當沖用區塊 5；1,000 次、seed 11）、CAGR、Sharpe、最大回撤、勝率、成本拖累、換手（多日：每日新進／K；當沖：每日 100%）、平均持有日、平均檔數、被擋比例（漲跌停、參與率）、TX 的平均 |β| 與平均名目。市場中性組合以 0 為基準，另列同期 TAIEX 報酬指數供對照。
9. **「對業界有吸引力」的門檻（資訊性，事前寫死）**：機構主要成本、分數位置、P1 ∪ P2 的年化淨 Sharpe ≥ 1.0、年化淨報酬 95% 區間下界 > 0、且 P2 單獨年化淨報酬 > 0。三者都成立記為「達標」，否則記為「未達標」；散戶欄只作對照。達標不觸發任何前瞻登記，只決定報告的結論措辭。
10. **程序與輸出**：`backtest/twbacktest/scenario_suite.py`（合成資料測試先過：空頭 band 的方向與借券費、當沖的成本與漲跌停跳過、滾動 beta 的事前性、合併的權重）；輸出 `grid.csv`、`daily.csv`、`report.json`、`summary.md`；中英雙語報告 `docs/institution_cost_scenarios_results.md` 與 Word 檔。

判讀限制（事前）：

- P2 已經看過；參數在 P1 選、P1 ∪ P2 判讀含 P1，選擇偏誤同 §13.14。
- 機構成本是假設值（手續費 0.03%、滑價 0.05%、借券 2%），未查證各券商實際報價；放空宇宙假設流動性宇宙全部可借，實際借券供給（尤其中小型股）會縮小空頭這一腿。
- 現股當沖假設所有流動性宇宙股票可先賣後買；實際當沖標的名單排除部分股票與處置股。
- 空頭位置以「每日等權重設」近似，與實際放空持倉的名目漂移不同；借券費以年率除以 252 按日扣。
- TX 的滾動 beta 以組合自身 60 個區間的報酬估計，對沖的是歷史 beta，不是當日曝險。
- 3 億與 10 億的參與率上限是硬性篩選，不是市場衝擊模型。

**§13.19 結果（2026-10-25，`twbacktest.scenario_suite`，輸出 `artifacts/diagnostics/backtest/scenario-suite-b0-d0-20261025T150000Z/`；完整表格見 [機構成本六情境報告](institution_cost_scenarios_results.md)）。** 選定參數：B0 h = 10、K = 50、m = 3、q = 5%；D0 h = 10、K = 50、m = 2、q = 5%。**36 格（6 情境 × 2 模型 × 3 市場）沒有一格達到第 9 項的門檻。** 最接近的是 D0 S2（多日多空加滾動 β 台指期疊加）兩市場合併：P1 ∪ P2 淨 +16.8% [+1.4%, +33.7%]、Sharpe 0.83，但 P2 −6.5%。判讀：（1）機構成本只讓 S1 的拖累由 9.6–17% 降到 8–13%／年，淨報酬仍為 −6.3% 到 +7.6%、P2 −26% 到 −63%，瓶頸是空頭腿（P2 空頭腿 −61% 到 −89%，對上指數 +81–89%）：持倉診斷顯示 h = 10 的排序本質上是短期反轉訊號，空頭腿放空的是近 20–60 日漲幅最大的股票（規模與宇宙相同、不是大型股），P1 反轉成立、P2 動能延續，被放空的強勢股事後年化 +75–79%（宇宙 +57.5%）、被買進的落後股只有 +21–27%；這與 §13.16 不矛盾：變體 A 恢復的是全宇宙的殘差 Rank IC，但 band 組合實際進場的前後 50 檔在 P2 的殘差目標多減空價差已為 −0.4%／10 日（P1 +0.9–1.2%）、原始 alpha 價差 −0.8% 到 −1.4%，殘差化只改了標籤裡的市場、規模與產業，沒有改組合的動能暴露；（2）TX 疊加因組合 β ≈ −0.5 而做多台指期，TX 腿在 P2 貢獻 +19–58%，是 beta 不是 alpha；（3）當沖毛年化 +123–179%（每邊每日 0.25–0.33%）對機構往返 0.31%／腿、拖累 156%／年，淨 TWSE −17% 到 −33%、TPEX +7–12% 但區間跨 0 且 P2 為負，散戶拖累 263–363%；滑價歸零時 TWSE 淨 +17%／+32%（Sharpe 0.8／1.5），即只有零滑價才成立；（4）3 億與 10 億的參與率上限擋掉多日 83–97% 的候選買單、把當沖部署率壓到 15–70%，TPEX 的正值在機構規模下消失。處置：未達標，不觸發前瞻登記；結論措辭為「在機構成本下仍無對業界有吸引力的策略」。

**§13.19 補充：反轉暴露診斷（2026-10-25 於實作與計分前登記；描述性，使用者指示執行）。** 目的：把 S1 多空組合的損益拆成「對動能與 beta 的結構性暴露」和「扣掉暴露後剩下的排序能力」，並預覽以週轉率條件化反轉是否改善 P2，作為 G1 的判讀基礎。學界與業界的對應做法：殘差反轉（Blitz 等 2013，對因子取殘差後 Sharpe 0.62 → 1.28）、產業與風格中性化（東吳證券純化反轉因子）、以流動性或成交量條件化反轉（Avramov、Chordia、Goyal 2006；Nagel 2012）。

1. **資料**：B0 `t-lgbm-alpha-residual_v1a` 與 D0 `t-xs-scratch` 的 h = 10 集成分數（同 §13.19）；TWSE 與 TPEX 各自的流動性宇宙（同 §13.14）；目標為原始遠期報酬（t+1 開盤到第 10 日收盤）、原始 alpha（對市場指數）與殘差目標 `residual_h10`（來自同一 run 的 `targets.parquet`）。
2. **每列特徵（只用訊號日及之前的資料）**：前 20 日與前 60 日對數報酬（還原收盤）；60 日滾動 beta（個股對市場指數的日對數報酬，共變異數除以變異數，少於 40 個觀測為缺值）；週轉率跳升 = log（前 5 日平均成交金額／前 60 日成交金額中位數）。
3. **分數變體**：V0 = §13.14 的平滑百分位分數；V1 = 每日在宇宙內把 V0 對截距、前 20 日報酬、前 60 日報酬、beta 做 OLS，取殘差（分數正交化）。
4. **組合建構**（每日形成一批、持有 10 日、各批等權，與 §13.17 步驟 0 相同的 cohort 法；前後四分位、金額中性）：C0 plain；C1 動能配平 = 在前 20 日報酬的五分位內分別排序與配對（兩腿的動能分布相同）；C2 週轉率條件化 = 把宇宙依週轉率跳升分三等分，各自計算 plain 價差並分開報告（預期：高跳升組的反轉價差最大，P2 亦然）。V0 與 V1 各配 C0、C1，V0 配 C2。
5. **指標**：每批多減空價差（原始報酬、原始 alpha、殘差目標）、年化毛報酬（× 252／10）、區塊 20 的 circular block bootstrap 95% 區間（1,000 次、seed 11）、依 `cost_profiles.json` 兩個成本檔估的年化成本拖累（每批兩腿進出各一次：機構約 23%／年，散戶 TWSE 約 36%、TPEX 約 44%）與淨年化；分 P1、P2、P1 ∪ P2。另報 V0 與 V1 分數對前 20 日報酬、前 60 日報酬、beta 的平均橫斷面相關，確認正交化成立。
6. **事前寫下的判讀規則**：(a) 若 V1－C1 在 P1 的原始價差保留 V0－C0 的一半以上且 P2 的原始價差不再顯著為負，暴露是 P2 虧損的主因，建構層可處理；(b) 若 V1－C1 在 P1 的價差低於 V0－C0 的四分之一，價量模型的多日能力基本上就是反轉本身；(c) 若 C2 的高跳升組在 P1 與 P2 的價差都高於低跳升組，條件化反轉有效，G1 中籌碼特徵（當沖比、融資變化、法人買賣超）做同一件事的先驗機率提高。三種結果都不改變幅度對成本的結論；本診斷不觸發任何前瞻登記。
7. **程序**：`backtest/twbacktest/reversal_exposure.py`（合成資料測試：正交化後相關為 0、分數與動能無關時 C1 = C0、三等分分組完整）；輸出 `grid.csv`、`daily.csv`、`report.json`、`summary.md`；結果寫入本節與 `docs/institution_cost_scenarios_results.md`。

**反轉暴露診斷結果（2026-10-25，輸出 `artifacts/diagnostics/backtest/reversal-exposure-b0-d0-20261025T170000Z/`；完整表格見報告第 3.8 節）。** 原始分數 V0 與特徵的平均橫斷面 Spearman 相關：對 60 日 beta −0.24 到 −0.45（P2 更強）、對前 20 日報酬 −0.12 到 −0.26、對前 60 日報酬 −0.10 到 −0.26；正交化後的 V1 全部在 ±0.08 以內。原始遠期報酬的每批（10 日）多減空價差，V0 plain → V1 動能配平：B0 TWSE P1 +0.49% → +0.62%、P2 **−1.78% [−3.06%, −0.53%] → +0.09% [−0.68%, +0.81%]**、P1 ∪ P2 −0.13% → **+0.47% [+0.07%, +0.88%]**；B0 TPEX P2 −1.46% → +0.82% [+0.16%, +1.44%]、P1 ∪ P2 +0.29% → +0.99% [+0.63%, +1.35%]；D0 TWSE P2 −1.24% → +0.16%、P1 ∪ P2 +0.02% → +0.49% [+0.12%, +0.88%]；D0 TPEX P2 −0.54% → +0.91% [+0.03%, +1.75%]、P1 ∪ P2 +0.73% → +1.12% [+0.71%, +1.51%]。**判讀 (a) 成立於四個（模型, 市場）**：P1 保留率 1.0 到 1.3、P2 不再顯著為負，P2 的虧損主要來自分數對 beta 與動能的結構性暴露，不是排序能力消失；(b) 不成立（價量模型不是只有反轉）；(c) 週轉率條件化**沒有一致效果**（高跳升組的價差只在 B0 TPEX 的 P1 與 P2 都高於低跳升組，其他三組至少一個期間相反），免費的週轉率代理無法分出會延續的強勢股。幅度對成本：V1 動能配平的年化毛價差 TWSE +12%、TPEX +25% 到 +28%，但每 10 日全數換手的機構成本拖累約 25%／年，機構淨 TWSE −13%、TPEX −0.3% 到 +3%，散戶淨 −18% 到 −24%；純化讓訊號跨期穩定，沒有讓它變大。處置：不觸發前瞻；下一個可登記的實驗是把正交化分數與動能配平放進 §13.19 的 band 建構（持有約 40 日、換手每日 2–5%，機構拖累 8–13%／年）重算 S1／S2，這是建構層的問題，與籌碼資料無關；G1 的子分析照 §13.18 補登記的設計進行。

**§13.19 補充二：正交化分數與動能配平的 band 建構（2026-10-25 於實作與計分前登記；描述性，使用者指示執行）。** 把反轉暴露診斷的兩個處理放進 §13.19 的 S1／S2：

1. **分數變體**：V0 = §13.14 平滑百分位分數（§13.19 原設定）；V1 = 每日在宇宙內把 V0 對截距、前 20 日報酬、前 60 日報酬、60 日 beta 做 OLS 取殘差（特徵定義同補充一第 2 項）。
2. **兩腿配平**：`none` = §13.19 原建構（單一排序，前 K 做多、反向前 K 放空）；`momentum` = 每日依前 20 日報酬把宇宙分成五分位，每個五分位各跑一套 band（多頭與空頭各 K／5 個位置，每日新進上限 ⌈K／5／10⌉），五個子組合等權；持有中的股票若換了五分位就離開原子組合的排序而被賣出，這部分的換手照實計入成本。兩腿因此在進場時有相同的動能分布。
3. **格子**：四個組合 (V0, none)、(V1, none)、(V0, momentum)、(V1, momentum)，其中 (V0, none) 即 §13.19 的 S1／S2 基線。**不重新選參數**：h、K、m 固定為 §13.19 已選定的值（B0：10／50／3；D0：10／50／2），避免二次選擇偏誤。只算 S1 與 S2（S2 的 TX 滾動 β 疊加同 §13.19 第 5 項）；兩個成本檔、滑價 0／1／2 倍、分數位置與機構 3 億／10 億、TWSE／TPEX／TW 合併，全部照 §13.19 的表格格式。
4. **指標與門檻**：同 §13.19 第 8、9 項；吸引力門檻按 (變體, 配平) 分別判定，只作資訊。
5. **事前寫下的預期**：依補充一的結果，(V1, momentum) 的 S1 在 P2 的毛報酬應明顯高於基線、P1 不低於基線的一半；年化成本拖累應與基線同量級（8–13%）或因五分位換手而略高；若 (V1, momentum) 的 S1 在 P1 ∪ P2 機構淨報酬為正且區間下界 > 0，代表純化後的訊號在 band 建構下付得起機構成本，但 Sharpe ≥ 1 與 P2 > 0 兩項仍需同時成立才算達標。
6. **程序**：`scenario_suite` 新增 `construction` 設定（`score_variants`、`leg_matching`）與 `multi_day.fixed` 參數；特徵計算抽成 `exposure_features.py` 與補充一共用；設定檔 `backtest/config/scenario_suite_v1.json`；合成資料測試：V1 分數與特徵無關、五分位子組合的總位置數等於 K；結果寫入本節與 `docs/institution_cost_scenarios_results.md`。

**補充二結果（2026-10-25，輸出 `artifacts/diagnostics/backtest/scenario-suite-v1-b0-d0-20261025T180000Z/`；完整表格見報告第 3.9 節）。** (V1, none)：同一組 band 參數下 S1 的 P1 ∪ P2 機構淨報酬 B0 TWSE −6.3% → **+17.3% [−0.6%, +37.3%]**（Sharpe 1.04）、TPEX −2.5% → **+19.4% [+4.9%, +34.4%]**（1.50）、TW −4.4% → **+18.3% [+5.5%, +33.6%]**（1.46）；D0 TWSE −5.0% → +10.7%（0.62）、TPEX +7.6% → **+28.8% [+13.5%, +43.6%]**（2.16）、TW +1.3% → **+19.7% [+6.3%, +34.7%]**（1.53）；P2 全部轉正（+4.7% 到 +23.2%），最大回撤 −11% 到 −18%，組合 β 由 −0.5 降到 −0.15 到 −0.20，換手每日 3–5.5%、拖累 9–15%／年。S2 加 TX 腿再加 +2% 到 +9%。**第 9 項門檻（機構）：10 格達標**（B0 S1 TPEX／TW、B0 S2 三市場、D0 S1 TPEX／TW、D0 S2 三市場），B0 TWSE S1（下界 −0.6%）與 D0 TWSE S1（Sharpe 0.62）未達；散戶成本下 B0 TW S1 +14.7% [+1.9%, +29.7%]、S2 +19.6% [+6.5%, +34.2%] 亦為正。(·, momentum)：五分位子組合因持股換五分位即被賣出，持有由 20–43 日掉到 4 日、換手 10%／日、拖累 20–25%／年，P1 全部為負，此建構不可行；正交化單獨即可。容量：3 億元 B0 TW +21.8%（1.71）、TPEX +28.0%（1.99）、D0 TW +9.0%；10 億元 81–97% 候選買單被擋、淨 −8% 到 +11%。**判讀限制**：正交化沒有擬合參數、band 參數固定，但「對 beta 與動能正交化」是看過 P2 失效後才決定的設計，P2 結果不是樣本外證據，達標只在資訊層面；空頭腿假設流動性宇宙全部可借券而最強結果在 TPEX；絕對報酬 +17% 到 +29% 仍低於同期 TAIEX 報酬指數 +41%，贏的是市場中性的風險調整報酬。處置：本節結論由「無對業界有吸引力的策略」改為「**正交化後的多日多空在機構成本下達到資訊性門檻，待 P3 確認**」；登記 P3 前瞻（2026-06 之後，V1、無配平、固定參數、兩個成本檔）為下一個實驗；G1 的分期與分強勢股子分析照 §13.18 進行，並以 V1 分數為基線。

**§13.19 補充三：補充二結果的穩健性檢查（2026-10-25 於計分前登記；描述性，使用者指示執行）。** 補充二的 Sharpe 1.0–2.2 在 3.4 年樣本的標準誤約 ±0.7–1.0，且設計在看過 P2 後決定；本補充不改變任何判定，只回答兩個問題：(a) 結果是否靠單一年份撐起；(b) 結果是否只在 §13.19 選定的那一組 (h, K, m) 成立。

1. **逐年表**：補充二主要成本、分數位置的每日淨序列（`daily.csv`），依訊號日的曆年（2023、2024、2025、2026 至 5 月）各算：淨報酬（複利）、年化 Sharpe（該年日報酬的均值除以標準差乘 √252）、最大回撤、勝率、交易日數；對 S1 與 S2、V0／V1、無配平、兩個模型、三個市場、兩個成本檔各列一組。
2. **V1 全格子**：以 V1 分數、無配平，對 h ∈ {5, 10} × K ∈ {30, 50} × m ∈ {2, 3} 的 8 組參數各算 S1，TWSE 與 TPEX 分別，機構主要成本、分數位置，分 P1、P2、P1 ∪ P2 列出年化毛報酬、淨報酬與區間、Sharpe、換手、拖累。**不重新選參數**：補充二的結論仍以 §13.19 選定的 (10, 50, 3)／(10, 50, 2) 為準，全格子只用來看鄰近參數是否同向。
3. **借券宇宙**：請 `tw_chip_data` 第一階段新增免費品項「得為融資融券有價證券」與「得為借券賣出有價證券」每日名單（證交所、櫃買各一；以公告日起生效，`available_at` 同 T+1 開盤前規則）。之後補充二的空頭腿只在可借券名單內選股並重算（登記為補充四，待資料到位），這是 TPEX 結果能否成立的前提。
4. **事前寫下的判讀**：(a) 若四個曆年中任一年的 S1 V1 淨報酬為負且該年占 P1 ∪ P2 總淨報酬的貢獻 < 0，記為「非單年撐起」只有在其餘年份皆為正時成立；若 P1 ∪ P2 的正報酬有一半以上來自單一年份，報告必須寫明。(b) 若 8 組參數中至少 6 組的 P1 ∪ P2 機構淨報酬為正、且沒有一組的 P2 顯著為負（區間上界 < 0），視為鄰近參數同向；否則補充二的結果對參數敏感，達標格數要打折解讀。
5. **程序**：`scenario_suite` 寫出 `yearly.csv`（每個最終序列的逐年摘要）並允許 `construction.grid_score_variant` 與 `multi_day.grid_markets` 讓選擇格子以 V1 分數在多個市場計算；設定檔 `backtest/config/scenario_suite_v1_grid.json`；結果寫入本節與報告第 3.10 節。

**補充三結果（2026-10-25，輸出 `scenario-suite-v1-b0-d0-20261025T190000Z/yearly.csv` 與 `scenario-suite-v1-grid-b0-d0-20261025T190000Z/`；完整表格見報告第 3.10 節）。** (a) 逐年（S1、V1、機構）：B0 四個曆年全部為正，TWSE +1.0%／+16.8%／+12.4%／+23.2%（2026 至 5 月）、TPEX +24.4%／+22.5%／+3.0%／+12.7%、TW +12.3%／+20.0%／+7.9%／+18.1%，無單一年份占總淨報酬一半以上 → **非單年撐起**；D0 TWSE 2025 年 −1.5%、D0 TPEX 由 +48.4%／+36.7% 遞減到 +12.7%／+3.3%，前兩年占八成以上，D0 的結果**集中在 2023–2024 且逐年遞減**，是選擇效應或訊號衰減由 P3 分辨。(b) V1 全格子：四個（模型, 市場）的 8 組參數 P1 ∪ P2 機構淨報酬全部為正，TPEX 16 組區間下界全部 > 0，P2 只有兩組為負且區間跨 0 → **鄰近參數同向**；§13.19 選定的參數在 V1 格子居中而非最佳。(c) 借券名單已向 `tw_chip_data` 提出（第一階段新增品項），補充四待資料到位。兩項檢查不改變補充二的限制（事後設計、可借券假設、絕對報酬低於指數）。

**§13.19 補充四：執行不確定性與可借券宇宙（2026-10-26 於實作與計分前登記；使用者指示）。** 使用者指出：名單上「可以借」與「借得到」是兩回事；買賣至少能掛市價、以滑價承受，借券沒有這個退路，所以放空的可得性要用機率模擬，而不是假設全部借得到。極端的流動性鎖死（漲跌停鎖住以外）先排除。本補充重算補充二的 S1／S2（V1、無配平、固定參數）並加入四層：

1. **可借券宇宙**：空頭腿候選只取當日「平盤下得融（借）券賣出」名單內的股票（機構用借券名單、散戶用融券名單；名單缺值日沿用前一有效名單並計數）。不在名單 → 借不到（機率 1）。
1b. **可借量上限（2026-10-26 資料到位後補登記，先於第 2 項套用）**：`tw_chip_data` 的 `sbl` 表每檔每日有交易所公布的「次一營業日可借券賣出限額」`sbl_next_quota`（股）與融券的 `short_limit`、`short_balance`；訊號日 t 的列在 t 23:59:59 可用、t+1 開盤生效。機構的空頭腿每檔新進名目 ≤ `sbl_next_quota` × 執行日開盤價，散戶 ≤ (`short_limit` − `short_balance`) × 開盤價；限額為 0 的候選跳過並計數，限額不足位置的候選依封頂規則縮小部位。這是可觀測的管制上限，不是實際出借供給，所以第 2 項的機率層仍保留，作為供給面的敏感度。名單覆蓋面：`eligibility` 每日約 TWSE 1,140 檔、TPEX 744 檔，`sbl_eligible` 約 99%，名單本身只排除少數暫停或跌停封鎖的股票，主要的限制來自本項的限額。
2. **借不到的機率**：名單內每檔每日以 Bernoulli 抽樣決定能否借到，機率依訊號日 20 日成交金額中位數在宇宙內的三等分分層。主要假設：機構（借券）大／中／小 = 2%／10%／25%，散戶（融券）= 5%／20%／50%；敏感度 × 0（全部借得到，即補充二）與 × 2。抽樣失敗 → 該候選跳過、換下一名（計「借不到」次數與比例）；已持有的空頭部位另加每日被召回（借券回補）機率 0（主要）與 0.5%（敏感度），被召回者次一開盤回補並付成本。隨機數以 seed 11 起 5 個重複，報告 5 個重複的平均與極差，區間用平均序列的 block bootstrap。
3. **借券費分層**：機構借券年費依同一三等分 1.5%／3%／6%（取代單一 2%）；散戶融券手續費 0.08% 不變、保證金 90% 留作擔保品不生息。
4. **買賣的可得性**：買進與賣出視為市價單必成交，代價是滑價（主要與 2 倍滑價已在成本敏感度內），漲跌停鎖住的開盤照既有規則不買不賣；不另設成交機率。
5. **資本情境改為「封頂而非跳過」**：原 band 引擎在 3 億與 10 億對超過 20 日成交金額中位數 5% 的候選是整檔跳過、換下一名，導致組合被迫往排序後段的大型流動股移動（§13.19 的 3 億結果因此劣於分數位置，見本補充第 7 項的說明）；改為部位名目封頂在 5% × 成交金額中位數、超出部分留現金（與當沖情境相同；封頂後不足位置 10% 的候選仍跳過），報告名目部署率與被封頂比例。兩種規則並列一次以示差異，之後以封頂為準。**第 5 項先於其餘各項實作（2026-10-26，`simulate_band` 的 `cap_mode`，`cost_profiles.json` 的 `participation_mode = scale`），因為它是模擬規則的錯誤而非資料問題；§13.14 的 `tw_strategy` 設定保留 `skip` 以維持舊結果可重現。**
6. **指標與判讀**：同補充二；另列借不到比例、召回次數、名單過濾掉的候選比例、名目部署率。事前判讀：若主要假設下 B0 的 S1 在 TW 的 P1 ∪ P2 機構淨報酬仍為正且區間下界 > 0，補充二的結論在可得性修正後成立；若轉負或區間跨 0，報告寫明「補充二的結果依賴借得到小型股」，並以 TWSE 單市場、大中型分層的結果作為可行版本。
7. **對 §13.19 資本情境結果的說明（事後解讀，不改數字）**：3 億的虧損不是成本，是選股被規則改寫：位置 600 萬元對 5% 參與率上限需要每日成交金額 1.2 億元以上，TPEX 門檻 1,000 萬元起的宇宙大半不合格，候選買單被擋 83% 到 97%，組合實際買到的是排序後段的大型流動股，排序能力被稀釋；1 億元的位置 200 萬元只要求 4,000 萬元成交金額，TWSE 宇宙幾乎全部合格，所以與分數位置幾乎相同。跳過規則把容量問題變成了選股問題，第 5 項的封頂規則才是業界實際的做法。
8. **程序**：`scenario_suite` 新增 `execution` 設定（名單路徑、分層機率、召回機率、費率分層、重複數、封頂規則）；合成資料測試：機率 0 時與補充二逐列相同、機率 1 時空頭腿為空、封頂後部署率小於 1；設定檔 `backtest/config/scenario_suite_v1_exec.json`；結果寫入本節與報告新節。

**補充四第 5 項結果（2026-10-26，`scenario-suite-v1-scale-b0-d0-20261026T010000Z/`，V1、無配平、機構主要成本、P1 ∪ P2；報告第 3.9 節有跳過與封頂並列的完整表）。** 封頂規則下 3 億與 10 億全部為正且區間下界 > 0：B0 TW 3 億 +13.1% [+3.4%, +24.1%]（Sharpe 1.32、部署率 67%）、10 億 +15.2% [+5.6%, +25.3%]（2.02、47%）；B0 TPEX 10 億由跳過規則的 −5.8% 變為 +20.5% [+7.5%, +33.6%]；D0 TW 3 億 +14.0%、10 億 +12.7%；D0 TWSE 10 億由 −6.1% 變為 +7.9%。候選被擋比例由 81–97% 降到 1–30%（TWSE）與 26–87%（TPEX，10 億時仍有 85% 連位置的 10% 都裝不下），被封頂比例 10–51%。判讀：§13.19 原資本情境的虧損是跳過規則造成的選股改寫，不是策略在大資金下失效；封頂後的真實形態是名目部署率 40–80%、淨報酬以總資本計被稀釋，部署資本的報酬高於分數位置。自此所有 `scenario_suite` 回測以封頂為準；§13.14 的 `tw_strategy` 保留跳過以維持舊表可重現。第 1–4 項仍待可借券名單。

### 14. 遠端驗證邊界

Python environment、Poetry lock、lint、pytest、data preparation、model prefetch、GPU
training 與 validation 全部在 RunPod 執行。本機只做 source/config 編輯、shell 靜態
檢查、S3 sync、Pod 控制與 artifact 下載。本機 Python/PyTorch/CUDA 狀態不是本專案
runtime 證據。

---

## English

### 1. Research question

Under strict point-in-time inputs, next-open execution, and chronological
evaluation, can a finance-K-line-pretrained Kronos-base with LoRA predict an individual stock's
or ETF's `h_start`-through-14 holding-day benchmark-relative alpha and absolute direction
more accurately than simple rules, traditional ML, and compact causal DL
baselines? Each `a-signal64`/`b-signal64` group/seed uses one experiment workflow and one Pod
to train `alpha_point` then `absolute_direction`, testing independent robust point regression
and fixed-return-band classification. Existing quantile and cost-sensitive dual-direction
experiments remain controls; their outcomes are not interchangeable task results.

Validation Sharpe or one backtest is not sufficient proof of tradable alpha,
and model output is not a complete trading rule. Technical baselines measure
incremental model value. Retained `direction64` V1/V2/V3 multitask presets inject ten explicitly
defined historical technical channels into the Kronos predictor without replacing it with
baseline rules. The latest two independent tasks do not enable those technical channels.

### 2. Fixed assumptions and scope

- Instrument-level, not portfolio-level, forecasting.
- Daily OHLCV only.
- US and Taiwan stocks/ETFs; no futures or options in this PoC.
- Signal after close `t`; entry at the next market session's raw open, without skipping missing bars.
- Entry day counts as holding day one.
- `h_start` is configurable as 1, 2, or 3; horizons run through fixed day 14.
- Alpha labels are benchmark-relative adjusted execution log returns; independent absolute direction uses the stock's adjusted simple return and fixed ±1% band.
- CAPM abnormal return reserved for a diagnostic/ablation.
- Independent point regression uses normalized Smooth L1 and independent direction uses CE, without shared LoRA updates or mixed task losses. Legacy quantiles use pinball and a ranking head uses same-day loss; `direction64` V1/V2 retain cost-based softmax CE, whereas V3 uses conditional ordinal three-class NLL. All three retain five-level quantile post-processing.

### 3. Model hypothesis

The independent-task hypothesis asks whether separating robust alpha-point prediction from
absolute-stock down/neutral/up classification improves reproducible holdout results against
task-aligned baselines with the same Kronos-base/LoRA-64, data and scale path. Each task has
its own fine-tuned encoder, head, optimizer and checkpoint; separation is not presumed superior.
Both initialize independently from the same pretrained revision, without shared trainable
parameters or mixed losses. One parent workflow manages both child models, rather than requiring
two manually launched experiments. Sequential execution avoids holding both full models in GPU
memory at once; different seeds can run concurrently on separate Pods.
The ±1% band is an initial candidate, not a proven optimum. Full-train market/horizon class
audits must pass before fitting; test labels never select the threshold.

The retained multitask hypothesis is that a shared finance-pretrained encoder, historical-benchmark
conditioning and readable past-only scales improve cost-sensitive absolute-stock/alpha
direction discrimination on identical data/splits without buying apparent classification gains
with uncontrolled quantile errors. V3 selects by population probability quality, macro-F1 and
decision-error guards; V2 retains equal-market/task/horizon balanced-decision BA as a control.
All retain raw population accuracy, majority baselines, up precision/recall and
normalized pinball. Original quantile, correlation, ranking and calibration hypotheses remain
controls. Neither one horizon's majority accuracy nor higher coverage establishes direction
skill, and this version does not presume 60% accuracy.

If zero-return, past-only momentum, GBDT, GRU, DLinear, or PatchTST consistently
outperforms the full model under the same protocol, no foundation-model gain may
be claimed.

### 4. Dataset identity

Every run binds dataset profile and selected sources; raw Parquet, bar-store
manifest, symbol-index, and cutoff-range hashes/sizes/counts; provider, market,
symbol, type, and date provenance; benchmark-mapping hash; the
`h_start`-independent storage preparation and pipeline digests; the training
selection's `h_start`, fixed `max_horizon=14`, and full horizon sequence; the
train-only runtime-calibration sample count, seed, and checkpoint-persisted robust scales; split
counts; and cutoff-exclusion audit. Formal comparisons cannot mix profiles,
universes, dataset versions, or benchmark mappings. EODHD versus official
Taiwan source differences must be disclosed.

### 5. Labels and causality

For cutoff `t` and horizon `h`:

```text
entry = next market session's raw regular-session open; never postpone a missing entry
exit  = h-th shared holding day's raw close
alpha_h = log(asset adjusted gross return) - log(benchmark adjusted gross return)
```

Instrument and benchmark use identical timestamps. Inputs contain only both
point-in-time adjusted histories through `cutoff_at`. Future entries, exits,
returns, CAPM diagnostics, and labels must never enter input tensors.

Tests cover future-benchmark label-only changes, global adjusted-scale
invariance, split continuity and volume direction, cutoff-bounded contexts, and
fail-closed benchmark calendar gaps.

### 6. Splits and stages

Build compact runtime ranges over existing bars using complete market sessions
and minimum liquidity. Production train precedes 2025-06-01; validation is
[2025-06-01, 2025-12-01), test [2025-12-01, 2026-06-01). Last label dates must
precede each upper boundary. A/B fully traverse the same validation/test source.
See README and `configs/data_cleaning.json` for cleaning and annual weighting. Runtime
sample indexes and main-model DataLoaders do not pre-expand all windows/labels; baselines
separately build disk-backed input/label caches in bounded batches. Both retain genuine extreme returns.

| Item | Stage 1 | Stage 2 |
|---|---|---|
| Purpose | Validate the complete script/data/GPU/loss/checkpoint/validation path | Full-train PoC result |
| Train samples | Dynamic annual weighting; 5% of eligible count per epoch, capped at 500,000 | Dynamic annual weighting; presentations equal the full eligible count |
| Epoch limit | 2 | 5 |
| Validation cadence | 20%/40%/60%/80%/100% of every epoch | Same |
| Early stopping | Five consecutive non-improving validations of the configured monitor and two training intervals at the minimum LR; active from epoch 1. Independent point/direction use all-horizon Smooth L1/CE; original quantiles use normalized pinball; V1 uses fixed-weight pinball plus both CEs; V2 uses equal-market BA and a pinball guard; V3 uses probability/decision quality and that guard (sections 7 and 9) | Same |
| Retention | Best five full validation-ranked checkpoints plus completion result | Same |
| Validation/test | Every eligible window, without subsampling | Same |
| Architecture | Kronos-base + LoRA + shared resampler + conditioner + alpha head | Identical |
| Initialization | Original pretrained base | Original pretrained base |
| Continue Stage 1 checkpoint | No | No |

The architecture row describes the standard quantile Stage 1/2 presets; independent-task
heads and acceptance criteria are specified in sections 7 and 12.
Stage 1 is not the earliest 5% of time and does not shrink validation/test.
Both standard default configs require the same architecture digest. A short final training batch
is deterministically filled from the beginning of the same target set, and the
padding count is written to the training summary. Stage 2 restarts from the same
pretrained revisions so the Stage 1 smoke run is not hidden extra training.

### 7. Objective

#### 7.1 Current primary path: two independent tasks

The two child models in `a-signal64`/`b-signal64` each train and select by one objective:

```text
alpha_point: mean(SmoothL1((prediction - alpha) / train_scale[h], beta=1.0))
absolute_direction: mean(CrossEntropy(logits[h], fixed_return_class[h]))
fixed_return_class: down if return < -0.01; up if return > +0.01; otherwise neutral
```

The direction return is the stock's adjusted simple return from next-open to horizon-close,
not alpha and not cost-adjusted. Equality at either boundary is neutral. Comparing log-return
labels to `log1p(-0.01)` and `log1p(+0.01)` implements the same rule. Both tasks average valid
samples/horizons equally, without quantile, ranking, legacy dual-direction/perturbation losses
or interval calibration. `alpha_point` is a robust location, not labelled q50 or guaranteed
to be a conditional mean. Full-validation task loss drives checkpoint selection/early stop
at the stable `primary_5d/selection_score`, mode `min`, covering all horizons rather than day five.
Training explicitly records `train/normalized_smooth_l1` or `train/cross_entropy`, alongside
`train/loss` and `validation/loss`. Selection scores cannot be ranked across tasks.

Both retain paired Kronos/LoRA, the resampler, benchmark conditioning, historical-scale and
direct-benchmark numerical paths, market embeddings and horizon embeddings. The head mean-pools
latent tokens before horizon-specific reading; it does not retain V3's ordinal direction head,
horizon-query attention, market-state gating, train priors or PCGrad. Point outputs are multiplied
by fixed train-fitted robust scales. Here `explicit_output_scale` retains extended scale features,
not per-sample volatility multiplication of predicted locations. Neither task outputs interval
widths or trading rules.

#### 7.2 Retained controls: quantiles and V1/V2 multitask objectives

The following objectives apply to retained `quantiles` experiments. Their primary objective
is q10/q50/q90 pinball loss, equally averaged after
train-only per-horizon normalization. Training adds same-day ranking loss through
an independent score head (default weight 0.05), not directly through q50.
Original presets select by uncalibrated normalized pinball:

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
selection_score = mean_h(normalized_pinball_h), h=h_start,...,14
```

The aggregate is also exposed at `primary_5d/selection_score` to preserve the
stable RunPod checkpoint-monitor path. It still covers every horizon.
Checkpoint mode is `min`.

`direction64` starts from LoRA64 settings and adds cost-band classification, horizon queries,
ten technical channels and representation regularization. Costs assume Taiwan 1,000-share
lots and US direct-broker 50-share units, using cutoff-known prices only. Definitions and
primary references are in [README architecture](../README.md#model-en).
V1 estimates frozen `sQ,sAbs,sAlpha` by forwarding at most 1,024 train-only windows through the
untrained model. Even with a fixed calibration-sample seed, these values can depend on model
initialization seed. Training uses
`pinball + (sQ/sAbs)*CE_abs + (sQ/sAlpha)*CE_alpha + 0.05*ranking + 0.05*sQ*adversarial`,
with equal horizon weights. Full-validation `direction/selection_loss` includes only the
first three terms; old pinball fields retain their meaning. Checkpoints persist references
without refitting on resume. Completed dual-direction baselines retain their independent
objectives. V2 does not change labels, data rules or cost scenarios and therefore requires
neither baseline retraining nor repeated CPU preparation.

The original `direction64` remains reproducible. `a-direction64-v2`/`b-direction64-v2` use separate
`a_direction64_v2.yaml`/`b_direction64_v2.yaml` files. Its direction head directly reads the
existing 20 train-standardized scale features, plus four bounded inputs formed by cost thresholds
divided by asset/relative 20-day volatility times `sqrt(horizon)`. Ratios are bounded to
`[-20,20]` then transformed with `asinh`, without changing exact label thresholds or clipping
genuine returns. History signal `f` is separated from train prior `π`; population logits are
`z=f+τ*log(π)`, default `τ=1`. Natural-population `CE(z)` uses fixed train market weights,
not an additional inverse-class multiplier. Capped water filling uses annually weighted
train-population proportions `p_m`: `w_m=min(k/p_m, market_weight_max)`, solving `k` so
`sum(p_m*w_m)=1`. The train-weighted mean is 1 and the final effective weight cannot exceed
`market_weight_max` (default 3); this is not clipping followed by normalization.
Population probabilities are `softmax(z)`;
`argmax(f)` is separately reported as `balanced_decision`, not substituted for raw accuracy
or interpreted as a trading win probability. Motivation comes from
[Menon et al.](https://research.google/pubs/long-tail-learning-via-logit-adjustment/).
Inference direction schema 2 identifies probabilities with `probability_source="population_logits"`.
`balanced_decisions.absolute`/`.alpha` map each holding period to class indices (`0=down`,
`1=neutral`, `2=up`) from `history_residual_logits_without_train_prior`, not probabilities.

At most 8,192 train windows are sampled in bounded market/year strata with fixed calibration
seed 59. Annual-decay-weighted aggregates recover the training population, **without any
model forward**. References are train pinball for zero-centered scaled null quantiles,
`log(3)` and `log(3)`, so training seed or untrained-model initialization cannot alter task
coefficients. Priors, fixed market weights, references and sampling provenance persist and
restore on resume. Horizon losses remain equally weighted.

[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
projects three-task gradients only on shared trainable LoRA parameters when their actual dot
product is negative. Non-LoRA parameters and representation regularization retain ordinary
total-loss gradients. No conflict means no gradient change; `monitor` only measures.
Both production training and GPU batch-probe clipping set `error_if_nonfinite=True`; gradient
management does not mask nonfinite values, and detecting a nonfinite gradient norm raises before
that step's `optimizer.step()` rather than updating weights.
Norms/cosines/negative conflicts and nonfinite counts are summarized every 100 optimizer steps.
The exact training-step position is recorded in `gradient/runtime_optimizer_step`. The tracking
canonical step may denote the latest logging boundary, not the exact GPU-update step or timestamp.
The conservative extra-buffer peak of `28 × LoRA numel` bytes for three tasks cannot exceed
`512 MiB`. Batch autotuning uses the same production backward path rather than underestimating
AMP/accumulation cost with ordinary backward. The literature motivates a test, not a claim of
observed conflicts or improved direction performance on this dataset.

V2 minimizes its own definition of `direction/selection_loss`:

```text
selection_loss = 1 - mean_market(mean_task(mean_horizon(balanced_decision_BA)))
                 + max(pinball/first_full_validation_anchor - 1 - 0.01, 0)*1
```

BA averages class recalls with true support in each slice; supports remain explicit.
The first **complete full** validation pinball freezes the anchor, with 1% relative degradation
tolerance and penalty weight 1 beyond it. The anchor is set once, checkpointed and restored
on resume. Test participates in no fitting. Each run's first-validation anchor is independent,
so `selection_loss` must not rank seeds. Cross-seed comparisons use the identical holdout's
raw pinball and separately reported raw/balanced direction metrics.
Configuration ranges and CLI selection are in
[README objectives](../README.md#training-en).

A/B estimate robust scales from their own training labels, so those scales may differ.
Normalized pinball and normalized Smooth L1 support within-run checkpoint selection, but their
cross-group magnitudes alone cannot establish superiority. Point comparisons use shared-holdout
raw MAE, RMSE, correlation, sign agreement and within-market RankIC; quantile controls additionally
compare raw pinball and interval quality. Fixed-label independent classification compares CE,
accuracy, BA, macro-F1 and class support on the same holdout, not coverage.

### 8. Comparator suite

Store baselines independently by task, but use one `baseline --experiment a-signal64`
(or `b-signal64`) workflow. It checks both task caches locally and rents no Pod when both are complete.
Otherwise one Pod builds point and same-±1%-label classification families sequentially, skipping
completed work. `readiness --baseline --experiment a-signal64` checks both current cleaning rules
and completion. Paired entries do not change active selection; advanced individual-task entries
remain in the README appendix. Task results are not interchangeable. Runs sharing data and task
reuse the completed baseline across training seeds; main-model LoRA capacity/seed does not
trigger retraining. New baselines do not redo CPU preparation. Neural point baselines use
normalized Smooth L1. Trees train with absolute error (MAE), but still select by full-validation
normalized Smooth L1; reports must disclose this training/selection objective difference.
Classification uses neural CE/tree log loss. Rules predict hard classes; their
probability scores use deterministic predictions with an explicit numerical floor, not learned
or calibrated probabilities. Full validation/test still cover the shared eligible membership.

Neural baselines randomly permute and traverse all train samples once per non-early-stopped
epoch; GBDT also fits all train rows without annual `sample_weight`. Main-model presets use
dynamic sampling with `yearly_sampling_decay=0.8`. An identical eligible population therefore
does not imply identical annual training weights. Neural baselines retain validation-plateau
and low-LR early-stop gates; GBDT uses full validation, patience and the best fitted model,
without the neural optimizer's low-LR schedule. Rules perform no gradient training. Reports
must not describe every training policy or compute budget as identical.

Baseline identity includes task, label/loss parameters and baseline numerical implementation,
not main-model LoRA capacity or training seed. Each task preserves independent weights or rule
parameters, best-validation results and full-test results. Neural resume restores optimizer,
scheduler, RNG and sample cursor; GBDT retains warm-start models and horizon/iteration cursors.
Legacy quantile/cost-direction results cannot satisfy new-task completion. Main-model final
testing reuses baseline results after task, population and scale checks, without baseline
retraining or inference.

The following comparator details apply to existing quantile/cost-direction experiments.
Independent tasks retain the same model families with the aligned outputs/objectives above,
without synthetic quantiles:

All comparators read the same cutoff-inclusive asset/benchmark contexts and use
the same eligible train universe and validation split, with the annual-weighting differences above:

- constant distributions: always-buy and zero-return
- rules: relative momentum/reversal, MA crossover, RSI, MACD, volatility-scaled
- traditional ML: per-horizon quantile GBDT
- causal DL: paired GRU, DLinear, compact PatchTST
- full model: Kronos-base LoRA plus dynamic benchmark conditioning

Rule residual quantiles, GBDT/neural parameter fitting, and robust scales use train
labels only. Validation selects models/checkpoints and controls early stopping.
Neural and GBDT baselines also train absolute-stock/alpha three-class tasks with costs,
horizons and full evaluation membership matching the directional main model. Rules supply
hard classes without fictitious CE. Learned baselines select by their own fixed-training-reference
pinball-plus-dual-CE objective while retaining quantile metrics. Cross-architecture comparisons
use task accuracy, balanced accuracy, macro-F1, up precision/recall and signal fraction,
not reference-dependent weighted selection losses.
Frozen main-model interval-tail calibration fits validation only after checkpoint selection;
test never tunes these parameters. Integrated `adaptive64`/`adaptive128` add market residuals,
numerical/benchmark ranking paths, market-weighted training pinball, sample-clock decay and
stronger dropout/weight decay. The original six capacity presets remain available; defaults
and boundaries are documented in [README](../README.md#training-en).

### 9. Validation metrics

Independent tasks minimize full-validation all-horizon normalized Smooth L1 (`alpha_point`)
or CE (`absolute_direction`). Point diagnostics include MAE, RMSE, Pearson, sign agreement and
valid date-level RankIC; classifiers report accuracy, BA, macro-F1, class shares and confusion,
not coverage. Selection metric for original presets: all-horizon normalized pinball; `direction64`
minimizes fixed-reference pinball plus both CEs; `direction64-v2` uses equal-market direction
BA and a frozen pinball guard. Different selection-loss definitions cannot be ranked directly.
V3 uses population CE/Brier ratios against train priors, macro-F1, opposite-error/collapse penalties
and the frozen pinball guard; section 13 defines the complete formula.
Retained `direction64` V1/V2/V3 presets additionally report absolute/alpha
confusion matrices, accuracy, balanced accuracy, macro-F1, majority accuracy, up precision,
recall and signal fraction by horizon, market and month. Classification precision is not
realized trading profit, and three-class accuracy is not the old binary q50 sign agreement.
V2 separately reports `balanced_decision` confusion/accuracy/BA/macro-F1 while preserving
raw accuracy and population probabilities without renaming or overwriting them. Probability
diagnostics include train-prior CE skill, 20 confidence bins and joint-up thresholds/precision/counts.
Daily confusion aggregation has a hard 4,096-entry limit: overflow fails rather than subsamples.
These probabilities are not assumed calibrated.

For every quantile-task horizon report raw/normalized pinball, q50 raw/normalized MAE, q50
Pearson correlation, q50 sign agreement, q10–q90 coverage, width, and coverage
error. These quantile controls retain research diagnostics including date-level RankIC/IR, equal-weight
long/short return, Sharpe, drawdown, turnover, five-level signal distribution,
and slices by market, asset type, provider, and year.

Independent point evaluation publishes only within-market/horizon RankIC/IR from cross-sectional
metrics, not long/short returns, Sharpe, turnover or five-level signals. Independent classifiers
publish no return ranking. Both independent tasks report market/month and other subgroups,
plus input-context historical-drawdown slices at 20%/40%; these are diagnostics, not sample
exclusion thresholds. Neither task has quantile coverage or interval width.

Cross-sectional diagnostics are neither portfolio-model outputs nor checkpoint
selection metrics. Transaction costs must be explicit, and one Sharpe is not
proof of tradability.
New RankIC uses `spearman_average_ties_v1`: ties receive average ranks, and dates with constant
predictions/targets have no valid correlation. Reports must read `rank_ic_status`,
`rank_ic_evaluated_dates` and `rank_ic_ir_status` instead of presenting no-valid-date numeric
placeholders as estimates. Historical unmarked values have not automatically been recomputed.

### 10. Test policy

Production RunPod training selects the best checkpoint using full validation;
after training completes, the same Pod automatically runs the full test benchmark. When
interval calibration is enabled, the selected checkpoint is fixed, calibration
fits full validation only, and test retains both raw and calibrated metrics. Retained adaptive
quantile presets additionally report `calibrated_static` and a separate `calibrated_online` prequential
diagnostic. Online width updates use only returns whose scheduled exit occurred strictly
before the forecast date under the existing market calendar; q50, the model and the frozen
artifact are unchanged. Never present this as a fully frozen holdout result.
Baseline test scores come from the completed cache without refitting or inference.
Only after all models within the run pass test-membership and return-scale consistency checks
is `test_unlocked=true` published. Test never selects checkpoints, controls early
stopping, frozen calibration or online hyperparameters. If test results drive model changes, that population
has become an iterative research evaluation set. Independent generalization claims
then require a new time interval not used in those decisions; a different dataset
version alone does not restore independence.

### 11. Checkpoint and reproducibility

Production Stage 1/2 permits neither `max_steps` nor fixed-step cadence. The
target set, batch size, gradient accumulation, and epoch count determine the
optimizer budget, with exactly five validations scheduled per complete epoch.

Independent point/direction and original quantile presets select by their task-specific
`primary_5d/selection_score`; retained `direction64` V1/V2/V3 use their version-specific
`direction/selection_loss`. An acceptable checkpoint is full-validation-selected, stores the trainable union,
optimizer/scheduler/RNG/resolved config, binds run/data/model/source revisions,
passes artifact and strict-key verification, and declares
`model_output_schema_version=5.0` for quantiles/retained multitask models or `signal-1.0`
with an explicit `prediction_task` for independent tasks. Mismatched tasks, schemas and
architectures fail closed; resume cannot cross tasks or direction versions. Confirm success
using terminal lifecycle, run manifest, best pointer, trainer state, and raw
validation JSON—not a W&B chart or README alone.

V2 also persists direction-schema-2 train priors, fixed references, market weights, the first
full-validation anchor and PCGrad random-order/step state. The new direction head requires a
new run, not an original `direction64` checkpoint resumed as v2. Old presets remain reproducible.

The leaderboard retains at most the best five full resumable checkpoints.
Normal completion and early stopping also save one `completion-result/` with
the final trainable weights, resolved config, stop reason, actual step/sample
counts, and validation metrics, without duplicating optimizer/scheduler state.

### 12. Minimum acceptance

Stage 1 uses reproducible annual-weighted dynamic sampling, presenting 5% of the
eligible train count (capped at 500,000) per epoch for up to two epochs. Early stopping cannot
trigger until five consecutive non-improving validations and two training intervals
at the minimum LR have both occurred; it is enabled from epoch 1. Stage 1 passes remote ruff/pytest,
forward/backward, validation-ranked save/reload, inference-schema smoke, and
lifecycle termination. It outputs ordered `[B,15-h_start,3]` alpha quantiles for
`h_start in {1,2,3}`, without natural-language/fact generation. Standard presets have no classifier;
direction presets additionally validate both three-class heads.

Standard Stage 2 matches the standard Stage 1 architecture. Independent point/direction presets
have their own task/architecture and cannot resume across tasks; all retain the same pretrained
revisions and dynamic sampling over the full training population. Point output is
`[B,15-h_start]`; direction logits/probabilities are `[B,15-h_start,3]` with the fixed-return band,
not quantile/ranking/cost-classification outputs. Baselines use one aligned protocol. Raw JSON
persists task-specific horizon/aggregate/subgroup metrics, plus valid cross-sectional diagnostics
for point/quantile tasks, not fabricated return rankings for classifiers. Provenance is retained
and provider code remains outside training.

Stage 1 proves script viability only. A single-seed Stage 2 is still a PoC.
Stronger claims require multiple seeds, dispersion, and predefined ablations.

### 13. Priority ablations

The current primary path uses `a-signal64`/`b-signal64` to manage two independent models, with
at least three training seeds (42, 43, 44) within one data group and each model's completed task-aligned
baselines. First compare same-task shared-holdout metrics and seed dispersion, then study capacity
or another single factor; never compare point and classifier selection scores against each other.
Use `train --experiment a-signal64 --seed 42 --seed 43 --seed 44` (`b-signal64` for B):
three Pods, three parents and six independent models. Separate commands per seed remain available
for choosing different GPUs. Each Pod's runtime applies to the whole pair, not separately to each model.
Parent `status` is COMPLETE only after both correct tasks finish training and full test; `resume`
skips completed children and finishes outstanding training/test, and `download` obtains both child
results without duplicating weights. Complete commands are in
[README training operations](../README.md#train-en).

The retained historical multitask comparison is `direction64-v2` versus `direction64-v3` within
one data group, also across multiple seeds, with V1 `direction64` as the reproduction control.
It compares bundled paths, priors, gradient management and selection, not a single component's
causal effect or the architecture of the current two single-task heads.
The original `adaptive64`/`adaptive128` capacity control remains available with everything else
fixed and only LoRA rank/alpha changed; comparisons against `lora64` also evaluate a bundle.
`partial` remains secondary and does not replace the Kronos/LoRA backbone.

V3 uses `a_direction64_v3.yaml`/`b_direction64_v3.yaml` and `market_ordinal_v3`. Lower crossing
probability `a` and conditional upper crossing `b` form `[1-a,a*(1-b),a*b]`, trained by three-class
NLL. Its gradient equals lower-boundary BCE over every sample plus conditional upper-boundary
BCE only on non-down samples, not an equal-weight sum of separately averaged BCEs. This retained
version keeps costs, absolute/alpha tasks, quantiles, ranking, LoRA, PCGrad and train-only references.
This follows [CORN](https://arxiv.org/abs/2111.08851), not a financial-accuracy guarantee.
Inspired by [MASTER](https://ojs.aaai.org/index.php/AAAI/article/view/27767), seven existing
past-only benchmark state features generate bounded gates for three token streams and twenty
numerical features. Cleaning, baseline semantics and DataLoader remain unchanged; matching
data/cost baseline results are reusable. Selection equally averages market/task/horizon
`(CE/priorCE+Brier/priorBrier+1-population_macro_F1)/3`, adding opposite-error and collapse
penalties (each default 0.25) plus the retained frozen pinball guard. Collapse means true share
at least 5% and predicted share below 1%; configurable thresholds/weights never remove samples.
BA, population/balanced decision accuracy, per-class precision/recall and confidence bins remain;
ordinal MAE, opposite errors, Brier skill, collapsed classes and score components are added.
Test fits no prior/threshold. V2/V3 cannot resume across architectures; selection loss does not
rank runs. See [README historical V3 objectives](../README.md#legacy-objectives-en) for parameters and the full formula.

Optional later single-factor ablations, without changing outputs or splits, include:

1. Gated benchmark conditioning versus asset-only (gate fixed to zero).
2. Kronos LoRA versus frozen Kronos plus trainable downstream modules.
3. Kronos versus compact PatchTST/DLinear with identical histories, horizons and task loss, disclosing annual sampling and optimizer differences; GBDT point training with MAE is not an identical-loss control.
4. Price-index benchmark input versus total-return-adjusted benchmark input.
5. CAPM abnormal-return diagnostic analysis without replacing the main label.

Each ablation needs a new run identity and must never overwrite a checkpoint
with different semantics.

#### 13.1 Single-factor training-order control: `*-signal64-datemix`

The historical `DateMarketSampler` sorts by (date, market) and emits contiguous 128-sample
blocks. Each main-model micro-batch therefore comes from one date and market, and a
256-sample effective batch covers only about two date/market groups. Stocks on one date share
market and style shocks, so each update is dominated by the common outcome of very few dates.
Neural baseline blocks instead hold consecutive dates of one stock; the batch compositions differ.

Hypothesis: date-concentrated batches make the model chase date-level common shocks, visible as
early best checkpoints, seed sensitivity, long-horizon class drift and volatile per-date losses.
`a-signal64-datemix`/`b-signal64-datemix` add only the bounded local shuffle
`training.date_mixing_buffer_samples: 16384`; every other numerical setting equals the matching
`*-signal64`. Each epoch keeps the same samples and annual quotas, the model architecture digest is
unchanged, and task baselines are reused. This is a new run identity and resumes no old checkpoint.

```bash
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64-datemix
bash scripts/runpod_workflow.sh train --experiment a-signal64-datemix --seed 42
```

Comparison procedure:

1. Compare seed 42 with the existing `a-signal64` seed 42 on the **complete validation set**
   first; do not choose a method from test results.
2. Check `training_batch_composition` in `summary.json`: distinct date/market groups per
   optimizer step in the mixed order should far exceed `unmixed_reference_order`.
3. Pre-registered criteria that use only validation metrics already recorded in `metrics.jsonl`:
   - alpha: best normalized Smooth L1 below the historical seed 42 best of 0.4548, with the best
     point after the first evaluation;
   - direction: best CE below 1.0284; at that point, 10–14 day ECE below the historical seed 42
     best-point range of 0.0864–0.1037; market-macro CE (1.0197) and macro-F1 (0.3592) must not
     worsen.
4. Only if the validation criteria hold, complete the other seeds and read the final test,
   comparing GRU/GBDT and other learned baselines with date-paired intervals. The test period has
   been used repeatedly, so it is not confirmation on entirely unseen data. The final benchmark's
   `daily_signal_loss` and per-horizon predicted class shares are supporting diagnostics for
   per-date loss volatility and long-horizon class drift.
5. Retain the result even if any criterion fails, and add no other change in the same round so the
   effect remains attributable.

Outcome: `a-signal64-datemix` seed 42 (R24) failed both the alpha and direction criteria, so no
further seeds run. See [model development roadmap section 5.6](../MODEL_DEVELOPMENT_ROADMAP.md) for
the full values, batch composition and interpretation.

#### 13.2 Alpha checkpoint re-selection by cross-sectional Rank IC-IR (inference only, pre-registered)

Independent alpha models select checkpoints by complete-validation normalized Smooth L1. At this
signal-to-noise ratio, evaluation losses differ by only 0.2%–2%, while cross-sectional ranking
quality can differ much more. This section is an inference-time diagnostic: no retraining, no
change to `checkpoint_monitor`, no rewrite of the best-checkpoint pointer, and no replacement of
published results.

Pre-registered rule (written on 2026-10-18, before any test result of a non-best checkpoint was
read):

1. Score eqIR: from one validation evaluation, average
   `validation/cross_sectional/by_market/<market>/<h>d/rank_ic_ir` over h = 1–14 into a market
   score, then weight US, TWSE and TPEX equally. An evaluation with any missing market or horizon
   value is not eligible.
2. Candidates are only checkpoints retained by the run's validation leaderboard; evaluation points
   that were not retained have no weights to export.
3. Choose the highest eqIR; when it equals the loss-selected best checkpoint, export nothing extra.
4. Scope: the alpha children of the three `a-signal64` seeds (R21–R23). Classifiers select on CE,
   itself a proper scoring rule aligned with the task, and are not re-selected. R24 (datemix) is a
   recorded negative result and is excluded.

Results computed from `metrics.jsonl`:

| Run | Loss-selected | Its eqIR | eqIR-selected | Its eqIR | Extra export |
| --- | --- | ---: | --- | ---: | --- |
| R21 alpha | `checkpoint-003680` | 0.328 | `checkpoint-022079` | 0.533 | Yes |
| R22 alpha | `checkpoint-011040` | 0.432 | `checkpoint-018399` | 0.490 | Yes |
| R23 alpha | `checkpoint-018399` | 0.519 | `checkpoint-018399` | 0.519 | No (same) |

Execution: run complete validation and test inference with
`export-signal-predictions --checkpoint RUN_ID/checkpoint-NNNNNN`; a named checkpoint must be on
the leaderboard or it is refused. The validation loss and per-market Rank IC-IR recomputed in
`split.json` must match the same step in `metrics.jsonl`. The analysis computes three-seed ensembles and the error decomposition separately
for loss-selected and eqIR-selected checkpoints. Test is recorded only and must not change this
rule, choose seeds or swap checkpoints afterwards.

Limits:

- Taking the maximum over five retained checkpoints is selection-biased, so validation eqIR is
  optimistic.
- Each market's validation Rank IC has only 112 valid dates, so eqIR gaps between candidates
  (for example R22's 0.432 against 0.490) may still be noise. R22's eqIR choice also trades mean
  Rank IC for stability (eqIC 0.0637→0.0525).
- Reports list the loss-selected and eqIR-selected groups side by side, each with its three-seed
  ensemble, and show Rank IC per market. Test differences between the groups are descriptive only
  and must not be read as evidence about which selection rule is better.
- The equal-market score is driven by Taiwan: US 1–14 day mean Rank IC is only 0.017–0.047, and
  Taiwan's validation IC has reversed on test before (TPEX). eqIR therefore need not represent
  transferable signal.
- The outcome is only evidence for a later decision on formally changing `checkpoint_monitor`; that
  change needs code changes, a new preset and three-seed training.

#### 13.3 Phase 1: direction information in free Taiwan data (CPU only, pre-registered)

The P0 per-sample decomposition splits three-class CE into "does the return move beyond ±1%" and
"up or down given a move". All of the classifiers' gain over the train prior comes from the first
part; the second is no better than the prior at any horizon, for Kronos and GBDT alike. For
example, Taiwan 1-day validation improves by 5.8% on the first part and by −0.3% on the second
(Kronos three-seed mean).

This section tests whether free Taiwan flow, futures and US overnight data let GBDT beat the
prior on "up or down". It is a prerequisite for spending GPU training on new features. It does
not retrain Kronos or change the existing dataset, labels or baselines. Operation is described in
the README's "Taiwan information test".

Fixed settings (written 2026-10-18, before any auxiliary download or model fit), all recorded in
`configs/research/tw_information_test.json`:

1. **Samples**: the cleaned TWSE and TPEX samples of the `a-absolute-direction64` data group with
   the existing splits (train to 2025-06-01, validation to 2025-12-01, test to 2026-06-01).
2. **Labels**:
   - Entry and exit as in `execution_total_return`: the open of session t+1 to the close of
     session t+h.
   - Main horizons 1/2/3/5; 10/14 for comparison only.
   - Main band max(cost floor, k_h × σ20 × √h), with floors of 0.585% for stocks and 0.385% for
     ETFs. k_h is calibrated on train only to a neutral share of 1/3.
   - The fixed ±1% band is a comparison for `F0` and `all_pre_open` only.
3. **Feature sets**: `F0`, `F0_long`, `short_only`, `flows`, `market`, `us_evening`, `us_pre_open`,
   `all_evening`, `all_pre_open`. Contents and publication times are in the README.
4. **Models**:
   - sklearn HistGradientBoosting: learning rate 0.05, max leaf nodes 31, min samples leaf 500,
     L2 1.0, at most 400 iterations, seed 42.
   - Fitted on train only. The boosting stage is checked every 10 stages and chosen on the last
     20% of train, separated from the earlier part by 15 sessions.
   - Validation is never used for fitting or tuning.
5. **Primary metric**: pooled Taiwan validation CE of "up or down", over non-neutral labels only.
   The reference is the train class shares per market (Jeffreys-smoothed).
6. **Criterion**: an added feature set (any set other than `F0`) passes when, on at least two of
   the 1/2/3-session horizons, all three of the following hold:
   1. Its "up or down" CE improves on the prior by at least 1%, i.e.
      (prior − model)/prior ≥ 0.01.
   2. The 95% interval of its gap to the prior has a lower bound above 0.
   3. The 95% interval of its gap to `F0` has a lower bound above 0.

   Intervals come from a circular block bootstrap over dates with 5-session blocks, 1,000
   replicates and seed 7; the statistic is a ratio of per-date sums. The test passes when the
   report's `criteria.information_found` is true.
7. **Consequences**:
   - Pass: the passing sets enter the Taiwan model design (new labels and horizons). The design
     is reviewed separately, and GPU training needs separate approval.
   - Fail: these free data carry no direction information that beats the prior at 1–3 sessions.
     No GPU training is spent; the next step is decided after a review.
8. **Secondary questions** (descriptive; they do not affect the criterion):
   - Lookback: the "up or down" gaps of `F0_long` and `short_only` to `F0`, with intervals
     (`window_question_direction_gap_vs_F0`).
   - Signal time: `all_evening` versus `all_pre_open`.
   - Band: fixed ±1% versus the main band.
   - Alpha: normalized Smooth L1 and per-market Rank IC of every set.
   - Test metrics and a simple backtest: each day, buy the top 10% by signal whose signal is
     positive, net of the cost floor. Overlapping multi-day holdings make the t statistic
     optimistic.

Limits:

- There are 8 added sets and 3 horizons, so there are multiple comparisons. Requiring the same
  set to pass on two horizons reduces chance passes.
- A GBDT failure does not rule out other models using the data, but it means the signal is too
  weak to estimate stably at this sample size.
- When an auxiliary source is `failed`, the sets that depend on it do not take part, are listed in
  `skipped_feature_sets`, and the decision uses the remaining sets.
- Validation covers only about six months (2025-06 to 2025-12). Intervals are wide, and the
  conclusion applies to that period.

#### 13.4 Stage 1 backtest: Taiwan ranking signals net of costs (local, pre-registered)

§13.3 shows that free data do not lift absolute direction above the prior, yet several models keep a valid Taiwan cross-sectional ranking (Rank IC) on test. This section tests whether those ranking signals beat passive holding after real Taiwan costs. It is a research backtest: no retraining and no RunPod. Code and configuration are in `backtest/`, with its own uv environment; the pre-registered settings are recorded in `backtest/config/stage1.json`.

Fixed settings (written 2026-10-19, before any backtest computation):

1. **Data**:
   - TWSE and TPEX rows of the P0 evaluation population, whose row order is verified against every export by its sample digest.
   - Signal dates: validation 2025-06-02 to 2025-11-10 (112 dates), test 2025-12-01 to 2026-05-11 (103 dates).
   - Returns: adjusted log returns from the open of session t+1 to the close of session t+h (`absolute_targets`).
   - Index returns: absolute minus alpha, identical within a market and date (TAIEX or the TPEx index).
2. **Signals (13)**: each signal uses the prediction for the same horizon as the holding period.
   - Kronos alpha (8): R21 (checkpoint 003680), R21-IR (022079), R22 (011040), R22-IR (018399), R23 (018399), R24 datemix (007360), and two three-seed ensembles, LOSS (R21, R22, R23) and IR (R21-IR, R22-IR, R23), averaging point predictions.
   - Baseline alpha (2): GRU, GBDT.
   - Direction controls (3): the mean probabilities of the R21–R23 direction models, GRU direction and GBDT direction, with signal P(up) − P(down).
3. **Trading rule**:
   - After the close of session t, rank the stocks; buy the top K equally weighted at the next open, hold h sessions and sell at the close of session t+h.
   - The next batch is selected after that close and bought at the following open.
   - Each of the h start dates forms one series, equivalent to splitting capital into h sleeves.
   - Grid: h ∈ {1, 2, 3, 5, 10}; K ∈ {10, 20, 50}; market ∈ {TWSE, TPEX, TW (both ranked together)}.
   - Versions: long-only is primary; the hedged version subtracts the matching index return, with futures costs approximated as 0.
4. **Costs**:
   - Buy: commission 0.1425% × discount.
   - Sell: commission 0.1425% × discount plus transaction tax (0.3% for stocks, 0.1% for ETFs).
   - h = 1 is a same-day round trip, taxed at 0.15% for stocks.
   - Slippage: 10 bps per side on TWSE and 20 bps on TPEX.
   - Primary scenario: discount 0.6, slippage ×1, conservative turnover (full replacement each batch).
   - Sensitivity: discount {1.0, 0.6, 0.28} × slippage {0, 1, 2}× × turnover {conservative, optimistic}. Optimistic turnover means names held into the next batch pay no extra cost.
   - Not simulated: minimum commission per order, price limits, liquidity and board-lot constraints.
5. **Metrics**:
   - Primary metric: batch net return in **net excess** of the same-date, same-window, cost-free equal-weight population.
   - Others: excess over the index, annualized values (× 252/h), IR, Sharpe, maximum drawdown, batch hit rate, turnover, cost drag, gross excess, and break-even cost per side.
   - IR = mean batch excess ÷ its standard deviation × √(252/h), computed within each start series and then averaged.
   - 95% intervals: a circular moving-block bootstrap of the daily batch-excess series in signal-date order, block length max(5, 2h), 1,000 replicates, seed 11.
6. **Selection**: validation only.
   - In each market (TWSE, TPEX, TW), among alpha signal × h × K under long-only and the primary costs, the combination with the highest net-excess IR is the **primary configuration**.
   - Test runs once. The full grid is descriptive; the headline is the primary configuration and is not re-selected.
7. **Gate to the next stage**: the primary configuration's validation net-excess 95% interval has a lower bound above 0 **and** its test net excess is positive. Stage 3 verifies only passing primary configurations on new data from 2026-06-01 onward.

Limits:

- Taking the best of 150 configurations per market overstates validation; test checks this bias.
- Validation and test each cover about five months, so intervals are wide.
- The population already passed the data-cleaning rules, which may exclude stocks later suspended or delisted. The bias is shared by every strategy and the equal-weight benchmark, but it makes absolute returns optimistic.
- Limit-up opens that cannot be bought, insufficient volume and minimum commissions are not simulated, so results are optimistic until stage 3 adds them.

#### 13.5 Stage 1b: hold buffer with per-position accounting (local, pre-registered)

The §13.4 primary configurations failed the gate in all three markets: the 10-session configurations chosen on validation lost 37% to 44% a year in net excess on test. The post-hoc analysis found two things:

- One-day rankings carry a gross edge on both validation and test. The top K beat the same-date equal weight by about 25–30 bps a day on TWSE and 15–20 bps on TPEX, but a round trip in the primary cost scenario costs about 52 bps, which consumes the edge.
- About 85% of the top K overlap between adjacent days. The §13.4 optimistic turnover charges no repeat costs, but it omits the overnight gap of held names and still applies the day-trade tax, so it is not evidence that lower turnover helps.

This section uses per-position accounting to test whether a hold buffer that lowers turnover turns the ranking's gross edge into net excess return after costs. The code shares `backtest/` with §13.4; the pre-registered settings are recorded in `backtest/config/stage1b.json`.

**Contamination statement**: this section was conceived from the §13.4 post-hoc analysis of test (the one-day gross edge, decile returns and low-cost scenarios), so test is no longer unseen data. Test results here are exploratory only; the only confirmatory test is stage 3, on data from 2026-06-01 onward, with models and rules frozen in advance.

Fixed settings (written 2026-10-19, before any stage 1b strategy computation; only the accounting-identity check in item 2 ran before writing):

1. **Data**: the same population, splits, 13 signals and row-order checks as §13.4.
2. **Accounting identity**:
   - For one stock's rows on adjacent signal dates d and d+1, `abs[d, h+1] − abs[d+1, h] = log(open(d+2) / open(d+1))` holds for every h = 1…13, which gives open-to-open holding returns.
   - Check before writing: all 96,166 adjacent pairs in validation and 93,402 in test satisfy it (maximum difference ≤ 9×10⁻⁸). Adjacent split dates are therefore consecutive sessions, and every row uses the same adjusted price series.
   - The run re-checks the identity for stocks and for the index (absolute minus alpha) and stops if fewer than 99.9% of pairs are within 10⁻⁵.
3. **Trading rule** (simulated separately for each market, signal and parameter set):
   - After the close of day d, rank all stocks of the market by the horizon-s prediction (1 = best).
   - Held stocks ranked ≤ b × K are kept; held stocks ranked > b × K are sold.
   - Empty slots are filled with the best-ranked stocks not yet held, up to K stocks. b = 1 replaces the portfolio with the top K every day and is the no-buffer control.
   - All trades execute in the opening call auction of day d+1; there is no maximum holding period.
   - Price limits: a buy whose open is 9.5% or more above the previous close cannot be filled, and its slot stays in cash until the next execution; a sell whose open is 9.5% or more below the previous close cannot be filled, so the stock is held and re-evaluated on the next signal date. A missing previous-day row counts as fillable; the final liquidation ignores the limit-down rule.
   - Capital is split into K equal slots. A sold slot's net proceeds buy the new stock in full; there is no rebalancing.
   - A held stock with no row on the next signal date exits at the close of day d+2 (`abs[d, 2]`), and its slot stays in cash at the next execution.
   - Validation and test are simulated separately: positions are opened after the close of the first signal date and fully liquidated at the open after the last signal date.
4. **Costs**:
   - Buy: commission 0.1425% × discount, plus slippage.
   - Sell: commission 0.1425% × discount, plus transaction tax (0.3% for stocks, 0.1% for ETFs) and slippage. Every position is held at least overnight, so the day-trade tax never applies.
   - Slippage: 10 bps per side on TWSE and 20 bps on TPEX.
   - Primary scenario: discount 0.6, slippage ×1. Sensitivity: discount {1.0, 0.6, 0.28} × slippage {0, 1, 2}×.
5. **Grid**: s ∈ {1, 3, 5}; K ∈ {10, 20, 50}; b ∈ {1, 2, 4}; market ∈ {TWSE, TPEX, TW (both ranked together)}; all 13 signals are candidates (direction signals rank by P(up) − P(down)). Each market has 351 candidates.
6. **Metrics** (daily, non-overlapping open-to-open series):
   - Primary metric: the portfolio's daily net return in **net excess** of the same-date, same-market, daily-rebalanced, cost-free equal weight of all stocks.
   - Others: annualized (× 252) net excess, IR, CAGR, Sharpe, maximum drawdown, daily hit rate, daily one-way turnover (newly bought slots ÷ K), holding period (mean, median, p90), annual cost drag, break-even cost per side (annual gross excess ÷ annual two-way turnover), price-limit blocks and missing-row exits.
   - IR = mean daily net excess ÷ its standard deviation × √252.
   - Hedged version: subtract each held slot's index (TAIEX or the TPEx index) open-to-open return every day, with futures costs approximated as 0; descriptive only.
   - 95% intervals: a circular moving-block bootstrap of the daily net-excess series, block length 10 sessions, 1,000 replicates, seed 11.
7. **Selection**: validation only.
   - In each market, the candidate among the 351 with the highest net-excess IR under long-only and the primary costs is the **primary configuration**; ties are broken by signal, s, K and b.
   - Test runs once. The full grid is descriptive; the headline is the primary configuration and is not re-selected.
8. **Gate**: the primary configuration's validation net-excess 95% interval has a lower bound above 0 **and** its test mean net excess is positive.
   - Passing only means the configuration earns a stage 3 forward test.
   - If no market passes, the lower-turnover route stops and work returns to the models themselves.

Limits:

- Taking the best of 351 candidates per market overstates validation, and test has already been seen in post-hoc analysis, so it cannot fully check that bias.
- Volume and liquidity, minimum commissions, board lots, and dividend income tax and the supplementary health premium are still not simulated; the population's survivorship bias is as in §13.4.
- Price limits are approximated from adjusted opening gaps and may be misjudged on ex-dividend dates.
- Slots are not rebalanced, so weights drift with prices.

#### 13.6 Holding-aligned evaluation: open-to-open re-evaluation of existing signals (local, pre-registered)

The §13.4 and §13.5 results are in the [backtest results report](backtest_stage1_results.md): the top-K edge exists only during sessions and reverses overnight. The current labels (1 day = the next session's open to close) and Rank IC evaluation do not match the intended trading style: signals after the close, buy at the next open, hold days to two weeks, and sell at an open. On 2026-10-19 it was decided to align labels and evaluation with that holding, in two parts:

- **A (this section)**: build a holding-aligned evaluation and re-evaluate existing signals on open-to-open returns. Local computation only; no model training.
- **B (pre-registered later in a separate section)**: train GBDT on open-to-open labels to check whether a learnable multi-day edge exists. B's criteria use this section's evaluation definitions.

This section is descriptive only: it has no gate and does not re-select models. Test has been consulted repeatedly in earlier analyses and is exploratory only. The code is in `backtest/`; the settings are recorded in `backtest/config/holding_eval.json`.

Fixed settings (written 2026-10-19, before any open-to-open metric was computed):

1. **Data**: the same population, splits, 13 signals and row-order checks as §13.4; the run re-checks the §13.5 accounting identity and stops if it fails.
2. **Open-to-open labels**:
   - For a stock on signal date d and holding h sessions: `R(d, h) = abs[d, h+1] − abs[d+h, 1] = log(open(d+1+h) / open(d+1))`. It needs the same stock's row on signal date d+h; otherwise it is missing.
   - The index is derived the same way from absolute minus alpha; open-to-open alpha = stock minus index.
   - Holding periods: h = 5 and 10 are primary; h = 3 is auxiliary.
3. **Signals**: each signal uses its own prediction for horizon h; direction signals use P(up) − P(down).
4. **Metrics** (validation and test; TWSE and TPEX separately):
   - **Rank IC**: daily cross-sectional Spearman (average ties, at least 3 values, non-constant) against open-to-open alpha, with the Rank IC against the current label listed for comparison. Mean and IR (mean ÷ standard deviation) are reported.
   - **Deciles**: rank by score each day; the top 10% (at least 1 stock, rounded) and bottom 10% open-to-open simple returns, each minus the same-date, same-market equal weight of stocks with labels, plus the top-minus-bottom spread.
   - **Session versus overnight**: the top decile's excess log return split into the h sessions (each day's open to close) and the remaining overnight part, both relative to equal weight.
   - **Fixed-parameter holding portfolio**: the §13.5 engine and accounting with score horizon s = h, K = 20, b = 2, the primary costs (discount 0.6, slippage ×1), long-only; markets TWSE, TPEX and combined. Reported: annual net excess, IR, gross excess, cost drag, daily turnover and mean holding period.
   - Intervals: circular moving-block bootstrap of daily series, 1,000 replicates, seed 11; block length max(5, 2h) for h-session label series and 10 for the portfolio's daily series.
5. **Interpretation**:
   - Descriptive only, with no gate.
   - A signal whose validation top-decile excess has a 95% lower bound above 0 in a market at a primary horizon is listed as "worth watching", but no decision rests on it.
   - Decisions rest only on B or on later pre-registered tests.

Limits:

- 13 signals × 2 markets × 3 horizons create multiple comparisons, so some "worth watching" entries will occur by chance.
- h-session labels overlap; the block bootstrap only partly handles the autocorrelation.
- Signal dates near the end of a split lack complete h-session labels, so the usable validation and test dates shrink with h.
- Other limits are as in §13.5 (liquidity, minimum commissions, board lots, dividend taxes, survivorship bias).

#### 13.7 Local Kronos inference: parity check and checkpoint selection (local, pre-registered)

§13.6 shows that checkpoints of the same run differ widely overnight (R22's loss-selected checkpoint drags −213.8 bps overnight, its Rank IC-IR checkpoint only −10.7 bps), and that TWSE 10-session R22-IR is the only signal positive on both validation and test. On 2026-10-19 it was decided that training stays on RunPod and inference needed for backtests runs locally (M5 Pro). This section fixes two things: the parity check that local inference must pass first, and the rule for selecting checkpoints by a holding metric. The settings are recorded in `backtest/config/local_parity.json` and `backtest/config/checkpoint_selection.json`.

Fixed settings (written 2026-10-19, before any local inference):

1. **Local inference**:
   - Runs in the separate uv subproject `inference/`, apart from the Poetry and RunPod environments and from `backtest/`.
   - Loads each checkpoint's own `resolved-config.yaml` and `adapter.safetensors`; the Kronos source and base weights use the versions pinned by sha256 and revision.
   - Device: MPS fp32, or CPU fp32 where MPS lacks support. The device actually used must pass the parity check itself.
   - Scope: only the TWSE and TPEX rows of the P0 evaluation population. Each row records its stock and date, and `backtest/` maps rows one-to-one onto the evaluation arrays; the mapping must be complete.
   - Inference changes no model, checkpoint or lifecycle record.
2. **Parity check (P1)**:
   - Reference: the validation predictions of R22 checkpoint-018399 exported on a RunPod RTX 5090 in bf16 (`export-20261017T223254Z-ce4974f3`).
   - Local inference runs the same checkpoint on the validation Taiwan rows.
   - Passing requires all of:
     1. Across date × market × horizon groups, at least 99% have a cross-sectional Spearman (local against reference) ≥ 0.999.
     2. Using the §13.6 definitions, the open-to-open Rank IC for TWSE and TPEX at 5 and 10 sessions differs by ≤ 0.005.
     3. Under the same conditions, the top-decile excess differs by ≤ 5 bps.
   - Also recorded: samples per second, peak memory and the maximum absolute prediction difference.
   - On failure, neither checkpoint selection nor §13.8 proceeds; the cause is reported for discussion.
3. **Checkpoint selection (P2)**:
   - Candidates: the 5 retained checkpoints of each of three alpha runs.
     - R21 (`run-20261015T064558Z-f04e155d4299416f-alpha`): 003680, 007360, 011040, 018399, 022079.
     - R22 (`run-20261015T064559Z-624f212a27894308-alpha`): 003680, 007360, 011040, 014720, 018399.
     - R23 (`run-20261015T064600Z-13b8b4376ea5413b-alpha`): 003680, 007360, 018399, 022079, 025759.
   - Rule: for each run, select the checkpoint with the highest validation TWSE 10-session top-decile open-to-open excess (§13.6 definition); ties go to the later step.
   - Selection uses validation only. The 3 selected checkpoints are also run on test, descriptively.
   - No gate; the selected checkpoints are examined descriptively in §13.8.

Limits:

- Selection and description use the same validation period, so the selected checkpoints' validation values are optimistic; only §13.8's unseen data can confirm them.
- 15 candidates over 112 validation dates; the rule looks only at TWSE 10-session results.
- Local inference is fp32 and the reference bf16; the parity check measures their difference and does not guarantee identical rows.

**§13.7 amendment (2026-10-19, after the local parity check failed; the registered text above is kept unchanged)**

Local inference result: under the original settings, MPS fp32 inference of R22 checkpoint-018399 on the validation Taiwan rows (97,363 rows, 52 per second) failed the parity check.

| Criterion | Result | Passed |
| --- | --- | --- |
| At least 99% of groups with cross-sectional Spearman ≥ 0.999 | 0% (3,136 groups; median 0.995, minimum 0.908) | no |
| Rank IC difference ≤ 0.005 | Maximum 0.0017 | yes |
| Top-decile difference ≤ 5 bps | Maximum 11.1 bps (TPEX, 5 sessions) | no |

Diagnosis: the difference comes from numerical precision, not systematic bias or a code error.

- Local minus reference shifts by 0.04–0.25 bps on average (prediction standard deviation 83 bps), the regression slope is 1.000–1.002, and the overall correlation is 0.997.
- Computing on CPU with the same bf16 numerics as RunPod gives a one-date cross-sectional Spearman of 0.9996–0.9999; rounding only the frozen weights to bf16 does not help. The difference comes from bf16 rounding during computation.
- MPS bf16 is inaccurate (Spearman 0.65–0.93) and CPU bf16 too slow (about 2.4 rows per second); neither is feasible.

Amended settings:

1. Inference uses the existing RunPod GPU export workflow (`export-signal-predictions`, CUDA bf16), the same code and numerics as the reference exports. Local work is scoring and backtesting only; the `inference/` subproject is diagnostic and decides nothing.
2. The parity check is withdrawn. Integrity checks follow P0: every export's row-order digest must match the baseline, and its recomputed validation selection score must match that checkpoint's validation record at the same step in `metrics.jsonl` (within 10⁻⁶).
3. The selection rule is unchanged. Of the 15 candidates, 5 were exported in P0 (R21 003680 and 022079; R22 011040 and 018399; R23 018399); the other 10 are added in one export job with validation and test together.
4. The selection code reads validation only; test is computed only for the 3 selected checkpoints, enforced in code, and is descriptive.
5. Item 5 of §13.8, "runs only after the §13.7 parity check passes", becomes: inference uses the RunPod GPU export workflow, and scoring starts only after the export integrity checks pass. Forward-data inference needs the export code extended to a `forward` split, which will be described before implementation.

#### 13.8 Stage 3: forward test on unseen data from 2026-06-01 onward (pre-registered)

This section is the first confirmatory test of TWSE 10-session R22-IR, a candidate found after the fact in §13.6. Its criteria are therefore written before the forward data is prepared, with the model, market, horizon and parameters all frozen. The settings are recorded in `backtest/config/stage3.json`.

Fixed settings (written 2026-10-19, before the forward data download and any forward inference):

1. **Data**:
   - A new dataset is built with the existing data-preparation workflow on a RunPod CPU Pod: the same profile (`us_tw_eodhd`), universe and cleaning rules, ending at the latest trading day available at preparation time.
   - A new `forward` split covers signal dates on or after 2026-06-01 with complete 14-session labels. The train, validation and test definitions are unchanged.
   - Data checks (all must pass before scoring):
     1. The new dataset's manifest is ready.
     2. For the `forward` split's Taiwan rows, the §13.5 accounting identity holds for at least 99.9% of pairs (within 10⁻⁵).
     3. Over the test period shared with the current dataset, at least 99% of Taiwan rows have `absolute_targets` (h = 1…14) within 10⁻⁵ at every horizon.
2. **Labels and metrics**: the §13.6 open-to-open definitions, deciles and fixed portfolio, computed within the `forward` split only.
3. **Primary hypothesis (the only one that decides)**: R22-IR (R22 checkpoint-018399), TWSE, h = 10. Both conditions must hold to pass:
   1. The 95% interval lower bound of the mean top-decile open-to-open excess is above 0. The interval uses a circular moving-block bootstrap with block 20, 1,000 replicates and seed 11.
   2. The fixed portfolio (s = 10, K = 20, b = 2, primary costs, long-only) has a positive mean annual net excess.

   An interval that crosses 0 is a failure, with no further interpretation.
4. **Secondary description (does not decide)**:
   - ENS-IR, R23 (checkpoint-018399) and the 3 checkpoints selected in §13.7.
   - Controls: the loss-selected R21 (003680) and R22 (011040).
   - R22-IR results on TPEX and at 5 sessions.
5. **Procedure**:
   - Runs only after the §13.7 parity check passes.
   - Each frozen model is run on the `forward` split once; models, markets, horizons and parameters are not re-selected.
   - Results are written up as a bilingual report.
6. **Outcome**:
   - Pass: TWSE 10-session R22-IR becomes a candidate signal for the trading rules, with continued forward monitoring.
   - Fail: it is not adopted, and work returns to improving the models.

Limits:

- About 80 usable signal dates with overlapping 10-session windows give wide intervals; a failure does not prove the absence of an edge, but the signal cannot be adopted.
- Only the summer-to-autumn 2026 market is covered.
- The new dataset's adjusted prices change history after post-June corporate actions; scoring stays within the new dataset.
- Other limits are as in §13.5 (liquidity, minimum commissions, board lots, dividend taxes, survivorship bias).

**§13.8 amendment (2026-10-20, before the forward data download and any forward inference; the registered text above is kept unchanged)**

Reasons:

1. **Only Taiwan data is needed**:
   - Stage 3 scores Taiwan only. A Taiwan sample's inputs are the stock itself and its Taiwan benchmark (`TAIEX.TW`, `TPEX.TWO`), and its prices come from the official TWSE and TPEx sources; it uses no US data.
   - Keeping `us_tw_eodhd` would mean re-downloading the full US history for rows that are never scored.
2. **An explicit exception is needed**: after the §13.7 amendment, inference runs through the RunPod GPU export. The current checkpoint loading rule requires the dataset to be identical to the training dataset, so forward inference needs the allowed differences written down.
3. **The bottom decile is needed to interpret the top decile**: P2 found that on test both the top and the bottom decile beat the equal-weight average.

Amended settings:

1. **Data** (replaces the first two items of the original item 1):
   - A new dataset is built with the existing workflow on a RunPod CPU Pod, with profile `tw_only`: official TWSE and TPEx data only.
   - The start date (2016-01-01), universe mode, cleaning rules, benchmark mapping and fixed split boundaries equal those of the shared evaluation snapshot used for inference so far.
   - The exclusive end date is the UTC date on which the CPU Pod is created, written to `backtest/config/stage3.json` before the Pod starts.
   - A new `forward` split: signal date d with 2026-06-01 ≤ d < end date, whose 14th-session label date is before the end date. The train, validation and test definitions are unchanged.
   - Holidays: only dates on or after 2026-06-01 may be added, for unscheduled closures (such as typhoons) or adjusted trading days announced by TWSE. Each addition is listed in the report; the calendar before 2026-06-01 is unchanged.
2. **Data checks** (replace the original item 1 check list; run locally; all must pass before any GPU work):
   1. The new dataset's manifest is ready.
   2. For the `forward` split's Taiwan rows, the §13.5 accounting identity holds for at least 99.9% of pairs (within 10⁻⁵).
   3. Against the shared evaluation snapshot's test period, matched by symbol and signal date:
      - At least 99% of the old Taiwan rows are found in the new dataset.
      - At least 99% of the matched rows have `absolute_targets` (h = 1…14) within 10⁻⁵ at every horizon.
3. **Inference scope** (new):
   - Only these differences are allowed: dataset identity and artifacts, profile (`us_tw_eodhd` → `tw_only`), end date, `forward` boundary, and the holidays allowed in item 1.
   - Model, input length, horizons, label definitions, the rest of the cleaning rules and the train, validation and test boundaries must equal the checkpoint's training settings, checked item by item in code.
   - `export.json` records the identities of both datasets.
   - Inference uses the same GPU model (RTX 5090), torch, Python and bf16 settings as the 2026-10-19 exports. If no RTX 5090 is available, the work waits; the GPU is not changed.
   - The `forward` split is inferred for 6 checkpoints: R21 003680, 018399 and 022079; R22 011040 and 018399; R23 018399.
4. **Pipeline consistency check** (new; must pass before scoring):
   - In the same GPU job, R22-IR infers the test split of the new dataset.
   - Matched by symbol and signal date with the test Taiwan rows of the 2026-10-19 export, at least 99% of the old rows must be found.
   - The predictions are expected to be bit-identical. If they are not, both of the following must hold:
     - (a) On the matched rows, the correlation of predictions pooled over all 14 horizons is at least 0.999.
     - (b) The TWSE 10-session top-decile excess recomputed per §13.6 is within 5 bps of the +103.2 bps in the selection report.
   - Failure: no scoring; the cause is found and recorded before any re-run.
5. **Procedure** (adds to the original item 5):
   - Order: CPU data preparation → local data checks → GPU inference (with the consistency check) → local scoring.
   - No forward metric is computed before all data checks and the consistency check pass. The scoring code refuses to run if the check records are incomplete.
   - A technical re-run of inference before scoring is allowed, with the cause recorded; scoring uses the results of a single inference run.
   - After scoring starts there are no re-runs, and models, markets, horizons and parameters are not re-selected.
6. **Descriptive additions** (do not decide): the primary hypothesis and every secondary signal also list the bottom-decile excess and the top-minus-bottom spread next to the top decile.

Additional limits:

- The new dataset holds Taiwan only; US rows are out of scope and affect no decision in this section.
- The Taiwan instrument list is whatever the official sources return at download time and may miss stocks delisted during the period (survivorship bias, as in §13.4).
- The consistency check covers the test period only; data problems confined to the forward period are caught by data check 2.

**§13.8 supplement (2026-10-20, before implementation and before obtaining any forward data)**

Reason: the training data pipeline's split logic is part of the dataset fingerprint, so a new split would invalidate every existing dataset; and if the model proves valuable, an independent inference flow is needed anyway.

1. Forward data comes from an independent forward-inference pipeline; no training dataset is built. This replaces amendment item 1's "a new dataset is built with the existing workflow on a RunPod CPU Pod":
   - Taiwan data from 2025-03-01 to the end date is fetched locally (Taiwan network) with the existing TWSE and TPEx providers.
   - Normalization, adjustment, eligibility, benchmark mapping and cleaning all call the same functions as the training data pipeline.
2. There are two populations, both using the validation/test sample rule:
   - `overlap`: signal dates 2025-12-01 ≤ d < 2026-06-01, labels before 2026-06-01, equal to the original test definition.
   - `forward`: defined as registered.
3. Data checks:
   - Check 3 compares against `overlap`.
   - New check 4 (before GPU work): on `overlap` Taiwan rows, at least 99% of rows have model inputs within a maximum relative difference of 10⁻⁶ of the shared evaluation snapshot.
4. Inference loads no dataset and checks only the model contract; exports record the input package hash. The pipeline consistency check uses `overlap`.
5. Everything else is unchanged: GPU model, the 6 checkpoints, criteria, procedure and descriptive additions.

**§13.8 result (2026-10-20)**: the primary hypothesis failed and, under item 6, is not adopted; see the [forward test results report](stage3_forward_results.md).

#### 13.9 Capturability analysis (validation, exploratory; written 2026-10-20 before the analysis)

Background: §13.8 failed, yet TWSE open-to-open Rank IC is positive in validation, test and the forward period. A long-only top decile held for 10 sessions uses only a small part of the ranking information, and stock trading costs are high. The instruments the user can trade: long stocks, TAIEX futures (standard, mini, micro), and stock futures (which allow shorting). This section finds which implementable construction turns the most ranking information into net excess after costs. The settings are recorded in `backtest/config/capturability.json`.

1. **Data scope**:
   - Only validation (signal dates 2025-06-02 to 2025-11-10, 116 dates) is used for comparison and selection.
   - Test and the §13.8 forward period have been seen; they are exploratory references only and take no part in selection.
   - No training and no new inference: the 2026-10-19 validation exports are used. Signals are the 6 registered and selected signals of §13.8.
2. **A: where the ranking information lies (before costs)**:
   1. Segments: with or without stock futures (TAIFEX underlying list, contract listed and traded that day), liquidity terciles (average traded value over the 20 sessions before the signal date, within the same date and market), and TWSE versus TPEX. For each segment: Rank IC, top- and bottom-decile excess, and the spread (h = 5, 10).
   2. Decile curves: the ten decile excesses at h = 5 and 10.
   3. Decay: Rank IC for h = 1…13, and the session/overnight split of the top and bottom deciles.
3. **B: implementable portfolios (after costs)**; the candidates are fixed:
   - B0 (reference): long-only top 20 stocks (the §13.6 fixed portfolio).
   - B1: B0 hedged with TAIEX futures (hedge ratio 1 and rolling beta).
   - B2: stock-futures long-short among the stocks with futures: long the top, short the bottom, equal dollar amounts.
   - B3: stock-futures rank-weighted long-short over every stock with futures, market neutral.
   - Each candidate is compared at 5- and 10-session rebalancing, with score smoothing (the mean of recent days' scores) on or off.
   - Costs: stocks use the primary scenario; futures use the 0.002% futures transaction tax, a per-contract commission, volume-tiered slippage scenarios and monthly rolls. The commission and the capital (which sets contract granularity) come from the user; until then they are parameters with sensitivities listed.
4. **Futures data**: TAIFEX daily futures trading data (public download form) for the validation period and the 20 sessions before it. Contract codes map to stock codes through TAIFEX's current underlying list; unmatched contracts are counted. Stock futures returns are approximated by the underlying stock's open-to-open return.
5. **Interpretation**: exploratory, with no gate. Validation selects one construction (signal, portfolio, rebalancing and smoothing); a new pre-registration then confirms it on data after 2026-10-20. Results are written as a bilingual report.

Interpretation limits:

- Validation has only 116 dates, so intervals are wide; constructions are compared in pairs.
- Futures open (08:45) before stocks (09:00), and basis, dividend adjustments and rolls make the approximate returns imprecise.
- TAIFEX's current list may miss contracts delisted during the period.
- Margin, margin calls and futures capacity are not simulated beyond contract granularity.

#### 13.10 Cross-sectional rank task: rank labels and baselines (written 2026-10-21 before implementation)

Background: neither the numeric alpha nor the absolute direction robustly beat the baselines; only relative ranking within the same date and market has positive Rank IC (Kronos alpha: validation 0.099–0.119, test 0.091–0.124). The project changes to predicting the cross-sectional rank. The label window is unchanged: signals at the t0 close, entry at the t1 open, exit at the close of session h (h = 1…14). Entry thresholds and exit rules are set by the user's hermes agent, which backtests daily rolling predictions; the model only supplies the ranking at each horizon. The old alpha and direction models and baselines are not kept compatible.

1. **Task**: `cross_sectional_rank`; its only parameter is the minimum group size `rank_min_group_size` (default 20).
2. **Labels**:
   - Source: the existing benchmark-relative adjusted log return (t+1 open to t+h close). Each market has one benchmark (TAIEX.TW for TWSE, TPEX.TWO for TPEX, VTI.US for US), so within a date and market the rank of the relative return equals the rank of the absolute return.
   - Groups: same split, same cutoff date, same market; only stocks in that split's sample ranges.
   - Transform: average rank r for ties, then (r − 0.5) / n, then the inverse standard normal distribution.
   - Groups with fewer than `rank_min_group_size` stocks get no label (NaN) and are masked in training. Evaluation still computes Rank IC on the raw returns and is not affected by the mask.
3. **Label storage (Parquet)**:
   - One store per split, next to that split's bar store: `rank-labels/<identity>/`.
   - `shards/bucket-XXXX/shard.parquet` keeps the bar store's buckets; each symbol is one row group with the cutoff time and `rank_h01…rank_h14` (float32).
   - `label-index.parquet` records each symbol's file, row group, row count and first sample ordinal; `rank-labels.json` records the identity, each file's sha256 and per-market group statistics.
   - The identity combines the bar-store manifest, the sample ranges, the split, the horizons, the transform, the minimum group size and the label code's hash; any change rebuilds the store.
   - The dataset reads labels by (symbol, cutoff) after checking timestamps; the dataloader pipeline is unchanged.
4. **Baselines** (models and seeds from `configs/baseline.json`):
   - GBDT: one squared-error regressor per horizon; rows without labels are dropped.
   - GRU, DLinear, PatchTST: one output per horizon; the loss is the mean squared error over labelled values.
   - The 6 rule signals: the raw signal value is the ranking score.
   - always_buy and zero_return: constant 0, a no-ranking-skill reference.
5. **Evaluation and selection**:
   - Primary metric: mean daily Rank IC for each (market, horizon), on raw returns, as in the existing evaluation.
   - Selection score: the negative of the Rank IC averaged over the three markets with equal weight and over the 14 horizons; lower is better. Early stopping and checkpoint selection both use it.
   - Supporting: mean squared error and correlation on the rank labels, Rank IC-IR per market and horizon, and the existing top-minus-bottom quintile diagnostic.
6. **Execution**: the labels are built at the start of the baseline GPU Pod (bounded multiprocessing: returns, ranks, Parquet writes). Kronos training later uses the same labels.
7. **Interpretation**: these baselines only set the reference, with no gate. Kronos `a-rank64` is registered separately and compared with them. Date mixing (`*-rank64-datemix`) is evaluated after `a-rank64`.

Interpretation limits:

- US has most of the samples; selection weights markets equally, so Taiwan (TWSE, TPEX) carries two thirds.
- Validation has about 116 trading dates, so Rank IC intervals remain wide.

**§13.10 result (2026-10-21)**: baseline `baseline-5fb3473a…` completed. Market-equal Rank IC (validation → test): PatchTST 0.057 → 0.035, GRU 0.056 → 0.007, GBDT 0.044 → 0.020, DLinear 0.033 → 0.003, moving-average crossover 0.020 → 0.043. US holds on test at 0.04–0.056; Taiwan is near 0 on test except at 1 session (TWSE 1-session about 0.08 for GBDT, GRU and PatchTST). Seed 42 only; the standard error of the daily Rank IC is about 0.01–0.018 per market and horizon.

#### 13.11 Kronos cross-sectional rank model `a-rank64` (written 2026-10-21 before implementation and training)

Background: the §13.10 baselines show ranking skill on validation, but on test Taiwan keeps it only at 1 session. This section registers the Kronos rank model's settings, metric groups and decisions. Reporting Taiwan 1 session separately from long horizons was decided after seeing the §13.10 test results, so decisions on test are a pilot; final confirmation uses data after 2026-10-20 and will be registered separately.

1. **Model and training**:
   - Preset `a-rank64`: the same architecture as `a-alpha-point64` (Kronos-base, LoRA 64, combined numerical branch, market embedding). The head outputs one ranking score per horizon, not multiplied by return scales.
   - Loss: mean squared error on the §13.10 gauss-rank labels; values without labels are excluded and horizons are weighted equally.
   - Data: group A (from 2021-01-01) with unchanged splits; the rank labels are the same Parquet stores as the §13.10 baselines.
   - Optimization settings follow `a-alpha-point64` (5 epochs, the same learning rates and schedule). Early stopping and checkpoint selection both use the §13.10 selection score (minus the validation Rank IC averaged over the three markets with equal weight and over the 14 horizons).
   - Seed 42 only; item 6 decides whether more seeds run.
2. **Evaluation samples**: validation only selects checkpoints; test decides this section; data after 2026-10-20 gives the final confirmation.
3. **Metrics and groups**:
   - Basic quantity: the Spearman Rank IC per (market, horizon, signal date), computed on raw returns (benchmark-relative adjusted log returns) as in §13.10.
   - Horizon groups: short (h = 1), medium (h = 2–5), long (h = 6–14).
   - Market groups: Taiwan (TWSE and TPEX weighted equally) and US.
   - Each cell (market group × horizon group) has a daily value: the mean over the group's horizons, and for Taiwan the mean of TWSE and TPEX. The cell's metric is the mean of the daily values over all signal dates.
   - 95% intervals: a circular moving-block bootstrap of the daily series, 1,000 replicates, seed 11; block length 5 sessions for short, 10 for medium and 28 for long.
   - Paired comparisons: on the same signal dates, subtract the comparator's daily value from Kronos's and compute the interval the same way.
   - Each market and each horizon is also reported, descriptively.
4. **Comparators** (fixed from validation):
   - Primary: PatchTST, the §13.10 baseline with the best validation selection score (market-equal Rank IC 0.057).
   - Secondary descriptions: GBDT, GRU and DLinear; the moving-average crossover is best on test, but that is a post hoc observation and stays descriptive.
5. **Decisions** (test, seed 42):
   - H1 (Taiwan short horizon): the 95% interval lower bound of Kronos's Taiwan short cell is above 0.
   - H2 (Taiwan short horizon beats the baseline): the 95% interval lower bound of the paired Kronos-minus-PatchTST difference in the Taiwan short cell is above 0.
   - H3 (Taiwan long horizon, decided independently): the 95% interval lower bound of Kronos's Taiwan long cell is above 0.
   - The medium group and US are descriptive. An interval that crosses 0 fails, with no further interpretation.
6. **Disposition**:
   - H1 and H2 pass: Taiwan 1-session ranking becomes a candidate signal for hermes; seeds 43 and 44 run, and a forward confirmation is registered separately.
   - H1 passes and H2 fails: Taiwan 1-session ranking exists, but Kronos does not beat PatchTST; the forward confirmation compares the two, with no default preference for Kronos.
   - H1 fails: Kronos has no Taiwan short-horizon ranking skill on test; work returns to model design.
   - H3 only decides whether Taiwan long horizons can be candidates; it does not change the H1/H2 disposition.
   - Date mixing (`a-rank64-datemix`) is registered separately after these results.
7. **Procedure**:
   - Training runs on a RunPod GPU, launched only when the Pod has at least 12 usable CPUs.
   - `export-signal-predictions` exports the selected checkpoint's per-sample validation and test predictions; the baselines' per-sample predictions are downloaded from the volume.
   - Scoring runs locally in `backtest/`, with settings written to `backtest/config/rank_pilot.json` before scoring. Results are written as a bilingual report.

Interpretation limits:

- Test was seen in §13.10, and the groups were chosen after those results, so the test decisions are only a pilot.
- Taiwan test has about 103 signal dates, and h-session labels overlap; the block bootstrap only partly handles the autocorrelation.
- One seed only; §13.10 shows that 0.01–0.02 differences between models are often within noise.
- During test (2025-12 to 2026-05), Taiwan long-horizon ranking failed for every learned model; this may be a property of the period rather than a model defect.

**§13.11 result (2026-10-21)**: H1 passes, Taiwan short cell 0.0642 [0.0271, 0.1015]; H2 fails, Kronos minus PatchTST −0.0022 [−0.0211, 0.0182]; H3 fails, Taiwan long cell 0.0082 [−0.0440, 0.0637]. Under item 6: Taiwan 1-session ranking exists, but Kronos does not beat PatchTST, so the forward confirmation compares the two; seeds 43 and 44 do not run; Taiwan long horizons are not candidates. The selected checkpoint is the first validation, checkpoint-003680 (about 0.2 epoch). The consistency check passed (504 values, largest difference 2.8×10⁻¹⁷). See the [rank model pilot results report](rank_pilot_results.md).

#### 13.12 Training data and feature library for the Taiwan main line (rewritten 2026-10-22 before data preparation and implementation; replaces the `78c257c` draft)

Background:

- **§13.11 results**: Kronos `a-rank64` peaked at the first validation (about 0.2 epoch) with a market-equal Rank IC of 0.060 and did not improve afterwards. On test, Taiwan 1 session is 0.064 [0.027, 0.102] and passes, but it does not beat PatchTST (paired difference −0.002 [−0.021, 0.018]); Taiwan 6–14 sessions is 0.008 [−0.044, 0.064] and fails. Every paired difference between Kronos and PatchTST in the six cells crosses 0. Taiwan long horizons failed on test for every learned model (validation 0.043–0.076 → test −0.037 to 0.012), while the moving-average crossover rose from 0.030 to 0.063.
- **The literature does not settle pooled versus local training; it depends on market size and model family**: Jiang, Kelly and Xiu (JF 2023; daily chart CNNs in 26 markets) find that a US-trained model applied directly beats local retraining on average (Sharpe 0.69 vs 0.29), but the gains concentrate in markets with fewer than about 300 stocks, while markets with 600–3,000 stocks gain little or lose; Choi, Jiang and Zhang (RAPS 2025; monthly firm characteristics, Taiwan included) find market-specific training better; Cakici and Zaremba (JBF 2026; 45 markets) find the two comparable; Hellum, Pedersen and Rønn-Nielsen (2023) find a single global model better. Taiwan's cleaned universe has about 600 TWSE and 290 TPEX stocks per day, the end of the range where US data help little.
- **This project's evidence**: about 83% of Kronos's loss came from US stocks, yet its Taiwan validation Rank IC was on par with the other models; pooling is not the current bottleneck, and nothing shows it helps.
- **Decision (2026-10-22)**: the main line becomes Taiwan only from 2016, and walk-forward retraining replaces single training. The four arms of the earlier draft (pooled / Taiwan only × 2021 / 2016) do not run: they answer a data-engineering question, the target market is Taiwan, and Taiwan only cuts the training data to about one sixth of the pooled set. The US market leaves the main line.

1. **Dataset**:
   - Profile `tw_only` (`twse_official` + `tpex_official`), date contract `[2016-01-01, 2026-06-01)`, built with the existing `configure` and `cpu-prepare`; the data-preparation scripts stay frozen and unedited.
   - Cleaning rules are those of the existing datasets. Storage still follows the fixed split (train to 2025-06-01, validation to 2025-12-01, test to 2026-06-01) as three bar stores; this plan's modelling no longer uses that split (item 4), and all three stores apply the cleaned sample rule used for validation and test.
   - The control plane gains the experiment group letter `t` (`tw_only`, 2016-01-01 to 2026-06-01); the existing `a`/`b` mean US and Taiwan pooled from 2021/2016 and are not reused.
   - Expected size: about 1.63 million Taiwan samples (2016 to 2025-05); about 600 TWSE and 290 TPEX stocks per evaluation day.
2. **Labels**: as in §13.10: the benchmark-relative adjusted log return from the t1 open to the close of session h (h = 1…14) for a signal at the t0 close; rank labels are the gauss-rank within the same signal date and market, with groups under 20 stocks left unlabelled. Grouping depends only on the signal date and market, not on the split, so the label definition is identical across the three stores.
3. **Feature library (Alpha158-style, hand-written, no LLM generation)**:
   - The list is written to `configs/features/alpha_tw_v1.json` before any model trains and is not changed afterwards; a change is a new version with a new identity.
   - Content: the operator and window families of Qlib's Alpha158 (9 candlestick shapes; open, high, low and VWAP relative to the close at lags 0–4; volume lags; ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN and VSUMD over windows 5, 10, 20, 30 and 60), plus Taiwan additions: benchmark-relative 5/20/60-session returns, the within-date, within-market percentile rank of the 20-session mean dollar volume, and the market indicator (TPEX or not). 183 features in all; the data have no VWAP, so the typical price (high + low + close) / 3 stands in for it.
   - Computed only from adjusted OHLCV and the benchmark, using data up to and including the signal date. Normalization is a robust z-score estimated on each walk-forward fold's training window, clipped to ±3, with missing values set to 0; no data outside the fold may be used.
   - Stored like the rank labels: one Parquet store per bar store under `features/<identity>/`, where the identity hashes the bar-store manifest, the feature list and the feature code. GBDT and DL share the same store.
4. **Walk-forward**:
   - Folds are calendar quarters: the first is 2023-Q1 (from 2023-01-02) and the last 2026-Q2 (signal dates with complete labels before 2026-06-01), 14 folds in all.
   - Fold q trains on `[2016-01-01, start of q − 15 sessions)` (15 = the longest horizon, 14, plus 1, so labels never overlap the prediction period); the last 12 months of that window are the fold's validation (early stopping and checkpoint selection) and the rest is train. The window start is fixed (expanding window). Every fold trains from scratch, without the previous fold's weights.
   - Feature normalization, early stopping and selection use only data inside the fold's window; the quarter's predictions come only from that fold.
   - Evaluation periods are reported in three parts: P1 = 2023-01-02 to 2025-05-30 (never evaluated by any model), P2 = 2025-06-02 to 2026-05 (the validation and test periods already seen in §13.10 and §13.11) and P3 = after 2026-06-01 (forward, registered separately).
5. **Procedure**:
   1. On RunPod CPU, create the selection with `configure --profile tw_only --start 2016-01-01 --end 2026-06-01`, then run `cpu-prepare` (ask the user before creating the Pod).
   2. Build the rank labels and the feature library for the three stores with a new CPU job (the research family of `runpod_tmux_launch.sh`) and download them to the local `artifacts/`.
   3. Models and backtests follow §13.13.

Interpretation limits:

- Taiwan only gives up the cross-market sample size; if the §13.13 DL model does not beat LightGBM on Taiwan, that does not show pooled training would fail too.
- The feature library is a fixed hand-written list with no factor screening; screening would add another layer of selection bias and is out of scope.
- Taiwan data for 2016–2020 use the same cleaning rules, but the official sources may lack stocks delisted during the period (survivorship bias, as in §13.4).

#### 13.13 Taiwan cross-sectional ranking pipeline: feature LightGBM baseline, cross-stock DL, ablations and top-k dropout backtest (written 2026-10-22 before implementation)

Background: the goal is a Taiwan daily ranking model that can serve as the basis of a trading strategy, validated the way practitioners do: a GBDT baseline on the same features, walk-forward retraining, Rank IC and ICIR, and a top-k dropout portfolio with costs. Evidence: on the public Qlib Alpha360 benchmark, cross-stock DL (HIST 0.067, IGMTF 0.061, GATs 0.060) beats LightGBM 0.050 and the plain Transformer is worst at 0.033; MASTER (AAAI 2024) reports Rank IC 0.076 against XGBoost 0.050, but a third-party reproduction reached only 0.030, so the reproducibility risk is controlled with ablations and multiple seeds; QuantBench finds IC-type losses best and 3-month rolling retraining best, with no retraining worst; Forecast Collapse (arXiv 2608.14106) shows that time-series foundation models optimize per-series objectives and that a cross-sectional correlation term raises IC substantially; the Taiwan appendix of the Kronos paper shows generic time-series foundation models at about 0 zero-shot Rank IC on Taiwan. LLM-in-the-loop alpha mining is excluded on cost; Kronos is no longer the backbone (§13.11).

1. **Models** (data, labels, features and walk-forward as in §13.12):
   - **B0 `t-lgbm-alpha` (baseline, built first)**: LightGBM, one regressor per horizon, target the gauss-rank label, inputs the feature library. Hyperparameters fixed before training: `num_leaves` 31, `learning_rate` 0.05, `feature_fraction` 0.8, `bagging_fraction` 0.8, `min_data_in_leaf` 200, early stopping (50 rounds) on the fold validation's horizon-equal Rank IC. Seeds 42, 43 and 44, reported individually and as the mean ensemble. If `lightgbm` cannot be installed on RunPod CPU, scikit-learn's `HistGradientBoostingRegressor` with matching parameters is used and the report says so.
   - **D0 `t-xs-scratch` (main model)**: a cross-stock transformer. An intra-stock encoder (patch transformer over the 128-session OHLCV window) with the feature-library vector as a side input → all stocks of one signal date and market as one batch (the existing `date_market_sampler`) → inter-stock attention (1–2 layers) → market-state gating (from that date's and market's cross-sectional feature summary) → ranking scores for 14 horizons. Loss = mean squared error on the gauss-rank labels + λ × (1 − the date's cross-sectional Pearson correlation), λ = 1, fixed before training. Model width 128, 2 + 2 layers, about 1–3M parameters. Seeds 42, 43 and 44.
   - **D1 pluggable encoders** (only after D0 is done; each is a single-factor change to D0): the contract is raw window `(B, L, F)` in, `(B, D_enc)` out; normalization lives inside each adapter; frozen encoders must cache embeddings per (symbol, date) as Parquet, so walk-forward refits train only the cross-stock module. Order: `t-xs-kronos-small` (frozen) → `t-xs-kronos-base-lora` (only if the former is effective) → `t-xs-chronos2-small` (frozen, Apache-2.0). TimesFM 2.5 only if budget remains.
   - **Ablations** (single-factor changes to D0, seed 42): A1 inter-stock attention off; A2 λ = 0 (mean squared error only); A3 no feature side input; A4 market-state gating off.
   - The new DL training entry does not go through the §13.10 baseline-suite contract; its preflight instead checks that B0's walk-forward manifest exists.
2. **Metrics**:
   - The Spearman Rank IC per (market, horizon, signal date), computed on raw returns (as in §13.11).
   - Cells: Taiwan short (h = 1), Taiwan holding (h = 5–10, matching holding periods of days to two weeks) and Taiwan long (h = 11–14); Taiwan weights TWSE and TPEX equally. Each market and horizon is also reported.
   - A cell's Rank IC is the mean of its daily values; Rank ICIR is that mean divided by the standard deviation of the daily values; the share of positive days is also reported.
   - 95% intervals: circular moving-block bootstrap, 1,000 replicates, seed 11; block length 5 sessions for short, 20 for holding and 28 for long. Paired comparisons subtract on the same signal dates and use the same method.
   - The primary decision period is P1 ∪ P2 (2023-01-02 to 2026-05, about 830 signal dates); P1 and P2 are also reported separately.
3. **Top-k dropout backtest** (local `backtest/`, settings written to `backtest/config/topk_dropout.json` before scoring):
   - Markets separately: TWSE k = 50 with 5 replaced per day; TPEX k = 30 with 3 replaced per day. Each day, by the previous close's ranking score, the n lowest-ranked holdings are sold and the n highest-ranked non-holdings bought at the t+1 open; per-position open-to-open accounting (as in §13.5).
   - The horizon score used: the h in {1, 5, 10} with the highest TWSE net IR on P1; once chosen it is fixed for P2 and the forward period.
   - Costs follow the primary scenario of `backtest/config/stage1.json` (commission discount 0.6, slippage ×1, conservative turnover); discounts 1.0 and 0.28 and slippage ×0 and ×2 are also reported.
   - Reported: annualized net excess (over the same market's equal-weight universe), IR, turnover and maximum drawdown; intervals from a monthly-block bootstrap, 1,000 replicates, seed 11.
4. **Decisions** (fixed beforehand; B0 and D0 are three-seed mean ensembles):
   - **H-A (DL beats the baseline)**: the paired Rank IC difference D0 − B0 has a 95% interval lower bound above 0 in both the Taiwan short and the Taiwan holding cells, over P1 ∪ P2.
   - **H-B (tradeable)**: the TWSE top-k dropout annualized net excess has a 95% interval lower bound above 0 under the primary cost scenario, and D0's IR is not below B0's.
   - **H-C (forward confirmation)**: data after 2026-06-01 through the existing forward pipeline, with criteria registered separately.
   - **Pretrained encoders**: a D1 model counts as effective only if its paired difference against D0 has an interval lower bound above 0 in the Taiwan short or holding cell; otherwise it is ineffective.
   - **Ablations** are descriptive, to attribute D0's gain to its parts; no thresholds.
5. **Disposition**:
   - H-A fails in both cells: B0 becomes the strategy baseline (it must still pass H-B to count as tradeable); DL stays research and gets no more GPU.
   - H-A passes in at least one cell: D0 is the candidate model and proceeds to D1 and H-B.
   - H-B fails: ranking skill exists but is not tradeable; work returns to the strategy layer (rotation frequency, holding period, costs), not the model; H-C does not run.
   - H-B passes: H-C is registered and confirmed on P3.
6. **Procedure and order** (GPU last):
   1. The §13.12 dataset, labels and feature library (CPU).
   2. B0 walk-forward (RunPod CPU or local, with a bounded number of workers).
   3. D0 with three seeds (RunPod GPU, RTX 4090 class; 14 folds × 3 seeds, estimated 10–15 hours and about 5–10 USD; ask the user before creating the Pod).
   4. Ablations A1–A4 (seed 42, estimated 8–10 hours).
   5. D1 encoders (frozen modes first build the embedding cache, then each fold trains only the cross-stock module).
   6. Top-k dropout backtest (local).
   7. H-C registration and forward confirmation.
   - Every model's per-sample walk-forward predictions are stored as Parquet (all signal dates of P1 ∪ P2) for local scoring and backtests; results are written as bilingual reports.

7. **Implementation registration (2026-10-23, fixed before training, no D0 result seen)**:
   - D0 training parameters: AdamW lr 5e-4, weight decay 1e-4, at most 20 epochs, early stopping
     on the fold validation's daily Rank IC (14 horizons and both markets weighted equally) with
     patience 4, gradient clip 1.0, bf16 autocast on CUDA; one batch is every stock of one signal
     date and market; the feature side input is cross-sectionally z-scored within the batch
     (clipped to ±5); windows are normalized as log prices relative to the last close and log
     volume relative to the window median. Encoder patch 8, width 128, 2 layers; inter-stock
     attention 2 layers, 4 heads, feed-forward 256, dropout 0.1.
   - D1 encoders: `t-xs-kronos-small` uses the last-position hidden state of
     `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`; `t-xs-chronos2-small` averages the
     `embed` output of `autogluon/chronos-2-small` (28M, Apache-2.0) over variates and patches,
     with the five OHLCV channels as the variates of one series. Both are frozen and their
     embeddings are cached once per (symbol, cutoff).
   - Paired comparison: D − B0 subtracts the cell series on common signal dates; circular
     moving-block bootstrap (1,000 replicates, seed 11, blocks 5/20/28); H-A is decided on the
     short and holding cells over P1 ∪ P2.
   - Entry point: `runpod_workflow.sh xs-train --experiment <name>`, one GPU Pod per experiment.

Interpretation limits:

- P2 was seen in §13.10 and §13.11, and P1 is evaluated for the first time here, but the data rules and model design were decided after seeing P2's aggregate results; only P3 is a true confirmation.
- Ablations and D1 use seed 42 only; §13.10 shows that 0.01–0.02 differences between models are often within noise.
- A third-party reproduction of MASTER reached less than half the paper's Rank IC; D0 is an independent implementation of that design and is not guaranteed to reach the paper's numbers.
- TPEX liquidity is low, and machine-learning stock selection alpha in Taiwan is often limited by investability (Chang and Chiu, SSRN 2026); the backtest does not model volume caps, round lots, minimum commissions or dividend taxes.
- If `lightgbm` cannot be installed and scikit-learn is used instead, the baseline is still B0, but comparability with the public Qlib benchmark falls.

**§13.13 B0 result (2026-10-23)**: B0 `t-lgbm-alpha` (`launch-20261022T234922Z-22048038`, 14 folds × 14 horizons × 3 seeds = 588 models) on P1 ∪ P2: short cell 0.1178 [0.1091, 0.1266], holding cell 0.0859 [0.0650, 0.1061], long cell 0.0834 [0.0541, 0.1129]; P2 alone: holding 0.0343 [−0.0004, 0.0726], long 0.0107 [−0.0372, 0.0600]. The item 3 top-k dropout applied to B0 selects h = 10; TWSE P1 ∪ P2 annualized net excess under the primary costs is −6.3% [−19.0%, +7.1%] with IR −0.58, so H-B fails. Under item 5: B0 is the strategy baseline but not tradeable, H-C does not run for B0, and strategy-layer changes need their own registration. D0 (H-A) runs as step 3 of item 6 and is recorded separately. See the [B0 baseline and backtest results report](lgbm_baseline_results.md).

**§13.13 D0 result (2026-10-23)**: D0 `t-xs-scratch` (`launch-20261023T015746Z-224647797`, 14 folds × 3 seeds = 42 fits, RTX PRO 4500, 4.8 hours) on P1 ∪ P2: short cell 0.1221 [0.1129, 0.1311], holding 0.0921 [0.0739, 0.1096], long 0.0893 [0.0664, 0.1121]; seeds 0.1185 / 0.1181 / 0.1209 in the short cell. Paired D0 − B0: short +0.0043 [+0.0015, +0.0071] (passes), holding +0.0062 [−0.0012, +0.0136] (fails), so **H-A passes in one cell** and under item 5 D0 is the candidate model and proceeds to the ablations and D1. D0's top-k dropout (item 3) selects h = 10; TWSE P1 ∪ P2 annualized net excess −7.3% [−17.9%, +3.5%], IR −0.79 (below B0's −0.58), so **H-B fails**. GPU utilization was only 13–17% (the per-date grouping is the bottleneck); that Pod has no clock-monitor record. Ablations A1–A4 and D1 are recorded separately.

**§13.13 ablation A1 result (2026-10-23)**: `t-xs-a1-no-cross` (inter-stock attention off, seed 42, `launch-20261023T075624Z-2596926448`, 14 fits, 70 minutes) on P1 ∪ P2: short 0.1195 [0.1098, 0.1284], holding 0.0879, long 0.0832; paired against B0 the short cell is +0.0017 [−0.0014, +0.0047] (not significant). D0's single seed 42 has a short cell of 0.1185, on a par with A1: the D0 ensemble's +0.0043 over B0 comes from the three-seed ensemble, not from inter-stock attention. A2–A4 and D1 are deferred by the user's decision of 2026-10-23 pending the §13.16 label results.

#### 13.14 Taiwan strategy layer: cost-aware long-only band portfolio with a TAIEX futures overlay (written 2026-10-23 before implementation and scoring)

Background: both B0 and D0 of §13.13 rank (P1 ∪ P2 short-cell Rank IC 0.118 / 0.122), but the registered top-k dropout (5 of 50 replaced daily) fails H-B under Taiwan costs: about 10% daily turnover costs about 17% per year, more than the P1 gross excess; in P2 the h = 5–14 Rank IC falls to 0.02–0.05 while h = 1 keeps 0.11. Under §13.13 item 5 the work returns to the strategy layer. This section registers the Taiwan version of the industry's cost-mitigation practice: a buy/hold band (the single most effective method in Novy-Marx and Velikov 2016), signal smoothing, a turnover cap, a liquidity universe, cost awareness (Jensen, Kelly, Malamud and Pedersen 2024), and TAIEX futures (transaction tax 0.002%) instead of stocks for the beta hedge and for the market exposure of idle cash. The futures enter no model training; they are used only in the backtest layer, with data from TAIFEX's official daily download rather than a third-party API.

1. **Inputs**: walk-forward predictions of B0 `t-lgbm-alpha` and D0 `t-xs-scratch` (three-seed ensembles; D1 added if effective); the dataset's raw `market.parquet` (unadjusted open/close and volume, adjusted close); TAIFEX TX daily rows 2022-12 to 2026-06 (`twbacktest.taifex download-contract`), with a continuous front month that rolls at the open of the expiry month's third Wednesday and chains opens by ratio.
2. **Universe**: stocks only (no ETFs or indices); markets separately, TWSE decides and TPEX is descriptive. Liquidity filter: the median dollar volume (close × volume) over the 20 sessions before the signal date ≥ NT$30 million (TWSE) / NT$10 million (TPEX), and close ≥ NT$10; fixed before training, not selected.
3. **Signal**: one horizon's ensemble score (h chosen from the item 7 grid); converted to a within-market percentile rank each day, then EMA-smoothed per symbol with a 3-session half-life (fixed), reset after a gap of more than 5 sessions.
4. **Portfolio rule (band)**: buy only when entering the top K, sell only when falling out of the top M (M = m × K); checked every session on the previous close's smoothed score and executed at the next open; at most ⌈K/10⌉ new buys per session; capital in K equal slots with per-position open-to-open accounting (the §13.5 engine: the buffer rule of `positions.simulate` plus the daily entry cap); limit-up opens are not bought and limit-down opens not sold (as in §13.13). The primary analysis uses fractional slots (large capital or odd lots); a board-lot feasibility sensitivity at NT$10 million and NT$30 million of capital leaves a slot in cash, and counts it, when it cannot buy one lot (1,000 shares); the NT$20 minimum commission is below 0.01% at both capital levels and is reported, not simulated.
5. **Costs**: stocks as in the primary scenario of `backtest/config/stage1.json` (commission 0.1425% × 0.6 discount, transaction tax 0.3%, slippage TWSE 0.1% / TPEX 0.2%), with discounts 1.0 / 0.28 and slippage ×0 / ×2 also reported. TX per side: 0.002% transaction tax on contract value, NT$50 commission per contract, 1 tick (NT$200) slippage; a roll pays both sides once.
6. **TAIEX futures overlay** (all on the continuous front-month TX series, position adjusted at each open to the previous session's stock value):
   - **Hedged**: short notional = β × stock value, β = 1 (fixed); a margin reserve of 15% of the notional (about 5% initial margin plus a 10% mark-to-market buffer) is taken from total capital, and stocks use the rest; when the reserve falls below 10% of the notional it is replenished at the next open by selling stocks pro rata and one margin call is counted. The hedged benchmark is 0.
   - **Cash-equitized**: cash not invested in stocks (empty slots, blocked buys, lot shortfalls) holds an equal long TX notional, with margin taken from that cash. Same benchmark as long-only.
   - **TX long reference**: holding only the front-month TX long (with roll costs) against the TAIEX price index, to report the basis and roll drag.
7. **Selection and decisions**:
   - Grid: h ∈ {5, 10}, K ∈ {30, 50}, m ∈ {2, 3}, 8 combinations; for each model the selected parameters are those with the highest long-only TWSE P1 net excess IR (over the equal-weight liquid universe of the same market) under the primary costs, then fixed for P2, P1 ∪ P2 and the forward period.
   - **H-D (tradeable strategy layer)**: the selected TWSE long-only portfolio's P1 ∪ P2 annualized net excess over the equal-weight universe has a 95% interval lower bound above 0 under the primary costs (monthly-block bootstrap, 1,000 replicates, seed 11). B0 and D0 are judged separately.
   - **H-D-hedged**: the hedged version with the same parameters has a P1 ∪ P2 annualized net (absolute) return with a 95% lower bound above 0.
   - Forward candidate: among the models passing H-D, the one with the higher TWSE P1 ∪ P2 net excess IR; none if both fail.
8. **Report (fixed columns)**: besides net excess with its interval and IR, always list annualized net return, CAGR, Sharpe, maximum drawdown, turnover, cost drag, excess over the TAIEX price index and over the equal-weight universe, the share of buys blocked by board-lot feasibility and the number of margin calls.
9. **Disposition**: H-D passes → register H-C (P3 after 2026-06-01) with the candidate model; H-D fails → long-only stock holding cannot turn the ranking skill into returns under Taiwan costs, and the next candidates are a long-short inside the stock-futures universe (§13.9 capturability) or an h = 1 day-trade route, each needing its own registration.
10. **Procedure**: `backtest/config/tw_strategy.json` is written and committed before scoring; synthetic-data tests first verify the band, the entry cap, lot feasibility, the hedge and the margin accounting (including calls); then B0 and D0 are scored and written up as a bilingual report.

Interpretation limits:

- One of 8 parameter sets is selected on P1, so P1 carries selection bias; the P1 ∪ P2 decision includes P1, and P3 remains the true confirmation.
- The liquidity thresholds and β = 1 are fixed reasonable values, not optimized; a single β ignores stock-level beta differences.
- The margin model is simplified (a fixed reserve ratio and call rule) and does not use TAIFEX's daily margin tables; futures liquidity and the after-hours session are not modelled.
- Volume participation limits for large capital, dividend taxes and securities lending are not modelled; lot feasibility is reported at two capital levels only.

**§13.14 result (2026-10-23)**: both B0 and D0 select h = 10, K = 50, m = 3 (TWSE P1 long-only net excess IR 0.42 / 0.52). Daily turnover 2.0% / 2.5%, cost drag 3.4% / 4.8% per year; P1 net excess +5.4% / +5.5% (interval spans 0), P2 gross excess −27% / −31%, P1 ∪ P2 annualized net excess B0 −4.7% [−19.7%, +10.5%] and D0 −5.4% [−18.2%, +7.7%], so **H-D fails**; the hedged P1 ∪ P2 annualized net return is −12.5% / −13.3% with 28 / 27 margin calls, so **H-D-hedged fails**. Absolute: long-only annualized net return +23.8% / +23.1%, Sharpe 1.72 / 1.50, while the TAIEX total-return index made +41% per year (the dataset's "TAIEX index" is the official total-return index, so the "price index" of item 8 reads as the total-return index). Capital scenarios (fractional, NT$10 million with lots / odd lots, NT$30 million, NT$100 million, NT$300 million, NT$1 billion; odd lots with 0.1% extra slippage, minimum commissions, a 5% participation cap): NT$10 million with lots loses 1.5–2.7 points and odd lots recover it; NT$300 million concentrates in large liquid names through the participation cap and narrows the P1 ∪ P2 net excess to −2.9% / −1.3% with the interval still spanning 0; no capital scenario passes H-D. Forward candidate: none. Under item 9 the next step is a separate registration of a stock-futures long-short or an h = 1 day-trade route. See the [strategy-layer results report](strategy_layer_results.md).

#### 13.15 Stock-futures short-side day trade: tradeability of the h = 1 signal (written 2026-10-23 before implementation and scoring)

Background: the §13.14 long-only holding fails H-D in every capital scenario; the stock-futures universe diagnostic (strategy-layer report, section 7) shows that the h = 1 ranking keeps 80–90% of its skill among names with stock futures (liquid segment Rank IC 0.10–0.11, P2 0.08–0.09) and that the h = 1 gross spread comes almost entirely from the short side (bottom decile next-session intraday −0.36% to −0.45%, top decile only +0.04% to +0.07%, negative universe intraday mean). Shorting stocks is constrained by borrow, the uptick rule and the day-trade tax; stock futures pay a 0.002% tax and need no borrow. This section registers a short-side day trade backtested on official TAIFEX futures prices.

1. **Data**: the TAIFEX daily download of every contract (`twbacktest.taifex download`, 2022-12 to 2026-06): each stock-futures contract's daily open, settlement, volume and session; underlying mapping from the TAIFEX stock-futures list (contract code → stock symbol, 2,000 or 100 shares per contract). The TX front month as in §13.14. Signals are the h = 1 ensemble scores of B0 and D0.
2. **Universe**: on signal date t, underlyings whose near-month stock futures traded in the regular session and whose near-month 20-session mean volume is ≥ 100 contracts (standard-contract equivalents); the standard contract (2,000 shares) is preferred, and names with only a small contract (100 shares) are priced on it and counted. On an expiry day the next month is used. Names without a futures price on the next session are skipped and counted.
3. **Positions and prices**: ranked within the universe by the previous close's h = 1 score, the bottom q (q ∈ {10%, 20%}, ⌈n × q⌉ names) is shorted at the futures open (08:45) of session t+1 and covered at that session's settlement price (computed from trades before 13:30, the official daily mark); no overnight position. Versions: **short-only** (decides); **long-short** (plus an equal-notional long of the top q); **hedged** (short-only plus an equal-notional long TX, open to settlement).
4. **Capital and contracts**: the §13.14 capital amounts (fractional, NT$10 million, NT$30 million, NT$100 million, NT$300 million, NT$1 billion; no odd-lot scenario for futures). Target gross notional = L × capital with L = 1 (fixed); notional per name = total / names; contracts = ⌊notional per name / (open × shares)⌋, names below one contract skipped and counted; contracts per name ≤ 5% of that contract's volume on the day. Margin uses TAIFEX's highest tier (initial 20.25%, maintenance 15.53%); margin usage is reported and L × 20.25% > 100% is infeasible.
5. **Costs** (per side): 0.002% tax on notional; NT$30 commission per contract (NT$60 also reported); 1 tick slippage (2 ticks also reported), the tick following the underlying's price band (< NT$10: 0.01; 10–50: 0.05; 50–100: 0.1; 100–500: 0.5; 500–1,000: 1; ≥ 1,000: 5) × shares per contract. Primary scenario: 1 tick, NT$30. TX per side: 0.002% tax, NT$50 per contract, 1 point.
6. **Metrics and intervals**: daily return = that day's P&L / capital (fractional: notional-weighted return × L), gross (no costs) and net; annualized return (× 252), CAGR, Sharpe, maximum drawdown, hit rate, cost drag, mean names and contracts, shares blocked by contract size or participation, margin usage. Intervals: circular block bootstrap of the daily net returns, block 5, 1,000 replicates, seed 11. Periods P1, P2, P1 ∪ P2 by signal date.
7. **Selection and decisions**:
   - q is chosen on the combined TWSE plus TPEX universe (the stock-futures universe is not split by market), P1, short-only, fractional, primary costs, by the highest net Sharpe; B0 and D0 separately.
   - **H-E (tradeable short day trade)**: the selected q's short-only version has a P1 ∪ P2 annualized net (absolute) return with a 95% lower bound > 0, and a positive P2 annualized net return. B0 and D0 are judged separately.
   - Long-short and hedged versions are descriptive, as are capital scenarios; if the NT$10 million scenario skips more than 50% of names for lack of a whole contract, that capital is marked infeasible in the report.
   - Forward candidate: among the models passing H-E, the higher P1 ∪ P2 net Sharpe.
8. **Report (fixed columns)**: for every version, capital scenario, cost scenario and period: annualized gross return, annualized net return with interval, CAGR, Sharpe, maximum drawdown, hit rate, cost drag, mean names, mean contracts, blocked shares, margin usage; plus the standard deviation of the daily differences between the futures open and the stock open and between the settlement and the stock close (timing noise).
9. **Disposition**: H-E passes → register H-C (the P3 forward period after 2026-06-01, with actual futures prices) for the candidate model; H-E fails → the h = 1 signal is not tradeable with the available instruments and costs either, model and strategy layers return to research, and the next registered experiment targets the signal itself (residualized labels, multi-horizon blending).
10. **Procedure**: `backtest/config/futures_daytrade.json` is written and committed before scoring; synthetic-data tests first cover contract sizing, participation, costs and margin; then B0 and D0 are scored and written up bilingually.

Interpretation limits:

- Contract codes are mapped through the 2026-10 underlying list, so stock futures delisted during the period cannot be mapped (survivorship bias, about 38 codes).
- The 08:45 futures open is a tradable price, but liquidity between 08:45 and 09:00 is thinner than after the stock open; slippage is counted in ticks, and insufficient opening volume is not modelled.
- The settlement price is the official mark; the actual cover trades around 13:30 and may deviate from it.
- Margin uses the highest tier for every name while actual per-name tiers are lower; intraday margin calls are not modelled.
- q is chosen on P1 and the P1 ∪ P2 decision includes P1; P3 is the confirmation.

**§13.15 result (2026-10-23)**: both models select q = 10% (about 13 names a day). On official TAIFEX futures opens and settlements the short-only P1 ∪ P2 annualized gross return is +61.5% (B0) and +64.5% (D0), positive in P1 and P2, but under the primary costs the annualized net return is −63.4% [−91.5%, −36.0%] (B0) and −60.5% [−86.9%, −33.7%] (D0), negative in P2 as well, so **H-E fails**. The cost drag of about 125% a year is almost all tick slippage (about 0.23% per side); the break-even cost per side is about 0.12%. Long-short and hedged versions are worse or not rescued; all six capital scenarios are negative. Under item 9 the h = 1 signal is not tradeable with the available instruments and costs, model and strategy layers return to research, and the next step is §13.16. See the [stock-futures short day-trade results report](futures_daytrade_results.md).

#### 13.16 Residualized labels and multi-horizon blending: stability of multi-session ranking in concentrated-style periods (written 2026-10-23 before implementation)

Background: in P1 the Rank IC of B0 and D0 (§13.13) eases from 0.13 at h = 1 to 0.11 at h = 10, but in P2 (from 2025-06, a mega-cap rally in which TAIEX doubled in a year while the equal-weight universe rose a third) h = 5–14 fall to 0.02–0.05 while h = 1 keeps 0.11; the §13.14 long-only strategy and h = 10 inside the stock-futures universe fail for that reason. The literature and industry practice (strategy-layer report and the discussion preceding this registration): factor half-lives matched to rebalancing (Flint and Vermaak), fast and slow signals weighted by decay (Gârleanu and Pedersen 2013), multi-scale signal blending (Han, Zhou and Zhu 2016; Jiang, Kelly and Xiu 2023), and residualizing returns against market, size and industry so the model learns relative performance (the standard risk-model neutralization). The current label removes the market direction only, not size and industry exposure. This section registers a label change without a model change, verified first on the cheapest model, B0.

1. **Data**: training data, feature library and walk-forward split exactly as in §13.12 and §13.13. One new company reference table `configs/reference/tw_company_reference_20261022.csv` from the TWSE OpenAPI `opendata/t187ap03_L` (listed) and the TPEX OpenAPI `mopsfin_t187ap03_O` (OTC), snapshot of 2026-10-22: symbol, market, two-digit industry code, issued common shares. It covers 99.2% (TPEX) and 99.8% (TWSE) of the names still trading in 2026-05, and 90% / 93% of all samples including delisted names.
2. **Residualized labels** (new label-store schema `cross-sectional-gauss-rank-residual-v1`, identity including the residualization spec and the reference table digest; the existing store is untouched): within each (signal date, market) group, each horizon's raw alpha target (already relative to the market index) is regressed cross-sectionally on an intercept, a size proxy and industry dummies, and the residual is gauss-ranked. Industries with a single name in a group, or groups below `min_group_size`, are merged into "other".
   - **Variant A (primary)**: size proxy = log of the median dollar volume over the 20 sessions before the signal date (available daily, no look-ahead); industry = the reference table's two-digit code, missing values as an "unknown" class.
   - **Variant B (sensitivity)**: size proxy = log(raw close on the signal date × issued shares from the reference table); rows missing from the table use variant A's proxy. Share counts are a snapshot, so companies with past stock dividends or capital reductions are biased; the report says so.
3. **Models**: B0 `t-lgbm-alpha` with unchanged settings, hyperparameters and seeds, retrained on variant A and variant B labels (RunPod CPU) through a new `--label-variant`; the feature library is reused. D0 and D1 are retrained on the new labels only after B0 passes item 5 (D1's embedding cache does not depend on labels and is not recomputed).
4. **Metrics**: Spearman Rank IC per (market, horizon, signal date) against two targets: (a) the **residual target** (what the model predicts; ranking skill) and (b) the **raw alpha target** (what a long-only book earns; usability). Cells, periods and bootstrap as in §13.13. The paired difference between the new and the old B0 on common signal dates is also reported.
5. **Decision (H-F, residualization stabilizes multi-session ranking)**: variant A's B0 must have (i) a P2 holding-cell (h = 5–10) Rank IC against the residual target with a 95% lower bound > 0, and (ii) a P1 ∪ P2 short-cell Rank IC against the raw alpha target not below the old B0 minus 0.01 (paired-difference lower bound > −0.01). Both are required. Variant B is descriptive unless variant A fails and variant B passes, in which case variant B becomes primary and the report flags it as a second chance.
6. **Multi-horizon blending** (strategy layer, descriptive): the `tw_strategy` signal becomes a decay-weighted mean of the h = 1–14 scores (weights ∝ exp(−h/τ), τ ∈ {3, 7} as a new grid dimension beside the single h), other rules as in §13.14; run for the new-label B0 and the old B0. Descriptive only; the §13.14 H-D rule is unchanged.
7. **Disposition**: H-F passes → D0 is retrained on the new labels (three seeds), then the ablations and D1 are decided on the new labels; §13.14 and §13.15 are rerun with the new models under their own rules. H-F fails → the multi-session failure in concentrated-style periods is not caused by label exposure, multi-session horizons leave the main line, and the models serve only as h = 1 fast signals (§13.15).
8. **Procedure**: the reference table is fetched from the official OpenAPIs by a script and committed (with snapshot date and sha256); the label-variant spec `configs/labels/residual_v1.json` is written before implementation; synthetic-data tests first cover the regression residuals, industry merging and the identity; then the label stores are built and B0 variants A and B trained on RunPod CPU; results are written up bilingually.

Interpretation limits:

- Industry and share counts are a 2026-10-22 snapshot; the 7–10% of samples that are delisted have no industry (unknown class) or shares (variant A proxy), a survivorship bias.
- Residualization removes linear size and industry exposure; non-linear and other style exposures (momentum, volatility) remain in the target.
- Rank IC on the residual target is not tradeability: a long-only book still bears size and industry factor returns, and only the §13.14 decisions measure tradeability.
- P2 has been seen; H-F is a repair designed after observing the failure, and P3 is the true confirmation.

**§13.16 variant A result (2026-10-23)**: B0 retrained on the `residual_v1a` labels (`launch-20261023T103805Z-2637116736`, 588 models, `t-lgbm-alpha-residual_v1a`). Against the residual target: short P1 ∪ P2 0.1008 [0.0954, 0.1061] (ICIR 1.19), holding P2 **0.0646 [0.0457, 0.0856]** (condition (i) passes), long P2 0.0549 [0.0324, 0.0763]. Against the raw alpha target: short P1 ∪ P2 0.1175, paired difference to the old B0 −0.0002 [−0.0013, +0.0009] (condition (ii) passes); holding P2 0.0434 [0.0103, 0.0794], paired difference +0.0090 [+0.0055, +0.0125]; long P2 0.0239 [−0.0216, 0.0692] (old B0 0.0107). **H-F passes.** Residualization did not hurt h = 1, and the P2 multi-session ranking against the residual target returns to 0.05–0.06 with intervals above 0; the improvement against raw alpha is smaller (holding +0.009) because a long book still earns the size and industry factor returns. Under item 7: D0 is retrained on the new labels and §13.14 / §13.15 are rerun with the new models. Variant B is recorded separately.

### 14. Remote-validation boundary

Python environment, Poetry lock, lint, pytest, data preparation, model prefetch,
GPU training, and validation all run on RunPod. The local machine only edits
source/config, performs shell-level static checks, syncs S3, controls Pods, and
downloads artifacts. Local Python/PyTorch/CUDA state is not runtime evidence.
