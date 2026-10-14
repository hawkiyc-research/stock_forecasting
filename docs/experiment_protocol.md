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

### 14. Remote-validation boundary

Python environment, Poetry lock, lint, pytest, data preparation, model prefetch,
GPU training, and validation all run on RunPod. The local machine only edits
source/config, performs shell-level static checks, syncs S3, controls Pods, and
downloads artifacts. Local Python/PyTorch/CUDA state is not runtime evidence.
