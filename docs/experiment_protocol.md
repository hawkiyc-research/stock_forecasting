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

**§13.14 結果（2026-10-23）**：B0 與 D0 都選出 h = 10、K = 50、m = 3（TWSE P1 多頭版淨超額 IR 0.42／0.52）。每日換手 2.0%／2.5%、成本拖累 3.4%／4.8%／年；P1 淨超額 +5.4%／+5.5%（區間跨 0），P2 毛超額 −27%／−31%，P1 ∪ P2 年化淨超額 B0 −4.7% [−19.7%, +10.5%]、D0 −5.4% [−18.2%, +7.7%]，**H-D 未通過**；對沖版 P1 ∪ P2 年化淨報酬 −12.5%／−13.3%、28／27 次追繳，**H-D-hedged 未通過**。絕對報酬：多頭版年化淨報酬 +23.8%／+23.1%，Sharpe 1.72／1.50，但 TAIEX 報酬指數同期 +41%／年（本節所稱「TAIEX 指數」在資料集中為官方報酬指數，含息；第 8 項的「價格指數」應讀為報酬指數）。整張可行性：1,000 萬元資本下 34–37% 的候選買單買不到一張。前瞻候選模型：無。依第 9 點，下一步為另行登記股票期貨多空或 h = 1 當沖路線。見 [台股策略層結果報告](strategy_layer_results.md)。

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

**§13.14 result (2026-10-23)**: both B0 and D0 select h = 10, K = 50, m = 3 (TWSE P1 long-only net excess IR 0.42 / 0.52). Daily turnover 2.0% / 2.5%, cost drag 3.4% / 4.8% per year; P1 net excess +5.4% / +5.5% (interval spans 0), P2 gross excess −27% / −31%, P1 ∪ P2 annualized net excess B0 −4.7% [−19.7%, +10.5%] and D0 −5.4% [−18.2%, +7.7%], so **H-D fails**; the hedged P1 ∪ P2 annualized net return is −12.5% / −13.3% with 28 / 27 margin calls, so **H-D-hedged fails**. Absolute: long-only annualized net return +23.8% / +23.1%, Sharpe 1.72 / 1.50, while the TAIEX total-return index made +41% per year (the dataset's "TAIEX index" is the official total-return index, so the "price index" of item 8 reads as the total-return index). Lot feasibility: at NT$10 million, 34–37% of candidate buys cannot afford one lot. Forward candidate: none. Under item 9 the next step is a separate registration of a stock-futures long-short or an h = 1 day-trade route. See the [strategy-layer results report](strategy_layer_results.md).

### 14. Remote-validation boundary

Python environment, Poetry lock, lint, pytest, data preparation, model prefetch,
GPU training, and validation all run on RunPod. The local machine only edits
source/config, performs shell-level static checks, syncs S3, controls Pods, and
downloads artifacts. Local Python/PyTorch/CUDA state is not runtime evidence.
