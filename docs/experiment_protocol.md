# 條件式 Alpha 模型實驗協定

## 中文

### 1. 研究問題

主要問題：在嚴格 point-in-time、next-open execution 與 chronological evaluation
下，以金融 K-line 預訓練的 Kronos-base 經 LoRA 微調後，能否比簡單規則、傳統 ML
與小型 causal DL baseline 更準確地預測個別股票／ETF 從可設定的第 1、2 或 3 個
持有交易日起至固定第 14 日、
相對市場 benchmark 的 alpha 條件分布，並辨識超過成本門檻的個股絕對／alpha 方向？

本協定不把 validation Sharpe 或單次回測視為「可交易 alpha」的充分證據，也不把模型
輸出直接當交易規則。技術指標 baseline 保留為對照；各版方向 preset 另將十個有定義的
歷史技術通道輸入 Kronos predictor，不用 baseline 規則取代 backbone。

### 2. 固定假設與範圍

- 商品層級預測，不是投資組合預測。
- 日線 OHLCV only。
- 美國與台灣股票、ETF；期貨與選擇權不在本 PoC。
- 收盤 `t` 後產生訊號，下一市場交易日 raw open 進場，不跳過缺成交日。
- 進場日算第 1 個持有交易日。
- `h_start` 可設定為 1、2 或 3；horizons 是 `h_start` 到固定第 14 日。
- label 固定為 benchmark-relative adjusted execution log return。
- CAPM abnormal return 只作未來 diagnostic/ablation。
- quantile 用 pinball，獨立 ranking head 用同日排序 loss；`direction64`／`direction64-v2` 另有個股絕對／alpha 三分類 CE，原五級方向後處理仍保留。

### 3. 模型假說

主假說：共享金融預訓練 encoder、歷史 benchmark conditioning 與可讀出的歷史尺度
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
`configs/data_cleaning.json` 為準；不建立完整 windows／labels，不刪除真實極端報酬。

| 項目 | Stage 1 | Stage 2 |
|---|---|---|
| 目的 | 驗證完整腳本、資料、GPU、loss、checkpoint 與 validation | 完整 train split 的 PoC 結果 |
| train 樣本 | 動態年度加權取樣；每 epoch 呈現有效數的 5%，上限 500,000 | 動態年度加權取樣；每 epoch 呈現次數等於有效 train 數 |
| epoch 上限 | 2 | 5 |
| validation cadence | 每個 epoch 的 20%／40%／60%／80%／100% | 同左 |
| early stopping | 設定的 validation monitor 連續 5 次未改善，且完成最低 LR 的兩個訓練間隔；第 1 epoch 起生效。原有實驗用 normalized pinball，`direction64` 用固定加權 pinball＋雙 CE，v2 用等市場方向 BA 與 pinball 護欄 | 同左 |
| retention | validation 最佳 5 個完整 checkpoints，加上 completion result | 同左 |
| validation/test | 完整有效 windows，不抽樣 | 同左 |
| 模型架構 | Kronos-base + LoRA + shared resampler + conditioner + alpha head | 完全相同 |
| 初始化 | 原始 pretrained base | 原始 pretrained base |
| 接續 Stage 1 checkpoint | 否 | 否 |

Stage 1 不是「最早 5% 時間」，也不是縮短 validation/test。兩個預設 config 的
`config.model_architecture_digest()` 必須相同；此 digest 包含 `h_start`／輸出維度，
最後不足一個 training batch 的 target set 會從同一集合開頭確定性補齊，且將補齊數量
寫入 training summary。Stage 2 重新從相同 pretrained revision 開始，以免把 Stage 1
script smoke 當成額外訓練資料。

### 7. Objective

主 objective 是 q10/q50/q90 pinball loss。每個 horizon 以 train-only robust scale
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
先用最多 1,024 筆 train-only windows 量測 `sQ,sAbs,sAlpha`，固定後訓練：
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
pinball 適合各 run 內的選模，不能只憑其跨組大小判定 A/B 優劣。跨組報告須同時
比較共同 holdout 上的 raw pinball、raw MAE、correlation、方向一致率及 coverage。

### 8. Comparator suite

所有 baseline 只能讀相同 cutoff-inclusive asset/benchmark contexts，並用相同 train
subset 與 validation split：

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

- 原有實驗：all-horizon mean normalized pinball；`direction64`：固定 reference 加權的 pinball＋雙 CE；`direction64-v2`：等市場方向 BA 與凍結 pinball 護欄，皆越低越好，不跨定義比較 selection loss。
- V3：population CE／Brier 對 train prior 的比率與 macro-F1，加相反方向／塌縮罰分及凍結 pinball 護欄；完整公式見第 13 節。

`direction64` 另報每個 horizon／市場／月份的兩個三分類 confusion matrix、accuracy、
balanced accuracy、macro-F1、majority accuracy、up precision／recall／訊號比例。
分類 precision 不等同回測獲利，三分類 accuracy 也不是舊 q50 二元方向一致率。
v2 另列 `balanced_decision` 的 confusion／accuracy／BA／macro-F1，而原 raw accuracy
與人口分布機率保持原義；不能改名或覆蓋。機率診斷包含 train-prior CE skill、20 個
confidence bins 與聯合 up 的門檻／precision／count；逐日 confusion 聚合硬上限
4,096 entries，超限報錯而非抽樣。這些機率未被假定為已校準。

每個 `h_start`–14 日 horizon 另報告：

- raw pinball 與 normalized pinball
- q50 MAE 與 normalized MAE
- q50 Pearson correlation
- q50 sign agreement
- q10–q90 interval coverage、width 與 coverage error

研究 diagnostics：

- date-level cross-sectional RankIC/IR
- equal-weight long/short net log return、Sharpe、max drawdown、turnover
- 五級 postprocess signal distribution
- market、asset type、provider、year slices

cross-sectional diagnostics 不是投資組合模型輸出，也不是主要 checkpoint metric。
transaction-cost 假設必須明列；單一 Sharpe 不可解讀為已證明可交易。

### 10. Test policy

正式 RunPod 訓練以完整 validation 選定最佳 checkpoint；訓練完成後，由同一 Pod
自動執行完整 test benchmark。若啟用區間校準，先固定該 checkpoint，僅用完整
validation 擬合凍結校準係數，再套用於 test；raw 與 calibrated 指標均保留。
整合實驗額外列出 `calibrated_static` 對照和 `calibrated_online` 逐日診斷：
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

1. 由完整 validation 選出：原有實驗看 `primary_5d/selection_score`；各版方向實驗看其版本定義的 `direction/selection_loss`。
2. 保存 trainable parameter union、optimizer、scheduler、RNG state 與 resolved config。
3. 綁定 run ID、dataset artifacts、model architecture、Kronos source/model/tokenizer
   revisions 與 bounded training-source digest。
4. 通過 artifact SHA-256/size 與 strict trainable-key restore。
5. 使用 `model_output_schema_version=5.0`；舊 checkpoint fail closed。

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

Stage 2：

- 使用相同 architecture digest、100% train split、相同 pretrained revisions。
- 完成所有 baseline 與完整模型的同協定 validation。
- raw JSON 有 per-horizon、aggregate、cross-sectional 與 subgroup metrics。
- 資料 provenance 可追溯，且 training code 無 provider/API import。

研究主張：

- Stage 1 只證明腳本可運作。
- 單一 seed Stage 2 只算 PoC。
- 要主張模型優於 baseline，至少需多 seed、報告 dispersion，並完成預先定義的 ablation。

### 13. 優先 ablation

當前方向比較使用相同資料組別的 `direction64-v2` 與 `direction64-v3`，各至少三個
training seeds（42、43、44）；原版保留為重現對照。這是整套路徑／prior／梯度／選模
方法的比較，不得把差異歸因於單一元件。CLI 使用 `train --experiment a-direction64-v3`
或 `b-direction64-v3`，搭配可重複的 `--seed`；runtime、GPU 型號及多 Pod 參數
維持原有規則，完整命令見 [README 訓練操作](../README.md#train-zh)。
原 `adaptive64`／`adaptive128` 的容量對照仍保留：其餘設定固定，只改 LoRA rank／alpha；
與 `lora64` 比較也是整套方法比較。`partial` 保留為次要研究線，不取代 Kronos／LoRA。

V3 的 `a_direction64_v3.yaml`／`b_direction64_v3.yaml` 使用 `market_ordinal_v3`：
下界條件機率 `a`、上界條件機率 `b` 組成 `[1-a,a*(1-b),a*b]`；以三分類 NLL 訓練，
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
參數及完整公式見 [README V3 模型](../README.md#training-zh)。

後續可在不改 output contract 與資料切分下，另作以下單因子消融：

1. gated benchmark conditioner vs asset-only（benchmark gate 固定為 0）。
2. Kronos LoRA vs frozen Kronos + trainable downstream modules。
3. Kronos vs compact PatchTST/DLinear，在相同 histories、horizons、loss 下比較。
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
evaluation, can a finance-K-line-pretrained Kronos-base with LoRA predict the
`h_start`-through-14 holding-day benchmark-relative alpha distribution of an individual stock
or ETF more accurately than simple rules, traditional ML, and compact causal DL
baselines, while identifying absolute-stock and alpha directions beyond cost hurdles?

Validation Sharpe or one backtest is not sufficient proof of tradable alpha,
and model output is not a complete trading rule. Technical baselines measure
incremental model value. Both direction presets separately inject ten explicitly defined historical
technical channels into the Kronos predictor without replacing it with baseline rules.

### 2. Fixed assumptions and scope

- Instrument-level, not portfolio-level, forecasting.
- Daily OHLCV only.
- US and Taiwan stocks/ETFs; no futures or options in this PoC.
- Signal after close `t`; entry at the next market session's raw open, without skipping missing bars.
- Entry day counts as holding day one.
- `h_start` is configurable as 1, 2, or 3; horizons run through fixed day 14.
- Label fixed to benchmark-relative adjusted execution log return.
- CAPM abnormal return reserved for a diagnostic/ablation.
- Pinball trains quantiles and an independent score head receives ranking loss; `direction64`/`direction64-v2` add absolute/alpha three-class CE while retaining five-level quantile post-processing.

### 3. Model hypothesis

The primary hypothesis is that a shared finance-pretrained encoder, historical-benchmark
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
See README and `configs/data_cleaning.json` for cleaning and annual weighting;
no full windows/labels are stored and genuine extreme returns remain.

| Item | Stage 1 | Stage 2 |
|---|---|---|
| Purpose | Validate the complete script/data/GPU/loss/checkpoint/validation path | Full-train PoC result |
| Train samples | Dynamic annual weighting; 5% of eligible count per epoch, capped at 500,000 | Dynamic annual weighting; presentations equal the full eligible count |
| Epoch limit | 2 | 5 |
| Validation cadence | 20%/40%/60%/80%/100% of every epoch | Same |
| Early stopping | Five consecutive non-improving validations of the configured monitor and two training intervals at the minimum LR; active from epoch 1. Original presets use normalized pinball; `direction64` uses fixed-weight pinball plus both CEs; v2 uses equal-market direction BA with a pinball guard | Same |
| Retention | Best five full validation-ranked checkpoints plus completion result | Same |
| Validation/test | Every eligible window, without subsampling | Same |
| Architecture | Kronos-base + LoRA + shared resampler + conditioner + alpha head | Identical |
| Initialization | Original pretrained base | Original pretrained base |
| Continue Stage 1 checkpoint | No | No |

Stage 1 is not the earliest 5% of time and does not shrink validation/test.
Both default configs require the same architecture digest. A short final training batch
is deterministically filled from the beginning of the same target set, and the
padding count is written to the training summary. Stage 2 restarts from the same
pretrained revisions so the Stage 1 smoke run is not hidden extra training.

### 7. Objective

The primary objective is q10/q50/q90 pinball loss, equally averaged after
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
At most 1,024 train-only windows establish frozen `sQ,sAbs,sAlpha`. Training uses
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

A/B estimate robust scales from their own training labels, so those scales may
differ. Normalized pinball supports within-run checkpoint selection, but its
cross-group magnitude alone cannot establish superiority. A/B reports must also
compare raw pinball, raw MAE, correlation, direction accuracy, and coverage on
the shared holdout population.

### 8. Comparator suite

All comparators read the same cutoff-inclusive asset/benchmark contexts and use
the same train subset and validation split:

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

Selection metric: original presets minimize all-horizon normalized pinball; `direction64`
minimizes fixed-reference pinball plus both CEs; `direction64-v2` uses equal-market direction
BA and a frozen pinball guard. Different selection-loss definitions cannot be ranked directly.
V3 uses population CE/Brier ratios against train priors, macro-F1, opposite-error/collapse penalties
and the frozen pinball guard; section 13 defines the complete formula.
Direction presets additionally report absolute/alpha
confusion matrices, accuracy, balanced accuracy, macro-F1, majority accuracy, up precision,
recall and signal fraction by horizon, market and month. Classification precision is not
realized trading profit, and three-class accuracy is not the old binary q50 sign agreement.
V2 separately reports `balanced_decision` confusion/accuracy/BA/macro-F1 while preserving
raw accuracy and population probabilities without renaming or overwriting them. Probability
diagnostics include train-prior CE skill, 20 confidence bins and joint-up thresholds/precision/counts.
Daily confusion aggregation has a hard 4,096-entry limit: overflow fails rather than subsamples.
These probabilities are not assumed calibrated.

For every horizon report raw/normalized pinball, q50 raw/normalized MAE, q50
Pearson correlation, q50 sign agreement, q10–q90 coverage, width, and coverage
error. Research diagnostics include date-level RankIC/IR, equal-weight
long/short return, Sharpe, drawdown, turnover, five-level signal distribution,
and slices by market, asset type, provider, and year.

Cross-sectional diagnostics are neither portfolio-model outputs nor checkpoint
selection metrics. Transaction costs must be explicit, and one Sharpe is not
proof of tradability.

### 10. Test policy

Production RunPod training selects the best checkpoint using full validation;
after training completes, the same Pod automatically runs the full test benchmark. When
interval calibration is enabled, the selected checkpoint is fixed, calibration
fits full validation only, and test retains both raw and calibrated metrics. Integrated
presets additionally report `calibrated_static` and a separate `calibrated_online` prequential
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

An acceptable checkpoint is validation-selected, stores the trainable union,
optimizer/scheduler/RNG/resolved config, binds run/data/model/source revisions,
passes artifact and strict-key verification, and declares
`model_output_schema_version=5.0`. Old checkpoints fail closed. Confirm success
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

Stage 2 uses the same architecture and pretrained revisions with 100% train,
evaluates all baselines under one protocol, persists per-horizon/aggregate/
cross-sectional/subgroup raw JSON, retains provenance, and keeps provider code
out of training.

Stage 1 proves script viability only. A single-seed Stage 2 is still a PoC.
Stronger claims require multiple seeds, dispersion, and predefined ablations.

### 13. Priority ablations

The current direction comparison is `direction64-v2` versus `direction64-v3` within one data
group, each with at least three training seeds (42, 43, 44). The original remains a reproduction
control. This compares a bundle of paths, priors, gradient management and selection, not the
causal effect of a single component. Use `train --experiment a-direction64-v3` or
`b-direction64-v3` with repeated `--seed`; runtime, GPU selection and multi-Pod options retain
their existing behavior. Complete commands are in [README training operations](../README.md#train-en).
The original `adaptive64`/`adaptive128` capacity control remains available with everything else
fixed and only LoRA rank/alpha changed; comparisons against `lora64` also evaluate a bundle.
`partial` remains secondary and does not replace the Kronos/LoRA backbone.

V3 uses `a_direction64_v3.yaml`/`b_direction64_v3.yaml` and `market_ordinal_v3`. Lower crossing
probability `a` and conditional upper crossing `b` form `[1-a,a*(1-b),a*b]`; three-class NLL
retains costs, absolute/alpha tasks, quantiles, ranking, LoRA, PCGrad and train-only references.
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
rank runs. See [README V3 model](../README.md#training-en) for parameters and the full formula.

Optional later single-factor ablations, without changing outputs or splits, include:

1. Gated benchmark conditioning versus asset-only (gate fixed to zero).
2. Kronos LoRA versus frozen Kronos plus trainable downstream modules.
3. Kronos versus compact PatchTST/DLinear with identical histories/horizons/loss.
4. Price-index benchmark input versus total-return-adjusted benchmark input.
5. CAPM abnormal-return diagnostic analysis without replacing the main label.

Each ablation needs a new run identity and must never overwrite a checkpoint
with different semantics.

### 14. Remote-validation boundary

Python environment, Poetry lock, lint, pytest, data preparation, model prefetch,
GPU training, and validation all run on RunPod. The local machine only edits
source/config, performs shell-level static checks, syncs S3, controls Pods, and
downloads artifacts. Local Python/PyTorch/CUDA state is not runtime evidence.
