# stock_forecasting：金融 OHLCV 時序預測

[繁體中文](#中文) · [English](#english)

## 中文

### 閱讀導覽

- [專案定位與授權](#overview-zh)、[模型架構與輸出](#model-zh)
- [目前兩個獨立任務](#signal-tasks-zh)
- [資料來源、連續性與流動性](#data-zh)、[固定時間切分](#splits-zh)
- [離線管線](#pipeline-zh)、[訓練、loss 與全量評估](#training-zh)
- [RunPod 操作手冊](#operations-zh)：[帳號與儲存](#setup-zh) → [設定與同步](#configure-zh)
  → [CPU prepare](#cpu-zh) → [baseline](#baseline-zh) → [任務／實驗與多 Pod 訓練](#train-zh)
  → [下載結果](#download-zh)
- [產物與推論](#artifacts-zh)、[歷史尺度表徵診斷](#probes-zh)、[CLI 參數速查](#cli-reference-zh)
- [驗收與結果解讀](#acceptance-zh)、[資料與模型參考](#references-zh)
- [歷史實驗附錄：容量、adaptive 與 V1／V2／V3](#historical-experiments-zh)

已有資料與所選任務 baseline 的使用者，直接從「任務／實驗與多 Pod 訓練」開始。
只有程式或 YAML 有修改才需要先同步；有 Pod 使用同一 volume 時不可覆寫共用程式。

<a id="overview-zh"></a>

### 專案定位與授權

本專案以美國與台灣普通股、ADR／TDR，以及經稽核且可映射的非槓桿股票型 ETF
日線 OHLCV 資料，微調金融領域預訓練的時序基礎模型。系統只處理數值時序：

- 輸入與輸出都是數值張量，不提供自然語言生成或事實重建功能。
- 不把外部 API 放進訓練迴圈。
- 預測從可設定的 `h_start`（1、2 或 3）到固定第 14 個持有交易日。
  分別訓練 alpha 穩健點估計與固定報酬帶的個股漲／平／跌模型。
  過往分位數及多任務研究設定見文末[歷史實驗附錄](#historical-experiments-zh)。
- 提供可供下游系統重用的數值 encoder 介面。

這是研究與能力驗證用的 PoC，不是投資建議、交易系統或可保證獲利的模型。
模型的職責是提供可檢驗的 alpha 與個股方向訊號；交易規則、成本、部位、風控與回測
由下游系統（例如 Hermes agent）結合模型輸出及傳統指標處理，不是本模型的預測目標。

#### 授權與版本

本專案自有程式碼及明示發布的模型新增部分，僅限自然人免費研究、學習、實驗、
非商業 hobby project，以及使用本人資金進行個人交易。**公司、法人、基金、量化
交易公司、任何組織用途、商業產品與付費服務均不獲授權**，也不得代管第三方資金。
完整條文見 [LICENSE](LICENSE) 與 [MODEL_LICENSE](MODEL_LICENSE)。這是
source-available 個人用途授權，不是 OSI 定義的開源授權。

Kronos 原始碼、預訓練權重與 tokenizer 保留原有 MIT 授權；本專案不限制上游獨立
授予的權利。詳見 [第三方聲明](THIRD_PARTY_NOTICES.md)。GitHub 可能將自訂授權顯示
為 Other；以授權全文為準。架構與報告快照見 [版本紀錄](RELEASES.md)。

<a id="model-zh"></a>

### 模型架構與數值輸出

<a id="signal-tasks-zh"></a>

#### 目前架構：一組實驗、兩個獨立模型

目前入口為 A 組 `a-signal64` 與 B 組 `b-signal64`。每次選一個資料組別及 seed，
建立一台 Pod 與一個父 run ID；在同一台 Pod 依序訓練 alpha 點估計及個股漲／平／跌兩個模型，
各自完成完整 validation／test。兩個子模型各有 Kronos／LoRA、optimizer、輸出頭與
checkpoint，不混合回歸與分類 loss，也不共享可訓練權重或梯度。
依序執行使兩個完整模型不必同時常駐 VRAM；不是合併 loss，也不是要求分開租兩台 Pod。
查詢、下載與接續使用父 run ID，仍可指定子 run ID 查閱單一模型。任務契約如下：

| 組內模型任務 | 輸出與語意 | 訓練與完整 validation 選模 |
| --- | --- | --- |
| `alpha_point` | `alpha_point`，形狀 `[batch, 15-h_start]`；benchmark-relative adjusted execution log return 的穩健點估計，不命名為 q50，也不宣稱是期望報酬 | train-only 尺度正規化 Smooth L1（Huber 形式），`point_loss_beta=1.0` |
| `absolute_direction` | `absolute_direction_logits`，形狀 `[batch, 15-h_start, 3]`；softmax 後為個股 `down, neutral, up` 機率 | 個股絕對報酬三分類 cross entropy |

獨立方向任務以同一個 next-open 至 horizon close 的個股 adjusted simple return 分類：
低於 −1% 為 down、介於 −1% 與 +1%（含端點）為 neutral、高於 +1% 為 up。
`neutral_return_threshold=0.01` 是固定報酬門檻，**不是交易成本**；不另預測 alpha 分類。
±1% 是初始候選，不宣稱最佳。Baseline 建置在 fitting 前掃描完整 train，按市場／持有期
稽核類別分布；缺少任一類、最小類別比例低於 1% 或最大類別比例高於 98% 都會阻止訓練。
主模型啟動前再核對這份 audit，不重掃行情、不用 test 標籤挑門檻。
兩個獨立任務沿用 A-LoRA64 的 Kronos、benchmark conditioning、尺度特徵與 LoRA-64，
不建立分位數／ranking 頭，也不做區間校準或 adaptive 校準。它們的
`direction.enabled=false` 只關閉舊雙方向多任務模組，不會關閉 `absolute_direction` 任務。
舊實驗仍保留原輸出，不把點估計複製成三個假分位數；獨立任務使用 `signal-1.0` schema。

`alpha_point > 0` 表示預測個股相對 benchmark 有正的超額 log return，並不表示個股一定
上漲，也不是「贏大盤的機率」。是否跑贏 benchmark 可直接由 alpha 點估計的正負判讀，
**不需要另一個 alpha 方向分類器**。回歸評估中的 `direction_agreement` 只是預測 alpha
與實際 alpha 的符號一致率，不是額外 head、分類 loss 或個股漲跌正確率。

兩個任務各自執行相同的輸入路徑：截至 `t` 的個股／benchmark OHLCV → 視窗標準化
→ Kronos／LoRA → resampler 與 benchmark conditioning → mean-pooled 表徵、market／horizon
embedding → 所選任務 head；20 維歷史尺度特徵與 benchmark latent 另直接進入 numerical branch。
點估計 head 每個 horizon 輸出一個值，分類 head 輸出三個 logits。
目前兩個 preset **不啟用**十技術通道、horizon-query attention、市場狀態 gating、train prior、
PCGrad、ranking 或表徵擾動正則化，也不使用交易成本特徵／標籤。

要使用目前成對架構，明確指定 `--experiment a-signal64` 或 `--experiment b-signal64`。
組內兩份 YAML 分別設定模型，不需手動建立兩個實驗。標準
`stage1_kronos_base_lora.yaml`／`stage2_kronos_base_lora.yaml` 仍是 `quantiles`；
省略 `--experiment` 時沿用 configure 的 active selection，**不會自動切成最新任務**。

生產設定使用 `NeoQuasar/Kronos-base` 與
`NeoQuasar/Kronos-Tokenizer-base`。預設 LoRA 實驗凍結 Kronos predictor 基礎權重，只在
`q_proj`、`k_proj`、`v_proj`、`out_proj`、`w1`、`w2`、`w3`
注入 LoRA；resampler、benchmark conditioner 與所選任務 head 可訓練。官方 source 固定為 commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`，必要的 source snapshot 與 MIT license
隨專案一併同步；preflight 與建模會逐檔驗證 SHA-256，不會在 RunPod 執行 Git。
模型與 tokenizer 權重另分別固定為
Hugging Face commits `2b554741eca47781b64468546e77fef3e85130e6` 與
`0e0117387f39004a9016484a186a908917e22426`，下載、離線 smoke test、config 與
checkpoint 都會綁定 revisions。

`QuantForecastModel.encode_ohlcv(...)` 顯式輸出：

- `last_hidden_state`
- `attention_mask`
- `latent_tokens`

下游系統可重用這些數值表示，接入其他預測、排序或風險分析模組，而不改變所選任務的輸出契約。

benchmark 的歷史同時透過動態 gated cross-attention 與 resampler latent 直接分支
影響輸出；尺度分支在 Kronos 視窗標準化前讀取歷史輸入。benchmark 在收盤 t
之後的資料只供離線 label construction 使用，不會出現在模型輸入。

#### 小型數值特徵分支

生產預設 `combined`，保留 Kronos-base 與 conditioner。數值分支與 LoRA rank／部分解凍
選項分開設定；20 個特徵的順序固定如下：

| 位置 | 特徵 | 定義 |
| --- | --- | --- |
| 1–3 | 個股、benchmark、相對報酬的 20 日波動 | 最近 20 個日 log return 的母體標準差；相對報酬為兩者之差 |
| 4–6 | 個股、benchmark、相對報酬的 60 日波動 | 同上，視窗為 60 個日 log return |
| 7–8 | 個股與 benchmark 的歷史價格 CV | 128-bar context 內 adjusted close 的母體標準差除以平均值 |
| 9–14 | 個股、benchmark、相對報酬的 5／20 日平均 log return | 只使用歷史日報酬 |
| 15–16 | 60 日相關係數與 beta | 個股與 benchmark 的歷史共變異數及波動 |
| 17 | 相對報酬 20 日 downside deviation | 負報酬平方平均的平方根 |
| 18–19 | 個股、benchmark 的當日／20 日平均成交量比 | `log((volume_t + 1) / (mean_volume_20 + 1))` |
| 20 | benchmark 60 日回撤 | 當下 close 相對歷史視窗最高 close 的 log ratio |

所有統計僅使用截至 `t` 的共同觀測資料，至少需要 61 個有效 bars；不年化、不使用未來
報酬，也不使用未來公司行動改寫輸入。前八個 feature 使用 `log(x + 1e-8)`，其餘使用
可保留正負號的 `asinh`，再以 train-only
calibration sample 的 median／IQR 標準化；IQR 下限 `1e-3`，標準化值裁切至 `[-10, 10]`。
完整 schema、train dataset 指紋與統計值保存在 checkpoint 的 `runtime_scale_features`，
推論只恢復統計，不以 validation、holdout 或當次輸入重新 fit。

目前獨立任務與歷史 head 共用的尺度 MLP 為 `20 → 32 → 16`；benchmark resampler tokens
先 mean pooling，再做 `512 → 16`。獨立任務將 head hidden 投影為 16 維，三者串接後經
`48 → 32 → output_dim` 作 numerical residual：點估計 `output_dim=1`，分類 `output_dim=3`。
點估計乘固定的 train-fitted robust scale；分類輸出 logits。不使用上下尾 gate、成本門檻或
區間寬度。新任務的 `explicit_output_scale` 沿用尺度特徵契約，不表示建立分位數區間。

Production config 啟用 `explicit_output_scale`，因此 feature mode 必須為 `scales`
或 `combined`。自訂實驗如需移除尺度路徑，
必須同步關閉明確尺度控制。切換 mode 不改變 dataset namespace，但必須建立新的 training run。
是否改善預測必須由新訓練的評估結果證明，加入尺度資訊本身不等於已提升 alpha 能力。

#### Backbone 選擇

目前整合的是金融 OHLCV 預訓練的 Kronos-base，原因是輸入領域相符、已有可固定 revision
的公開權重與 tokenizer，並可在單 GPU 上進行 LoRA／部分解凍實驗。來源見
[Kronos 官方程式庫](https://github.com/shiyu-coder/Kronos) 與
[論文](https://arxiv.org/abs/2508.02739)。這不是它優於所有時序模型的結論；
更換 backbone 必須控制資料、context、head 與評估集合，不能只比不同專案的總分。

<a id="data-zh"></a>

### 資料來源與樣本規則

#### 資料來源與可選資料集

正常 RunPod workflow 透過 `bash scripts/runpod_workflow.sh configure` 選擇
profile；`FIN_TS_DATASET_PROFILE` 是腳本驗證 selection 後傳入 Pod 的內部值：

| profile | 實際來源 | 狀態 | 適用情境 |
| --- | --- | --- | --- |
| `tw_only` | TWSE 官方 + TPEx 官方 | 可用 | 零美股 API 費用的研究路徑 |
| `us_only_eodhd` | EODHD 美國股票/ETF | 可用 | 先驗證美股能力 |
| `us_tw_eodhd` | EODHD + TWSE + TPEx | 預設 PoC | 美台跨市場完整 PoC |
| `us_tw_massive` | Massive + TWSE + TPEx | 僅保留型別化介面 | 取得適合授權後再實作 |

EODHD 路徑預設可發現 active 與 delisted 美國股票/ETF，減少只保留存活標的造成的 survivorship bias。若費用或呼叫額度有限，可在 `configure` 使用 `--universe explicit` 搭配 `--stocks`、`--etfs`，或在 all 模式使用 `--symbol-limit` 縮小 universe。輸出 manifest 會列出 profile、實際 provider、market、symbol、asset type、日期範圍與每個 split 的樣本數。

`--universe all` 的 discovery 是準備當下 EODHD 回傳的 active/delisted 清單，並非
每個歷史交易日各自重建的 point-in-time constituents。以 `--start 2016-01-01 --end 2026-06-01` 為例，日期契約是 `[2016-01-01, 2026-06-01)`：區間中途上市的
商品只會從 provider 可取得的第一個交易日開始，區間中途下市的商品只會保留到最後
可取得日；同時在區間內上市又下市的商品，只要 EODHD delisted discovery 有回傳且
帳戶有權限，就會納入。`explicit` 只處理明列的 ticker；使用 `--symbol-limit` 時則只
處理限制後的子集。raw row 的 `is_active` 是 discovery 當下狀態，不是逐日上市狀態。

EODHD 官方另註明：2018 年後下市的商品可取得 EOD、fundamentals、dividends 與
splits；2018 年前下市的商品只保證 EOD。因此這些舊下市商品的 EOD rows 仍可能進入
raw／training windows，但 split-adjusted volume 的輔助資料覆蓋不能視為完整。下載
manifest 會列出 `delisted_pre_2018_auxiliary_coverage_warning` 的數量與 symbols，避免
把「有價格歷史」誤寫成「所有公司行動資料都完整」。

EODHD 是 PoC 資料，不應被描述成交易所級真實行情。跨 provider 的 adjusted price、公司行動、delisted history、時區與資料修訂可能不同；正式比較前必須先做重疊標的抽樣對帳。

#### 交易時間、benchmark 與調整資料

每筆樣本在交易日 `t` 收盤後產生訊號；下一個**市場交易日**的 regular-session raw
open 進場，該日算第 1 個持有交易日，持有 `h` 日時在第 `h` 個市場交易日的
raw close 出場。Alpha label 是商品與 benchmark 在完全相同 entry/exit timestamps 的
total-return log return 差，`h ∈ {h_start,…,14}`，其中 `h_start ∈ {1,2,3}`。
`absolute_direction` 使用同一 entry／exit 的個股 total-return log return，不減 benchmark；
以 `log1p(-0.01)`／`log1p(0.01)` 對應固定 ±1% simple-return 三分類門檻。
個股缺 bar 或零成交量時不能將 entry／exit 延後到下一個有資料的日期。

預設 benchmark policy：

- 美國普通股、ADR 與白名單股票型 ETF：`VTI.US`。
- TWSE 普通股、TDR 與白名單股票型 ETF：`TAIEX.TW`，其 adjusted anchor 使用官方發行量加權股價報酬指數。
- TPEx 普通股：`TPEX.TWO`，其 adjusted anchor 使用櫃買報酬指數。
- 只有經稽核白名單內、可映射到既定 benchmark 的非槓桿股票型 ETF 才進入訓練；
  槓桿、反向、債券、商品、波動率與未稽核 ETF 一律 fail closed。
  `benchmark_mapping_path` 只能改變白名單 ETF 的 benchmark，不能擴張訓練 universe。

`VTI.US` 是美國商品建立 benchmark-relative label 與 benchmark context 的必要資料
依賴，不是 `--symbol-limit` 的一般候選商品，也不會成為自己的訓練 target
（`self_benchmark` 會排除它）。因此限制是在 ETF 與 stock 各自選完 N 檔後才確認
VTI：若已選到便不重複，否則額外補入。這讓 N 個 ETF target candidates 不會被
benchmark 占掉一席；raw universe 最多是 `N ETF + N stock + 1 VTI`，但實際可訓練
target 數仍可能因資料長度、benchmark mapping 或品質 gate 而更少。

raw O/H/L/C 永久保留；台灣 `volume` 也是官方 raw field。EODHD 官方定義的
`volume` 已做 split adjustment，因此管線用完整 Historical Splits response 反推出
當時的未調整 `volume`，並把 vendor 值保留為 `split_adjusted_volume`，不會再乘一次
split factor。模型視窗把 vendor/官方 total-return factor 正規化到
`cutoff_at`，再套用到歷史 O/H/L/C，因此收盤後推論不會因未來公司行動而回寫輸入；
volume 只依 split/share change 調整，不用現金股利調整。EODHD 保留
`adjusted_close`；所有日期範圍都對每個 symbol 使用 Historical Splits API。官方將
這個 endpoint 列入 EOD Historical Data — All World 且每個 request 為 1 API call；
管線不使用另屬 Calendar 產品的 `calendar/splits`。每個 symbol 因此需要一個 EOD
history request 加一個 split-history request，兩者都可 cache／續傳，也會列入資訊性
request 估算；估算值不會阻止完整資料集執行。provider 對 2018 年前下市商品的上述
輔助覆蓋例外則依前述 warning 顯式保留。台股使用 TWSE/TPEx 官方除權息資料與官方
報酬指數，並從既有月度官方 benchmark rows 取得實際交易日，不會把一般週一至週五
一律當成開市日。
這可避免股票分割或除權息造成的人為跳空，同時維持下一日 raw open 的可交易 entry
語意。

#### 資料完整性、連續性與最低流動性

主模型與 baseline 的 train、validation、test 使用同一份
`configs/data_cleaning.json` 規則，推論沿用其中所有只需歷史資料的檢查：

- **日曆**：使用離線 `exchange-calendars==4.13.2`，US 對應 XNYS，TWSE／TPEx
  對應 XTAI。不把週一至週五一律當交易日，也不把個股與 benchmark 同時缺資料
  誤當休市。日曆版本固定；臨時休市等差異須核對交易所公告，不自行補價格。
  `calendar_overrides` 已補列 2016–2018 年八個週六交易日，並排除套件漏列的
  2022-02-04、2023-01-18、2024-10-31 休市日。依據包括
  [櫃買中心 2016 年日曆](https://wwwov.tpex.org.tw/storage/zh-tw/web/bulletin/trading_date/trading_date_105.htm)、
  [2017 年日曆](https://wwwov.tpex.org.tw/storage/zh-tw/web/bulletin/trading_date/trading_date_106.htm)、
  [證交所 2018 年日曆](https://www.twse.com.tw/staticFiles/product/publication/0001002657.pdf)、
  [2022 年春節公告](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080817d22b9cb017e3336c4e203e3.pdf)、
  [櫃買中心 2023 年日曆副本](https://www.honsec.com.tw/uploads/images/112%E5%B9%B4%E6%9C%89%E5%83%B9%E8%AD%89%E5%88%B8%E6%AB%83%E6%AA%AF%E8%B2%B7%E8%B3%A3%E5%B8%82%E5%A0%B4%E9%96%8B%EF%BC%88%E4%BC%91%EF%BC%89%E5%B8%82%E6%97%A5%E6%9C%9F%E8%A1%A8.pdf)，
  及 [2024-10-31 休市公告](https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e30192dd5179bc0327.pdf)。
- **完整序列**：128 個 input bars 必須恰好是截至 `t` 的 128 個連續市場交易日；
  output 必須包含接續的 14 個市場交易日，即使 `h_start` 為 2 或 3 也相同。
  個股與 benchmark 都須完整對齊。缺一天就排除該 window；不 forward-fill、
  不用更早月份／年份湊滿 128 筆、不壓縮停牌日、不改 label horizon。
- **有效行情**：整段 input／output 的 OHLC 與 adjusted close 必須有限且大於零，
  high／low 須符合 OHLC 邊界。個股每日 volume 必須有限且大於零；交易型 ETF
  benchmark 亦同。非交易型指數 benchmark 允許零 volume，但價格不能缺漏。
- **最低流動性**：截至 `t` 的最近 60 個市場交易日，以當時 raw close × raw volume
  估計每日成交金額，其中位數須達 USD 1,000,000（美股）或 TWD 10,000,000（台股）。
  不用未來成交量決定當日流動性，不做外匯 API 查詢。開發用 input 少於 60 bars
  時使用整個 input 長度；正式 128-bar 設定固定使用 60 日。
  這是可調整的工程門檻，不是已證明最佳的投資門檻；須一起檢視排除率。
- **保留極端報酬**：真實、有限且通過上述檢查的大漲跌不刪除、不截尾；
  絕對 adjusted log return 超過 0.5 只計入診斷。既有 prepared ranges 的舊極端值
  排除旗標不再決定 runtime 樣本。資料不足或無效數值與真實市場極端值分開處理。
- **時間邊界**：train cutoff 在 2025-06-01 之前；validation 為
  `[2025-06-01, 2025-12-01)`；test 為 `[2025-12-01, 2026-06-01)`。
  每筆最晚 label 日期必須嚴格早於所屬 split 的右界，不能使用下一 split 的 ground truth。
- **共同評估來源**：A/B 的 training 各讀其設定的歷史跨度；validation／test 均讀
  同 profile、universe、revision、結束日之 **2016-01-01 起點的既有 bar-store**。
  相同日期不足以保證不同下載快照相同，因此不各自讀 A/B 快照，也不取兩者交集。
  共用來源不存在時明確停止，不自動下載或回退到另一份資料。

執行時從不可變 bar-store 平行建立小型 `prepared/sample-universes/` cutoff-range
索引；依 CPU quota、可用記憶體與每 worker 預算限制並行度與 pending tasks。
`FIN_TS_CLEANING_WORKERS` 可設定 worker 上限（預設 8），中斷後重用完成的 bucket。
不複製完整 windows／labels，不改 raw、CPU prepare 的 manifest 或既有結果。
`universe.json` 保存每 split／market 的候選數、有效數、各排除原因與保留極端值數；
逐商品稽核保存在 `parts/*.json`。未來無法成交的樣本是「無有效 ground truth」，
不能將其排除後的評分解讀成無停牌／下市風險的可交易回測。

改動清理規則會改變實際訓練樣本，必須為受影響的 A/B 資料組別與 prediction task
各建置對應 baseline；舊模型與
結果保留，但不能當成新集合的公平基準。單純切換 LoRA／解凍容量不重建 baseline。
此流程重用既有行情，不要求重跑 CPU prepare 或重新下載資料。

<a id="splits-zh"></a>

#### 固定 Train／Validation／Holdout 時間切分

生產 Stage 1／2 固定採用以下 exclusive 日期切分，不依資料量或最早日期重新算比例：

| split | 預測日 `t` 範圍 | 用途 |
| --- | --- | --- |
| Train | 起始日（預設 `2016-01-01`）≤ `t` < `2025-06-01` | 訓練、label scale 與 feature scale 校準 |
| Validation | `2025-06-01` ≤ `t` < `2025-12-01` | 最佳 checkpoint 排名、early stopping |
| Holdout / test | `2025-12-01` ≤ `t` < `2026-06-01` | 訓練結束後的最終模型及 baseline 評估 |

每筆樣本所有 horizon 的實際 `label.end_at` 都必須嚴格小於所屬 split 的結束日，包括
holdout 的 `2026-06-01`。因此每段末端約 14 個交易日不會成為完整 horizon 的評分起點；
固定日期模式不再另外疊加舊的 20 日 purge／14 日 embargo。Validation／holdout 的歷史
輸入可以向前跨過其起點，因為預測時已可取得那些歷史資料；未來 label 不可跨界。
`2026-06-01` 之後的日期不會加入這三個 split，6–8 月保留作後續回測。

CPU preparation 與本機 readiness 會核對 prepared manifest 的日期契約：每個市場的
validation、holdout 各至少須有 **80 個不同預測日期**，記錄在
`split_audit.dates_by_market`。這是 prepared 資料的檢查；runtime 清理後的有效集合
另以 `sample-universes` 稽核和實際評估結果為準，不能把 prepare 的筆數當成清理後筆數。80 日是資料覆蓋的工程門檻，不是統計
顯著性或跨所有行情的保證；大量股票樣本也不能當成同樣多的獨立時間樣本。

日期與持久化 preparation 契約屬於 dataset identity；若仍使用歷史比例切分，必須建立
符合固定日期的新 namespace，不能只修改 ready marker。已完成本表日期切分的資料，
切換 Stage、A/B 或容量實驗不需重跑 CPU prepare。Runtime 清理規則改變時，重建的是
小型樣本索引及對應 baseline，不是行情或 bar store。

<a id="pipeline-zh"></a>

### 離線資料管線

資料取得與模型訓練是兩個互斥階段：

```text
外部 API
  │
  ▼
不可變 raw JSON cache
  │
  ▼
canonical daily OHLCV Parquet + download-manifest.json
  │
  ▼
可續傳 symbol bar store + prepared 候選 cutoff ranges
  │
  ▼
bar-store/index/ranges + dataset-manifest.json
  │
  ▼
runtime 連續性／流動性清理索引
  │
  ▼
Lazy DataLoader 動態建立 context/label（完全離線）
  │
  ▼
Stage 1 / Stage 2 訓練
```

raw OHLCV 以不可變的壓縮 Parquet 分批寫入。CPU preparation 不再展開每一個
128-bar window，也不預先落地 label；它以 128 個 hash buckets 建立按 symbol
排列、每個 symbol 一個 Parquet row group 的壓縮 bar store，並只保存小型
`symbol-index.parquet` 與連續有效 cutoff ranges。每個 scan、compaction、quality
與 split bucket 都有原子 checkpoint；Pod 到達 max runtime 時會以
`waiting_for_preparation` 結束，下一個相同 dataset namespace 的 CPU Pod 從尚未完成的
scan partition 或 bucket 接續，已完成項目直接跳過。只有 `_SUCCESS.json`
發布後才回收 `.work` 暫存分區。

資料以 dataset request 自動映射到 `/runpod-volume/datasets/<dataset-request-sha256>/`，
不由使用者手填 DATA_ROOT。CPU prepare 不展開完整 windows；runtime 再依清理規則建立有效索引。

<details>
<summary>下載快取、配額與準備流程的完整性規則</summary>

資料下載器具備：

- provider-specific QPS throttle。
- EODHD、TWSE、TPEx 以獨立迴圈平行抓取；retryable request 使用指數退避。
- 完整計畫的預估 HTTP requests 只作資訊與容量規劃；台灣最終 plan 使用官方
  benchmark sessions，另保留 pre-calendar weekday upper bound；兩者都不作
  dataset admission gate。
- `max_api_calls` 只限制單次 CPU attempt 的 EODHD network attempts（含 retry）；
  TWSE／TPEx 不受 request-count 上限限制。EODHD、TWSE 與 TPEx 都以同一個預設
  1 分鐘的 `--maxBackoff` 作為各自的退避退出邊界。cache hit 不扣額度，未完成時
  保留進度並由下一個 Pod 續接。
- `--max-api-calls`、`--eodhd-qps`、`--taiwan-qps` 與 `--maxBackoff` 是每次 CPU Pod
  launch 的 acquisition policy，由 `runpod_workflow.sh cpu prepare` 設定；它們不屬於
  immutable dataset selection，也不會改變 dataset request identity。
- 任一 provider 先退出都不會取消其他 provider；每個成功完成的 provider 會先以
  dataset request、training security scope 與 materialization revision 綁定的
  SHA-256 checkpoint 原子發布。只有全部 provider 迴圈退出後，流程才發布續傳狀態
  或依固定順序合併已驗證的 provider checkpoints。
- Dataset contract 改版或日期改變會建立新的 immutable namespace；新 namespace
  可唯讀命中其他 dataset namespace 中相同 cache revision 與 request identity 的 raw JSON cache，
  但新的 Parquet、manifest 與 progress 只會寫入自己的 namespace。
- QPS 與 `max_api_calls` 都不是 provider 的每日／每週 quota，也不代表 EODHD
  不同 endpoint 的計費 call units。
- 暫時性 provider 錯誤、429、單次 request budget 或 acquisition time budget
  耗盡時，保留成功的 raw responses 與已完成的 provider materialization checkpoints、
  寫入 `download-progress.json`，並允許下一個 CPU Pod 直接重用已完成 provider，
  只對未完成 provider 補未快取的 requests 並重新 materialize。
- CPU workflow 預設保留 max runtime 的 25% 給 canonical data cleaning 與 symbol
  bar-store/index 建置（最多 2 小時，亦可用 `--prepareReserve` 明確設定）；下載完成的 raw Parquet、request log
  與 download manifest 會先發布成 durable `downloaded` checkpoint。後續 Pod 可完全
  跳過 API，從 durable bucket checkpoints 接續；只有 bar store、cutoff ranges 與
  readiness 都驗證通過才成為 `ready`。
- API token 不進 cache key、request log 或 manifest。
- raw cache 與直接執行的下載／準備 CLI 拒絕靜默覆寫。
- Parquet 與 manifest 的 SHA-256、row count 與 provenance 綁定。

訓練器只接受 `_SUCCESS.json`、完整 shard/index/range 契約與 `state=ready` 的 dataset
manifest；CPU readiness 與訓練 preflight 會逐 shard 核對 size 與 SHA-256，而不是只驗證
小型 index。模型訓練、評估及推論程式不呼叫 EODHD、TWSE、TPEx 或 Massive。

</details>

<a id="training-zh"></a>

### 訓練與評估協定

#### Stage 1／Stage 2 與動態 sampling

| 項目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 用途 | 有界流程與模型 smoke run | 全訓練歷史的正式實驗 |
| 每 epoch 樣本呈現預算 | 有效 train 數的 5%，最多 500,000 | 有效 train 總數 |
| Training sampling | 年度衰減動態取樣；不是固定 5% 商品—日期集合 | 年度衰減動態取樣；不保證每個 window 恰好一次 |
| Epoch 上限 | 2 | 5 |
| Validation cadence | 每 epoch 的 20%／40%／60%／80%／100%，共 5 次完整 validation | 同左 |
| Early stopping | 設定的 validation monitor 連續 5 次未改善，且完成最低 LR 的兩個間隔；第 1 epoch 起生效。各任務／版本的 monitor 定義見下表 | 同左 |
| 保存結果 | validation 最佳 5 個完整 checkpoints，加上完成時的精簡權重 | 同左 |
| Validation / test | 完整有效集合，不抽樣、不 padding、不 drop_last | 同左 |
| 底層 Stage 設定（歷史分位數預設） | `configs/stage1_kronos_base_lora.yaml` | `configs/stage2_kronos_base_lora.yaml` |
| 初始化 | 原始 pretrained base | 原始 pretrained base；不接續 Stage 1 權重 |

訓練 DataLoader 以 bounded sampler 狀態從有效 cutoff ranges 動態取樣，按需讀取一個 symbol
row group、建立 128-bar asset/benchmark context，並在記憶體中計算從 `h_start` 到第
14 個持有交易日的 alpha 及個股絕對報酬，再按所選任務建立 label。主模型不預先落盤
逐-window 或逐-label 資料集；baseline 的有界 input cache 另見下節。Stage 1
每 epoch 呈現 `min(valid train cutoffs × 5%, 500,000)` 個樣本；Stage 2 的呈現次數
等於全部 valid train cutoffs 數。年度衰減會使較新的 windows 重複呈現、較舊 windows
未必在同一 epoch 出現；seed 與 epoch 決定可重現的順序與接續位置。兩者都用固定
大小 batch，最後不足一個 batch 時確定性補齊並記錄數量。年度／市場分組索引採 mmap，
不在 RAM 展開所有 windows。這個 out-of-core 設計可直接處理
完整長歷史資料，不需要把全部 bars 或所有可能 window 載入 RAM。

Stage 是訓練資料預算與流程層級，不是模型版本。目前兩個獨立任務使用各自的 Stage 2
實驗 YAML，必須在[主模型訓練](#train-zh)明確選擇；上列底層 Stage YAML 仍是歷史分位數預設，
兩者的 `model_architecture_digest()` 相同。歷史容量設定見[附錄](#legacy-presets-zh)。
同一 run 的中斷恢復見[接續訓練](#resume-zh)，不是從 Stage 1 微調到 Stage 2。

Production Stage 1/2 不接受 `max_steps`、固定 step validation cadence 或獨立的固定
step checkpoint cadence。optimizer budget 由每 epoch 的呈現次數、batch size、gradient
accumulation 與 epoch 數推導；每次 epoch-relative validation 都參與最佳 5 個 checkpoint
排名。正常跑完或 early stopping 都會另外原子發布唯一的 `completion-result/`，其中保存
當下的可訓練權重、resolved config、停止原因、實際步數／樣本數與最後 validation metrics，
但不重複保存已無續傳需求的 optimizer/scheduler state。

#### 訓練目標與學習率

所有 monitor 都以完整 validation 計算，越低越好；同一鍵名不代表同一分數定義。

| 任務／實驗系列 | Checkpoint monitor | 分數定義 |
| --- | --- | --- |
| 目前 `alpha_point` | `primary_5d/selection_score` | 全 horizon 平均 normalized Smooth L1 |
| 目前 `absolute_direction` | `primary_5d/selection_score` | 全 horizon 平均個股三分類 CE |

##### 目前獨立點估計與個股方向目標

獨立 `alpha_point` 只最小化 `mean(SmoothL1((prediction − alpha) / train_scale, beta=1))`；
獨立 `absolute_direction` 只最小化固定三分類 CE。兩者都對有效樣本與全部持有期等權平均，
不混合 pinball、排序或舊方向 regularizer。`primary_5d/selection_score` 仍為 `min`
監控鍵，語意分別是完整 validation 的 normalized Smooth L1 與 CE，而不是只看第 5 日。
Training 分別記錄 `train/normalized_smooth_l1` 或 `train/cross_entropy`，並保留
`train/loss`、`validation/loss`；不得把不同任務的 selection score 當成同一指標排名。
目前兩種 preset 均使用 validation-plateau、task／LoRA LR `3e-5`／`5e-6`、
dropout `0.05` 與 weight decay `0.01`，不沿用 adaptive 的 sample-plateau 或市場 loss 權重。
兩者的訓練排程、資料取樣與數值尺度各隨自己的 checkpoint 保存及接續。

##### Train-only 尺度

Stage 1／2 的 label robust scales 都從完整 **train partition** 的同一個確定性最多
50,000 筆樣本校準，seed 固定為 59，不隨 Stage 1 的 5% 訓練抽樣縮小。
尺度特徵的 median／IQR 也使用相同 sample-count／seed 契約，統計快取與 dataset
manifest SHA 綁定；兩種校準都不讀 validation／holdout。

##### Validation-driven plateau 排程

目前兩個獨立任務皆採 warmup 後的 validation-driven plateau 排程：兩次未改善即將 LR 乘 0.3，最低為
初始 LR 的 0.09；到達最低 LR 後，必須再完成兩次 validation 間隔的訓練，才允許 early stop。
checkpoint 保存 plateau 狀態，resume／更換 batch plan 不會把已降低的 LR 重設。

#### 全量 validation／test 與比較方式

最終 `validation stage` 是既有工作流程名稱；新固定日期模式實際評估 **holdout/test**。
完整模型必須重新推論，不能沿用 checkpoint validation snapshot。所有例行 validation
與最終 test 都使用完整 split，`evaluation_max_samples` 與 `baseline_max_samples_per_split`
固定為 `null`；沒有 20,000 筆上限。指標按完整資料的樣本數加權，預測分批移出 GPU 並使用
磁碟暫存，逐塊彙總。同一資料組別的主模型與 baseline 使用相同 train-calibrated label scales，並驗證完整有序
symbol/date SHA-256 相同。A/B 的 train-only scales 可能不同，跨組比較 normalized Smooth L1 或 normalized pinball
須同時核對尺度；評估集合則必須完全相同，不能只比日期。

完整評估依 `symbol → cutoff` 的固定順序逐檔列舉所有有效 windows；runtime 清理後的
`prepared/sample-universes/<identity>/cutoff-ranges.parquet` 儲存精確可用區間，不是估計筆數；
不能以 CPU prepare 舊候選範圍的筆數代替實際評估集合。Dataset 只保存這些
小型索引與有上限的商品快取，在 DataLoader 取 batch 時才產生 context／label，
不會預先展開巨大的 window 資料集。最後不足一個 batch 的資料照常評估，不重複補齊。
舊設定中的 `evaluation_max_samples` 若仍有數值，會明確警告並忽略，不能切回抽樣。
這項固定完整遍歷只適用於 validation／testing；training 保留動態 sampling、
各 epoch 的隨機順序與同日同市場分組，並在讀取 batch 時動態產生 windows。

最終 testing 會在當次 GPU 重新測量純 inference 的 batch size，不做 backward，
也不沿用另一張 GPU 的 training batch plan。預取深度依 inference／training 實測速度、
RAM、容器 shared memory、同時存活的 worker pools 與 pinned-memory 複本限制；
使用多 worker、pinned memory 與非同步 GPU transfer。結果的 `execution` 記錄實際筆數、
batch、worker、prefetch、吞吐量與 GPU 峰值記憶體。這些是執行效能資料，不是預測效能。

報告包含逐月、逐市場及各 horizon 指標，以及完整模型減去各 baseline 的逐日平均任務 loss 差：
點估計用 normalized Smooth L1，純分類用 CE，歷史分位數用 normalized pinball，皆對全 horizon 等權。
95% 區間以預測日期為單位，使用 14 日 circular moving-block
bootstrap（1,000 次、seed 42），不把同日股票各自當獨立樣本；此區間未作多重比較校正。
這是配對 loss 差的區間，不是 RankIC 或 balanced accuracy 的信賴區間。
Holdout 排名僅描述結果，不可再用來挑 checkpoint、反覆調參或宣稱已涵蓋所有未來行情。

各任務的結果範圍如下，不能用另一種任務的欄位代替：

| 任務 | 主要品質指標 | 不輸出的指標／訊號 |
| --- | --- | --- |
| `alpha_point` | normalized Smooth L1、MAE／RMSE／normalized MAE、bias、correlation 及有效狀態、alpha 符號一致率、逐市場／horizon 的 date-level RankIC | 分位數、coverage、個股漲跌分類機率、long-short 績效、Sharpe、五級交易訊號 |
| `absolute_direction` | CE、Brier、accuracy、balanced accuracy 與缺類狀態、macro-F1、opposite error、ECE、majority accuracy、混淆矩陣、各類 support／precision／recall／預測占比 | alpha 分類、報酬點估計、RankIC、分位數與組合交易績效 |


兩個獨立任務與 baseline 的逐模型比較在 `comparison.rows`。歷史任務欄位見[附錄](#legacy-results-zh)。
RankIC 使用標準 Spearman average ties，
方法標記為 `rank_ic_method=spearman_average_ties_v1`；同日預測或 target 為常數時，
該日沒有有效 correlation，不因排序時的列順序產生 RankIC。`rank_ic_status`、
`rank_ic_evaluated_dates`、`rank_ic_ir_status` 區分有效值與保留的數值占位值。
沒有此方法標記的歷史結果未自動重算，不能當成相同定義的 ties 結果。

#### Baseline 前置建置與快取

Baseline 是主模型之前的獨立流程。Rule、GBDT、GRU、DLinear、PatchTST 都使用完整
train／validation／test，並按所選 prediction task 建置相應結果；舊分位數或雙成本方向
結果不能當成目前兩個獨立任務的 baseline。操作命令見[baseline 建置](#baseline-zh)。

##### 目前兩個獨立任務

| 任務 | 神經 baseline 訓練 | GBDT 訓練 | 完整 validation 選模與最終 test |
| --- | --- | --- | --- |
| `alpha_point` | train-scale normalized Smooth L1，與主模型相同 beta | `absolute_error`（MAE），不是 Smooth L1 | 統一 evaluator 的 normalized Smooth L1 選模；完整回歸／符號／RankIC 指標 |
| `absolute_direction` | 個股絕對方向 CE，不訓練 alpha 分類 | 每個 horizon 的 log loss classifier | 相同 ±1% 含端點 neutral band；CE 選模與完整機率／類別指標 |

固定規則產生 task-aligned point／hard class，不訓練分類器；分類機率評分使用明列
`0/-30` logit floor 的確定性預測，不宣稱是學到或校準過的機率。兩個新任務都不訓練
分位數、ranking 或雙成本方向頭，也不使用舊方向 loss reference。比較結果在 `comparison.rows`。

##### 共同訓練、儲存與重用規則

神經模型每 epoch 做五次完整 validation、最多五個 epochs；GBDT 每八輪新增樹後
做完整 validation、最多 200 輪，停用 sklearn 內部 validation 抽樣。兩者均採五次未改善
patience、保存最佳 validation 模型；神經模型另用 validation-plateau 與最低 LR 觀察規則。
GBDT 沒有神經 optimizer 的 LR 排程，固定規則不適用 early stopping。
Baseline 神經模型以 blockwise permutation 每 epoch 遍歷完整 train 一次；GBDT 也不套
主模型的年度衰減 sample weights。主模型的年度加權動態 sampling 與此不同，因此相同
資料全集、完整評估及 patience 不等於相同訓練權重、FLOPs 或訓練時間；比較須揭露這些差異。

完成的權重／規則參數、最佳 validation 指標、完整 test 預測及指標保存在
`/runpod-volume/baselines/<baseline-id>/`，全部成功後才發布 `complete.json`。
主模型啟動前檢查對應 baseline，testing 直接讀快取，**不會再次訓練或推論 baseline**。
身分包含資料期間／內容、清理與 universe／horizon 語意、prediction task、beta／neutral
threshold 及 baseline 自身數值參數／實作；不綁主模型 LoRA 容量、training seed、
README 或部署腳本。修改 task 相關數值設定則必須使用相符結果，不能因日期相同就重用。
若完整 GBDT 無法放進記憶體預算，流程拒絕執行，不會退回抽樣。

Validation/test 共用 `inputs/<split>/metadata.npy`（membership）與 `targets.npy`，
個股方向另共用 `absolute_targets.npy`。`evaluation_data` 引用共同資料；每個模型目錄
只保存自己的預測、權重與指標，不重複保存 `membership.npy`／`targets.npy`。
獨立任務的 `predictions.npy`：點估計為 `[N,H]`；分類為 `[N,H,3]`，順序 down／neutral／up。
分類分數在神經模型為 raw logits、GBDT 為 `log(max(probability, 1e-30))`、
規則為 `0/-30`；統一由 softmax 轉成評分機率，不把這三種來源說成同一個學習式分類頭。
收尾串流核對內容後原子更新引用、移除副本；
若中斷，重跑只完成儲存收尾，不重訓。這項整理不改資料、標籤或 baseline 數值身分。
主模型最終 `validation-benchmark.json` 使用結果 schema `6.0`，明列
`selection_split=validation`、`evaluation_split=test`，配對檢查完成後才發布 `test_unlocked=true`。
這些不是 baseline `complete.json` 的欄位；baseline 完成由其 `state=complete`、模型與產物驗證判定。
相同評估 schema 也須核對任務、有效樣本及尺度，
不能將不同任務的舊結果冒充完成。不同主模型 seed 可重用匹配結果，不重做 CPU prepare 或下載行情。

<a id="operations-zh"></a>

### RunPod 完整操作手冊

#### 執行位置與安全邊界

本機需要 Bash、系統 Python 3、AWS CLI 與 curl；台灣 relay 部署另需 Google Cloud CLI。
本機只編輯程式、操作 credentials／selection、上下載產物與監控 Pod，不載入模型。
系統 Python 的 stdlib 控制／契約測試、shell 語法與獨立靜態檢查，不需要本機 ML 環境。

Python 套件管理沿用 Poetry，環境固定在雲端專案根目錄
`/runpod-volume/stock_forecasting/.venv`。Dependency resolution、canonical lockfile、
完整 pytest、模型／CUDA smoke test、資料準備及訓練均在 RunPod 執行。
不在本機建立或檢查 ML environment，也不執行 `poetry install`、`poetry lock`、
`uv sync` 或 `uv lock`；本機殘留 lockfile 不作為遠端環境依據。

一般操作以 configure 和實驗名稱選設定，不手動改 `.env` 或生成的 selection／manifest。
開發者可修改 YAML 以建立新的受控設定，但需刷新 selection、在無 Pod 使用 volume 時同步，
並以新 run 執行；不能把數值契約變更當成舊 run 的 resume。
遠端 Python 為 `>=3.12,<3.13`；需要套件更新時由 workflow 取得排他環境寫入 lease，
不會為安裝依賴重新下載行情或執行 CPU prepare。

`bash scripts/runpod_workflow.sh` 是**本機控制入口**；
`bash scripts/runpod_tmux_launch.sh` 是 **SSH 進 Pod 後的工作入口**。
建立 Pod 不代表訓練已開始；tmux 可讓 SSH 斷線後繼續工作。
本機 guard 仍需維持開機、連網，並負責終止指定 Pod。

`<RUN_ID>`／`<POD_ID>` 是需替換的值；不要原樣輸入。
所有可選參數、預設值與 alias 集中在 [CLI 參考](#cli-reference-zh)。

##### 讓 agent 暫時操作 RunPod：本機 SSH socket

**只有當使用者希望讓 agent 協助部署、啟動訓練或檢查 RunPod 時，才使用
`runpod-ssh-socket.sh`。** 在專案根目錄執行：

```bash
bash runpod-ssh-socket.sh
bash runpod-ssh-socket.sh 8h
bash runpod-ssh-socket.sh --pod-id "<POD_ID>"
bash runpod-ssh-socket.sh 8h --pod-id "<POD_ID>"
```

`SOCKET_TTL` 為可選的位置參數，預設 `4h`，接受正整數加上 `m`、`h` 或 `d`，
例如 `30m`、`8h`、`1d`。新增的可選參數 `--pod-id POD_ID` 只連線指定的
運行中 Pod，可與 TTL 一起使用；若該 Pod 不存在、無權存取或未運行，腳本會
報錯，不會改連其他 Pod。`--help` 顯示說明。專案位置依腳本所在目錄動態取得，
從其他工作目錄呼叫時也適用。

它使用既有、權限為 `600` 或 `400` 的專案 `.env` 與 REST v2 查詢入口，
不帶參數時，會自動為目前 RunPod API key 可存取的所有運行中 Pod 建立 socket；
只提供 TTL 而未指定 `--pod-id` 時，也會連線全部運行中 Pod。不限制專案名稱、
network volume 或 workflow 身分，也不需要設定 `RUNPOD_NETWORK_VOLUME_ID`。
每台 Pod 使用獨立 socket 與到期程序，全程不要求終端選擇。
優先使用 direct SSH；沒有 direct
endpoint 時使用 API 提供的 RunPod SSH proxy。對應的 SSH 公鑰必須已在
RunPod 設定完成；proxy 連線不支援 SCP／SFTP，詳見
[RunPod SSH 文件](https://docs.runpod.io/pods/configuration/use-ssh)。
登入先嘗試 `~/.ssh/id_ed25519_runpod`，失敗或不存在時，再依序嘗試
`~/.ssh/` 下其他私鑰（包含子目錄）。使用加密私鑰時，先以 `ssh-add` 解鎖；
腳本不會詢問或保存密碼，也不會修改 SSH 設定。

成功後會印出 socket、到期時間、agent 可使用的 SSH 命令，以及檢查／提前關閉命令。
將這段輸出提供給 agent 即可；本次連線索引保存於
`.runpod/ssh-socket/latest.json`，其中 `connections` 以 Pod ID 為 key，記錄各台
成功連線的 metadata；`failures` 記錄失敗的 Pod 與原因。每台 Pod 的每次連線
使用獨立的私有暫存目錄。agent 應先檢查對應 metadata 的 `expires_at_epoch`
及 `check_command`，再使用 `ssh_command` 執行工作；此命令只重用 socket，
失效時直接失敗。部分 Pod 連線失敗時，其他成功的 socket 仍可使用，但腳本
會回傳非零 exit code，避免把部分成功當成全部完成。

多台 Pod 的 API 查詢與 SSH 認證使用 bounded thread pool；待處理 futures
不超過 worker 數量。worker 數量同時依 Pod 數量、CPU 核心數、可用記憶體與
服務負載上限決定，最多 `4` 個；可透過環境變數 `RUNPOD_SSH_SOCKET_WORKERS`
降低並行度。記憶體規劃保留 `512 MiB`，每個存續連線及 active worker 各估算
`64 MiB`。若無法估算記憶體或不足以安全建立全部連線，會在連線前報錯；
可使用 `--pod-id` 縮小範圍。同一台 Pod 的私鑰仍依優先順序逐一嘗試。

TTL 是 socket 接受新共用連線的期限。到期會執行 `ssh -O stop`，
已建立的 SSH 工作及 detached tmux 可以繼續；本機睡眠、重啟或網路中斷可能
讓連線提早失效，睡眠期間到期則於恢復後停止接受新連線。
這個期限與 Pod runtime／guard 截止時間分開管理。腳本只建立連線，部署與訓練
仍使用既有 workflow 與 tmux 入口。

此檔案刻意放在專案根目錄，排除於既有上傳 allowlist，因此不會上傳 RunPod、
不加入 code file manifest，也不參與 SHA 計算；只修改這個腳本不會改變 workflow
身分或觸發重新同步／準備資料。`.runpod/` 的連線 metadata 也不會上傳。
README 本身仍遵循既有文件同步規則。

<a id="setup-zh"></a>

#### 1. 建立帳號資源、憑證與儲存

本機控制端需要 `bash`、Python 3、AWS CLI 與 `curl`。GPU Pod、CPU Pod、
network volume 的建立與 Pod 查詢、啟停、終止都使用 RunPod REST API v2。
CPU Pod 的 vCPU 數量只接受 `2`、`4`、`8`、`16` 或 `32`。
On-demand Pod 沒有供應商端的執行時間上限；本專案以本機 guard 控制截止時間。
此處的系統 Python 3 只供無第三方相依的 manifest/JSON control helper 使用，
不代表建立、載入或
檢查本機專案 Python environment。RunPod 官方文件：

- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [S3-compatible API](https://docs.runpod.io/storage/s3-api)
- [RunPod Secrets](https://docs.runpod.io/pods/templates/secrets)
- [REST API v2](https://api.runpod.io/v2/openapi.json)

在 RunPod Console 建立 project-scoped RunPod API key 與另一組 S3 API key，
再建立下列固定名稱的 RunPod Secrets：

- `huggingface_token`：必要，用來預抓固定 revision 的 Kronos model 與
  tokenizer。
- `wandb_api_key`：必要，用於訓練與 validation tracking。
- `eodhd_api_token`：只有 `us_only_eodhd` 或 `us_tw_eodhd` profile
  需要；`tw_only` 不需要。

包含台灣市場的 profile 另需位於 GCP `asia-east1`（台灣）的 TPEx Cloud Run relay。
relay 部署腳本會為每次通過驗證的部署建立唯一名稱的
`tpex_relay_token_<timestamp>_<nonce>` RunPod Secret，並把
secret 名稱寫回本機 `.env`；不要手動建立固定名稱的 TPEx secret。

不要複製、開啟或手動修改 `.env`。以隱藏輸入方式建立 credential-only
`.env`；腳本會原子寫入並固定權限為 `600`：

```bash
bash scripts/runpod_workflow.sh credentials
```

接著由腳本建立 network volume。成功回傳的 volume ID、datacenter、S3 region
與 endpoint 會自動寫回同一個 `.env`，不需要複製 ID：

```bash
bash scripts/runpod_workflow.sh volume deploy \
  --name stock-forecasting \
  --size-gb 100 \
  --datacenter EU-RO-1
```

若 `.env` 已登記 volume，volume script 預設不會再建立另一個可能計費的
volume；只有刻意使用 `--force-new` 才會建立並改登記新 volume。

<details>
<summary>台灣市場必需：TPEx Cloud Run relay 部署與驗證</summary>

RunPod 機房若被 TPEx data endpoint 以 HTTP 403 拒絕，台灣市場 profile 必須先部署
受限的 Cloud Run relay。建立已啟用 billing 的獨立 GCP project，安裝 Google Cloud
CLI，登入要用來部署的帳號；不需要建立 Cloudflare token、GCP API token、自訂
subdomain、Pub/Sub topic 或 relay 儲存空間：

```bash
gcloud auth login
```

部署腳本會啟用 Cloud Run、Cloud Build、Artifact Registry、Secret Manager 與 IAM
API，建立專用 runtime service account，為 source build 使用的 Compute Engine default
service account 加入 `roles/run.builder`，建立／更新單一 Secret Manager secret，並
設定 unauthenticated network ingress。執行部署的 GCP principal 因此必須具備這些
管理動作所需權限。在個人持有、只用於此 relay 的新 project，首次設定可使用 Project
Owner；多人或正式環境應改用等價的最小權限組合。Google 對 source deployment
列出的基礎角色為 `roles/run.sourceDeveloper`、
`roles/serviceusage.serviceUsageConsumer`、runtime identity 上的
`roles/iam.serviceAccountUser`，而此腳本額外需要啟用 API、建立 service account、
管理 Secret／Secret IAM、設定 Cloud Run public invoker 與授予 build role 的權限。
腳本不會嘗試把這些管理角色授予目前登入者。

`--allow-unauthenticated` 只表示 RunPod 能連到 managed `run.app` HTTPS endpoint；
應用層仍要求長度受限的共享 token。若組織政策禁止 unauthenticated Cloud Run，
部署會 fail closed，不能把 relay 改成沒有應用層驗證的公開 proxy。

```bash
bash scripts/runpod_workflow.sh tpex-relay configure
bash scripts/runpod_workflow.sh tpex-relay deploy
```

`configure` 只把 GCP project ID、固定區域 `asia-east1`、service／Secret 名稱與首次
自動產生的共享 relay token 合併寫入本機 `.env`；`gcloud` 登入 credential 留在本機
Google Cloud CLI credential store，不會寫進 `.env`、source 或 Pod。Cloud Run 會
自動提供 `run.app` subdomain，互動流程不會要求自訂 domain。

`configure` 會保留既有 RunPod network volume、S3、RunPod API key、已啟用的 relay
URL，以及遷移前的 Cloudflare 欄位。Cloudflare 欄位不再被新 workflow 使用；在
Cloud Run live verification 與後續 CPU Pod 實際成功前，腳本也不會刪除舊 Worker
或撤銷舊 token。若 volume 已部署完成，不要重跑 `credentials` 或 `volume deploy`。

`deploy` 會使用本機 `.env` 內既有的 `RUNPOD_API_KEY`，以 Bearer 認證呼叫
RunPod REST API v2。任何 GCP 寫入前，先以唯讀的 `GET /v2/account/secrets`
檢查存取權，不會在預檢階段建立資源；完成 Cloud Run 驗證後，才以
`POST /v2/account/secrets` 建立 Secret。控制程式固定傳送明確的專案 API-client
`User-Agent`。若 API 拒絕請求，腳本會保留 HTTP 狀態與安全的錯誤細節，同時遮蔽
API key 與 relay token。API key 只保存在本機 `.env`，不會隨 source sync 上傳；
預檢或 Secret 建立失敗時，`deploy` 會停止，也不會把新的 relay metadata
啟用到本機 `.env`。

REST API v2 預檢通過後，部署器才會建立 GCP 資源。共享 token 以 Secret Manager
的數字 version 掛入特定 Cloud Run revision，不使用會漂移的 `latest`。新 revision
產生前，Node.js Buildpack 會透過 `gcp-build` 強制執行 relay 單元測試；測試或 build
失敗就不會部署 revision。新 revision 上線後，部署器先驗證 authenticated warmup，
再實際驗證 `dailyQuotes`、`exDailyQ`、
`ROE` 與 `inx` 四個精確路徑；官方 route probe 之間固定間隔 2 秒。全部取得含官方
資料表的 JSON 後，才建立新的 RunPod
Secret 並原子更新本機 `TPEX_PROXY_URL` 與 secret reference。若 live verification 或
RunPod Secret 建立失敗，Cloud Run revision 可能已存在，但該輪仍屬未完成，本機仍
保留先前 URL／Secret reference。修正原因後重跑 `tpex-relay deploy` 即可。

- 以 source deployment 上傳 [`cloudrun/tpex-relay`](cloudrun/tpex-relay)，固定使用
  GCP `asia-east1`（台灣）、Node.js 22、1 vCPU、512 MiB、60 秒 request timeout、
  request-based CPU throttling 與 startup CPU boost。
- 設定 service-level `min instances=0`、`max instances=1`、container
  concurrency `1`。沒有 request 時可 scale to zero；單一 instance／單一 request
  防止 Cloud Run autoscaling 放大既有 `--taiwan-qps`。relay 不再另設一個與 CLI
  衝突的 QPS limiter。
- 只允許 `GET`、共享 token、固定 TPEx origin、四個專案使用中的 path，以及各
  path 的固定 query schema；TPEx 若回傳 redirect，最多跟隨三次且每一跳都必須維持
  相同 HTTPS origin。每次 upstream request 的總 timeout 為 30 秒、response body
  上限為 16 MiB；relay 不進行 provider retry。重新導向回應若設定工作階段 Cookie，
  只會在驗證同源後承接到下一跳，且 Cookie 數量與 header bytes 都有硬上限；Cookie
  不會回傳給呼叫端。
  跨 origin、缺少 Location 或 Cookie 超限會立即拒絕；相同 URL 與 Cookie 狀態再次
  出現時，會判定為沒有進展的 redirect loop。它不是通用或開放式 proxy。
- 成功的 2xx response 必須可解析為 JSON object，但 relay 回傳原始 bytes，不重排或
  改寫 TPEx payload。非 2xx response 保留 status 與有上限的 body，讓既有 provider
  指數退避決定何時退出；relay 自身不會形成無限 retry loop。

這個 MVP 不使用 Pub/Sub、資料庫、Cloud Storage 或固定出口 IP。Cloud Run 使用預設
動態 egress；`asia-east1` 是台灣 region，但 region 本身不是 TPEx 永遠接受該 IP 的
保證，所以四路 live verification 才是建立 CPU Pod 前的必要 gate。若日後仍出現
依 egress IP 而變的 403，再評估 Serverless VPC Access ＋ Cloud NAT 固定 IP；若所有
GCP 台灣出口都被拒絕，才改用台灣本地 VPS relay。

`min instances=0` 配合 request-based billing 時，不會為閒置 Cloud Run instance
支付運算費；但 request、source build、Artifact Registry image 儲存、Secret Manager
與網路流量仍各自依 GCP 定價與免費額度計費，不能把整體服務視為保證免費。可隨時
查看 control-plane 狀態或重新執行完整 live verification：

```bash
bash scripts/runpod_workflow.sh tpex-relay status
bash scripts/runpod_workflow.sh tpex-relay verify
```

TPEx client 仍使用原始 `https://www.tpex.org.tw` endpoint 與 public query params
計算 request SHA-256；Cloud Run URL、relay token 與 transport 模式都不會進入 raw
cache key 或 dataset request identity。切換 relay 後，既有成功的 TWSE、TPEx 與
EODHD JSON cache 會照常續用，只對缺少的 TPEx response 經 relay 發出請求。CPU
workflow 只在確定需要 provider acquisition 時，緊接 `stock-forecasting-download` 前呼叫已驗證的
`/_internal/warmup`；該 request 不會呼叫 TPEx。它不放在 tmux 啟動開頭，避免完整
pytest 與 Hugging Face prefetch 期間 relay 又 scale to zero。若完整 raw checkpoint
已可重用，連 warmup 都不會執行。

舊的 `tpex-proxy configure|deploy|verify|status` workflow 名稱暫時保留為相容 alias，
但實際呼叫的已是 Cloud Run 腳本；新操作請使用 `tpex-relay`。
相關官方文件：

- [安裝 Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- [Cloud Run 區域](https://docs.cloud.google.com/run/docs/locations)
- [從 source 部署 Cloud Run](https://docs.cloud.google.com/run/docs/deploying-source-code)
- [Node.js Buildpack 與 `gcp-build`](https://docs.cloud.google.com/docs/buildpacks/nodejs)
- [Cloud Run IAM 角色](https://docs.cloud.google.com/run/docs/reference/iam/roles)
- [Cloud Run autoscaling](https://docs.cloud.google.com/run/docs/about-instance-autoscaling)
- [Cloud Run minimum instances 與 billing](https://docs.cloud.google.com/run/docs/configuring/min-instances)
- [Cloud Run Secret Manager 整合](https://docs.cloud.google.com/run/docs/configuring/services/secrets)
- [Cloud Run 定價](https://cloud.google.com/run/pricing)
- [RunPod REST API v2 OpenAPI 規格](https://api.runpod.io/v2/openapi.json)

</details>

<a id="configure-zh"></a>

#### 2. 選擇資料與設定，然後同步

在本機專案目錄執行。不帶參數的 `configure` 是互動模式；下列為選項與固定範例。

##### `configure` 參數與資料範圍

`--universe`、`--stocks`、`--etfs` 與 `--symbol-limit` **只控制美國資料
範圍**，不會篩選台股。`us_tw_eodhd` 永遠由「依 universe 選出的 EODHD
美國目標證券」加上「指定日期範圍內 TWSE／TPEx 官方端點回傳、且符合
普通股／TDR／白名單非槓桿股票型 ETF 契約的台灣資料」組成。

`--data-profile` 可選值如下：

| 值 | 美國資料 | 台灣資料 | 是否需要 `eodhd_api_token` |
| --- | --- | --- | --- |
| `tw_only` | 無 | TWSE／TPEx 普通股、TDR、經稽核且可映射的非槓桿股票型 ETF，以及官方 benchmark；`--universe` 必須為 `all` | 否 |
| `us_only_eodhd` | 依 `--universe` 選出的 EODHD 普通股（包含 ADR）與經稽核且可映射的非槓桿股票型 ETF，並自動補入 `VTI.US` benchmark | 無 | 是 |
| `us_tw_eodhd` | 與 `us_only_eodhd` 相同的美國證券範圍 | 與 `tw_only` 相同的台灣目標證券範圍 | 是 |

`--universe` 可選值如下：

| 值 | 意義 | 可搭配的選項 |
| --- | --- | --- |
| `all` | 對含美國資料的 profile，透過 EODHD discovery 取得 active 與 delisted 普通股／ADR，再套用非槓桿股票型 ETF 白名單；對 `tw_only`，表示完整的台灣目標證券範圍 | 美國 profile 可選擇搭配 `--symbol-limit`；不得同時提供 `--stocks` 或 `--etfs` |
| `explicit` | **只限制美國資料**；至少要提供一個 `--stocks` 或 `--etfs`。系統仍會自動補入 `VTI.US` | 只能用於 `us_only_eodhd` 或 `us_tw_eodhd`；不得搭配 `--symbol-limit` |

「完整美國目標證券範圍」在此指 EODHD 帳戶可取得且 discovery 回傳的
active／delisted common stocks（包含 ADR），以及程式白名單內的非槓桿股票型
ETF；使用含美國資料的 profile、`--universe all`，並完全省略 `--symbol-limit`。
槓桿、反向、債券、商品與波動率 ETF 不會成為訓練目標，explicit benchmark
mapping 也不能繞過此限制。這不保證涵蓋 provider 未回傳或帳戶未授權的商品。

所有使用者可設定的 `configure` 選項如下：

| 選項 | 可選值／格式 | 意義與限制 |
| --- | --- | --- |
| `--stage` | `stage1`、`stage2` | 選擇固定的訓練 config。非互動模式必填 |
| `--data-profile` | `tw_only`、`us_only_eodhd`、`us_tw_eodhd` | 決定實際使用的 provider 與市場組合。非互動模式必填 |
| `--dataset-revision` | 1～64 字元；英數開頭，之後可用英數、`.`、`_`、`-`；預設 `v1` | provider 修訂歷史資料時，用新 label 強制建立新的 immutable dataset namespace |
| `--start` | `YYYY-MM-DD`；預設 `2016-01-01` | 所有市場共用的 inclusive 起始日，須早於 `2025-06-01` 且留足 context 與 train 資料 |
| `--end` | `YYYY-MM-DD`；無預設值 | 所有市場共用的 exclusive 結束日，須至少為 `2026-06-01`；超出此日的資料不會進入固定 train／validation／holdout |
| `--h-start` | `1`、`2`、`3`；預設 `1` | 最早輸出的累積持有期，之後逐日至第 14 日；仍於 `t+1` raw open 評估進場。依任務輸出 alpha 或個股方向，只改 runtime labels／模型輸出，不改 dataset identity |
| `--feature-mode` | 新版 production 使用 `scales` 或 `combined`；預設 `combined` | 其他模式只適用關閉 explicit output scale 的自訂／舊設定；不改變 dataset namespace |
| `--universe` | `all`、`explicit` | 控制美國商品選取方式；對 `tw_only` 只能使用 `all`。非互動模式必填 |
| `--stocks` | 逗號或空白分隔的美國 ticker；可重複提供 | `explicit` 模式中的美國普通股／ADR，例如 `"AAPL,BABA"`；會用 EODHD discovery 驗證型別，不影響台股 |
| `--etfs` | 逗號或空白分隔的美國 ticker；可重複提供 | `explicit` 模式中的非槓桿股票型 ETF，例如 `"SPY,QQQ"`；必須在經稽核白名單內，不影響台股 |
| `--symbol-limit` | 正整數 N | **小規模容量／流程驗證用，不是完整美國市場模式。**只適用於含美國資料的 `all` 模式。discovery 後把白名單內的非槓桿股票型 ETF 與普通股／ADR 分開，各自依 active → delisted、ticker 字母順序取最多 N 檔；若某一類少於 N 就全取。這不是隨機或代表性抽樣。接著確認必要的 `VTI.US` benchmark：已在 N 檔 ETF 內就不重複，否則額外補入，所以 raw universe 最多 `2N+1` 檔。要完整美國目標證券範圍就不要提供此選項 |
| `--interactive` | 無值 flag | 明確開啟互動式選單；直接執行 `configure` 而不帶選項時會自動使用此模式 |
| `--reuse-current` | 無值 flag | 保留 active selection 的 stage、feature mode、日期、horizon 與 universe，只刷新目前 config；資料 identity 若將改變則拒絕。不可搭配 `--interactive`；不要搭配其他設定選項，因為它們會被既有 selection 值取代 |

互動模式的預設值是 `stage1`、`us_tw_eodhd`、dataset revision `v1`、起始日
`2016-01-01`、`h_start=1`、`feature_mode=combined` 與 US universe `all`。`--end` 刻意沒有預設值，提示時若留白會繼續
要求輸入，不會自動採用本機當日。Provider acquisition policy 不在 `configure` 設定；
若在此命令提供 `--max-api-calls`、`--eodhd-qps`、`--taiwan-qps` 或 `--maxBackoff`，
會以未知選項拒絕，不會建立另一個 selection。
底層 helper 的 `--project-root` 由 `runpod_workflow.sh` 自動注入，不是使用者
設定資料範圍的選項，不要自行提供。

首次準備某份資料時用 `stage1` 建立資料與模型快取；這不要求先訓練 Stage 1。
已準備好的資料可選 `stage2`，不需再開 CPU Pod。完整美台市場的 B 組資料設定：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --dataset-revision v1 \
  --start 2016-01-01 \
  --end 2026-06-01 \
  --h-start 1 \
  --feature-mode combined \
  --universe all
```

A 組把 `--start` 改成 `2021-01-01`，其他資料參數保持一致；**A 的評估仍需共用 B
資料來源，所以兩份資料都要已完成 prepare。** `--experiment a-signal64` 等任務選項在
後面的 train 命令選擇，不是 configure 的參數。

只下載指定美股、不含台灣市場的設定範例：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_only_eodhd \
  --start 2016-01-01 \
  --end 2026-06-01 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

這個例子含上述四檔美股與必要的 `VTI.US` benchmark。若改成 `us_tw_eodhd`，
台股仍為完整合格 universe，不受這四個 ticker 限制；純台股則使用
`--data-profile tw_only --universe all`，不帶 `--stocks`／`--etfs`。
這些是不同資料範圍的替代選擇，不應在已有資料時逐一照抄執行。

Provider quota、QPS 與此次新增 API attempts 在建立 CPU Pod 時設定，不是 configure
的資料參數。依帳戶剩餘額度設定 `--max-api-calls`；完整 request 計畫超出單次額度時
會續傳，不會縮小 universe。價格與供應商限制請查
[EODHD Pricing](https://eodhd.com/pricing)、
[API Limits](https://eodhd.com/financial-apis/api-limits) 與
[User API](https://eodhd.com/financial-apis/user-api)，不要把程式預設值當作帳戶保證額度。

腳本會建立 `.runpod/selections/<selection-id>.json` 與
`.runpod/active-selection.json`。兩者都不含 secret 且被 `.gitignore` 排除。
目前使用 selection schema 3；舊 schema 不含完整 `h_start` 處理契約，因此更新程式碼後
必須重新執行 `configure`，不會自動轉換。修改 `--end`
會按設計建立新的 dataset request namespace；既有 network-volume 檔案不會被刪除。
Selection identity 包含 stage/config 與資料選擇；dataset request identity 只包含會改變
持久化資料的 profile、日期、universe、symbol limit、revision 與 storage preparation 契約。
**改 LoRA 或主模型學習率不是新資料集**，不可混淆兩種 identity。QPS、API budget 與最大退避只記錄在 CPU launch
metadata、download progress 與 download manifest，不會進入 selection identity。

只修改 `h_start` 會產生新的 training selection SHA，但 `h_start=1`、`2`、`3` 共用相同
dataset request SHA、raw Parquet、symbol bar store、品質 cutoff ranges 與 split audit。
DataLoader 在訓練時才選取對應的 `h_start...14` label，train-only robust scales 也在該次
訓練啟動時由 train split 動態抽樣估計，並寫入每個 checkpoint 供續訓、評估與推論精確
還原。因此切換 `h_start` 不會重建資料、不會掃描 API
cache，更不會呼叫 provider；新 selection 由啟動流程核對所選 dataset manifest 與 artifacts，不要求額外 CPU finalization。日期、symbol universe、provider request 或
`dataset-revision` 改變時才會建立不同的 dataset namespace。

若 provider 可能修訂歷史資料，且確實要為相同 profile/date/universe 建立新快照，
請在 `configure` 明確加入新的 `--dataset-revision <label>`；CPU workflow 不會
覆寫完整的既有 namespace，同 identity 的未完成工作依續傳流程恢復，身分不符的殘留則拒絕混用。

可用下列指令檢視目前選擇，不需開啟 JSON：

```bash
bash scripts/runpod_workflow.sh selection show
```

`.env` 只保存本機 RunPod/S3 credential、GCP relay metadata、relay shared token，
以及腳本回填的 volume、RunPod Secret reference 與 TPEx Cloud Run URL；
stage、資料範圍、runtime 與 config 不從 `.env` 讀取。不要 `source .env`，也不要
把 API key、token 或 secret value 寫進 README、config、shell script 或提交
紀錄。`gcloud` credential 只留在本機 Google Cloud CLI credential store。Pod 只會
收到 RunPod Secret reference 解析出的 relay token 與非敏感 `run.app` URL，不會收到
本機 account-level RunPod/S3 或 GCP deployment credential。

##### 驗證 S3 並上傳程式碼

先執行 read-only S3 權限檢查，再預覽明確的上傳 allowlist：

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/runpod_workflow.sh sync --dry-run
```

確認清單且該 volume 已無 Pod 使用後，才實際上傳並驗證 remote code readiness：

```bash
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh readiness --code-only
```

上傳器會掃描 allowlisted source、config、script、test、`README.md` 與
`pyproject.toml` 的 secret pattern，逐檔上傳並核對遠端大小，最後才發布
`lifecycle/stage1/code.json`。`.env`、cache、資料、checkpoint 與本機
artifact 不會上傳。`poetry.lock` 也不會上傳；它會依 approved RunPod image
的 Python/PyTorch/CUDA 環境在 network volume 上重新產生。

任何 allowlisted 程式碼或 config 修改後，都要重新執行 `--dry-run`、
`--apply` 與 readiness check。所選 config 修改後需用 `configure --reuse-current` 刷新 selection；此選項保留資料範圍，
若資料 identity 會改變則拒絕。未使用的其他 config 修改不會改變這份 selection。只要所選資料集已完成 prepare，且資料內容與 preparation
契約未改變，切換 selection 或更新非資料程式不需重跑 CPU preparation。
GPU gate 仍會驗證所選資料集完整性及目前模型的離線快取，不得略過。

##### 已完成 prepare 後切換 Stage 或 A/B

先用 `selection show` 核對原資料範圍。只改訓練 Stage、h_start 或 feature mode，不需
CPU finalization；資料 profile、revision、日期與 universe 須仍指向已準備好的 namespace。
例如從同資料的 Stage 1 改 Stage 2，重新 configure 時保留原資料參數，只改 `--stage stage2`，
再依本節同步 selection。不要為此執行 `cpu prepare --max-api-calls 1`。

如果已上傳實驗 YAML，可直接以 `train --experiment` 選 A/B 與獨立預測任務；
launcher 自動發布該實驗的 immutable selection，不修改全域 active selection，也不再上傳
原始碼。缺少資料或 baseline 時會在付費建立前報錯；應完成缺少的前置項目，不改 manifest
或重新下載已有資料。Resume／validate 則以 run ID 還原原 selection，見[接續訓練](#resume-zh)。

<a id="cpu-zh"></a>

#### 3. 首次準備資料，或接續未完成的 CPU 工作

**已完成相同資料範圍的 prepare 時跳過本步。** 首次建置需先 configure `stage1`；
目前 creator 遇到 `stage2` 會選 `cpu-finalize`，它只重驗既有資料，不能建立缺少的 bar store。
不需要先訓練 Stage 1 模型。若已完成準備而只切換訓練 Stage／容量，直接進入 baseline／train。

CPU Pod 只能使用 active selection；如果尚未執行 `configure`、config SHA 已改變，
或 selection JSON 不完整，建立前就會失敗。在本機不帶任何選項執行時會進入
互動模式，依序詢問 workload 最長執行時間、這次 Pod 可新增的 EODHD network-attempt
上限、EODHD QPS、TWSE／TPEx 每個 provider 的 QPS、data cleaning/bar-store construction
保留時間、三個 provider 共用的最大單次退避、vCPU 數與 CPU flavor。
`--max-api-calls` 沒有可直接按 Enter 接受的預設值，必須依帳戶當下剩餘額度明確輸入；
其他項目按 Enter 分別
使用 6 小時、16 QPS、每個台灣 provider 0.5 QPS、自動保留、1 分鐘、8 vCPU 與
`cpu3g`。自動保留是 max runtime 的 25%，最多 2 小時；預設 6 小時會保留 90 分鐘。
最後還必須輸入 `y` 或 `yes` 才會建立可能計費的 Pod，直接按 Enter、輸入 `n` 或
`no` 都會安全取消：

```bash
bash scripts/runpod_workflow.sh cpu prepare
```

只要提供任一參數，就會使用非互動模式。此模式必須明確提供 `--max-api-calls`；
EODHD／Taiwan QPS 與資源選項未提供時才採用上述預設值。因此自動化腳本可明確提供
全部參數，不需要修改 `.env` 或重新執行 `configure`：

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve 2h \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

若希望先在命令列提供互動提示的預設值，再由使用者確認或覆寫，可加入
`--interactive`：

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --interactive \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve auto \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

`--max-api-calls` 是正整數，只計入本次 CPU Pod 的 EODHD cache miss 與 retry；它不是
帳戶每日總額度，新的 CPU Pod 也不會自動扣除前一次 Pod 的帳戶使用量。`--eodhd-qps`
與 `--taiwan-qps` 必須大於零，預設分別為 `16` 與 `0.5`；Taiwan 值是每個 provider
各自的 limiter，因此 TWSE 與 TPEx 同時執行時是兩個獨立的 0.5 QPS 上限。
`--maxRuntime`、明確的 `--prepareReserve` 與 `--maxBackoff` 都接受正整數加 `m`、`h` 或 `d`；
`--prepareReserve auto` 使用上述自動公式，明確值必須短於 max runtime。
`--maxBackoff` 預設為 `1m`，同時套用於 EODHD、TWSE 與 TPEx：若下一次由指數退避或
`Retry-After` 得到的等待時間**超過**此值，只有該 provider 迴圈會退出（等於上限仍會
等待）。EODHD 會在 `--max-api-calls` 用完或超過這個退避邊界時停止，兩者任一先發生
即生效。
`--cpuNumber` 只接受 `2`、`4`、`8`、`16` 或 `32`。`--cpuFlavor` 只接受下表六個
RunPod 值；其他數值會在建立 Pod 前失敗：

| Flavor | 世代 | 類型 | RAM / vCPU | 32 vCPU RAM | Container disk 上限 |
| --- | ---: | --- | ---: | ---: | ---: |
| `cpu3c` | CPU3 | Compute-Optimized | 2 GB | 64 GB | 10 GB/vCPU |
| `cpu3g` | CPU3 | General Purpose | 4 GB | 128 GB | 10 GB/vCPU |
| `cpu3m` | CPU3 | Memory-Optimized | 8 GB | 256 GB | 10 GB/vCPU |
| `cpu5c` | CPU5 | Compute-Optimized | 2 GB | 64 GB | 15 GB/vCPU |
| `cpu5g` | CPU5 | General Purpose | 4 GB | 128 GB | 15 GB/vCPU |
| `cpu5m` | CPU5 | Memory-Optimized | 8 GB | 256 GB | 15 GB/vCPU |

Container disk 預設要求 30 GB；若較小的 vCPU 數使該值超過表中的上限，建立腳本
會自動降到該 flavor 的合法上限。若另以 `RUNPOD_CPU_CONTAINER_DISK_GB` 明確指定
超過上限的值，則會在建立 Pod 前失敗。

指令會輸出 Pod ID、外部 hard-limit guard 與 SSH 後應執行的 workflow。
由 RunPod Console 的 Connect 頁面取得 SSH 命令。登入 Pod 後執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh cpu-prepare
```

如需即時查看，可 attach 到 tmux；離開時用 `Ctrl-b d`，不要停止 session：

```bash
tmux -L stock-forecasting-cpu-prepare attach -t stock-forecasting-cpu-prepare
```

<details>
<summary>CPU preparation 產物、資源規劃與資料身分細節</summary>

`cpu-prepare` 會依序：

1. 建立 persistent directory layout、Poetry 2.4.0 與 remote Python 3.12
   `.venv`，並在 RunPod image 內產生 canonical `poetry.lock`。
2. 驗證專案內固定的 Kronos source SHA-256，從 Hugging Face cache 驗證固定的
   model/tokenizer revisions，執行完整 pytest；遠端流程不會 clone Git repository。
3. 從 mounted immutable selection 重新驗證 stage、profile、日期、universe、
   config SHA-256 與 Pod environment，再依該 selection 下載資料；cache hit 不會
   再次呼叫 provider。腳本會同時讀取使用者要求的 vCPU 數、RunPod 提供的
   `RUNPOD_CPU_COUNT` 與容器實際可見核心數，採三者最小值作為 worker 數；
   EODHD、TWSE 與 TPEx 以三個獨立頂層迴圈平行抓取，迴圈內再依商品／官方 benchmark
   所列實際交易日使用 thread pool；bar-store preparation 的 raw scan、bucket compaction、
   candidate ranges 與 split ranges 使用 `spawn` process pool，pytest worker 也不會超過
   這個有效核心數。process 數不是直接照搬 vCPU 數：程式會取 cgroup 與作業系統可用
   記憶體的較小值、保留 parent process headroom，最多只把 60% 的當下可用記憶體列入
   worker 預算，再依該階段最重 task 的保守膨脹估算降低 worker 數。raw scan 會先把來源
   Parquet 的細碎 row groups 合併為約 `4 * batch_rows` 的粗粒度來源分區；每個 process
   仍只以 `batch_rows` 為上限串流讀取，在 Pod 本機 `/tmp` 建立 bucket 暫存，最後只向
   Network Volume 原子發布一個分區 Parquet 與 checkpoint。每個 child 的 Arrow／BLAS
   native thread 固定為 1，避免 process 與 native
   thread 相乘。若單一 bucket 的估算已超過安全預算，process pool 不會啟動，已完成的
   checkpoint 仍可由較大記憶體的後續 Pod 接續。每個 provider 有自己的 QPS limiter。
   `--max-api-calls` 只限制 EODHD，
   TWSE／TPEx 沒有專案端 request-count ceiling；三個 provider 都由同一個
   `--maxBackoff` 值控制各自的退避邊界。
   任一 provider 先退出都不會取消另外兩個；完成的 provider 會先原子發布 durable
   Parquet/request-log checkpoint。主流程 join 全部迴圈後才合併已驗證的 checkpoints
   或發布續傳狀態。完整 request 估算只作資訊；workflow 自動保留 max runtime 的 25%（最多 2 小時；預設 6 小時即
   90 分鐘）給 data cleaning/bar-store construction，也可用 `--prepareReserve` 調整。
4. 建立並驗證下列 persistent artifacts：

   | 遠端路徑 | 內容 |
   | --- | --- |
   | `/runpod-volume/datasets/<dataset-request-sha256>/api-cache/` | provider raw response cache |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-progress.json` | 續傳 attempt、cache 數量與 provider／budget／runtime 等待狀態 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/provider-checkpoints/` | 可驗證並跨 CPU Pod 重用的 provider materialization checkpoints |
   | `/runpod-volume/datasets/<dataset-request-sha256>/raw/market.parquet` | durable `downloaded` checkpoint 的 canonical daily OHLCV |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/shards/` | 依 symbol row group 壓縮且可隨機讀取的 OHLCV bars |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/symbol-index.parquet` | symbol → shard/row-group 索引 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/cutoff-ranges.parquet` | prepared 候選 cutoff ranges；runtime 另套用完整連續性／流動性規則 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/_SUCCESS.json` | bar store 完成與完整性 checkpoint |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/execution-plan.json` | 建置中各階段的記憶體預算、有效 process 數與續用 task 數；成功後回收 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/scan-index.json` | 建置中來源分區至 bucket row group 的精確索引；成功後回收 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/scan-partitions/` | 粗粒度、可續傳的 raw scan 分區；成功後回收 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-manifest.json` | 實際 provider、profile、symbols 與下載 provenance |
   | `/runpod-volume/datasets/<dataset-request-sha256>/dataset-manifest.json` | split counts、hash 與資料契約 |
   | `/runpod-volume/datasets/<dataset-request-sha256>/manifests/api-request-log.jsonl` | 不含 token 的 request audit |
   | `/runpod-volume/cache/huggingface/` | 離線 Kronos model/tokenizer cache |
   | `/runpod-volume/cache/hf-models.json` | 固定 model revisions 與 cache manifest |

5. raw Parquet、download manifest 與 request log 完整驗證後，先在
   `/runpod-volume/lifecycle/stage1/cpu-preparation.json` 發布 `downloaded`
   執行狀態。沒有 exit code 的 `downloaded` 是同一 Pod 內的中間 checkpoint，外部 guard
   不會誤判為終態；若剩餘時間少於 cleaning reserve，腳本才以 exit code 75 將它標記為
   可續傳終態，下一個 CPU Pod 直接從 checkpoint 執行清理，不再呼叫 provider。bar-store
   建置期間若剩餘時間到達安全截止點，則以 `waiting_for_preparation` 與 exit code 75
   結束；下一個 CPU Pod 會依 `scan-index.json` 完全跳過已完成的來源分區與 compacted
   buckets。若中斷發生在單一來源分區內，最多只重做該分區；已原子發布的分區不會
   重寫，也不需要列舉數萬個 segment 目錄。舊版未完成的 segment checkpoint 若與目前
   差異僅為 scan 執行演算法，會以目錄 rename 隔離，不會變更 raw Parquet 或重新呼叫
   provider。worker 數與執行時記憶體規劃只影響排程，
   不屬於 dataset identity；改用不同 vCPU／RAM 的 CPU Pod 不會重新下載 provider 資料，
   也不會使既有 bucket checkpoint 失效。其後才將
   launch 專屬的隱藏 dataset-manifest 暫存檔直接寫在同一個 dataset root，確保其中的
   artifact 相對路徑在發布前驗證與發布後都指向相同檔案；驗證通過後才以同檔案系統
   rename 原子發布為 `dataset-manifest.json`。接著將
   dataset request、storage preparation、所選 provider 與 bar-store 的 data-content
   identity、resolved artifact hashes，以及建立時的 selection/stage/config provenance
   寫入 `/runpod-volume/lifecycle/stage1/dataset.json`。這個檔案只表示不可變資料已
   `ready`，不承載 `preparing`、失敗或續傳狀態；CPU 工作的所有執行狀態只寫入
   `cpu-preparation.json`，因此 lint、setup 或下載失敗不會覆寫已完成的 dataset
   readiness。它是最近一次 CPU prepare 的相容性摘要，不是訓練資料的選擇依據；
   主模型與 baseline 都按 active selection 讀取 `datasets/<dataset_request_sha256>/`
   內的 manifest。若新的 active selection 改變資料請求，舊 selection 的 marker 會先移入
   `lifecycle/stage1/history/`，只移動 canonical pointer，不刪除舊資料集。只有全部檢查
   成功才發布新的 dataset marker，CPU guard 再依獨立的
   `cpu-preparation.json` 終態自動終止 Pod。

資料身分分成三層，避免修改非資料程式就重建整份資料：dataset request SHA 只包含
profile、日期、universe、明確的 dataset revision，以及會改變持久化 bar/cutoff 的
storage preparation 欄位；provider materialization digest 只包含該 provider 的 endpoint
參數、商品篩選、日期邊界、解析、調整與數值驗證；bar-store digest 只包含清理、benchmark
資格、cutoff 與時間切分的數值語意。QPS、API 次數上限、retry/backoff、Cloud Run relay、
worker/process 數、記憶體估算、checkpoint 目錄布局、logging、錯誤文字、lifecycle、CLI
wrapper、訓練與驗證程式都不屬於資料內容身分。完整 code release hash 仍保留作 provenance
與上傳完整性檢查，但不會單獨讓已驗證資料失效。若 provider 語意真的改變，只隔離並重建
相應 provider materialization 與其下游資料，既有 request-key 相同的 raw API cache 可重用；
若只有 bar-store 語意改變，只隔離並重建衍生 bar-store，不重新呼叫 provider。

任何 CPU 或 GPU 工作在讀寫 volume 前，都會先證明 `/runpod-volume` 是**精確的
mount point**：優先使用 `mountpoint`，否則使用 `findmnt`，最後才檢查
`/proc/self/mountinfo`。建立 Pod 時由本機選定的 volume ID 會以
`RUNPOD_EXPECTED_VOLUME_ID` 傳入，並與 RunPod 自動提供的 `RUNPOD_VOLUME_ID`
逐字比對；路徑正確但 ID 不同也會立即停止。專案固定放在
`/runpod-volume/stock_forecasting`，資料、Kronos cache、W&B transaction 與模型則
位於 volume root 下的獨立子目錄；任何 persistent path 都不得使用會被 Pod
重建清除的 `/workspace`。這個 layout 避免把專案目錄本身當成 mount target，
也避免 network volume 掛載時遮蔽同名的 Pod 內建目錄。

</details>

Pod 終止後，以 S3 lifecycle 為準，不要依賴已消失的 SSH session：

```bash
bash scripts/runpod_workflow.sh status
```

##### Provider quota 與跨 CPU Pod 續傳

EODHD、TWSE 與 TPEx 使用彼此獨立的平行迴圈。retryable 的 429、暫時性網路錯誤或
provider 5xx 都採指數退避；台灣官方端點由 Pod IP/WAF 暫時回傳的 403 也視為
retryable。三個 provider 共用同一個 `--maxBackoff` 設定（預設 `1m`），但各自獨立
計算退避並在下一次等待超過該值時退出。EODHD 另外受到該 CPU attempt 的
`--max-api-calls` 限制，兩個 EODHD 邊界任一先發生就停止其迴圈；TWSE／TPEx 則沒有
request-count 上限。共同 acquisition deadline 仍可讓任何迴圈進入
`waiting_for_resume`，以保留 data cleaning 時間。流程遵守下列契約：

1. 已成功取得的每個 raw JSON response 仍保留在該 dataset request 專屬的
   `api-cache/`。完整的單一 provider 可原子發布至 `provider-checkpoints/`，但不完整的
   provider staging Parquet 與 aggregate `state=ready` marker 都不會發布。
2. EODHD 先達 `--max-api-calls` 或先超過最大退避時，都只退出 EODHD 迴圈，
   TWSE／TPEx 繼續；任一台灣 provider 先超過最大退避時，也不會停止 EODHD 或另一個
   台灣 provider。最大退避越界會開啟該 provider client 的共享 circuit breaker；
   已送出的 in-flight request 可能完成，但同一 provider 的其他 worker 不會再啟動新
   request。主流程一定等到所有已選 provider 迴圈退出，才決定下一步與允許 CPU Pod
   結束。
3. `download-progress.json` 記錄 attempt number、各 provider cache／network counts、
   EODHD limited count，以及每個 provider 的 `complete`、`waiting_for_budget`、
   `waiting_for_provider` 或 `waiting_for_resume` outcome；provider 錯誤另記已等待總秒數、
   最後等待、下一次 proposed backoff、三者共用的最大值，以及該次 launch 實際使用的
   `max-api-calls`／EODHD QPS／Taiwan QPS。HTTP status、`Retry-After` 與
   rate-limit headers 只在供應商有回傳時記錄；資料契約錯誤另記安全的 provider、
   operation、symbol/date/month 與 exception type，不記錄 token 或 response body。
   `complete` outcome 另記 materialization checkpoint identity，以及該 checkpoint
   是本次新發布或直接重用。
4. 若任一迴圈仍未完成，CPU preparation lifecycle 依整體 outcome 進入
   `waiting_for_budget`、`waiting_for_provider` 或 `waiting_for_resume`，GPU readiness
   維持不通過，CPU Pod 才自動終止；已完成 provider 的 durable checkpoints 仍會保留。
   若三者都完成，則以固定 provider 順序合併通過驗證的 checkpoints，繼續 data
   cleaning，不會提早關閉 Pod。
5. 額度恢復後，**不要重新 `configure`、不要改 `--dataset-revision`、不要刪除
   cache**。在本機再次建立 CPU Pod 時，依新的剩餘額度重新輸入 `--max-api-calls`；
   QPS 也可依當次 provider 狀態調整，兩者都不會改變 selection。登入後重新啟動同一個
   workflow：

   ```bash
   bash scripts/runpod_workflow.sh cpu prepare
   # Run after connecting to the newly created CPU Pod:
   cd /runpod-volume/stock_forecasting
   bash scripts/runpod_tmux_launch.sh cpu-prepare
   ```
6. 新 attempt 會先驗證並重用具有相同 provider materialization request 與 data-content
   digest 的 provider checkpoints。只有未完成或內容身分不相容的 provider
   才重新播放已快取 responses 並對缺少的 request 呼叫 provider。只有全部資料、
   manifest 與 selection gate 都通過後，lifecycle 才會變成 `ready`。`all` 模式的
   discovery response 也屬於同一份 immutable cache，因此跨日續傳不會重新取得一份
   已漂移的商品清單。

EODHD 偶爾會在已上市商品的日資料中回傳全零或其他無法通過 canonical OHLCV contract
的 placeholder row。下載器不補值、不改價，而是只丟棄無效 source rows，將數量記入
`dropped_source_rows`，並繼續保留同檔商品的有效 observations。TWSE 的早期除權資料
有時存在於 `TWT49U` 主表，但 detail 端點回傳「無相關資料」；此時保留可驗證的
price factor，share multiplier 使用 identity `1.0`，並在
`missing_share_multiplier_details_by_provider` 明確記錄 volume-adjustment coverage gap，
不把未知比例偽造成完整資料。

`bash scripts/runpod_workflow.sh status` 會在 `cpu_prepare` lifecycle 下方顯示 download
attempt、已快取 response 數、此次 network request 數、可用的完整 request 估算與
安全的 provider error 摘要，不需要手動開啟 JSON。

這同時提供 request-level 與 provider-materialization-level 續傳，不是 HTTP response
的 byte-range 續傳。每個成功完成的 API request 是 raw-cache 續傳單位；每個通過 hash
與 identity 驗證的完整 provider checkpoint 是 materialization 續傳單位。若
`download-progress.json` 的 identity 與目前 dataset request
不同，流程會 fail closed，避免混用不同 profile、日期或 universe。過小的
`--max-api-calls` 會產生可續傳的 `waiting_for_budget`，不是失敗，也不要求重新
`configure`；可用相同 selection 直接建立下一個 CPU Pod。401／403 或其他非暫時性
設定錯誤才標記為 `failed`，應先修正 Secret。只有確實要建立新的
provider 資料快照時才改 `--dataset-revision`，新 revision 不會沿用舊 snapshot cache。

若狀態是 `waiting_for_budget`、`waiting_for_provider`、`waiting_for_resume`、
`waiting_for_preparation`、`failed` 或 `timed_out`，用以下命令下載 CPU 紀錄；腳本會解析 lifecycle 中的
`launch_id`、`log_path` 與 `progress_path`，不需手動查 JSON 或輸入遠端路徑：

```bash
bash scripts/runpod_workflow.sh cpu-logs
```

CPU 結束後先檢查狀態；baseline 使用 `readiness --baseline`，主模型使用
`readiness --gpu`。建立命令會自行處理前置檢查，無須手動修改 readiness marker；
baseline 會重用仍有效的近期檢查結果：

```bash
bash scripts/runpod_workflow.sh status
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64
```

<a id="baseline-zh"></a>

#### 4. 建置或接續同組兩種 baseline

先完成所需 train 與共用 evaluation 資料的 CPU prepare，再同步程式與設定。
成對 baseline 命令一次處理同組的 alpha 點估計及個股漲／平／跌兩套比較模型：

```bash
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1
```

```bash
bash scripts/runpod_workflow.sh baseline \
  --experiment a-signal64 \
  --maxRuntime 24h \
  --gpuId "NVIDIA GeForce RTX 4090"
```

B 組改為 `b-signal64`。`--data-center` 僅篩選庫存，須選已登記 volume 的機房。
`--maxRuntime`／`--gpuId` 及 alias 的預設值仍是 `12h`／RTX 5090。
同一次只接受一個 `--experiment NAME`，不接受 `--seed`、`--launchWorkers`。
兩個 task cache 都在本機查詢；已全部完成直接跳過，不租 Pod。若有未完成工作，只建一台
Pod，依序檢查／建置兩種 baseline，跳過已完成的一種；24h 是整台的共同期限。
兩套結果維持獨立數值身分，跨主模型 seed 共用，不是每個 seed 重訓一份。

唯讀確認兩種 baseline 與資料規則：

```bash
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64
```

輸出各 task 的狀態；只有兩者都完成且資料規則通過才回傳 0。成對操作不要求手動切換
alpha／方向設定，不改全域 active selection。省略 `--experiment` 仍保留原本 active-selection
單任務行為，不能當成整組完成查詢；進階單任務入口見[附錄](#single-model-zh)。

獨立點估計、固定報酬三分類、舊分位數／成本方向 baseline 不能互相冒充完成結果；
任務及其 beta／neutral threshold 納入 baseline 契約，不綁主模型 LoRA 容量或 training seed。
同資料、同任務、同 baseline 參數的主模型可共用結果。選擇新任務不重做 CPU prepare，
不重新下載行情；既有模型結果保留，不覆寫成另一種任務。
已準備好的資料只因更新主模型設定而刷新 selection 時，用 `configure --reuse-current`；
不重跑 `cpu prepare`／`cpu-finalize`。

`readiness --baseline --experiment a-signal64` 是整組唯讀查詢，**同時檢查目前資料篩選規則及 baseline 是否完成**。
它核對所選 training bar-store、`configs/data_cleaning.json` 指定的共用 validation／test
bar-store，以及符合當前資料與 baseline 訓練契約的完整結果。A 組不能以自己的評估快照
代替共用 B 組來源。不要求 Kronos／HF cache，也不讀取主模型 YAML 的訓練設定或
「最近一次 CPU prepare」紀錄；不建立 Pod、不重新訓練。

預設只顯示完成狀態、資料規則、有效樣本筆數、baseline ID 及下一步：

- `Baseline: COMPLETE` 且 `Data rules: PASS`：所有模型、評估指標、權重／訓練結果、
  共用評估資料與預測檔均通過現有完整性檢查；三個 split 的筆數與當前有效索引一致。
  exit code 為 **0**，可重用這份 baseline，不需要重訓。
- `Baseline: NOT COMPLETE`：目前資料與規則沒有匹配的完整結果。即使資料規則已通過，
  仍須執行 `baseline` 建置或接續；exit code 為 **1**。
- `STORAGE FINALIZATION REQUIRED`：已有訓練結果，但尚須完成共用結果的儲存整理；
  exit code 為 **1**，執行 `baseline` 完成整理，不需重訓。
- 資料、規則、結果檔缺損或遠端查詢失敗：exit code 為 **2**，輸出具體錯誤，
  不會假報完成，也不把連線失敗當成「尚未建置」。

`readiness --baseline` 與 `baseline` 共用本機 10 分鐘的成功檢查快取。
程式版本、volume、資料選擇、清理規則及遠端 artifacts 均未變更時，後續建立或重試只核對
小型 manifests 與分頁物件清單中的版本／大小，不再逐一查詢所有 shards，也不重印完整報告。
超時、程式或規則改變、檔案被替換／刪除、清理索引新建完成時自動重新驗證；遠端查詢失敗時
不採用舊結果。首次直接執行 `baseline` 也會完成檢查，無須先另跑 `readiness`。
此快取只加速資料輸入檢查，不是 baseline 模型身分或訓練完成紀錄；每次完成查詢仍核對
baseline 結果。建立 Pod 時的程式部署檢查、即時 Pod 衝突檢查與掛載後資料驗證也保留。
唯讀查詢不要求本機查詢腳本與遠端部署逐檔相同；正式啟動前仍須同步修改過的程式，
且必須通過部署版本檢查。同步程式不等於 baseline 需要重訓。

資料檢查區分候選筆數與有效筆數，CLI 的 `Eligible windows` 只顯示後者：

- `prepared_candidate_counts` 僅代表既有 CPU prepare 的候選筆數，**不是清理後有效 windows**。
- `eligible_sample_counts` 才是目前連續性、有效行情與最低流動性規則下的筆數。
  只接受符合目前規則、來源快照、128-bar 視窗與固定 split 的 `sample-universes` 索引；
  validation／test 的數字來自共用評估來源。
- `cleaning_state=ready` 表示已核對當前清理索引與稽核筆數；`pending_build` 表示仍須
  在 baseline 流程中先建立索引，尚未確認的筆數為 `null`，不以舊筆數代填。
  此時資料可供建置，但不是清理或 baseline 訓練已完成；唯讀完成查詢不會回傳成功。
  無須重跑 CPU prepare、重新下載行情或手動修改 manifest。
- 共用來源缺失、索引損壞、規則／來源不符或實際使用的 split 沒有有效樣本時，檢查失敗。
  舊規則的索引保留原樣，不會被當成新規則的已完成索引。

本機驗證小型 metadata 的 checksum、prepared shard/index 物件大小及 cleaned index 的
存在與非空狀態；Pod 掛載後串流驗證實際資料與 cleaned range index 的 checksum。
這不是在本機重新掃描所有行情。`readiness --gpu` 保留給主模型，不是 baseline 的前置要求。
`readiness --baseline` 與 `baseline`／`train` 共用同一套已完成結果驗證，
查詢或建立命令都不接受不符合當前篩選規則的舊 baseline。

主模型的 `readiness --gpu`、Pod 內檢查及訓練紀錄同樣使用所選資料集的 manifest；
Kronos cache 另外依目前 config 的 repository、固定 revision 與離線驗證結果檢查，
不與 CPU prepare 當時的共用 HF manifest 檔案雜湊綁定。切換至已完成 prepare 的
A／B 資料集，無須重跑 CPU prepare、下載或切分，也不會使既有 baseline 失效。
續訓／評估則驗證 run 本身記錄的資料 manifest、原始 checksum 及該 run 固定的 selection；
不會以另一組最後完成 prepare 的摘要取代該 run 的資料身分。

`baseline --experiment a-signal64` 自動讀 `.env` 的 network volume，並由 configure 的資料選項
建立兩個 task selection；不改變 active selection。在本機檢查完成 manifest
和所有輸出物件大小；完全匹配就顯示 cache hit 並結束，**不建立 Pod**。若建立 Pod，使用
Console 提供的 SSH 連線後執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh baseline
```

工作由 detached tmux 執行；不需要持續維持 SSH。查看即時 log：

```bash
tmux -L stock-forecasting-baseline attach -t stock-forecasting-baseline
```

按 `Ctrl-b d` 離開不會中斷工作。完成／失敗／逾時後由**本機 guard**終止 Pod，
因此本機須保持開機與網路；baseline 不依賴 Pod 自我終止。先在 Console 或唯讀
`recover --pod-id <POD_ID>` 確認原 Pod 已結束；`status` 的持久化結果不能證明已停止計費。
需要接續未完成的一種或兩種任務時，再執行同一成對命令：

```bash
bash scripts/runpod_workflow.sh baseline --experiment a-signal64 \
  --maxRuntime 24h --gpuId "NVIDIA GeForce RTX 4090"
```

若先前已完整完成，這次應直接命中快取；若逾時或失敗，會建立新 Pod，SSH 後仍執行同一個
tmux baseline 命令，接續已保存的 job／checkpoint。完整 baseline 可能超過單次 24 小時，
此參數是單次 workload 上限，不是完成時間保證。可用 `readiness --baseline --experiment a-signal64` 唯讀確認
`COMPLETE` 與 `PASS` 後，再進入主模型建立流程。注意：`baseline` 不是唯讀查詢命令；
快取未完成時會建立付費 Pod。`train` 若找不到匹配的完整 baseline，會在本機拒絕建立
Pod，不會自動執行 baseline 訓練。


每個 dataset、prediction task 與 baseline 數值契約組合各建立一份結果；只有三者相同的
主模型才共用。Baseline 的基本參數、early stopping 與模型清單來自
`configs/baseline.json`，再套用所選任務的 objective 與 beta／neutral threshold。
主模型的 Kronos revision、LoRA 容量、training seed、feature mode、學習率與 checkpoint
不參與 baseline 啟動或快取判定；但切換 prediction task、point loss beta 或 neutral
threshold 必須使用相符的 baseline。資料內容／切分或 baseline 自身數值契約改變也須
建立對應結果，不能混用不同股票—日期集合或任務。

切換至**已完成 CPU prepare 的** A 組（2021 起始）時，先在本機設定下列完整資料選擇；
此例接著建置成對 baseline。切回 B 組時將 `--start` 改為 `2016-01-01`、
`--experiment` 改為 `b-signal64`，仍會一次處理兩個任務。
其餘資料選項須維持原先 prepare 設定。程式與實驗 YAML 已同步時，不因切換而重傳：

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage2 --data-profile us_tw_eodhd --dataset-revision v1 \
  --start 2021-01-01 --end 2026-06-01 \
  --h-start 1 --feature-mode combined --universe all
bash scripts/runpodctl_project.sh gpu list
bash scripts/runpod_workflow.sh baseline --experiment a-signal64 --maxRuntime 24h --gpuId "NVIDIA GeForce RTX 4090"
```

此流程不執行 prepare、不重新切分或下載行情，也不需要手動調整 manifest、SHA 或路徑。
如果資料檢查失敗，先處理明確指出的資料問題；不要直接重跑 CPU prepare。

<details>
<summary>Baseline 資源調校、儲存與中斷恢復</summary>

baseline 以 `configs/baseline.json` 管理參數及資源：預設同張 GPU 最多兩個 deep jobs，
另有一個 CPU rule／GBDT job；先按 CPU、cgroup 記憶體、GPU 可用記憶體與 `/dev/shm`
計算可准入數量，並限制 DataLoader workers、prefetch、原生 BLAS threads 與 job deadline。
神經模型與 CPU job 在預算允許時同時執行，資源不足會明確降並行度或拒絕執行。
完整 GBDT 所需 RAM 及全量 validation 成本遠高於舊 20,000 筆方案；請以 resource-plan
與進度 log 判斷硬體需求，不要以降低資料量繞過檢查。`.pt`／`.pkl` 是本專案受信任的
hash-scoped 輸出，勿載入第三方不可信權重。

每個 deep baseline 先以 GPU probe 排除超出記憶體預算的 batch，再分別以真實 train／
validation windows 聯合實測 batch size、worker 數與 prefetch；選擇端到端吞吐在最佳值
95% 內、資源占用較低的組合。探測使用模型副本，不更新正式權重、optimizer 或 RNG。
神經模型直接接收固定形狀的連續 tensor batch，在 pin memory 前完成合併；不建立逐筆
Kronos metadata／padding，亦不預先展開全資料集。training／validation／testing 交替
使用單一 worker pool，切換時釋放上一個 pool，再從已提交的 sample cursor 恢復。
CPU workers／GBDT threads 同時受 affinity 與 cgroup v1／v2 CPU
quota 限制，不會把主機核心數直接當成容器可用核心數。調校結果與測量保存在各 job 的
`runtime-plan.json`；可在 `configs/baseline.json` 的 `resources` 調整 batch／prefetch
上限、probe 次數及保存間隔。`auto_batch=false` 才使用固定 `batch_size`。
動態數值讀取只解碼必要的 OHLCV／調整價格／時間欄位；baseline 各 worker 使用有界
Parquet metadata cache（預設最多 128 個檔案、512 MiB 的保守 metadata 記憶體估計），
由 `resources.parquet_cache_files`／`parquet_cache_bytes` 調整。快取不包含展開後的
windows，spawn 不會傳遞檔案 handles；worker 記憶體預算預設為 1 GiB。

排程預留完整 GBDT 與 GPU jobs 的共同 host RAM（預設保留 20% 安全空間），CPU inputs
完成後優先啟動 GBDT。若連一個 GPU job 都無法安全重疊，會拒絕執行並提示較大 RAM，
不會默默延後成 CPU-only 尾段。GPU jobs 全部完成後，GBDT 在下一次 native fit 使用
釋出的 CPU threads；`live-resources.json` 記錄配置，`progress.json` 記錄資料等待比例。
GBDT 的總工作量仍可能比神經模型長；此排程不保證完全消除 CPU-only 尾段。
Baseline 規劃可保守計入 cgroup 中乾淨、未映射檔案快取的 50%，再套用上述安全空間；
不計入匿名記憶體、shared memory、mapped／dirty／writeback 頁面，仍受所有有效的
容器與主機可用記憶體上限限制。`resources.reclaimable_file_cache_fraction` 可調整此比例，
統計不可用時回到原始 headroom；`resource-plan.json` 同時記錄原始預算及快取折抵量。
這是保守的准入估計，不代表作業系統保證能即時回收；CPU prepare 的規劃不受影響。

經審核且數值契約不變的執行效率修正，可透過
`configs/baseline_execution_compatibility.json` 的**完整 source hash 白名單**保留既有
baseline ID、inputs、完成的 jobs 與 resume checkpoint。`execution-contract.json` 記錄
實際程式版本；任何不在白名單內的 baseline 程式修改仍使 ID 失效，資料期間、模型參數、
標籤或校準契約改變也不會被忽略。不同 Python 版本使用同一個穩定 AST hash 格式。

續訓使用相同的上述 `baseline`／tmux 命令，不需指定 checkpoint 路徑。神經模型在第一個
batch 後、預設每 300 秒及完整 validation 前後保存 `resume.pt`，包含權重、optimizer、
LR scheduler、亂數狀態與 epoch 內樣本位置。更換硬體／batch size 後從該位置接續同一
隨機 epoch；warmup 與 validation 邊界以樣本數對齊。未完成的 validation 會重跑完整
split，不能把部分結果當成 early stopping 依據。training 保留動態抽樣，validation／test
仍逐產品完整、固定順序動態取出 windows，沒有展開成常駐記憶體中的巨大資料集。

CPU tabular input cache 每 300 秒先 flush／fsync 再發布續作位置；重啟從已確認的 row
繼續，不重建已完成 split。GBDT 每完成一個 horizon（歷史分位數另按 quantile）的增量 fitting 就保存；
rules 從尚未完成的 rule 接續。中斷中的原生 GBDT fit、完整 validation 或最終 test 會重跑
該工作單元。這些快取與 checkpoint 屬於 baseline 訓練產物，不需要新行情 API call，
也不要求重跑 CPU prepare。進度預設每 30 秒輸出，完整結果仍由 `complete.json` 認定。

</details>

<a id="train-zh"></a>

#### 5. 選擇任務／實驗並建立訓練 Pod

<a id="gpu-catalog-zh"></a>

##### GPU 資源查詢

先查詢 GPU 型號與庫存目錄，取得完整 `gpuId`；訓練預設型號是
`NVIDIA GeForce RTX 5090`。不帶參數會列出所有機房，不會自動以 `.env` 的機房篩選：

```bash
bash scripts/runpodctl_project.sh gpu list
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1
bash scripts/runpodctl_project.sh gpu list --data-center EU-SE-1 --search "5090"
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1 --output json
```

| 參數 | 預設／格式 | 說明 |
| --- | --- | --- |
| `--data-center ID` | 預設不篩選；例如 `EU-RO-1`、`EU-SE-1` | 依完整機房 ID 篩選，不區分大小寫；不是 `EU` 等區域前綴。不會改變 `.env` 或 Pod 建立位置 |
| `--search TEXT` | 預設不篩選 | 在 GPU 顯示名稱或完整 `gpuId` 中作不區分大小寫的子字串搜尋，可和機房篩選合用 |
| `--output table`、`--output json` | `table` | 表格提供 VRAM、Secure／Community 每小時美元價格、完整 `gpuId` 與機房庫存；JSON 保留符合條件 GPU 的 API 欄位，包含其他機房資料 |
| `-h`、`--help` | 無值 flag | 顯示 GPU 查詢的參數說明 |

`--` 表示 API 未回報資訊；`NONE` 表示該機房回報無庫存，這些 GPU **仍會出現在結果**。
查詢結果不是可建立 Pod 的保證或容量預約；篩選後的表格庫存欄才是指定機房的庫存。
掛載既有 network volume 的 Pod 必須使用該 volume 所在機房；查詢其他機房不會讓
既有 volume 跨區掛載，也不會重新建立 volume。請把完整 `gpuId` 傳給下列建立命令，
不要把表格中可能被截短的 GPU 顯示名稱當成 ID。

##### 選擇資料組別與成對實驗

先完成所需資料的 CPU prepare，以及該組兩個任務的 baseline。現行成對入口每個
「組別 × seed」只建立一台 GPU Pod，在其中依序完成兩套獨立模型。它不是 DDP，
不是兩種 loss 合併更新，也不要求使用者分開操作兩個子模型。

| 實驗名稱 | 訓練資料起點 | 組內兩份設定檔（`configs/experiments/`） |
| --- | --- | --- |
| `a-signal64` | 2021-01-01 | `a_alpha_point64.yaml`、`a_absolute_direction64.yaml` |
| `b-signal64` | 2016-01-01 | `b_alpha_point64.yaml`、`b_absolute_direction64.yaml` |

資料終點皆為 2026-06-01（不含）；A/B 共用相同 validation/test 來源及有效樣本規則。
`--experiment` 保留 configure 的 profile、universe、revision 與 h_start。
一個父 run ID 對應 `alpha_point`、`absolute_direction` 兩個子 run；兩者各自從相同
pretrained Kronos-base 初始化，LoRA rank 64／alpha 128，不共用可訓練參數或 optimizer。
先完成 alpha 模型的訓練／test，再釋放其程序資源並執行方向模型；一個模型 early stop
不代表另一個模型完成，guard 不能因第一個模型完成而終止整台 Pod。

Task／LoRA LR 為 `3e-5`／`5e-6`，warmup 為 256,000 次樣本呈現。
Training 保留年度加權動態 sampling：`0.8 ** (最新訓練年份 − cutoff 年份)`；
年度配額按有效 window 數乘權重正規化，記錄配額、loss 與樣本數。
Validation/test 完整循序遍歷。兩個模型分別以完整 validation 的 normalized Smooth L1／CE
選 checkpoint 及 early stop，不做區間校準。`FORECAST_METRIC_WORKERS` 可限制診斷執行緒
（預設上限 4，另受 CPU／記憶體限制）。

請明確指定成對實驗名稱；省略 `--experiment` 仍沿用 active selection，不會自動切換架構。
歷史容量／V1／V2／V3 設定與進階單模型入口放在[文末附錄](#historical-experiments-zh)，
不作為一般成對訓練步驟。選擇已同步的組別、seed 或配對本身不重做 CPU prepare／baseline，
也不修改資料與模型的數值契約。

##### 在本機建立單台或多台訓練 Pod

首次部署或修改程式／YAML 後，上傳一次全部設定；同一 volume 尚有 Pod 時不可覆寫共用
程式或環境。只切換已上傳的成對實驗與 seed 不需再次同步：

```bash
bash scripts/runpod_workflow.sh sync --dry-run
bash scripts/runpod_workflow.sh sync --apply
```

單一 A 組、seed 42，一台 Pod 執行兩個模型：

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-signal64 \
  --seed 42 \
  --maxRuntime 72h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

同組三個 seed，建立三台 Pod，每台都有 alpha 與方向兩個模型：

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-signal64 \
  --seed 42 --seed 43 --seed 44 \
  --launchWorkers 2 \
  --maxRuntime 72h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

這是三個父 run、六個子模型，**不是六台 Pod**。改為 `b-signal64` 即為 B 組。
同時指定 A、B 與三個 seed 才是六台 Pod。每個 `--seed` 只帶一個整數；GPU 型號不同時，
分開執行三次 `train`，各自選一個 seed 與 GPU。每個組合只帶自己的不可變 selection、
pair manifest、run ID、log 與 guard；之後 configure 不改變已建立的實驗。

| 參數 | 預設 | 用法與限制 |
| --- | --- | --- |
| `--experiment NAME` | 沿用 active selection | 現行入口 `a-signal64`／`b-signal64`；可重複不同名稱，與 seed 建立笛卡兒組合；含進階／歷史名稱至多 22 種，每次至多 30 個組合 |
| `--seed N` | 所選 YAML／run 的 training seed | 可重複不同整數，0–4294967295；每個組別／seed 一台 Pod，組內兩個模型同 seed，各自獨立 RNG／權重 |
| `--launchWorkers N`／`--launch-workers N` | `2` | 本機建立請求的 worker 上限，1–6；不是模型並行數、GPU 數或 DataLoader workers |
| `--maxRuntime DURATION`／`--max-runtime DURATION` | `12h` | 每台 Pod 的共同停止請求期限，涵蓋兩個模型與評估，不是每個子模型各獲一次 runtime；正整數加 `m`／`h`／`d` |
| `--gpuId GPU_ID`／`--gpu-id GPU_ID` | `NVIDIA GeForce RTX 5090` | 這批 Pod 使用同一型號，完整 ID 加引號；不同型號分開建立 |

Seed 不修改 YAML、active selection、完整評估集合或尺度校準 seed；相同 data/task baseline
可跨 seed 重用。Resume 保留原 seed，不接受換 seed；不同 GPU 的非確定性運算仍可能使同 seed
有細微數值差異。建立前檢查兩個任務的完整 baseline、資料與程式；任一缺失就拒絕付費建立。

若缺貨／API 錯誤使部分 Pod 建立失敗，輸出各組合結果並回傳非零；成功 Pod 保留自己的 guard，
不被一起刪除、不自動重租。只重試失敗的組合。`--launchWorkers 1` 僅限制建立請求，
不會讓已租用的 Pod 排隊。費用按每台 Pod 累加。建立輸出提供父 run ID、兩個子 run ID、
seed、Pod ID 與 guard log；後續通常只需保存父 run ID。

GPU 建立請求遭拒時，只保存本機建立嘗試紀錄，不發布遠端 run。Pod 建立成功且
本機 guard 啟動後，才發布遠端 run metadata；因此缺貨重試不會新增未啟動的實驗清單項目。

##### 在每台 Pod 內啟動訓練

建立 Pod 不會自動開始訓練。從 Console SSH 登入各台後只啟動一次：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-train
```

`stage1-train` 是既有 workflow 名稱；成對 manifest 決定組內兩個 Stage 2 模型，
不需再次 configure 或分別啟動子模型。查看該台 log：

```bash
tmux -L stock-forecasting-train attach -t stock-forecasting-train
```

SSH 斷線不停止 detached tmux。同台依序執行兩套獨立訓練／完整 test，不同父 run 才在
不同 Pod 並行；每個模型各自調校 batch、worker、prefetch。模型自己的 checkpoints、
evaluation、loss 與 RNG 分開保存，共用唯讀行情、pretrained cache 與已完成 baseline。
同一父 run 不可由兩台 Pod 重疊處理；cache 建置使用既有協調機制，不複製整個 window dataset。

只有兩個模型都成功完成訓練與 test，成對工作才算 COMPLETE；失敗／逾時依父 run guard
安全收尾，接續時跳過已完成模型。共用 runtime 到期可能只完成第一個模型，不得把它當成
整組成功。CPU prepare、baseline、同步與共用環境更新仍不得和讀取同一 volume 的主模型重疊。

<a id="run-status-zh"></a>

##### 查詢 run ID、執行設定與完成狀態

以下指令在**本機專案目錄**執行，直接使用 `.env` 的 network volume 設定；
不需要先啟動 Pod，也不需要提供 volume ID 或檔案路徑。先列出最近的 runs：

```bash
bash scripts/runpod_workflow.sh runs
```

成對實驗預設只列一筆父 run，並列出兩個子模型 ID 與狀態，不把它們重複列成三個實驗。
`status <父 RUN_ID>` 顯示兩套模型的詳細紀錄；也可直接指定子 ID。每筆顯示實驗名稱、
training／最終 evaluation／整體狀態、run 初始化時間、
最後活動時間，以及當時的 configure 設定：stage、data profile、dataset revision、
資料起訖日期（end exclusive）、h-start、feature mode、universe、symbol limit、
股票／ETF 清單與 YAML 路徑。設定來自**該 run 自己保存的 selection／manifest**，
不是目前的 active selection；後來切換 configure 不會改變過去 run 的顯示。
`symbol-limit=none` 表示未設上限；`stocks=-`／`etfs=-` 表示未指定個別清單，
不是該市場沒有產品，實際範圍仍由 universe 與 data profile 決定。

查詢特定參數實驗，或僅列出完整完成的 runs：

```bash
bash scripts/runpod_workflow.sh runs --experiment a-signal64
```

同一實驗的 seed 查詢；`runs` 與 `status RUN_ID` 會顯示保存的 training seed，
歷史紀錄未保存時顯示 `unknown`，不以目前 YAML 補值：

```bash
bash scripts/runpod_workflow.sh runs --experiment a-signal64 --seed 43
```

```bash
bash scripts/runpod_workflow.sh runs --state complete
```

從列表複製 run ID 後，查詢該 run 是否完成及詳細時間、Pod／launch ID、LoRA 記錄：

```bash
bash scripts/runpod_workflow.sh status "<RUN_ID>"
```

成對 `COMPLETE` 必須同時核對兩個子模型的任務、training 完成證據與 `split=test` 最終
報告；只有父 lifecycle 寫 ready、只有其中一個模型完成，或錯用另一任務產物都不算完成。
單模型歷史 run 仍使用自己的 training 與最終報告判定。
`TRAINED` 只代表訓練完成，尚未完成最終評估；`TRAINING`／`EVALUATING` 表示紀錄中的
進行階段；`INCOMPLETE` 表示中斷、失敗或完成紀錄所需的結果缺失；`NOT_STARTED`
表示只有建立前的 selection，沒有訓練開始證據；`UNKNOWN`／`ERROR` 不會算成功。
清單的 `split=test` 才是 holdout；早期 run 若最後評估的是 validation，會如實顯示
`split=validation`，不會稱為 holdout 完成。W&B 同步狀態另外列出，不會混同模型完成狀態。

這是持久化結果／lifecycle 的唯讀查詢，不是即時 GPU 或 Pod 存活探測。
成對父 `Initialized` 是 pair manifest 建立時間；子模型則是訓練程式建立 run manifest 的時間。
兩者都不是 configure 時間，也不等同每次
resume 的 GPU 開始運算時間；`Last activity` 是最近一筆保存的狀態／結果時間。
詳細查詢會分別顯示 training 完成與 evaluation 開始／完成時間；未記錄的欄位明示
`unknown`／`unrecorded`，不以現在的設定補值。若停止期限與完成寫入競態造成舊 lifecycle
仍是 `timed_out`，會同時保留原狀態並以該 run 的已保存完成結果判定，不只看一個標籤。

| 參數 | 指令／預設 | 說明 |
| --- | --- | --- |
| `RUN_ID`／`--run-id RUN_ID` | `status`；兩者擇一 | 指定 run 詳情；不要填 Pod ID 或 baseline ID |
| `--experiment NAME` | `runs`；不篩選 | 依保存的實驗名稱精確篩選，如 `a-signal64`、`b-signal64`；歷史未記錄名稱的 run 仍可在不篩選的清單查到 |
| `--seed N` | `runs`；不篩選 | 依保存的 training seed 篩選，範圍 0–4294967295；可與 `--experiment`、`--state` 併用 |
| `--state STATE` | `runs`；不篩選 | `complete`、`trained`、`training`、`evaluating`、`incomplete`、`not_started`、`unknown`、`error` |
| `--limit N` | `runs`；`20` | 每頁 1–200 筆，以 run 配發時間由新到舊排列，不依後來的 resume 時間排序 |
| `--offset N` | `runs`；`0` | 跳過 N 筆符合篩選的 run；有更多候選時輸出下一頁 offset，翻頁須保留相同篩選 |
| `--workers N` | `runs`；`2` | 1–8 個並行查詢上限，另依可見 CPU、可用記憶體與 API 上限降低；只讀 metadata，不載入模型權重或 dataset |
| `--output table\|json` | `runs`、`status RUN_ID`；`table` | 預設為按 run 分組的人可讀摘要；`json` 保留結構化設定、時間與狀態，供程式使用 |
| `--timezone ZONE` | `runs`、`status RUN_ID`；`Asia/Taipei` | 人可讀時間的 IANA 時區，如 `UTC`；JSON 保留來源時間與 UTC offset |

`status RUN_ID` 的 exit code：`0`＝完整完成、`1`＝找到但尚未完整完成、`2`＝查無 run、
參數錯誤或讀取／身分錯誤。`runs` 查詢成功為 `0`，清單中有未完成 run 不算查詢失敗；
遇到讀取錯誤會列出錯誤並回傳 `2`，不默默略過。這些查詢不修改 selection 或結果，
不要求為查詢上傳程式，不觸發 CPU prepare、baseline 或訓練。

原本不帶 run ID 的 `status` 仍保留程式、CPU prepare、dataset、baseline、下載進度、
run 清單與 W&B 總覽：

```bash
bash scripts/runpod_workflow.sh status
```

總覽的單一階段 `ready` 不等於整個 run 完成；請用 `status RUN_ID` 查看該 run 的判定。
整組 baseline 是否完成用 `bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64`
（B 組改為 `b-signal64`）；不帶 experiment 的舊查詢僅核對 active selection 的任務。

##### 時間限制與 guard 恢復

`--maxRuntime` 到期時，本機 guard 發出該 Pod、該 run 的停止請求。Pod 完成目前的
validation／checkpoint 保存或 benchmark 模型／seed 的原子寫入，回報安全邊界後
才被終止。**這是停止請求時間，不是保證的費用上限**：長段落可能超時並繼續計費。
Baseline 仍使用其原有硬上限。每台 Pod 都有自己的本機 guard；macOS 以
`caffeinate` 防止控制端睡眠，但關機、斷電或網路中斷仍會影響監控。

控制端恢復後，先唯讀查看，再決定是否套用修復；`--pod-id` 可只處理指定 Pod：

```bash
bash scripts/runpod_workflow.sh recover
bash scripts/runpod_workflow.sh recover --apply --pod-id "<POD_ID>"
```

省略 `--pod-id` 會檢查本專案所有 Pod；`--apply` 才允許重新啟動缺失的 guard
或終止已核實完成的 Pod。恢復不重置原本期限；狀態不明、查詢失敗或 lifecycle
不符合 Pod/run 身分時，不會把它當成已完成。SSH/tmux 畫面不是完成狀態的唯一依據。

若已在 RunPod Console 手動刪除 Pod，可在本機清理殘留的 guard、GPU 時脈監視器
及該 guard 的 `caffeinate`。先預覽，再明確套用：

```bash
bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>"
```

```bash
bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>" --apply
```

此命令先以既有 API 唯讀確認指定 Pod 已不存在；Pod 仍存在、API 查詢失敗、PID 重用
或程序身分無法核對時拒絕清理。省略 `--apply` 不送出 signal；帶 `--apply` 才對核對過的
本機程序送 `SIGTERM`。保留監控與 guard 日誌，不關閉共用 SSH socket、不操作遠端 Pod，
不影響其他 Pod 的監控。`--pod-id` 必填；`--guard-dir PATH` 選填，預設
`RUNPOD_GUARD_LOG_DIR` 或 `~/.local/state/runpod-guards`；`-h`／`--help` 顯示用法。
這不同於 `recover --apply`：後者會恢復 guard 或終止已確認完成的 Pod，不是清理孤兒程序。

##### 本機 GPU 降頻監控與通知

新建或重新啟用訓練、baseline 或 validation 的 GPU guard 時，會自動啟動獨立的
本機 GPU 監視器；CPU preparation 不啟用它。不需額外監控終端或 agent SSH socket。
監視器使用自己的 SSH 連線，
唯讀執行 `nvidia-smi -q -x`，預設每 30 秒記錄 GPU UUID、SM／graphics／memory 時脈、
負載、溫度、功耗／上限、降頻原因與累計計時。它不修改 GPU 設定、不停止訓練，
也不建立／刪除 Pod；連線失敗不影響原 guard。更新程式不會替既有 guard 自動補上
監控或重設期限。SSH 預設使用 `~/.ssh/id_ed25519_runpod`；加密金鑰先用 `ssh-add`
解鎖。監視器不保存密碼、不轉送 SSH agent，也不延長／關閉 agent socket。

判讀參考 [NVIDIA Clock Event Reasons／Counters](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：
溫度／硬體供電降頻旗標或**本次監控期間新增**的相關計時立即記錄，歷史累計值不算。
`SW Power Cap` 單獨出現為 `power_limited`，不是故障判定。降頻推播採最近五分鐘滾動窗口：
時脈**低於同 GPU 正常觀測基準的 80%** 的有效高負載時間**嚴格超過窗口的一半**，
才提示 `suspected_low_clock`；不要求連續低時脈，恰好 80% 或恰好一半都不觸發。
基準為同 GPU 無異常高負載觀測的 SM 時脈中位數，不是規格表的最大 Boost 時脈。
窗口須走完且至少一半窗口有有效高負載資料；依觀測間隔秒數加權，不以採樣筆數代替時間。
分母固定為整個窗口（預設 300 秒），低時脈須累計超過 150 秒；閒置／低負載不算
低時脈時間，也不縮小分母。資料缺口重置窗口；這是抽樣估計，不是連續硬體 trace。
偶發降頻旗標或低功耗上限仍記錄在 `gpu-status`，不單獨推送降頻通知。
這些是工程預設，不是 NVIDIA 故障門檻。功耗上限低於裝置回報預設值的 90% 另列設定警示。
`P2`、低於最大時脈、閒置／checkpoint 保存期間降頻不單獨判異常。
必要欄位缺失為 `limited_telemetry`，持續兩分鐘會記錄無法完整監控；未建立正常基準時，
不能排除從啟動就存在的不明低時脈。SSH 啟動寬限內為 `waiting_for_ssh`，逾時為
`unavailable`，不是正常。

```bash
bash scripts/runpod_workflow.sh gpu-status
bash scripts/runpod_workflow.sh gpu-status --pod-id "<POD_ID>"
bash scripts/runpod_workflow.sh gpu-status --run-id "<RUN_ID>" --output json
bash scripts/runpod_workflow.sh gpu-notify-test
```

`gpu-status` 只讀本機紀錄，不呼叫雲端 API；`--pod-id`、`--run-id` 預設不篩選，
可同時使用。`--output` 為 `table`（預設）或 `json`；`--guard-dir PATH` 預設
`RUNPOD_GUARD_LOG_DIR` 或 `~/.local/state/runpod-guards`；`-h`／`--help` 顯示說明。
`gpu-notify-test` 無參數，送出明確標為測試的 macOS 通知。`status` 無參數的原有
總覽保留並附上 GPU 狀態；不改變 `status RUN_ID` 的訓練／最終評估完成判定。

通知包含 Pod、run、GPU、正常時脈基準與低時脈時間占比；同事件預設 15 分鐘冷卻，
只有先前已推播的降頻事件，確認恢復後才再通知。SSH 監控無法取得資料／連線恢復的
通知獨立保留，不將監控故障誤報為 GPU 降頻。
`submitted` 只代表已交給系統，不保證在專注模式下顯示。紀錄保存於 guard log 旁的
`<pod-id>.gpu-events.jsonl`（5 MiB、兩份輪替備份）及 `<pod-id>.gpu-monitor.json`。
`stale`／`stopped` 不能解讀成 GPU 正常。本機仍需開機、連網；非 macOS 保留
紀錄及 CLI 查詢。每個 Pod 使用獨立、有逾時的監控程序，不共用 GPU 判定狀態。

進階設定於啟動 workflow 前以環境變數傳入，不修改模型 YAML 或資料／baseline 身分：

| 環境變數 | 預設／允許值 |
| --- | --- |
| `RUNPOD_GPU_MONITOR` | `1` 啟用；`0` 只停用 GPU 監控，不停用原 guard |
| `RUNPOD_GPU_MONITOR_SSH_KEY` | `~/.ssh/id_ed25519_runpod`；可指定替代私鑰檔案路徑 |
| `RUNPOD_GPU_MONITOR_NOTIFICATIONS` | `auto`（macOS 通知）；`0` 停用；`1` 要求通知，非 macOS 回報不支援 |
| `RUNPOD_GPU_MONITOR_POLL_SECONDS` | `30`；10–300 秒 |
| `RUNPOD_GPU_MONITOR_CONFIRM_SECONDS` | `120`；30–3600 秒，資料缺失／斷線／恢復確認 |
| `RUNPOD_GPU_MONITOR_CLOCK_WINDOW_SECONDS` | `300`；60–3600 秒，低時脈時間占比的滾動窗口 |
| `RUNPOD_GPU_MONITOR_COOLDOWN_SECONDS` | `900`；60–86400 秒 |
| `RUNPOD_GPU_MONITOR_STARTUP_GRACE_SECONDS` | `300`；0–1800 秒 |
| `RUNPOD_GPU_MONITOR_HIGH_UTILIZATION_PCT` | `80`；10–100 |
| `RUNPOD_GPU_MONITOR_CLOCK_DROP_FRACTION` | `0.20`；0.05–0.8；低於 0.20 仍以 80% 時脈為推播上限，更大值要求更深跌幅 |
| `RUNPOD_GPU_MONITOR_QUERY_TIMEOUT_SECONDS` | `15`；3–30 秒，逾時只中止監視器自己的查詢 |

##### Loss 記錄與 W&B 補傳

每個 run 的 `metrics.jsonl`、`summary.json` 與 W&B 保存 training loss、各學習率、
已處理樣本數與完整 validation 指標。獨立點估計另列 `train/normalized_smooth_l1`，
獨立方向另列 `train/cross_entropy`。歷史 pinball／ranking 日誌解讀見[附錄](#legacy-results-zh)。
`loss_log_points_per_epoch`
預設 250，另記錄 validation 邊界；`evaluations_per_epoch: 5` 對應每 epoch 的
20%／40%／60%／80%／100%。W&B 使用 `trainer/global_step` 與
`benchmark_validation/global_step`，loss 和 validation 的同 step 紀錄不互相覆蓋。

`status` 會列出各 run 的 W&B component；`online_finished`／`synced` 才代表送達，
`offline_pending`、`sync_failed` 或 workflow 結束後的 `online_running` 需要補傳。
線上初始化失敗可保存 offline transaction，不必丟棄已完成的模型結果。

補傳以單模型為單位：先從父 run 的 `status` 取得子 `MODEL_RUN_ID`，使用
`resume <MODEL_RUN_ID>`（training 未完成）或 `validate <MODEL_RUN_ID>`
建立該模型的專用 Pod，不以父 ID 同步兩個模型；**不要啟動 training/validation tmux**，SSH 登入後改執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_wandb_sync.sh "<MODEL_RUN_ID>"
```

新建的 run-scoped Pod 只允許補傳自身 run；省略參數也只處理自身 run，不掃描其他
正在訓練的實驗。腳本保留所有接續／重評估 transaction，使用專案固定 W&B 版本的
legacy sync 路徑；完成後終止補傳 Pod，不在其內再啟動訓練。

<a id="resume-zh"></a>

##### 接續訓練與獨立 holdout 驗證

先確認原 run 的 Pod 已終止。接續訓練直接指定原 run，不需要切換 active selection：

```bash
bash scripts/runpod_workflow.sh resume \
  --maxRuntime 12h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  "<RUN_ID>"
```

省略 run ID 時，只有唯一一個未完成且可接續的 run 才自動選用；若存在多個候選，
會列出 ID 並拒絕猜測。指令讀取該 run 保存的 selection/config，建立 Pod 前驗證
checkpoint 與數值訓練契約。優先使用有效且更新的 temporary checkpoint，否則選擇
retained checkpoints 中 global step 最大者，而非 validation 分數最佳者。
成對 resume 會略過已完成訓練與 test 的模型，恢復未完成模型；只有訓練完成而 test 未完成時，
只補該模型的評估；兩者皆完成時拒絕 `resume`，不租用 Pod。
歷史單模型已完成 training 時仍使用 `validate`。

新 Pod 沿用 run ID，還原權重、optimizer、scheduler、RNG、進度與 early-stop 狀態；
SSH 後仍執行 `bash scripts/runpod_tmux_launch.sh stage1-train`。切換容量不是 resume；
改變模型、資料或數值訓練設定需開新 run。只改選擇器或顯示／路徑控制，不應使既有
資料或 baseline 失效。

若 training 已完成，但要接續或重跑 holdout：

```bash
bash scripts/runpod_workflow.sh validate \
  --maxRuntime 8h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  "<RUN_ID>"
```

`validate` 省略 run ID 會選最新完成 training 的 run；省略 runtime/GPU 時分別為
`12h`／`NVIDIA GeForce RTX 5090`。`--resume`（預設）沿用已完成的 benchmark
工作；`--no-resume` 停用續評估；`--force` 強制重算主模型結果並停用續評估。
baseline 仍讀取已建好的結果，不因此重新訓練。SSH 後執行：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-validate
```

父 run 的 `validate` 在同一台 Pod 處理尚未完成評估的子模型，各自驗證原 checkpoint；不重新訓練。
若兩份 test 已完成且未指定 `--force`，直接在本機跳過、不開 Pod；`--force` 才重算兩份結果。
不同 run 的 validation 可以並行，但不能和同一父 run 或任一子模型的 training/validation 重疊。

<a id="download-zh"></a>

#### 6. 下載 checkpoint、loss 與最終評估結果

本機下載不需要 volume ID 或遠端路徑。多實驗比較建議明確指定每次建立時回報的 run ID：

```bash
bash scripts/runpod_workflow.sh download "<RUN_ID>"
bash scripts/runpod_workflow.sh download --checkpointScope best "<RUN_ID>"
bash scripts/runpod_workflow.sh download --resume --checkpointScope all "<RUN_ID>"
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `RUN_ID` | 省略時選最新已完成結果的 run | 根據 run-scoped terminal 記錄的時間選擇，不使用「最後啟動 Pod」推測 |
| `--checkpointScope all\|best`／`--checkpoint-scope all\|best` | `all` | `all` 為 leaderboard 仍保留的 best-5 集合；`best` 為 validation-selected best |
| `--resume` | 不啟用 | 目標目錄已存在時必須帶此 flag，填補／更新已知檔案，不刪除其他本機 checkpoint |
| `-h`／`--help` | 無 | 顯示下載用法 |

指定父 run 時，一個下載命令取得兩個模型；只有兩份正式 test 結果齊全才算成對下載成功。
父目錄 `artifacts/runpod/<父_RUN_ID>/` 保存 `pair.json` 與 `download-result.json`；
兩個模型的權重各只保存一份，放在 `artifacts/runpod/<子_RUN_ID>/`，由父結果以相對路徑索引。
`--resume` 同時補齊兩個子模型，`--checkpointScope` 分別作用於兩份 leaderboard。
下載 worker 預設最多 2，另依 CPU／可用記憶體限制；`RUNPOD_RESULT_DOWNLOAD_WORKERS=1|2`
可降低 I/O 並行度，不把權重載入 Python RAM。明確指定單模型 ID 仍保留原下載功能。

每個模型的目錄包括 run manifest、resolved config、
leaderboard、所選 checkpoint、`completion-result/`、`metrics.jsonl`、`summary.json`、
training/validation lifecycle、training completion 與最終 `validation-benchmark.json`。
歷史啟用區間校準時另含 `interval-calibration.json`。不會重建已被 retention 政策刪除的
checkpoint；`--resume` 也不會把已下載的 `all` 自動裁切成 `best`。

Checkpoint 內的 validation metrics 用於選模；最終 `validation-benchmark.json`
是 holdout 與 baseline 比較。比較前核對 run ID、設定、有效評估集合與完成狀態。
尺度表徵診斷另用 [下載診斷結果](#在本機下載診斷結果)中的 `download-probes`，不包含在
這個下載命令內。

<a id="artifacts-zh"></a>

### 訓練與推論產物

`download` 會一併下載 `metrics.jsonl` 與 `summary.json`；啟用區間校準的 run 亦下載
`interval-calibration.json`。新訓練缺少必需檔案時會明確報錯；歷史 run 未曾產生的
日誌或未啟用的校準檔不會被要求存在。每筆 loss 紀錄包含
run/session ID、時間、optimizer step、累計呈現樣本數與 epoch fraction；接續訓練會
append 新 session，不覆寫舊曲線。獨立任務的 `train/loss`、`validation/loss` 分別是
normalized Smooth L1 或 CE，另以 `train/normalized_smooth_l1`／`train/cross_entropy`
標出實際目標，沒有假 pinball。歷史分位數與多任務日誌見[附錄](#legacy-results-zh)。
loss 日誌依 `loss_log_points_per_epoch` 分段平均記錄（預設每 epoch 250 點），並涵蓋
每次 validation 邊界；每次寫入都 flush/fsync，W&B 停用時仍保留在 network volume。

父 run 的 pair manifest 保存組別、seed、兩個任務及各自的不可變 selection；不是第三個模型。
以下權重與訓練產物各由兩個子模型獨立保存：

- `adapter.safetensors`：LoRA、resampler、benchmark conditioner、該任務的輸出 head
  與該實驗解凍的 Kronos 權重。
- `resolved-config.yaml`
- `trainer-state.json`
- optimizer / scheduler state
- run manifest、checkpoint leaderboard 與 best-checkpoint pointer
- `completion-result/`：正常跑完或 early stop 當下的最終可訓練權重、停止原因與稽核計數
- selection ID/SHA、dataset request SHA、stage config SHA 與 requested dataset contract
- dataset manifest 摘要、architecture digest、Kronos source/model/tokenizer
  revisions 與 bounded training implementation digest
- 訓練選模 validation metrics、最終 holdout 與 baseline 比較
- 尺度分支啟用時的 `runtime_scale_features` schema、train-only normalization 與指紋

Checkpoint resume 必須符合該 run 的 task/output schema（舊分位數 `5.0`、獨立任務 `signal-1.0`），不能跨任務接續，並且必須通過 RunPod run
identity、artifact integrity、validation-selection、資料、模型與訓練程式碼契約檢查；
任何不相容的 schema 都會 fail closed。

唯讀推論與 `probe-scales` 允許舊比例切分、無新分支的 checkpoint 在原 resolved config、
資料與 artifact hashes 一致時使用歷史實作相容路徑；此例外不適用於 training resume，
也不允許拿舊 checkpoint 配上新固定日期 config 或把舊報告當作新 holdout 結果。

低階推論使用子 `MODEL_RUN_ID`（可由父 run 的 status 查到），父 ID 本身沒有模型權重。
推論也必須在已建立專案 Poetry environment 且掛載相同 network volume 的 RunPod Pod
內執行，不在本機載入 checkpoint。替換 run／checkpoint 識別碼，並將 `<MARKET_PARQUET>`
換成含目標商品及 benchmark 的實際輸入檔；不依賴 SSH session 中未必存在的 `DATA_ROOT`：

```bash
poetry run stock-forecasting-infer \
  --config "/runpod-volume/savedModel/<MODEL_RUN_ID>/<CHECKPOINT>/resolved-config.yaml" \
  --checkpoint "/runpod-volume/savedModel/<MODEL_RUN_ID>/<CHECKPOINT>" \
  --input "<MARKET_PARQUET>" \
  --symbol AAPL.US
```

所有推論包含 `forecast`、資料 provenance、encoder shape 與 checkpoint metadata。
獨立點估計的 `forecast.by_horizon` 只有 `alpha_point`；獨立方向則只有
`absolute_direction_probabilities`，並明列 `down, neutral, up` 順序及含端點的 neutral band。
兩者不回傳 ranking、五級交易訊號、成本情境或校準分位數。
<a id="probes-zh"></a>

### 歷史尺度表徵診斷

`probe-scales` 讀取既有 checkpoint，固定 Kronos / LoRA / resampler / conditioner / head
權重，使用獨立 ridge 線性探針檢查各層是否仍可讀出歷史尺度。它不是重新訓練 forecast
模型，也不會新增尺度特徵分支。此功能沿用 checkpoint 原有的 train / validation
資料切分，不修改日期邊界，不建立 test loader，不計算未來 alpha 標籤。

`probe-scales` **不會自動建立 GPU Pod，也不能在本機執行模型診斷**。執行順序如下：

1. **本機控制端：確認程式就緒。** 已同步且未修改 source 時不用再同步；需更新時，
   等該 volume 無任何 Pod 後才執行同步。 透過 `bash scripts/runpod_workflow.sh sync --apply` 與既有雲端部署流程
   更新程式碼；GPU readiness 必須通過，原始 network volume 中須已有可用的專案環境。
   本機與 volume 必須同時更新至支援診斷 lifecycle 的版本，再建立 Pod 與本機 guard；
   更新檔案不會替已在運行的舊 guard 加上新的監控項目。
2. **本機控制端：建立 GPU Pod。** 若已有掛載同一 volume、且未執行訓練或 validation 的 GPU Pod，可直接使用。
   沿用既有 Pod 時，須確認其本機 guard 已支援診斷 lifecycle 且仍在運行。
   若沒有，使用[進階單模型操作](#single-model-zh)的專用診斷 Pod 建立命令，
   不要為單模型診斷建立成對訓練 Pod。它仍須通過資料、模型與 baseline readiness，
   配置 run identity 與 guard，但不會自動啟動訓練；診斷仍由下一步的 tmux 命令啟動。
3. **GPU Pod：透過 tmux 啟動診斷。** SSH 登入該 Pod，執行下列診斷命令。
   **不要執行**建立 Pod 後提示的
   `runpod_tmux_launch.sh stage1-train`，也不要啟動 `stage1-validate`。

在 GPU Pod 內，不指定 checkpoint 時，自動使用目前掛載 volume 中最新已完成訓練的
run，並選取該 run 的 validation-selected best checkpoint：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh probe-scales
```

指定模型時，`--checkpoint` 填入子模型／歷史單模型的 **run ID**，不是父 ID，也不需完整路徑：

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh probe-scales \
  --checkpoint "<RUN_ID>" \
  --train-samples 16384 \
  --validation-samples 4096 \
  --batch-size 16 \
  --ridge-alpha 10 \
  --seed 42
```

「最新完成」依各 run 已正式發布的 `completion-result/training-result.json` 中
`created_at` 判定，包含正常完成與 early stopping；不依目錄修改時間、run ID 的字串
順序、目前 active selection 或最高 checkpoint step 判定。沒有完成紀錄的中途訓練不會
被自動選中；完成時間相同時以 run ID 作固定排序。選取最新 run 後，會驗證其
`best-checkpoint.json` 與保留的 checkpoint；若資料損壞、缺少最佳 checkpoint 或存在
未完成的 selection transaction，就明確失敗，不會悄悄改用較舊模型。
完成紀錄本身格式不合法時也會停止掃描；可明確指定其他 run，或先處理損壞的產物。

`--checkpoint RUN_ID` 使用該 run 經完整性驗證的最佳 checkpoint；明確指定 run 不要求
訓練已完成，但 checkpoint 必須已完整提交且 GPU lease 可取得。為相容既有命令，仍接受
完整的 `/runpod-volume/savedModel/<run-id>` 或其 `checkpoint-NNNNNN` 絕對路徑；
不接受相對路徑、路徑跳脫或 symlink。
設定一律取自所選 checkpoint 的 `resolved-config.yaml`，
不採用目前 active Stage 設定。原 checkpoint 綁定的 bar store、dataset manifest 與
model/tokenizer cache 必須仍存在且契約一致；不能直接改指向另一個資料版本。
若 run 留有未完成的 checkpoint-selection transaction，診斷會先拒絕執行；須由原訓練
流程完成復原，不會在唯讀診斷中觸發 checkpoint 自動修復或刪除。

本機使用 `runpod_workflow.sh` 建立 Pod；Pod 內的診斷一律使用
`runpod_tmux_launch.sh probe-scales` 啟動，與前述部署及訓練流程一致。
此命令會啟動 `stock-forecasting-probe-scales` 背景 session。看到 `Detached tmux session started`
後，即可中斷 SSH，不需要保持終端連線。

需要查看即時輸出時，在 Pod 仍運行期間重新 SSH 登入後 attach；按 `Ctrl-b d` 可離開畫面而
不中止工作，不要按 `Ctrl-c` 當作 detach：

```bash
tmux -L stock-forecasting-probe-scales attach -t stock-forecasting-probe-scales
```

診斷沿用既有的 timeout 與**本機監控終止**流程：runner 取得排他 GPU lease 後，先發布
獨立的 running 標記；成功、失敗或逾時後，先保存持久化 log / status，再發布診斷終態。
本機 `terminate_runpod_after.sh` 透過 S3 讀取標記，驗證 Pod ID、Pod 建立時的 owner run ID、
launch ID 與狀態後，使用本機 `runpodctl_project.sh pod delete` 終止該 Pod。
owner run ID 是本次 Pod 的識別，不是 `--checkpoint` 指定的歷史模型 run ID。
診斷不呼叫 Pod 端終止 API，也不需要將本機 RunPod API key 放入 Pod。
不改寫訓練 / validation 的 lifecycle 完成標記。
若 session 已存在、啟動前檢查不通過，或 runner 無法取得 GPU lease，會保留 Pod，避免
中斷既有工作；這些情況不發布診斷終止訊號，本機 guard 仍按該 Pod 的設定監控。
診斷 runner 自身的 timeout 使用 Pod 建立時的 `MAX_RUNTIME_SECONDS`（上例為 2 小時），
另有 60 秒強制中止寬限。這是診斷程序的上限，不是雲端費用保證；本機 guard 仍須成功
讀取終態並取得 RunPod API 確認，才算 Pod 已終止。
看到 `awaiting Pod termination by the local guard` 表示診斷計算已結束，正等待本機 guard
下一次成功輪詢；runner 在等待期間繼續持有 GPU lease，避免其他工作在終止前插入。
**SSH 可以斷線，但執行 guard 的本機必須保持開機、連網，且 guard 程序不可中止。**

launcher 會印出這次工作的確切路徑。執行狀態與診斷數值報告分開保存：

```text
/runpod-volume/logs/tmux/stock-forecasting-probe-scales/<launch-id>/
  combined.log                 # Worker stdout/stderr and local-guard handoff messages
  status.json                  # Terminal job state: succeeded / failed / timed_out
  runner.sh                    # Quoted arguments, timeout, lease and local-guard handoff

/runpod-volume/lifecycle/diagnostics/representation-scales/<pod-id>.json
                               # running / succeeded / failed / timed_out; Pod and owner identity
```

`status.json` 在工作結束時發布；tmux 啟動成功不代表模型診斷成功，`succeeded` 也不代表
RunPod API 已確認終止。終止請求與重試記錄位於**本機**的
`~/.local/state/runpod-guards/<pod-id>.log`（或建立 Pod 時印出的自訂 Guard log 路徑），
不是 Pod 內的 `pod-shutdown` 目錄；仍須以 RunPod 的 Pod 狀態確認是否已終止。
無法讀取終態時須檢查本機 guard 與 RunPod 狀態，不可假定已停止計費。Pod 終止後不能再 attach，應從持久化
network volume 讀取 log、status 與報告。
可用下列命令查看所有診斷參數：

```bash
bash scripts/runpod_tmux_launch.sh probe-scales --help
```

自動選取 run 的 metadata 掃描採有界 thread pool，
可用 `--selection-workers 1..8` 設定上限；實際數量還會受可見 CPU 與可用記憶體限制，
不載入所有 run 的模型權重。GPU 記憶體不足時可降低 `--batch-size`，抽樣列不受 batch size 影響。
SSH session 缺少 Pod 環境變數時，腳本會使用既有 allowlist PID 1 importer 載入，
不需要手動重設 Stage、volume 或 credential 變數。

診斷內容：

- 四種讀取方式：Kronos 的 asset / benchmark masked mean 串接、兩者最後有效 token
  串接、兩組 resampler latent mean 串接，以及 alpha head 實際使用的 conditioned mean。
- 八個歷史目標：asset、benchmark、兩者差值的 20 / 60 個交易日對數報酬標準差，以及
  asset / benchmark 全輸入視窗的收盤價 `std / mean`。採 as-of 調整後的輸入價格、
  `ddof=0`、不年化；至少需要 61 根有效 K 線。這些尺度不是未來持有期 alpha。
- Train / validation 各自固定種子、無放回均勻抽樣，所有讀取方式共用同一組樣本。
  探針 train 候選為原資料版本的**完整 train split**，不保證等於 Stage 1 模型曾見過的
  5% 子集；不是每檔股票或每個日期等權抽樣。
- 特徵與目標的平均值、標準差只在 train 擬合；固定 ridge alpha，不用 validation 選參。
  同時比較 train 平均值基準與固定種子打亂 train 標籤的 ridge 對照。
- 輸出 train / validation R²、MAE、RMSE、Pearson r、相對 train 平均值的 MSE skill，
  以及各市場 validation 指標、特徵維度、常數特徵數及每個特徵對應的 train 樣本數。

每次執行建立新的獨立目錄：

```text
/runpod-volume/diagnostics/representation-scales/<run-id>/<checkpoint>/probe-<UTC>-<id>/
  status.json                  # Only state=complete establishes a completed diagnostic
  probe.log
  report.json                  # Metrics, settings, data/checkpoint/source SHA-256 provenance
  summary.md                   # Full Traditional Chinese section, then full English section
  samples.jsonl                # Split-local row order, sample IDs, cutoffs, symbols and markets
  probes.npz                   # Train-only scalers, ridge coefficients and shuffle permutation
  validation_predictions.npz   # Historical targets and predictions in validation row order
```

不覆寫 checkpoint、best pointer、既有 validation 報告或完成標記。失敗留下的 partial
結果不可當作完成報告；重跑會建立新目錄。現有 `download` 命令仍只處理原本的訓練 / validation
產物，不會自動下載本診斷目錄。診斷目錄與 tmux 日誌都在持久化 network volume，Pod 終止後
仍保留，可透過該 volume 的 S3 介面取回。

#### 在本機下載診斷結果

在**本機專案根目錄**執行一個命令，即可下載最新完成的歷史尺度表徵診斷：

```bash
bash scripts/runpod_workflow.sh download-probes
```

若要下載指定模型 run 的最新完成診斷，只需加上 `PROBE_RUN_ID`：

```bash
bash scripts/runpod_workflow.sh download-probes "<RUN_ID>"
```

`PROBE_RUN_ID` 是被診斷模型的 run ID，與執行診斷時 `--checkpoint` 使用的 run ID
相同。不指定時搜尋整個專案 volume；指定時只搜尋該 run。兩種用法都只下載最新完成的
一份結果。若曾對同一模型重跑診斷，由腳本自動選取，不需要使用者區分儲存目錄。

腳本自動讀取專案 `.env` 中的 network volume ID、S3 憑證、region 與 endpoint，
自行解析 checkpoint、診斷目錄及本機位置。**不需要輸入 volume ID、路徑或其他識別碼，
也不需要先列出遠端檔案。** Pod 已終止仍可下載，不會建立 Pod、重跑診斷或下載模型權重。

最新結果依 `state: complete` 的診斷報告 `created_at` 判定，不依訓練完成時間或
目錄名稱猜測；執行中與失敗的結果不會被選中。下載時自動檢查七個產物是否完整，
驗證 report / status 與所選結果一致，並核對報告記錄的三個數值產物 SHA-256。
找不到完成結果、缺檔或驗證失敗會明確報錯，不會把不完整檔案當作成功下載。

成果保存於被 Git 排除的 `artifacts/diagnostics/representation-scales/`，
保留各次結果且不互相混合。命令完成後直接印出下載位置與 `summary.md` 路徑；
先閱讀摘要，再查看 `report.json` 的完整指標。中斷後重跑同一命令即可，不需要
`--resume`；驗證成功後只更新該次診斷的已知產物，不刪除其他檔案。
`probe.log` 會隨結果一併下載。

#### 解讀診斷結果

`probes.npz` 以 `<readout>__feature_mean/feature_scale/coef/intercept` 儲存探針。
計算順序為 `Xz = (X - feature_mean) / feature_scale`、
`Yz = Xz @ coef.T + intercept`；前 8 欄為真實標籤探針，後 8 欄為打亂標籤對照，
兩組分別以 `Y = Yz * target_scale + target_mean` 還原尺度。目標欄位順序見
`report.json.target_contract.names`。原始高維表徵不落盤。

解讀時先確認 validation 同時優於平均值與打亂標籤對照。MSE skill 定義為
`1 - MSE_probe / MSE_train_mean_baseline`，正值才代表勝過該基準；R² 的分母則使用
validation 自身平均值，兩者不能混為一談。常數目標的 R²、常數向量的 Pearson r、
零分母的 skill 以 JSON `null` / Markdown `N/A` 表示。Train 高但 validation 低，應先檢查
探針過擬合或分布差異。低分只表示目前 pooling + 線性探針無法讀出，不足以證明資訊消失；
不同層維度不同，分數差不能直接解讀為資訊損失。重疊視窗與共用 benchmark 並非獨立樣本，
本功能不提供顯著性或自動架構裁決；可讀出歷史尺度也不等於能預測未來 alpha。
預設 16,384 / 4,096 筆為可調整的成本上限，不是統計充分性的保證。

已建立專案環境的雲端 Pod 可先執行不下載模型的合成資料／mock checkpoint 契約測試：

```bash
cd /runpod-volume/stock_forecasting
.venv/bin/python -m pytest tests/test_representation_scale_probe.py
```

<a id="cli-reference-zh"></a>

### CLI 參數速查

以下涵蓋本 README 使用的專案命令。`[ARG]` 表示可省略，`<ARG>` 表示須替換的值，
不要把方括號或角括號原樣輸入。除明確列出的旗標外，不要把底層 helper 的參數加到
workflow。`bash scripts/runpod_workflow.sh --help`（或 `-h`、`help`）列出入口；
不是每個 shell 子命令都支援自己的 `--help`。
一般 workflow 自動讀取專案 `.env` 的 volume／credential 設定，不需額外傳 volume ID 或資料路徑。

#### 本機控制命令

下表「命令」均接在 `bash scripts/runpod_workflow.sh` 後面；這些命令不在本機執行模型計算。

| 命令 | 參數、預設值與行為 |
| --- | --- |
| `credentials` | 無 CLI 參數；互動式隱藏輸入憑證 |
| `tpex-relay configure`、`deploy`、`verify`、`status` | 不接受額外 CLI 參數；`configure` 互動設定，`deploy` 部署，`verify` 驗證，`status` 唯讀查詢；`tpex-proxy` 是同一入口的別名 |
| `volume deploy` | `--name NAME` 預設 `stock-forecasting`（1–63 個英數、`.`、`_`、`-`，首字須英數）；`--size-gb N` 預設 `100`，範圍 10–4000；`--datacenter ID` 預設 `EU-RO-1`；`--force-new` 明確建立另一個計費 volume 並改登記它，**不搬移舊資料**。已登記 volume 時預設直接跳過 |
| `configure` | 完整選項與限制見前文「configure 參數與資料範圍」；無參數為互動模式。`-h`／`--help` 顯示底層 parser 說明；`--project-root` 由 workflow 注入，不需使用者提供 |
| `selection show` | 無額外參數；顯示 active selection |
| `sync` | `--dry-run` 預設，只檢查／列出上傳清單；`--apply` 才上傳。二者擇一，不接受其他參數 |
| `cpu prepare` | 無參數或 `--interactive` 開啟互動確認。非互動時 `--max-api-calls N` 必填且為正整數；`--eodhd-qps Q` 預設 `16`、`--taiwan-qps Q` 預設 `0.5`，均須大於零；`--maxRuntime D` 預設 `6h`；`--prepareReserve D` 預設 `auto`（25% runtime，最多 2h），明確值須短於 runtime；`--maxBackoff D` 預設 `1m`；`--cpuNumber N` 預設 `8`，可選 2／4／8／16／32；`--cpuFlavor F` 預設 `cpu3g`，可選 `cpu3c`、`cpu3g`、`cpu3m`、`cpu5c`、`cpu5g`、`cpu5m` |
| `readiness` | 必須擇一：`--code-only` 核對程式上傳；`--gpu` 核對主模型訓練依賴；`--baseline` 同時核對當前資料篩選規則與 baseline 完成結果、不要求主模型 HF cache。`--baseline --experiment a-signal64`（或 `b-signal64`）查詢兩個任務，省略實驗沿用 active selection；`--experiment` 只供 baseline 模式使用。完成為 exit 0、尚未完成／待整理為 1、驗證錯誤為 2 |
| `train`、`baseline` | `--maxRuntime D` 預設 `12h`；`--gpuId ID` 預設 `NVIDIA GeForce RTX 5090`。`train` 接受可重複的 `--experiment NAME`、`--seed N`（0–4294967295；省略沿用 training seed）及 `--launchWorkers N`（1–6，預設 2，別名 `--launch-workers`）；實驗 × seeds 最多 30 個組合。`a-signal64`／`b-signal64` 每組別、每 seed 建立一台 Pod，依序執行兩個獨立模型，runtime 共用。`baseline` 接受單一 `--experiment NAME`，成對入口先在本機查兩個 task cache、只建置未完成部分，不改 active selection；不接受 seed／launchWorkers。省略實驗仍使用 active selection，**不自動變成成對實驗**；進階單模型入口見附錄。兩者無 run ID 位置參數 |
| `resume [RUN_ID]` | 父 ID 接續未完成模型或補齊其 test，略過已完成模型；整對已完成則拒絕、不開 Pod。單模型保留原接續功能。省略 ID 須唯一可接續候選，多個候選拒絕猜測。自動讀取原 selection；`--maxRuntime D` 預設 `12h`、`--gpuId ID` 同上；接續最新可用 checkpoint，不是 best |
| `validate [RUN_ID]` | 省略 ID 選最新完成 training 的 run，使用原 selection。`--maxRuntime D` 預設 `12h`、`--gpuId ID` 同上。`--resume` 預設沿用完成工作；`--no-resume` 停用續評估；`--force` 重算並停用續評估，三者擇一。父 ID 在一 Pod 處理兩模型；兩份 test 已完成且無 `--force` 時本機跳過、不開 Pod；`--force` 重算兩份。不重訓 baseline |
| `runs` | `--experiment NAME`、`--seed N`、`--state STATE`、`--limit N`、`--offset N`、`--workers N`、`--output table\|json`、`--timezone ZONE`；列出 run ID、seed、執行時間與固定設定，詳見[查詢 run](#run-status-zh) |
| `status [RUN_ID]` | 指定 run 時可用 `--run-id RUN_ID` 取代位置參數，另有 `--output table\|json`、`--timezone ZONE`；無參數保留專案總覽，指定 ID 則查該 run 是否完整完成 |
| `cpu-logs` | 無參數；下載 CPU 工作紀錄 |
| `gpu-status` | 只讀本機 GPU 紀錄；`--pod-id ID`、`--run-id ID` 為可合併的可選篩選；`--output table\|json` 預設 `table`；`--guard-dir PATH` 預設 `RUNPOD_GUARD_LOG_DIR` 或 `~/.local/state/runpod-guards`；`-h`／`--help` 顯示說明 |
| `gpu-notify-test` | 無參數；送出明確標示為測試的 macOS 桌面通知，不查詢／操作 Pod |
| `recover` | 預設僅診斷；`--apply` 才恢復 guard／終止已確認可終止的 Pod；`--pod-id ID` 限定一個 Pod；`--confirmations N` 預設 `2`，至少 2；`--confirmation-delay-seconds N` 預設 `5`，不可負數；`--guard-dir PATH` 為進階控制端紀錄位置，預設 `RUNPOD_GUARD_LOG_DIR` 或 `~/.local/state/runpod-guards`；`-h`／`--help` 顯示說明 |
| `cleanup-monitors` | 必填 `--pod-id ID`；預設 dry-run，`--apply` 才終止已核對的本機 guard／GPU monitor／guard-owned caffeinate。先查 API 證實 Pod 不存在；存在、API 失敗或程序身分異常皆拒絕。`--guard-dir PATH` 預設 `RUNPOD_GUARD_LOG_DIR` 或 `~/.local/state/runpod-guards`；`-h`／`--help`。不刪日誌、不關共用 SSH、不操作遠端 |
| `download [RUN_ID]` | 父 ID 一次下載兩模型；省略 ID 選最新完整完成的結果，成對子模型不當成額外實驗自動選用。`--checkpointScope all` 預設下載各自 retained checkpoints，`best` 取各自最佳；`--resume` 補齊／更新兩份本機結果，**不是接續訓練**；`-h`／`--help` 顯示說明。明確子 ID 保留單模型下載 |
| `download-probes [PROBE_RUN_ID]` | 省略 ID 下載最新成功診斷；指定被診斷模型的 training run ID，下載該模型最新成功診斷。不需 checkpoint ID、probe ID、volume ID 或路徑；`-h`／`--help` 顯示說明 |

時間 `D` 只接受正整數加 `m`、`h`、`d`（例如 `30m`、`12h`、`2d`），不接受 `1.5h`。
可用的同義旗標：`--max-runtime` = `--maxRuntime`、`--gpu-id` = `--gpuId`、
`--checkpoint-scope` = `--checkpointScope`；CPU prepare 另接受 `--maxApiCalls`、
`--eodhdQps`、`--taiwanQps`，以及 `--prepare-reserve`、`--max-backoff`、
`--cpu-number`、`--cpu-flavor`。文件範例使用同一組主要拼法。
GPU train／validate 的 runtime 是**安全保存後停止的請求時間，不是費用硬上限**；baseline／CPU
的期限行為見各自操作章節。GPU 查詢的 `--data-center` 不是 train／baseline 的選項。

`bash scripts/verify_runpod_s3_access.sh` 不接受使用者 CLI 選項；它從 `.env` 讀取 volume
與 S3 設定。`bash scripts/runpodctl_project.sh gpu list` 的全部選項見[GPU 資源查詢](#gpu-catalog-zh)。

#### Pod 內工作與診斷

`bash scripts/runpod_tmux_launch.sh WORKFLOW` 的一般 `WORKFLOW` 為 `cpu-prepare`、
`cpu-finalize`、`stage1-train`、`stage1-validate`、`baseline`；這些名稱後面不接受額外參數，
設定由建立 Pod 時的 selection／環境傳入。只有 `probe-scales` 轉交下列診斷參數：

| 參數 | 預設值／限制 | 說明 |
| --- | --- | --- |
| `--checkpoint MODEL_RUN_ID` | 最新完成訓練的單模型 run | 使用其 validation-selected best checkpoint；成對實驗填子模型 ID，不填父 ID |
| `--train-samples N` | `16384`，至少 2 | train 表徵抽樣上限，不會重新訓練 forecast 模型 |
| `--validation-samples N` | `4096`，至少 2 | validation 表徵抽樣上限，不使用 holdout |
| `--batch-size N` | `16`，正整數 | GPU 表徵提取 batch；OOM 時降低 |
| `--num-workers N` | `0`，範圍 0–16 | 診斷 DataLoader 的 worker 數，與 train／test loader 自動調校不同 |
| `--ridge-alpha X` | `10`，有限正數 | ridge 正則化係數 |
| `--seed N` | `42`，整數 0–4294967293 | 固定抽樣與 shuffled-label 對照 |
| `--selection-workers N` | auto，範圍 1–8 | 掃描完成訓練 metadata 的 I/O 上限，另受 CPU／可用記憶體限制 |
| `-h`、`--help` | 無值 flag | 顯示診斷參數，不啟動診斷 |

`bash scripts/runpod_wandb_sync.sh [MODEL_RUN_ID]` 在新建 run-scoped Pod 中只處理自身 run；
省略 ID 也不掃描別的實驗，指定不同 run 或 `--help` 會在取得工作 lease 前拒絕。
只有歷史未分 scope 的 Pod 保留省略 ID 掃描全部的行為；不應用它處理並行中的實驗。
腳本取得 lease 後退出會進入既有終止流程，因此只在專用、閒置補傳 Pod 執行，
完成後仍須用本機狀態確認 Pod 已終止；不可把 help 當成安全的線上檢查命令。

#### 遠端低階資料與推論 CLI

以下為 `poetry run stock-forecasting-*` 的完整選項，只供已準備好環境的雲端 Pod
除錯，不取代本機 workflow；正常流程不要求手填資料路徑或 identity。
這四個 Python CLI 均接受 `-h`／`--help`，但仍須在雲端既有環境執行。

`poetry run stock-forecasting-download`：

| 參數 | 預設值／用途 |
| --- | --- |
| `--profile NAME` | `FIN_TS_DATASET_PROFILE`，否則 `us_tw_eodhd`；可用 `tw_only`、`us_only_eodhd`、`us_tw_eodhd`；parser 另保留尚未實作的 `us_tw_massive`，不可用於正式取得資料 |
| `--start DATE`、`--end DATE` | 必填 `YYYY-MM-DD`，前者 inclusive、後者 exclusive |
| `--symbols SYMBOL ...`、`--etf-symbols SYMBOL ...` | 美股 explicit 清單，以空白分隔；皆省略時 discovery；不篩選台股 |
| `--symbol-limit N` | 預設不限；各取最多 N 檔美股與白名單 ETF，再補 VTI；只作有界驗證 |
| `--interval 1d` | 只接受 `1d` |
| `--output PATH` | 必填，canonical Parquet 目的地，不允許靜默覆寫 |
| `--manifest-root PATH` | 預設由 output 推導 dataset 根目錄（output 位於 `raw/` 時用其上一層），保存下載 manifest |
| `--raw-cache-root PATH` | 預設 `<manifest-root>/api-cache` |
| `--provider-checkpoint-root PATH` | 預設 raw cache 同層的 `provider-checkpoints`；儲存可續傳 provider Parquet |
| `--progress-path PATH` | 預設 `<manifest-root>/download-progress.json` |
| `--cache-revision LABEL` | `RUNPOD_DATASET_REVISION`，否則 `v1`；隔離 provider cache 修訂 |
| `--dataset-request-sha256 HASH`、`--selection-id ID`、`--selection-sha256 HASH`、`--launch-id ID` | workflow provenance；分別預設來自 `RUNPOD_DATASET_REQUEST_SHA256`、`RUNPOD_SELECTION_ID`、`RUNPOD_SELECTION_SHA256`、`RUNPOD_LAUNCH_ID`，未設定則無值；不要自行編造 |
| `--max-api-calls N` | 此低階 CLI 預設 `100000`；只限制 EODHD 單次取得資料的 network attempts，含 retry |
| `--eodhd-qps Q`、`--taiwan-qps Q` | 預設 `16`、`0.5`；每個 provider 的請求節流 |
| `--max-backoff-seconds S` | `RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS`，否則 `60`；注意此處單位為秒，不是 workflow 的 `1m` |
| `--workers N` | `FIN_TS_CPU_WORKERS`，未設定時低階預設 `1`；正式 CPU workflow 會按硬體資源配置，不以此 fallback 作正式並行設定 |
| `--acquisition-deadline-epoch-seconds T`、`--preparation-reserve-seconds S` | 預設無值；Unix 絕對截止時間與保留給資料準備的秒數，由 workflow 配置 |
| `--exclude-delisted` | 預設包含下市股；此旗標排除 EODHD 下市商品 |

`poetry run stock-forecasting-prepare`：

| 參數 | 預設值／用途 |
| --- | --- |
| `--input PATH`、`--output DIR` | 必填，已下載 raw Parquet 與可續傳 bar-store 目錄；不下載行情 |
| `--download-manifest PATH`、`--dataset-manifest PATH` | 預設 dataset 根目錄的 `download-manifest.json`、`dataset-manifest.json`；後者必須直接位於該根目錄且不得覆寫已 ready manifest |
| `--benchmark-mapping PATH` | 選填 JSON mapping；未指定使用預設 benchmark policy，不放寬 ETF 白名單 |
| `--fixed-evaluation` | 無值 flag；啟用 production 的 2025-06／2025-12／2026-06 exclusive 切分；省略為歷史比例模式 |
| `--window-size N` | `128` 個 bars |
| `--h-start N` | `1`；可選 1／2／3，最大 horizon 固定 14 |
| `--max-abs-log-return X` | `0.5`，舊 preparation 候選範圍的極端值標記；production runtime 依 `configs/data_cleaning.json` 重建有效集合，不以此刪除真實極端報酬 |
| `--train-fraction X`、`--validation-fraction X` | `0.70`、`0.15`，僅比例模式使用 |
| `--purge-bars N` | `20`，比例模式的 purge；固定日期模式改用 label end 邊界 |
| `--stride N`、`--sample-stride N` | 相容欄位只接受 `5`、`1`；不是可任意修改的抽樣步距 |
| `--target-horizon N`、`--diagnostic-horizons N ...` | 相容欄位必須為 `5`、`1 20`；實際輸出仍為 h-start 至 14 日 |
| `--embargo-bars N`、`--effective-embargo-bars N` | 相容欄位必須為 `5`、`14`；固定日期模式不疊加比例切分的 embargo |
| `--flat-volatility-multiplier X` | `0.25`，保留於 preparation provenance 的相容欄位，不是分類訓練目標 |
| `--bucket-count N`、`--batch-rows N` | `128`、`1000000`，bar-store 分桶數與讀取分塊上限 |
| `--deadline-epoch-seconds T` | 選填 Unix 絕對時間，於安全邊界暫停以便續傳 |
| `--workers N` | 正整數；`FIN_TS_CPU_WORKERS`，未設定時低階預設 `1`；實際各 phase 依 CPU／記憶體下調上限 |

`poetry run stock-forecasting-infer`：`--config PATH`、`--checkpoint PATH`、`--input PATH`
必填，分別為 resolved config、checkpoint／run 目錄與含商品及 benchmark 的行情檔；
`--symbol SYMBOL` 在多商品輸入時必填；`--as-of TIMESTAMP` 是選填的 inclusive UTC
截止時間，未提供則使用可用資料的最新共同日期；`--output PATH` 選填保存 JSON，省略只輸出至 stdout。
`probe-scales` 的 Python CLI 使用上述相同診斷參數；操作時使用 tmux 入口。

外部工具的範例旗標：`tmux -L SOCKET attach -t SESSION` 中 `-L` 選專用 socket、
`-t` 選 session；不要換成其他工作的 socket。`gcloud auth login` 使用互動登入；
額外選項見 [gcloud auth login 官方參考](https://cloud.google.com/sdk/gcloud/reference/auth/login)。
`.venv/bin/python -m pytest TEST_FILE` 的 `-m` 是 Python 模組執行，`TEST_FILE` 限定測試範圍；
pytest 的 `-k EXPR` 篩選測試、`-q` 簡潔輸出、`-h` 顯示全部第三方選項，只在雲端 Pod 使用。

<a id="acceptance-zh"></a>

### 驗收與結果解讀

1. Training 使用動態 sampling；validation/test 完整、固定順序遍歷清理後的有效集合，
   不遺漏最後一個短 batch，不以估計數量或抽樣代替全量評估。
2. 所有 split 遵守相同的連續 input/output、行情有效性與流動性規則；保留真實極端報酬。
   A/B 評估共用同一份來源，不只核對日期範圍或筆數。
3. 兩個獨立任務各有自己的架構與權重，均從同一 pretrained base 開始。未完成 run 的接續必須還原原架構、optimizer、RNG 與進度。
4. Readiness 核對所選 dataset 自身的 manifest 與實際 artifacts；不要求另一份資料最後寫入的
   共用 CPU marker 在 stage/config 上相同，也不能混用不同 profile、universe 或資料內容。
5. 獨立任務分別只輸出 `[B,15-h_start]` alpha point 或 `[B,15-h_start,3]` 個股三分類；
   固定 ±1% band 含端點為 neutral，不把 alpha up 當個股上漲、不輸出假分位數。
6. 輸入與 train-only 正規化不使用未來資料。Validation 用於選模，test 不參與 fitting；
   目前兩個獨立任務不做區間校準。完整記錄各自的 train／validation loss，
   不將回歸與分類分數混排。歷史任務的校準與日誌契約見[附錄](#legacy-results-zh)。
7. 主模型在相同評估集合上讀取同任務的已完成 baseline 結果，不重訓 baseline。
   點估計報回歸／符號／有效 RankIC，純分類報機率與類別診斷；
   兩者均保留市場／月份切片，不混用不同任務指標。
8. 多 Pod 驗收須涵蓋 selection、run ID、checkpoint、log、lease 與 guard 隔離；
   一台完成不能終止另一台，sync／共用環境更新不能覆寫使用中的資源。
9. 每個成對實驗／seed 只建立一台 Pod、回報一個父 run；兩個子模型依序執行且完全獨立。
   完成判定須核對兩份正確任務的 training 與 test artifacts；接續略過已完成部分，
   父下載一次取得兩份結果，不重複存權重或觸發 baseline 重訓。

Stage 1 證明流程可運作，不證明 alpha。單一 seed、單一 holdout 或高 GPU utilization
也不能建立穩健的預測優勢；需搭配多 seed、受控比較及未參與調參的後續回測。

工程證據有各自的版本與範圍，不代表後續變更或正式全量訓練已自動通過：
[整合市場適應驗收](docs/integrated_adaptation_verification.md)、
[多 Pod 控制與文件驗收](docs/runpod_parallel_verification.md)、
[清理與容量實驗驗證](docs/cleaning_experiments_verification.md)、
[tensor 資料管線驗證](docs/baseline_tensor_pipeline_validation.md)、
[baseline runtime 驗證](docs/baseline_runtime_validation.md)、
[完整評估流程驗證](docs/performance_workflow_validation.md)。
模型分析見 [reports](reports)，版本快照見 [RELEASES.md](RELEASES.md)。

<a id="references-zh"></a>

### 主要資料與模型參考

- [Kronos 論文](https://arxiv.org/abs/2508.02739)
- [Kronos 官方程式庫](https://github.com/shiyu-coder/Kronos)
- [TimesFM 官方程式庫](https://github.com/google-research/timesfm)
- [Google Research: TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Chronos 官方程式庫](https://github.com/amazon-science/chronos-forecasting)
- [Chronos 論文](https://arxiv.org/abs/2403.07815)
- [Uni2TS / Moirai 官方程式庫](https://github.com/SalesforceAIResearch/uni2ts)
- [Moirai 論文](https://arxiv.org/abs/2402.02592)
- [EODHD EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE 除權除息計算說明](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx 報酬指數](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive stocks pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)

<a id="historical-experiments-zh"></a>

### 歷史實驗附錄（非現行訓練入口）

本附錄收錄已退出主線的容量、adaptive 及 direction64 V1／V2／V3 實驗，
僅供重現與解讀歷史結果；設定、輸出與命令仍保留，並未刪除其功能。
目前架構與新實驗入口請使用[兩個獨立任務](#signal-tasks-zh)及[主模型訓練](#train-zh)。
歷史 preset 的資料、baseline、loss 與 checkpoint 契約不能當成目前獨立任務的結果。

<a id="model-versions-zh"></a>

#### 歷史方向實驗 V1／V2／V3：範圍與差異

V1、V2、V3 指保留的 **direction64 多任務系列**，不是 Stage 1／Stage 2、Git release 版本，
也不是目前兩個獨立任務的別名。三版都同時學 alpha 分位數、個股絕對方向、alpha 方向與
獨立排序；兩種方向都使用成本帶。以下是歷史實驗的重現契約，不會把這些目標加回新任務。

| 版本 | A／B 實驗名稱 | 方向 architecture／schema |
| --- | --- | --- |
| V1 | `a-direction64`／`b-direction64` | `legacy`／方向 schema 1 |
| V2 | `a-direction64-v2`／`b-direction64-v2` | `market_residual_v2`／方向 schema 2 |
| V3 | `a-direction64-v3`／`b-direction64-v3` | `market_ordinal_v3`／方向 schema 3 |

| 比較項目 | V1：雙方向初版 | V2：尺度直連與先驗分離 | V3：有序機率與市場狀態選擇 |
| --- | --- | --- | --- |
| 標籤／labels | 個股絕對報酬及 alpha 各分 down／neutral／up；門檻是預測日已知的成本帶 | 與 V1 相同，不改報酬、成本或清理規則 | 與 V2 相同，不以 test 調門檻 |
| 方向 head／機率結構 | horizon query 讀取 token；每個任務三個自由 logits | 保留 horizon query；歷史訊號 `f` 與 train prior `π` 分離，人口 logits `z=f+τ log π` | 兩個條件 logit 導出 `[1-a, a(1-b), ab]` 三類有序機率，不是三個自由 logits |
| 尺度與市場特徵 | 方向 head 未直接讀既有 20 維尺度分支 | 直接讀 20 維尺度及四個成本／波動比值；歷史 residual 加市場分支，market ID 選 prior | 保留 V2 路徑；benchmark 歷史狀態控制三組 token 與尺度特徵 gate |
| Loss reference／任務尺度 | 未訓練模型 forward 的 pinball／雙 CE；最多 1,024 筆，即使抽樣 seed 固定仍受模型 seed 影響 | train-only null quantile reference 與固定 `CE=log(3)`；最多 8,192 筆，市場／年份分層，與模型初始化無關 | 沿用 V2 的固定 reference；分類 loss 改成 conditional-chain NLL |
| 共享梯度／PCGrad | 加權總 loss 一次 backward | 只對共享 LoRA 的三組任務梯度做 PCGrad；其他參數仍用總 loss | 與 V2 相同；有序分類 NLL 的梯度仍傳入共享 LoRA |
| Checkpoint／early-stop 選模 | reference 加權的 pinball＋兩個 CE | 市場／任務／horizon 平均 balanced-decision BA，加凍結 pinball 退化 guard | 人口分布 CE／Brier 相對 prior 的 skill、macro-F1，加 opposite-error／collapse penalties 與 pinball guard |
| Baseline 比較 | 對齊 quantiles 與雙成本方向；不是純個股方向 baseline | 數值契約匹配時共用同一成本多任務 baseline；新增 balanced／prior 診斷不代表 baseline 改成 V2 head | 相同契約下仍用同一 baseline；不得把 baseline 當成 V3 ordinal 架構 |
| Resume／接續 | 只接續同一 V1 run | 必須建立 V2 新 run，不能接 V1 checkpoint | 必須建立 V3 新 run，不能接 V1／V2 checkpoint |

三版控制不變的部分：Kronos-base、LoRA rank 64／alpha 128、無部分解凍、個股與 benchmark
各五個技術通道、horizon-query 讀取、ranking 權重 `0.05`、表徵擾動權重 `0.05`、
task／LoRA LR `3e-5`／`5e-6`、dropout `0.05`、weight decay `0.01`、validation-plateau 排程，
以及只在選模完成後擬合的靜態區間校準。完整 loss／選模公式見[歷史訓練目標](#legacy-objectives-zh)。
版本升級是可驗證的設計差異，不代表準確率已提升；應在相同樣本、horizon、任務與 seed
對照下比較各項原始指標，不能直接比較不同版本定義的 `selection_score`。

#### 歷史 `quantiles` 輸出與架構

以下僅適用既有 `quantiles` 任務，不適用目前兩個獨立任務：

- `alpha_quantiles`: `[batch, 15-h_start, 3]`。
- 第二維依序是持有 `h_start`、`h_start+1`、…、14 個交易日；`h_start`
  只能是 1、2 或 3，生產設定預設為 1。舊 resolved config 保留原 horizon。
- 第三維固定為 q10、q50、q90。
- 單位是商品相對其 benchmark 的 adjusted execution log return。
- `ranking_scores`：啟用獨立 ranking head 時為 `[batch, 15-h_start]`；是無量綱排序分數，
  不是報酬、機率，也不能當成 q50。

原有容量與 adaptive 實驗沒有分類 head。`direction64`、`direction64-v2` 與 `direction64-v3` 另輸出個股絕對方向與 alpha
方向 logits，各為 `[batch, 15-h_start, 3]`，順序固定為 `down, neutral, up`；推論的
`direction` 欄位包含兩者機率、成本情境及門檻，不能將 alpha up 解讀為個股一定上漲。
既有 alpha 輸出契約及 `model_output_schema_version=5.0` 保留，方向欄位另以 schema 1
描述；v2 方向 schema 為 2，V3 為 3，另外明列人口分布機率與去先驗的 balanced decision。
`softmax(z)` 才是人口分布預測機率；balanced decision 不能冒充原始機率或原始 accuracy。
推論 JSON 的 `direction.probability_source="population_logits"` 明列機率來源；
`balanced_decisions.absolute` 與 `.alpha` 是以持有期字串為 key 的類別索引，
`0=down`、`1=neutral`、`2=up`，不是機率。其 `source`：
V2 為 `history_residual_logits_without_train_prior`；V3 為
`population_log_probabilities_minus_train_log_prior`，另明列 `probability_structure=ordered_conditional_chain`。
`train_prior` 保存所用訓練先驗。V3 的三個輸出是條件鏈導出的 log probabilities，不是三個獨立 logits。
分位數實驗仍可由 q10/q50/q90 後處理成 `strong_bearish`、`bearish`、`neutral`、
`bullish`、`strong_bullish`；這套五級訊號不是方向分類器，也不是額外訓練目標。
不相容的 output schema 會被拒絕載入。

```text
OHLCV through close t (asset + benchmark)
  ├─ normalized windows → Kronos → resampler → benchmark conditioning → head trunk
  │                                └─ benchmark latent ────┐
  └─ 20 historical numerical features ──────────────────────┴─ numerical encoding

head trunk ── base quantile parameters ──┐
numerical encoding ── quantile residual ─┼─ q50 + positive tail widths → raw q10/q50/q90
head trunk + encoding ── market residual*┘                                  │
                                                         frozen validation calibration
head trunk ── ranking head ─────────────┐                 (+ separate delayed-feedback
numerical encoding ── ranking residual*─┴─ ranking scores   diagnostic on holdout)

* Additional paths enabled by adaptive64 / adaptive128; calibration keeps q50 unchanged.
```

`direction64` 在 tokenizer 的 token embedding 與 Kronos transformer 之間加入零初始化的
技術通道 projection；個股、benchmark 各五個通道。三組 resampler／conditioned tokens
由各 horizon 的獨立 query 做 cross-attention，再交給兩個三分類 head。CE 梯度經共享
resampler 回傳 Kronos／LoRA；不 detach 表徵，也不以指標規則替代 backbone。
此原版設定不開啟 adaptive 的 market residual、ranking numeric residual 或適應式校準。
`direction64-v2` 保留同一 Kronos／LoRA 主體、十技術通道及持有期讀取，另把已存在的
20 維 train-only 標準化尺度特徵直接送入方向 head。每個 horizon 加上兩個精確成本門檻
各除以「個股／相對報酬 20 日波動 × sqrt(horizon)」的四個比值；預測輸入的比值限於
`[-20,20]` 後作 `asinh`，不可達的 lower band 只在這個特徵路徑轉為有限值。
精確方向 label、成本情境與極端報酬保持不變。方向市場 residual 讀取歷史表徵，
市場 ID 只選擇輸出分支與 train prior，不直接把市場／成本 metadata 加入歷史訊號 logits。

`quantiles` 任務直接學習條件 alpha 分布；不是先預測 raw-return q50 再減 benchmark q50。

#### 歷史分位數的尺度與數值輸出頭

以下的三參數、q50 與 tail-gate 設計只適用歷史 `quantiles` head：
原 head 的 horizon-conditioned hidden 另做 `512 → 16`；三者串接後經 `48 → 32 → 3`，
加在原 head 的三個 raw quantile parameters 上，位置在原 head LayerNorm **之後**。
head 另接入 US／TWSE／TPEx／unknown 的市場 embedding。輸出不再只有加法 residual：
以歷史相對報酬 20 日波動乘 `sqrt(horizon)` 作為正尺度，限制於各 horizon train robust
scale 的 0.1–10 倍，再乘上可學習的 0.25–4 倍正值 gate。`decoupled_output_scale`
讓上下區間寬度各自使用正值 gate；q50 location 使用 train robust scale，不跟著
歷史波動同步縮放。q10／q90 由 q50 減去／加上正值寬度，因此保持分位數順序。
市場 embedding、residual 輸出層與 scale gate 零初始化；
gate 的初始倍率為 1。Head、特徵轉換與 pinball loss 使用 FP32，
Kronos 計算仍沿用 BF16 mixed precision。

`adaptive64`／`adaptive128` 整合實驗另啟用 `market_residual_hidden_dim: 64`：
head hidden 與上述 48 維特徵串接，經 LayerNorm、64 維 MLP，再由 US／TWSE／TPEx／unknown
各自的零初始化輸出層修正 location 與上下尾參數；不複製 Kronos backbone。
`ranking_numeric_features: true` 將同一份 48 維歷史特徵經獨立 `48 → 32 → 1` residual
送入 ranking score。排序輸出層不直接修改 q50；共享 encoder 仍接受兩種目標的梯度。

#### 歷史 direction64 V1／V2／V3 的方向任務與十個技術通道

本節只描述上表的歷史多任務系列；`absolute-direction64` 不使用這些成本標籤或技術通道。
`a-direction64`／`b-direction64` 是 V1，帶 `-v2`／`-v3` 後綴分別為 V2／V3。
三版都以 `lora64` 設定為基礎：rank 64、alpha 128、無部分解凍，
task／LoRA LR 為 `3e-5`／`5e-6`，dropout `0.05`、weight decay `0.01`、validation plateau
排程及靜態區間校準。舊容量與 adaptive 設定仍保留；新實驗重新由固定 pretrained
Kronos 初始化，不把舊 checkpoint 當作可直接接續的新架構。v2 必須建立新 run，
不能將原版 `direction64` checkpoint 接續成 v2；同一個 v2 run 的中斷仍可正常 resume。

v2 以歷史特徵學習方向訊號 `f`，另儲存 train-only、市場 × 任務 × horizon 的類別先驗 `π`：
人口分布 logits 為 `z = f + τ*log(π)`，`τ=class_balance_power`，預設 1。
Training 使用自然類別比例的 `CE(z)`，不再乘 inverse-class 權重；報告與推論的原始
accuracy、CE 與人口分布機率來自 `z`／`softmax(z)`。`argmax(f)` 是另列的
`balanced_decision`，以等類別召回的 decision rule 檢查是否只學到市場多數類別，
不是已校準的交易勝率。這借鑑 [Menon et al. 的 logit adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)，
不能由該文的非金融實驗推論本專案已達 60% 或已改善泛化。

兩個方向 label 使用同一個 next-open／h-th-close 的 adjusted execution return：
絕對方向用個股 log return；alpha 方向用個股減 benchmark log return。假設 ex-ante
round-trip 成本比例為 `0 ≤ c < 1`，`r < log(1-c)` 為 down，`r > log(1+c)` 為 up，邊界及中間
皆為 neutral。Alpha 的成本帶代表要求相對優勢至少超過個股成本，不是兩腿避險的淨損益。
大盤跌 8%、個股跌 4% 會得到 absolute down、alpha up；做多篩選不能只看 alpha up。
Down 只表示不利變動，不表示已計入借券成本或可獲利放空。

低價股票的最低費用可能使 `c ≥ 1`。這時保留原樣本與報酬，up 仍須超過
`log(1+c)`，down 邊界不可達，其他有限報酬為 neutral；不刪除樣本，也不把成本壓到 100% 以下。
對外 JSON 的 `lower_log_return` 為 `null`，內部標籤比較使用負無限大。
模型與 GBDT 的成本特徵使用 `100 × tanh(log 門檻)` 保持有限；這只轉換預測輸入，
不改精確分類門檻、alpha 分位數或真實極端報酬。

成本設定在 `model.direction.costs`，不在訓練時查詢券商網站：

| 市場／參數 | 預設情境 |
| --- | --- |
| `taiwan_shares`／`us_shares` | 台股 1,000 股（整張倍數）；美股 50 股（50 股倍數），直接海外券商帳戶 |
| 台股佣金 | 元大、富邦公開一般牌告各 `0.1425%`、每邊至少 NT$20，先各算費用再等權平均；不是議價後的市場加權平均 |
| 台股賣出稅 | 股票 `0.3%`；本專案准入的股票型 ETF／TDR `0.1%`；不假設當沖減稅 |
| 美股佣金 | IBKR Pro Fixed `max(0.005 × 股數, US$1)`、上限成交額 1%，與 Schwab 線上上市股票 US$0 等權平均，每邊各算一次 |
| 美股規費 | SEC 賣出比例 `0.0000206`；TAF `0`（2026 Q4 暫停情境）；CAT 每股每邊 `0.000003` 作保守通用預留，並非每家都另收 |
| `slippage_bps_per_side` | 每邊 5 bps 的價差／滑價合併假設，不是券商平均手續費；可設 0 或壓力情境 |

所有名目金額以預測日已知的最後 raw-close 等值價格 × 股數估算，絕不以未來進出場價
決定 label 門檻。以台股 NT$100、1,000 股為例，總成本帶約 `0.685%`；美股 US$20、
50 股約 `0.20303%`，均含上述滑價假設。這是固定的 `listed_cash_2026_10_09` 成本情境，
不是 2016–2026 的歷史費率重建，也不含換匯、股息預扣稅、ADR 保管費或融券成本。
費率變更須建立新 run，不能在既有 run 內改變 label 定義。
費率來源：[元大](https://www.yuanta.com.tw/eYuanta/agent/Node/Index?C1=2021102007380647&C2=2021102008672547&ID=2021102008672547&Level=2&MainId=00415)、
[富邦](https://www.fbs.com.tw/DomesticStocks/DomesticStocks)、[證交所](https://www.twse.com.tw/zh/about/company/guide.html)、
[IBKR](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)、[Schwab](https://www.schwab.com/pricing)。
規費情境依據 [SEC 2026 費率公告](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)
及 [FINRA 2026 Q4 TAF 暫停公告](https://www.govinfo.gov/content/pkg/FR-2026-09-23/html/2026-19392.htm)；
CAT 預留參考 IBKR 公開規費表，並非所有券商的統一收費。
TDR 辨識涵蓋 `91XX` 與 `91XXXX` 兩種代碼，依據
[證交所證券分類](https://isin.twse.com.tw/isin/class_main.jsp?chklike=Y&issuetype=J&market=1)。

下列五個通道分別作用於個股與 benchmark，合計十個，使用 point-in-time OHLCV。
`SMA_n` 包含當前 bar；prior20 範圍不含當前 bar；每一個 token 都只能讀其歷史前綴。

| 每個 stream 的通道 | 精確定義 |
| --- | --- |
| close／MA 趨勢 | `log(C_t / SMA20_t)` |
| 短期 MA 趨勢 | `log(SMA5_t / SMA20_t)` |
| 中期 MA 趨勢 | `log(SMA20_t / SMA60_t)` |
| 前期區間位置 | `(2*C_t - max(H_{t-20:t-1}) - min(L_{t-20:t-1})) / (max(H) - min(L))`；零範圍輸出 0 |
| 成交量確認 | 最近 20 次 `sign(C_j-C_{j-1})*V_j` 總和除以同段成交量總和；無成交量輸出 0 |

不足完整 lookback 的通道輸出 0，不延伸輸入或補缺成交日；通道值限於 `[-5,5]` 只限制
feature leverage，不裁切 target 或排除真實極端報酬。MA／區間突破參考
[Brock et al. (1992)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1992.tb04681.x)，
趨勢及成交量確認參考 [Neely et al.](https://files.stlouisfed.org/files/htdocs/wp/2010/2010-008.pdf)。
5／20／60 日與連續值表示是本專案的可驗證設計，不宣稱是原論文在本資料上已證實的最佳值。
持有期 query 的動機參考 [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363)，
但本模型保留 Kronos，不複製其完整架構。

<a id="legacy-objectives-zh"></a>

#### 歷史訓練目標與選模

下列 monitor 都使用完整 validation，越低越好；不得跨版本以同名分數直接排名。
Train-only 尺度的計算沿用[訓練章節](#training-zh)的資料隔離契約。

| 任務／實驗系列 | Checkpoint monitor | 分數定義 |
| --- | --- | --- |
| 歷史 quantiles，無雙方向 head | `primary_5d/selection_score` | 全 horizon 平均 normalized pinball |
| 歷史 V1 `direction64` | `direction/selection_loss` | 固定 reference 加權 pinball＋雙 CE |
| 歷史 V2 `direction64-v2` | `direction/selection_loss` | balanced-decision BA 損失＋凍結 pinball guard |
| 歷史 V3 `direction64-v3` | `direction/selection_loss` | 人口機率品質、macro-F1、誤判／塌縮罰分＋pinball guard |


##### 歷史分位數目標

以下分位數與多任務目標只適用於既有 `quantiles` preset。
Normalized pinball 是把每個 horizon 的分位數誤差除以該 horizon 的 **train-only robust
scale** 後計算 pinball，再對有效樣本、horizons 與 q10／q50／q90 平均；越低越好。
它不是百分比、方向準確率或 correlation，也不保證 80% coverage。
`primary_5d/selection_score` 是保留的監控鍵名，實際選模分數涵蓋全部 horizons，
不是只看第 5 日。

對誤差 `u = (真實 alpha − 預測分位數) / train_scale[h]`，單一分位數的 loss 為
`max(q × u, (q − 1) × u)`。

訓練目標為 normalized pinball 加上權重 `0.05` 的 pairwise logistic ranking loss；
ranking 使用獨立 score head，不直接把 q50 當排序分數。
排序只比較同一截止日、同一市場的不同股票，排除重複 padding 與近乎相同的標籤；
每個 microbatch 最多 256 對，loss 仍涵蓋所有 forecast horizons。runtime-only date/market
索引改善同組股票在 batch 中相遇的機會，對清理後 train windows 做年度衰減動態抽樣；不改 bar-store。
原有實驗的 checkpoint／early stop 看完整 validation 的 normalized pinball；
V1／V2／V3 則使用上表及以下公式的多任務分數。
所有版本均不以 ranking regularizer 或 holdout
選模，也不做 prediction／parameter ensemble。

##### V1：未訓練模型 reference 與加權總 loss

`direction64` 的初始 loss 加權採 training-only reference normalization：先以固定 seed 59
讀取最多 1,024 筆 train windows，量測未更新模型的平均 `sQ, sAbs, sAlpha`，之後凍結。
預設 `absolute_weight=alpha_weight=1`，實際公式為：

```text
L = pinball + (sQ/sAbs)*CE_absolute + (sQ/sAlpha)*CE_alpha
    + 0.05*ranking + 0.05*sQ*representation_regularization
```

每個 CE 對所有樣本與 t1–t14 等權平均；不因較遠 horizon 而改權重。初始化時三個主任務
在各自 reference 尺度上權重為 1:1:1，**不是把原始 loss 直接相加，也不保證梯度相等**。
這借鑑 [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) 的初始 loss 比例思想，
不是完整 GradNorm 動態權重演算法；1:1:1 是需驗證的中性起點，並非金融預測通用最佳值。
各 loss、實際 CE 係數、validation 子任務 loss 都寫入 `metrics.jsonl`／W&B，reference
另保存在 checkpoint 與 completion result，resume 不重算。

表徵正則化借鑑 [Feng et al. 的股票方向 adversarial training](https://arxiv.org/abs/1810.09936)：
在方向 head 的 LayerNorm 後隱向量上，沿兩個 CE 梯度的單位方向加入 `0.01*sqrt(hidden_dim)`
的 L2 擾動，再計算兩個 reference-normalized CE 的平均。擾動方向 stop-gradient；
不修改原 OHLCV、不增加第二次 Kronos forward，不在 validation/test 加擾動。
`representation_weight=0.05`／`representation_radius=0.01` 是保守初值，不是原論文最優超參數。

原版 `direction64` checkpoint monitor 為 `direction/selection_loss`：完整 validation 的前三項 loss，
使用同一組凍結 reference，不含 ranking／adversarial regularizer。既有 `primary_5d/selection_score`
仍只表示 pinball，不偷偷改它的語意。方向結果分絕對／alpha，包含每個 horizon、市場、月份
的 confusion matrix、accuracy、balanced accuracy、macro-F1、majority baseline、up precision／recall
及訊號比例，避免大量預測 neutral 製造高準確率。`joint_up` 另記錄兩者同時 up 的命中數及 precision。
Balanced accuracy 平均該切片中有真實樣本的類別 recall；macro-F1 固定以三個類別平均，
缺少真實樣本的類別仍可由 confusion matrix／support 識別。
這些分類 accuracy 不可直接與舊 q50 sign agreement 當成同一指標比較。

##### V2：尺度直連、固定 reference、先驗分離與 PCGrad

`direction64-v2` 的 reference 不執行未訓練模型 forward：固定 calibration seed 59，
最多 8,192 筆 train windows，以有界的市場／年份分層抽樣及年度衰減權重估計先驗。
樣本預算先按有效市場分配，再依各年份的有效數 × 年度權重分配；統計以權重還原
training population。`sQ` 是零中心的 `[-train_scale, 0, +train_scale]` null quantiles
對 train labels 的 normalized pinball；`sAbs=sAlpha=log(3)` 為三分類均勻機率的 CE。
同一資料與設定不會因 training seed 的隨機模型初始化改變 CE 係數。
預設訓練公式仍保留 pinball、排序及表徵正則化：

```text
L_v2 = pinball + (sQ/log(3))*market_weighted_CE_absolute(z)
       + (sQ/log(3))*market_weighted_CE_alpha(z)
       + 0.05*ranking + 0.05*sQ*representation_regularization
```

市場權重從年度加權的 train population mass 固定計算，以 capped water filling
同時滿足 train 加權平均為 1，且最終有效權重不超過 `market_weight_max`（預設 3）。
對有效市場比例 `p_m` 使用 `w_m=min(k/p_m, market_weight_max)`，求解 `k` 使
`sum(p_m*w_m)=1`；不是先截斷再正規化，因為後者可能再次超過上限。
它不依每個 microbatch 重新正規化、不從 validation/test 擬合，也不另乘類別逆頻率。
Reference、先驗、市場權重與抽樣 provenance 寫入 checkpoint，resume 原樣還原。
各 horizon 維持等權；沒有把遠期 loss 改成低權重。

共享多任務梯度採 [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
的條件投影：三個加權任務為 pinball＋ranking、absolute CE、alpha CE，只對共享可訓練
LoRA 參數在實際 gradient dot product 為負時投影。沒有負衝突時與原本 backward 相同；
方向／alpha／ranking heads、技術 projection、數值分支及其他參數保留普通 total-loss
梯度，表徵正則化也保留原梯度。`monitor` 模式只量測、不投影。
正式訓練與 GPU batch probe 的 clipping 均設定 `error_if_nonfinite=True`，梯度管理不掩蓋
非有限值；偵測到非有限 gradient norm 時會在更新前報錯，不執行該步 `optimizer.step()`。
每 100 個 optimizer steps 彙總 gradient norms、cosines、負衝突比例及 nonfinite 計數，
以 `gradient/runtime_optimizer_step` 記錄精確的訓練步數位置。Tracking 的 canonical
step 可能是最近的 logging 邊界，不是當次 GPU 更新的確切步數或時間戳；分析梯度
診斷時應使用 runtime 欄位對齊。這不是宣稱已觀測到衝突或保證投影會改善方向率。
額外 FP32 buffer 以三任務的 `28 × LoRA numel` bytes 保守峰值計算，硬性上限
`512 MiB`，超限會拒絕而非忽略；
batch autotune 使用和正式訓練相同的梯度路徑，並涵蓋 AMP／gradient accumulation。

v2 同樣使用 `direction/selection_loss`，但其定義由該 run 的 v2 設定明列，不與原版混排：

```text
selection_loss = 1 - mean_market(mean_task(mean_horizon(balanced_decision_BA)))
                 + max(pinball/first_full_validation_anchor - 1 - 0.01, 0)*1
```

較低較好；`BA` 是該切片有真實支持的類別 recall 平均。市場、兩個任務、horizons
先各自等權平均，避免 US 樣本數主導選模；市場／類別 support 仍逐項揭露。
回歸護欄的 anchor 只在第一輪**完整** validation 完成後凍結，不用部分結果、不逐輪更新；
超過 anchor 1% 的相對 pinball 退化才加罰分（預設係數 1）。Anchor 隨 checkpoint
保存和 resume 還原；test 不擬合先驗、reference、anchor 或任何方向門檻。
各 run 的首輪 validation anchor 獨立，因此 `selection_loss` 只用於該 run 內選模，
不能用來跨 seed 排名。跨 seed 比較使用相同 holdout 的 raw pinball，以及分列的
raw／balanced direction 指標；不能把各自的回歸護欄基準當成共同評分尺度。
原始人口 accuracy 保留，去先驗 decision 的 accuracy／BA 明列在 `balanced_decision`，
不以後者取代原 accuracy。另報 CE 對 train-prior 基準的 skill、20 個 confidence bins、
聯合 up 的機率門檻／precision／樣本數及逐日 confusion。機率摘要採固定 bins，逐日
聚合最多 4,096 個 entries；超出上限明確失敗，不默默丟棄樣本或無限制累積記憶體。

下表欄位均位於 `model.direction`，不是新增 workflow CLI 旗標；已上傳的 v2 preset
可由 `--experiment` 選擇，調整數值契約仍需新 run：

| 設定欄位 | v2 preset／允許值 | 用途 |
| --- | --- | --- |
| `architecture` | `market_residual_v2`；原版 `legacy` 保留 | 選方向 head 架構，不能跨架構 resume |
| `loss_reference_samples` | `8192`；32–8192 | 有界 train-only 分層 reference 樣本上限；calibration seed 為 `data.calibration_seed=59` |
| `class_balance_power` | `1.0`；0–1 | `τ`，人口 logits 的 train-prior 強度，balanced decision 去除同一先驗項 |
| `market_residual_hidden_dim` | `64`；8–256 | 方向歷史訊號的數值 projection／市場 residual 中間維度 |
| `prior_smoothing` | `1.0`；大於 0、至多 100 | train 市場／任務／horizon 類別先驗 pseudo-count |
| `market_weight_max` | `3.0`；1–4 | 最終有效市場權重上限；capped water filling 同時使 train 加權平均為 1 |
| `gradient_management` | `pcgrad`；`none`、`monitor`、`pcgrad` | 共享 LoRA 負衝突投影、唯讀量測或普通 backward |
| `gradient_diagnostics_interval` | `100`；1–10000 | optimizer-step 診斷彙總間隔，不是 validation cadence |
| `probability_bins` | `20`；5–128 | 有界 confidence／joint-up 機率診斷 bins |
| `max_daily_entries` | `4096`；128–16384 | 逐日指標聚合 entry 硬上限，超限不抽樣 |
| `selection_pinball_guard_tolerance` | `0.01`；0–0.1 | 相對第一輪完整 validation pinball 的容許退化 |
| `selection_pinball_guard_weight` | `1.0`；大於 0、至多 10 | 超過容許退化後的 checkpoint 罰分倍率 |

`absolute_weight`／`alpha_weight` 仍為 1（大於 0、至多 10），可調兩任務權重；
`representation_weight=0.05`（0–1）、`representation_radius=0.01`（大於 0、至多 0.1）
及既有成本設定保留。v2 不修改 CPU prepare、資料清理、動態 training sampling 或
完整 validation/test；資料規則、成本情境及 baseline 數值／實作契約均匹配時，
可重用已完成的雙方向 baseline，不需因此重訓。不能僅憑日期或成本相同跳過契約檢查。

##### V3：有序方向、市場狀態選擇與機率品質選模

`a-direction64-v3`／`b-direction64-v3`（`a_direction64_v3.yaml`／`b_direction64_v3.yaml`）
保留 Kronos-base／LoRA64、alpha 分位數、獨立排序、十技術通道、直接尺度路徑、
train-only reference、市場權重與 PCGrad；只替換方向讀取及選模結構。
清洗規則、baseline 標籤／成本、資料來源與切分不變，既有 baseline 可重用。
V3 必須建立新 run，不能接續 V2 checkpoint；同 V3 的中斷仍可 resume。

方向分類借鑑 [CORN 的條件機率順序結構](https://arxiv.org/abs/2111.08851)：
`a=P(return>lower)`、`b=P(return>upper | return>lower)`，
`P(down)=1-a`、`P(neutral)=a*(1-b)`、`P(up)=a*b`。
上界跨越機率不會大於下界跨越機率。使用穩定 log-sigmoid 的三分類負對數似然（NLL），
等價於全樣本下界 BCE 加上**僅 non-down 樣本**的上界條件 BCE，再以全樣本數平均；
不是兩項 BCE 各自平均後相加。`log(3)` reference 仍對應三分類均勻 null 機率，
而非假設兩個條件機率都是 0.5。固定 train 先驗轉為條件座標 log-odds，歷史訊號學習
其 residual。Population 機率為 `softmax(log_probability)`；balanced decision 另用
`log_probability - τ*log(train_prior)`，不是可作勝率的校準機率。
此結構借鑑 ordinal 文獻，不宣稱論文已證明本資料可達 60% 方向率。

特徵選擇借鑑 [MASTER](https://ojs.aaai.org/index.php/AAAI/article/view/27767)，
不引入全市場 N×N attention、不新增指標或刪除極端報酬。只使用截至 `t` 的既有 benchmark
std20／std60、context close CV、mean5／mean20、volume ratio20、drawdown60，
沿用 train-fitted 標準化。小型 gate 網路產生三條 token stream 的 `3*softmax` gate
與 20 個數值特徵的 `2*sigmoid` gate，動態選擇 asset／benchmark／conditioned 表徵與
尺度特徵；gate 不讀未來、標籤、成本或 batch population，也沒有市場狀態直加 logits 的捷徑。
新增計算為 batch tensor kernels，不改 DataLoader、worker 或預取流程。

V3 仍最小化 `direction/selection_loss`，但其定義獨立於 V2：

```text
selection_loss = mean_market_task_horizon(
                    (CE / CE_train_prior + Brier / Brier_train_prior + 1 - population_macro_F1) / 3
                    + 0.25 * balanced_opposite_error + 0.25 * collapse_deficit)
                 + frozen_pinball_guard
```

CE／Brier 基準為凍結 train 市場／任務／horizon 先驗，在完整 validation labels 上評分，
不從 validation 擬合新先驗；markets／tasks／horizons 等權。BA 保留作診斷，不再單獨選模。
CE 與 Brier 的機率品質依據見 [Gneiting／Raftery 的 scoring-rule 理論](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。
含 F1／護欄的整體分數是工程複合選模規則，不宣稱整體也是 strictly proper scoring rule。
`balanced_opposite_error` 平均真 down 判 up、真 up 判 down 的條件錯誤率，排除無真實支持
的 tail；此處評分使用 population argmax，並非 `balanced_decision` 的預測。
`collapse_deficit` 對真實占比至少 5% 的類別，計算 `max(1-predicted_share/0.01,0)` 後平均。
這是可設定護欄，不刪樣本、不宣稱預測占比少於 1% 必然錯誤。
分項分數／罰分、類別占比、塌縮類別、ordinal MAE、相反方向錯誤、既有 precision／recall、
CE skill、Brier skill、confidence bins 均記錄，並同時診斷 balanced decision。
初始權重是工程預設，不是文獻保證的最佳金融參數。Pinball anchor、test 不擬合、
跨 run 不以 selection loss 排名的規則與 V2 相同；coverage 不列入方向選模目標。

| 設定欄位 | V3 預設／允許值 | 用途 |
| --- | --- | --- |
| `selection_class_support_min` | `0.05`；大於 0、至多 0.5 | 觸發塌縮護欄的最低真實類別占比 |
| `selection_prediction_share_min` | `0.01`；大於 0、至多 0.1 | 預測占比不足時的線性 deficit 基準 |
| `selection_collapse_weight` | `0.25`；0–2 | 類別塌縮選模罰分倍率 |
| `selection_opposite_weight` | `0.25`；0–2 | 真 down／up 相反方向誤判罰分倍率 |

這四項僅影響 V3；`architecture=market_ordinal_v3`，其餘方向參數沿用上表。
CLI 使用既有 `--experiment`／`--seed`／`--gpuId`／`--maxRuntime`，沒有新增必填參數：

```bash
bash scripts/runpod_workflow.sh train --experiment a-direction64-v3 --seed 42 \
  --gpuId "NVIDIA GeForce RTX 5090" --maxRuntime 72h
```

`b-direction64-v3` 選 B 組；seed 43／44 各自執行獨立 train 命令。這會建立正式訓練 Pod，
不是本機訓練或驗證；仍須先完成既有 code sync 與 readiness。

#### 歷史排程與區間校準

原 Stage 1/2、`lora32`／`lora64`／`partial` 及 direction64 V1–V3 採用
主文的 validation-driven plateau；`adaptive64`／`adaptive128` 的差異如下。

歷史 `adaptive64`／`adaptive128` 整合實驗使用 `sample_plateau`：按實際已處理 windows 計數，在 1,000,000 到 4,000,000
次呈現之間做 cosine 衰減至初始 LR 的 0.09，同時保留 plateau；兩者取較低倍率，
不以 A/B 各自的 epoch 長度決定衰減位置。256,000 次呈現 warmup、低 LR 的兩次驗證保障
及樣本計數都保存在 scheduler/checkpoint，resume 不會重設。設定項為
`training.sample_decay_start`／`sample_decay_end`，且 `warmup_samples ≤ start < end`。
task LR 為 `1e-5`、LoRA LR 為 `5e-6`；resampler、conditioner、head 與 LoRA dropout
均為 `0.10`，AdamW weight decay 為 `0.03`。不增加 epoch 上限。

整合實驗 `training.market_loss_weights: [1, 2, 2, 1]` 的順序是 US、TWSE、TPEx、unknown，
只加權 training pinball，每筆權重必須在 `(0, 4]`；不在單一市場 microbatch 內重新
正規化而抵銷權重。Ranking 仍使用 `0.05` 權重。Validation/test 保持未加權的完整集合，
另報告 `market_macro` 等市場平均診斷；不改 checkpoint monitor 或 baseline 目標。

以下校準只適用啟用 interval calibration 的歷史 `quantiles` preset；目前兩個獨立任務不做區間校準。
選定 checkpoint 後才用完整 validation 擬合區間校準，q50 與 ranking score 保持不變。
原設定的 `validation.interval_calibration.mode: static`（預設）保留 market/horizon 校準。
整合實驗使用 `regime_adaptive`，輸出分開保存：

| 結果欄位 | 使用規則 |
| --- | --- |
| 原有 raw metrics | 未校準預測；adaptive 對照以此選 checkpoint／early stop，不使用校準後的指標 |
| `calibrated_static` | 原 market/horizon validation 校準對照 |
| `calibrated` | 凍結的市場 × horizon × 歷史波動狀態校準；只 fit validation |
| `calibrated_online` | 獨立的逐日 prequential 診斷；只用已成熟的過往 test 預測誤差更新寬度 |

波動狀態為截至預測日的相對報酬 `std20/std60`，以 validation 三分位劃分三組。
每個有效市場日期先給相同權重，再以 `recency_half_life_sessions: 63` 衰減；
不足 `minimum_group_samples: 128` 或 `minimum_group_dates: 20` 時使用市場／pooled
係數，其他組以 `有效日期數 / (有效日期數 + shrinkage_dates)` 向母群收縮，預設
`shrinkage_dates: 60`。這些參數均位於 `validation.interval_calibration`。

線上診斷固定 `online_learning_rate: 0.05`，對每天已成熟的上下尾違約率分別更新
log-width multiplier，目標單尾 10%；更新幅度再乘 `當日樣本數 / (當日樣本數 + 128)`。
`online_max_multiplier: 2.0` 限定相對凍結寬度為 0.5–2 倍。到期日採既有離線交易日曆的
`t+h`，且必須**嚴格早於**當次預測日才可讀取報酬，不以日曆天數或下一筆有效股票資料代替。
它不回寫 validation 校準檔、不調整模型、不參與選模；不可把這個 online 結果冒充完全
凍結的 test。各校準結果都有 horizon、市場、月份、市場 × 月份的 coverage、width 與
pinball。這是經驗性校準，不承諾未來或每個子群必達 80% coverage。

#### 歷史評估指標

| 任務 | 主要品質指標 | 不輸出的指標／訊號 |
| --- | --- | --- |
| 歷史 `quantiles` | pinball、coverage／width、q50 誤差／符號／相關性、ranking 與啟用的歷史方向診斷 | 不把 q50 冒充新任務的 `alpha_point`，也不把舊成本分類當固定 ±1% 分類 |

歷史雙成本方向與 baseline 另有 `comparison.direction`，不是目前獨立任務的 `comparison.rows`。

##### 保留的分位數／雙成本方向 baseline

以下只適用歷史 quantile／direction64 路徑。GRU、DLinear、PatchTST 保留 alpha 分位數，
並以共享表徵訓練絕對／alpha 雙三分類；GBDT 保留分位數 regressors，另建每 horizon
兩組 classifiers。標籤、進出場時點、成本帶與歷史 direction64 相同；啟用雙方向時以固定 reference
加權 normalized pinball＋雙 CE 選模，quantile-only baseline 則保留分位數目標。
Rule 的 alpha residual quantiles 以完整 train 校準。

歷史方向設定由 `configs/baseline.json` 的 `direction` 管理：`enabled` 預設 `true`，
`loss_reference_samples` 預設 1,024（32–8,192），`absolute_weight`／`alpha_weight` 預設
各 1.0（大於 0、不超過 10）。`costs` 使用與主模型相同的市場別成本欄位，台股 1,000 股、
美股 50 股；只用截止預測日已知的價格估算門檻。神經模型在更新前估計固定 reference，
GBDT 使用固定 train 子集的平滑類別先驗與 intercept-only 分位數 reference；正式 fitting、
validation、test 均不抽樣。Reference 及其加權係數隨 checkpoint 保存並於續訓恢復。
`metrics.jsonl` 記錄訓練／validation loss；神經模型記錄實際訓練 batch／區間平均，
GBDT 另以固定 train reference 子集記錄訓練診斷（明列筆數），不減少正式 fitting 或完整評估。
不同模型的 reference 不同，不能以加權
`selection_loss` 跨架構排名，應比較各任務指標。

完整評估另輸出 `direction`：兩任務的 accuracy、balanced accuracy、macro-F1、混淆矩陣、
majority accuracy、up precision／recall、signal fraction、joint-up precision，均有
horizon／市場／月份切面。有機率的模型另列 CE。規則以各自的歷史個股／相對訊號配合成本
門檻輸出 hard classes；`always_buy` 固定 up、`zero_return` 固定 neutral，不捏造機率或 CE。
原有 q50 正負號一致率仍保留，不能和成本敏感三分類 accuracy 混為同一指標。
主模型 testing 的 `comparison.direction` 直接彙整快取結果；成本、horizons 或樣本集合
不一致就拒絕比較。方向模型啟動前也在本機檢查成本一致性，不因此把 baseline 綁定 Kronos 容量。

歷史成本方向的 `inputs/<split>/absolute_targets.npy` 與 `direction_bands.npy` 同樣每個 split
只存一份；每個模型只保存自己的 `direction-probabilities.npy`（`N×2×H×3`）或規則的
`direction-classes.npy`（`N×2×H`）。任務順序為 absolute、alpha，類別順序為 down、neutral、up。
只有 alpha 分位數的舊 baseline 不能視為雙方向任務已完成；也不能將上述 `N×2×H` 的
雙任務結果當成目前單一 `absolute_direction` 結果。重用前一律檢查目前數值契約。

<a id="legacy-presets-zh"></a>

#### 歷史設定與重現入口

以下十六份設定仍可用於重現，但不是目前獨立 alpha／個股漲平跌模型。
A/B 資料日期與主文相同；切換 `--experiment` 不會把其任務自動改為目前架構，
須使用各自匹配的 baseline，且不能跨任務或架構接續 checkpoint。

| 實驗名稱 | 設定檔（位於 `configs/experiments/`） | Kronos 可訓練容量 |
| --- | --- | --- |
| `a-lora32`、`b-lora32` | `a_lora32.yaml`、`b_lora32.yaml` | LoRA rank 32、alpha 64 |
| `a-lora64`、`b-lora64` | `a_lora64.yaml`、`b_lora64.yaml` | LoRA rank 64、alpha 128 |
| `a-partial`、`b-partial` | `a_partial.yaml`、`b_partial.yaml` | 前 10 層 LoRA-32；解凍最後 2 層與 final norm，解凍層不重複套 LoRA |
| `a-adaptive64`、`b-adaptive64` | `a_adaptive64.yaml`、`b_adaptive64.yaml` | 市場分支、排序特徵、正則化／樣本排程與適應校準對照；LoRA rank 64、alpha 128 |
| `a-adaptive128`、`b-adaptive128` | `a_adaptive128.yaml`、`b_adaptive128.yaml` | 與 adaptive64 相同整合設定；只改 LoRA rank 128、alpha 256 |
| `a-direction64`、`b-direction64` | `a_direction64.yaml`、`b_direction64.yaml` | A-LoRA64 基礎，加雙三分類、十技術通道、horizon query 與表徵正則化；rank 64、alpha 128 |
| `a-direction64-v2`、`b-direction64-v2` | `a_direction64_v2.yaml`、`b_direction64_v2.yaml` | 同樣 rank 64、alpha 128；方向 head 直接尺度路徑、train prior 分離、固定 reference、市場權重與共享 LoRA 負衝突 PCGrad |
| `a-direction64-v3`、`b-direction64-v3` | `a_direction64_v3.yaml`、`b_direction64_v3.yaml` | 同樣 rank 64、alpha 128；成本有序條件機率、市場狀態特徵選擇、population 機率品質與誤判護欄選模 |

全部從相同 pretrained Kronos-base 初始化；`partial` 為次要研究線。
原六份容量設定的 task／LoRA／解凍層 LR 為 `3e-5`／`5e-6`／`1e-6`；
所有 preset warmup 為 256,000 次樣本呈現，年度加權動態 sampling 與完整評估規則沿用主文。
各系列的 loss、選模、校準與整合設定差異見[歷史訓練目標](#legacy-objectives-zh)。
`FORECAST_METRIC_WORKERS` 亦限制其診斷／校準執行緒（預設上限 4）。

保留的分位數容量對照也能一次建立三台 Pod；每個 `--experiment` 都帶一個名稱：

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-lora32 \
  --experiment a-lora64 \
  --experiment a-partial \
  --launchWorkers 2 \
  --maxRuntime 12h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

上述命令會建立三台付費訓練 Pod；僅在重現歷史容量對照時使用。
將 `a-` 改成 `b-` 可重現 B 組，`--seed` 與其他建立參數沿用[CLI 參數速查](#cli-reference-zh)。

<a id="legacy-results-zh"></a>

#### 歷史日誌、產物與驗收

整合實驗的 `train/loss` 不等於未加權 `train/pinball_loss`；加權部分另記為
`train/weighted_pinball_loss`。分析 pinball 的 train/validation gap 時，統一比較
`train/pinball_loss` 與 `validation/aggregate/normalized_pinball`；方向實驗也可使用
`validation/direction/pinball_loss`，不要將其與方向選模用的 `validation/loss` 混淆。

以下是舊分位數與多任務日誌：`train/loss` 是 market-weighted normalized pinball
加上加權 ranking（原設定的市場權重全為 1），另列未加權的 `train/pinball_loss`、
`train/weighted_pinball_loss`、`train/ranking_loss` 與各參數組 LR。這是無方向 head 實驗的
total-loss 定義；方向實驗另包含已記錄係數的 absolute／alpha CE 及表徵正則化，
v2 另保存 train priors、固定市場權重、共享梯度診斷、population／balanced-decision 指標
及第一輪完整 validation 的 pinball anchor；不可把不同版本的 `direction/selection_loss` 混排。
`validation/loss` 記錄該 run 的 `checkpoint_monitor`，不是所有實驗共用的 loss：
歷史 `quantiles` 且無雙方向 head 時為完整 validation 各 horizon 平均的 normalized pinball，不含 ranking；
原版 `direction64` 為 pinball 加上凍結 reference 加權的兩項 CE；
`direction64-v2` 則為等市場／任務／horizon 的 balanced-decision BA 選模損失，加上
凍結首輪完整 validation pinball anchor 的退化罰分。
`direction64-v3` 則用 population CE／Brier 對 train prior 的比率、macro-F1、相反誤判／塌縮
罰分與相同 pinball 護欄；分項皆記錄，BA 不再單獨選模。
判斷 pinball 的 train/validation gap 時，應比較 `train/pinball_loss` 與
`validation/aggregate/normalized_pinball`（方向實驗亦另列
`validation/direction/pinball_loss`），不能把 training total loss 或方向選模損失
直接當作 validation pinball。歷史雙方向任務的泛化則分別檢查完整 validation 的
absolute／alpha CE、accuracy、balanced accuracy 與市場別指標。

下列欄位只適用於舊分位數任務：獨立 `ranking_scores`，以及若存在與此 checkpoint 權重相符的 validation 校準，另外回傳的
`calibrated_forecast`；否則為 `null` 並附 `interval_calibration_status`。不生成自然語言解釋。
歷史 adaptive 分位數實驗的單次推論使用凍結的 validation regime 校準，狀態為
`validation_fitted_frozen_regime`；不暗中載入 test 的 online 回饋狀態。

歷史輸出驗收另須核對：

舊分位數任務 `alpha_quantiles` 為 `[B,15-h_start,3]`、q10 ≤ q50 ≤ q90；
`ranking_scores` 與中位數報酬分開；歷史 direction64 V1／V2／V3 另輸出兩組三分類機率，不生成自然語言。
V2 的 population probabilities／raw accuracy 與 `balanced_decision` 必須分開，不能偷換指標。

啟用區間校準的分位數任務只在選模後擬合；frozen test 不參與 fitting，
獨立 online 診斷只使用到期後的過往標籤。分位數任務另報 pinball、coverage／width
與 ranking，保留市場／月份切片，不與目前獨立任務指標混用。

---


<a id="single-model-zh"></a>

#### 進階單模型操作（非過時模型）

現行成對實驗內部仍使用 `a-alpha-point64`、`a-absolute-direction64`、
`b-alpha-point64`、`b-absolute-direction64` 四個 task preset。這些單模型入口未移除，
也不是歷史 V1／V2／V3；只供明確需要單一任務的消融、診斷或既有 run 重現。
一般訓練使用上文 `a-signal64`／`b-signal64`，不需先後執行下面兩種任務。

```bash
bash scripts/runpod_workflow.sh train --experiment a-alpha-point64 --seed 42 --maxRuntime 2h --gpuId "NVIDIA GeForce RTX 5090"
```

要單獨執行方向任務可改用 `a-absolute-direction64`；B 組改 `b-` 前綴。
`baseline --experiment NAME` 也保留這四個名稱的單任務行為；其任務 selection 會成為
active selection，與不改 active selection 的成對操作不同。`readiness --baseline`
不帶 experiment 仍查 active selection；`runs --experiment NAME` 仍可精確篩選子模型。

`probe-scales`、W&B 單模型補傳、低階 inference 必須使用 `status <父 RUN_ID>` 列出的
子 `MODEL_RUN_ID`，不是父 ID；這些功能沒有將兩份權重合成同一 checkpoint。
若需要專用診斷 Pod，可用上述單模型命令建立，但不要啟動 training tmux，改依
[尺度診斷](#probes-zh)啟動 `probe-scales`。診斷完成與本機 guard 的原有規則不變。

## English

### Navigation

- [Scope and license](#overview-en), [architecture and outputs](#model-en)
- [Current independent tasks](#signal-tasks-en)
- [Data sources, continuity and liquidity](#data-en), [fixed time splits](#splits-en)
- [Offline pipeline](#pipeline-en), [training, losses and full evaluation](#training-en)
- [RunPod operations](#operations-en): [account/storage](#setup-en) → [configure/sync](#configure-en)
  → [CPU preparation](#cpu-en) → [baselines](#baseline-en) → [tasks/experiments and multi-Pod training](#train-en)
  → [downloads](#download-en)
- [Artifacts and inference](#artifacts-en), [scale diagnostics](#probes-en), [CLI reference](#cli-reference-en)
- [Acceptance and interpretation](#acceptance-en), [model/data references](#references-en)
- [Appendix: obsolete experiments, including V1/V2/V3](#historical-experiments-en)

If data and the selected task's baselines are ready, start at tasks/experiments and multi-Pod training. Synchronize first only
when source or YAML changed; never overwrite shared source while a Pod owns the volume.

<a id="overview-en"></a>

### Project scope and license

This project fine-tunes a finance-pretrained time-series foundation model on
daily OHLCV data for US/Taiwan common stocks, ADRs/TDRs, and audited
benchmark-mappable unleveraged equity ETFs. The system processes only
numerical time series:

- Inputs and outputs are numerical tensors; no natural-language generation or
  fact-reconstruction interface is provided.
- The training loop never calls an external market-data API.
- Horizons run from configurable `h_start` (1, 2, or 3) through the fixed 14th holding day.
  Independent tasks predict robust alpha points or fixed-return-band stock down/neutral/up
  classes. Superseded model families remain available for historical reproduction;
  their contracts are in the [obsolete-experiment appendix](#historical-experiments-en).
- It exposes reusable numerical encoder representations for downstream systems.

This is a research and capability-validation PoC. It is not investment advice,
a production trading system, or a claim of guaranteed profitability.
The model supplies testable alpha and absolute-stock direction signals. Downstream systems
(such as a Hermes agent) combine those signals with conventional indicators to handle trading
rules, costs, positions, risk controls and backtests; these are not the current prediction targets.

#### License and versions

Project-owned code and expressly released model additions are available only to
individuals for free personal research, learning, experimentation, non-commercial
hobby projects, and trading with their own personal funds. **Companies, legal
entities, funds, quantitative trading firms, all organizational uses, commercial
products, and paid services are not licensed.** Managing third-party capital is
also prohibited. See [LICENSE](LICENSE) and [MODEL_LICENSE](MODEL_LICENSE).
This is source-available personal-use software, not OSI open-source software.

Kronos source, pretrained weights, and tokenizer retain their original MIT terms.
This project does not restrict rights independently granted upstream. See
[third-party notices](THIRD_PARTY_NOTICES.md). GitHub may show a custom license as
Other; the complete text controls. See [release history](RELEASES.md) for
architecture and report snapshots.

<a id="model-en"></a>

### Architecture and numerical outputs

<a id="signal-tasks-en"></a>

#### Current architecture: one experiment, two independent models

The current entry points are `a-signal64` and `b-signal64`. Choose a data group and seed once
to create one Pod and one parent run ID. That Pod sequentially trains the alpha-point and
absolute-stock down/neutral/up models, including each model's complete validation/test.
Each child has independent Kronos/LoRA weights, optimizer, head and checkpoints. The losses
and gradients are not combined. Sequential execution avoids holding two full models in VRAM;
it does not require two rented Pods. Query, download and resume by the parent run ID; child
IDs remain available for individual inspection. The two model contracts are:

| Model task within the pair | Output and meaning | Training and full-validation selection |
| --- | --- | --- |
| `alpha_point` | `alpha_point`, shape `[batch, 15-h_start]`; a robust point estimate of benchmark-relative adjusted execution log return, not labelled q50 or claimed to be an expected return | Train-scale-normalized Smooth L1 (Huber form), `point_loss_beta=1.0` |
| `absolute_direction` | `absolute_direction_logits`, shape `[batch, 15-h_start, 3]`; softmax gives absolute-stock `down, neutral, up` probabilities | Absolute-stock three-class cross entropy |

The independent direction task labels the stock's adjusted simple return from the same next-open
entry to horizon-close exit: below −1% is down, −1% through +1% inclusive is neutral, and above
+1% is up. `neutral_return_threshold=0.01` is a fixed return threshold, **not a trading cost**;
there is no additional alpha classifier. This is an initial candidate, not an established optimum.
The baseline workflow audits all training windows by market/horizon before fitting, blocking
missing classes, minimum class share below 1% or maximum share above 98%. Main-model launch
checks that stored audit again without rescanning prices; test labels never select the threshold.
Both tasks retain A-LoRA64's Kronos, benchmark conditioning, scale features and LoRA-64,
without quantile/ranking heads or interval/adaptive calibration. `direction.enabled=false`
disables only the legacy dual-direction multitask module, not the `absolute_direction` task.
Old experiments retain their output. Point forecasts are not duplicated into synthetic quantiles;
independent tasks use the `signal-1.0` schema.

`alpha_point > 0` predicts positive benchmark-relative excess log return, not necessarily an
absolute stock gain or a probability of beating the benchmark. Its sign already expresses the
predicted relative direction; **no separate alpha classifier is needed**. Regression
`direction_agreement` measures agreement between predicted and observed alpha signs; it is
not another head, classification loss or absolute-stock direction accuracy.

Each independent task follows the same input path: asset/benchmark OHLCV through `t` → window
normalization → Kronos/LoRA → resampling and benchmark conditioning → mean-pooled representation
plus market/horizon embeddings → selected task head. The 20 historical scale features and
benchmark latents also directly enter a numerical branch. The point head emits one value per
horizon; the classifier emits three logits. The current
presets **do not enable** the ten technical channels, horizon-query attention, market-state gates,
train priors, PCGrad, ranking or representation-perturbation regularization, and use no
transaction-cost features or labels.

Select the current paired architecture explicitly with `--experiment a-signal64` or
`--experiment b-signal64`. Each pair uses two task-specific YAML files without requiring
two separately launched experiments. Standard
`stage1_kronos_base_lora.yaml` / `stage2_kronos_base_lora.yaml` still select `quantiles`.
Omitting `--experiment` retains configure's active selection; it does **not** automatically
switch to the latest task.

Production configs use `NeoQuasar/Kronos-base` and
`NeoQuasar/Kronos-Tokenizer-base`. Standard LoRA experiments freeze the base weights. LoRA is
injected into `q_proj`, `k_proj`, `v_proj`, `out_proj`, `w1`, `w2`, and `w3`;
the resampler, benchmark conditioner, and selected task head remain trainable. The official source is pinned to
commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`; the required source snapshot
and MIT license are synchronized with the project. Preflight and model
construction verify each source file by SHA-256 and never invoke Git on RunPod.
Model and tokenizer weights are separately pinned to
Hugging Face commits `2b554741eca47781b64468546e77fef3e85130e6` and
`0e0117387f39004a9016484a186a908917e22426`; downloads, offline smoke tests,
configs, and checkpoints bind those revisions.

`QuantForecastModel.encode_ohlcv(...)` explicitly returns:

- `last_hidden_state`
- `attention_mask`
- `latent_tokens`

Downstream systems can reuse these numerical representations in other forecasting,
ranking or risk-analysis modules without changing the selected task's output contract.

Benchmark history affects both current tasks through gated cross-attention and a direct
resampler-latent branch. Scale features are extracted before window normalization. Benchmark
observations after close `t` are used only for offline labels, never model inputs.

#### Small numerical feature branch

Production defaults to `combined`, retaining Kronos-base and the conditioner.
The numerical branch is independent of LoRA rank or partial-unfreezing capacity.
Its 20 features have the following fixed order:

| Positions | Features | Definition |
| --- | --- | --- |
| 1–3 | Asset, benchmark, relative 20-day volatility | Population standard deviation of the last 20 daily log returns; relative returns are asset minus benchmark |
| 4–6 | Asset, benchmark, relative 60-day volatility | Same definition over the last 60 daily log returns |
| 7–8 | Asset and benchmark context price CV | Population standard deviation of adjusted closes divided by their mean over the 128-bar context |
| 9–14 | Asset, benchmark, relative 5/20-day mean log return | Historical daily returns only |
| 15–16 | 60-day correlation and beta | Historical asset/benchmark covariance and volatility |
| 17 | Relative 20-day downside deviation | Root mean squared negative return |
| 18–19 | Asset/benchmark current-to-20-day mean volume ratio | `log((volume_t + 1) / (mean_volume_20 + 1))` |
| 20 | Benchmark 60-day drawdown | Log ratio of current close to the historical window peak |

Statistics use only aligned observations available through `t`, require at least
61 valid bars, and are not annualized. Future returns and future corporate actions
never enter features. The first eight values use `log(x + 1e-8)`; the others use
signed `asinh`. Values are normalized with
train-only median/IQR statistics (IQR floor `1e-3`), and clipped to `[-10, 10]`.
Checkpoints bind the feature schema, training dataset fingerprint, and aggregates
in `runtime_scale_features`. Inference restores them without refitting on
validation, holdout, or the inference batch.

For both independent tasks, the scale MLP is `20 → 32 → 16`, mean-pooled benchmark
latents use `512 → 16`, and the horizon-conditioned trunk uses `512 → 16`. Their
concatenation passes through `48 → 32 → output_dim`: `output_dim=1` for alpha point
regression or `3` for absolute direction. The point output is multiplied by its horizon's
train-only robust scale; class outputs are raw logits. No positive tail-width gates,
quantile ordering or interval calibration exist on these paths.

Production `explicit_output_scale` requires `scales` or `combined` feature mode.
Removing the scale branch in a custom configuration also requires disabling
explicit scale control. Changing the mode preserves the dataset namespace but requires a new run.
Predictive improvement must be demonstrated by new evaluation results; adding
scale information alone is not evidence of improved alpha forecasting.

#### Backbone choice

The integrated backbone is the finance-OHLCV-pretrained Kronos-base: its domain matches the
inputs, public model/tokenizer revisions can be pinned, and LoRA/partial-unfreezing experiments
fit the single-GPU workflow. See the [official repository](https://github.com/shiyu-coder/Kronos)
and [paper](https://arxiv.org/abs/2508.02739). This is not a claim of universal superiority;
backbone comparisons must control data, context, heads and evaluation membership, not merely
compare aggregate scores from different projects.

<a id="data-en"></a>

### Data sources and sample rules

#### Data sources and selectable datasets

The normal RunPod workflow selects a profile through
`bash scripts/runpod_workflow.sh configure`. `FIN_TS_DATASET_PROFILE` is an
internal value passed to the Pod only after the script validates the selection:

| profile | Actual sources | Status | Use case |
| --- | --- | --- | --- |
| `tw_only` | Official TWSE + official TPEx | Available | Research without US API cost |
| `us_only_eodhd` | EODHD US stocks/ETFs | Available | Validate US-market capability first |
| `us_tw_eodhd` | EODHD + TWSE + TPEx | Default PoC | Full US/Taiwan PoC |
| `us_tw_massive` | Massive + TWSE + TPEx | Typed interface only | Implement after obtaining suitable rights |

The EODHD path can discover both active and delisted US stocks/ETFs by default,
reducing survivorship bias. When budget or quota is constrained, use
`--universe explicit` with `--stocks` and `--etfs`, or use `--symbol-limit` in
all-universe mode. Manifests record the profile, actual providers, markets,
symbols, asset types, date range, and per-split sample counts.

Discovery under `--universe all` is the active/delisted snapshot returned by
EODHD at preparation time, not point-in-time constituents reconstructed for
every historical session. For `--start 2016-01-01 --end 2026-06-01`, the date
contract is `[2016-01-01, 2026-06-01)`: an instrument listed during the range
starts at its first provider-available session, and one delisted during the
range ends at its last available session. An instrument both listed and
delisted inside the range is included when EODHD delisted discovery returns it
and the account is entitled to it. `explicit` processes only named tickers;
`--symbol-limit` processes only its selected subset. A raw row's `is_active`
value is the discovery-time state, not a daily listing-state history.

EODHD separately documents EOD, fundamentals, dividends, and splits for
instruments delisted after 2018, but guarantees only EOD for pre-2018
delistings. Those older EOD rows can therefore still enter raw data and training
windows, while their split-adjusted-volume auxiliary coverage cannot be treated
as complete. The download manifest records their count and symbols under
`delisted_pre_2018_auxiliary_coverage_warning`, so price history is not
misrepresented as complete corporate-action history.

EODHD is PoC data and must not be represented as an exchange-grade market feed.
Adjusted prices, corporate actions, delisted history, time zones, and revisions
can differ across providers. Sample reconciliation on overlapping instruments
is required before formal comparisons.

#### Execution timing, benchmarks and adjusted data

Each sample emits a signal after trading-day `t` closes. Entry occurs at the
next **market session's** raw regular-session open, which counts as holding day
one. A horizon `h` exits at the raw close of the `h` th market session, for
`h ∈ {h_start,...,14}` and `h_start ∈ {1,2,3}`. Alpha point/quantile targets are the
difference between instrument and benchmark adjusted total-return log returns over
identical entry and exit timestamps. The independent absolute-direction classifier instead
labels the instrument's adjusted simple return using its fixed neutral band. Historical
cost-direction labels retain their [separate archived contract](#model-versions-en).
Missing or zero-volume asset bars never postpone entry or exit to the next available row.

Default benchmark policy:

- US common stocks, ADRs, and allowlisted equity ETFs: `VTI.US`.
- TWSE common stocks, TDRs, and allowlisted equity ETFs: `TAIEX.TW`, whose adjusted anchor uses the official TAIEX total
  return index.
- TPEx common stocks: `TPEX.TWO`, whose adjusted anchor uses the official TPEx return
  index.
- Only audited allowlisted unleveraged equity ETFs that can map to an approved
  benchmark enter training. Leveraged, inverse, bond, commodity, volatility,
  and unaudited ETFs fail closed. `benchmark_mapping_path` may change the
  benchmark of an allowlisted ETF but cannot expand the training universe.

`VTI.US` is a required data dependency for US benchmark-relative labels and
benchmark context. It is not an ordinary `--symbol-limit` candidate and cannot
be its own training target (`self_benchmark` excludes it). The workflow first
selects N ETF candidates and N stock candidates, then ensures VTI is present:
an already-selected VTI is not duplicated; otherwise it is added. This prevents
the benchmark from consuming one of the N ETF candidate slots. The raw universe
is therefore at most `N ETFs + N stocks + 1 VTI`, while data-length, benchmark
mapping, and quality gates can reduce the actual trainable-target count.

Raw O/H/L/C is retained permanently, as is official Taiwan raw volume. EODHD
defines its EOD `volume` as already split-adjusted, so the pipeline uses the
complete Historical Splits response to reconstruct contemporaneous unadjusted
`volume` and retains the vendor value as `split_adjusted_volume`; it never
multiplies that value by the split factor again. Model windows normalize each vendor or
official total-return factor to `cutoff_at` before applying it to historical
O/H/L/C, so future corporate actions cannot rewrite an after-close inference
input. Volume is adjusted only for splits/share changes, never for cash
dividends. EODHD retains `adjusted_close` and uses the per-symbol Historical
Splits API for every date range. EODHD lists that endpoint under EOD
Historical Data — All World at one API call per request; the pipeline does not
use `calendar/splits`, which belongs to Calendar-enabled products. Each symbol
therefore requires one EOD-history request plus one split-history request. Both
are cacheable/resumable and included in the informational request estimate; the
estimate never blocks a complete dataset. The provider's pre-2018 delisting
exception is retained explicitly in the warning described above. Taiwan uses
official TWSE/TPEx ex-right/ex-dividend data and return indices. Existing monthly
benchmark rows provide the actual trading sessions, so ordinary weekdays are
not blindly treated as open sessions. This removes artificial corporate-action
gaps while retaining the tradable next-day raw-open entry semantics.

#### Data completeness, continuity and minimum liquidity

Main-model and baseline train/validation/test share `configs/data_cleaning.json`.
Inference applies the same checks that depend only on past observations:

- **Calendar:** pinned offline `exchange-calendars==4.13.2`, XNYS for US and XTAI
  for TWSE/TPEx. Weekdays are not automatically sessions. Simultaneous gaps in
  asset and benchmark are not automatically holidays. Exceptional closures need
  exchange-announcement verification; prices are never fabricated.
  `calendar_overrides` includes eight Saturday sessions in 2016–2018 and removes
  the missing closures on 2022-02-04, 2023-01-18, and 2024-10-31. Sources are
  [TPEx 2016](https://wwwov.tpex.org.tw/storage/zh-tw/web/bulletin/trading_date/trading_date_105.htm),
  [TPEx 2017](https://wwwov.tpex.org.tw/storage/zh-tw/web/bulletin/trading_date/trading_date_106.htm),
  [TWSE 2018](https://www.twse.com.tw/staticFiles/product/publication/0001002657.pdf),
  [TWSE 2022 Lunar New Year](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff8080817d22b9cb017e3336c4e203e3.pdf),
  [TPEx 2023 calendar copy](https://www.honsec.com.tw/uploads/images/112%E5%B9%B4%E6%9C%89%E5%83%B9%E8%AD%89%E5%88%B8%E6%AB%83%E6%AA%AF%E8%B2%B7%E8%B3%A3%E5%B8%82%E5%A0%B4%E9%96%8B%EF%BC%88%E4%BC%91%EF%BC%89%E5%B8%82%E6%97%A5%E6%9C%9F%E8%A1%A8.pdf),
  and the [2024-10-31 closure](https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d69236c2e30192dd5179bc0327.pdf).
- **Complete sequences:** inputs must be the exact 128 consecutive market sessions
  through `t`; outputs must cover the next 14 sessions even when `h_start` is 2 or 3.
  Both streams must align. A missing session excludes the window: no forward fill,
  reaching farther back to collect 128 rows, compressed suspensions, or shifted horizons.
- **Valid observations:** OHLC and adjusted close must be finite and positive with
  consistent high/low bounds throughout input and output. Asset volume and traded
  ETF benchmark volume must be finite and positive. Non-traded index benchmarks may
  have zero volume, but must still provide complete valid prices.
- **Minimum liquidity:** the trailing 60-session median of raw close × raw volume
  through `t` must reach USD 1,000,000 or TWD 10,000,000. Future volume is never used
  for this decision, and no FX API is needed. Development inputs shorter than 60 bars
  use their full input length; production 128-bar inputs use 60 sessions. These are
  configurable engineering defaults, not proven optimal investment thresholds;
  always review exclusion rates alongside results.
- **Real extremes remain:** valid finite extreme returns are neither removed nor
  winsorized. Absolute adjusted log returns above 0.5 are diagnostic counts only.
  Old extreme-transition flags in prepared candidate ranges no longer determine
  runtime membership. Invalid numbers and inadequate history are distinct from
  genuine market extremes.
- **Boundaries:** train cutoffs precede 2025-06-01; validation is
  `[2025-06-01, 2025-12-01)` and test `[2025-12-01, 2026-06-01)`. Every last label date
  must be strictly earlier than its split's upper boundary.
- **Shared evaluation source:** A/B retain separate training histories, but both
  validation/test use the existing **2016-01-01-start bar store** with the same
  profile, universe, revision, and end date. Identical date ranges do not make two
  downloaded snapshots identical. There is no A/B intersection or fallback to a
  different snapshot. Missing shared data stops the workflow without downloading it.

Runtime builds a compact `prepared/sample-universes/` range index in bounded
parallel workers over immutable prepared bars. CPU quota, available memory, and
per-worker estimates bound workers and pending tasks. `FIN_TS_CLEANING_WORKERS`
sets the worker cap (default 8); completed buckets resume after interruption.
No full windows/labels are copied and no raw data, CPU-preparation manifests, or
completed results are rewritten. `universe.json` records candidates, accepted
windows, exclusion reasons, and retained extremes per split/market; `parts/*.json`
contains per-symbol audits. Excluding unexecutable future labels does not make the
result a tradable backtest free from halt or delisting risk.

Changing these rules changes training membership and requires a new matching baseline
for each A/B history, prediction task and baseline numerical contract. Old weights/results
remain but are not fair references for the new population. Changing only LoRA/unfreezing
capacity or main-model seed reuses a matching baseline.
Existing market data is reused without rerunning CPU preparation or downloading data.

<a id="splits-en"></a>

#### Fixed Train/Validation/Holdout periods

Production Stage 1/2 use fixed exclusive dates, independent of dataset size or its
earliest observation:

| Split | Forecast date `t` | Purpose |
| --- | --- | --- |
| Train | Start (default `2016-01-01`) ≤ `t` < `2025-06-01` | Training and label/feature scale calibration |
| Validation | `2025-06-01` ≤ `t` < `2025-12-01` | Checkpoint ranking and early stopping |
| Holdout / test | `2025-12-01` ≤ `t` < `2026-06-01` | Final post-training model and baseline evaluation |

Every horizon's actual `label.end_at` must be strictly before its split end,
including the holdout end `2026-06-01`. Approximately the final 14 trading dates
of each interval therefore cannot be full-horizon forecast origins. Fixed-date
mode does not additionally apply the legacy 20-day purge/14-day embargo.
Historical inputs may cross a validation/holdout start because those observations
were already available at prediction time; forward labels may not cross the end.
Dates from `2026-06-01` onward are excluded from all three splits, reserving
June–August for subsequent backtests.

CPU preparation and local readiness validate the prepared manifest's contract, including
at least **80 distinct forecast dates per market in each evaluation split**, recorded in
`split_audit.dates_by_market`. This checks prepared data. The cleaned runtime population
must separately be read from sample-universe audits and actual evaluation results;
prepared candidate counts are not post-cleaning counts. This is an
engineering coverage floor, not a guarantee of significance or all-regime
generalization. Many stocks are not equally many independent time observations.

Dates and storage-preparation semantics belong to dataset identity. Historical proportional
splits require a namespace prepared for the fixed-date contract, not a relabelled ready marker.
Once these fixed splits are prepared, changing stage, A/B selection, or capacity does not
require CPU preparation again. Runtime-cleaning changes rebuild compact sample indexes and
the affected baselines, not market data or the bar store.

<a id="pipeline-en"></a>

### Offline data pipeline

Acquisition and model training are separate phases:

```text
External APIs
  │
  ▼
Immutable raw JSON cache
  │
  ▼
Canonical daily OHLCV Parquet + download-manifest.json
  │
  ▼
Resumable symbol bar store + prepared candidate cutoff ranges
  │
  ▼
bar-store/index/ranges + dataset-manifest.json
  │
  ▼
Runtime continuity/liquidity sample-universe index
  │
  ▼
Lazy DataLoader builds contexts/labels on demand (fully offline)
  │
  ▼
Stage 1 / Stage 2 training
```

Raw OHLCV is written incrementally as immutable compressed Parquet. CPU
preparation never expands every 128-bar window and never persists labels. It
coalesces fragmented source row groups into coarse scan partitions of roughly
`4 * batch_rows`, streams at most `batch_rows` at a time, builds temporary bucket
parts on Pod-local `/tmp`, and atomically publishes only one indexed Parquet per
source partition to the Network Volume. Compaction reads exact bucket row groups
from `scan-index.json`, without listing tens of thousands of segment directories.
It then uses 128 hash buckets to build a compressed, symbol-oriented bar store
with one Parquet row group per symbol, plus small `symbol-index.parquet` and
contiguous valid-cutoff ranges. Every partition, compaction, quality, and split
bucket has an atomic checkpoint. At max runtime the Pod exits as
`waiting_for_preparation`; the next CPU Pod in the same dataset namespace skips
completed scan partitions and buckets. `.work` partitions are reclaimed only
after `_SUCCESS.json` is published.

Dataset requests automatically map to `/runpod-volume/datasets/<dataset-request-sha256>/`;
users do not supply DATA_ROOT. CPU preparation does not expand full windows; runtime cleaning
builds the eligible-sample index afterward.

<details>
<summary>Download cache, quota and preparation integrity rules</summary>

The downloader provides:

- Provider-specific QPS throttling.
- Independent parallel EODHD, TWSE, and TPEx loops with exponential backoff for
  retryable requests.
- Estimated HTTP requests for the complete plan are informational capacity data.
  The final Taiwan plan uses official benchmark sessions and records its
  pre-calendar weekday upper bound separately; neither is an admission gate.
- `max_api_calls` limits only EODHD network attempts, including retries, in one
  CPU attempt. TWSE/TPEx have no request-count ceiling. EODHD, TWSE, and TPEx
  each use the same default one-minute `--maxBackoff` value as their independent
  backoff exit boundary. Cache hits do not consume the EODHD counter; incomplete
  work is saved for another Pod.
- `--max-api-calls`, `--eodhd-qps`, `--taiwan-qps`, and `--maxBackoff` are
  acquisition policy values for one CPU Pod launch. They are set by
  `runpod_workflow.sh cpu prepare`, are not part of the immutable dataset
  selection, and cannot change dataset request identity.
- One provider exiting never cancels another. Each completed provider first
  atomically publishes a SHA-256 checkpoint bound to the dataset request,
  training security scope, and materialization revision. Only after every
  provider loop exits does the workflow publish a resume state or merge the
  validated provider checkpoints in deterministic order.
- A dataset-contract or date change creates a new immutable namespace. That
  namespace may read matching cache revisions and request identities from raw JSON caches in older
  dataset namespaces, while its Parquet files, manifests, and progress remain
  confined to the new namespace.
- Neither QPS nor `max_api_calls` represents a provider's daily/weekly quota or
  EODHD's endpoint-specific billed call units.
- After a temporary provider failure, HTTP 429, per-attempt request-budget
  exhaustion, or acquisition-time exhaustion, successful raw responses and
  completed provider-materialization checkpoints remain durable.
  `download-progress.json` is updated, and a later CPU Pod reuses completed
  providers while only incomplete providers replay cache and request misses.
- The CPU workflow reserves 25% of max runtime for canonical cleaning and
  symbol bar-store/index construction by default, capped at 2 hours and explicitly configurable
  through `--prepareReserve`. Once
  acquisition completes, raw Parquet, the request log, and the download manifest
  become a durable `downloaded` checkpoint. A later Pod can skip every API call;
  it resumes durable bucket checkpoints; only a verified bar store, cutoff
  ranges, and readiness can become `ready`.
- Cache identities, request logs, and manifests that exclude API tokens.
- A raw cache and direct download/preparation CLIs that refuse silent overwrite.
- SHA-256, row-count, and provenance bindings for artifacts.

Training accepts only a complete shard/index/range contract with `_SUCCESS.json`
and a dataset manifest whose state is `ready`. CPU readiness and training
preflight verify every shard's size and SHA-256, not only the small indexes.
Training, evaluation, and inference never call EODHD, TWSE, TPEx, or Massive.

</details>

<a id="training-en"></a>

### Training and evaluation protocol

#### Stage 1/Stage 2 and dynamic sampling

| Item | Stage 1 | Stage 2 |
| --- | --- | --- |
| Purpose | Bounded workflow/model smoke run | Formal experiments on the full training history |
| Sample presentations per epoch | 5% of eligible train count, capped at 500,000 | Full eligible train count |
| Training sampler | Annual-decay dynamic sampling, not one fixed 5% membership | Annual-decay dynamic sampling, not exactly-once traversal |
| Epoch limit | 2 | 5 |
| Validation cadence | Full validation at 20%/40%/60%/80%/100% of each epoch | Same |
| Early stopping | Five non-improving evaluations of the configured monitor plus two minimum-LR intervals, active from epoch 1. Current independent tasks use normalized Smooth L1 / CE | Same |
| Stored results | Best five validation-ranked full checkpoints plus compact completion weights | Same |
| Validation / test | Full eligible populations, no sampling, padding or drop_last | Same |
| Standard config | `configs/stage1_kronos_base_lora.yaml` | `configs/stage2_kronos_base_lora.yaml` |
| Initialization | Original pretrained base | Original pretrained base, not Stage 1 weights |

The training DataLoader uses a bounded-state dynamic sampler over valid cutoff ranges. It
loads one symbol row group on demand, constructs aligned 128-bar asset and
benchmark contexts, and computes task-required alpha or absolute-stock return labels from
`h_start` through holding day 14 in memory. No per-window or per-label dataset is written. Stage 1 presents
5% of the valid-train count per epoch, capped at 500,000; Stage 2 presents the full
valid-train count. Annual decay means newer windows can repeat and older windows
need not appear in the same epoch. Seed and epoch determine reproducible order and
resume position. Both use fixed-size batches and record deterministic final-batch
padding. The compact date/market index is memory-mapped, not an in-memory expansion
of every window. This out-of-core design
handles long full-market history without loading all bars or all potential windows
into RAM.

The standard stage configs retain `quantiles` and have matching `model_architecture_digest()` values. Current tasks
use explicit experiment YAML files; see [task selection](#train-en). [Resume](#resume-en)
continues one interrupted run, not Stage 1 weights into Stage 2.

Production Stage 1/2 rejects `max_steps`, fixed-step validation cadence, and a
separate fixed-step checkpoint cadence. The optimizer budget is derived only
from the per-epoch presentation budget, batch size, gradient accumulation, and epoch count. Every
epoch-relative validation participates in the best-five checkpoint ranking.
Normal completion and early stopping both atomically publish one
`completion-result/` containing the current trainable weights, resolved config,
stop reason, actual step/sample counts, and final validation metrics, without
duplicating optimizer/scheduler state that is no longer needed for resume.

#### Objectives and learning rates

Every monitor uses full validation and is minimized. A shared key does not imply a shared score definition.

| Task / experiment family | Checkpoint monitor | Score definition |
| --- | --- | --- |
| Current `alpha_point` | `primary_5d/selection_score` | Mean normalized Smooth L1 across all horizons |
| Current `absolute_direction` | `primary_5d/selection_score` | Mean absolute-stock three-class CE across all horizons |

##### Current independent point and absolute-direction objectives

Independent `alpha_point` minimizes only
`mean(SmoothL1((prediction − alpha) / train_scale, beta=1))`; independent
`absolute_direction` minimizes only fixed-label three-class CE. Both average valid samples
and all horizons equally, without pinball, ranking or legacy direction regularizers.
The stable `primary_5d/selection_score` monitor remains `min`, representing full-validation
normalized Smooth L1 or CE respectively, not just day five. Training logs
`train/normalized_smooth_l1` or `train/cross_entropy`, alongside `train/loss` and
`validation/loss`. Selection scores from different tasks are not directly comparable.
Both current presets use validation-plateau, task/LoRA LR `3e-5`/`5e-6`, dropout `0.05`
and weight decay `0.01`, not adaptive sample-plateau or market loss weights. Each task's
scheduler, sampling state and numerical scales persist in its own checkpoint for resume.

##### Train-only numerical scales

Stage 1/2 calibrate label robust scales on the same deterministic, at-most-50,000
sample from the full **train partition**, using seed 59 rather than shrinking it
with Stage 1's 5% sample-presentation budget. Feature median/IQR calibration uses the same
sample-count/seed contract. Aggregate caches bind the training dataset manifest
SHA; neither calibration reads validation or holdout.

##### Validation-plateau scheduling

The current independent tasks use validation-driven plateau: two non-improving validations after warmup multiply neural learning rates by 0.3,
down to 0.09 of their original values. Early stopping additionally requires two
completed training/validation intervals at the minimum rate. Checkpoints preserve
the plateau state; resuming or changing runtime batch plans never resets reductions.

#### Full validation/test and comparisons

The existing workflow name `validation stage` scores **holdout/test** for fixed-date
runs. The full model recomputes predictions rather than reusing checkpoint validation.
Every routine validation and final test visits its entire split, with
`evaluation_max_samples` and `baseline_max_samples_per_split` set to `null`.
Predictions leave the GPU batch by batch and use disk-backed aggregation with exact
sample weighting. The main model and task-matched baselines for one training data group share
train-calibrated label scales where applicable and verified ordered symbol/date SHA-256.
A/B train-only scales can differ, so cross-group normalized pinball or Smooth L1 comparisons
must also check their denominators.
Their evaluation populations must match exactly, not merely have the same date limits.

Full evaluation enumerates every valid window in fixed `symbol → cutoff` order.
The cleaned runtime `prepared/sample-universes/<identity>/cutoff-ranges.parquet`
contains exact valid ranges, not estimated counts. Old CPU-preparation candidate
counts cannot substitute for this actual evaluation population. The dataset retains compact indexes and bounded instrument caches, building
contexts/labels only when the DataLoader fetches a batch; no giant window dataset
is expanded. The final short batch is retained without padding or duplication.
A numeric legacy `evaluation_max_samples` now warns and is ignored, never enabling
sampling. This exhaustive traversal applies only to validation/testing: training
retains dynamic sampling, epoch-wise random order and same-date/market grouping,
with windows constructed dynamically at batch-read time.

Final testing remeasures inference batch sizes on the current GPU without backward
passes or reusing another GPU's training plan. Prefetch accounts for measured
inference/training speed, RAM, container shared memory, live worker pools and pinned
copies. Multiple workers, pinned memory and asynchronous transfers are used.
The result's `execution` section records actual counts, batch size, workers,
prefetch, throughput and peak GPU memory; these are runtime measurements, not
predictive-performance scores.

Reports include month, market and horizon breakdowns. Paired full-model-minus-baseline
comparisons use datewise mean loss with equal horizon weights: normalized pinball for
quantiles, normalized Smooth L1 for alpha point, or CE for absolute direction. Their 95%
intervals use a 14-date circular moving-block bootstrap (1,000 draws, seed 42), not
independent stock-row resampling. These are loss-difference intervals, **not** RankIC or
balanced-accuracy intervals. Intervals are not adjusted for multiple comparisons. Holdout rankings
are descriptive, never a source for checkpoint selection or repeated tuning,
and do not establish coverage of every future market regime.

Metrics are task-specific; fields from another task are not substitutes:

| Task | Quality metrics | Outputs/metrics not supplied |
| --- | --- | --- |
| `alpha_point` | Normalized Smooth L1, MAE/RMSE/normalized MAE, bias, correlation and validity status, alpha-sign agreement, date-level RankIC by market/horizon | Quantiles, coverage, absolute-stock class probabilities, long-short performance, Sharpe, five-level trading signals |
| `absolute_direction` | CE, Brier, accuracy, balanced accuracy and missing-class status, macro-F1, opposite error, ECE, majority accuracy, confusion, class supports/precision/recall/predicted shares | Alpha classification, point-return forecasts, RankIC, quantiles or portfolio performance |

Independent-task model/baseline comparisons appear in `comparison.rows`. Archived output and
metric definitions are separate in the [obsolete-experiment appendix](#historical-experiments-en).
RankIC uses standard Spearman average ties, marked `rank_ic_method=spearman_average_ties_v1`.
A date with constant predictions or targets has no valid correlation; row order cannot manufacture
a nonzero RankIC. `rank_ic_status`, `rank_ic_evaluated_dates` and `rank_ic_ir_status` distinguish
valid estimates from retained numeric placeholders. Historical results without the method marker
are not automatically recomputed or assumed to use the same tie definition.

#### Prerequisite baseline building and reuse

Baselines are an independent prerequisite for each matching prediction task and dataset.
Rules, GBDT, GRU, DLinear and PatchTST use full train/validation/test. Historical quantile or
dual-cost-direction results cannot replace baselines for the current independent tasks.
See [baseline operations](#baseline-en) for launch commands.

##### Current independent tasks

| Task | Neural baseline fitting | GBDT fitting | Full-validation selection and final test |
| --- | --- | --- | --- |
| `alpha_point` | Train-scale-normalized Smooth L1, with the same beta as the main model | `absolute_error` (MAE), not Smooth L1 | Shared evaluator selects using normalized Smooth L1; complete regression/sign/RankIC metrics |
| `absolute_direction` | Absolute-stock CE, no alpha classifier | One log-loss classifier per horizon | Same inclusive ±1% neutral band; CE selection and complete probability/class metrics |

Fixed rules supply task-aligned point/hard-class forecasts, not trained classifiers. Probability
scores use deterministic predictions with an explicit `0/-30` logit floor, not learned or
calibrated probabilities. Neither current task trains quantile, ranking or dual-cost-direction
heads or uses legacy direction loss references. Comparisons appear in `comparison.rows`.

##### Shared training, storage and reuse rules

Neural models validate five times per epoch for at most five epochs. GBDT validates after every
eight additional boosting rounds, up to 200, without sklearn's internal validation subsampling.
Both use five-evaluation patience and retain the best full-validation model. Neural models
also retain validation-plateau and minimum-LR observation rules; GBDT has no neural-optimizer
LR schedule and fixed rules have no early stopping. Neural baselines traverse the full train
population once per epoch using blockwise permutation. Neither these neural baselines nor GBDT
applies the main model's annual-decay weights. Equal populations, full evaluation and patience
therefore do not imply equal training weights, FLOPs or wall-clock costs; disclose the differences.

Weights/rule parameters, best-validation metrics, complete test predictions and metrics live
under `/runpod-volume/baselines/<baseline-id>/`; `complete.json` is published only after all
required results succeed. Main training requires this cache; main testing reads its metrics
**without retraining or re-running baseline inference**. Identity covers data periods/content,
cleaning/universe/horizon semantics, prediction task, point-loss beta or neutral threshold and
baseline numerical parameters/implementation, not main-model LoRA capacity, training seed,
README or deployment scripts. Matching dates alone cannot permit different tasks or numerical
settings. If full GBDT exceeds the memory budget, it fails instead of silently subsampling.

Validation/test share `inputs/<split>/metadata.npy` (membership) and `targets.npy`; absolute
direction also shares `absolute_targets.npy`. `evaluation_data` references these common arrays.
Each model stores its own predictions, weights and metrics, not duplicate membership/target arrays.
For independent tasks, `predictions.npy` has shape `[N,H]` for point forecasts or `[N,H,3]` for
absolute-direction scores, ordered down/neutral/up. Neural models store raw logits, GBDT stores
`log(max(probability, 1e-30))`, and rules store `0/-30`; the common evaluator uses softmax to
obtain probabilities. These are not all outputs of the same learned classifier structure.
Finalization streams verification, atomically publishes references, then removes
redundant copies. Interrupted cleanup resumes storage finalization without retraining; it does
not change periods, labels or baseline numerical identity. The main model's final
`validation-benchmark.json` uses result schema `6.0` and records `selection_split=validation`,
`evaluation_split=test` and, after paired checks pass, `test_unlocked=true`. These are not fields
of baseline `complete.json`; baseline completion requires its `state=complete` plus model/artifact
verification. Matching evaluation schemas still require matching tasks, eligible populations
and scales. Main-model seeds reuse matching results without CPU preparation or market downloads;
two independent tasks need two baselines, not six baselines for three seeds each.

<a id="operations-en"></a>

### Complete RunPod operations guide

#### Execution locations and safety boundaries

The local controller needs Bash, system Python 3, AWS CLI and curl; Taiwan-relay deployment
also needs Google Cloud CLI. Locally edit source, manage credentials/selections, transfer
artifacts and supervise Pods; do not load models. Stdlib control/contract tests, shell syntax
checks and standalone static analysis require no local ML environment.

Retain Poetry for Python package management. The cloud environment belongs at
`/runpod-volume/stock_forecasting/.venv`. Resolve dependencies and the canonical lockfile,
run full pytest/model/CUDA checks, prepare data and train on RunPod. Do not create or inspect
a local ML environment or run `poetry install`, `poetry lock`, `uv sync` or `uv lock`
locally. A leftover local lockfile is not evidence of the cloud runtime.

Normal operation uses configure and experiment names; do not edit `.env` or generated
selections/manifests manually. Developers may edit YAML for a new controlled configuration,
refresh its selection, sync while no Pod owns the volume, and start a new run. A numerical
contract change is not a resume of the old run. Remote Python is `>=3.12,<3.13`.
Dependency updates require the workflow's exclusive environment-write lease and do not
trigger market downloads or CPU preparation.

`bash scripts/runpod_workflow.sh` is the **local control entry point**.
`bash scripts/runpod_tmux_launch.sh` runs **inside the Pod after SSH login**.
Creating a Pod does not start training. Detached tmux survives SSH disconnection;
the local guard host must still remain powered and online to terminate its assigned Pod.

Replace `<RUN_ID>`/`<POD_ID>` with actual values. Supported options, defaults and aliases
are collected in the [CLI reference](#cli-reference-en).

##### Temporary agent access to RunPod: local SSH socket

**Use `runpod-ssh-socket.sh` only when you want an agent to help deploy, start
training, or inspect RunPod.** Run it from the project root:

```bash
bash runpod-ssh-socket.sh
bash runpod-ssh-socket.sh 8h
bash runpod-ssh-socket.sh --pod-id "<POD_ID>"
bash runpod-ssh-socket.sh 8h --pod-id "<POD_ID>"
```

`SOCKET_TTL` is an optional positional parameter defaulting to `4h`. It accepts
a positive integer followed by `m`, `h`, or `d`, such as `30m`, `8h`, or `1d`.
The new optional `--pod-id POD_ID` parameter connects only to the specified
running Pod and can be combined with TTL. If that Pod is absent, inaccessible,
or not running, the helper fails without switching to another Pod. `--help`
prints usage. The project location is derived from the script's directory,
so invocation from another working directory also works.

The helper uses the existing project `.env` (mode `600` or `400`) and REST v2
query wrapper. With no arguments, it automatically creates sockets for every
running Pod accessible to the configured RunPod API key. Providing only TTL
without `--pod-id` also connects to all running Pods, regardless of project name,
network volume, or workflow identity. `RUNPOD_NETWORK_VOLUME_ID` is not required.
Each Pod gets its own socket and expiration process, without terminal selection.
Direct SSH is preferred;
when no direct endpoint exists, it uses the API-provided RunPod SSH proxy. The SSH
public key must already be configured in RunPod. Proxy connections do not support
SCP/SFTP; see the
[RunPod SSH documentation](https://docs.runpod.io/pods/configuration/use-ssh).
Authentication first tries `~/.ssh/id_ed25519_runpod`. If it fails or is absent,
the helper tries other private keys under `~/.ssh/`, including subdirectories,
in sequence. Unlock encrypted keys with `ssh-add` beforehand. The helper does
not prompt for or store passwords or change SSH configuration.

On success it prints the socket, expiration, agent SSH command, and commands to
check or close access early. Give this output to the agent. The invocation index
is saved in `.runpod/ssh-socket/latest.json`: `connections` maps Pod IDs to
successful connection metadata, while `failures` records failed Pods and their
reasons. Every connection gets a separate private temporary directory. The agent
should check the corresponding `expires_at_epoch` and run `check_command` before
using `ssh_command`. That command reuses only the socket and fails if unavailable.
If some Pods fail, successful sockets remain usable, but the helper returns a
nonzero exit code so partial success is not reported as full completion.

Multi-Pod API queries and SSH authentication use a bounded thread pool with no
more pending futures than workers. Worker count accounts for Pod count, CPU cores,
available memory, and service load, capped at `4`. Set the environment variable
`RUNPOD_SSH_SOCKET_WORKERS` to lower concurrency. Memory planning reserves
`512 MiB` and estimates `64 MiB` for each retained connection and active worker.
If memory cannot be estimated or is insufficient for all connections, the helper
fails before connecting; use `--pod-id` to narrow the scope. Private keys for the
same Pod are still tried sequentially in priority order.

TTL limits acceptance of new shared connections. Expiration runs `ssh -O stop`,
allowing established SSH work and detached tmux to continue. Host sleep, reboot,
or a network interruption can end access early; expiration during sleep stops
new connections after the host resumes. This deadline is managed separately
from the Pod runtime and guard deadline. The helper creates access; deployment
and training still use the existing workflow and tmux entry points.

This file intentionally lives at the project root, outside the existing upload
allowlist. It is never uploaded to RunPod, added to the code file manifest, or
included in SHA calculation. Editing only this helper does not change workflow
identity or require source sync or data preparation. Connection metadata under
`.runpod/` is also excluded. The README itself retains its existing document
sync rules.

<a id="setup-en"></a>

#### 1. Create account resources, credentials and storage

The local control machine needs `bash`, Python 3, AWS CLI, and `curl`.
GPU Pod, CPU Pod, and network volume creation, along with Pod lookup, actions,
and termination, use RunPod REST API v2. CPU Pod vCPU count must be `2`, `4`,
`8`, `16`, or `32`. On-demand Pods have no provider-side runtime limit; this
project enforces deadlines with a local guard. This system Python runs dependency-free manifest and
JSON control helpers only; it does not create, load, or validate a local
project Python environment. Official RunPod references:

- [Network volumes](https://docs.runpod.io/storage/network-volumes)
- [S3-compatible API](https://docs.runpod.io/storage/s3-api)
- [RunPod Secrets](https://docs.runpod.io/pods/templates/secrets)
- [REST API v2](https://api.runpod.io/v2/openapi.json)

In the RunPod Console, create a project-scoped RunPod API key and a separate S3
API key. Then create these fixed-name RunPod Secrets:

- `huggingface_token`: required to prefetch the pinned Kronos model and
  tokenizer revisions.
- `wandb_api_key`: required for training and validation tracking.
- `eodhd_api_token`: required only by the `us_only_eodhd` and
  `us_tw_eodhd` profiles; it is not required by `tw_only`.

Profiles containing Taiwan data also require a TPEx Cloud Run relay in GCP
`asia-east1` (Taiwan). Each verified relay deployment creates a uniquely named
`tpex_relay_token_<timestamp>_<nonce>` RunPod Secret and records that secret
name in the local `.env`; do not create a fixed-name TPEx secret manually.

Do not copy, open, or manually edit `.env`. Create the credential-only file
through hidden input; the script writes it atomically with mode `600`:

```bash
bash scripts/runpod_workflow.sh credentials
```

Create the network volume through the script. The returned volume ID,
datacenter, S3 region, and endpoint are written back to the same `.env`
automatically:

```bash
bash scripts/runpod_workflow.sh volume deploy \
  --name stock-forecasting \
  --size-gb 100 \
  --datacenter EU-RO-1
```

If `.env` already registers a volume, the script will not create another
potentially billable volume by default. Only an intentional `--force-new`
creates and registers a replacement.

<details>
<summary>Required for Taiwan profiles: deploy and verify the TPEx Cloud Run relay</summary>

When a RunPod datacenter receives HTTP 403 from TPEx data endpoints, deploy the
restricted Cloud Run relay before using a Taiwan-market profile. Create a
dedicated GCP project with billing enabled, install the Google Cloud CLI, and
authenticate the deployment account. No Cloudflare token, GCP API token,
custom subdomain, Pub/Sub topic, or relay storage is required:

```bash
gcloud auth login
```

The deployer enables the Cloud Run, Cloud Build, Artifact Registry, Secret
Manager, and IAM APIs; creates a dedicated runtime service account; grants
`roles/run.builder` to the Compute Engine default account used for the source
build; creates or updates one Secret Manager secret; and configures
unauthenticated network ingress. The GCP principal running it must therefore
be authorized for those administrative mutations. Project Owner is acceptable
for first-time setup in a new personal project dedicated to this relay; a
shared or production project should use an equivalent least-privilege set.
Google lists `roles/run.sourceDeveloper`,
`roles/serviceusage.serviceUsageConsumer`, and
`roles/iam.serviceAccountUser` on the runtime identity as the base source
deployment roles. This script additionally needs permission to enable APIs,
create service accounts, manage the secret and its IAM policy, set the public
Cloud Run invoker policy, and grant the builder role. It does not grant these
administrative roles to the logged-in principal.

`--allow-unauthenticated` only makes the managed `run.app` HTTPS endpoint
reachable from RunPod. The application still requires a length-bounded shared
token. An organization policy that blocks unauthenticated Cloud Run causes a
fail-closed deployment; never replace application authentication with an open
general-purpose proxy.

```bash
bash scripts/runpod_workflow.sh tpex-relay configure
bash scripts/runpod_workflow.sh tpex-relay deploy
```

`configure` merges the GCP project ID, fixed `asia-east1` region, service and
secret names, and an automatically generated shared relay token into local
`.env`. The `gcloud` login stays in the local Google Cloud CLI credential store
and never enters `.env`, source, or a Pod. Cloud Run provides a managed
`run.app` hostname; there is no custom-domain prompt.

The update preserves all existing RunPod volume, S3, API-key, and activated
relay values, as well as legacy Cloudflare fields during migration. The new
workflow does not use those Cloudflare values or delete the old Worker/token.
Keep the old deployment until Cloud Run passes live verification and a CPU Pod
has succeeded. Do not rerun `credentials` or `volume deploy` when the volume
already exists.

`deploy` uses the existing `RUNPOD_API_KEY` in the local `.env` with Bearer
authentication for RunPod REST API v2. Before any GCP write, it checks access
with read-only `GET /v2/account/secrets` and creates no resource during that
preflight. After Cloud Run verification, it creates the Secret with
`POST /v2/account/secrets`. The control script sends an explicit project
API-client `User-Agent`. If the API rejects a request, it preserves the HTTP
status and safe error details while redacting the API key and relay token. The
API key remains in the local `.env` and is excluded from source
synchronization. If the preflight or Secret creation fails, `deploy` stops
without activating new relay metadata in the local `.env`.

Only after the REST API v2 preflight does the deployer mutate GCP. The shared token
is mounted from a numbered Secret Manager version into one Cloud Run revision;
it never uses a drifting `latest` reference. Before a revision is produced, the
Node.js buildpack must pass the relay unit tests through `gcp-build`; a failed
test or build cannot deploy a revision. After the revision is live, the deployer
verifies authenticated warmup and then all four exact routes:
`dailyQuotes`, `exDailyQ`, `ROE`, and `inx`, with two seconds between official
route probes. Only official table-shaped JSON from every route allows it to
create a new RunPod Secret and atomically activate
`TPEX_PROXY_URL` plus the secret reference in local `.env`. A live-verification
or RunPod Secret failure can leave the new Cloud Run revision deployed, but the
workflow remains incomplete and the previous local URL/reference stays active.
Fix the reported cause and rerun `tpex-relay deploy`.

- Source deployment uploads [`cloudrun/tpex-relay`](cloudrun/tpex-relay) with
  GCP `asia-east1` (Taiwan), Node.js 22, 1 vCPU, 512 MiB, a 60-second request
  timeout, request-based CPU throttling, and startup CPU boost.
- Service-level minimum instances is `0`, maximum instances is `1`, and
  container concurrency is `1`. It scales to zero while idle and cannot
  multiply the existing `--taiwan-qps` through autoscaling. The relay does not
  add a second fixed QPS limiter that could conflict with the CPU CLI setting.
- It accepts only `GET`, the shared token, the fixed TPEx origin, the four paths
  used by this project, and each path's exact query schema. One upstream request
  has a 30-second total timeout and a 16 MiB response limit. Redirects are
  limited to three same-origin hops; bounded session cookies can be carried to
  the next same-origin hop but are never returned to the caller. Cross-origin,
  missing-Location, cookie-limit, and no-progress loops fail closed. The relay
  never performs provider retry and is not a general or open proxy.
- A successful 2xx response must parse as a JSON object, while the relay returns
  the original bytes rather than rewriting the official payload. A bounded
  non-2xx response retains its status and body so the existing provider-level
  exponential backoff decides when to stop.

This MVP uses no Pub/Sub, database, Cloud Storage, or static outbound address.
Cloud Run uses dynamic default egress. Although `asia-east1` is a Taiwan
region, region selection does not guarantee that TPEx will accept every egress
address, so four-route live verification is the deployment gate. If 403 later
varies by egress address, evaluate Serverless VPC Access plus Cloud NAT for a
static address. Move to a Taiwan domestic VPS only if TPEx rejects all tested
GCP Taiwan egress.

With request-based billing and minimum instances `0`, idle Cloud Run instances
do not incur compute charges. Requests, source builds, Artifact Registry image
storage, Secret Manager, and network traffic are still governed by their own
GCP pricing and free allowances; the service is not guaranteed to be entirely
free. Inspect control-plane state or rerun complete live verification at any
time:

```bash
bash scripts/runpod_workflow.sh tpex-relay status
bash scripts/runpod_workflow.sh tpex-relay verify
```

The TPEx client still hashes the original `https://www.tpex.org.tw` endpoint
and public query parameters for request identity. The Cloud Run URL, relay token,
and transport mode do not enter raw-cache keys or dataset-request identity.
After switching transports, all successful TWSE, TPEx, and EODHD JSON cache
entries remain reusable; only missing TPEx responses pass through the relay.
The CPU workflow calls authenticated `/_internal/warmup` immediately before
`stock-forecasting-download`, and only when provider acquisition is actually required.
Warmup never contacts TPEx. It is intentionally not placed at tmux startup,
because the complete pytest suite and Hugging Face prefetch could let the relay
scale back to zero before acquisition. Reusing a complete raw checkpoint skips
even the warmup request.

The old `tpex-proxy configure|deploy|verify|status` workflow name remains as a
compatibility alias, but it dispatches to the Cloud Run scripts. Use
`tpex-relay` for new operations.
Official references:

- [Install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- [Cloud Run locations](https://docs.cloud.google.com/run/docs/locations)
- [Deploy Cloud Run from source](https://docs.cloud.google.com/run/docs/deploying-source-code)
- [Node.js buildpack and `gcp-build`](https://docs.cloud.google.com/docs/buildpacks/nodejs)
- [Cloud Run IAM roles](https://docs.cloud.google.com/run/docs/reference/iam/roles)
- [Cloud Run autoscaling](https://docs.cloud.google.com/run/docs/about-instance-autoscaling)
- [Cloud Run minimum instances and billing](https://docs.cloud.google.com/run/docs/configuring/min-instances)
- [Cloud Run Secret Manager integration](https://docs.cloud.google.com/run/docs/configuring/services/secrets)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
- [RunPod REST API v2 OpenAPI specification](https://api.runpod.io/v2/openapi.json)

</details>

<a id="configure-en"></a>

#### 2. Choose data/configuration and synchronize

Run from the local project root. With no arguments, `configure` is interactive; options and
fixed examples follow.

##### `configure` options and dataset scope

`--universe`, `--stocks`, `--etfs`, and `--symbol-limit` control **only the US
dataset scope**; they never filter Taiwan instruments. `us_tw_eodhd` always
combines the EODHD US target-security scope selected by `--universe` with the
TWSE/TPEx data that satisfies the common-stock/TDR/audited-unleveraged-equity-
ETF contract over the selected date range.

Available `--data-profile` values are:

| Value | US data | Taiwan data | Requires `eodhd_api_token` |
| --- | --- | --- | --- |
| `tw_only` | None | TWSE/TPEx common stocks, TDRs, audited benchmark-mappable unleveraged equity ETFs, and official benchmarks; `--universe` must be `all` | No |
| `us_only_eodhd` | EODHD common stocks (including ADRs) and audited benchmark-mappable unleveraged equity ETFs selected by `--universe`, plus the automatically added `VTI.US` benchmark | None | Yes |
| `us_tw_eodhd` | The same US security scope as `us_only_eodhd` | The same Taiwan target-security scope as `tw_only` | Yes |

Available `--universe` values are:

| Value | Meaning | Compatible options |
| --- | --- | --- |
| `all` | For profiles containing US data, use EODHD discovery for active and delisted common stocks/ADRs, then apply the audited unleveraged-equity-ETF allowlist. For `tw_only`, use the complete Taiwan target-security scope | US profiles may optionally use `--symbol-limit`; do not provide `--stocks` or `--etfs` |
| `explicit` | Restrict **only the US scope** and require at least one `--stocks` or `--etfs` value. The workflow still adds `VTI.US` automatically | Only `us_only_eodhd` and `us_tw_eodhd`; cannot be combined with `--symbol-limit` |

Here, the “complete US target-security scope” means active/delisted common
stocks (including ADRs) returned by EODHD discovery and the unleveraged equity
ETFs in the audited program allowlist. Use a US-containing profile with
`--universe all` and omit `--symbol-limit` entirely. Leveraged, inverse, bond,
commodity, and volatility ETFs cannot become training targets, and an explicit
benchmark mapping cannot bypass this restriction. The workflow cannot guarantee
instruments that the provider omits or the account cannot access.

All user-facing `configure` options are:

| Option | Values or format | Meaning and restrictions |
| --- | --- | --- |
| `--stage` | `stage1`, `stage2` | Select the fixed training config. Required in non-interactive mode |
| `--data-profile` | `tw_only`, `us_only_eodhd`, `us_tw_eodhd` | Select the actual provider and market combination. Required in non-interactive mode |
| `--dataset-revision` | 1-64 characters; start with an alphanumeric character, followed by alphanumerics, `.`, `_`, or `-`; default `v1` | Use a new label to force a new immutable dataset namespace after a provider revises historical data |
| `--start` | `YYYY-MM-DD`; default `2016-01-01` | Inclusive start shared by all markets; must precede `2025-06-01` and leave enough context/training history |
| `--end` | `YYYY-MM-DD`; no default | Exclusive end shared by all markets; must be at least `2026-06-01`. Later rows never enter the fixed train/validation/holdout splits |
| `--h-start` | `1`, `2`, or `3`; default `1` | First cumulative holding-day horizon through day 14; entry stays at the `t+1` raw open. The task selects alpha or absolute direction; changes runtime labels/model output, not dataset identity |
| `--feature-mode` | Production uses `scales` or `combined`; default `combined` | Other modes require custom/legacy configs with explicit output scale disabled; dataset namespace is unchanged |
| `--universe` | `all`, `explicit` | Select the US-instrument strategy; `tw_only` accepts only `all`. Required in non-interactive mode |
| `--stocks` | Comma- or space-separated US tickers; repeatable | US common stocks/ADRs in `explicit` mode, such as `"AAPL,BABA"`; provider type is verified against EODHD discovery and does not affect Taiwan data |
| `--etfs` | Comma- or space-separated US tickers; repeatable | Unleveraged US equity ETFs in `explicit` mode, such as `"SPY,QQQ"`; each ticker must be in the audited allowlist and does not affect Taiwan data |
| `--symbol-limit` | Positive integer N | **A bounded capacity/workflow check, not a complete-US-market mode.** Only valid for a US-containing `all` profile. After discovery, split allowlisted unleveraged equity ETFs from common stocks/ADRs, then keep up to N of each by active → delisted and ticker order; take all when a type has fewer than N. This is neither random nor representative sampling. Then ensure the required `VTI.US` benchmark is present: do not duplicate it if it is among the N ETFs, otherwise add it, so the raw universe is at most `2N+1`. Omit this option for the complete US target-security scope |
| `--interactive` | Flag with no value | Explicitly open the interactive prompts; invoking `configure` with no options enables this mode automatically |
| `--reuse-current` | Flag with no value | Refresh the config snapshot using the existing active selection, preserving its dataset scope, stage, horizon and feature mode. Cannot be combined with `--interactive`; previous selection values take precedence over other selection options. It fails if refreshing would change dataset identity |

Interactive defaults are `stage1`, `us_tw_eodhd`, dataset revision `v1`, start
date `2016-01-01`, `h_start=1`, `feature_mode=combined`, and US universe `all`. `--end` intentionally has no default:
leaving its prompt blank asks again instead of selecting the local current date.
Provider acquisition policy is not configured here. Supplying `--max-api-calls`,
`--eodhd-qps`, `--taiwan-qps`, or `--maxBackoff` to this command is rejected as
an unknown option instead of creating another selection. The lower-level helper's
`--project-root` is injected by `runpod_workflow.sh`; it is not a user-facing
dataset-scope option and should not be supplied manually.

Use `stage1` to prepare a dataset and model cache for the first time; this does not require
training a Stage 1 model. For already-prepared data, select `stage2` without another CPU Pod.
The complete US/Taiwan B-group dataset is configured as follows:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_tw_eodhd \
  --dataset-revision v1 \
  --start 2016-01-01 \
  --end 2026-06-01 \
  --h-start 1 \
  --feature-mode combined \
  --universe all
```

For A, change `--start` to `2021-01-01` and preserve all other dataset values.
**A still evaluates on shared B data, so both datasets must be prepared.** Task presets
such as `--experiment a-signal64` belong to the later train command, not configure.

Example for named US instruments with no Taiwan data:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage1 \
  --data-profile us_only_eodhd \
  --start 2016-01-01 \
  --end 2026-06-01 \
  --universe explicit \
  --stocks "AAPL,MSFT" \
  --etfs "SPY,QQQ"
```

This contains the four named US instruments and required `VTI.US` benchmark.
Changing the profile to `us_tw_eodhd` adds the complete eligible Taiwan universe,
not just four instruments. Taiwan-only uses `--data-profile tw_only --universe all`
without `--stocks`/`--etfs`. These are alternative dataset choices, not a sequence
to copy over an already-prepared selection.

Provider quota, QPS and new API-attempt budgets are configured when creating a CPU Pod,
not in configure. Set `--max-api-calls` from remaining account quota; a larger complete
request plan resumes across attempts instead of shrinking the universe. Check
[EODHD Pricing](https://eodhd.com/pricing),
[API Limits](https://eodhd.com/financial-apis/api-limits) and the
[User API](https://eodhd.com/financial-apis/user-api) for current account limits.
Program defaults do not establish subscription entitlements.

The script creates `.runpod/selections/<selection-id>.json` and
`.runpod/active-selection.json`. They contain no secrets and are excluded by
`.gitignore`. Selection schema 3 binds the complete `h_start` preparation
contract. Older selections are not migrated automatically, so rerun `configure`
after updating the code. Changing `--end` intentionally creates a new dataset request
namespace; existing network-volume files are not deleted. Selection identity includes stage/config and dataset choices; dataset-request identity
only includes profile, dates, universe, symbol limit, revision and storage-preparation
semantics. **Changing LoRA or main-model learning rates does not create a dataset.**
These identities serve different purposes.
QPS, API budget, and maximum backoff are recorded only in CPU launch metadata,
download progress, and the download manifest; they never enter selection identity.

Changing only `h_start` creates a new training selection SHA, while `h_start=1`,
`2`, and `3` share the same dataset request SHA, raw Parquet, symbol bar store,
quality cutoff ranges, and split audit. The DataLoader selects the corresponding
`h_start...14` labels at training time, and train-only robust scales are sampled
dynamically from the train split when that run starts, then persisted in every
checkpoint for exact resume, evaluation, and inference restoration. Switching `h_start`
therefore neither rebuilds data nor scans API caches and sends no provider
request. Launch preflight checks that selection's dataset manifest and artifacts without
requiring another CPU-finalization step. A different date range, symbol universe, provider request, or
`dataset-revision` creates a different dataset namespace.

When provider history may have been revised and a new snapshot is intentional
for the same profile/date/universe, pass a new explicit
`--dataset-revision <label>` to `configure`. The CPU workflow never overwrites a
complete existing namespace and resumes matching incomplete work rather than mixing incompatible partial data.

Inspect the active selection without opening JSON:

```bash
bash scripts/runpod_workflow.sh selection show
```

`.env` stores only local RunPod/S3 credentials, GCP relay metadata, the relay
shared token, and script-managed volume, RunPod Secret reference, and TPEx
Cloud Run URL values. Stage, data range,
runtime, and config are never read from
`.env`. Do not `source .env`, and never put API keys, tokens, or secret values
in the README, configs, shell scripts, or commit history. Pods receive RunPod
Secret-resolved relay tokens and a non-secret `run.app` URL. The local `gcloud`
credential stays in the Google Cloud CLI credential store; account-level
RunPod, S3, and GCP deployment credentials never enter a Pod.

##### Verify S3 and upload source code

Run the read-only S3 access check, then preview the explicit upload allowlist:

```bash
bash scripts/verify_runpod_s3_access.sh
bash scripts/runpod_workflow.sh sync --dry-run
```

After reviewing the list and confirming no Pod owns the volume, upload and verify code readiness:

```bash
bash scripts/runpod_workflow.sh sync --apply
bash scripts/runpod_workflow.sh readiness --code-only
```

The uploader scans allowlisted source, configs, scripts, tests, `README.md`, and
`pyproject.toml` for secret patterns, uploads and size-checks every file, and
only then publishes `lifecycle/stage1/code.json`. It does not upload `.env`,
caches, data, checkpoints, or local artifacts. It also excludes `poetry.lock`;
the approved RunPod image regenerates that file for its Python/PyTorch/CUDA
environment on the network volume.

After any allowlisted source or config change, rerun `--dry-run`, `--apply`, and
the readiness check. After editing the selected config, refresh with `configure --reuse-current`,
which preserves dataset scope and rejects a changed dataset identity. Editing another
unused config does not change this selection. Switching selections or changing non-data code does not require
CPU preparation again when the selected dataset is already prepared and its data
content/preparation contract is unchanged. The GPU gate still verifies that
dataset's integrity and the current model's offline cache; never bypass it.

##### Switch stage or A/B after preparation

Use `selection show` to verify the original scope. Changing stage, h_start or feature mode
does not require CPU finalization; profile, revision, dates and universe must still select
a prepared namespace. For Stage 1 → Stage 2 on the same data, retain the original dataset
arguments, change only `--stage stage2`, and synchronize the selection as described here.
Do not run `cpu prepare --max-api-calls 1` for this switch.

Once task YAML files are uploaded, choose
A/B and a current task with
`train --experiment`. The launcher publishes that experiment's immutable selection without
changing the global active selection or uploading source again. Missing data/baselines fail
before paid creation; complete only the missing prerequisite, never rewrite manifests or
redownload existing data. Resume/validate restore selection by run ID; see [resume](#resume-en).

<a id="cpu-en"></a>

#### 3. Prepare new data or resume incomplete CPU work

**Skip this step for a dataset whose preparation is complete.** Configure `stage1` for initial
preparation: the current creator maps `stage2` to `cpu-finalize`, which revalidates existing
artifacts and cannot build a missing bar store. No Stage 1 model training is required.
If only stage/capacity changes for prepared data, proceed to baseline/train.

The CPU Pod accepts only the active selection. Pod creation fails before any
compute is rented when `configure` has not run, the config SHA changed, or the
selection JSON is incomplete. With no options, the local command is interactive:
it prompts for maximum workload runtime, the maximum additional EODHD network
attempts for this Pod, EODHD QPS, per-provider TWSE/TPEx QPS, time reserved for
data cleaning/bar-store construction, the maximum retry backoff shared by all three
providers, vCPU count, and CPU flavor. `--max-api-calls` has no Enter-to-accept
default and must be
entered explicitly from the account's current remaining quota. Pressing Enter
accepts 6 hours, 16 EODHD QPS, 0.5 QPS for each Taiwan provider, an automatic
reserve, 1 minute, 8 vCPUs, and `cpu3g`. The automatic reserve is 25% of max
runtime, capped at 2 hours; the six-hour default reserves 90 minutes.
The final prompt requires `y` or `yes` before creating a potentially billable
Pod; Enter, `n`, or `no` cancels safely:

```bash
bash scripts/runpod_workflow.sh cpu prepare
```

Providing any option selects non-interactive mode. This mode requires an explicit
`--max-api-calls`; omitted QPS and resource options retain their defaults.
Automation can therefore provide all values without editing `.env` or rerunning
`configure`:

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve 2h \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

Add `--interactive` to use command-line values as prompt defaults that the
user can confirm or replace:

```bash
bash scripts/runpod_workflow.sh cpu prepare \
  --interactive \
  --max-api-calls 80000 \
  --eodhd-qps 16 \
  --taiwan-qps 0.5 \
  --maxRuntime 10h \
  --prepareReserve auto \
  --maxBackoff 1m \
  --cpuNumber 16 \
  --cpuFlavor cpu5g
```

`--max-api-calls` is a positive integer counting only EODHD cache misses and
retries in this CPU Pod. It is not the account's total daily quota, and a new
Pod does not automatically subtract account usage by earlier Pods.
`--eodhd-qps` and `--taiwan-qps` must be positive, defaulting to `16` and `0.5`.
The Taiwan value is per provider, so concurrent TWSE and TPEx loops each have an
independent 0.5-QPS limiter. `--maxRuntime`, an explicit `--prepareReserve`, and
`--maxBackoff` accept a positive integer followed by `m`, `h`, or `d`.
`--prepareReserve auto` uses the
formula above; an explicit reserve must be shorter than max runtime.
`--maxBackoff` defaults to `1m` and applies to EODHD, TWSE, and TPEx. Only the
affected provider loop exits when its next exponential or `Retry-After` delay
would **exceed** this limit; a delay equal to the limit is still performed.
EODHD stops at whichever comes first: `--max-api-calls` exhaustion or this
backoff boundary. `--cpuNumber` must be `2`, `4`, `8`, `16`, or `32`.
`--cpuFlavor` accepts only the six RunPod values below; any other value fails
before Pod creation:

| Flavor | Generation | Type | RAM / vCPU | RAM at 32 vCPUs | Container-disk limit |
| --- | ---: | --- | ---: | ---: | ---: |
| `cpu3c` | CPU3 | Compute-Optimized | 2 GB | 64 GB | 10 GB/vCPU |
| `cpu3g` | CPU3 | General Purpose | 4 GB | 128 GB | 10 GB/vCPU |
| `cpu3m` | CPU3 | Memory-Optimized | 8 GB | 256 GB | 10 GB/vCPU |
| `cpu5c` | CPU5 | Compute-Optimized | 2 GB | 64 GB | 15 GB/vCPU |
| `cpu5g` | CPU5 | General Purpose | 4 GB | 128 GB | 15 GB/vCPU |
| `cpu5m` | CPU5 | Memory-Optimized | 8 GB | 256 GB | 15 GB/vCPU |

The default container-disk request is 30 GB. If a small vCPU count makes that
value exceed the table's limit, the creator caps it at the legal flavor limit.
An explicit `RUNPOD_CPU_CONTAINER_DISK_GB` value above that limit fails before
Pod creation.

The command prints the Pod ID, external hard-limit guard, and the workflow to
run after SSH login. Obtain the SSH command from the RunPod Console Connect
page. Inside the Pod, run:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh cpu-prepare
```

Attach for live observation if needed. Detach with `Ctrl-b d`; do not stop the
session:

```bash
tmux -L stock-forecasting-cpu-prepare attach -t stock-forecasting-cpu-prepare
```

<details>
<summary>CPU-preparation artifacts, resource planning and data identity</summary>

`cpu-prepare` performs the following sequence:

1. Creates the persistent directory layout, Poetry 2.4.0, and the remote Python
   3.12 `.venv`, then generates the canonical `poetry.lock` inside the RunPod
   image.
2. Verifies the bundled Kronos source SHA-256 values, verifies the pinned
   model/tokenizer revisions from the Hugging Face cache, and runs the complete
   pytest suite. The remote workflow never clones a Git repository.
3. Revalidates the stage, profile, dates, universe, config SHA-256, and Pod
   environment from the mounted immutable selection before downloading. Cache
   hits do not call the provider again. The script reads the requested vCPU
   count, RunPod's `RUNPOD_CPU_COUNT`, and the cores visible to the container,
   then uses the minimum as its worker count. EODHD, TWSE, and TPEx run as three
   independent top-level acquisition loops; each loop uses a thread pool across
   instruments or actual sessions listed by official benchmark history.
   Raw scan, bucket compaction, candidate ranges, and split ranges use `spawn`
   process pools, and pytest workers never exceed the effective CPU count. The
   process count is not copied directly from the vCPU count: the planner takes
   the lower cgroup/OS available-memory estimate, reserves parent-process
   headroom, assigns at most 60% of currently available memory to workers, and
   lowers each phase's process count using a conservative estimate for its
   largest task. Raw scan coalesces fragmented source row groups into coarse
   partitions of roughly `4 * batch_rows`; each process streams at most
   `batch_rows`, uses Pod-local `/tmp` for bucket intermediates, and atomically
   publishes one indexed partition Parquet to the Network Volume. Each child limits
   Arrow/BLAS native threads to one so process and native-thread counts cannot
   multiply. If one bucket alone exceeds the safe estimate, no process pool is
   started and completed checkpoints remain available for a later Pod with more
   memory. Each provider has its own QPS limiter.
   `--max-api-calls` limits only EODHD; TWSE/TPEx have no project-side request
   counter ceiling. All three providers use the same `--maxBackoff` value for
   their independent retry boundary. One provider exiting never
   cancels the other two. A completed provider first atomically publishes its
   durable Parquet/request-log checkpoint. The process joins all loops before
   merging validated checkpoints or publishing a resume state. The
   complete-plan estimate is informational. The workflow automatically reserves
   25% of max runtime for cleaning/bar-store construction (90 minutes for the
   default six-hour runtime and capped at 2 hours), configurable with
   `--prepareReserve`.
4. Creates and verifies these persistent artifacts:

   | Remote path | Contents |
   | --- | --- |
   | `/runpod-volume/datasets/<dataset-request-sha256>/api-cache/` | Provider raw-response cache |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-progress.json` | Resume attempt, cache counts, and provider/budget/runtime wait state |
   | `/runpod-volume/datasets/<dataset-request-sha256>/provider-checkpoints/` | Validated provider-materialization checkpoints reusable across CPU Pods |
   | `/runpod-volume/datasets/<dataset-request-sha256>/raw/market.parquet` | Canonical daily OHLCV in the durable `downloaded` checkpoint |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/shards/` | Compressed OHLCV bars with one row group per symbol |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/symbol-index.parquet` | Symbol-to-shard/row-group index |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/cutoff-ranges.parquet` | Prepared candidate cutoffs; runtime additionally applies complete continuity/liquidity rules |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/_SUCCESS.json` | Completed bar-store integrity checkpoint |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/execution-plan.json` | In-progress memory budget, effective processes, and reused-task counts by phase; reclaimed after success |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/scan-index.json` | Exact in-progress source-partition to bucket-row-group index; reclaimed after success |
   | `/runpod-volume/datasets/<dataset-request-sha256>/prepared/bar-store/.work/scan-partitions/` | Coarse resumable raw-scan partitions; reclaimed after success |
   | `/runpod-volume/datasets/<dataset-request-sha256>/download-manifest.json` | Actual providers, profile, symbols, and download provenance |
   | `/runpod-volume/datasets/<dataset-request-sha256>/dataset-manifest.json` | Split counts, hashes, and data contract |
   | `/runpod-volume/datasets/<dataset-request-sha256>/manifests/api-request-log.jsonl` | Request audit without tokens |
   | `/runpod-volume/cache/huggingface/` | Offline Kronos model/tokenizer cache |
   | `/runpod-volume/cache/hf-models.json` | Pinned model revisions and cache manifest |

5. After validating raw Parquet, the download manifest, and the request log,
   publishes a `downloaded` execution state to
   `/runpod-volume/lifecycle/stage1/cpu-preparation.json`. A `downloaded` marker
   without an exit code is an
   intermediate checkpoint in the same Pod, so the external guard does not treat it
   as terminal. Only when the remaining time is below the cleaning reserve does exit
   code 75 make it a resumable terminal state; the next CPU Pod then cleans directly
   from the checkpoint without provider calls. If bar-store construction reaches
   its safe deadline, it exits with `waiting_for_preparation` and code 75; the next
   Pod uses `scan-index.json` to skip completed source partitions and compacted
   buckets. If interruption occurs inside one source partition, only that partition
   is redone; atomically published partitions are never rewritten, and the workflow
   does not list tens of thousands of segment directories. Incomplete checkpoints
   from the prior segment layout are directory-renamed aside when the raw and semantic
   preparation contract is unchanged; raw Parquet and provider caches are untouched.
   Worker count and runtime memory planning affect scheduling only and are not part
   of dataset identity, so resuming on a CPU Pod with different vCPU/RAM does not
   repeat provider downloads or invalidate finished bucket checkpoints.
   A launch-specific hidden dataset-manifest staging file is written directly in
   the same dataset root, so every relative artifact path resolves to the same file
   both before and after publication. After verification, a same-filesystem rename
   atomically publishes it as `dataset-manifest.json`. The workflow then binds the
   dataset request, storage-preparation contract, selected-provider and bar-store
   data-content identities, resolved artifact hashes, and creation-time
   selection/stage/config provenance to
   `/runpod-volume/lifecycle/stage1/dataset.json`. That file represents only
   immutable `ready` data; it never carries preparing, failure, or resumable
   execution states. Every CPU execution state goes to `cpu-preparation.json`, so
   a lint, setup, or acquisition failure cannot overwrite completed dataset
   readiness. This is a compatibility summary of the latest CPU preparation,
   not a training dataset selector. Both main-model and baseline workflows read
   manifests under the active selection's `datasets/<dataset_request_sha256>/`.
   If a new active selection changes the dataset request, the old
   selection marker moves to `lifecycle/stage1/history/`; this moves only the
   canonical pointer and does not delete the old dataset. The new dataset marker
   is published only after every check passes, and the CPU guard then terminates
   the Pod from the independent terminal state in `cpu-preparation.json`.

Data identity has three layers so unrelated code changes do not rebuild the
dataset. The dataset-request SHA contains only profile, dates, universe, the
explicit dataset revision, and storage-preparation fields that alter persisted
bars or cutoff ranges. A provider-materialization digest contains only that
provider's endpoint parameters, universe filtering, date boundaries, parsing,
adjustment, and numerical validation. The bar-store digest contains only
cleaning, benchmark eligibility, cutoff, and chronological split semantics. QPS,
API-call ceilings, retry/backoff, the Cloud Run relay, worker/process counts,
memory estimation, checkpoint-directory layout, logging, error prose, lifecycle
code, CLI wrappers, training, and validation are outside data-content identity.
The complete code-release hash remains provenance and upload-integrity evidence,
but cannot invalidate verified data by itself. A real provider-semantic change
quarantines and rebuilds only the affected provider materialization and its
downstream artifacts while raw API cache entries with identical request keys
remain reusable. A bar-store-only semantic change quarantines and rebuilds only
the derived bar store without calling a provider again.

Before any CPU or GPU workflow reads or writes persistent data, it proves that
`/runpod-volume` is the **exact mount point**: use `mountpoint` first, then
`findmnt`, and finally `/proc/self/mountinfo`. The volume ID selected locally is
passed as `RUNPOD_EXPECTED_VOLUME_ID` and compared byte-for-byte with RunPod's
automatically supplied `RUNPOD_VOLUME_ID`; a correct-looking path with the wrong
ID fails immediately. The project lives at
`/runpod-volume/stock_forecasting`, while data, the Kronos cache, W&B
transactions, and models use separate children of the volume root. Persistent
paths may never use the restart-cleared `/workspace`. This layout keeps the
project itself from becoming the mount target and prevents a network-volume
mount from hiding a same-named image directory.

</details>

After the Pod terminates, use the S3 lifecycle as the authority instead of the
now-unavailable SSH session:

```bash
bash scripts/runpod_workflow.sh status
```

##### Provider quotas and cross-Pod resume

EODHD, TWSE, and TPEx use independent parallel loops. Retryable HTTP 429,
temporary network failures, and provider 5xx responses use exponential backoff.
A temporary HTTP 403 from a Taiwan official endpoint due to Pod IP/WAF policy is
also retryable. All three providers share one `--maxBackoff` setting (default
`1m`) but maintain independent backoff state and exit when their next delay
would exceed it. EODHD is additionally bounded by the CPU attempt's
`--max-api-calls`; whichever EODHD boundary is reached first stops its loop.
TWSE/TPEx have no request-count ceiling. The shared acquisition deadline can
still place any loop in `waiting_for_resume` to protect cleaning time. The
workflow follows this contract:

1. Every successful raw JSON response remains in the dataset request's
   `api-cache/`. A complete provider may atomically publish under
   `provider-checkpoints/`, but incomplete provider staging Parquet and the
   aggregate `state=ready` marker are never published.
2. If EODHD reaches `--max-api-calls` or its backoff boundary first, only its
   loop exits; TWSE/TPEx continue. A Taiwan provider crossing its backoff boundary
   likewise does not stop EODHD or the other Taiwan provider. Crossing the
   boundary opens a shared circuit breaker for that provider client. Requests
   already in flight may finish, but sibling workers cannot start another request
   for the stopped provider. The main process always waits for every selected
   provider loop to exit before the CPU Pod may finish.
3. `download-progress.json` records the attempt, per-provider cache/network
   counts, the limited EODHD count, and each provider's `complete`,
   `waiting_for_budget`, `waiting_for_provider`, or `waiting_for_resume` outcome.
   Provider errors also record cumulative wait, last wait, next proposed backoff,
   the shared maximum, and the launch's effective `max-api-calls`, EODHD QPS,
   and Taiwan QPS. HTTP status, `Retry-After`, and rate-limit headers are stored
   when supplied. Data-contract errors additionally identify the safe provider,
   operation, symbol/date/month, and exception type. Tokens and response bodies
   are not stored. A `complete` outcome also records its materialization
   checkpoint identity and whether that checkpoint was published or reused.
4. If any loop is incomplete, the CPU-preparation lifecycle becomes
   `waiting_for_budget`, `waiting_for_provider`, or `waiting_for_resume`; GPU
   readiness stays blocked, while durable checkpoints from completed providers
   remain available. If all loops complete, validated provider checkpoints are
   merged in a fixed order and cleaning continues instead of closing the Pod early.
5. After quota becomes available, **do not reconfigure, change
   `--dataset-revision`, or delete the cache**. When creating the next CPU Pod,
   enter a new `--max-api-calls` value for the current remaining quota. QPS may
   also change for that attempt without changing the selection. Then connect to
   it and start the same workflow:

   ```bash
   bash scripts/runpod_workflow.sh cpu prepare
   # Run after connecting to the newly created CPU Pod:
   cd /runpod-volume/stock_forecasting
   bash scripts/runpod_tmux_launch.sh cpu-prepare
   ```
6. The new attempt first validates and reuses provider checkpoints with the same
   provider-materialization request and data-content digest. It replays cached
   responses only for incomplete or content-incompatible providers and calls
   them only for missing requests. The lifecycle becomes `ready` only
   after all data, manifests, and selection gates pass. The `all`-mode discovery
   response is part of the same immutable cache, so a cross-day resume does not
   replace it with a drifted instrument list.

EODHD occasionally returns all-zero or otherwise invalid placeholder rows inside
an instrument's daily history. The downloader neither imputes nor rewrites those
prices: it drops only source rows that cannot satisfy the canonical OHLCV
contract, counts them in `dropped_source_rows`, and retains the instrument's
valid observations. Early TWSE actions can exist in the `TWT49U` main table while
the detail endpoint returns no record. In that case the verified price factor is
retained, the unknown share multiplier stays at identity `1.0`, and
`missing_share_multiplier_details_by_provider` explicitly records the resulting
volume-adjustment coverage gap instead of inventing a ratio.

`bash scripts/runpod_workflow.sh status` prints the download attempt, cached
response count, network requests in that attempt, the full request estimate when
available, and a safe provider-error summary below the `cpu_prepare` lifecycle, so no
JSON file needs to be opened manually.

This provides both request-level and provider-materialization-level resume, not
byte-range resume within one HTTP response. Each successfully completed API
request is a raw-cache resume unit; each complete provider checkpoint that
passes hash and identity validation is a materialization resume unit. A progress identity
that differs from the current profile, dates, universe, or dataset request
fails closed. An undersized `--max-api-calls` produces resumable
`waiting_for_budget`, not failure; create another CPU Pod with the same selection.
A 401/403 or other non-temporary configuration error produces `failed` and
requires correcting the Secret. Change
`--dataset-revision` only for an intentional new provider snapshot; a new
revision does not reuse the old snapshot cache.

If the state is `waiting_for_budget`, `waiting_for_provider`,
`waiting_for_resume`, `waiting_for_preparation`, `failed`, or `timed_out`, download CPU logs
with the following command. It resolves `launch_id`, `log_path` and `progress_path` from
lifecycle records without manual JSON inspection or remote paths:

```bash
bash scripts/runpod_workflow.sh cpu-logs
```

After CPU work finishes, check status. Baselines use `readiness --baseline`; the main
model uses `readiness --gpu`. Creation commands handle preflight automatically;
baselines reuse still-valid recent checks. Never edit readiness markers:

```bash
bash scripts/runpod_workflow.sh status
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64
```

<a id="baseline-en"></a>

#### 4. Build or resume both baseline families for one group

Prepare the training and shared evaluation sources, then synchronize source/configuration.
One paired baseline command handles both alpha-point and absolute-stock direction comparators:

```bash
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1
```

```bash
bash scripts/runpod_workflow.sh baseline \
  --experiment a-signal64 \
  --maxRuntime 24h \
  --gpuId "NVIDIA GeForce RTX 4090"
```

Use `b-signal64` for B. The catalog data center must match the registered volume's location.
Runtime/GPU defaults remain `12h`/RTX 5090, with the documented aliases. Baseline accepts
one `--experiment NAME`, not `--seed` or `--launchWorkers`. Both task caches are checked locally;
if both are complete no Pod is rented. Otherwise one Pod sequentially builds/resumes both
families, skipping a family already complete. The 24h limit covers the whole Pod.
Independent numerical cache identities remain reusable across main-model seeds.

Read-only completion plus current-data-rule verification for both tasks:

```bash
bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64
```

The query prints each task's status and returns 0 only when both are complete and pass data
rules. Pair operations do not require manual task switching or change the global active
selection. Omitting `--experiment` retains legacy active-selection single-task behavior,
not whole-pair completion. Advanced individual-model operations are in the
[appendix](#single-model-en).

Independent point, fixed-return classification and legacy quantile/cost-direction baselines are
not interchangeable. The task and its beta/neutral threshold enter the baseline contract, not
main-model LoRA capacity or training seed. Models with identical data/task/baseline parameters
reuse completed results. Choosing a new task does not rerun CPU preparation or download prices;
existing results remain intact rather than being overwritten with different task semantics.
When updating only the main-model config for prepared data, refresh the
selection with `configure --reuse-current`; do not rerun CPU preparation/finalization.

`readiness --baseline --experiment a-signal64` is a paired read-only query that **checks both current data-cleaning rules and
baseline completion**. It verifies the selected training bar store, the shared validation/test
source selected by `configs/data_cleaning.json`, and complete results matching the current data
and baseline training contract. A cannot substitute its own evaluation snapshot for the shared
B source. No Kronos/HF cache, main-model training YAML, or shared "most recent CPU prepare"
record is required. This query never creates a Pod or retrains models.

The default output contains completion, data-rule status, eligible counts, baseline ID and next action:

- `Baseline: COMPLETE` with `Data rules: PASS`: every required model, metric, weight/training
  artifact, shared evaluation array and prediction file passed the existing integrity checks;
  all three split counts match the current eligible indexes. Exit code **0** means this baseline
  is reusable without retraining.
- `Baseline: NOT COMPLETE`: no complete result matches the selected data and rules. Even if
  data rules pass, run `baseline` to build or resume. Exit code is **1**.
- `STORAGE FINALIZATION REQUIRED`: training results exist but shared-result storage needs
  finalization. Exit code is **1**; run `baseline` to finalize without retraining.
- Invalid data/rules/results, missing artifacts, or failed remote reads produce exit code **2**
  with a specific error, never a false completion or a network failure mislabeled as an unbuilt cache.

`readiness --baseline` and `baseline` share a local ten-minute successful-check cache.
If the code release, volume, dataset selection, cleaning policy and remote artifacts are unchanged,
creation/retries check only small manifests and paginated object revision/size listings, without
issuing another individual query for every shard or printing the full report again. Expiration,
code/policy changes, replaced/deleted artifacts or newly built cleaning indexes trigger full
verification. Failed remote queries never fall back to stale success. Running `baseline` directly
performs the initial check; a separate `readiness` command is optional. This cache only accelerates
data-input admission; it is not a baseline model identity or completion record. Every status query
still verifies baseline results. Deployment-version checks, live Pod conflict checks and mounted
data verification remain mandatory for Pod creation. A read-only query does not require local
query scripts to match the deployed release file for file; actual launches still require changed
source to be synchronized and deployment verification to pass. Source synchronization does not
by itself require baseline retraining.

Candidate and eligible counts have distinct meanings; the CLI's `Eligible windows` shows only the latter:

- `prepared_candidate_counts` are historical CPU-preparation candidate counts, **not eligible
  windows after runtime cleaning**.
- `eligible_sample_counts` come from a `sample-universes` index matching the current
  continuity, valid-bar and liquidity rules, source snapshot, 128-bar window and fixed splits.
  Validation/test counts come from the shared evaluation source.
- `cleaning_state=ready` means the current index and accepted-window audit counts were
  checked. `pending_build` means the baseline workflow must first build the missing indexes;
  unverified counts are `null`, never filled from preparation counts. Available build inputs
  do not mean cleaning or baseline training is complete; the read-only completion query does not succeed.
  No repeated CPU preparation, market-data download or manual manifest edit is needed.
- Missing shared sources, corrupt indexes, inconsistent current policy/source metadata,
  or empty populations for used splits fail admission. Old-rule indexes remain untouched
  and cannot satisfy the current-rule check.

Local checks cover small metadata checksums, prepared shard/index object sizes and the
existence/nonempty size of the cleaned range index. Mounted admission streams data and
cleaned range-index checksums. The control host does not rescan every market-data row.
`readiness --gpu` remains the main-model gate, not a baseline prerequisite. `readiness --baseline`,
`baseline` and `train` share the same completed-result validator; neither status nor launch accepts
old results that fail the current cleaning rules.

Main-model `readiness --gpu`, mounted admission, and run provenance also use the
selected dataset's own manifest. Kronos is checked independently against the
current config's repositories, pinned revisions, and offline verification result,
not the shared HF manifest file hash from an earlier CPU preparation. Switching
between already-prepared A/B datasets requires no CPU preparation, downloads, or
resplitting and does not invalidate completed baselines. Resume/evaluation checks
use the dataset manifest and original checksums recorded by that run plus the
run's pinned selection, never the most recently prepared dataset's shared summary.

`baseline` reads the volume from `.env` and uses the active selection. It checks the
complete manifest and artifact sizes **locally before allocating a paid Pod**.
A complete matching cache exits immediately without a Pod. Otherwise connect with
the Console's SSH command, then run:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh baseline
```

The detached tmux job survives SSH disconnection. To observe progress:

```bash
tmux -L stock-forecasting-baseline attach -t stock-forecasting-baseline
```

Detach with `Ctrl-b d`. The **local guard** terminates the Pod after completion,
failure or timeout; keep the local host powered and connected. Baselines do not
depend on Pod self-termination. Confirm the original Pod has ended through Console or read-only
`recover --pod-id <POD_ID>`; persisted `status` results do not establish that billing stopped.
To resume unfinished work or check the cache, run:

```bash
bash scripts/runpod_workflow.sh baseline --experiment a-signal64 --maxRuntime 24h --gpuId "NVIDIA GeForce RTX 4090"
```

Use the same paired `--experiment` as the unfinished build. Completed task families are
reused independently; both complete means a cache hit. An incomplete build instead creates a
new Pod; launch the same remote baseline tmux command to resume saved jobs and
checkpoints. Full baselines may take longer than 24 hours; this is a per-Pod workload
limit, not a completion-time guarantee. Use `readiness --baseline --experiment a-signal64` to confirm
`COMPLETE` and `PASS` before launching the main model. `baseline` is not a read-only
status query: an incomplete cache creates a paid Pod. Missing complete baselines cause
`train` to reject creation locally rather than silently starting baseline training.


Each dataset, prediction task and baseline numerical-contract combination has its own
results; main models reuse them only when all three match. Baseline base parameters,
early stopping and model lists come from `configs/baseline.json`, with the selected task's
objectives and beta/neutral threshold applied. Main-model Kronos revisions, LoRA capacity,
training seed, feature mode, learning rate and checkpoints do not control baseline
admission or cache identity. Changing the prediction task, point-loss beta or neutral
threshold requires a matching baseline, as do changed data/splits or baseline numerical
contracts; different stock-date populations or tasks cannot be mixed.

To select an A dataset (starting in 2021) whose **CPU preparation is already complete**,
use the following after the current source/YAML release has been synchronized. To switch
to B, change `--start` to `2016-01-01` and `--experiment` to `b-signal64`; retain the
other options used for its original preparation. The same command handles both tasks:

```bash
bash scripts/runpod_workflow.sh configure \
  --stage stage2 --data-profile us_tw_eodhd --dataset-revision v1 \
  --start 2021-01-01 --end 2026-06-01 \
  --h-start 1 --feature-mode combined --universe all
bash scripts/runpodctl_project.sh gpu list
bash scripts/runpod_workflow.sh baseline --experiment a-signal64 --maxRuntime 24h --gpuId "NVIDIA GeForce RTX 4090"
```

This does not prepare/split data again or download market data. No manual manifest,
SHA or path editing is required. If data admission fails, investigate the reported
data issue instead of automatically rerunning CPU preparation.

<details>
<summary>Baseline resource tuning, persistence and interruption recovery</summary>

`configs/baseline.json` controls numerical parameters and resource limits. Defaults
allow two deep-model experiments on one GPU plus one CPU rule/GBDT job. Admission
uses CPU count, cgroup memory, free GPU memory and `/dev/shm`; DataLoader workers,
prefetch, native BLAS threads and job deadlines are bounded. GPU and CPU jobs overlap
when budgets permit. Insufficient resources explicitly reduce concurrency or reject
execution. Full GBDT RAM and full-validation cost can be much higher than the former
20,000-row workflow; inspect resource plans/logs instead of bypassing checks through
subsampling. Load `.pt`/`.pkl` only from trusted project-generated hash-scoped output.

Each deep baseline first uses GPU probes for memory admission, then jointly measures
batch size, workers and prefetch on real train/validation windows. It selects a
lower-resource configuration within 95% of the best end-to-end throughput. Probes use
isolated model copies without changing production weights, optimizer or RNG state.
Neural jobs receive fixed-shape contiguous tensor batches assembled before pinning,
without per-window Kronos metadata/padding or materializing the complete window dataset.
Training, validation and testing alternate one worker pool; phase changes release the
previous pool and resume from the committed sample cursor.
CPU workers and GBDT threads respect both affinity
and cgroup v1/v2 bandwidth quotas rather than assuming all host cores are available.
Per-job `runtime-plan.json` records the measurements and selected settings. Configure
batch/prefetch ceilings, probe repetitions and checkpoint intervals under `resources`
in `configs/baseline.json`; `auto_batch=false` selects the fixed `batch_size` instead.
Dynamic numerical readers decode only required OHLCV, adjustment and timestamp columns.
Each baseline worker has a bounded Parquet metadata cache: at most 128 files and a
512 MiB conservative metadata estimate by default, controlled by
`resources.parquet_cache_files`/`parquet_cache_bytes`. It never caches expanded windows
or serializes file handles into spawned workers. The default worker reserve is 1 GiB.

Admission reserves host RAM jointly for full GBDT and GPU jobs, with 20% headroom by
default. GBDT starts before rule jobs once CPU inputs are complete. If even one GPU
job cannot safely overlap, admission requests more RAM instead of silently deferring
GBDT to a CPU-only tail. Once GPU jobs finish, the next native GBDT fit receives the
released CPU threads. `live-resources.json` records allocation and `progress.json`
records input-wait fractions. GBDT may still outlast the neural experiments; this
scheduler does not guarantee elimination of the CPU-only tail.
Baseline admission may credit 50% of clean, unmapped cgroup file cache before applying
the safety margin. Anonymous memory, shared memory, mapped, dirty and writeback pages
receive no credit; all valid container and host memory limits still apply. Configure
the fraction with `resources.reclaimable_file_cache_fraction`. Missing statistics fall
back to raw headroom, and `resource-plan.json` records both raw availability and cache
credit. This conservative estimate is not an OS reclamation guarantee and does not
change CPU-prepare resource planning.

Reviewed execution-only changes with unchanged numerical contracts can preserve the
baseline ID, inputs, completed jobs and resume checkpoints through the **exact source
hash allowlist** in `configs/baseline_execution_compatibility.json`.
`execution-contract.json` records the actual implementation. Unknown baseline source
changes still invalidate identity, as do changed data periods, model parameters,
labels or calibration contracts. Python versions share a stable AST serialization.

Resume with the same `baseline` and tmux commands above; no checkpoint path is needed.
Neural jobs save `resume.pt` after the first batch, every 300 seconds by default, and
before/after full validation. It retains weights, optimizer, LR scheduler, RNG state
and the within-epoch sample cursor. A new hardware/batch plan continues the same
random epoch order; warmup and validation boundaries are sample-aligned. Interrupted
validation restarts the full split and never supplies partial early-stopping metrics.
Training remains dynamically sampled; validation/test dynamically enumerate every
valid per-instrument window in fixed order without a giant resident window dataset.

The CPU tabular input cache flushes/fsyncs before committing its row cursor every 300
seconds and retains completed splits. GBDT saves after each horizon's incremental fit for
independent tasks, or each horizon/quantile for quantile regressors; rule jobs reuse
completed rules. An interrupted native GBDT fit, full validation
or final test restarts that work unit. These are baseline artifacts: no new market-data
API calls or CPU prepare reruns are required. Progress is reported every 30 seconds
by default; only `complete.json` denotes a complete baseline result.

</details>

<a id="train-en"></a>

#### 5. Choose tasks/experiments and create training Pods

<a id="gpu-catalog-en"></a>

##### Query GPU resources

Query the GPU model and stock catalog to obtain a complete `gpuId`. Training defaults
to `NVIDIA GeForce RTX 5090`. With no options, the query covers all data centers;
it does not implicitly filter using the data center in `.env`:

```bash
bash scripts/runpodctl_project.sh gpu list
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1
bash scripts/runpodctl_project.sh gpu list --data-center EU-SE-1 --search "5090"
bash scripts/runpodctl_project.sh gpu list --data-center EU-RO-1 --output json
```

| Option | Default or format | Behavior |
| --- | --- | --- |
| `--data-center ID` | No filter; e.g. `EU-RO-1`, `EU-SE-1` | Case-insensitive exact data-center ID, not a region prefix such as `EU`. Does not change `.env` or the Pod deployment location |
| `--search TEXT` | No filter | Case-insensitive substring of the GPU display name or complete `gpuId`; can be combined with the data-center filter |
| `--output table`, `--output json` | `table` | Tables show VRAM, Secure/Community USD hourly prices, complete `gpuId` and data-center stock. JSON retains matching GPUs' API fields, including other data centers |
| `-h`, `--help` | Flag with no value | Show GPU query options |

`--` means the API did not report a value; `NONE` means that data center reports no
stock, and these GPUs **remain in the results**. A catalog query is neither a capacity
reservation nor a guarantee that Pod creation will succeed. For a filtered table, read
the stock column for the selected data center. A Pod using an existing network volume
must be created in that volume's data center: querying another location neither makes
the volume attachable across regions nor creates a replacement volume. Pass the complete
`gpuId` to creation commands, not a potentially truncated GPU display name.

##### Choose the data group and paired experiment

Complete CPU preparation and both task-specific baselines first. Each current group/seed
combination creates one single-GPU Pod that sequentially runs two independent models.
This is not DDP, a combined-loss model, or two separately operated training jobs.

| Paired experiment | Training-data start | Two configs under `configs/experiments/` |
| --- | --- | --- |
| `a-signal64` | 2021-01-01 | `a_alpha_point64.yaml`, `a_absolute_direction64.yaml` |
| `b-signal64` | 2016-01-01 | `b_alpha_point64.yaml`, `b_absolute_direction64.yaml` |

Both data ranges end at 2026-06-01 exclusive and use the same validation/test source and
eligibility rules. `--experiment` preserves configure's profile, universe, revision and h_start.
One parent run owns the `alpha_point` and `absolute_direction` child runs. Both initialize
from the same pretrained Kronos-base with LoRA rank 64/alpha 128, but have separate trainable
weights and optimizers. The alpha model finishes training/test before its process resources
are released and the direction model starts. One child's early stop/completion cannot finish
the other child or terminate the entire Pod.

Task/LoRA learning rates are `3e-5`/`5e-6`, with 256,000 sample presentations of warmup.
Training retains dynamic sampling weighted by `0.8 ** (latest_training_year - cutoff_year)`;
annual quotas are proportional to eligible-window count times the weight. Validation/test
fully traverse the fixed eligible sets. Each model selects checkpoints and early-stops by
its own full-validation normalized Smooth L1 or CE, without interval calibration.
`FORECAST_METRIC_WORKERS` caps diagnostics at 4 by default, additionally CPU/memory bounded.

Specify the paired experiment explicitly. Omitting `--experiment` retains the active
selection and does not switch architectures. Archived capacity/V1/V2/V3 presets and advanced
individual-model entry points are in the [final appendix](#historical-experiments-en), not
the normal paired-training workflow. Pairing does not change numerical contracts or require
CPU preparation, market-data downloads, or retraining already matching baselines.

##### Create one or multiple training Pods locally

Upload source/configs once on initial deployment or after edits; never overwrite shared
source/runtime while a Pod owns the volume. Changing uploaded paired experiments or seeds
does not require another upload:

```bash
bash scripts/runpod_workflow.sh sync --dry-run
bash scripts/runpod_workflow.sh sync --apply
```

One A-group seed, one Pod, two models:

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-signal64 \
  --seed 42 \
  --maxRuntime 72h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

Three seeds of the same group create three Pods, with both models in each Pod:

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-signal64 \
  --seed 42 --seed 43 --seed 44 \
  --launchWorkers 2 \
  --maxRuntime 72h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

This means three parent runs and six child models, **not six Pods**. Use `b-signal64` for B.
Selecting both groups with three seeds creates six Pods. Each `--seed` takes one integer.
For different GPUs, issue three commands with one seed and the desired GPU each. Immutable
per-launch selections and pair manifests isolate run IDs, logs and guards; later configure
operations cannot change existing launches.

| Option | Default | Behavior |
| --- | --- | --- |
| `--experiment NAME` | Active selection | Current choices `a-signal64`/`b-signal64`; repeat distinct names to combine with seeds; up to 22 names including advanced/archived presets, at most 30 combinations |
| `--seed N` | Selected YAML/run training seed | Repeat distinct integers, 0–4294967295; one Pod per group/seed; both children use that seed with independent RNG/weights |
| `--launchWorkers N` / `--launch-workers N` | `2` | Local creation-request workers, 1–6; not model concurrency, GPU count or DataLoader workers |
| `--maxRuntime DURATION` / `--max-runtime DURATION` | `12h` | One shared stop-request deadline per Pod covering both models and evaluations, not a fresh runtime for each child; positive integer plus `m`/`h`/`d` |
| `--gpuId GPU_ID` / `--gpu-id GPU_ID` | `NVIDIA GeForce RTX 5090` | One GPU model for this batch; quote the full ID and launch separately for different models |

Seed overrides do not mutate YAML, active selection, evaluation membership or scale-calibration
seed. Matching baselines are reusable across seeds. Resume preserves the original seed;
different GPU implementations may still yield small numerical differences. Both tasks' data,
code and completed baselines must pass preflight before paid creation starts.

Partial creation failures report each combination and exit nonzero. Successfully created Pods
keep their guards; they are not deleted or automatically rented again. Retry only failed
combinations. `--launchWorkers 1` serializes creation requests, not paid Pods. Costs add across
Pods. Launch output includes the parent run, both child IDs, seed, Pod ID and guard log;
normally retain the parent run ID for subsequent operations.

A rejected GPU creation request leaves only a local attempt receipt, not a remote run.
Remote run metadata is published only after Pod creation succeeds and its local guard starts,
so capacity retries do not add unstarted experiments to the run inventory.

##### Start training inside each Pod

Pod creation does not start training. SSH into each Pod and launch once:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-train
```

`stage1-train` is the existing workflow name. The paired manifest selects both Stage 2 models;
do not configure again or start separate child jobs. Attach to the Pod's log:

```bash
tmux -L stock-forecasting-train attach -t stock-forecasting-train
```

Detached tmux survives SSH disconnection. Models run sequentially inside their Pod; different
parents may run concurrently on different Pods. Each model independently tunes batch size,
workers and prefetch, and stores checkpoints, evaluations, loss and RNG. Read-only bars,
pretrained cache and matching completed baselines are shared. The same parent cannot run on
two Pods simultaneously; cache coordination does not duplicate the full window dataset.

The pair is COMPLETE only after both models finish training and test. Failure/timeout uses
the parent guard's safe shutdown; resume skips completed models. A shared runtime may finish
only the first model and must not be reported as full success. CPU preparation, baseline
building, source sync and shared-runtime updates remain exclusive with main-model readers.

<a id="run-status-en"></a>

##### Find run IDs, execution settings and completion status

Run these commands in the **local project directory**. They use the network volume configured
in `.env`; no running Pod, volume ID argument or file path is required. List recent runs:

```bash
bash scripts/runpod_workflow.sh runs
```

Each pair appears once under its parent ID, with both child IDs/states, rather than as three
separate experiments. `status <PARENT_RUN_ID>` expands both models; child status remains
available directly. Each entry groups the run ID, experiment name, training/final-evaluation/overall state, run
initialization time, last activity and recorded configure settings: stage, data profile, dataset
revision, start/end-exclusive dates, h-start, feature mode, universe, symbol limit, stock/ETF
lists and YAML path. Settings come from **that run's saved selection/manifest**, not today's
active selection. Changing configure later does not change historical entries.
`symbol-limit=none` means no configured cap. `stocks=-`/`etfs=-` means no explicit list,
not an empty market; universe and data profile still determine the scope.

Find a current task experiment, or list only fully completed runs:

```bash
bash scripts/runpod_workflow.sh runs --experiment a-signal64
```

Filter a specific experiment by seed. `runs` and `status RUN_ID` display the recorded training
seed, or `unknown` for historical records without one, never the current YAML's value:

```bash
bash scripts/runpod_workflow.sh runs --experiment a-signal64 --seed 43
```

```bash
bash scripts/runpod_workflow.sh runs --state complete
```

Copy a run ID from the list to inspect completion, detailed timestamps, Pod/launch IDs and
recorded LoRA settings:

```bash
bash scripts/runpod_workflow.sh status "<RUN_ID>"
```

For a pair, `COMPLETE` requires both children's training completion and final `split=test`
reports with `state=ready`, matching their task identities. A parent marker alone is insufficient.
Individual historical runs retain their recorded evaluation split and cannot be relabeled as holdout.
`TRAINED` means training is complete but final evaluation is still pending;
`TRAINING`/`EVALUATING` describe the recorded active phase. `INCOMPLETE` means interruption,
failure or missing results needed to support a completion marker. `NOT_STARTED` means only
a pre-launch selection exists, without evidence that training started. `UNKNOWN`/`ERROR` are
not success. Only `split=test` identifies holdout evaluation; older runs evaluated on validation
remain labeled `split=validation`. W&B delivery is shown separately from model completion.

This is a read-only query of persisted results/lifecycle, not a live GPU or Pod-liveness check.
For a parent, `Initialized` is when its pair manifest was allocated; for a child it is when training
created its model manifest. Neither is the configure timestamp or the GPU compute start of each
resume attempt. A not-yet-started pair still displays its embedded configure settings.
`Last activity` is the latest recorded state/result
timestamp. Details separate training completion from evaluation start/completion. Unrecorded
fields display `unknown`/`unrecorded` instead of borrowing current settings. If a cutoff/completion
race left a `timed_out` lifecycle, the query retains that raw state but uses the run's durable
completion results instead of interpreting one label alone.

| Parameter | Command/default | Meaning |
| --- | --- | --- |
| `RUN_ID` / `--run-id RUN_ID` | `status`; use one form | Inspect one run, not a Pod ID or baseline ID |
| `--experiment NAME` | `runs`; no filter | Exact recorded experiment, e.g. `a-signal64`, `b-signal64`; older unnamed runs remain visible without this filter |
| `--seed N` | `runs`; no filter | Recorded training seed, 0–4294967295; can be combined with `--experiment` and `--state` |
| `--state STATE` | `runs`; no filter | `complete`, `trained`, `training`, `evaluating`, `incomplete`, `not_started`, `unknown`, `error` |
| `--limit N` | `runs`; `20` | 1–200 entries per page, newest run allocation first, not ordered by a later resume time |
| `--offset N` | `runs`; `0` | Skip N matching runs; use the printed next-page offset with the same filters when more candidates remain |
| `--workers N` | `runs`; `2` | 1–8 concurrent-query ceiling, further bounded by visible CPUs, available memory and the API cap; reads metadata, not weights or datasets |
| `--output table\|json` | `runs`, `status RUN_ID`; `table` | Human-readable grouped summaries by default; `json` preserves structured settings, timestamps and states for automation |
| `--timezone ZONE` | `runs`, `status RUN_ID`; `Asia/Taipei` | IANA timezone for human-readable times, e.g. `UTC`; JSON retains source timestamps and offsets |

`status RUN_ID` exits `0` for fully complete, `1` for an existing but not fully complete run,
and `2` for a missing run, invalid arguments or read/ownership errors. `runs` exits `0` for a
successful query even when entries are unfinished; read errors are reported and return `2`,
never silently skipped. Queries do not change selections/results, require a source upload,
or trigger CPU preparation, baseline building or training.

The existing no-argument `status` keeps the code, CPU preparation, dataset, baseline, download
progress, run listing and W&B overview:

```bash
bash scripts/runpod_workflow.sh status
```

A phase-level `ready` in the overview does not certify the entire run; use `status RUN_ID`
for that run's completion decision. Check baseline completion with
`bash scripts/runpod_workflow.sh readiness --baseline --experiment a-signal64` (use `b-signal64` for B).

##### Runtime limits and guard recovery

At `--maxRuntime`, each local guard publishes a stop request bound to its Pod and run.
The Pod completes its current validation/checkpoint transaction or benchmark model/seed result
before acknowledging a safe boundary and terminating. **This is a stop-request time, not a
guaranteed cost ceiling**: long sections may overrun and continue billing. Baselines retain
their existing hard limit. Each Pod has a separate local guard; macOS uses `caffeinate`
to prevent sleep, but shutdown, power loss or network loss can still disrupt monitoring.

After recovering the control machine, inspect first and explicitly apply recovery if needed.
`--pod-id` limits recovery to one Pod:

```bash
bash scripts/runpod_workflow.sh recover
bash scripts/runpod_workflow.sh recover --apply --pod-id "<POD_ID>"
```

Without `--pod-id`, all project Pods are inspected. Only `--apply` permits re-arming missing
guards or terminating verified-completed Pods. Recovery does not reset the original deadline.
Unknown state, transport errors or a mismatched Pod/run lifecycle are not treated as completion.
An SSH/tmux screen alone is not authoritative completion evidence.

After manually deleting a Pod in RunPod Console, clean up its orphaned local guard, GPU-clock
monitor and guard-owned `caffeinate`. Preview first, then apply explicitly:

```bash
bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>"
```

```bash
bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>" --apply
```

The command first makes a read-only API check that the Pod no longer exists. An existing Pod,
API failure, reused PID or unverifiable process identity blocks cleanup. Without `--apply` it
sends no signal; with it, only verified local processes receive `SIGTERM`. Logs are retained;
shared SSH sockets, remote resources and other Pods' monitors are untouched. `--pod-id` is required.
Optional `--guard-dir PATH` defaults to `RUNPOD_GUARD_LOG_DIR` or `~/.local/state/runpod-guards`;
`-h`/`--help` shows usage. This is not `recover --apply`, which can re-arm guards or terminate
verified-completed Pods rather than cleaning up orphaned processes.

##### Local GPU throttling monitoring and notifications

New or re-armed training, baseline and validation GPU guards automatically start an independent
local GPU monitor; CPU preparation does not. No extra terminal or agent SSH socket is required.
Every 30 seconds by default, its own
SSH connection runs only `nvidia-smi -q -x`, recording GPU UUID, SM/graphics/memory clocks, load,
temperature, power/limits, clock-event reasons and duration counters. It never changes GPU settings,
stops training, or creates/deletes Pods. Connection failures do not affect the lifecycle guard.
Updating files does not retrofit running guards or reset deadlines. The default SSH identity is
`~/.ssh/id_ed25519_runpod`; unlock encrypted keys with `ssh-add`. The monitor never stores passwords,
forwards the SSH agent, or extends/closes agent sockets.

Interpretation follows [NVIDIA Clock Event Reasons/Counters](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
thermal/hardware-power flags or **newly accumulated** durations are recorded immediately, not historical
counter totals. `SW Power Cap` alone means `power_limited`, not a hardware fault. Clock notifications
use a rolling five-minute window: valid high-load time **below 80% of the same GPU's normal clock
reference** must total **strictly more than half of the window**. Dips need not be consecutive; exactly 80%
or exactly half does not trigger. The reference is the median of healthy high-load SM clocks, not
the specification's maximum Boost clock. A full window must elapse with at least half of it covered
by valid high-load intervals. Fractions are time-weighted, not sample-count-weighted. The denominator is
the full window (300 seconds by default), requiring more than 150 seconds of low clocks. Idle/low-load
intervals do not count as low-clock time and do not shrink the denominator. Missing-data gaps reset
the window. This is sampled estimation, not a continuous
hardware trace. Isolated flags or low configured power limits remain in `gpu-status` without independently
pushing clock alerts. These are engineering defaults, not NVIDIA fault thresholds.
A power limit below 90% of the device's reported default gets a separate settings warning.
P2, clocks below maximum, idle periods and checkpoint pauses do not individually imply abnormal
throttling. Missing fields produce `limited_telemetry`, with an incomplete-monitoring record after two
minutes; without a normal reference the monitor cannot rule out unexplained low clocks from startup.
SSH boot grace reports `waiting_for_ssh`; subsequent
failure reports `unavailable`, never healthy.

```bash
bash scripts/runpod_workflow.sh gpu-status
bash scripts/runpod_workflow.sh gpu-status --pod-id "<POD_ID>"
bash scripts/runpod_workflow.sh gpu-status --run-id "<RUN_ID>" --output json
bash scripts/runpod_workflow.sh gpu-notify-test
```

`gpu-status` only reads local records, with no cloud calls. Optional `--pod-id` and `--run-id` filters
can be combined; neither filters by default. `--output` accepts `table` (default) or `json`.
`--guard-dir PATH` defaults to `RUNPOD_GUARD_LOG_DIR` or `~/.local/state/runpod-guards`;
`-h`/`--help` shows help. `gpu-notify-test` takes no options and submits an explicitly labelled macOS
test notification. No-argument `status` retains the original overview and appends GPU telemetry;
the training/final-evaluation completion verdict of `status RUN_ID` is unchanged.

Notifications identify the Pod, run, GPU, normal clock reference and low-clock time fraction.
Identical events have a default 15-minute cooldown; confirmed recovery is only pushed for an event
previously pushed. SSH monitoring-unavailable/recovered notifications remain separate; monitoring
failure is not GPU throttling. `submitted` means handed to the OS,
not guaranteed visible under Focus mode. Logs sit beside the guard log as `<pod-id>.gpu-events.jsonl`
(5 MiB with two rotated backups) and `<pod-id>.gpu-monitor.json`. `stale`/`stopped` is not a healthy
GPU verdict. The local host must stay powered and online; other platforms retain logs and CLI queries.
Each Pod has an independent, timeout-bounded monitor and separate detection state.

Set optional environment variables before the workflow; they do not modify model YAML or data/baseline identity:

| Environment variable | Default / allowed values |
| --- | --- |
| `RUNPOD_GPU_MONITOR` | `1` enables; `0` disables GPU monitoring only, not the lifecycle guard |
| `RUNPOD_GPU_MONITOR_SSH_KEY` | `~/.ssh/id_ed25519_runpod`; alternative identity file path |
| `RUNPOD_GPU_MONITOR_NOTIFICATIONS` | `auto` enables macOS notifications; `0` disables; `1` requests them, reporting unsupported on other platforms |
| `RUNPOD_GPU_MONITOR_POLL_SECONDS` | `30`; 10–300 seconds |
| `RUNPOD_GPU_MONITOR_CONFIRM_SECONDS` | `120`; 30–3600 seconds for missing-data, disconnect and recovery confirmation |
| `RUNPOD_GPU_MONITOR_CLOCK_WINDOW_SECONDS` | `300`; 60–3600 seconds for the rolling low-clock time fraction |
| `RUNPOD_GPU_MONITOR_COOLDOWN_SECONDS` | `900`; 60–86400 seconds |
| `RUNPOD_GPU_MONITOR_STARTUP_GRACE_SECONDS` | `300`; 0–1800 seconds |
| `RUNPOD_GPU_MONITOR_HIGH_UTILIZATION_PCT` | `80`; 10–100 |
| `RUNPOD_GPU_MONITOR_CLOCK_DROP_FRACTION` | `0.20`; 0.05–0.8; values below 0.20 cannot raise the notification threshold above 80%; larger values require a deeper drop |
| `RUNPOD_GPU_MONITOR_QUERY_TIMEOUT_SECONDS` | `15`; 3–30 seconds; only the monitor's query is killed on timeout |

##### Loss logs and W&B recovery

Each run's `metrics.jsonl`, `summary.json` and W&B record training loss, learning rates,
processed samples and full validation metrics. Independent point tasks also record
`train/normalized_smooth_l1`; independent direction tasks record `train/cross_entropy`.
Historical pinball/ranking and multitask loss fields are described in the
[archived artifact guide](#legacy-artifacts-en).
`loss_log_points_per_epoch` defaults to 250, plus validation boundaries;
`evaluations_per_epoch: 5` corresponds to 20%/40%/60%/80%/100% of each epoch.
W&B uses `trainer/global_step` and `benchmark_validation/global_step`; same-step
loss and validation rows do not overwrite each other.

`status` lists W&B components for each run. Only `online_finished`/`synced` confirms
delivery; `offline_pending`, `sync_failed`, or `online_running` after workflow termination
requires recovery. Failed online initialization may preserve an offline transaction without
discarding completed model results.

W&B recovery is per model: obtain a child `MODEL_RUN_ID` from parent `status`, then provision
the matching dedicated Pod using `resume <MODEL_RUN_ID>` for unfinished training
or `validate <MODEL_RUN_ID>` for completed training. A parent ID does not sync both models.
**Do not start training/validation tmux.**
After SSH login, run:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_wandb_sync.sh "<MODEL_RUN_ID>"
```

A newly created run-scoped Pod may sync only its own run. Omitting the argument also selects
only that run, not other active experiments. All resume/re-evaluation transactions are
preserved and synced using the project's pinned W&B legacy sync path. The recovery Pod is
terminated afterward; do not start training in it.

<a id="resume-en"></a>

##### Resume training and run standalone holdout validation

Confirm the original run's Pod has terminated. Resume by run ID without changing active selection:

```bash
bash scripts/runpod_workflow.sh resume \
  --maxRuntime 12h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  "<RUN_ID>"
```

With no run ID, automatic selection requires exactly one incomplete resumable candidate.
Multiple candidates are listed and rejected instead of guessed. The creator restores the run's
stored selection/config and validates its checkpoint and numerical training contract before
renting a Pod. It prefers a valid newer temporary checkpoint, otherwise the retained checkpoint
with greatest global step, not the best validation score. Pair resume skips models with completed
training and test; it resumes unfinished training or only the missing test for a trained child.
A fully completed pair rejects resume before Pod creation. Completed individual-model training
continues to use `validate`.

The new Pod preserves the run ID and restores weights, optimizer, scheduler, RNG, progress and
early-stop state. After SSH login, run `bash scripts/runpod_tmux_launch.sh stage1-train`.
Changing capacity is not resume: model, data or numerical-training changes require a new run.
Selection, display or path-control changes alone do not invalidate prepared data or baselines.

For completed training whose holdout needs resuming or recomputing:

```bash
bash scripts/runpod_workflow.sh validate \
  --maxRuntime 8h \
  --gpuId "NVIDIA GeForce RTX 5090" \
  "<RUN_ID>"
```

Omitting the run ID selects the latest completed training run; runtime/GPU defaults are
`12h`/`NVIDIA GeForce RTX 5090`. `--resume` (default) reuses completed benchmark jobs;
`--no-resume` disables evaluation resume; `--force` recomputes main-model results and
disables resume. Prebuilt baselines are still read, not retrained. After SSH login:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh stage1-validate
```

Parent validation handles unfinished child evaluations sequentially on one Pod, without retraining.
If both tests are already complete and `--force` is absent, it skips locally without renting a Pod;
`--force` recomputes both. Different parents may validate concurrently, but no parent/child
training or validation may overlap another workflow of the same pair.

<a id="download-en"></a>

#### 6. Download checkpoints, loss logs and final evaluations

Downloads need neither a volume ID nor remote paths. For multi-experiment comparisons, specify
each run ID reported at creation:

```bash
bash scripts/runpod_workflow.sh download "<RUN_ID>"
bash scripts/runpod_workflow.sh download --checkpointScope best "<RUN_ID>"
bash scripts/runpod_workflow.sh download --resume --checkpointScope all "<RUN_ID>"
```

| Option | Default | Behavior |
| --- | --- | --- |
| `RUN_ID` | Latest completed results if omitted | Select by run-scoped terminal timestamps, not the most recently launched Pod |
| `--checkpointScope all\|best` / `--checkpoint-scope all\|best` | `all` | `all` is the retained best-five leaderboard set; `best` is the validation-selected best |
| `--resume` | Off | Required for an existing target; fills/refreshes known files without deleting other local checkpoints |
| `-h` / `--help` | None | Show download usage |

One parent download obtains both models and succeeds only when both final test results are complete.
`artifacts/runpod/<PARENT_RUN_ID>/` stores `pair.json` and `download-result.json`; weights are stored
once per model under `artifacts/runpod/<CHILD_RUN_ID>/`, referenced by relative paths from the parent.
`--resume` applies to both children, and checkpoint scope applies to each leaderboard separately.
At most two I/O workers run, further bounded by CPUs and available memory; set
`RUNPOD_RESULT_DOWNLOAD_WORKERS=1|2` to reduce concurrency. Weights are not loaded into Python RAM.
Explicit individual-model IDs retain the original download behavior.

Each model directory contains its run manifest, resolved config, leaderboard, selected
checkpoints, `completion-result/`, `metrics.jsonl`, `summary.json`, training/validation
lifecycles, training completion and final `validation-benchmark.json`. Historical runs with interval
calibration also include `interval-calibration.json`. Retention-deleted checkpoints are not
recreated; `--resume` never prunes a previous `all` download into `best`.

Checkpoint validation metrics select the model; final `validation-benchmark.json` compares
holdout results and baselines. Verify run IDs, configs, eligible evaluation membership and
completion before comparison. Scale representation probes are separate: use `download-probes`
under [Download diagnostic results locally](#download-diagnostic-results-locally).

<a id="artifacts-en"></a>

### Training and inference artifacts

`download` includes `metrics.jsonl` and `summary.json`, plus `interval-calibration.json`
for runs with interval calibration enabled. Missing required artifacts fail clearly
for new runs; historical logs that were never produced and calibration files for
uncalibrated runs are not required. Loss records contain run
and session IDs, timestamps, optimizer steps, cumulative sample presentations and
epoch fractions. Resume appends a new session without overwriting old curves.
Independent tasks record normalized Smooth L1 or CE as `train/loss` and `validation/loss`,
with explicit `train/normalized_smooth_l1` / `train/cross_entropy` names and no fabricated
pinball. `validation/loss` records the configured task's full-validation checkpoint monitor,
not a universal cross-task score. See the [archived artifact guide](#legacy-artifacts-en)
for obsolete multitask losses and output schemas.
Losses are averaged between the configured
`loss_log_points_per_epoch` points (default 250), including validation boundaries.
Every local record is flushed/fsynced and remains on the network volume even with W&B disabled.

The parent pair manifest records group, seed, task identities and both immutable selections;
it is not a third model. Each child independently stores at least:

- `adapter.safetensors`: trainable LoRA, resampler, benchmark-conditioner, the selected task's output head,
  and any Kronos weights unfrozen by the selected experiment.
- `resolved-config.yaml`
- `trainer-state.json`
- Optimizer and scheduler state
- Run manifest, checkpoint leaderboard, and best-checkpoint pointer
- `completion-result/`: final trainable weights, stop reason, and audit counters at normal or early-stopped completion
- Selection ID/SHA, dataset request SHA, stage-config SHA, and requested dataset contract
- Dataset-manifest summary, architecture digest, Kronos source/model/tokenizer
  revisions, and bounded training-implementation digest
- Training-selection validation metrics and final holdout/baseline comparisons
- `runtime_scale_features` schema, train-only normalization and fingerprint when enabled

Checkpoint resume must match the run's task/output schema (`5.0` for legacy quantiles,
`signal-1.0` for independent tasks), never cross task boundaries, and must pass
the RunPod run-identity, artifact-integrity, validation-selection, dataset,
model, and training-source contract checks. Any incompatible schema fails
closed.

Read-only inference and `probe-scales` permit historical proportional-split
checkpoints without the new branches when their original resolved settings,
dataset, and artifact hashes still agree. This exception never authorizes training
resume, loading an old checkpoint under a new fixed-date config, or relabelling
old scores as new holdout results.

Inference also runs inside a RunPod Pod with the project Poetry environment and
the same network volume mounted; do not load the checkpoint locally. Replace the run/checkpoint
IDs and `<MARKET_PARQUET>` with an actual input file containing the target and benchmark.
Use a child `MODEL_RUN_ID` from parent status, not the parent ID; inference loads one model at a time.
This example does not depend on `DATA_ROOT` being inherited by the SSH session:

```bash
poetry run stock-forecasting-infer \
  --config "/runpod-volume/savedModel/<MODEL_RUN_ID>/<CHECKPOINT>/resolved-config.yaml" \
  --checkpoint "/runpod-volume/savedModel/<MODEL_RUN_ID>/<CHECKPOINT>" \
  --input "<MARKET_PARQUET>" \
  --symbol AAPL.US
```

All inference returns `forecast`, data provenance, encoder shapes and checkpoint metadata.
Independent point `forecast.by_horizon` contains only `alpha_point`; independent direction
contains only `absolute_direction_probabilities`, with explicit `down, neutral, up` order and
inclusive neutral-band boundaries. Neither task emits ranking, five-level trading signals,
cost scenarios or calibrated quantiles. Archived inference fields are documented in the
[obsolete-experiment appendix](#legacy-artifacts-en).
No natural-language explanation is generated.

<a id="probes-en"></a>

### Historical scale representation diagnostics

`probe-scales` loads an existing checkpoint, freezes Kronos / LoRA / resampler / conditioner /
head weights, and fits separate ridge probes to test historical-scale decodability. It does
not retrain the forecasting model or add a numerical feature branch. It preserves the
checkpoint's train/validation split membership and date boundaries, creates no test loader,
and calculates no future alpha labels.

`probe-scales` **does not create a GPU Pod automatically and cannot run model diagnostics
on the local control machine**. Follow this sequence:

1. **Local control machine: confirm source readiness.** Already-synced unchanged source needs
   no upload. If an update is needed, wait until no Pod owns the volume before synchronizing. Use `bash scripts/runpod_workflow.sh sync --apply`
   and the existing cloud deployment workflow. GPU readiness must pass, and the original
   network volume must already contain a usable project environment.
   Update both the local checkout and volume to support the diagnostic lifecycle before
   creating the Pod and local guard. Updating files does not add monitoring to an already
   running older guard process.
2. **Local control machine: create a GPU Pod.** Reuse an idle GPU Pod mounting that same
   volume only if its local guard supports the diagnostic lifecycle and is still running.
   Otherwise use the dedicated diagnostic creation command under
   [advanced individual-model operations](#single-model-en), not the paired training entry.
   Data/model/baseline readiness and a guarded run identity are still required. Creation
   does not start training; the next step starts diagnostics through the existing tmux route.
3. **GPU Pod: start diagnostics through tmux.** SSH into the Pod and run the commands below.
   **Do not execute** the
   suggested `runpod_tmux_launch.sh stage1-train` command or start `stage1-validate`.

Inside the GPU Pod, omit the checkpoint option to select the latest completed training run
on the mounted volume and use that run's validation-selected best checkpoint:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh probe-scales
```

Pass a child-model or historical individual-model **run ID**, not the parent ID or a full path:

```bash
cd /runpod-volume/stock_forecasting
bash scripts/runpod_tmux_launch.sh probe-scales \
  --checkpoint "<RUN_ID>" \
  --train-samples 16384 \
  --validation-samples 4096 \
  --batch-size 16 \
  --ridge-alpha 10 \
  --seed 42
```

"Latest completed" uses `created_at` in each atomically published
`completion-result/training-result.json`, including normal completion and early stopping.
It does not use directory modification times, run-ID chronology, the active selection or
the highest checkpoint step. In-progress runs without completion metadata are excluded;
equal completion timestamps are deterministically ordered by run ID. The selected run's
`best-checkpoint.json` and retained checkpoint must pass integrity checks. Missing/corrupt
selection artifacts or pending transactions fail explicitly, without falling back to an
older model. Malformed completion metadata also stops discovery; select another run
explicitly or address the damaged artifacts first.

`--checkpoint RUN_ID` selects that run's integrity-validated best checkpoint. Explicit
selection does not require completed training, but the checkpoint must be fully committed
and the GPU lease must be available. For compatibility, canonical absolute
`/runpod-volume/savedModel/<run-id>` and exact `checkpoint-NNNNNN` paths remain supported;
relative paths, path traversal and symlinks are rejected. Configuration always comes
from that checkpoint's `resolved-config.yaml`, never the current active stage selection.
The original bound bar store, dataset manifest and model/tokenizer cache must still exist
and pass compatibility checks; do not substitute another dataset version.
Runs with a pending checkpoint-selection transaction are rejected before shared checkpoint
validation can repair anything. Recover through the original training workflow first;
the read-only diagnostic never triggers checkpoint reconciliation or deletion.

Use `runpod_workflow.sh` locally to create the Pod, then use
`runpod_tmux_launch.sh probe-scales` inside the Pod to start diagnostics, consistently
with the deployment and training workflows above. This starts the detached
`stock-forecasting-probe-scales` session. After `Detached tmux session started` appears,
SSH may disconnect without stopping the job.

While the Pod is still running, reconnect over SSH and attach to view live output.
Use `Ctrl-b d` to detach without stopping work; do not use `Ctrl-c` to detach:

```bash
tmux -L stock-forecasting-probe-scales attach -t stock-forecasting-probe-scales
```

Diagnostics reuse the existing timeout and **local monitoring/termination** flow. After
acquiring the exclusive GPU lease, the runner publishes an independent running marker.
Success, failure and timeout persist logs/status before publishing a terminal diagnostic
signal. The local `terminate_runpod_after.sh` reads it over S3, validates its Pod ID,
Pod-creation owner run ID, launch ID and state, then invokes the local
`runpodctl_project.sh pod delete`. The owner run ID identifies this Pod allocation, not
the historical model run selected by `--checkpoint`. Diagnostics do not call the Pod-side
termination API or require the local RunPod API key inside the Pod. Training and validation
lifecycle completion markers remain unchanged. An existing session, rejected launch
preflight or failure to acquire the GPU lease emits no diagnostic termination signal,
protecting existing work; the local guard still follows that Pod's configured policy.
The diagnostic runner uses `MAX_RUNTIME_SECONDS` from Pod creation (two hours above),
plus a 60-second forced-termination grace period. This bounds the diagnostic process, not
cloud charges: termination still requires successful local-guard polling and RunPod API
confirmation.
`awaiting Pod termination by the local guard` means computation has finished and the
runner is waiting for the next successful local guard poll. It retains the GPU lease
while waiting so another job cannot start just before termination.
**SSH may disconnect, but the local guard host must remain powered on, online, and keep
the guard process running.**

The launcher prints the exact paths for this invocation. Execution status is stored
separately from numerical diagnostic reports:

```text
/runpod-volume/logs/tmux/stock-forecasting-probe-scales/<launch-id>/
  combined.log                 # Worker stdout/stderr and local-guard handoff messages
  status.json                  # Terminal job state: succeeded / failed / timed_out
  runner.sh                    # Quoted arguments, timeout, lease and local-guard handoff

/runpod-volume/lifecycle/diagnostics/representation-scales/<pod-id>.json
                               # running / succeeded / failed / timed_out; Pod and owner identity
```

`status.json` is published when the job exits. A launched session does not establish a
successful diagnostic, and `succeeded` does not establish confirmed Pod termination.
Termination requests and retries are recorded **locally** in
`~/.local/state/runpod-guards/<pod-id>.log` (or the custom Guard log path printed at Pod
creation), not a Pod-side `pod-shutdown` directory. Confirm termination from RunPod's
actual Pod state. Unreadable terminal signals require checking the local guard and actual Pod state;
do not assume billing has stopped.
After Pod termination, attach is unavailable; retrieve logs, status and reports from the
persistent network volume instead.
List all diagnostic options with:

```bash
bash scripts/runpod_tmux_launch.sh probe-scales --help
```

Automatic run discovery uses a bounded metadata
thread pool; `--selection-workers 1..8` sets its upper limit, further constrained by visible
CPUs and available memory. It never loads every run's model weights. Reduce `--batch-size`
if extraction runs out of GPU memory; sample membership is batch-size invariant.
For SSH sessions, the script reuses the existing allowlisted PID 1 environment importer;
no manual stage, volume or credential exports are required.

The diagnostic includes:

- Four readouts: concatenated asset/benchmark Kronos masked means, concatenated last valid
  Kronos tokens, concatenated resampler latent means, and the exact conditioned-token mean
  consumed by the alpha head.
- Eight past-only targets: asset, benchmark and asset-minus-benchmark daily log-return
  standard deviations over 20/60 trading returns, plus each stream's full-context close-price
  `std / mean`. Inputs use as-of adjusted prices, `ddof=0`, no annualization and at least 61
  valid bars. These are not future holding-period alpha targets.
- Seeded uniform sampling without replacement within each original split, shared by all
  readouts. Probe train candidates cover the **full train split**, not necessarily the 5%
  subset seen during Stage 1. Sampling is not equal-weighted by symbol or date.
- Train-only feature/target scalers and a fixed ridge alpha, without validation tuning.
  Controls use the train target mean and a ridge probe fitted to shuffled train targets.
- Train/validation R², MAE, RMSE, Pearson r and MSE skill relative to the train-mean baseline;
  per-market validation metrics, feature dimensions, constant-feature counts and train
  samples per feature.

Every execution creates a new independent directory:

```text
/runpod-volume/diagnostics/representation-scales/<run-id>/<checkpoint>/probe-<UTC>-<id>/
  status.json                  # Only state=complete establishes a completed diagnostic
  probe.log
  report.json                  # Metrics, settings, data/checkpoint/source SHA-256 provenance
  summary.md                   # Full Traditional Chinese section, then full English section
  samples.jsonl                # Split-local row order, sample IDs, cutoffs, symbols and markets
  probes.npz                   # Train-only scalers, ridge coefficients and shuffle permutation
  validation_predictions.npz   # Historical targets and predictions in validation row order
```

Checkpoints, best pointers, existing validation reports and completion markers are not
overwritten. Partial artifacts from a failed attempt are not completed results; retrying
creates a new directory. The existing `download` command remains scoped to training and
validation artifacts and does not automatically retrieve this diagnostic directory.
Diagnostic outputs and tmux logs survive Pod termination on the persistent network volume
and can be retrieved through its S3 interface.

#### Download diagnostic results locally

From the **local project root**, use one command to download the latest completed
historical scale representation diagnostic:

```bash
bash scripts/runpod_workflow.sh download-probes
```

To download the latest completed diagnostic for a particular model run, add only
`PROBE_RUN_ID`:

```bash
bash scripts/runpod_workflow.sh download-probes "<RUN_ID>"
```

`PROBE_RUN_ID` is the diagnosed model's run ID, the same ID accepted by `--checkpoint`
when starting diagnostics. Omission searches the entire project volume; an explicit ID
restricts the search to that run. Both forms download just the latest completed result.
If diagnostics were repeated for one model, the script selects the result automatically;
users do not need to distinguish storage directories.

The script reads the network volume ID, S3 credentials, region and endpoint from the
project `.env`, then resolves the checkpoint, diagnostic directory and local destination.
**No volume ID, path, additional identifier or preliminary remote listing is required.**
Downloads work after Pod termination without creating a Pod, rerunning diagnostics or
retrieving model weights.

The latest result is selected by report `created_at` among diagnostics with
`state: complete`, not by training completion time or directory names. Running and
failed results are excluded. The downloader checks all seven artifacts, verifies that
report / status match the selected result, and checks the three numerical artifact
SHA-256 values recorded in the report. Missing completed results, missing files and
verification failures produce explicit errors, never a successful incomplete download.

Results are kept separately under the Git-ignored
`artifacts/diagnostics/representation-scales/`. The command prints the result directory
and `summary.md` path. Read the summary first, then `report.json` for full metrics.
After interruption, rerun the same command without `--resume`. Only the selected
diagnostic's known artifacts are refreshed after verification; other files are not
deleted. The download includes `probe.log`.

#### Interpret diagnostic results

`probes.npz` stores `<readout>__feature_mean/feature_scale/coef/intercept`.
Reconstruction uses `Xz = (X - feature_mean) / feature_scale` and
`Yz = Xz @ coef.T + intercept`. The first eight columns predict true targets and the last
eight form the shuffled-label control. Restore each group with
`Y = Yz * target_scale + target_mean`; target order is in
`report.json.target_contract.names`. High-dimensional extracted representations are not saved.

Check whether validation outperforms both controls. MSE skill is
`1 - MSE_probe / MSE_train_mean_baseline`; positive values beat that baseline. R² instead
uses the validation mean in its denominator. Constant-target R², constant-vector Pearson r
and zero-denominator skill are JSON `null` / Markdown `N/A`. High train but low validation
scores can indicate probe overfitting or distribution shift. A weak pooling + linear probe
does not establish absence of information. Readout dimensions differ, so score gaps alone
do not establish information loss. Overlapping windows/shared benchmarks are not independent
samples; this feature makes no significance claim or automatic architecture decision.
Scale decodability does not establish future-alpha predictability. The default 16,384 /
4,096 sample limits bound cost; they do not guarantee statistical sufficiency.

An existing cloud Pod with the project environment can first run the download-free
synthetic-data/mock-checkpoint contract tests:

```bash
cd /runpod-volume/stock_forecasting
.venv/bin/python -m pytest tests/test_representation_scale_probe.py
```

<a id="cli-reference-en"></a>

### CLI parameter reference

This reference covers project commands used in this README. `[ARG]` is optional;
`<ARG>` is a value to replace. Do not type the brackets literally. Do not pass lower-level
helper options to the workflow unless listed here. `bash scripts/runpod_workflow.sh --help`
(also `-h` or `help`) lists entry points; not every shell subcommand has its own `--help`.
Normal workflows read volume/credential settings from the project's `.env`; no extra
volume ID or data path is needed.

#### Local control commands

Append the commands below to `bash scripts/runpod_workflow.sh`. They do not run models locally.

| Command | Options, defaults and behavior |
| --- | --- |
| `credentials` | No CLI options; interactive hidden credential entry |
| `tpex-relay configure`, `deploy`, `verify`, `status` | No extra CLI options; respectively configure interactively, deploy, verify, or read status. `tpex-proxy` is an alias for this entry point |
| `volume deploy` | `--name NAME` defaults to `stock-forecasting` (1–63 alphanumeric, `.`, `_`, `-` characters, starting with an alphanumeric); `--size-gb N` defaults to `100`, range 10–4000; `--datacenter ID` defaults to `EU-RO-1`; `--force-new` creates and registers another billable volume and **does not migrate existing data**. An already registered volume is reused by default |
| `configure` | All options and restrictions are listed under “Configure parameters and dataset scope” above. No options opens interactive mode. `-h`/`--help` shows the underlying parser help; the workflow injects `--project-root`, which users need not supply |
| `selection show` | No extra options; display the active selection |
| `sync` | `--dry-run` is the default and only checks/lists planned uploads; `--apply` uploads. Choose one; no other options are accepted |
| `cpu prepare` | No options or `--interactive` opens interactive confirmation. Non-interactive calls require positive `--max-api-calls N`; `--eodhd-qps Q` defaults to `16`, `--taiwan-qps Q` to `0.5`, both positive; `--maxRuntime D` defaults to `6h`; `--prepareReserve D` defaults to `auto` (25% of runtime, capped at 2h), with an explicit duration shorter than runtime; `--maxBackoff D` defaults to `1m`; `--cpuNumber N` defaults to `8`, choices 2/4/8/16/32; `--cpuFlavor F` defaults to `cpu3g`, choices `cpu3c`, `cpu3g`, `cpu3m`, `cpu5c`, `cpu5g`, `cpu5m` |
| `readiness` | Choose one: `--code-only` checks uploaded code; `--gpu` checks main-model dependencies; `--baseline` checks current data-cleaning rules and baseline completion without the main-model HF cache. Add `--experiment a-signal64` or `b-signal64` to `--baseline` to check both tasks; omission retains active selection. `--experiment` is only accepted in baseline mode. Exit 0: complete; 1: incomplete/pending finalization; 2: verification error |
| `train`, `baseline` | `--maxRuntime D`: `12h`; `--gpuId ID`: `NVIDIA GeForce RTX 5090`. `train` accepts repeated `--experiment NAME`, repeated `--seed N` (0–4294967295; training seed when omitted), and `--launchWorkers N` (1–6, default 2; alias `--launch-workers`), at most 30 experiment × seed combinations. Each `a-signal64`/`b-signal64` group/seed creates one Pod running two independent models sequentially under one runtime. `baseline` accepts one `--experiment NAME`: a paired entry checks both task caches locally and builds only unfinished work without changing active selection; no seed/launchWorkers. Omitting experiment keeps active selection, **not an implicit pair**. Advanced individual entries remain in the appendix. Neither accepts a positional run ID |
| `resume [RUN_ID]` | A parent resumes unfinished models or missing test evaluation, skipping completed models; a complete pair rejects before renting. Individual-model resume remains supported. Omission requires exactly one eligible candidate. Restore original selection; `--maxRuntime D`: `12h`, GPU as above; use latest usable checkpoint, not best |
| `validate [RUN_ID]` | Omission selects latest completed training with original selection; `--maxRuntime D`: `12h`, GPU as above. `--resume` (default) reuses completed jobs; `--no-resume` disables continuation; `--force` recomputes and disables resume; choose one. Parent validation handles both models in one Pod. Both tests complete without `--force`: skip locally/no Pod; `--force`: recompute both. No baseline retraining |
| `runs` | `--experiment NAME`, `--seed N`, `--state STATE`, `--limit N`, `--offset N`, `--workers N`, `--output table\|json`, `--timezone ZONE`; list run IDs, seeds, execution times and frozen settings; see [run queries](#run-status-en) |
| `status [RUN_ID]` | `--run-id RUN_ID` is an alternative to the positional ID; also `--output table\|json`, `--timezone ZONE`; no arguments retain the project overview, an ID checks that run's full completion |
| `cpu-logs` | No options; download CPU workflow logs |
| `gpu-status` | Read local GPU records; optional, combinable `--pod-id ID` and `--run-id ID`; `--output table\|json` defaults to `table`; `--guard-dir PATH` defaults to `RUNPOD_GUARD_LOG_DIR` or `~/.local/state/runpod-guards`; `-h`/`--help` shows help |
| `gpu-notify-test` | No options; submit an explicitly labelled macOS notification test without querying/modifying Pods |
| `recover` | Diagnostic-only by default; `--apply` restores guards/terminates Pods confirmed safe to terminate; `--pod-id ID` restricts scope; `--confirmations N` defaults to `2`, minimum 2; `--confirmation-delay-seconds N` defaults to `5`, nonnegative; `--guard-dir PATH` is an advanced control-host log location, defaulting to `RUNPOD_GUARD_LOG_DIR` or `~/.local/state/runpod-guards`; `-h`/`--help` shows help |
| `cleanup-monitors` | Required `--pod-id ID`; dry-run by default, `--apply` terminates verified local guard/GPU-monitor/guard-owned caffeinate processes. API must confirm Pod absence; existence, API failures or identity errors block cleanup. `--guard-dir PATH` defaults to `RUNPOD_GUARD_LOG_DIR` or `~/.local/state/runpod-guards`; `-h`/`--help`. No log deletion, shared-SSH closure or remote changes |
| `download [RUN_ID]` | A parent downloads both models; omission selects latest complete results without separately auto-selecting linked children. `--checkpointScope all` (default) obtains each retained checkpoint set; `best` selects each best. `--resume` fills/refreshes both local results, **not training continuation**; `-h`/`--help` shows help. Explicit child IDs retain individual downloads |
| `download-probes [PROBE_RUN_ID]` | Without an ID, download the latest successful diagnostic; supply the diagnosed model's training run ID to download its latest successful diagnostic. No checkpoint ID, probe ID, volume ID or path is needed; `-h`/`--help` shows help |

Durations `D` accept only a positive integer followed by `m`, `h` or `d` (e.g. `30m`,
`12h`, `2d`), not `1.5h`. Supported aliases are `--max-runtime` = `--maxRuntime`,
`--gpu-id` = `--gpuId`, and `--checkpoint-scope` = `--checkpointScope`. CPU prepare
also accepts `--maxApiCalls`, `--eodhdQps`, `--taiwanQps`, `--prepare-reserve`,
`--max-backoff`, `--cpu-number` and `--cpu-flavor`. Examples use consistent primary spellings.
GPU train/validate runtime is a **request to stop after safely saving, not a hard cost cap**;
baseline/CPU deadline behavior is described in their operation sections. GPU catalog
`--data-center` is not a train/baseline option.

`bash scripts/verify_runpod_s3_access.sh` takes no user CLI options and reads volume/S3
settings from `.env`. All `bash scripts/runpodctl_project.sh gpu list` options are covered
under [Query GPU resources](#gpu-catalog-en).

#### In-Pod workflows and diagnostics

Normal `WORKFLOW` values for `bash scripts/runpod_tmux_launch.sh WORKFLOW` are
`cpu-prepare`, `cpu-finalize`, `stage1-train`, `stage1-validate` and `baseline`. They take
no extra arguments: selection/environment values are passed when the Pod is created.
Only `probe-scales` forwards the following diagnostic options:

| Option | Default or limit | Behavior |
| --- | --- | --- |
| `--checkpoint MODEL_RUN_ID` | Latest completed individual model | Use its validation-selected best checkpoint; supply a child model ID, not a parent ID |
| `--train-samples N` | `16384`, minimum 2 | Train representation sample cap; does not retrain the forecast model |
| `--validation-samples N` | `4096`, minimum 2 | Validation representation sample cap; does not use holdout |
| `--batch-size N` | `16`, positive integer | GPU representation-extraction batch size; reduce on OOM |
| `--num-workers N` | `0`, range 0–16 | Diagnostic DataLoader workers, separate from train/test loader auto-tuning |
| `--ridge-alpha X` | `10`, finite positive number | Ridge regularization strength |
| `--seed N` | `42`, integer 0–4294967293 | Reproducible sampling and shuffled-label control |
| `--selection-workers N` | Auto, range 1–8 | I/O concurrency cap for scanning completed-training metadata, also limited by CPU/available memory |
| `-h`, `--help` | Flag with no value | Show diagnostic options without launching a diagnostic |

`bash scripts/runpod_wandb_sync.sh [MODEL_RUN_ID]` in a new run-scoped Pod processes only its
own run, even when the ID is omitted. A different ID or `--help` is rejected before lease
admission. Only historical unscoped Pods retain all-run discovery; do not use that behavior
with concurrent experiments. Once admitted, exit invokes the existing termination flow.
Use a dedicated idle sync Pod, then confirm its termination locally; help is not a safe
live-training diagnostic.

#### Remote low-level data and inference CLIs

These are the full options for the `poetry run stock-forecasting-*` developer CLIs.
They are for debugging on a cloud Pod with an existing environment, not replacements
for local workflows. Normal operation does not require manual data paths or identities.
All four Python CLIs accept `-h`/`--help`, still within the existing cloud environment.

`poetry run stock-forecasting-download`:

| Option | Default or purpose |
| --- | --- |
| `--profile NAME` | `FIN_TS_DATASET_PROFILE`, otherwise `us_tw_eodhd`; implemented choices are `tw_only`, `us_only_eodhd`, `us_tw_eodhd`. The parser also reserves unimplemented `us_tw_massive`; do not use it for production acquisition |
| `--start DATE`, `--end DATE` | Required `YYYY-MM-DD`, inclusive start and exclusive end |
| `--symbols SYMBOL ...`, `--etf-symbols SYMBOL ...` | Space-separated explicit US lists; omitting both enables discovery. Does not filter Taiwan instruments |
| `--symbol-limit N` | Unlimited by default; cap US stocks and allowlisted ETFs separately at N, then ensure VTI is included. For bounded verification only |
| `--interval 1d` | Only `1d` is supported |
| `--output PATH` | Required canonical Parquet destination; no silent overwrite |
| `--manifest-root PATH` | Derive the dataset root from output by default (use the parent of `raw/` when output is in `raw/`); stores the download manifest |
| `--raw-cache-root PATH` | Defaults to `<manifest-root>/api-cache` |
| `--provider-checkpoint-root PATH` | Defaults to `provider-checkpoints` beside the raw cache; stores resumable provider Parquet |
| `--progress-path PATH` | Defaults to `<manifest-root>/download-progress.json` |
| `--cache-revision LABEL` | `RUNPOD_DATASET_REVISION`, otherwise `v1`; isolates provider-cache revisions |
| `--dataset-request-sha256 HASH`, `--selection-id ID`, `--selection-sha256 HASH`, `--launch-id ID` | Workflow provenance, respectively from `RUNPOD_DATASET_REQUEST_SHA256`, `RUNPOD_SELECTION_ID`, `RUNPOD_SELECTION_SHA256`, `RUNPOD_LAUNCH_ID`, or unset. Do not invent values |
| `--max-api-calls N` | This low-level CLI defaults to `100000`; limits EODHD network attempts per acquisition, including retries |
| `--eodhd-qps Q`, `--taiwan-qps Q` | `16`, `0.5`; provider request throttles |
| `--max-backoff-seconds S` | `RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS`, otherwise `60`; seconds, unlike workflow `1m` |
| `--workers N` | `FIN_TS_CPU_WORKERS`, otherwise the low-level fallback `1`; production CPU workflows configure hardware-aware concurrency rather than using this fallback as the production setting |
| `--acquisition-deadline-epoch-seconds T`, `--preparation-reserve-seconds S` | Unset by default; absolute Unix deadline and reserved preparation seconds, configured by the workflow |
| `--exclude-delisted` | Delisted instruments are included by default; this flag excludes EODHD delisted instruments |

`poetry run stock-forecasting-prepare`:

| Option | Default or purpose |
| --- | --- |
| `--input PATH`, `--output DIR` | Required downloaded raw Parquet and resumable bar-store directory; does not download market data |
| `--download-manifest PATH`, `--dataset-manifest PATH` | Default to `download-manifest.json` and `dataset-manifest.json` in the dataset root. The latter must reside directly in that root and cannot overwrite a ready manifest |
| `--benchmark-mapping PATH` | Optional JSON mapping; otherwise use the default benchmark policy. Does not relax the ETF allowlist |
| `--fixed-evaluation` | Flag enabling production exclusive split boundaries at 2025-06/2025-12/2026-06; omitting it uses legacy proportional splits |
| `--window-size N` | `128` bars |
| `--h-start N` | `1`, choices 1/2/3; maximum horizon is fixed at 14 |
| `--max-abs-log-return X` | `0.5`, legacy preparation extreme-transition flag; production runtime rebuilds membership from `configs/data_cleaning.json` and does not use this flag to remove genuine extreme returns |
| `--train-fraction X`, `--validation-fraction X` | `0.70`, `0.15`, used only by proportional splitting |
| `--purge-bars N` | `20`, proportional-split purge; fixed-date mode uses label-end boundaries instead |
| `--stride N`, `--sample-stride N` | Compatibility fields accept only `5`, `1`, not arbitrary sampling strides |
| `--target-horizon N`, `--diagnostic-horizons N ...` | Compatibility fields must be `5`, `1 20`; actual output still covers h-start through day 14 |
| `--embargo-bars N`, `--effective-embargo-bars N` | Compatibility fields must be `5`, `14`; fixed-date mode does not additionally apply proportional-split embargo |
| `--flat-volatility-multiplier X` | `0.25`, a compatibility field retained in preparation provenance, not a classification objective |
| `--bucket-count N`, `--batch-rows N` | `128`, `1000000`; bar-store bucket count and input chunk limit |
| `--deadline-epoch-seconds T` | Optional absolute Unix deadline; pause at a safe boundary for continuation |
| `--workers N` | Positive integer; `FIN_TS_CPU_WORKERS`, otherwise low-level fallback `1`. Individual phases lower actual concurrency according to CPU/memory limits |

`poetry run stock-forecasting-infer` requires `--config PATH`, `--checkpoint PATH` and
`--input PATH`: resolved config, checkpoint/run directory, and market data containing
both instrument and benchmark. `--symbol SYMBOL` is required for multiple target symbols;
optional `--as-of TIMESTAMP` is an inclusive UTC cutoff, otherwise use the latest available
common date. Optional `--output PATH` saves JSON; without it, output goes only to stdout.
The `probe-scales` Python CLI uses the diagnostic options above; operate it through tmux.

External-tool example options: in `tmux -L SOCKET attach -t SESSION`, `-L` selects the
dedicated socket and `-t` selects the session; do not substitute another workflow's socket.
`gcloud auth login` logs in interactively; additional options are documented in the
[official gcloud auth login reference](https://cloud.google.com/sdk/gcloud/reference/auth/login).
In `.venv/bin/python -m pytest TEST_FILE`, `-m` runs a Python module and `TEST_FILE` limits
the test scope. Pytest `-k EXPR` selects tests, `-q` reduces output and `-h` lists third-party
options. Run these only on the cloud Pod.

<a id="acceptance-en"></a>

### Acceptance and interpretation

1. Training uses dynamic sampling. Validation/test enumerate the complete cleaned population in
   fixed order, retaining the final short batch; estimates or sampling never replace full evaluation.
2. All splits use the same continuous input/output, valid-observation and liquidity rules while
   retaining real extreme returns. A/B share one evaluation source, not merely dates or row counts.
3. Both independent tasks have their own architecture and weights, initialized from the same
   pretrained base. Resume restores the original architecture, optimizer, RNG and progress.
4. Readiness checks the selected dataset's own manifest and actual artifacts. Another dataset's
   last-written CPU summary need not match stage/config, but profile, universe and content cannot
   be silently substituted.
5. Independent tasks output only `[B,15-h_start]` alpha points or `[B,15-h_start,3]`
   absolute-stock logits (probabilities after softmax). The fixed ±1% neutral band includes its boundaries; positive alpha is
   not absolute stock up, and no synthetic quantiles are emitted.
6. Inputs and train-only normalization exclude future data. Validation selects checkpoints;
   neither current task fits interval calibration. Record each task's train/validation loss
   and evaluate the selected checkpoint on the untouched complete holdout.
7. Main evaluation reads completed same-task baselines on identical membership without retraining.
   Points report regression/sign/valid RankIC and classifiers report probability/class diagnostics.
   Both retain market/month slices without mixing task-specific metrics. Obsolete experiments
   retain their separate [archived evaluation contracts](#legacy-objectives-en).
8. Multi-Pod acceptance covers independent selections, run IDs, checkpoints, logs, leases and
   guards. One Pod finishing must not terminate another; sync/runtime updates cannot overwrite
   in-use shared resources.
9. Each paired experiment/seed creates one Pod and one parent ID; its two models execute
   sequentially with independent training state. Completion checks both task-correct training/test
   artifacts. Resume skips completed work, and one parent download obtains both results without
   duplicate weights or triggering baseline retraining.

Stage 1 establishes workflow viability, not alpha. One seed, one holdout or high GPU utilization
does not demonstrate robust predictive superiority. Stronger claims require multiple seeds,
controlled comparisons and subsequent backtests untouched by tuning.

Engineering evidence is version- and scope-specific; it does not automatically validate later
changes or production-scale training:
[integrated market-adaptation verification](docs/integrated_adaptation_verification.md),
[multi-Pod control and documentation verification](docs/runpod_parallel_verification.md),
[cleaning/capacity verification](docs/cleaning_experiments_verification.md),
[tensor-pipeline verification](docs/baseline_tensor_pipeline_validation.md),
[baseline runtime verification](docs/baseline_runtime_validation.md), and
[full-evaluation workflow verification](docs/performance_workflow_validation.md).
See [reports](reports) for model analyses and [RELEASES.md](RELEASES.md) for version snapshots.

<a id="references-en"></a>

### Primary model and data references

- [Kronos paper](https://arxiv.org/abs/2508.02739)
- [Official Kronos repository](https://github.com/shiyu-coder/Kronos)
- [Official TimesFM repository](https://github.com/google-research/timesfm)
- [Google Research: TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Official Chronos repository](https://github.com/amazon-science/chronos-forecasting)
- [Chronos paper](https://arxiv.org/abs/2403.07815)
- [Official Uni2TS / Moirai repository](https://github.com/SalesforceAIResearch/uni2ts)
- [Moirai paper](https://arxiv.org/abs/2402.02592)
- [EODHD EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [EODHD API limits](https://eodhd.com/financial-apis/api-limits)
- [EODHD pricing](https://eodhd.com/pricing)
- [EODHD delisted data coverage](https://eodhd.com/financial-apis/delisted-stock-companies-data-2)
- [TWSE OpenAPI](https://openapi.twse.com.tw/)
- [TWSE ex-right/ex-dividend calculation](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx OpenAPI](https://www.tpex.org.tw/openapi/)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
- [Massive stocks pricing](https://massive.com/pricing?product=stocks)
- [Massive market-data terms](https://massive.com/legal/market-data-terms-of-service)

<a id="historical-experiments-en"></a>

### Appendix: obsolete experiments retained for reproduction

**This appendix documents superseded architectures, not the current experiment workflow.**
V1 (`direction64`), V2 (`direction64-v2`) and V3 (`direction64-v3`) are all obsolete multitask
experiments, not versions of the current independent point/classification architecture.
Normal new experiments use `a-signal64` or `b-signal64`, each containing both current tasks.

Settings, CLI names, weights and descriptions remain available for deliberate historical
reproduction and result interpretation. Nothing is removed or silently redirected. Do not
use archived launch examples to test current-model changes or resume an old run under a
different task. Historical baselines cannot substitute for current task-matched baselines.

<a id="legacy-presets-en"></a>

#### Archived experiment names and capacity controls

These sixteen A/B presets are retained for historical reproduction, not new-model entry points.
A starts at 2021-01-01 and B at 2016-01-01; both end at 2026-06-01 exclusive and use the shared
evaluation source. None is an alias for `alpha-point64` or `absolute-direction64`.
Original task/LoRA/unfrozen learning rates are `3e-5`/`5e-6`/`1e-6`; adaptive variants use the
separately documented rates below. Partial unfreezing remains a historical secondary research control.

| Archived experiment | Config files under `configs/experiments/` | Retained architecture / capacity |
| --- | --- | --- |
| `a-lora32`, `b-lora32` | `a_lora32.yaml`, `b_lora32.yaml` | LoRA rank 32, alpha 64 |
| `a-lora64`, `b-lora64` | `a_lora64.yaml`, `b_lora64.yaml` | LoRA rank 64, alpha 128 |
| `a-partial`, `b-partial` | `a_partial.yaml`, `b_partial.yaml` | LoRA-32 on the first 10 blocks; unfreeze the last 2 blocks and final norm without redundant LoRA |
| `a-adaptive64`, `b-adaptive64` | `a_adaptive64.yaml`, `b_adaptive64.yaml` | Market/ranking paths, regularization, sample scheduling and adaptive calibration control; LoRA rank 64, alpha 128 |
| `a-adaptive128`, `b-adaptive128` | `a_adaptive128.yaml`, `b_adaptive128.yaml` | Same integrated settings; only LoRA rank/alpha change to 128/256 |
| `a-direction64`, `b-direction64` | `a_direction64.yaml`, `b_direction64.yaml` | A-LoRA64 base plus dual three-class heads, ten technical channels, horizon queries and representation regularization; rank 64, alpha 128 |
| `a-direction64-v2`, `b-direction64-v2` | `a_direction64_v2.yaml`, `b_direction64_v2.yaml` | Same rank 64/alpha 128; direct direction-scale path, separated train priors, data-only references, market weighting and shared-LoRA negative-conflict PCGrad |
| `a-direction64-v3`, `b-direction64-v3` | `a_direction64_v3.yaml`, `b_direction64_v3.yaml` | Same rank 64/alpha 128; cost-ordered conditional probabilities, market-state feature selection, population probability quality and decision-error selection guards |

For the retained historical capacity controls, create three Pods to compare A's three
capacities. Each repeated `--experiment` takes one name:

```bash
bash scripts/runpod_workflow.sh train \
  --experiment a-lora32 \
  --experiment a-lora64 \
  --experiment a-partial \
  --launchWorkers 2 \
  --maxRuntime 12h \
  --gpuId "NVIDIA GeForce RTX 5090"
```

<a id="model-versions-en"></a>

#### Historical direction experiments V1/V2/V3: scope and differences

V1, V2 and V3 identify the retained **direction64 multitask family**, not Stage 1/Stage 2,
Git release versions or aliases for the current independent tasks. All three jointly learn
alpha quantiles, absolute-stock direction, alpha direction and independent ranking. Both
direction tasks use cost bands. These are reproduction contracts for historical experiments,
not objectives added back to the new tasks.

| Version | A / B experiment names | Direction architecture / schema |
| --- | --- | --- |
| V1 | `a-direction64` / `b-direction64` | `legacy` / direction schema 1 |
| V2 | `a-direction64-v2` / `b-direction64-v2` | `market_residual_v2` / direction schema 2 |
| V3 | `a-direction64-v3` / `b-direction64-v3` | `market_ordinal_v3` / direction schema 3 |

| Dimension | V1: initial dual direction | V2: direct scales and prior separation | V3: ordered probabilities and market-state selection |
| --- | --- | --- | --- |
| Labels | Separate absolute-stock and alpha down/neutral/up labels with cutoff-known cost bands | Same returns, costs and cleaning rules as V1 | Same as V2; no test-based threshold tuning |
| Direction head / probability structure | Horizon queries read tokens; three free logits per task | Retained horizon queries; historical signal `f` separated from train prior `π`, population logits `z=f+τ log π` | Two conditional logits yield ordered probabilities `[1-a, a(1-b), ab]`, not three free logits |
| Scales and market features | Direction heads do not directly read the existing 20-scale branch | Direct access to 20 scales and four cost/volatility ratios; historical residual plus market branch, market ID selects prior | Retained V2 paths; historical benchmark state gates three token streams and scale features |
| Loss references / task scales | Untrained-model forward pinball / dual CE on at most 1,024 samples; depends on model seed even with fixed sampling seed | Train-only null-quantile reference and fixed `CE=log(3)` on at most 8,192 market/year-stratified samples, independent of model initialization | Same fixed V2 references; classification uses conditional-chain NLL |
| Shared gradients / PCGrad | One backward pass through the weighted total loss | PCGrad for three task-gradient groups on shared LoRA only; other parameters use the total loss | Same as V2; ordered classification NLL still backpropagates into shared LoRA |
| Checkpoint / early-stop selection | Reference-weighted pinball plus two CEs | Market/task/horizon mean balanced-decision BA with a frozen pinball-degradation guard | Population CE/Brier skill relative to prior, macro-F1, opposite-error/collapse penalties and pinball guard |
| Baseline comparison | Matching quantile and dual-cost-direction baselines, not pure absolute-direction baselines | Reuse the same cost-multitask baseline when its numerical contract matches; added balanced/prior diagnostics do not make the baseline a V2 head | Same baseline under a matching contract; it is not a V3 ordinal architecture |
| Resume | Resume the same V1 run | Start a new V2 run; do not resume a V1 checkpoint | Start a new V3 run; do not resume V1/V2 checkpoints |

Controls shared across the three versions: Kronos-base, LoRA rank 64 / alpha 128, no partial
unfreezing, five technical channels each for asset and benchmark, horizon-query reading,
ranking weight `0.05`, representation-perturbation weight `0.05`, task/LoRA LR `3e-5`/`5e-6`,
dropout `0.05`, weight decay `0.01`, validation-plateau scheduling, and static interval
calibration fitted only after checkpoint selection. Full loss and selection formulas are in
the [archived objectives](#legacy-objectives-en). Version changes describe testable design differences,
not proven accuracy improvements. Compare raw metrics on matching samples, horizons, tasks
and seeds; `selection_score` values with different version definitions are not directly comparable.

#### Historical `quantiles` outputs and architecture

The following applies only to the existing `quantiles` task, not the two current independent tasks:

- `alpha_quantiles`: `[batch, 15-h_start, 3]`.
- Dimension two contains holding periods `h_start`, `h_start+1`, ..., 14;
  `h_start` is restricted to 1, 2, or 3; production defaults to 1. Historical
  resolved configs retain their original horizons.
- Dimension three is fixed to q10, q50, and q90.
- Units are adjusted execution log return relative to the instrument's benchmark.
- `ranking_scores`: `[batch, 15-h_start]` when the independent ranking head is enabled;
  dimensionless ordering scores, not returns, probabilities, or replacements for q50.

Original capacity and adaptive presets have no classifier. `direction64`, `direction64-v2` and `direction64-v3` add absolute-stock
and alpha direction logits, each `[batch, 15-h_start, 3]` in `down, neutral, up` order.
Inference includes probabilities and the cost scenario/bands under `direction`; alpha up
does not imply an absolute gain. Core alpha schema `model_output_schema_version=5.0`
is retained; the original additive direction output has schema version 1. V2 uses direction
schema 2; V3 uses schema 3. Both separately identify population probabilities and prior-removed balanced decisions.
Only `softmax(z)` supplies population probabilities; balanced decisions do not replace raw accuracy
or masquerade as population probabilities.
Inference JSON explicitly declares `direction.probability_source="population_logits"`.
`balanced_decisions.absolute` and `.alpha` map holding-period strings to class indices:
`0=down`, `1=neutral`, `2=up`, not probabilities. Their `source` is
`history_residual_logits_without_train_prior` for V2 and
`population_log_probabilities_minus_train_log_prior` for V3. V3 additionally declares
`probability_structure=ordered_conditional_chain`; its three outputs are conditional-chain log
probabilities, not independent logits. `train_prior` stores the training priors.
Quantile presets retain five-level quantile post-processing (`strong_bearish` through
`strong_bullish`), which is not the new classifier or a separate training objective.
Incompatible output schemas fail closed.

```text
OHLCV through close t (asset + benchmark)
  ├─ normalized windows → Kronos → resampler → benchmark conditioning → head trunk
  │                                └─ benchmark latent ────┐
  └─ 20 historical numerical features ──────────────────────┴─ numerical encoding

head trunk ── base quantile parameters ──┐
numerical encoding ── quantile residual ─┼─ q50 + positive tail widths → raw q10/q50/q90
head trunk + encoding ── market residual*┘                                  │
                                                         frozen validation calibration
head trunk ── ranking head ─────────────┐                 (+ separate delayed-feedback
numerical encoding ── ranking residual*─┴─ ranking scores   diagnostic on holdout)

* Additional paths enabled by adaptive64 / adaptive128; calibration keeps q50 unchanged.
```

`direction64` injects a zero-initialized technical projection between tokenizer embeddings
and the Kronos transformer: five channels per instrument/benchmark stream. Horizon-specific
queries cross-attend instrument, benchmark and conditioned latent tokens, feeding two
three-class heads. CE gradients reach Kronos/LoRA through shared resamplers; representations
are not detached. The original preset does not enable adaptive market/ranking residuals or adaptive calibration.
`direction64-v2` retains Kronos/LoRA, ten technical channels and horizon-specific reading, while
routing the existing 20 train-standardized scale features directly to the direction head.
Each horizon also receives four ratios: the two exact cost thresholds divided by asset/relative
20-day volatility times `sqrt(horizon)`. Predictor ratios are bounded to `[-20,20]` and transformed
with `asinh`; an unreachable lower band becomes finite only on this feature path. Exact labels,
cost scenarios and genuine extreme returns remain unchanged. Direction market residuals read
historical representations. Market IDs select output branches and train priors, without directly
adding market/cost metadata to historical-signal logits.

The historical `quantiles` task predicts the conditional alpha distribution directly; it does not
predict a raw-return q50 and subtract a benchmark q50. Historical benchmark
state affects predictions through both gated cross-attention and a direct
resampler-latent branch. Historical scales are extracted before Kronos window
normalization. Benchmark data
after close t is used only for offline label construction and never enters the
model input. These past-only input and benchmark rules also apply to both independent tasks;
their outputs remain a robust alpha point or absolute-stock class logits, not quantiles.

#### Archived numerical output-scale and adaptive branches

For the historical `quantiles` task, the scale MLP is `20 → 32 → 16`. Mean-pooled benchmark resampler latents use
`512 → 16`; the original horizon-conditioned head hidden state uses another
`512 → 16`. Concatenated features pass through `48 → 32 → 3` and are added to
the original raw quantile parameters **after** the head LayerNorm, before the
ordered q10/q50/q90 transformation. A US/TWSE/TPEx/unknown market embedding
conditions the head. Historical relative 20-day volatility times `sqrt(horizon)`
explicitly controls output scale, bounded to 0.1–10 times each horizon's train
robust scale, with an additional learned positive multiplier bounded to 0.25–4.
With `decoupled_output_scale`, lower and upper widths use separate positive gates;
the q50 location uses the train robust scale instead of inheriting historical
volatility. Subtracting/adding positive widths to q50 preserves quantile ordering. The market
embedding, residual output layer, and scale gate start at zero; the gate initially
multiplies by one. The head, feature
transforms and pinball loss use FP32; Kronos retains BF16 mixed precision.

Integrated `adaptive64`/`adaptive128` presets also enable `market_residual_hidden_dim: 64`:
head hidden states and the 48 encoded features pass through LayerNorm and a 64-unit MLP,
then separate zero-initialized US/TWSE/TPEx/unknown outputs adjust location and tail parameters
without copying the backbone. `ranking_numeric_features: true` routes the same 48 past-only
features through an independent `48 → 32 → 1` residual into ranking scores. Ranking output
parameters do not directly change q50; shared encoders still receive gradients from both tasks.

#### Historical V1/V2/V3 direction supervision and ten technical channels

`a-direction64`/`b-direction64` retain the original architecture for reproduction;
`a-direction64-v2`/`b-direction64-v2` retain market-residual V2, while V3 is defined below.
All three historical `direction64` versions build on `lora64` (not the independent
`absolute-direction64` task):
rank 64, alpha 128, no partial unfreezing,
task/LoRA LR `3e-5`/`5e-6`, dropout `0.05`, weight decay `0.01`, validation plateau scheduling
and static interval calibration. Original capacity/adaptive presets remain reproducible.
New runs start from pinned pretrained Kronos, not an old checkpoint with a different architecture.
V2 requires a new run and cannot resume an original `direction64` checkpoint as v2;
an interrupted run with the same v2 architecture remains resumable.

V2 learns a history-dependent direction signal `f` separately from train-only class priors `π`
for each market × task × horizon. Population logits are `z = f + τ*log(π)`, where
`τ=class_balance_power` defaults to 1. Training uses natural-class-population `CE(z)`, without
an additional inverse-class multiplier. Raw accuracy, CE and population probabilities use
`z`/`softmax(z)`. Separately reported `balanced_decision` uses `argmax(f)` to assess a decision
rule targeting equal-class recall rather than a market-majority shortcut; it is not a calibrated
trading win probability. This draws on [Menon et al.'s logit adjustment](https://research.google/pubs/long-tail-learning-via-logit-adjustment/),
whose non-financial experiments do not establish 60% accuracy or improved generalization here.

Both labels retain next-open to h-th-close adjusted execution returns. Absolute direction
uses the stock log return; alpha direction subtracts the benchmark log return. With estimated
round-trip fraction `0 <= c < 1`, returns below `log(1-c)` are down, above `log(1+c)` are up; boundaries
and the interior are neutral. The alpha band is a minimum relative-edge hurdle, not a hedged
portfolio P&L. A stock losing 4% against a benchmark losing 8% is absolute down but alpha up.
Long screening must not rely on alpha up alone; down is not a profitable-short recommendation.

Minimum fees can make `c >= 1` for very low-priced stocks. Keep those samples and returns:
the up hurdle remains `log(1+c)`, the down boundary is unreachable, and other finite returns
are neutral. Neither samples nor costs are clipped. Public JSON uses `null` for
`lower_log_return`; internal label comparisons use negative infinity. Neural and GBDT cost
covariates use `100 * tanh(log threshold)` to stay finite. This transforms predictor inputs
only, not exact label thresholds, alpha quantiles, or genuine extreme returns.

`model.direction.costs` freezes the scenario without brokerage API calls during training:

| Market / parameter | Default scenario |
| --- | --- |
| `taiwan_shares` / `us_shares` | 1,000-share Taiwan lots / 50-share US units through a direct US brokerage account |
| Taiwan commission | Equal average of Yuanta/Fubon published standard `0.1425%`, minimum NT$20 per side; compute each bill first, not a negotiated volume-weighted industry average |
| Taiwan sell tax | Stocks `0.3%`, admitted equity ETFs/TDRs `0.1%`; no assumed day-trading concession |
| US commission | Average IBKR Pro Fixed `max(0.005*shares, US$1)`, capped at 1% of value, and Schwab US$0 online listed-equity commission; applied to each leg |
| US regulatory allowance | SEC sell rate `0.0000206`, TAF `0` in the 2026 Q4 suspension scenario, conservative CAT allowance `0.000003` per share per leg, not necessarily separately charged by every broker |
| `slippage_bps_per_side` | Assumed combined spread/slippage of 5 bps per side; configurable, not an observed average brokerage fee |

Known cutoff-price-equivalent raw close times quantity estimates both legs, never future entry
or exit prices. At NT$100/1,000 shares, the default band is approximately `0.685%`; at US$20/50
shares it is `0.20303%`, including assumed slippage. `listed_cash_2026_10_09` is a fixed scenario,
not historical 2016–2026 fee reconstruction. FX, dividend withholding, ADR custody and stock-borrow
costs are excluded. A changed scenario needs a new run. Sources:
[Yuanta](https://www.yuanta.com.tw/eYuanta/agent/Node/Index?C1=2021102007380647&C2=2021102008672547&ID=2021102008672547&Level=2&MainId=00415),
[Fubon](https://www.fbs.com.tw/DomesticStocks/DomesticStocks), [TWSE](https://www.twse.com.tw/en/about/company/guide.html),
[IBKR](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php), [Schwab](https://www.schwab.com/pricing).
The regulatory scenario follows the [SEC 2026 fee advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2)
and [FINRA 2026 Q4 TAF suspension notice](https://www.govinfo.gov/content/pkg/FR-2026-09-23/html/2026-19392.htm).
The CAT allowance follows IBKR's published regulatory-fee table, not a uniform charge at every broker.
TDR detection covers both `91XX` and `91XXXX`, following the
[TWSE security classification](https://isin.twse.com.tw/isin/class_main.jsp?chklike=Y&issuetype=J&market=1).

Each stream uses these five point-in-time channels, ten total. Moving averages include the
current bar; the prior range excludes it. Each token uses its prefix only.

| Channel per stream | Definition |
| --- | --- |
| Price trend | `log(C_t / SMA20_t)` |
| Short MA trend | `log(SMA5_t / SMA20_t)` |
| Medium MA trend | `log(SMA20_t / SMA60_t)` |
| Prior range position | `(2*C_t - max(H_{t-20:t-1}) - min(L_{t-20:t-1})) / (max(H)-min(L))`; zero for a zero-width range |
| Volume confirmation | Sum of the last 20 `sign(C_j-C_{j-1})*V_j` values divided by volume over the same period; zero for zero volume |

Incomplete lookbacks produce zero; no context extension or missing-day filling occurs.
Clipping channels to `[-5,5]` limits feature leverage, not labels or genuine extreme returns.
MA/range motivation follows [Brock et al. (1992)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1992.tb04681.x);
trend/volume motivation follows [Neely et al.](https://files.stlouisfed.org/files/htdocs/wp/2010/2010-008.pdf).
The 5/20/60-day continuous adaptation is an engineering choice, not a proven optimum for this dataset.
Horizon queries are motivated by [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363),
without replacing Kronos with TFT.

<a id="legacy-objectives-en"></a>

#### Archived training objectives, selection and calibration

These definitions do not apply to current independent tasks. Common train-only numerical
scales and validation-plateau rules are described in the [current training section](#training-en).
Original Stage 1/2, capacity presets and V1/V2/V3 retain validation-plateau; adaptive presets use
the sample-based schedule below. All retained runs still evaluate complete splits.

| Archived family | Checkpoint monitor | Score definition |
| --- | --- | --- |
| Historical quantiles without dual-direction heads | `primary_5d/selection_score` | Mean normalized pinball across all horizons |
| Historical V1 `direction64` | `direction/selection_loss` | Fixed-reference-weighted pinball plus two CEs |
| Historical V2 `direction64-v2` | `direction/selection_loss` | Balanced-decision BA loss plus frozen pinball guard |
| Historical V3 `direction64-v3` | `direction/selection_loss` | Population probability quality, macro-F1, mistake/collapse penalties and pinball guard |

##### Historical quantile objectives and shared train-only scales

The following quantile/multitask objectives apply only to existing `quantiles` presets.
Normalized pinball divides each horizon's quantile error by its **train-only robust scale**
before applying pinball loss, then averages valid samples, horizons and q10/q50/q90.
Lower is better. It is neither a percentage nor directional accuracy/correlation and does
not guarantee 80% coverage. The retained monitor name `primary_5d/selection_score`
covers every horizon, not only day five.

For `u = (true alpha − predicted quantile) / train_scale[h]`, one quantile's loss is
`max(q × u, (q − 1) × u)`.

Training adds pairwise logistic ranking loss with weight `0.05` to normalized
pinball. An independent score head handles ranking instead of treating q50 as its
score. Pairs must share the cutoff date and market and represent different
securities; duplicate padding and nearly tied labels are excluded. Each microbatch
uses at most 256 pairs across all forecast horizons. A runtime-only date/market
index improves within-group batch membership, with annual-decay dynamic sampling
over cleaned train windows; it does not modify the prepared bar store. Original presets select
by full-validation normalized pinball; V1 uses the fixed multitask score below;
V2 uses equal-market direction metrics with a regression guard, and V3 uses the probability/
decision-quality composite below.
Ranking regularization and holdout never select checkpoints. No prediction ensemble is used.

##### V1: untrained-model references and weighted total loss

`direction64` first estimates untrained-model mean losses `sQ, sAbs, sAlpha` from at most 1,024
training windows with seed 59, then freezes these references. Default task multipliers are 1:

```text
L = pinball + (sQ/sAbs)*CE_absolute + (sQ/sAlpha)*CE_alpha
    + 0.05*ranking + 0.05*sQ*representation_regularization
```

CE averages all samples and t1–t14 equally. Initial task contributions are 1:1:1 on their own
reference scales, not an unscaled sum or a guarantee of equal gradients. This borrows the
initial-loss-ratio idea from [GradNorm](https://proceedings.mlr.press/v80/chen18a.html), not the
complete dynamic GradNorm algorithm. Equal multipliers are a neutral starting hypothesis,
not a universal finance optimum. Raw component losses, CE coefficients and validation losses
are logged to `metrics.jsonl`/W&B. Checkpoints/completion results persist references; resume
does not refit them.

Following the representation perturbation idea in [Feng et al.](https://arxiv.org/abs/1810.09936),
the direction hidden vector after LayerNorm receives an L2 perturbation of
`0.01*sqrt(hidden_dim)` along the unit joint-CE gradient, and the two reference-normalized
adversarial CEs are averaged. Perturbation direction is detached, not the underlying features.
No raw OHLCV alteration or second Kronos forward is required; evaluation has no perturbation.
Weight `0.05` and radius `0.01` are conservative engineering initial values, not published optima.

Original `direction64` monitors `direction/selection_loss`, the first three loss terms on full validation with frozen
references, excluding ranking/adversarial terms. `primary_5d/selection_score` retains its original
pinball meaning. Direction results include per-horizon, market and month confusion matrices,
accuracy, balanced accuracy, macro-F1, majority baseline, up precision/recall and signal fraction,
so neutral collapse is visible. `joint_up` records precision/counts when both heads predict up.
Balanced accuracy averages recall over classes present in that slice; macro-F1 averages all
three fixed classes. Confusion matrices and supports expose absent classes.
Three-class accuracy is not interchangeable with the old q50 sign agreement.

##### V2: direct scales, fixed references, prior separation and PCGrad

`direction64-v2` references require no untrained-model forward. Fixed calibration seed 59
selects at most 8,192 train windows using bounded market/year strata and annual-decay weights.
The sample budget is allocated to eligible markets and then to years proportionally to eligible
windows × annual weight; weighted aggregates recover the training population.
`sQ` is train-label normalized pinball for zero-centered null quantiles
`[-train_scale, 0, +train_scale]`; `sAbs=sAlpha=log(3)` is uniform three-class CE.
For the same data/configuration, random model initialization from different training seeds cannot
change CE coefficients. Default training retains pinball, ranking and representation regularization:

```text
L_v2 = pinball + (sQ/log(3))*market_weighted_CE_absolute(z)
       + (sQ/log(3))*market_weighted_CE_alpha(z)
       + 0.05*ranking + 0.05*sQ*representation_regularization
```

Fixed market weights come from annually weighted train-population mass. Capped water filling
simultaneously enforces train-weighted mean 1 and a final effective weight no greater than
`market_weight_max` (default 3). For active market proportions `p_m`, it uses
`w_m=min(k/p_m, market_weight_max)` and solves `k` so `sum(p_m*w_m)=1`.
This is not clipping followed by normalization, which could violate the cap again.
Weights are neither re-normalized within a microbatch,
fitted on validation/test, nor multiplied by inverse class frequencies. Checkpoints preserve
references, priors, market weights and sampling provenance; resume restores rather than refits
them. Horizons remain equally weighted; distant-horizon losses are not downweighted.

Shared-task gradients use conditional [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
projection. The three weighted tasks are pinball plus ranking, absolute CE and alpha CE.
Only shared trainable LoRA parameters are projected when an actual gradient dot product is
negative. With no negative conflict, backward matches the original path. Direction/alpha/ranking
heads, technical projections, numerical branches and other parameters retain ordinary total-loss
gradients; representation regularization also keeps its ordinary gradient. `monitor` measures
without projecting. Both production training and GPU batch-probe clipping set
`error_if_nonfinite=True`; gradient management does not mask nonfinite values, and detecting a
nonfinite gradient norm raises before that step's `optimizer.step()` rather than updating weights.
Norms, cosines, negative-conflict fractions and nonfinite counts are summarized
every 100 optimizer steps, with the exact training-step position in `gradient/runtime_optimizer_step`.
The tracking canonical step may identify the latest logging boundary, not the exact GPU-update
step or timestamp; align gradient diagnostics using the runtime field. This neither asserts
that a conflict has already been observed nor guarantees a direction gain.
The conservative extra FP32-buffer peak for three tasks is
`28 × LoRA numel` bytes, with an enforced `512 MiB` limit; exceeding it fails rather than ignoring
the budget. Batch autotuning executes the same backward path as training, including AMP and
gradient accumulation.

V2 retains the monitor key `direction/selection_loss`, but declares a different definition in
its own run configuration; do not rank it against original reference-weighted selection losses:

```text
selection_loss = 1 - mean_market(mean_task(mean_horizon(balanced_decision_BA)))
                 + max(pinball/first_full_validation_anchor - 1 - 0.01, 0)*1
```

Lower is better. `BA` averages recalls for classes with true support in each slice. Markets,
the two tasks and horizons each receive equal weight so US sample count does not dominate
selection; market/class support remains explicit. The regression anchor freezes only after
the first **complete full** validation, never from partial results or subsequent re-estimation.
Pinball degradation beyond 1% of that anchor adds a penalty with default weight 1. Checkpoints
persist the anchor and restore it on resume. Test never fits priors, references, anchors or
direction thresholds.
Each run has an independent first-validation anchor, so `selection_loss` selects checkpoints
within that run and must not rank seeds. Compare seeds on the identical holdout using raw
pinball and separately reported raw/balanced direction metrics; their distinct regression-guard
anchors are not a common scoring scale.
Raw population accuracy remains intact; prior-removed decision accuracy/BA
are explicitly under `balanced_decision`, not replacements for raw accuracy. Diagnostics add CE
skill against train priors, 20 confidence bins, joint-up probability thresholds/precision/counts and
daily confusion aggregates. Fixed-size bins bound probability summaries; daily aggregation allows
at most 4,096 entries, failing explicitly rather than dropping samples or growing without bound.

The following settings belong under `model.direction`, not new workflow CLI flags. Uploaded
v2 presets are selected with `--experiment`; numerical-contract changes still require a new run:

| Setting | V2 preset / allowed values | Purpose |
| --- | --- | --- |
| `architecture` | `market_residual_v2`; retained original `legacy` | Direction-head architecture; no cross-architecture resume |
| `loss_reference_samples` | `8192`; 32–8192 | Bounded train-only stratified reference sample cap; `data.calibration_seed=59` |
| `class_balance_power` | `1.0`; 0–1 | `τ`, train-prior strength in population logits; balanced decisions remove the same prior term |
| `market_residual_hidden_dim` | `64`; 8–256 | Hidden dimension of the direction numerical projection/history-dependent market residual |
| `prior_smoothing` | `1.0`; greater than 0, at most 100 | Pseudo-count for train market/task/horizon class priors |
| `market_weight_max` | `3.0`; 1–4 | Final effective market-weight cap; capped water filling also enforces train-weighted mean 1 |
| `gradient_management` | `pcgrad`; `none`, `monitor`, `pcgrad` | Shared-LoRA negative-conflict projection, diagnostics-only mode or ordinary backward |
| `gradient_diagnostics_interval` | `100`; 1–10000 | Optimizer-step diagnostic interval, not validation cadence |
| `probability_bins` | `20`; 5–128 | Bounded confidence/joint-up probability diagnostic bins |
| `max_daily_entries` | `4096`; 128–16384 | Hard entry cap for daily aggregation; no subsampling on overflow |
| `selection_pinball_guard_tolerance` | `0.01`; 0–0.1 | Allowed degradation relative to first full-validation pinball |
| `selection_pinball_guard_weight` | `1.0`; greater than 0, at most 10 | Checkpoint penalty multiplier beyond the tolerance |

`absolute_weight`/`alpha_weight` remain 1 (greater than 0, at most 10) and still allow task weighting.
`representation_weight=0.05` (0–1), `representation_radius=0.01` (greater than 0, at most 0.1)
and existing cost settings remain. V2 does not change CPU preparation, data cleaning, dynamic
training sampling or full validation/test. Dual-direction baselines are reusable only when
their complete data, cost, task and baseline numerical/source contracts match; this is not a
claim that every historical cache ID remains valid after implementation changes.

##### V3: ordered direction, state-guided selection and probability-aware checkpoints

`a-direction64-v3`/`b-direction64-v3` use `a_direction64_v3.yaml`/`b_direction64_v3.yaml`.
They retain Kronos-base/LoRA64, alpha quantiles, independent ranking, ten technical channels,
direct scales, train-only references, market weighting and PCGrad. Only direction reading and
selection change. Cleaning, baseline labels/costs, sources and splits stay unchanged; matching
baseline contracts remain reusable. V3 needs a new run, not a V2 checkpoint resumed as V3; same-V3
interrupted runs remain resumable.

Following [CORN's conditional ordinal structure](https://arxiv.org/abs/2111.08851),
`a=P(return>lower)` and `b=P(return>upper | return>lower)` produce
`P(down)=1-a`, `P(neutral)=a*(1-b)`, `P(up)=a*b`. Crossing the upper boundary cannot be more
probable than crossing the lower boundary. Stable log-sigmoid three-class NLL equals lower
boundary BCE on every sample plus upper conditional BCE **only on non-down samples**, averaged
over all samples, not separately averaged BCEs. The retained `log(3)` reference means uniform
three-class null probabilities, not two conditional probabilities of 0.5. Train priors become
conditional log-odds offsets with history-dependent residuals. Population probabilities are
`softmax(log_probability)`; balanced decisions separately use
`log_probability - τ*log(train_prior)`, not calibrated trading win probabilities. Ordinal
literature motivates the structure but does not guarantee 60% direction accuracy here.

Inspired by [MASTER](https://ojs.aaai.org/index.php/AAAI/article/view/27767), feature gates use
only existing benchmark std20/std60, context close CV, mean5/mean20, volume ratio20 and
drawdown60 at cutoff `t`, with train-fitted normalization. A small network generates
`3*softmax` gates for asset/benchmark/conditioned token streams and `2*sigmoid` gates for twenty
numerical features. No future bars, labels, fees or batch population enter gates; state is not
added directly to logits. There is no new indicator, cross-stock N×N attention, extreme-return
removal or DataLoader/worker/prefetch change. Native batch tensor kernels implement routing.

V3 retains `direction/selection_loss` with a separate definition:

```text
selection_loss = mean_market_task_horizon(
                    (CE / CE_train_prior + Brier / Brier_train_prior + 1 - population_macro_F1) / 3
                    + 0.25 * balanced_opposite_error + 0.25 * collapse_deficit)
                 + frozen_pinball_guard
```

CE/Brier references score frozen train market/task/horizon priors on full validation labels;
they do not fit validation priors. Markets/tasks/horizons receive equal weight; BA remains a
diagnostic rather than the sole selection objective.
The probability-quality rationale for CE/Brier follows [Gneiting and Raftery's scoring-rule theory](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
The complete F1/guard-augmented selection score is an engineering composite, not itself claimed
to be a strictly proper scoring rule. `balanced_opposite_error` averages true
down→up and true up→down conditional error rates over supported tails, using population argmax,
not the separately reported `balanced_decision` rule. For classes with true share at least 5%,
`collapse_deficit` averages `max(1-predicted_share/0.01,0)`. It is a configurable guard, not
sample removal or proof that predictions below 1% are necessarily wrong. Score/guard components,
true/predicted shares, collapsed classes, ordinal MAE, opposite errors, precision/recall,
CE skill, Brier skill and confidence bins are recorded; balanced decisions are also audited.
Initial weights are explicit engineering defaults, not literature-established optimal financial
parameters. Frozen pinball anchors, no test fitting and no cross-run ranking by selection loss
remain as in V2; coverage is not a direction-selection objective.

| Setting | V3 default / allowed values | Purpose |
| --- | --- | --- |
| `selection_class_support_min` | `0.05`; greater than 0, at most 0.5 | Minimum true class share for the collapse guard |
| `selection_prediction_share_min` | `0.01`; greater than 0, at most 0.1 | Reference predicted share for the linear deficit |
| `selection_collapse_weight` | `0.25`; 0–2 | Collapse penalty multiplier |
| `selection_opposite_weight` | `0.25`; 0–2 | Opposite-direction mistake penalty multiplier |

These four settings affect V3 only; use `architecture=market_ordinal_v3`. Other direction settings
retain the table above. Historical V3 reproduction uses the existing CLI options:

```bash
bash scripts/runpod_workflow.sh train --experiment a-direction64-v3 --seed 42 \
  --gpuId "NVIDIA GeForce RTX 5090" --maxRuntime 72h
```

Use `b-direction64-v3` for B and separate train commands for seeds 43/44. This creates a formal
training Pod, not local training or QA; existing code sync and readiness gates still apply.

##### Archived adaptive schedules and calibration

Historical `adaptive64` / `adaptive128` presets use `sample_plateau`: a cosine multiplier decays from 1 to 0.09 between
1,000,000 and 4,000,000 actual window presentations, independently of A/B epoch lengths.
The lower of this multiplier and the existing plateau multiplier is applied. The 256,000-sample
warmup, sample counter, plateau state and two minimum-LR evaluation safeguard survive resume.
Configure `training.sample_decay_start`/`sample_decay_end` with `warmup_samples ≤ start < end`.
Task/LoRA LR are `1e-5`/`5e-6`; resampler, conditioner, head and LoRA dropout are `0.10`,
and AdamW weight decay is `0.03`. The epoch budget is not extended.

Those adaptive presets' `training.market_loss_weights: [1, 2, 2, 1]` orders US, TWSE, TPEx, unknown and
weights training pinball only. Each weight must be in `(0, 4]`; it is not renormalized within
a single-market microbatch, which would cancel the weighting. Ranking retains weight `0.05`.
Full validation/test remain unweighted and additionally expose equal-market `market_macro`
diagnostics. Neither the checkpoint monitor nor baseline objectives change.

For quantile presets with calibration enabled, full validation fits interval calibration after
checkpoint selection without changing q50 or ranking scores. Independent tasks do not run
this step. `validation.interval_calibration.mode: static` remains the quantile default;
adaptive quantile presets use `regime_adaptive` and keep results distinct:

| Result field | Protocol |
| --- | --- |
| Existing raw metrics | Uncalibrated output; adaptive controls select checkpoints/early stopping from these, never calibrated metrics |
| `calibrated_static` | Original validation-fitted market/horizon reference |
| `calibrated` | Frozen market × horizon × historical-volatility-regime calibration; fits validation only |
| `calibrated_online` | Separate daily prequential diagnostic using only matured past prediction errors |

Regimes use historical relative-return `std20/std60`, with validation tertile thresholds.
Each observed market date initially has equal weight, followed by exponential recency weighting
with `recency_half_life_sessions: 63`. Groups below `minimum_group_samples: 128` or
`minimum_group_dates: 20` fall back to market/pooled factors. Other groups shrink toward their
parent with weight `date_count / (date_count + shrinkage_dates)`, with `shrinkage_dates: 60`.
All these controls live under `validation.interval_calibration`.

Online diagnostics use `online_learning_rate: 0.05` to update log-width multipliers from
daily matured lower/upper violation rates, targeting 10% per tail, with update strength
`daily_sample_count / (daily_sample_count + 128)`. `online_max_multiplier: 2.0` bounds widths
to 0.5–2 times their frozen values. A label's scheduled exit is the existing offline market
calendar's `t+h` session and must be **strictly earlier** than the forecast date before the
label can be read. Calendar-day arithmetic and skipping to an older available stock bar are
not substitutes. Online feedback never rewrites frozen calibration, updates the model, or
selects checkpoints. Do not present prequential results as frozen holdout results. Calibrated
metrics include horizon, market, month and market × month coverage, width and pinball.
These are empirical procedures, not guaranteed future or subgroup 80% coverage.

##### Retained quantile / dual-cost-direction baselines

The following applies only to historical quantile/direction64 paths. GRU, DLinear and PatchTST
retain alpha quantiles and train absolute-stock/alpha three-class heads on shared representations.
GBDT retains quantile regressors and adds two classifiers per horizon. Labels, execution timing
and cost bands match historical direction64. When dual direction is enabled, fixed-reference
normalized pinball plus both CEs supplies selection; quantile-only baselines retain their quantile
objective. Fixed-rule residual quantiles use all train rows.

`configs/baseline.json` independently owns the legacy `direction` section: `enabled` defaults to `true`,
`loss_reference_samples` to 1,024 (32–8,192), and `absolute_weight`/`alpha_weight` to 1.0 each
(greater than zero, at most 10). `costs` uses the same market-specific fields as the main model,
with 1,000 Taiwan shares and 50 US shares and only cutoff-known prices. Neural models measure
fixed references before updates; GBDT uses smoothed class priors and intercept-only quantiles
on a fixed training subset. Actual fitting, validation and testing are not subsampled.
References and weights persist through resume. `metrics.jsonl` records training/validation losses:
neural models log actual training batches/interval means, while GBDT logs a fixed reference-sized
training diagnostic with its sample count, without reducing fitting or full evaluation populations.
Reference-weighted `selection_loss` is not comparable across architectures; compare task metrics.

Full evaluation adds `direction` metrics for both tasks: accuracy, balanced accuracy, macro-F1,
confusion, majority accuracy, up precision/recall, signal fraction and joint-up precision, by
horizon/market/month. Probabilistic models also report CE. Rules apply cost bands to their past-only
absolute/relative signals; `always_buy` always predicts up and `zero_return` neutral. Hard rules do
not pretend to provide probabilities or CE. Existing q50 sign agreement remains separate from
cost-sensitive three-class accuracy. Main testing assembles cached metrics in `comparison.direction`
and rejects mismatched costs, horizons or membership. Local main-model admission also verifies cost
alignment; this does not bind baseline identity to Kronos capacity. These legacy cost fields,
dual outputs and `comparison.direction` do not describe the current independent tasks.

For historical dual-direction baselines, labels `inputs/<split>/absolute_targets.npy` and `direction_bands.npy` also have one
copy per split. Each model retains only its `direction-probabilities.npy` (`N×2×H×3`) or rule
`direction-classes.npy` (`N×2×H`), ordered absolute/alpha and down/neutral/up.
An alpha-quantile-only baseline is not a completed dual-direction baseline, and the `N×2×H`
dual outputs are not current single-task `absolute_direction` outputs. Reuse always checks
the current numerical contract.

<a id="legacy-artifacts-en"></a>

#### Archived loss records, metrics and inference fields

The following pinball/ranking fields apply only to legacy quantile/multitask experiments.
`train/pinball_loss` remains unweighted; compare it with
`validation/aggregate/normalized_pinball` for the pinball generalization gap.
Direction experiments also expose `validation/direction/pinball_loss`; do not confuse
either pinball field with their checkpoint-selection `validation/loss`.
`train/weighted_pinball_loss` records the market-weighted training component.

The following describes legacy quantile/multitask logs: `train/loss` is
market-weighted normalized pinball plus weighted ranking (all market
weights are one for original presets). Unweighted `train/pinball_loss`,
`train/weighted_pinball_loss`, `train/ranking_loss` and per-group learning rates are logged separately.
This total-loss definition applies to presets without direction heads. Direction presets also
include recorded reference-weighted absolute/alpha CE and representation regularization.
V2 additionally persists train priors, fixed market weights, shared-gradient diagnostics,
population/balanced-decision metrics and the first full-validation pinball anchor. Different
versions' `direction/selection_loss` values are not comparable rankings.
`validation/loss` records the run's `checkpoint_monitor`, not one common loss across experiments:
independent tasks use their normalized Smooth L1 or CE; legacy quantiles without the dual-direction
module use mean normalized pinball over all full-validation
horizons, without ranking; original `direction64` uses pinball plus both CE terms weighted
by frozen references; `direction64-v2` uses equal-market/task/horizon balanced-decision BA
selection loss plus the regression penalty against its frozen first full-validation pinball anchor.
`direction64-v3` uses population CE/Brier ratios versus train priors, macro-F1, opposite-error/collapse
penalties and the same pinball guard. All components are recorded; BA alone no longer selects models.
Compare `train/pinball_loss` with `validation/aggregate/normalized_pinball` when diagnosing
the pinball generalization gap. Direction experiments also record
`validation/direction/pinball_loss`; neither training total loss nor direction selection loss
is validation pinball. Assess direction generalization separately using each absolute/alpha
task's full-validation CE, accuracy, balanced accuracy and market-specific metrics.

The following fields apply only to legacy quantile
tasks: independent `ranking_scores`, and `calibrated_forecast` when matching validation-fitted
checkpoint calibration exists; otherwise the latter is `null`, with `interval_calibration_status`.
Single-observation inference for adaptive quantile presets uses frozen validation-regime calibration with status
`validation_fitted_frozen_regime`; it never implicitly imports holdout online feedback state.

Archived `quantiles` metrics include pinball, coverage/width, q50 error/sign/correlation,
ranking and any enabled dual-direction diagnostics. These are not independent `alpha_point`
outputs or fixed ±1% absolute-direction labels. `comparison.direction` compares the two old
cost-based classifiers; the current independent tasks instead use `comparison.rows`.

`alpha_quantiles` has shape `[B,15-h_start,3]` with q10 ≤ q50 ≤ q90; `ranking_scores` is separate
from q50. V1/V2/V3 additionally emit two direction tasks, preserving their original population
and balanced-decision semantics. None generates natural-language output.


<a id="single-model-en"></a>

#### Advanced individual-model operations (not obsolete models)

Current pairs internally retain `a-alpha-point64`, `a-absolute-direction64`,
`b-alpha-point64` and `b-absolute-direction64`. These entries are not historical V1/V2/V3
and have not been removed. Use them only for an explicitly single-task ablation, diagnosis
or existing-run reproduction; normal training uses `a-signal64`/`b-signal64` once.

```bash
bash scripts/runpod_workflow.sh train --experiment a-alpha-point64 --seed 42 --maxRuntime 2h --gpuId "NVIDIA GeForce RTX 5090"
```

Use `a-absolute-direction64` for the individual classifier and `b-` for B.
`baseline --experiment NAME` also retains these individual tasks; unlike paired operations,
this older entry updates active selection. `readiness --baseline` without an experiment
still checks that active task, and `runs --experiment NAME` can explicitly filter child models.

`probe-scales`, individual W&B sync and low-level inference require the child `MODEL_RUN_ID`
shown by `status <PARENT_RUN_ID>`, not the parent ID. The two weights are not merged into one
checkpoint. For a dedicated diagnostic Pod, the single-model creation command above remains
available; do not start training tmux, and follow the [scale-probe workflow](#probes-en) instead.
Existing probe completion and local-guard behavior remain unchanged.
