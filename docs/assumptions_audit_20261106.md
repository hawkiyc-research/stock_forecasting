# 模型 × 策略組合的條件、假設與缺陷檢查 / Conditions, Assumptions and Defect Audit of Every Model × Strategy Cell

版本：2026-11-06。對象：§13.23 的 7 個模型 × 3 種策略 = 21 格（不論是否入選或進入確認期），加上並列規則（處置 D、停券強制回補 C）、成交價修正（F）、策略變體（15% 追蹤停損 L）與方式 B。資料來源：`backtest/config/scenario_suite_v1_p3b.json` 及其衍生設定、`cost_profiles*.json`、`strategy_decision_p3b.json`、`configs/labels/*.json`、`configs/walk_forward/tw_quarterly_v2.json`、`configs/features/*.json`、`configs/data_cleaning.json`、`src/stock_forecasting/data/sample_universe.py`、`backtest/twbacktest/*.py`，以及 2026-11-06 對 v1.1 資料集的直接計算。

Version: 2026-11-06. Scope: the 7 models × 3 strategies = 21 cells of §13.23 (whether or not shortlisted or in the confirmation period), plus the parallel rules (disposition D, forced cover C), the fill correction (F), the strategy variant (15% trailing stop L) and policy B. Sources: `backtest/config/scenario_suite_v1_p3b.json` and its derived configurations, `cost_profiles*.json`, `strategy_decision_p3b.json`, `configs/labels/*.json`, `configs/walk_forward/tw_quarterly_v2.json`, `configs/features/*.json`, `configs/data_cleaning.json`, `src/stock_forecasting/data/sample_universe.py`, `backtest/twbacktest/*.py`, and direct computations on the v1.1 dataset on 2026-11-06.

---

## 繁體中文

### 1. 共同條件與假設（所有 21 格）

#### 1.1 資料與宇宙

| 項目 | 條件 |
| --- | --- |
| 資料集 | v1.1（`fd8a60f5…`）：上市（TWSE）與上櫃（TPEX）全部股票的日線開高低收量，2016-01-01 至 2026-10-23；還原價只套除權息因子（有償現金增資採交易所的開盤競價基準），不含分割、減資、面額變更 |
| 大盤 | TWSE 用 `TAIEX.TW`（發行量加權股價報酬指數，含息）、TPEX 用 `TPEX.TWO`（櫃買報酬指數）；標籤與回測的「相對大盤」都用這兩條 |
| 交易日曆 | `exchange_calendars` XTAI 4.13.2，加上 2022-02-04、2023-01-18、2024-10-31、2026-07-10、2026-10-23 五個休市覆寫 |
| 資料清理（樣本宇宙） | 每個訊號日 t 需要 (a) 往前 128 個交易日無缺席、成交量 > 0；(b) **往後 14 個交易日無缺席、成交量 > 0**；(c) 60 日成交金額中位數 ≥ 新台幣 1,000 萬；(d) 大盤同期有資料。不符的列不進訓練、不出預測 |
| 回測宇宙（每日） | 預測列再篩：20 日成交金額中位數 ≥ 3,000 萬（上市）／1,000 萬（上櫃）、訊號日收盤 ≥ 10 元、排除 ETF |
| 市場合併 | 決策市場 TW = 上市書與上櫃書各 50%（兩個市場各自跑完整的組合，日報酬取平均） |

#### 1.2 標籤（所有模型共用的目標）

| 項目 | 條件 |
| --- | --- |
| 原始目標 | 相對大盤的還原對數報酬：t+1 開盤 → t+h 收盤，h = 1 至 14 |
| 標準標籤（B0） | 每個（訊號日, 市場）群組內把目標做高斯排名（cross-sectional gauss-rank） |
| 殘差標籤 a（F0、F2、D0、D1 × 2） | 群組內先對截距、log(20 日成交金額中位數)、產業虛擬變數（2026-10-22 公司參考快照；缺的歸 UNK，群組 < 30 檔不用產業）迴歸，殘差再高斯排名 |
| 殘差標籤 b（變體 B） | 同 a，規模改用 log(收盤 × 發行股數快照)；股數是單一快照，有股票股利或減資的公司有偏 |

#### 1.3 走動式訓練

| 項目 | 條件 |
| --- | --- |
| 排程 | `tw_quarterly_v2`：擴張視窗，訓練自 2016-01-01 起，每季一折，測試季 2023Q1 至 2026Q4（16 折），驗證期 12 個月、禁區 15 個交易日 |
| 預測 | 每折只對該季的訊號日出分數；P3 的最後 14 個交易日（10-01 之後）沒有標籤就沒有預測 |
| 早停 | LightGBM 最多 1,000 輪、驗證 Rank IC 50 輪無進步即停、取最佳輪；transformer 最多 20 個 epoch、驗證 Rank IC 4 個 epoch 無進步即停、取最佳 epoch |
| 集成 | 每折 seed 42／43／44 三個模型平均 |
| 模型保存 | P3 折（2026Q3、Q4）的模型與登錄表留存（補充二），其餘折只保存預測 |

#### 1.4 分數與組合建構

| 項目 | 條件 |
| --- | --- |
| 分數 | 多日策略用 h = 10 的分數，週頻用 h = 5；先在（日, 市場）內轉百分位，再按股票做半衰期 3 日的 EMA（缺口 > 5 日重置） |
| V1 正交化 | 每日在宇宙內把分數對 20 日報酬、60 日報酬、60 日 beta 迴歸取殘差，多空兩腿都用 |
| 多日 band（S1、S2） | K = 50 槽、緩衝 m = 3（排名掉出前 150 名才賣）、每日新進上限 ⌈K／10⌉ = 5 檔、槽位等權（進場時等額、之後隨價格漂移不再平衡）；多頭用正向排名，空頭用反向排名跑同一規則 |
| 多空書的資金口徑 | 淨報酬 = 多頭書日報酬 + 空頭書日報酬，兩本書各自 100% 投入（資本情境下兩腿各用全部資金）；即「3 億」= 多頭 3 億 + 空頭 3 億，毛曝險 2 倍，放空保證金與擔保品不計 |
| 成交與漲跌停 | 訊號日次一交易日的官方開盤價成交；開盤相對前收 ≥ +9.5% 不買、不回補，≤ −9.5% 不賣、不放空，延到下一個可成交的開盤；無下一筆資料（停牌、下市）以 t+2 收盤強制出場 |
| 台指期避險（S2） | 對書的毛報酬對該市場大盤的 60 日滾動 beta（至少 30 個觀測值），每日以 −beta × 資金的 TX 名目對沖；依結算價逐日結算、換月成本兩邊計；保證金占用不計 |
| 週頻多頭（S-A） | 每 ISO 週最後一個訊號日審視，K 與 m 在上市 P1 以 {30, 50} × {2, 3} 格子選出（各模型各選）、換手上限每次 K 的 50%，停牌／下市的強制出場與初次建倉不受上限 |
| 參與率上限（資本情境） | 每檔部位 ≤ 20 日成交金額中位數的 5%，超過的部分留現金；整股與零股規則依成本檔 |

#### 1.5 成本與執行

| 項目 | 機構 | 散戶 |
| --- | --- | --- |
| 手續費 | 0.03% | 0.0855% |
| 證交稅 | 0.3%（賣出；放空的賣出也付） | 同左 |
| 固定滑價（分數位置列） | 每邊 0.05% | 上市 0.1%、上櫃 0.2% |
| 固定滑價（資本情境列，成交價修正後） | 0（由自身衝擊取代，補充四修正二） | 0 |
| 放空 | 借券名單 + 次日限額，借不到機率大／中／小 2%／10%／25%，借券費依規模 1.5%／3%／6% 年率 | 融券名單 + 融券餘額空間，借不到 5%／20%／50%，融券手續費 0.08%，無借券費 |
| 隨機層 | 借不到抽樣與召回各 5 個 seed 重複平均；主要召回機率 0，敏感度 0.5%／日 | 同左 |
| 台指期 | 交易稅 0.002%、手續費每口 30 元、滑價 1 點 | 每口 50 元、1 點 |
| 資本情境 | 分數位置（不限資金）、3 億、10 億（補充五加 30 億，附記 3,000 萬、1 億） | 分數位置、1,000 萬（補充四加） |

#### 1.6 期間與判定

| 項目 | 條件 |
| --- | --- |
| 期間 | P1 = 2023-01-02 至 2025-07-01、P2 = 2025-07-01 至 2026-07-01、P1 ∪ P2 = 樣本內、P3 = 2026-07-01 至 2026-10-23（預測到 10-01，64 個區間） |
| 區間估計 | 21 日區塊 bootstrap、1,000 次、seed 11 |
| 門檻篩選 | 機構、TW、分數位置、V1：P1、P2、P1 ∪ P2 淨 Sharpe 各 ≥ 1.5，且 P1 ∪ P2 年化淨報酬 95% 區間下界 > 0 |
| P3 驗證 H-M | 入選格各自：P3 年化淨報酬 > 0，且 P3 最大回撤不深於該格 P1 ∪ P2 回撤的兩倍 |
| 確認期 P4' | 2026-10-23 起 12 個月，六格各自判定；H-L 回撤門檻 = 各格 P1 ∪ P2 回撤的兩倍；期中 2027-04-22 只報不判；凍結設定不得修改 |

### 2. 各模型的條件

| 模型 | 類型 | 輸入 | 標籤 | 主要超參數 | 正本 launch |
| --- | --- | --- | --- | --- | --- |
| B0 | LightGBM，每個視野一個 booster | `alpha_tw_v1`：Alpha158 式價量特徵（K 棒、價格與成交量滯後、滾動運算子、台灣相對報酬與成交金額排名、市場指標），只用 t 日以前的資料 | 標準高斯排名 | 回歸目標、num_leaves 31、learning rate 0.05、feature／bagging fraction 0.8、min_data_in_leaf 200、早停 50 | `launch-20261101T212237Z-2508230824` |
| F0（變體 A） | 同上 | `alpha_tw_v1` | 殘差 a | 同上 | `launch-20261101T220120Z-29216090` |
| B（變體 B） | 同上 | `alpha_tw_v1` | 殘差 b | 同上 | `launch-20261101T221954Z-1215615947` |
| F2 | 同上 | `alpha_chip_tw_v1`：`alpha_tw_v1` 全部欄位 + 籌碼 67 欄（三大法人流量與持股、融資融券、借券、當沖、外資持股、注意／處置事件、月營收；公布時刻採「次一開盤前」的保守規則，當沖為 T+2） | 殘差 a | 同上 | `launch-20261101T223737Z-1940816181` |
| D0 | 跨股票 transformer（從頭訓練）：個股 OHLCV 視窗的 patch 編碼器 + 特徵側輸入 → 同日同市場的股間注意力 → 市場狀態閘控 | OHLCV 視窗 + `alpha_tw_v1` 特徵（群組內 z 分數） | 殘差 a | learning rate 5e-4、weight decay 1e-4、最多 20 epoch、早停 4、損失 = 排名損失 + 相關項；bf16、同日同市場為一批 | `launch-20261102T063519Z-189413204` |
| D1-Kronos | 同 D0，但個股編碼器換成凍結的 Kronos-small（快取嵌入） | 同上 | 殘差 a | 同上 | `launch-20261102T071451Z-322624982` |
| D1-Chronos | 同 D0，編碼器換成凍結的 Chronos-2-small | 同上 | 殘差 a | 同上 | `launch-20261102T073405Z-1322713845` |

### 3. 各策略的條件

| 策略 | 建構 | 多日參數 | 持有與換手（F2，分數位置，機構，全段） |
| --- | --- | --- | --- |
| S1 多日多空 | 多頭 band + 空頭 band，各 100%；V1 正交化分數 | h = 10、K = 50、m = 3、每日新進 ≤ 5（七個模型一律用 F0 的參數，不各自在 P1 選） | 平均持有上市 31.6 日／上櫃 40.4 日，日均換手 2.8%，空頭腿名單外比例上市 57%／上櫃 73% |
| S2 多日多空 + 台指期 | S1 + 每日 −beta × 資金的 TX 對沖 | 同 S1；beta 視窗 60 | beta 平均 −0.15（S1 本身偏空），TX 名目約資金的 16%，TX 費用約 0.06%／年 |
| S-A 週頻多頭 | 週頻 band，純多頭，h = 5，V1 正交化 | K、m 各模型在上市 P1 選（結果多為 K = 50、m = 2 或 3）、換手上限 50% | 平均持有 22 到 28 日，每週換手約 16% 到 20% |

### 4. 21 格的狀態

機構、TW、分數位置、主要假設（開盤價成交；D、C 並列）；年化淨報酬／Sharpe。

| 模型 | S1 多日多空 | S2 多空 + 台指期 | S-A 週頻多頭 |
| --- | --- | --- | --- |
| B0 | +10.2%／0.80；未入選 | +16.8%／1.24；未入選 | +39.1%／2.05；未入選（P1 1.48） |
| F0 | +15.9%／1.25；未入選 | +22.3%／1.64；**入選，H-M 通過，確認期**；D + C 1.41 未過 | +40.5%／2.17；未入選（P1 1.45） |
| B | +12.6%／1.02；未入選 | +19.1%／1.45；未入選 | +36.4%／1.95；未入選 |
| F2 | +25.5%／2.11；**入選，H-M 通過，確認期**；D + C 1.32 未過 | +30.2%／2.38；**入選，H-M 通過，確認期**；D + C 1.70 過 | +53.4%／2.33；入選，H-M 不通過（P3 −13.4%） |
| D0 | +19.7%／1.52；未入選 | +24.5%／1.80；**入選，H-M 通過，確認期**；D + C 1.51 過 | +40.0%／2.06；未入選（P1 1.45） |
| D1-Kronos | +16.2%／1.28；未入選 | +20.4%／1.53；未入選（P2 1.49） | +44.2%／2.16；**入選，H-M 通過，確認期**；D + C 2.06 過 |
| D1-Chronos | +19.0%／1.48；未入選 | +24.4%／1.81；**入選，H-M 通過，確認期**；D + C 過門檻但 P3 −2.1% 未過 | +40.4%／2.04；未入選（P1 1.48） |

### 5. 並列規則、修正與變體的條件

| 項目 | 性質 | 條件 | 讀數（F2 × S2，機構，分數位置，全段） |
| --- | --- | --- | --- |
| D 處置期間不新進場 | 並列讀數（11-04、11-06 裁定） | 訊號日 t 的公告已出（announce ≤ t）且 t+1 開盤落在處置期間 → 兩腿都不新進場；已持有不強制出場、不加分盤撮合滑價 | 2.38 → 1.92 |
| C 停券強制回補 | 並列讀數 | 已持有空頭的訊號日列 `short_suspended` 為真 → t+1 開盤回補（跌停延後），且停券期間不新建空單；散戶 = 融券規則（比真實最後回補日晚至少一天），機構 = 上界 | 2.38 → 2.12；D + C 1.70 |
| F 成交價修正（方式 A） | 修正，取代資本情境數字 | 市價單進開盤集合競價：成交價 = 開盤價 × (1 ± Y σ_日 √(Q ÷ (f V)))，Y 0.7、f 10%（括弧 0.5／1.0、5%／20%，未校準）；分數位置列 Q → 0 不變；資本情境列不再扣固定滑價 | 散戶 1,000 萬 +21.5%／1.72；機構 3 億 −16.6%、10 億 −38.9% |
| L 15% 追蹤停損 | 策略變體（無停損 vs 停損） | 部位價值自進場後最高點回落 15% 當日出場；成交價 = S × (1 − 半檔位 − 衝擊 − c (S − 當日最低) ÷ S)，c 0.25；跳空在開盤價、跌停延後；多空鏡像 | 2.38 → 1.30；散戶 1,000 萬 F + L +5.5%／0.43 |
| 方式 B 盤中分批 | 容量評估（對照修正後的方式 A） | 成交價 = 開盤價 × exp(θ r_oc) × (1 ± 半檔位 ± Y σ_日 √(Q ÷ V))，θ 0.5、Y 0.7；不含停損 | 分數位置 2.01；散戶 1,000 萬 1.94；機構 3,000 萬 1.59、1 億 1.13、3 億 0.80 |

### 6. 缺陷檢查

嚴重度：**重大** = 會改變結論或違反已宣告的性質；**中** = 讓數字有系統性偏差但方向與量級可估；**輕** = 揭露即可。

#### 6.1 重大

**(1) 樣本宇宙的「往後 14 日」條件是前瞻排除。** `sample_universe.py` 要求訊號日之後 14 個交易日無缺席、成交量 > 0，不符的訊號日整列剔除——這用到了訊號日之後才知道的事（停牌、下市、零成交），而且不只影響標籤，連預測列、回測候選名單都少了這些日子。2026-11-06 直接在 v1.1 資料上計算：2023 年起 4.36% 的股票日因此被排除，被排除列的後 10 個交易日開盤到開盤平均報酬 −0.47%，未被排除的 +0.30%（中位數 −0.55% 對 −0.16%）。影響：多頭腿少看到一批之後表現差的名字（偏樂觀），空頭腿少了可以放空的名字（偏悲觀）；量級上 4% 的股票日、0.8 個百分點的差距，對全段年化的影響估計在 1 個百分點以內，但報告第 7.1 節「資料洩漏：無」的宣告不成立，必須改寫。修法：預測與回測候選用「只看過去 128 日」的宇宙，標籤才要求未來 14 日；這會改變資料集身分，屬新一輪的資料準備，須登記。

**(2) 多空書的槓桿口徑沒有宣告。** 淨報酬 = 多頭書 + 空頭書，兩本書各自 100% 投入；資本情境下兩腿各用全部資金，台指期對沖再按資金名目計。等於每 1 元資金有 2 元毛曝險，而且放空所需的保證金（散戶融券 90%、機構借券擔保品約 140%）與現金拖累都沒有計入。Sharpe 不受比例影響，但年化淨報酬、最大回撤與「3 億」的意義都是 2 倍曝險下的數字；若改以 1 元資金 = 1 元毛曝險（兩腿各 50%）報告，報酬與回撤大約減半，容量上限則約乘 2。這不是程式錯誤，是口徑，但報告沒有說明，必須在摘要與限制裡明寫，並補一列「單位毛曝險」的讀數。

**(3) 資本情境在兩個市場各算一次。** 每個市場的書都以全部資金（例如 3 億）模擬，TW 列再把兩市場各取 50% 平均；所以 TW 列的「3 億」實際上是上市 3 億 + 上櫃 3 億（再乘上 (2) 的兩倍曝險），而衝擊 ε 是按每檔 600 萬算的。若「3 億」指總資金，每市場應只投 1.5 億，每檔 300 萬，ε 按 √(Q) 縮為 0.71 倍，容量結論會往上修（約 1.4 倍）。需要決定口徑並重跑資本情境。

#### 6.2 中

**(4) 衝擊模型的三個未校準參數決定容量結論。** Y（0.5 到 1.0）、f（開盤量占比 5% 到 20%）與 5% 參與率上限都沒有台股實證；Y 從 0.7 改 0.5，機構 3 億的方式 B Sharpe 由 0.80 變 1.15。容量「約 3,000 萬」的結論對這三個參數都敏感，應以區間而非單點表述；取得逐筆或分鐘資料校準 f 是解法。

**(5) 台指期對沖的 beta 基準與對沖工具不一致。** beta 對該市場大盤（上櫃書對櫃買報酬指數）估計，但對沖工具是台指期；上櫃書的 TX 對沖會留下上櫃對大盤的殘差曝險。上市書沒有這個問題。修法：上櫃書的 beta 直接對 TX 估，或改用更貼近的工具（沒有櫃買期貨的流動性時，承認殘差）。

**(6) 空頭腿本身長期虧損、名單外比例極高。** 空頭腿單獨年化 −21.6%（它的價值在於壓低 beta 與回撤），上市 57%、上櫃 73% 的空頭候選不在可借名單，實際放空的是排名更後面的名字。沒有比較「多頭 band + 台指期對沖、不放空個股」這個更便宜的替代（省掉借券費、借不到、停券與處置的全部問題）。這是設計缺口，不是錯誤，建議登記為對照策略。

**(7) 兩條並列規則都偏樂觀。** D 不強制出場、不加分盤撮合滑價；C 比真實最後回補日晚至少一天、機構用上界；停損的跌停判斷只看開盤。這些已在協議與報告聲明，但確認期每季讀數應同時列 D + C 的數字（已安排）。

**(8) 產業與股數用單一快照。** 殘差標籤的產業虛擬變數與變體 B 的股數來自 2026-10-22 的快照，歷史上下市或改產業的公司被現在的分類描述；對 2023 年的標籤有輕微的前瞻成分。已揭露，量化未做。

#### 6.3 輕

**(9) 等權槽位不再平衡**：進場等額、之後隨價格漂移；對 K = 50 影響小，但長期持有的贏家權重會偏大。**(10) 漲跌停只在開盤判斷**：盤中鎖死不影響既有規則（開盤到開盤），但停損變體的盤中成交在跌停鎖死日被高估。**(11) 股利稅與借券利息**：還原價已含息，但股利所得稅（機構 21% 扣繳、散戶併入所得）與融券保證金的利息都未計。**(12) 標籤到收盤、交易到開盤**：標籤是 t+1 開盤到 t+h 收盤，策略持有的是開盤到開盤，視野也不等於實際持有（36 日對 10 日）；這是刻意的（band 規則），但分數的邊際價值在持有後段衰減，已有 EMA 平滑處理。**(13) 週頻 K、m 在上市 P1 選**：在樣本內選參數，已揭露。**(14) 深度模型不可逐次重現**（bf16、不同 GPU）、三 seed 集成已緩解。**(15) 門檻與候選集是看過第一版樣本外之後定的**，已揭露；P3 只有 64 日。

### 7. 建議的處理順序

1. 立即改寫兩份報告的「資料洩漏：無」與限制段落，加入 (1)、(2)、(3) 的揭露與量化。
2. 登記並決定口徑：(2) 多空書以「單位毛曝險」或「單位資金」報告；(3) 資本情境的資金是總額還是每市場。決定後重跑資本情境（約 20 分鐘的本機作業）。
3. 登記資料準備的修正 (1)：預測列與回測候選只用過去視窗；這是新資料集身分，排在確認期第一次延伸時一起做，避免再開一輪 Pod。
4. 登記對照策略 (6)：多頭 band + 台指期對沖、不放空個股。
5. (4)、(5) 作為下一輪的方法改進。

---

## English

### 1. Conditions and assumptions shared by all 21 cells

#### 1.1 Data and universe

| Item | Condition |
| --- | --- |
| Dataset | v1.1 (`fd8a60f5…`): daily OHLCV of every TWSE and TPEX stock, 2016-01-01 to 2026-10-23; adjusted prices carry ex-dividend / ex-rights factors only (paid rights issues on the exchange's opening-reference basis), no splits, capital reductions or par-value changes |
| Benchmarks | TWSE `TAIEX.TW` (the total-return weighted index, dividends included), TPEX `TPEX.TWO` (the TPEX total-return index); both the labels' "benchmark-relative" and the backtest's index columns use them |
| Calendar | `exchange_calendars` XTAI 4.13.2 plus five closure overrides: 2022-02-04, 2023-01-18, 2024-10-31, 2026-07-10, 2026-10-23 |
| Data cleaning (sample universe) | A signal date t needs (a) 128 sessions back without a missing session or zero volume; (b) **14 sessions forward without a missing session or zero volume**; (c) a 60-session median traded value ≥ NT$10M; (d) benchmark data over the same span. Rows that fail enter neither training nor the predictions |
| Backtest universe (daily) | Prediction rows further filtered: 20-session median traded value ≥ NT$30M (TWSE) / NT$10M (TPEX), signal-date close ≥ NT$10, no ETFs |
| Market combination | The decision market TW = 50% TWSE book + 50% TPEX book (each market runs the full construction, daily returns averaged) |

#### 1.2 Labels (the target every model shares)

| Item | Condition |
| --- | --- |
| Raw target | Benchmark-relative adjusted log return from the open of t+1 to the close of t+h, h = 1 to 14 |
| Standard label (B0) | Gauss-rank of the target within each (signal date, market) group |
| Residual label a (F0, F2, D0, both D1) | Within the group the target is first regressed on an intercept, log(20-session median traded value) and industry dummies (company reference snapshot of 2026-10-22; missing → UNK; no industry term for groups under 30 names), and the residual is gauss-ranked |
| Residual label b (variant B) | As a, with size = log(close × issued shares from the snapshot); a single snapshot of share counts biases companies with past stock dividends or capital reductions |

#### 1.3 Walk-forward training

| Item | Condition |
| --- | --- |
| Schedule | `tw_quarterly_v2`: expanding window from 2016-01-01, one fold per quarter, test quarters 2023Q1 to 2026Q4 (16 folds), 12-month validation, 15-session embargo |
| Predictions | Each fold scores its own quarter's signal dates only; the last 14 sessions of P3 (after 10-01) have no label and therefore no prediction |
| Early stopping | LightGBM up to 1,000 rounds, stop after 50 rounds without validation Rank IC improvement, best round kept; transformers up to 20 epochs, patience 4 on validation Rank IC, best epoch kept |
| Ensemble | Seeds 42 / 43 / 44 averaged per fold |
| Saved models | The P3 folds (2026Q3, Q4) keep their models and a registry (supplement 2); earlier folds keep predictions only |

#### 1.4 Scores and portfolio construction

| Item | Condition |
| --- | --- |
| Score | h = 10 for the multi-day strategies, h = 5 for the weekly; percentile within (date, market), then a per-stock EMA with a 3-session half-life (reset after a gap over 5 sessions) |
| V1 orthogonalization | Daily, within the universe, the score is regressed on the 20-session return, the 60-session return and the 60-session beta and the residual is used; both legs |
| Multi-day band (S1, S2) | K = 50 slots, buffer m = 3 (sell only when the rank falls out of the top 150), at most ⌈K/10⌉ = 5 new entries a day, equal-weight slots (equal at entry, then drifting, no rebalancing); the long book ranks ascending, the short book descending under the same rule |
| Capital basis of the long-short book | Net return = long-book daily return + short-book daily return, each book 100% invested (in the capital scenarios each leg uses the whole capital); "NT$300M" therefore means NT$300M long plus NT$300M short, gross exposure 2×, with no margin or collateral for the shorts |
| Fills and price limits | The official open of the session after the signal date; an open ≥ +9.5% over the previous close blocks buys and covers, ≤ −9.5% blocks sells and short sales, retried at the next fillable open; a row without a next bar (halt, delisting) exits at the close of t+2 |
| TX hedge (S2) | A 60-session rolling beta (at least 30 observations) of the book's gross return against that market's index, hedged daily with a TX notional of −beta × capital; marked daily at settlement, roll costs on both sides; margin not modelled |
| Weekly long (S-A) | Reviewed on each ISO week's last signal date; K and m chosen per model on TWSE P1 from {30, 50} × {2, 3}; replacements capped at 50% of K per rebalance; forced exits (halts, delistings) and the initial build are outside the cap |
| Participation cap (capital scenarios) | Each position ≤ 5% of the 20-session median traded value, the rest stays in cash; board-lot and odd-lot rules per cost profile |

#### 1.5 Costs and execution

| Item | Institution | Retail |
| --- | --- | --- |
| Commission | 0.03% | 0.0855% |
| Transaction tax | 0.3% on sales (the short sale pays it too) | same |
| Flat slippage (fractional rows) | 0.05% per side | 0.1% TWSE, 0.2% TPEX |
| Flat slippage (capital rows after the fill correction) | 0 (replaced by own impact, supplement 4 correction 2) | 0 |
| Shorting | SBL list plus next-day quota, borrow failure 2% / 10% / 25% by size tercile, borrow fee 1.5% / 3% / 6% a year by tercile | margin list plus margin-short room, failure 5% / 20% / 50%, short fee 0.08%, no borrow fee |
| Random layer | Failure draws and recalls averaged over 5 seeded replicates; primary recall 0, sensitivity 0.5% per day | same |
| TX futures | Tax 0.002%, commission NT$30 per contract, 1 point slippage | NT$50, 1 point |
| Capital scenarios | Fractional (unlimited), NT$300M, NT$1B (supplement 5 adds NT$3B; addendum NT$30M, NT$100M) | Fractional, NT$10M (added in supplement 4) |

#### 1.6 Periods and verdicts

| Item | Condition |
| --- | --- |
| Periods | P1 = 2023-01-02 to 2025-07-01, P2 = 2025-07-01 to 2026-07-01, P1 ∪ P2 = in sample, P3 = 2026-07-01 to 2026-10-23 (predictions to 10-01, 64 intervals) |
| Intervals | 21-day block bootstrap, 1,000 replicates, seed 11 |
| Threshold shortlist | Institution, TW, fractional, V1: net Sharpe ≥ 1.5 in each of P1, P2 and P1 ∪ P2, and the P1 ∪ P2 annualized-net 95% lower bound > 0 |
| P3 verification H-M | Each shortlisted cell: P3 annualized net return > 0 and P3 max drawdown within twice its P1 ∪ P2 drawdown |
| Confirmation period P4' | 12 months from 2026-10-23, six cells judged separately; H-L drawdown limit = twice each cell's P1 ∪ P2 drawdown; the 2027-04-22 interim reading is reported, not judged; frozen configuration |

### 2. Conditions of each model

| Model | Type | Inputs | Label | Main hyper-parameters | Current launch |
| --- | --- | --- | --- | --- | --- |
| B0 | LightGBM, one booster per horizon | `alpha_tw_v1`: Alpha158-style price-volume features (K-bar, price and volume lags, rolling operators, Taiwan relative returns and traded-value rank, market indicator), data up to day t only | standard gauss-rank | regression objective, num_leaves 31, learning rate 0.05, feature / bagging fraction 0.8, min_data_in_leaf 200, early stopping 50 | `launch-20261101T212237Z-2508230824` |
| F0 (variant A) | same | `alpha_tw_v1` | residual a | same | `launch-20261101T220120Z-29216090` |
| B (variant B) | same | `alpha_tw_v1` | residual b | same | `launch-20261101T221954Z-1215615947` |
| F2 | same | `alpha_chip_tw_v1`: every `alpha_tw_v1` column plus 67 chip columns (institutional flows and holdings, margin, SBL, day trading, foreign holdings, attention / disposition events, monthly revenue; publication times under the conservative "before the next open" rule, day-trade data at T+2) | residual a | same | `launch-20261101T223737Z-1940816181` |
| D0 | Cross-stock transformer trained from scratch: a patch encoder over the stock's OHLCV window plus a feature side input → inter-stock attention within the same date and market → market-state gating | OHLCV window + `alpha_tw_v1` features (z-scored within the group) | residual a | learning rate 5e-4, weight decay 1e-4, up to 20 epochs, patience 4, loss = ranking loss + correlation term; bf16, one (date, market) group per batch | `launch-20261102T063519Z-189413204` |
| D1-Kronos | As D0 with the frozen Kronos-small stock encoder (cached embeddings) | same | residual a | same | `launch-20261102T071451Z-322624982` |
| D1-Chronos | As D0 with the frozen Chronos-2-small encoder | same | residual a | same | `launch-20261102T073405Z-1322713845` |

### 3. Conditions of each strategy

| Strategy | Construction | Multi-day parameters | Holding and turnover (F2, fractional, institution, whole period) |
| --- | --- | --- | --- |
| S1 multi-day long-short | A long band plus a short band, each 100%; V1-orthogonalized score | h = 10, K = 50, m = 3, at most 5 entries a day (all seven models use F0's parameters, no per-model P1 selection) | mean holding 31.6 sessions TWSE / 40.4 TPEX, daily turnover 2.8%, short candidates outside the borrowable list 57% TWSE / 73% TPEX |
| S2 long-short + TX | S1 plus a daily TX hedge of −beta × capital | as S1; beta window 60 | mean beta −0.15 (S1 itself leans short), TX notional about 16% of capital, TX fees about 0.06% a year |
| S-A weekly long | Weekly band, long only, h = 5, V1 orthogonalized | K and m chosen per model on TWSE P1 (mostly K = 50, m = 2 or 3), replacement cap 50% | mean holding 22 to 28 sessions, weekly turnover about 16% to 20% |

### 4. Status of the 21 cells

Institution, TW, fractional, primary assumptions (fills at the open; D and C parallel); annualized net return / Sharpe.

| Model | S1 multi-day long-short | S2 long-short + TX | S-A weekly long |
| --- | --- | --- | --- |
| B0 | +10.2% / 0.80; not shortlisted | +16.8% / 1.24; not shortlisted | +39.1% / 2.05; not shortlisted (P1 1.48) |
| F0 | +15.9% / 1.25; not shortlisted | +22.3% / 1.64; **shortlisted, passes H-M, in confirmation**; D + C 1.41 fails | +40.5% / 2.17; not shortlisted (P1 1.45) |
| B | +12.6% / 1.02; not shortlisted | +19.1% / 1.45; not shortlisted | +36.4% / 1.95; not shortlisted |
| F2 | +25.5% / 2.11; **shortlisted, passes H-M, in confirmation**; D + C 1.32 fails | +30.2% / 2.38; **shortlisted, passes H-M, in confirmation**; D + C 1.70 passes | +53.4% / 2.33; shortlisted, fails H-M (P3 −13.4%) |
| D0 | +19.7% / 1.52; not shortlisted | +24.5% / 1.80; **shortlisted, passes H-M, in confirmation**; D + C 1.51 passes | +40.0% / 2.06; not shortlisted (P1 1.45) |
| D1-Kronos | +16.2% / 1.28; not shortlisted | +20.4% / 1.53; not shortlisted (P2 1.49) | +44.2% / 2.16; **shortlisted, passes H-M, in confirmation**; D + C 2.06 passes |
| D1-Chronos | +19.0% / 1.48; not shortlisted | +24.4% / 1.81; **shortlisted, passes H-M, in confirmation**; D + C clears the Sharpe threshold but P3 −2.1% fails | +40.4% / 2.04; not shortlisted (P1 1.48) |

### 5. Conditions of the parallel rules, the correction and the variants

| Item | Nature | Condition | Reading (F2 × S2, institution, fractional, whole period) |
| --- | --- | --- | --- |
| D no new entry under disposition | parallel reading (rulings of 11-04 and 11-06) | Announcement out by the signal date (announce ≤ t) and the t+1 open inside the disposition window → no new entry in either leg; held positions not forced out, no batched-auction slippage | 2.38 → 1.92 |
| C forced cover at a suspension | parallel reading | A held short whose signal-date row has `short_suspended` true covers at the t+1 open (limit-down waits) and no new short during the suspension; retail = the margin rule (at least a day later than the real last cover day), institution = an upper bound | 2.38 → 2.12; D + C 1.70 |
| F fill correction (policy A) | correction, replaces the capital-row numbers | Market orders in the opening call: fill = open × (1 ± Y σ_daily √(Q ÷ (f V))), Y 0.7, f 10% (brackets 0.5 / 1.0, 5% / 20%, uncalibrated); fractional rows unchanged (Q → 0); capital rows no longer pay the flat slippage | retail NT$10M +21.5% / 1.72; institution NT$300M −16.6%, NT$1B −38.9% |
| L 15% trailing stop | strategy variant (no stop vs stop) | Exit intraday once the position's value falls 15% from its running high since entry; fill = S × (1 − half a tick − impact − c (S − today's low) ÷ S), c 0.25; a gap exits at the open, a limit-down open waits; long/short mirrored | 2.38 → 1.30; retail NT$10M F + L +5.5% / 0.43 |
| Policy B worked through the session | capacity assessment (against corrected policy A) | fill = open × exp(θ r_oc) × (1 ± half a tick ± Y σ_daily √(Q ÷ V)), θ 0.5, Y 0.7; no stop | fractional 2.01; retail NT$10M 1.94; institution NT$30M 1.59, NT$100M 1.13, NT$300M 0.80 |

### 6. Defect audit

Severity: **major** = changes a conclusion or contradicts a stated property; **medium** = a systematic bias whose direction and size can be estimated; **light** = disclosure suffices.

#### 6.1 Major

**(1) The sample universe's "14 sessions forward" condition is a look-ahead exclusion.** `sample_universe.py` requires 14 sessions after the signal date without a missing session or zero volume and drops the whole signal row otherwise — that uses facts known only after the signal date (halts, delistings, zero-volume days), and it removes those days not only from the labels but from the prediction rows and the backtest's candidate list. Computed directly on the v1.1 data on 2026-11-06: 4.36% of stock-days since 2023 are excluded this way; the excluded rows' mean open-to-open return over the next 10 sessions is −0.47% against +0.30% for the included ones (medians −0.55% vs −0.16%). Effect: the long leg never sees a set of names that go on to do badly (optimistic), the short leg loses names it could have shorted (pessimistic); with 4% of stock-days and a 0.8-point gap the effect on whole-period annualized returns is estimated under one point, but the report's section 7.1 statement "data leakage: none" does not hold and must be rewritten. Fix: use a backward-only 128-session window for the prediction and backtest universe and keep the forward requirement for labels alone; that changes the dataset identity, a new data-preparation round to be registered.

**(2) The long-short book's leverage basis is undeclared.** Net return = long book + short book, each 100% invested; in the capital scenarios each leg uses the whole capital and the TX hedge is sized on that capital as well. Every NT$1 of capital therefore carries NT$2 of gross exposure, and the margin a short needs (retail margin short 90%, institution SBL collateral about 140%) and its cash drag are not charged. Sharpe is scale-free, but annualized net return, max drawdown and the meaning of "NT$300M" are 2×-exposure figures; reported per NT$1 of gross exposure (each leg 50%) returns and drawdowns roughly halve and the capacity limit roughly doubles. Not a code defect but a convention that the reports never state; it must be declared in the summaries and limitations, with an added "per unit gross exposure" reading.

**(3) The capital scenarios are counted once per market.** Each market's book is simulated with the whole capital (e.g. NT$300M), and the TW row then averages the two markets at 50% each; the TW row's "NT$300M" is therefore NT$300M in TWSE plus NT$300M in TPEX (on top of (2)), while the impact ε is computed with NT$6M per name. If "NT$300M" means the total, each market should hold NT$150M, NT$3M per name, and ε shrinks by √2 to 0.71×, moving the capacity verdict up by about 1.4×. The convention must be decided and the capital scenarios rerun.

#### 6.2 Medium

**(4) Three uncalibrated impact parameters decide the capacity verdict.** Y (0.5 to 1.0), f (the auction's share, 5% to 20%) and the 5% participation cap have no Taiwan evidence; moving Y from 0.7 to 0.5 lifts policy B's institution NT$300M Sharpe from 0.80 to 1.15. The "about NT$30M" verdict is sensitive to all three and should be stated as a range; calibrating f from tick or minute data is the remedy.

**(5) The TX hedge's beta benchmark and hedging instrument differ.** Beta is estimated against the market's own index (the TPEX book against the TPEX index) but the hedge trades TX futures, so the TPEX book keeps a residual TPEX-versus-TAIEX exposure. The TWSE book has no such problem. Fix: estimate the TPEX book's beta against TX directly, or accept and disclose the residual.

**(6) The short leg loses money on its own and is mostly unborrowable.** The short leg alone earns −21.6% a year (its value is lower beta and drawdown), and 57% (TWSE) / 73% (TPEX) of short candidates are outside the borrowable list, so the names actually shorted sit further down the ranking. The cheaper alternative — the long band plus a TX hedge with no stock shorts, which removes borrow fees, failures, suspensions and dispositions at a stroke — has not been compared. A design gap, not an error; register it as a comparison strategy.

**(7) Both parallel rules remain optimistic.** D forces no exit and adds no batched-auction slippage; C is at least a day late and the institution reads an upper bound; the stop's limit-down check looks at the open only. Declared in the protocol and the reports; the quarterly confirmation readings list D + C alongside (arranged).

**(8) Industry and share counts come from a single snapshot.** The residual labels' industry dummies and variant B's share counts come from the 2026-10-22 snapshot, so companies delisted or reclassified since are described by today's classification; a slight look-ahead component in the 2023 labels. Disclosed, not quantified.

#### 6.3 Light

**(9) Equal-weight slots are not rebalanced**: equal at entry, then drifting; small for K = 50, but long-held winners grow overweight. **(10) Price limits are judged at the open only**: intraday locks do not affect the open-to-open rules, but the stop variant's intraday fills are overstated on limit-down days. **(11) Dividend tax and margin interest**: adjusted prices carry dividends, but dividend tax (21% withholding for institutions, income tax for individuals) and interest on margin collateral are not charged. **(12) Labels run to the close, trades to the open**: the label is open t+1 to close t+h while the strategy holds open to open, and the horizon is not the holding period (36 sessions vs 10); deliberate (the band rule), with the EMA smoothing the score's decay over a holding. **(13) Weekly K, m chosen on TWSE P1**: in-sample parameter choice, disclosed. **(14) Deep models are not reproducible run to run** (bf16, different GPUs); the three-seed ensemble mitigates it. **(15) Thresholds and the candidate set were set after the first edition's out-of-sample results**, disclosed; P3 has 64 sessions only.

### 7. Suggested order of handling

1. Rewrite both reports' "data leakage: none" statement and limitations at once, disclosing and quantifying (1), (2) and (3).
2. Register and decide the conventions: (2) report the long-short book per unit gross exposure or per unit capital; (3) whether capital scenarios are totals or per market. Then rerun the capital scenarios (about 20 minutes locally).
3. Register the data-preparation fix (1): prediction rows and backtest candidates on the backward window only; a new dataset identity, best done together with the confirmation period's first extension to avoid an extra Pod round.
4. Register the comparison strategy of (6): long band plus TX hedge, no stock shorts.
5. (4) and (5) as the next round's method improvements.
