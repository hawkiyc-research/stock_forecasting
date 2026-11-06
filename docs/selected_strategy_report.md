# 台股量化選股策略（第二版）：七個模型、三種策略、門檻篩選與樣本外驗證 / A Quantitative Taiwan Stock Strategy (second edition): Seven Models, Three Strategies, the Threshold Shortlist and Out-of-Sample Verification

版本：2026-11-02，2026-11-04 補充 §7.5 執行層敏感度，2026-11-06 補充 §7.6 成交價模型與追蹤停損、§7.7 方式 B 與容量，並修訂摘要、6、9、10（取代 2026-10-29 的第一版；第一版封存於 `reports/deprecated/pre-1323-20261102/`，11-02 的 Word 版封存於 `reports/deprecated/pre-1323s3-20261104/`，11-04 的 Word 版封存於 `reports/deprecated/pre-1323s4-20261106/`）。完整版見 `docs/research_full_report.md`。
Version: 2026-11-02, with §7.5 (execution-layer sensitivity) added on 2026-11-04, §7.6 (fill model and trailing stop) and §7.7 (policy B and capacity) on 2026-11-06, the summary, 6, 9 and 10 revised (supersedes the first edition of 2026-10-29, archived under `reports/deprecated/pre-1323-20261102/`; the 11-02 Word files are archived under `reports/deprecated/pre-1323s3-20261104/`, the 11-04 Word files under `reports/deprecated/pre-1323s4-20261106/`). The full report is `docs/research_full_report.md`.

---

## 繁體中文

### 1. 一頁摘要

- **問題**：日線價量加免費籌碼資料的橫斷面排序模型，在台灣的交易成本與執行限制下，能不能變成一條可進入模擬交易的策略？
- **這一輪做了什麼**：修正還原價（有償現金增資）後，七個模型在 2016 到 2026-10-23 的完整資料上重訓；七個模型 × 三種策略 = 21 格一起用同一套事前門檻篩選；入選的格各自做樣本外驗證。
- **模型層**：價量＋籌碼的 LightGBM（F2）在所有期間都是最強的模型（樣本內 5 到 10 日 Rank IC 0.109，對純價量 +0.019）。三個跨股票 transformer 短天期略優於純價量 LightGBM，5 到 10 日沒有顯著差異。
- **門檻篩選**（機構主要成本、上市上櫃各半、分數位置；前段、後段、全段淨 Sharpe 各 ≥ 1.5、全段區間下界 > 0）：21 格中 **7 格入選**。
- **樣本外驗證（2026-07-01 至 10-01，64 個交易日；淨報酬 > 0 且回撤在樣本內兩倍之內）**：7 格中 **6 格通過**。最強的一格 **F2 × 多空＋台指期避險**：樣本內三段 Sharpe 2.17／2.84／2.38、年化 +30.2%；樣本外 **+19.6%**、Sharpe 1.48、回撤 −8.0%。唯一不通過的是 F2 × 週頻多頭（樣本外 −13.4%）。
- **處置**：六個通過的格一起進入 **2026-10-23 起 12 個月的確認期**，規則事前登記、期間內不得修改；期滿各自判定是否進入模擬交易。沒有策略在確認期之前進入模擬交易。
- **執行層補充敏感度（11 月 4 日，§7.5）**：加上「處置期間不新進場」與「停券起日強制回補」後，F2 × 多空＋避險全段 Sharpe 2.38 → 1.70、年化 +30.2% → +22.6%，樣本外 +20.4% 不變；下降超過事前預期，這套保守假設成為確認期每季讀數的參考。六格裡 F2 × S2、D0 × S2、D1-Kronos × S-A 仍過門檻，F2 × S1、F0 × S2、D1-Chronos × S2 未過並帶註記；散戶成本下只有 D1-Kronos × S-A 通過。
- **成交價的修正（11 月 6 日，§7.6）**：實際下單方式是市價單進開盤集合競價，資本情境的成交價改為「開盤價 × (1 ± 自身衝擊)」，數字據此修正：散戶 1,000 萬 F2 × S2 全段 +21.5%（Sharpe 1.72），機構 3 億 −16.6%、10 億 −38.9%——這套策略在「全部掛開盤」下容量遠低於 3 億。分數位置列（Q → 0）不變，確認期六格不受影響。
- **策略變體：15% 追蹤停損（§7.6）**：與無停損版本對照，七格全部較差、回撤沒有變淺（F2 × S2 分數位置全段 Sharpe 2.38 → 1.30）；停損版本不進入確認期。
- **稽核（11 月 6 日）**：21 格的條件與假設清單在 `docs/assumptions_audit_20261106.md`；三項重大問題（選樣前瞻、多空書 2 倍曝險未宣告、資本情境兩市場各算一次）見第 9 節 (g)，原本的「資料洩漏：沒有」改為揭露。
- **方式 B 與容量（11 月 6 日，§7.7）**：改為盤中分批下單，延遲成本約 5 個百分點（F2 × S2 分數位置 2.38 → 2.01），但 3 億仍只剩 +9.0%／0.80；容量約 3,000 萬（F2 × S2 散戶 1,000 萬 +24.1%／1.94、機構 3,000 萬 1.59、1 億 1.13），純多頭約 1 億，是中小型股宇宙與換手結構的限制。
- **要記得的限制**：候選集與門檻是看過第一版樣本外結果之後定的，所以這一輪的樣本外只對重訓後的分數乾淨；64 個交易日的區間寬 ±50 到 ±120 個百分點；多空策略的樣本外報酬全部來自上市、上櫃為負。

### 2. 研究問題

用每天收盤後可得的資料，對同一天、同一市場的股票排序，次日開盤成交，持有數天到數週。目標不是預測大盤，而是分辨「同一天裡誰相對強、誰相對弱」。第一版用十三天跑完七個階段把問題收斂到這裡；這一輪把資料修正、把全部模型重訓、把候選集擴大，再做一次。

### 3. 資料與防止偷看未來的規則

| 項目 | 設定 |
| --- | --- |
| 資料 | 上市上櫃全部股票日線價量 2016-01-01 至 2026-10-23（修訂 v1.1：還原價只套除權息因子，有償現金增資以交易所開盤競價基準還原）；12 個官方籌碼品項，每列帶公布時刻 |
| 時間點規則 | 特徵只用訊號日之前的資料；籌碼只用次日開盤前已公布的列；每季的模型只用該季開始前 15 個交易日以前的資料訓練 |
| 訓練 | 每季重訓、2016 起擴張；2023Q1 到 2026Q4 共 16 折；每折保存 validation Rank IC 最高的模型 |
| 樣本內 | 前段 2023-01-02 至 2025-06-30、後段 2025-07-01 至 2026-06-30 |
| 樣本外 | 2026-07-01 至 2026-10-01（64 個開盤到開盤區間） |
| 確認期 | 2026-10-23 起 12 個月（完全在資料結束日之後） |

### 4. 七個模型

| 代號 | 模型 | 特徵 | 標籤 |
| --- | --- | --- | --- |
| B0 | LightGBM | 183 欄價量 | 同日同市場排名 |
| F0 | LightGBM | 183 欄價量 | 殘差化報酬 a（扣除市場、市值、產業） |
| 變體 B | LightGBM | 183 欄價量 | 殘差化報酬 b |
| **F2** | LightGBM | 183 欄價量＋67 欄籌碼 | 殘差化報酬 a |
| D0 | 跨股票 transformer（從頭訓練，三種子集成） | 183 欄價量 | 殘差化報酬 a |
| D1-Kronos | 跨股票 transformer，Kronos-small 編碼器 | 同上 | 同上 |
| D1-Chronos | 跨股票 transformer，Chronos-2-small 編碼器 | 同上 | 同上 |

樣本內 Rank IC（短／5 到 10 日／長）：B0 0.117／0.087／0.086、F0 0.117／0.090／0.090、變體 B 0.113／0.085／0.083、**F2 0.129／0.109／0.105**、D0 0.121／0.095／0.096、D1-Kronos 0.121／0.094／0.092、D1-Chronos 0.121／0.093／0.092。模型層的數字只作資訊，不參與篩選。

### 5. 三種交易策略

分數在流動性宇宙內取百分位 → 3 日半衰期指數平滑 → 每天對 60 日 β、前 20 日與前 60 日報酬做橫斷面回歸取殘差（三種策略共用）。

| 策略 | 規則 | 參數（事前固定） | 曝險 |
| --- | --- | --- | --- |
| S1 多日多空 | 10 日分數；多頭 band（買進前 K、掉出前 m × K 才賣、每天最多進 K／10 檔）；空頭 band 反向排序，受可借券名單、限額、借不到機率與分層借券費約束 | K = 50、m = 3，七個模型相同 | 金額中性，毛 2 倍 |
| S2 多空＋台指期避險 | S1 加台指期 −β × 資本，β 為組合毛報酬對指數的前 60 日滾動 β | 同上 | 金額中性加 β 對沖 |
| S-A 週頻低換手多頭 | 5 日分數；每個 ISO 週最後一個交易日再平衡；前 K 多頭 band、掉出前 m × K 才賣；每週最多換掉 50% × K | (K, m) 各模型在上市、前段選定：F2 (30, 3)、D1-Kronos (30, 3)，其餘見完整版 | 100% 多頭 |

### 6. 成本與執行假設

機構：手續費 0.03%、滑價 0.05%、借券費依規模 1.5%／3%／6%；散戶：手續費 0.0855%、滑價 0.1%／0.2%、融券手續費 0.08%；證交稅 0.3%。放空只能在可借券名單內（覆蓋率上市 97.6%、上櫃 92.6%），受次日限額封頂，借不到機率依規模 2%／10%／25%，五次隨機重複平均。資本情境：分數位置；機構 3 億與 10 億。主表沒有處置股與停券強制回補的約束；這兩條規則是 11 月 4 日補登記的敏感度，見 7.5。成交價：分數位置列是開盤價（Q → 0）；資本情境列依實際下單方式（市價單進開盤集合競價）加上自身衝擊，ε = Y × σ_日 × √(Q ÷ (f × V))，Y 0.7、f 10%（未校準，見 7.6）。主策略沒有停損；15% 追蹤停損是另一組策略變體（7.6）。

### 7. 結果

#### 7.1 樣本內：21 格的門檻篩選（機構、上市上櫃各半、分數位置）

| 模型 × 策略 | P1 Sharpe | P2 Sharpe | 全段 Sharpe | 全段年化淨報酬 | 95% 區間 | 全段回撤 | 入選 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | 2.17 | 2.84 | **2.38** | +30.2% | [+18.5%, +43.0%] | −8.1% | **是** |
| F2 × S-A | 1.54 | 3.79 | 2.33 | +53.4% | [+28.1%, +82.9%] | −25.1% | **是** |
| D1-Kronos × S-A | 1.56 | 3.36 | 2.16 | +44.2% | [+22.4%, +67.7%] | −23.9% | **是** |
| F2 × S1 | 1.90 | 2.58 | 2.11 | +25.5% | [+13.4%, +39.2%] | −11.0% | **是** |
| D1-Chronos × S2 | 1.69 | 2.10 | 1.81 | +24.4% | [+11.2%, +39.0%] | −9.3% | **是** |
| D0 × S2 | 1.62 | 2.21 | 1.80 | +24.5% | [+11.1%, +37.8%] | −8.2% | **是** |
| F0 × S2 | 1.53 | 1.91 | 1.64 | +22.3% | [+8.2%, +36.6%] | −8.6% | **是** |
| 其餘 14 格 | — | — | 0.80 到 2.17 | — | — | — | 否 |

落選的 14 格裡，五個週頻多頭的全段 Sharpe 都在 1.95 到 2.17，但前段只有 1.44 到 1.48——高 Sharpe 幾乎全靠後段，正是門檻要擋的型態；其餘是多空格在某一段低於 1.5（完整表見完整版第 6.2 節）。

#### 7.2 樣本外（2026-07-01 至 10-01，64 個交易日）：入選 7 格

規則：(i) 年化淨報酬 > 0、(ii) 最大回撤不深於該格全段回撤的兩倍。

| 入選格 | 樣本外年化淨報酬 | 95% 區間 | Sharpe | 樣本外回撤 | 兩倍門檻 | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | **+19.6%** | [−38.2%, +87.8%] | 1.48 | −8.0% | −16.2% | **通過** |
| F2 × S1 | +14.2% | [−42.1%, +81.4%] | 1.06 | −8.3% | −22.0% | **通過** |
| D1-Kronos × S-A | +4.8% | [−57.6%, +63.2%] | 0.16 | −16.1% | −47.8% | **通過** |
| D1-Chronos × S2 | +14.5% | [−49.0%, +91.8%] | 1.03 | −10.0% | −18.6% | **通過** |
| D0 × S2 | +15.7% | [−38.2%, +77.9%] | 1.16 | −7.8% | −16.4% | **通過** |
| F0 × S2 | +7.6% | [−33.4%, +52.6%] | 0.60 | −7.0% | −17.2% | **通過** |
| F2 × S-A | −13.4% | [−120.3%, +75.8%] | −0.33 | −24.6% | −50.2% | 不通過（淨報酬 < 0） |

分市場：上市 F2 × S2 +47.3%（Sharpe 2.54）、D0 × S2 +34.6%、F0 × S2 +27.3%；上櫃的多空格全部為負（F2 × S2 −8.1%）。對照：等權流動性宇宙 +5.4%（回撤 −22.8%）、0050 +14.8%、加權報酬指數 +14.0%。

#### 7.3 散戶成本與資本情境（F2 × S2）

資本情境的成交價依實際下單方式計算（11 月 6 日修正，7.6）：市價單進開盤集合競價，成交價 = 開盤價 × (1 ± 自身衝擊)。散戶分數位置（成交價 = 開盤價）：全段 +29.1%（2.26）、樣本外 +24.0%（1.73）。散戶 1,000 萬：全段 +21.5%（1.72）、樣本外 -8.3%（-0.56）。機構 3 億：全段 -16.6%（-1.45）、樣本外 -58.2%（-3.91）；10 億：全段 -38.9%（-4.77）。滑價 0／1／2 倍（分數位置）：+31.6%／+30.2%／+28.7%。借不到倍率 × 0 到 × 2 與召回 0.5%（分數位置）：+27.8% 到 +30.9%。逐年（分數位置）：2023 +17.3%、2024 +34.2%、2025 +21.8%、2026（至 10-01）+40.6%，沒有一年回撤深過 −8%。散戶分數位置用同一套門檻（三段 Sharpe ≥ 1.5、全段區間下界 > 0）判：F2 × S2（1.95／2.94／2.26）、F2 × S1（1.67／2.63／1.98）、D1-Chronos × S2（1.54／2.22／1.75）入選，三格的樣本外都為正且回撤在兩倍之內；D0 × S2、F0 × S2、D1-Kronos × S-A 因前段 Sharpe 低於 1.5 出局。衝擊的量級與容量見 7.6、7.7。

#### 7.4 怎麼讀

- 樣本外 64 天的區間寬 ±50 到 ±120 個百分點，只能排除「策略壞了」，不能證明「策略有多好」。F2 × S2 是唯一一格四段期間 Sharpe 都明顯高於 1 的策略。
- 多空格的樣本外報酬全部來自上市，上櫃為負（上櫃空頭腿在 7 到 9 月的反彈中受損）；確認期用兩市場各半判定。
- 純多頭的週頻策略再一次在 7 月的廣度修正裡吃到全額回撤；D1-Kronos × S-A 通過只因它的樣本內回撤本來就深（−23.9%）、兩倍門檻很寬。
- 籌碼在策略層的增量（F2 − F0 配對差，全段年化）：S1 +9.6% [+1.6%, +18.1%]、S-A +12.9% [+4.2%, +23.1%] 宣稱有增量；S2 +7.8% [−0.8%, +16.3%] 不宣稱。

#### 7.5 執行層補充敏感度（2026-11-04）：處置期間不新進場、停券起日強制回補

主表把處置股當正常交易、已持有的空頭只有隨機召回的敏感度。11 月 4 日登記並實作兩條確定性規則（協定 §13.23 補充三）：**D** = 次一開盤落在處置期間（公告已出）的股票，多頭與空頭都不新進場；**C** = 已持有的空頭在該股暫停融券賣出的次一開盤強制回補，停券期間不新建空單（散戶的融券規則；機構以同一旗標作上界）。F2 × S2 在 D + C 下兩腿合計跳過約 9% 的候選、811 個區間內被強制回補約 400 次。

機構主要成本（年化淨報酬 / Sharpe / 最大回撤；全段 = P1 ∪ P2，樣本外 = P3 的 64 個區間；TW 兩市場各半、分數位置、主滑價）：

| 格 | 全段 基準 | 全段 D | 全段 C | 全段 D + C | 樣本外 基準 | 樣本外 D + C | D + C 讀法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +30.2% / 2.38 / -8.1% | +25.7% / 1.92 / -8.1% | +26.7% / 2.12 / -8.1% | **+22.6% / 1.70 / -8.3%** | +19.6% / 1.48 / -8.0% | **+20.4% / 1.39 / -8.3%** | 通過 |
| F2 × S1 | +25.5% / 2.11 / -11.0% | +18.9% / 1.49 / -11.2% | +22.1% / 1.85 / -11.1% | **+16.6% / 1.32 / -11.5%** | +14.2% / 1.06 / -8.3% | **+16.0% / 1.13 / -9.6%** | 未通過（Sharpe < 1.5） |
| D0 × S2 | +24.5% / 1.80 / -8.2% | +22.3% / 1.65 / -9.5% | +21.9% / 1.63 / -8.1% | **+20.3% / 1.51 / -9.3%** | +15.7% / 1.16 / -7.8% | **+3.2% / 0.23 / -10.4%** | 通過 |
| D1-Chronos × S2 | +24.4% / 1.81 / -9.3% | +23.3% / 1.73 / -8.8% | +24.6% / 1.86 / -8.6% | **+21.4% / 1.61 / -8.4%** | +14.5% / 1.03 / -10.0% | **-2.1% / -0.14 / -10.9%** | 未通過（樣本外 < 0） |
| F0 × S2 | +22.3% / 1.64 / -8.6% | +20.1% / 1.46 / -8.7% | +20.4% / 1.54 / -9.2% | **+19.0% / 1.41 / -9.2%** | +7.6% / 0.60 / -7.0% | **+26.8% / 1.87 / -8.5%** | 未通過（Sharpe < 1.5） |
| D1-Kronos × S-A | +44.2% / 2.16 / -23.9% | +40.2% / 2.06 / -24.7% | +44.2% / 2.16 / -23.9% | **+40.2% / 2.06 / -24.7%** | +4.8% / 0.16 / -16.1% | **+17.7% / 0.59 / -11.7%** | 通過 |
| F2 × S-A（未進確認期） | +53.4% / 2.33 / -25.1% | +38.3% / 1.97 / -24.3% | +53.4% / 2.33 / -25.1% | **+38.3% / 1.97 / -24.3%** | -13.4% / -0.33 / -24.6% | **+5.7% / 0.22 / -13.6%** | 通過 |

散戶成本（年化淨報酬 / Sharpe / 最大回撤；全段 = P1 ∪ P2，樣本外 = P3 的 64 個區間；TW 兩市場各半、分數位置、主滑價）：

| 格 | 全段 基準 | 全段 D | 全段 C | 全段 D + C | 樣本外 基準 | 樣本外 D + C | D + C 讀法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1% / 2.26 / -7.3% | +24.9% / 1.85 / -7.7% | +24.7% / 1.96 / -8.6% | **+19.4% / 1.45 / -8.7%** | +24.0% / 1.73 / -7.9% | **+22.5% / 1.51 / -8.4%** | 未通過（Sharpe < 1.5） |
| F2 × S1 | +24.2% / 1.98 / -10.5% | +18.5% / 1.44 / -10.6% | +20.3% / 1.70 / -11.5% | **+13.4% / 1.06 / -11.0%** | +19.8% / 1.43 / -8.4% | **+18.3% / 1.28 / -9.8%** | 未通過（Sharpe < 1.5） |
| D0 × S2 | +22.2% / 1.63 / -9.5% | +20.7% / 1.52 / -10.4% | +19.0% / 1.41 / -9.4% | **+15.6% / 1.17 / -9.9%** | +19.5% / 1.40 / -8.0% | **-1.0% / -0.07 / -10.6%** | 未通過（Sharpe < 1.5） |
| D1-Chronos × S2 | +23.3% / 1.75 / -9.1% | +19.9% / 1.47 / -9.4% | +18.5% / 1.40 / -8.6% | **+17.0% / 1.29 / -9.7%** | +3.8% / 0.27 / -11.9% | **-0.9% / -0.06 / -11.4%** | 未通過（Sharpe < 1.5） |
| F0 × S2 | +19.5% / 1.40 / -8.5% | +18.9% / 1.36 / -8.7% | +18.4% / 1.37 / -9.5% | **+16.3% / 1.19 / -9.7%** | +3.1% / 0.24 / -8.3% | **+21.9% / 1.51 / -8.1%** | 未通過（Sharpe < 1.5） |
| D1-Kronos × S-A | +41.2% / 2.02 / -24.4% | +37.2% / 1.91 / -26.4% | +41.2% / 2.02 / -24.4% | **+37.2% / 1.91 / -26.4%** | +1.5% / 0.05 / -16.3% | **+14.6% / 0.48 / -12.0%** | 通過 |
| F2 × S-A（未進確認期） | +49.7% / 2.18 / -25.3% | +35.4% / 1.82 / -24.6% | +49.7% / 2.18 / -25.3% | **+35.4% / 1.82 / -24.6%** | -17.9% / -0.45 / -24.7% | **+1.2% / 0.05 / -13.8%** | 通過 |

下降幅度超過事前預期（F2 × S2 全段 −7.5 個百分點／Sharpe −0.68），依登記的規則 **D + C 成為確認期每季讀數的保守參考**：六格裡 F2 × S2、D0 × S2、D1-Kronos × S-A 在保守假設下仍過門檻，**F2 × S1、F0 × S2、D1-Chronos × S2 未過，自此帶註記「執行層敏感度未通過」**；散戶只有 D1-Kronos × S-A 通過。影響主要來自處置規則，而且多頭與空頭兩腿都有：排序模型把兩端集中在隨後被處置的股票，P2 的多頭報酬有相當部分來自它們（F2 × S-A 的 P2 由 +114% 降到 +65%），樣本外則反過來（D 讓 F2 × S-A 的樣本外由 −13.4% 變成 +5.7%）。兩條規則仍偏樂觀（D 不強制出場、不加滑價；C 比真實最後回補日晚至少一天）；確認期六格與凍結設定不變，是否改以 D + C 為主要假設由使用者決定。

#### 7.6 成交價的修正（開盤集合競價市價單）與策略變體：15% 追蹤停損（2026-11-06）

**成交價的修正。** 實際下單方式是市價單進開盤集合競價：單一價格撮合、沒有價差，代價是自身單量推動撮合價的幅度。回測的成交價因此改為「開盤價 × (1 ± Y × σ_日 × √(Q ÷ (f × V)))」，Y 0.7、f（開盤量占全日量）10%，台股沒有實證參數，括弧（Y 0.5／1.0、f 5%／20%）明寫未校準。這是修正而不是敏感度：資本情境的數字一律以修正後的成交價計算；分數位置列因 Q → 0 不變，確認期六格不受影響。宇宙裡個股 20 日成交金額中位數只有 6,900 萬，3 億在 50 檔就把 61% 的部位推到 5% 參與率上限，每邊衝擊中位 0.85%；1,000 萬 0.25%。

修正後的資本情境（年化淨報酬（Sharpe））：

| 格 | 散戶 1,000 萬 全段 / 樣本外 | 機構 3 億 全段 / 樣本外 | 機構 10 億 全段 / 樣本外 |
| --- | --- | --- | --- |
| F2 × S2 | +21.5%（1.72） / -8.3%（-0.56） | -16.6%（-1.45） / -58.2%（-3.91） | -38.9%（-4.77） / -108.2%（-7.35） |
| F2 × S1 | +18.5%（1.51） / -3.8%（-0.21） | -21.7%（-1.95） / -61.7%（-4.07） | -37.3%（-4.52） / -108.2%（-6.98） |
| D0 × S2 | +11.4%（0.86） / -10.3%（-0.81） | -30.0%（-2.58） / -40.8%（-2.63） | -54.0%（-6.37） / -84.3%（-5.74） |
| D1-Chronos × S2 | +11.5%（0.89） / -16.3%（-1.07） | -29.6%（-2.48） / -58.2%（-3.91） | -50.1%（-5.73） / -83.7%（-5.57） |
| F0 × S2 | +14.5%（1.09） / -7.2%（-0.54） | -30.5%（-2.65） / -82.9%（-6.10） | -49.8%（-5.92） / -102.6%（-6.13） |
| D1-Kronos × S-A | +33.2%（1.65） / -5.5%（-0.19） | +2.1%（0.14） / -19.3%（-0.58） | -13.5%（-0.87） / -36.0%（-1.27） |
| F2 × S-A（未進確認期） | +36.5%（1.79） / -44.7%（-1.24） | +3.0%（0.18） / -67.5%（-2.05） | -15.1%（-0.91） / -99.4%（-2.90） |

**策略變體：15% 追蹤停損。** 持有部位的價值自進場後的最高點回落 15% 即盤中出場，成交價質量點在停損價、右尾往低價（半檔位 + 衝擊 + c × (停損價 − 當日最低)，c 0.25）。兩組策略都以修正後的成交價計算（年化淨報酬 / Sharpe / 最大回撤；全段 = P1 ∪ P2，樣本外 = P3；TW 兩市場各半、主滑價）：

| 格 | 成本檔 / 資金 | 期間 | 無停損 | 15% 追蹤停損 |
| --- | --- | --- | --- | --- |
| F2 × S2 | institution fractional | P1+P2 | +30.2% / 2.38 / -8.1% | +17.0% / 1.30 / -8.6% |
| F2 × S2 | institution fractional | P3 | +19.6% / 1.48 / -8.0% | +23.1% / 1.64 / -8.1% |
| F2 × S2 | retail 10M | P1+P2 | +21.5% / 1.72 / -11.1% | +5.5% / 0.43 / -15.8% |
| F2 × S2 | retail 10M | P3 | -8.3% / -0.56 / -5.9% | -40.3% / -3.15 / -11.0% |
| F2 × S1 | institution fractional | P1+P2 | +25.5% / 2.11 / -11.0% | +13.1% / 1.05 / -11.0% |
| F2 × S1 | institution fractional | P3 | +14.2% / 1.06 / -8.3% | +12.3% / 0.85 / -10.0% |
| F2 × S1 | retail 10M | P1+P2 | +18.5% / 1.51 / -13.5% | +4.0% / 0.32 / -20.3% |
| F2 × S1 | retail 10M | P3 | -3.8% / -0.21 / -6.8% | -40.9% / -2.91 / -11.9% |
| D0 × S2 | institution fractional | P1+P2 | +24.5% / 1.80 / -8.2% | +13.7% / 1.04 / -10.8% |
| D0 × S2 | institution fractional | P3 | +15.7% / 1.16 / -7.8% | -0.4% / -0.03 / -8.0% |
| D0 × S2 | retail 10M | P1+P2 | +11.4% / 0.86 / -13.0% | -2.5% / -0.20 / -23.7% |
| D0 × S2 | retail 10M | P3 | -10.3% / -0.81 / -5.9% | -46.0% / -4.60 / -11.8% |
| D1-Chronos × S2 | institution fractional | P1+P2 | +24.4% / 1.81 / -9.3% | +9.4% / 0.73 / -9.8% |
| D1-Chronos × S2 | institution fractional | P3 | +14.5% / 1.03 / -10.0% | -15.3% / -1.17 / -12.7% |
| D1-Chronos × S2 | retail 10M | P1+P2 | +11.5% / 0.89 / -12.5% | -3.3% / -0.27 / -29.1% |
| D1-Chronos × S2 | retail 10M | P3 | -16.3% / -1.07 / -6.0% | -31.5% / -2.64 / -8.9% |
| F0 × S2 | institution fractional | P1+P2 | +22.3% / 1.64 / -8.6% | +10.2% / 0.78 / -9.4% |
| F0 × S2 | institution fractional | P3 | +7.6% / 0.60 / -7.0% | -12.0% / -0.85 / -11.5% |
| F0 × S2 | retail 10M | P1+P2 | +14.5% / 1.09 / -11.0% | -3.8% / -0.29 / -22.6% |
| F0 × S2 | retail 10M | P3 | -7.2% / -0.54 / -7.4% | -33.5% / -2.98 / -12.3% |
| D1-Kronos × S-A | institution fractional | P1+P2 | +44.2% / 2.16 / -23.9% | +37.2% / 1.97 / -29.2% |
| D1-Kronos × S-A | institution fractional | P3 | +4.8% / 0.16 / -16.1% | +2.8% / 0.11 / -9.1% |
| D1-Kronos × S-A | retail 10M | P1+P2 | +33.2% / 1.65 / -28.6% | +27.0% / 1.50 / -33.9% |
| D1-Kronos × S-A | retail 10M | P3 | -5.5% / -0.19 / -15.8% | -29.5% / -1.51 / -14.0% |
| F2 × S-A（未進確認期） | institution fractional | P1+P2 | +53.4% / 2.33 / -25.1% | +45.0% / 2.22 / -26.1% |
| F2 × S-A（未進確認期） | institution fractional | P3 | -13.4% / -0.33 / -24.6% | -22.7% / -0.78 / -19.6% |
| F2 × S-A（未進確認期） | retail 10M | P1+P2 | +36.5% / 1.79 / -24.7% | +30.0% / 1.63 / -28.6% |
| F2 × S-A（未進確認期） | retail 10M | P3 | -44.7% / -1.24 / -26.1% | -40.7% / -1.61 / -19.9% |

- 停損版本在七格全部較差、回撤沒有變淺：F2 × S2 分數位置全段 Sharpe 2.38 → 1.30、年化 +30.2% → +17.0%，平均持有 36 → 20 日、換手加倍；超越量只占代價的三分之一，其餘是換手與停在低點後的反彈；純多頭的回撤反而更深。只有 F2 × S2 的樣本外略升（+23.1%）。停損版本不進入確認期。
- 全部並列規則（處置、停券與停損一起）：機構分數位置 F2 × S2 全段 +10.9% / 0.85 / -11.9%、樣本外 +17.6% / 1.30 / -7.1%。

#### 7.7 方式 B：盤中分批下單與容量（2026-11-06）

修正後的方式 A（7.6）顯示 3 億不可行，使用者登記方式 B 作容量評估：以開盤價為到達價、盤中分批執行，成交價 = 開盤價 × exp(θ × 當日開盤到收盤報酬) × (1 ± 半檔位 ± Y σ_日 √(Q ÷ V))，θ 0.5、Y 0.7，衝擊以全日量為分母，不含停損，加入機構 30 億情境。

分數位置與資本情境（F2 × S2 與純多頭格，其餘見完整報告 §6.9；分數位置的方式 A 成交價是開盤價，與 B 的差就是延遲成本）（年化淨報酬 / Sharpe / 最大回撤；全段 = P1 ∪ P2，樣本外 = P3；TW 兩市場各半、主滑價；基準 = 同一成本檔副本、無規則的參考跑）：

| 列 | 期間 | 方式 A（修正後） | 方式 B | B + D + C |
| --- | --- | --- | --- | --- |
| F2 × S2 機構 分數位置 | P1+P2 | +30.2% / 2.38 / -8.1% | +25.4% / 2.01 / -9.8% | +17.9% / 1.34 / -10.1% |
| F2 × S2 機構 分數位置 | P3 | +19.6% / 1.48 / -8.0% | +19.5% / 1.50 / -8.2% | +17.7% / 1.21 / -8.8% |
| F2 × S2 散戶 1,000 萬 | P1+P2 | +21.5% / 1.72 / -11.1% | +24.1% / 1.94 / -10.9% | +17.8% / 1.50 / -9.6% |
| F2 × S2 機構 3 億 | P1+P2 | -16.6% / -1.45 / -46.8% | +9.0% / 0.80 / -9.9% | -0.1% / -0.00 / -12.9% |
| F2 × S2 機構 10 億 | P1+P2 | -38.9% / -4.77 / -71.9% | -4.1% / -0.53 / -22.6% | -10.5% / -1.41 / -30.2% |
| F2 × S2 機構 30 億 | P1+P2 | （方式 A 未跑） | -15.1% / -2.48 / -39.4% | -20.9% / -3.50 / -49.4% |
| D1-Kronos × S-A 機構 3 億 | P1+P2 | +2.1% / 0.14 / -37.7% | +20.3% / 1.30 / -25.7% | +16.3% / 1.14 / -25.3% |
容量（方式 B 下全段淨 Sharpe ≥ 1.5 的最大資本情境）：

| 格 | 分數位置 | 3 億 | 10 億 | 30 億 | 容量（Sharpe ≥ 1.5） | 散戶 1,000 萬 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| F2 × S2 | 2.01 | 0.80 | -0.53 | -2.48 | 只有分數位置 | 1.94 |
| F2 × S1 | 1.73 | 0.36 | -0.32 | -1.63 | 只有分數位置 | 1.73 |
| D0 × S2 | 1.40 | -0.01 | -1.38 | -3.83 | 無 | 1.10 |
| D1-Chronos × S2 | 1.39 | -0.03 | -1.13 | -3.84 | 無 | 1.15 |
| F0 × S2 | 1.25 | -0.00 | -1.15 | -4.18 | 無 | 1.33 |
| D1-Kronos × S-A | 2.01 | 1.30 | 0.99 | -0.12 | 只有分數位置 | 1.78 |
| F2 × S-A（未進確認期） | 2.11 | 1.58 | 1.01 | 0.10 | 300M | 1.97 |

- 延遲成本約 5 個百分點、對執行時點不敏感（θ 0.25 到 0.75 差 1.5 個百分點），訊號不在開盤瞬間實現；「方式 B 延遲成本敏感度」通過。
- 3 億仍不達標（F2 × S2 +9.0%／0.80，對比修正後的方式 A −16.6%；整個文獻範圍的 Y 都不達標），10 億、30 億為負且部署率只剩一半以下；原因是中小型股宇宙與兩腿各 50 檔的換手結構，衝擊 ∝ √(Q ÷ V)。散戶 1,000 萬可行（F2 × S2 +24.1%／1.94）；補跑 3,000 萬與 1 億後，多空＋避險的容量約 3,000 萬（機構 3,000 萬 +21.3%／1.59；1 億 +14.7%／1.13），純多頭約 1 億（D1-Kronos × S-A 1.55、F2 × S-A 1.71；F2 × S-A 在 3 億仍 1.58）。
- 都是並列讀數，確認期六格不變；要管理 3 億以上需要擴大 K、降低換手或改宇宙，那是新策略。

### 8. 樣本是怎麼用的：洩漏與設計偏誤的交代

**資料洩漏：特徵、籌碼、標籤、訓練切分與策略輸入沒有；選樣有一處前瞻（2026-11-06 稽核，見第 9 節 (g)）。** 樣本宇宙要求訊號日之後 14 個交易日無缺席且有成交，這個算標籤必要的條件也套在預測列與回測候選上，所以接下來 14 天會停牌、下市或零成交的股票當天就不在候選名單裡（2023 年起 4.36% 的股票日，被剔除列後 10 日報酬平均 −0.47%，未剔除 +0.30%）。其餘：特徵、籌碼、標籤、訓練切分、策略輸入都只用訊號日之前（籌碼：次日開盤前）可得的資料；延伸沿用的折與舊 launch 逐位元相同。

**設計層的選擇偏誤：有。** 門檻篩選規則與「七個模型一起篩」是看過第一版樣本外結果之後定的，而第一版的樣本外（2026-06 至 10-02）與這一輪的樣本外（07-01 至 10-01）大部分重疊，所以**這一輪的樣本外對規則不是乾淨的，對重訓後的分數是未看過的**。這就是六格不直接進模擬交易、而是進入完全在資料結束日之後的確認期的原因。21 格的多重比較下「碰巧每段都過 1.5」的機率比第一版高，門檻沒有因此調整（調整本身就是事後決定）。

### 9. 限制（事前聲明，不是事後找理由）

(a) 還原價不含分割、減資與面額變更（2023 年起 100 次大跳動，98 次被停牌規則排除，另 2 檔不在樣本內）。(b) 樣本外只有 64 個交易日、一種市場狀態。(c) 籌碼公布時刻用保守規則；外資持股比例 z 分數欄恆缺值。(d) 放空腿的借不到機率、召回與借券費分層是參數假設；主表不含處置股與停券強制回補的約束，7.5 的兩條規則仍偏樂觀；資本情境的成交價修正（7.6）所用的衝擊係數與開盤量占比沒有台股實證、只有括弧，停損變體的超越量亦同；7.7 的容量結論取決於衝擊係數與 5% 參與率上限。(e) transformer 的結果不保證逐次重現。(f) 分點與集保資料未購買。(g) **2026-11-06 稽核的三項重大問題**：選樣的前瞻（往後 14 日條件套在候選名單上，4.36% 股票日，影響估計在 1 個百分點以內，修法排在確認期第一次延伸）；多空書的淨報酬是多頭書 + 空頭書、每 1 元資金 2 元毛曝險，保證金未計，年化與回撤是 2 倍曝險下的數字；資本情境在上市與上櫃各算一次全部資金，TW 列的「3 億」實為兩市場各 3 億，口徑決定後重跑。分數位置的 Sharpe 與確認期六格不受影響。

### 10. 下一步

1. 六個通過的格進入 2026-10-23 起 12 個月的確認期，各自判定：淨報酬 > 0 且回撤不深於各自樣本內的兩倍（F2 × S2 −16.2%、F2 × S1 −22.0%、F0 × S2 −17.2%、D0 × S2 −16.4%、D1-Chronos × S2 −18.6%、D1-Kronos × S-A −47.8%）；每季以延伸方式補新折、不重訓；2027-04-22 期中只報不判；期間內不改任何設定。
2. 上櫃空頭腿在樣本外的虧損歸因。
2b. 執行層寫實化一律登記為並列敏感度（11 月 4 日裁定）；確認期的每季讀數同時列 D + C、F + L 與全部保守假設下的數字。
2c. 容量：方式 B 下 3 億仍不達標，要管理 3 億以上必須擴大 K、降低換手或改宇宙（新策略）；多空＋避險的區間約 3,000 萬、純多頭約 1 億；停損不採用，若要風控改測組合層級的回撤降倉。
3. 事件研究（處置、注意股、當沖比跳升）；transformer 在籌碼特徵上重訓；券商分點資料——都要先登記再做，由使用者決定。

### 附錄：本文名稱與程式庫內部代號

| 本文名稱 | 程式庫代號 | 正本 launch |
| --- | --- | --- |
| B0 | `t-lgbm-alpha` × `alpha_tw_v1` | `launch-20261101T212237Z-2508230824` |
| F0 | `t-lgbm-alpha-residual_v1a` × `alpha_tw_v1` | `launch-20261101T220120Z-29216090` |
| 變體 B | `t-lgbm-alpha-residual_v1b` × `alpha_tw_v1` | `launch-20261101T221954Z-1215615947` |
| F2 | `t-lgbm-alpha-residual_v1a` × `alpha_chip_tw_v1` | `launch-20261101T223737Z-1940816181` |
| D0 | `t-xs-scratch` | `launch-20261102T063519Z-189413204` |
| D1-Kronos | `t-xs-kronos-small` | `launch-20261102T071451Z-322624982` |
| D1-Chronos | `t-xs-chronos2-small` | `launch-20261102T073405Z-1322713845` |
| S1、S2、S-A | `md_ls`、`md_ls_tx`、`wk_long` | 回測 `artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z/`、篩選 `strategy-decision-p3b-21cells-20261102T082429Z/` |
| 規則 | 研究協定 §13.20 補充一、§13.23 第 5 項與第 18 步實作註記、§13.22 第 2、3、5 項 | `docs/experiment_protocol.md` |

---

## English

### 1. One-page summary

- **Question**: can a cross-sectional ranking model on daily price-volume plus free chip data become a strategy fit for paper trading under Taiwan's trading costs and execution constraints?
- **What this round did**: after fixing the adjusted prices (paid rights issues), all seven models were retrained on the full 2016 to 2026-10-23 history; seven models × three strategies = 21 cells were screened together by one pre-registered threshold rule; every shortlisted cell was verified out of sample on its own.
- **Model level**: the price-volume + chip LightGBM (F2) is the strongest model in every period (in-sample 5-to-10-day Rank IC 0.109, +0.019 over price-volume only). The three cross-stock transformers beat the price-volume LightGBM slightly at the short horizon and not at 5 to 10 days.
- **Threshold shortlist** (institution primary cost, TWSE and TPEX half each, fractional capital; net Sharpe ≥ 1.5 in the first half, the second half and the whole, whole-period lower bound > 0): **7 of 21 cells shortlisted**.
- **Out-of-sample verification (2026-07-01 to 10-01, 64 sessions; net return > 0 and drawdown within twice the in-sample drawdown)**: **6 of 7 pass**. The strongest cell, **F2 × long-short + TX hedge**: in-sample Sharpe 2.17 / 2.84 / 2.38 and +30.2% a year; out of sample **+19.6%**, Sharpe 1.48, drawdown −8.0%. The one failure is F2 × weekly long (−13.4% out of sample).
- **Disposition**: the six passing cells enter a **12-month confirmation period from 2026-10-23**, with rules registered in advance and frozen; each is judged at the end for paper trading. No strategy enters paper trading before the confirmation period.
- **Execution-layer sensitivity added on 4 November (§7.5)**: with "no new entry during a disposition period" and "forced cover when short selling is suspended", F2 × long-short + hedge goes from Sharpe 2.38 to 1.70 and from +30.2% to +22.6% a year over the whole period while its out-of-sample +20.4% is unchanged; the fall exceeds the pre-registered expectation, so this conservative setting becomes the reference for the quarterly confirmation readings. Of the six cells F2 × S2, D0 × S2 and D1-Kronos × S-A still clear the thresholds, F2 × S1, F0 × S2 and D1-Chronos × S2 do not and carry a note; under retail costs only D1-Kronos × S-A passes.
- **The fill correction (6 November, §7.6)**: orders are market orders in the opening call auction, so capital-scenario fills become "open × (1 ± own impact)" and the numbers are corrected accordingly: retail NT$10M F2 × S2 whole period +21.5% (Sharpe 1.72), institution NT$300M −16.6% and NT$1B −38.9% — under "everything at the open" this strategy's capacity is far below NT$300M. The fractional row (Q → 0) is unchanged, so the six confirmation-period cells are unaffected.
- **Strategy variant: the 15% trailing stop (§7.6)**: against the no-stop version it is worse in all seven cells and makes no drawdown shallower (F2 × S2 fractional whole-period Sharpe 2.38 → 1.30); the stop version does not enter the confirmation period.
- **Audit (6 November)**: the conditions and assumptions of the 21 cells are listed in `docs/assumptions_audit_20261106.md`; three major findings (sample-selection look-ahead, undeclared 2× exposure of the long-short book, capital scenarios counted once per market) in section 9 (g); the earlier "data leakage: none" is replaced by the disclosure.
- **Policy B and capacity (6 November, §7.7)**: with orders worked through the session the delay cost is about 5 points (F2 × S2 fractional 2.38 → 2.01), but NT$300M still ends at +9.0% / 0.80; the capacity is about NT$30M (F2 × S2 retail NT$10M +24.1% / 1.94, institution NT$30M 1.59, NT$100M 1.13) and about NT$100M for long-only, a limit of the small-and-mid-cap universe and the turnover structure.
- **Limitations to keep in mind**: the candidate set and thresholds were set after the first edition's out-of-sample results, so this round's out-of-sample period is clean only for the retrained scores; 64 sessions give intervals ±50 to ±120 points wide; every long-short cell's out-of-sample return came from TWSE while TPEX was negative.

### 2. Research question

Rank the stocks of one market on one day from data available after the close, trade at the next open, hold for days to weeks. The aim is not to call the index but to tell which stocks are relatively strong or weak on the same day. The first edition narrowed the problem to this through seven stages in thirteen days; this round fixed the data, retrained every model, widened the candidate set and did it again.

### 3. Data and the rules against look-ahead

| Item | Setting |
| --- | --- |
| Data | Daily price-volume of every TWSE and TPEX stock, 2016-01-01 to 2026-10-23 (revision v1.1: ex-dividend / ex-rights factors only, paid rights issues adjusted on the exchange's opening basis); twelve official chip items, each row stamped with its publication time |
| Point-in-time rules | Features use only data before the signal day; chip rows only if published before the next open; each quarter's model trains only on data up to 15 sessions before the quarter starts |
| Training | Retrained each quarter on an expanding window from 2016; 16 folds from 2023Q1 to 2026Q4; each fold keeps the model with the best validation Rank IC |
| In sample | First half 2023-01-02 to 2025-06-30, second half 2025-07-01 to 2026-06-30 |
| Out of sample | 2026-07-01 to 2026-10-01 (64 open-to-open intervals) |
| Confirmation period | 12 months from 2026-10-23 (entirely after the data end) |

### 4. The seven models

| Code | Model | Features | Label |
| --- | --- | --- | --- |
| B0 | LightGBM | 183 price-volume | Same-day, same-market rank |
| F0 | LightGBM | 183 price-volume | Residual return a (market, size and industry removed) |
| Variant B | LightGBM | 183 price-volume | Residual return b |
| **F2** | LightGBM | 183 price-volume + 67 chip | Residual return a |
| D0 | Cross-stock transformer (from scratch, three-seed ensemble) | 183 price-volume | Residual return a |
| D1-Kronos | Cross-stock transformer, Kronos-small encoder | as above | as above |
| D1-Chronos | Cross-stock transformer, Chronos-2-small encoder | as above | as above |

In-sample Rank IC (short / 5-to-10-day / long): B0 0.117 / 0.087 / 0.086, F0 0.117 / 0.090 / 0.090, variant B 0.113 / 0.085 / 0.083, **F2 0.129 / 0.109 / 0.105**, D0 0.121 / 0.095 / 0.096, D1-Kronos 0.121 / 0.094 / 0.092, D1-Chronos 0.121 / 0.093 / 0.092. Model-level numbers are informational and play no part in the shortlist.

### 5. The three trading strategies

Scores are percentiled inside the liquid universe → smoothed with a 3-day half-life → residualized daily on 60-day beta and 20- and 60-day momentum (shared by all three strategies).

| Strategy | Rule | Parameters (fixed in advance) | Exposure |
| --- | --- | --- | --- |
| S1 long-short | 10-day score; long band (buy the top K, sell only when a name drops out of the top m × K, at most K / 10 entries a day); the short band uses the reverse ranking under the borrowable list, exchange limits, borrow-failure probability and tiered borrow fees | K = 50, m = 3, the same for all seven models | Dollar-neutral, gross 2× |
| S2 long-short + TX hedge | S1 plus TX futures at −β × capital, β the 60-day rolling beta of the book's gross return on the index | as above | Dollar-neutral plus beta hedge |
| S-A weekly low-turnover long | 5-day score; rebalance on the last session of each ISO week; top-K long band, sell only when a name drops out of the top m × K; at most 50% × K replaced a week | (K, m) selected per model on TWSE in the first half: F2 (30, 3), D1-Kronos (30, 3), the rest in the full report | 100% long |

### 6. Cost and execution assumptions

Institution: commission 0.03%, slippage 0.05%, borrow fee 1.5% / 3% / 6% by size; retail: commission 0.0855%, slippage 0.1% / 0.2%, short fee 0.08%; transaction tax 0.3%. Shorts only inside the borrowable list (coverage TWSE 97.6%, TPEX 92.6%), capped by the next-day limit, borrow failure 2% / 10% / 25% by size, average of five random repeats. Capital: fractional; institution 300M and 1B TWD. The main tables carry no disposition-stock or forced-cover constraint; those two rules are a sensitivity registered on 4 November, see 7.5. Fills: the fractional row is the open (Q → 0); capital rows follow the actual order type (market orders in the opening call) with own impact, ε = Y × σ_daily × √(Q ÷ (f × V)), Y 0.7, f 10% (uncalibrated, see 7.6). The main strategy has no stop; the 15% trailing stop is a separate strategy variant (7.6).

### 7. Results

#### 7.1 In sample: the 21-cell threshold shortlist (institution, TWSE and TPEX half each, fractional)

| Model × strategy | P1 Sharpe | P2 Sharpe | Whole Sharpe | Whole annual net | 95% interval | Whole drawdown | Shortlisted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | 2.17 | 2.84 | **2.38** | +30.2% | [+18.5%, +43.0%] | −8.1% | **yes** |
| F2 × S-A | 1.54 | 3.79 | 2.33 | +53.4% | [+28.1%, +82.9%] | −25.1% | **yes** |
| D1-Kronos × S-A | 1.56 | 3.36 | 2.16 | +44.2% | [+22.4%, +67.7%] | −23.9% | **yes** |
| F2 × S1 | 1.90 | 2.58 | 2.11 | +25.5% | [+13.4%, +39.2%] | −11.0% | **yes** |
| D1-Chronos × S2 | 1.69 | 2.10 | 1.81 | +24.4% | [+11.2%, +39.0%] | −9.3% | **yes** |
| D0 × S2 | 1.62 | 2.21 | 1.80 | +24.5% | [+11.1%, +37.8%] | −8.2% | **yes** |
| F0 × S2 | 1.53 | 1.91 | 1.64 | +22.3% | [+8.2%, +36.6%] | −8.6% | **yes** |
| The other 14 cells | — | — | 0.80 to 2.17 | — | — | — | no |

Among the 14 that fall out, five weekly longs sit at a whole-period Sharpe of 1.95 to 2.17 but only 1.44 to 1.48 in the first half — a high Sharpe resting almost entirely on the second half, exactly the pattern the thresholds are meant to stop; the rest are long-short cells below 1.5 in one period (full table in the full report, section 6.2).

#### 7.2 Out of sample (2026-07-01 to 10-01, 64 sessions): the 7 shortlisted cells

Rule: (i) annual net return > 0, (ii) maximum drawdown not deeper than twice the cell's whole-period drawdown.

| Shortlisted cell | OOS annual net | 95% interval | Sharpe | OOS drawdown | Twice limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | **+19.6%** | [−38.2%, +87.8%] | 1.48 | −8.0% | −16.2% | **pass** |
| F2 × S1 | +14.2% | [−42.1%, +81.4%] | 1.06 | −8.3% | −22.0% | **pass** |
| D1-Kronos × S-A | +4.8% | [−57.6%, +63.2%] | 0.16 | −16.1% | −47.8% | **pass** |
| D1-Chronos × S2 | +14.5% | [−49.0%, +91.8%] | 1.03 | −10.0% | −18.6% | **pass** |
| D0 × S2 | +15.7% | [−38.2%, +77.9%] | 1.16 | −7.8% | −16.4% | **pass** |
| F0 × S2 | +7.6% | [−33.4%, +52.6%] | 0.60 | −7.0% | −17.2% | **pass** |
| F2 × S-A | −13.4% | [−120.3%, +75.8%] | −0.33 | −24.6% | −50.2% | fail (net return < 0) |

By market: TWSE F2 × S2 +47.3% (Sharpe 2.54), D0 × S2 +34.6%, F0 × S2 +27.3%; every long-short cell is negative on TPEX (F2 × S2 −8.1%). Comparison columns: equal-weight liquid universe +5.4% (drawdown −22.8%), 0050 +14.8%, TAIEX total return +14.0%.

#### 7.3 Retail costs and capital (F2 × S2)

Capital-scenario fills follow the actual order type (corrected on 6 November, 7.6): market orders in the opening call auction, fill = open × (1 ± own impact). Retail fractional (fill = the open): whole period +29.1% (2.26), out of sample +24.0% (1.73). Retail NT$10M: whole +21.5% (1.72), out of sample -8.3% (-0.56). Institution NT$300M: whole -16.6% (-1.45), out of sample -58.2% (-3.91); NT$1B: whole -38.9% (-4.77). Slippage × 0 / × 1 / × 2 (fractional): +31.6% / +30.2% / +28.7%. Borrow failure × 0 to × 2 with 0.5% recalls (fractional): +27.8% to +30.9%. By year (fractional): 2023 +17.3%, 2024 +34.2%, 2025 +21.8%, 2026 (to 10-01) +40.6%, no yearly drawdown deeper than −8%. Judging the retail fractional row by the same thresholds (Sharpe ≥ 1.5 in all three periods, whole-period lower bound > 0): F2 × S2 (1.95 / 2.94 / 2.26), F2 × S1 (1.67 / 2.63 / 1.98) and D1-Chronos × S2 (1.54 / 2.22 / 1.75) are shortlisted, all three with a positive out-of-sample return and a drawdown within twice; D0 × S2, F0 × S2 and D1-Kronos × S-A drop out on a first-period Sharpe below 1.5. The impact magnitude and capacity are in 7.6 and 7.7.

#### 7.4 How to read it

- With 64 sessions the intervals are ±50 to ±120 points wide; the out-of-sample period can only rule out "the strategy is broken", not show how good it is. F2 × S2 is the only cell whose Sharpe is clearly above 1 in all four periods.
- Every long-short cell's out-of-sample return came from TWSE while TPEX was negative (the TPEX short leg suffered in the July-to-September rebound); the confirmation period judges on the two markets half each.
- The long-only weekly strategy again took the full drawdown of the July breadth correction; D1-Kronos × S-A passes only because its in-sample drawdown was already deep (−23.9%) and the two-times limit is wide.
- The chip increment at the strategy level (F2 − F0 paired difference, whole period annualized): S1 +9.6% [+1.6%, +18.1%] and S-A +12.9% [+4.2%, +23.1%] are claimed; S2 +7.8% [−0.8%, +16.3%] is not.

#### 7.5 Execution-layer sensitivity added on 2026-11-04: no new entries during a disposition period, forced cover when short selling is suspended

The main tables treat disposition stocks as trading normally and give held shorts only the random-recall sensitivity. Two deterministic rules were registered and implemented on 4 November (protocol §13.23 supplement 3): **D** = a stock whose next open lies inside an announced disposition period gets no new entry in either leg; **C** = a held short is covered at the open after the stock is suspended from margin short selling, and no new short is opened during the suspension (the retail margin rule; the institution reads the same flag as an upper bound). Under D + C, F2 × S2 skips about 9% of its candidates across both legs and is force-covered about 400 times over the 811 intervals.

Institution primary cost (annualized net return / Sharpe / max drawdown; whole = P1 ∪ P2, OOS = the 64 intervals of P3; TW half each, fractional, primary slippage):

| Cell | Whole baseline | Whole D | Whole C | Whole D + C | OOS baseline | OOS D + C | D + C reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +30.2% / 2.38 / -8.1% | +25.7% / 1.92 / -8.1% | +26.7% / 2.12 / -8.1% | **+22.6% / 1.70 / -8.3%** | +19.6% / 1.48 / -8.0% | **+20.4% / 1.39 / -8.3%** | pass |
| F2 × S1 | +25.5% / 2.11 / -11.0% | +18.9% / 1.49 / -11.2% | +22.1% / 1.85 / -11.1% | **+16.6% / 1.32 / -11.5%** | +14.2% / 1.06 / -8.3% | **+16.0% / 1.13 / -9.6%** | fail (Sharpe < 1.5) |
| D0 × S2 | +24.5% / 1.80 / -8.2% | +22.3% / 1.65 / -9.5% | +21.9% / 1.63 / -8.1% | **+20.3% / 1.51 / -9.3%** | +15.7% / 1.16 / -7.8% | **+3.2% / 0.23 / -10.4%** | pass |
| D1-Chronos × S2 | +24.4% / 1.81 / -9.3% | +23.3% / 1.73 / -8.8% | +24.6% / 1.86 / -8.6% | **+21.4% / 1.61 / -8.4%** | +14.5% / 1.03 / -10.0% | **-2.1% / -0.14 / -10.9%** | fail (OOS < 0) |
| F0 × S2 | +22.3% / 1.64 / -8.6% | +20.1% / 1.46 / -8.7% | +20.4% / 1.54 / -9.2% | **+19.0% / 1.41 / -9.2%** | +7.6% / 0.60 / -7.0% | **+26.8% / 1.87 / -8.5%** | fail (Sharpe < 1.5) |
| D1-Kronos × S-A | +44.2% / 2.16 / -23.9% | +40.2% / 2.06 / -24.7% | +44.2% / 2.16 / -23.9% | **+40.2% / 2.06 / -24.7%** | +4.8% / 0.16 / -16.1% | **+17.7% / 0.59 / -11.7%** | pass |
| F2 × S-A (not in confirmation) | +53.4% / 2.33 / -25.1% | +38.3% / 1.97 / -24.3% | +53.4% / 2.33 / -25.1% | **+38.3% / 1.97 / -24.3%** | -13.4% / -0.33 / -24.6% | **+5.7% / 0.22 / -13.6%** | pass |

Retail cost (annualized net return / Sharpe / max drawdown; whole = P1 ∪ P2, OOS = the 64 intervals of P3; TW half each, fractional, primary slippage):

| Cell | Whole baseline | Whole D | Whole C | Whole D + C | OOS baseline | OOS D + C | D + C reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1% / 2.26 / -7.3% | +24.9% / 1.85 / -7.7% | +24.7% / 1.96 / -8.6% | **+19.4% / 1.45 / -8.7%** | +24.0% / 1.73 / -7.9% | **+22.5% / 1.51 / -8.4%** | fail (Sharpe < 1.5) |
| F2 × S1 | +24.2% / 1.98 / -10.5% | +18.5% / 1.44 / -10.6% | +20.3% / 1.70 / -11.5% | **+13.4% / 1.06 / -11.0%** | +19.8% / 1.43 / -8.4% | **+18.3% / 1.28 / -9.8%** | fail (Sharpe < 1.5) |
| D0 × S2 | +22.2% / 1.63 / -9.5% | +20.7% / 1.52 / -10.4% | +19.0% / 1.41 / -9.4% | **+15.6% / 1.17 / -9.9%** | +19.5% / 1.40 / -8.0% | **-1.0% / -0.07 / -10.6%** | fail (Sharpe < 1.5) |
| D1-Chronos × S2 | +23.3% / 1.75 / -9.1% | +19.9% / 1.47 / -9.4% | +18.5% / 1.40 / -8.6% | **+17.0% / 1.29 / -9.7%** | +3.8% / 0.27 / -11.9% | **-0.9% / -0.06 / -11.4%** | fail (Sharpe < 1.5) |
| F0 × S2 | +19.5% / 1.40 / -8.5% | +18.9% / 1.36 / -8.7% | +18.4% / 1.37 / -9.5% | **+16.3% / 1.19 / -9.7%** | +3.1% / 0.24 / -8.3% | **+21.9% / 1.51 / -8.1%** | fail (Sharpe < 1.5) |
| D1-Kronos × S-A | +41.2% / 2.02 / -24.4% | +37.2% / 1.91 / -26.4% | +41.2% / 2.02 / -24.4% | **+37.2% / 1.91 / -26.4%** | +1.5% / 0.05 / -16.3% | **+14.6% / 0.48 / -12.0%** | pass |
| F2 × S-A (not in confirmation) | +49.7% / 2.18 / -25.3% | +35.4% / 1.82 / -24.6% | +49.7% / 2.18 / -25.3% | **+35.4% / 1.82 / -24.6%** | -17.9% / -0.45 / -24.7% | **+1.2% / 0.05 / -13.8%** | pass |

The fall exceeds the pre-registered expectation (F2 × S2 whole period −7.5 points / Sharpe −0.68), so by the registered rule **D + C becomes the conservative reference for every quarterly confirmation reading**: of the six cells F2 × S2, D0 × S2 and D1-Kronos × S-A still clear the thresholds under it, **F2 × S1, F0 × S2 and D1-Chronos × S2 do not and now carry the note "execution-layer sensitivity not passed"**; among retail cells only D1-Kronos × S-A passes. The effect comes mainly from the disposition rule and shows in both legs: the ranking models concentrate both ends of the ranking in stocks that are subsequently placed under disposition, a sizeable part of the P2 long-side return came from them (F2 × S-A's P2 falls from +114% to +65%), and out of sample it goes the other way (D turns F2 × S-A's OOS from −13.4% to +5.7%). Both rules remain optimistic (D forces no exit and adds no slippage; C is at least one day later than the real last cover day); the six confirmation-period cells and the frozen configuration are unchanged, and whether D + C becomes the primary assumption is the user's decision.

#### 7.6 The fill correction (market orders in the opening call) and a strategy variant: the 15% trailing stop (2026-11-06)

**The fill correction.** Orders are market orders in the opening call auction: a single-price match with no spread, whose cost is how far one's own order moves the matched price. The backtest's fill therefore becomes "open × (1 ± Y × σ_daily × √(Q ÷ (f × V)))", Y 0.7, f (the auction's share of daily volume) 10%; no Taiwan evidence exists for either, and the brackets (Y 0.5 / 1.0, f 5% / 20%) are labelled uncalibrated. This is a correction, not a sensitivity: capital-scenario numbers are always computed with the corrected fill; the fractional row is unchanged because Q → 0, so the six confirmation-period cells are unaffected. The universe's median 20-session traded value is only NT$69M; NT$300M across 50 names pushes 61% of positions to the 5% participation cap with a median impact of 0.85% per side; NT$10M pays 0.25%.

Corrected capital scenarios (annualized net return (Sharpe)):

| Cell | Retail NT$10M whole / OOS | Institution NT$300M whole / OOS | Institution NT$1B whole / OOS |
| --- | --- | --- | --- |
| F2 × S2 | +21.5% (1.72) / -8.3% (-0.56) | -16.6% (-1.45) / -58.2% (-3.91) | -38.9% (-4.77) / -108.2% (-7.35) |
| F2 × S1 | +18.5% (1.51) / -3.8% (-0.21) | -21.7% (-1.95) / -61.7% (-4.07) | -37.3% (-4.52) / -108.2% (-6.98) |
| D0 × S2 | +11.4% (0.86) / -10.3% (-0.81) | -30.0% (-2.58) / -40.8% (-2.63) | -54.0% (-6.37) / -84.3% (-5.74) |
| D1-Chronos × S2 | +11.5% (0.89) / -16.3% (-1.07) | -29.6% (-2.48) / -58.2% (-3.91) | -50.1% (-5.73) / -83.7% (-5.57) |
| F0 × S2 | +14.5% (1.09) / -7.2% (-0.54) | -30.5% (-2.65) / -82.9% (-6.10) | -49.8% (-5.92) / -102.6% (-6.13) |
| D1-Kronos × S-A | +33.2% (1.65) / -5.5% (-0.19) | +2.1% (0.14) / -19.3% (-0.58) | -13.5% (-0.87) / -36.0% (-1.27) |
| F2 × S-A (not in confirmation) | +36.5% (1.79) / -44.7% (-1.24) | +3.0% (0.18) / -67.5% (-2.05) | -15.1% (-0.91) / -99.4% (-2.90) |

**Strategy variant: the 15% trailing stop.** A held position leaves intraday once its value falls 15% from its running high since entry, with the fill's mass at the stop price and a right tail toward the low (half a tick + impact + c × (stop − today's low), c 0.25). Both strategies use the corrected fill (annualized net return / Sharpe / max drawdown; whole = P1 ∪ P2, OOS = P3; TW half each, primary slippage):

| Cell | Profile / capital | Period | No stop | 15% trailing stop |
| --- | --- | --- | --- | --- |
| F2 × S2 | institution fractional | P1+P2 | +30.2% / 2.38 / -8.1% | +17.0% / 1.30 / -8.6% |
| F2 × S2 | institution fractional | P3 | +19.6% / 1.48 / -8.0% | +23.1% / 1.64 / -8.1% |
| F2 × S2 | retail 10M | P1+P2 | +21.5% / 1.72 / -11.1% | +5.5% / 0.43 / -15.8% |
| F2 × S2 | retail 10M | P3 | -8.3% / -0.56 / -5.9% | -40.3% / -3.15 / -11.0% |
| F2 × S1 | institution fractional | P1+P2 | +25.5% / 2.11 / -11.0% | +13.1% / 1.05 / -11.0% |
| F2 × S1 | institution fractional | P3 | +14.2% / 1.06 / -8.3% | +12.3% / 0.85 / -10.0% |
| F2 × S1 | retail 10M | P1+P2 | +18.5% / 1.51 / -13.5% | +4.0% / 0.32 / -20.3% |
| F2 × S1 | retail 10M | P3 | -3.8% / -0.21 / -6.8% | -40.9% / -2.91 / -11.9% |
| D0 × S2 | institution fractional | P1+P2 | +24.5% / 1.80 / -8.2% | +13.7% / 1.04 / -10.8% |
| D0 × S2 | institution fractional | P3 | +15.7% / 1.16 / -7.8% | -0.4% / -0.03 / -8.0% |
| D0 × S2 | retail 10M | P1+P2 | +11.4% / 0.86 / -13.0% | -2.5% / -0.20 / -23.7% |
| D0 × S2 | retail 10M | P3 | -10.3% / -0.81 / -5.9% | -46.0% / -4.60 / -11.8% |
| D1-Chronos × S2 | institution fractional | P1+P2 | +24.4% / 1.81 / -9.3% | +9.4% / 0.73 / -9.8% |
| D1-Chronos × S2 | institution fractional | P3 | +14.5% / 1.03 / -10.0% | -15.3% / -1.17 / -12.7% |
| D1-Chronos × S2 | retail 10M | P1+P2 | +11.5% / 0.89 / -12.5% | -3.3% / -0.27 / -29.1% |
| D1-Chronos × S2 | retail 10M | P3 | -16.3% / -1.07 / -6.0% | -31.5% / -2.64 / -8.9% |
| F0 × S2 | institution fractional | P1+P2 | +22.3% / 1.64 / -8.6% | +10.2% / 0.78 / -9.4% |
| F0 × S2 | institution fractional | P3 | +7.6% / 0.60 / -7.0% | -12.0% / -0.85 / -11.5% |
| F0 × S2 | retail 10M | P1+P2 | +14.5% / 1.09 / -11.0% | -3.8% / -0.29 / -22.6% |
| F0 × S2 | retail 10M | P3 | -7.2% / -0.54 / -7.4% | -33.5% / -2.98 / -12.3% |
| D1-Kronos × S-A | institution fractional | P1+P2 | +44.2% / 2.16 / -23.9% | +37.2% / 1.97 / -29.2% |
| D1-Kronos × S-A | institution fractional | P3 | +4.8% / 0.16 / -16.1% | +2.8% / 0.11 / -9.1% |
| D1-Kronos × S-A | retail 10M | P1+P2 | +33.2% / 1.65 / -28.6% | +27.0% / 1.50 / -33.9% |
| D1-Kronos × S-A | retail 10M | P3 | -5.5% / -0.19 / -15.8% | -29.5% / -1.51 / -14.0% |
| F2 × S-A (not in confirmation) | institution fractional | P1+P2 | +53.4% / 2.33 / -25.1% | +45.0% / 2.22 / -26.1% |
| F2 × S-A (not in confirmation) | institution fractional | P3 | -13.4% / -0.33 / -24.6% | -22.7% / -0.78 / -19.6% |
| F2 × S-A (not in confirmation) | retail 10M | P1+P2 | +36.5% / 1.79 / -24.7% | +30.0% / 1.63 / -28.6% |
| F2 × S-A (not in confirmation) | retail 10M | P3 | -44.7% / -1.24 / -26.1% | -40.7% / -1.61 / -19.9% |

- The stop version is worse in all seven cells and makes no drawdown shallower: F2 × S2 fractional whole-period Sharpe 2.38 → 1.30 and return +30.2% → +17.0%, mean holding 36 → 20 sessions with turnover doubled; the overshoot is only a third of the cost, the rest is turnover and rebounds after stopping at lows; the long-only cells' drawdowns get deeper. Only F2 × S2's out-of-sample return rises slightly (+23.1%). The stop version does not enter the confirmation period.
- All parallel rules together (disposition, suspension and the stop): institution fractional F2 × S2 whole +10.9% / 0.85 / -11.9%, out of sample +17.6% / 1.30 / -7.1%.

#### 7.7 Policy B: orders worked through the session, and capacity (2026-11-06)

The corrected policy A (7.6) showed NT$300M to be infeasible, so the user registered policy B as a capacity assessment: the open as the arrival price, orders worked through the session, fill = open × exp(θ × the day's open-to-close return) × (1 ± half a tick ± Y σ_daily √(Q ÷ V)), θ 0.5, Y 0.7, impact over the full day's volume, no stop, an institution NT$3B scenario added.

Fractional and capital rows (F2 × S2 and the long-only cell; the rest in the full report §6.9; policy A's fractional fill is the open, so the difference to B is the delay cost) (annualized net return / Sharpe / max drawdown; whole = P1 ∪ P2, OOS = P3; TW half each, primary slippage; baseline = a reference run with the same cost-profile copy and no rule):

| Row | Period | Policy A (corrected) | Policy B | B + D + C |
| --- | --- | --- | --- | --- |
| F2 × S2 institution fractional | P1+P2 | +30.2% / 2.38 / -8.1% | +25.4% / 2.01 / -9.8% | +17.9% / 1.34 / -10.1% |
| F2 × S2 institution fractional | P3 | +19.6% / 1.48 / -8.0% | +19.5% / 1.50 / -8.2% | +17.7% / 1.21 / -8.8% |
| F2 × S2 retail NT$10M | P1+P2 | +21.5% / 1.72 / -11.1% | +24.1% / 1.94 / -10.9% | +17.8% / 1.50 / -9.6% |
| F2 × S2 institution NT$300M | P1+P2 | -16.6% / -1.45 / -46.8% | +9.0% / 0.80 / -9.9% | -0.1% / -0.00 / -12.9% |
| F2 × S2 institution NT$1B | P1+P2 | -38.9% / -4.77 / -71.9% | -4.1% / -0.53 / -22.6% | -10.5% / -1.41 / -30.2% |
| F2 × S2 institution NT$3B | P1+P2 | (not run under A) | -15.1% / -2.48 / -39.4% | -20.9% / -3.50 / -49.4% |
| D1-Kronos × S-A institution NT$300M | P1+P2 | +2.1% / 0.14 / -37.7% | +20.3% / 1.30 / -25.7% | +16.3% / 1.14 / -25.3% |
Capacity (the largest capital scenario with whole-period net Sharpe ≥ 1.5 under policy B):

| Cell | Fractional | 300M | 1B | 3B | Capacity (Sharpe ≥ 1.5) | Retail 10M |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| F2 × S2 | 2.01 | 0.80 | -0.53 | -2.48 | fractional only | 1.94 |
| F2 × S1 | 1.73 | 0.36 | -0.32 | -1.63 | fractional only | 1.73 |
| D0 × S2 | 1.40 | -0.01 | -1.38 | -3.83 | none | 1.10 |
| D1-Chronos × S2 | 1.39 | -0.03 | -1.13 | -3.84 | none | 1.15 |
| F0 × S2 | 1.25 | -0.00 | -1.15 | -4.18 | none | 1.33 |
| D1-Kronos × S-A | 2.01 | 1.30 | 0.99 | -0.12 | fractional only | 1.78 |
| F2 × S-A (not in confirmation) | 2.11 | 1.58 | 1.01 | 0.10 | 300M | 1.97 |

- The delay cost is about 5 points and insensitive to the execution time (θ 0.25 to 0.75 spans 1.5 points); the signal does not realize at the opening print; the "policy-B delay-cost sensitivity" passes.
- NT$300M still falls short (F2 × S2 +9.0% / 0.80 against the corrected policy A's −16.6%; short under the whole literature range of Y), NT$1B and NT$3B are negative with deployment below half; the cause is the small-and-mid-cap universe and the two-leg 50-name turnover structure, with impact scaling as √(Q ÷ V). Retail NT$10M works (F2 × S2 +24.1% / 1.94); with the added NT$30M and NT$100M rows, the capacity is about NT$30M for long-short + hedge (institution NT$30M +21.3% / 1.59; NT$100M +14.7% / 1.13) and about NT$100M for long-only (D1-Kronos × S-A 1.55, F2 × S-A 1.71; F2 × S-A still 1.58 at NT$300M).
- All parallel readings; the six confirmation-period cells are unchanged; managing NT$300M or more needs a larger K, lower turnover or another universe, i.e. a new strategy.

### 8. How the samples were used: leakage and design bias, stated plainly

**Data leakage: none in the features, chip rows, labels, training splits or strategy inputs; one look-ahead in sample selection (2026-11-06 audit, see section 9 (g)).** The sample universe requires 14 sessions after the signal date without a missing session or zero volume; that label-necessary condition is also applied to the prediction rows and backtest candidates, so a stock that will halt, delist or print zero volume within 14 sessions is absent from the day's candidate list (4.36% of stock-days since 2023; excluded rows' next-10-session return −0.47% on average, included +0.30%). Everything else: features, chip rows, labels, training splits and strategy inputs use only data available before the signal day (chip rows: before the next open); the folds carried over by extension are byte-identical to the old launches.

**Design-level selection bias: present.** The threshold rule and "screen all seven models together" were set after the first edition's out-of-sample results had been seen, and the first edition's out-of-sample period (2026-06 to 10-02) overlaps most of this round's (07-01 to 10-01), so **this round's out-of-sample period is not clean for the rules while it is unseen for the retrained scores**. That is why the six cells do not go straight to paper trading but into a confirmation period entirely after the data end. Under 21-cell multiple comparison the chance that some cell clears 1.5 in every period by luck is higher than in the first edition, and the thresholds were not tightened for it (tightening would itself be an after-the-fact decision).

### 9. Limitations (stated in advance, not found afterwards)

(a) Adjusted prices exclude splits, capital reductions and par-value changes (100 large jumps since 2023, 98 excluded by the halt rule, the other 2 names outside the sample). (b) The out-of-sample period has 64 sessions and one market state. (c) Chip publication times follow a conservative rule; the foreign-holding z-score column is always missing. (d) The short leg's borrow-failure probability, recalls and fee tiers are parameter assumptions; the main tables carry no disposition-stock or forced-cover constraint, and the two rules of 7.5 remain optimistic; the impact coefficient and auction share behind the capital-scenario fill correction (7.6) have no Taiwan evidence, only brackets, and so does the stop variant's overshoot; the capacity verdict of 7.7 rests on the impact coefficient and the 5% participation cap. (e) Transformer results are not reproducible run to run. (f) Broker-branch and depository data were not bought. (g) **Three major findings of the 2026-11-06 audit**: a look-ahead in sample selection (the 14-sessions-forward condition applied to the candidate list, 4.36% of stock-days, estimated effect under one point, fix scheduled with the confirmation period's first extension); the long-short book's net return is long book plus short book, NT$2 of gross exposure per NT$1 of capital with no margin charged, so returns and drawdowns are 2×-exposure figures; capital scenarios count the whole capital once per market, so the TW row's "NT$300M" is NT$300M in each market, to be rerun once the convention is decided. The fractional rows' Sharpe and the six confirmation-period cells are unaffected.

### 10. Next steps

1. The six passing cells enter the 12-month confirmation period from 2026-10-23, each judged on its own: net return > 0 and drawdown within twice its own in-sample drawdown (F2 × S2 −16.2%, F2 × S1 −22.0%, F0 × S2 −17.2%, D0 × S2 −16.4%, D1-Chronos × S2 −18.6%, D1-Kronos × S-A −47.8%); new folds added each quarter by extension, no retraining; the 2027-04-22 interim reading is reported, not judged; no setting changes during the period.
2. Attribution of the TPEX short leg's out-of-sample loss.
2b. Execution realism is always registered as a parallel sensitivity (the 4 November ruling); every quarterly confirmation reading also lists the numbers under D + C, F + L and all rules together.
2c. Capacity: NT$300M still falls short under policy B; managing NT$300M or more needs a larger K, lower turnover or another universe (a new strategy); about NT$30M for long-short + hedge and NT$100M for long-only; no stop loss, and for risk control test portfolio-level drawdown de-leveraging instead.
3. Event studies (disposition, attention stocks, day-trade-ratio jumps); the transformer retrained on chip features; broker-branch data — each registered before it is done, at the user's decision.

### Appendix: names used in this report and the code-base identifiers

| Name here | Code-base identifier | Current launch |
| --- | --- | --- |
| B0 | `t-lgbm-alpha` × `alpha_tw_v1` | `launch-20261101T212237Z-2508230824` |
| F0 | `t-lgbm-alpha-residual_v1a` × `alpha_tw_v1` | `launch-20261101T220120Z-29216090` |
| Variant B | `t-lgbm-alpha-residual_v1b` × `alpha_tw_v1` | `launch-20261101T221954Z-1215615947` |
| F2 | `t-lgbm-alpha-residual_v1a` × `alpha_chip_tw_v1` | `launch-20261101T223737Z-1940816181` |
| D0 | `t-xs-scratch` | `launch-20261102T063519Z-189413204` |
| D1-Kronos | `t-xs-kronos-small` | `launch-20261102T071451Z-322624982` |
| D1-Chronos | `t-xs-chronos2-small` | `launch-20261102T073405Z-1322713845` |
| S1, S2, S-A | `md_ls`, `md_ls_tx`, `wk_long` | Backtest `artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z/`, shortlist `strategy-decision-p3b-21cells-20261102T082429Z/` |
| Rules | Protocol §13.20 supplement 1, §13.23 item 5 and the step 18 note, §13.22 items 2, 3 and 5 | `docs/experiment_protocol.md` |
