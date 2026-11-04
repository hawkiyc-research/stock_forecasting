# 台股量化選股策略（第二版）：七個模型、三種策略、門檻篩選與樣本外驗證 / A Quantitative Taiwan Stock Strategy (second edition): Seven Models, Three Strategies, the Threshold Shortlist and Out-of-Sample Verification

版本：2026-11-02，2026-11-04 補充 §7.5 執行層敏感度並修訂摘要、6、9、10（取代 2026-10-29 的第一版；第一版封存於 `reports/deprecated/pre-1323-20261102/`，11-02 的 Word 版封存於 `reports/deprecated/pre-1323s3-20261104/`）。完整版見 `docs/research_full_report.md`。
Version: 2026-11-02, with §7.5 (execution-layer sensitivity) added on 2026-11-04 and the summary, 6, 9 and 10 revised (supersedes the first edition of 2026-10-29, archived under `reports/deprecated/pre-1323-20261102/`; the 11-02 Word files are archived under `reports/deprecated/pre-1323s3-20261104/`). The full report is `docs/research_full_report.md`.

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

機構：手續費 0.03%、滑價 0.05%、借券費依規模 1.5%／3%／6%；散戶：手續費 0.0855%、滑價 0.1%／0.2%、融券手續費 0.08%；證交稅 0.3%。放空只能在可借券名單內（覆蓋率上市 97.6%、上櫃 92.6%），受次日限額封頂，借不到機率依規模 2%／10%／25%，五次隨機重複平均。資本情境：分數位置；機構 3 億與 10 億。主表沒有處置股與停券強制回補的約束；這兩條規則是 11 月 4 日補登記的敏感度，見 7.5。

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

散戶成本：全段 +29.1%（Sharpe 2.26）、樣本外 +24.0%。機構 3 億：全段 +24.7%（2.20、部署率 71%）、樣本外 +36.4%；10 億：+15.6%（2.01、50%）、樣本外 +9.8%。滑價 0／1／2 倍：+31.6%／+30.2%／+28.7%。借不到倍率 × 0 到 × 2 與召回 0.5%：+27.8% 到 +30.9%。逐年：2023 +17.3%、2024 +34.2%、2025 +21.8%、2026（至 10-01）+40.6%，沒有一年回撤深過 −8%。

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

### 8. 樣本是怎麼用的：洩漏與設計偏誤的交代

**資料洩漏：沒有。** 特徵、籌碼、標籤、訓練切分、策略輸入都只用訊號日之前（籌碼：次日開盤前）可得的資料；延伸沿用的折與舊 launch 逐位元相同。

**設計層的選擇偏誤：有。** 門檻篩選規則與「七個模型一起篩」是看過第一版樣本外結果之後定的，而第一版的樣本外（2026-06 至 10-02）與這一輪的樣本外（07-01 至 10-01）大部分重疊，所以**這一輪的樣本外對規則不是乾淨的，對重訓後的分數是未看過的**。這就是六格不直接進模擬交易、而是進入完全在資料結束日之後的確認期的原因。21 格的多重比較下「碰巧每段都過 1.5」的機率比第一版高，門檻沒有因此調整（調整本身就是事後決定）。

### 9. 限制（事前聲明，不是事後找理由）

(a) 還原價不含分割、減資與面額變更（2023 年起 100 次大跳動，98 次被停牌規則排除，另 2 檔不在樣本內）。(b) 樣本外只有 64 個交易日、一種市場狀態。(c) 籌碼公布時刻用保守規則；外資持股比例 z 分數欄恆缺值。(d) 放空腿的借不到機率、召回與借券費分層是參數假設；主表不含處置股與停券強制回補的約束，7.5 的兩條規則仍偏樂觀。(e) transformer 的結果不保證逐次重現。(f) 分點與集保資料未購買。

### 10. 下一步

1. 六個通過的格進入 2026-10-23 起 12 個月的確認期，各自判定：淨報酬 > 0 且回撤不深於各自樣本內的兩倍（F2 × S2 −16.2%、F2 × S1 −22.0%、F0 × S2 −17.2%、D0 × S2 −16.4%、D1-Chronos × S2 −18.6%、D1-Kronos × S-A −47.8%）；每季以延伸方式補新折、不重訓；2027-04-22 期中只報不判；期間內不改任何設定。
2. 上櫃空頭腿在樣本外的虧損歸因。
2b. 決定是否以「處置期間不新進場＋停券強制回補」為主要執行假設（採用即另案登記、重新起算）；確認期的每季讀數同時列保守假設下的數字。
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

Institution: commission 0.03%, slippage 0.05%, borrow fee 1.5% / 3% / 6% by size; retail: commission 0.0855%, slippage 0.1% / 0.2%, short fee 0.08%; transaction tax 0.3%. Shorts only inside the borrowable list (coverage TWSE 97.6%, TPEX 92.6%), capped by the next-day limit, borrow failure 2% / 10% / 25% by size, average of five random repeats. Capital: fractional; institution 300M and 1B TWD. The main tables carry no disposition-stock or forced-cover constraint; those two rules are a sensitivity registered on 4 November, see 7.5.

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

Retail costs: whole period +29.1% (Sharpe 2.26), out of sample +24.0%. Institution 300M: whole +24.7% (2.20, deployed 71%), out of sample +36.4%; 1B: +15.6% (2.01, 50%), out of sample +9.8%. Slippage × 0 / × 1 / × 2: +31.6% / +30.2% / +28.7%. Borrow failure × 0 to × 2 with 0.5% recalls: +27.8% to +30.9%. By year: 2023 +17.3%, 2024 +34.2%, 2025 +21.8%, 2026 (to 10-01) +40.6%, no yearly drawdown deeper than −8%.

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

### 8. How the samples were used: leakage and design bias, stated plainly

**Data leakage: none.** Features, chip rows, labels, training splits and strategy inputs use only data available before the signal day (chip rows: before the next open); the folds carried over by extension are byte-identical to the old launches.

**Design-level selection bias: present.** The threshold rule and "screen all seven models together" were set after the first edition's out-of-sample results had been seen, and the first edition's out-of-sample period (2026-06 to 10-02) overlaps most of this round's (07-01 to 10-01), so **this round's out-of-sample period is not clean for the rules while it is unseen for the retrained scores**. That is why the six cells do not go straight to paper trading but into a confirmation period entirely after the data end. Under 21-cell multiple comparison the chance that some cell clears 1.5 in every period by luck is higher than in the first edition, and the thresholds were not tightened for it (tightening would itself be an after-the-fact decision).

### 9. Limitations (stated in advance, not found afterwards)

(a) Adjusted prices exclude splits, capital reductions and par-value changes (100 large jumps since 2023, 98 excluded by the halt rule, the other 2 names outside the sample). (b) The out-of-sample period has 64 sessions and one market state. (c) Chip publication times follow a conservative rule; the foreign-holding z-score column is always missing. (d) The short leg's borrow-failure probability, recalls and fee tiers are parameter assumptions; the main tables carry no disposition-stock or forced-cover constraint, and the two rules of 7.5 remain optimistic. (e) Transformer results are not reproducible run to run. (f) Broker-branch and depository data were not bought.

### 10. Next steps

1. The six passing cells enter the 12-month confirmation period from 2026-10-23, each judged on its own: net return > 0 and drawdown within twice its own in-sample drawdown (F2 × S2 −16.2%, F2 × S1 −22.0%, F0 × S2 −17.2%, D0 × S2 −16.4%, D1-Chronos × S2 −18.6%, D1-Kronos × S-A −47.8%); new folds added each quarter by extension, no retraining; the 2027-04-22 interim reading is reported, not judged; no setting changes during the period.
2. Attribution of the TPEX short leg's out-of-sample loss.
2b. Decide whether "no new entry during a disposition period + forced cover at a suspension" becomes the primary execution assumption (adopting it means a separate registration with a fresh clock); every quarterly confirmation reading also lists the numbers under the conservative setting.
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
