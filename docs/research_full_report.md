# 台股量化選股研究完整報告（第二版）：全線重訓、七個模型 × 三種策略的門檻篩選與樣本外驗證 / Quantitative Taiwan Stock Research, Full Report (second edition): the Full Retrain, the 21-Cell Threshold Shortlist and Out-of-Sample Verification

版本：2026-11-02，2026-11-04 補充 §6.7 執行層敏感度並修訂摘要、7.2、8、9.2（取代 2026-10-29 的第一版；第一版封存於 `reports/deprecated/pre-1323-20261102/`，11-02 的 Word 版封存於 `reports/deprecated/pre-1323s3-20261104/`）。
Version: 2026-11-02, with §6.7 (execution-layer sensitivity) added on 2026-11-04 and the summary, 7.2, 8 and 9.2 revised (supersedes the first edition of 2026-10-29, archived under `reports/deprecated/pre-1323-20261102/`; the 11-02 Word files are archived under `reports/deprecated/pre-1323s3-20261104/`).

---

## 繁體中文

### 1. 摘要

這份報告接續 2026-10-29 的第一版。第一版的結論是「事前規則選出的週頻多頭策略在樣本外因回撤未通過，沒有策略進入模擬交易」；同一天發現還原價對有償現金增資套錯，所有已出的模型分數都帶著這個誤差。使用者於 10 月 30 日決定全線重訓：修正資料（修訂 v1.1）、重建特徵庫與標籤庫、七個模型全部在 2016 到 2026-10-23 的完整資料上重訓，策略層候選集擴大為「七個模型 × 三種策略 = 21 格」一起用同一套門檻篩選，入選的格再各自做樣本外驗證。11 月 2 日另發現兩個訓練器自始沒有儲存任何模型，補上模型儲存規則後，用「延伸」而不是重訓的方式把舊 launch 的折沿用下來、只補 P3 需要的折並存權重。

最後的結果（機構主要成本、上市上櫃各半、分數位置）：

- **模型層**：價量＋籌碼的 LightGBM（F2）在所有期間都是最強的模型（樣本內 5 到 10 日 Rank IC 0.109，對純價量模型 +0.019、區間不跨 0）。三個跨股票 transformer（從頭訓練、Kronos-small 編碼器、Chronos-2 編碼器）短天期略優於純價量 LightGBM（+0.004 到 +0.005），5 到 10 日沒有顯著差異，與前兩輪結論相同。
- **門檻篩選**：21 格中 **7 格**三段期間淨 Sharpe 都 ≥ 1.5 且區間下界 > 0：F2 的三個策略、F0 的多空＋避險、兩個 transformer 的多空＋避險、Kronos 的週頻多頭。
- **樣本外驗證（2026-07-01 至 10-01，64 個交易日）**：7 格中 **6 格通過**（淨報酬 > 0 且回撤在樣本內兩倍之內）。最強的一格是 **F2 × 多空＋台指期避險**：樣本內三段 Sharpe 2.17／2.84／2.38、年化 +30.2%，樣本外 **+19.6%**、Sharpe 1.48、回撤 −8.0%（樣本內 −8.1%）。唯一不通過的是 F2 × 週頻多頭（樣本外 −13.4%），原因和第一版一樣：純多頭的回撤由市場決定，7 月那次廣度修正直接打在它身上。
- **處置**：六個通過的格一起進入 **2026-10-23 起 12 個月的確認期**（規則事前登記、期間內不得修改）；F2 × 週頻多頭維持研究狀態。這是這個專案第一次有策略走到確認期。
- **執行層補充敏感度（11 月 4 日，§6.7）**：加上「處置期間不新進場」與「停券起日強制回補」兩條規則後，F2 × 多空＋避險全段 Sharpe 由 2.38 降到 1.70、年化 +30.2% 降到 +22.6%，樣本外 +20.4% 不變；下降幅度超過事前預期，這套保守假設成為確認期每季讀數的參考。六格裡 F2 × S2、D0 × S2、D1-Kronos × S-A 在保守假設下仍過門檻，F2 × S1、F0 × S2、D1-Chronos × S2 未過，三格自此帶註記；散戶成本下只有 D1-Kronos × S-A 通過。影響主要來自處置規則：排序模型在兩端集中於隨後被處置的股票。

第 4 節保留第一版的研究歷程（七個階段），第 5 節記錄這一輪新增的三個階段（資料修正與重訓、模型儲存與延伸、候選集擴大），第 6 節是全部數字，第 7 節交代樣本的使用方式與限制。

### 2. 問題定義與目標

**問題**：在嚴格的時間點輸入、次日開盤成交與時間順序評估下，日線價量加免費籌碼資料的橫斷面排序模型，能不能在台灣的交易成本與執行限制下，變成一條可進入模擬交易的策略？

**成功標準（事前登記，本輪沿用第一版 10 月 29 日修訂後的門檻篩選）**：機構主要成本、上市上櫃各半、分數位置下，(a) 樣本內前段、後段與全段的淨 Sharpe **各自 ≥ 1.5**，且 (b) 全段淨報酬 95% 區間下界 > 0；通過的每一格各自做樣本外驗證：(i) 年化淨報酬 > 0 且 (ii) 最大回撤不深於該格樣本內回撤的**兩倍**。兩者成立才進入確認期。

**本輪與第一版的差別（全部在看任何新數字之前登記）**：候選集從「兩個 LightGBM × 三種策略」擴大為「七個模型 × 三種策略」，模型層的 Rank IC 不再是進入回測的門檻；樣本外期間改由資料結束日推算（最後一個完整季起算），6 月歸入樣本內；資料修訂 v1.1。

**明確不做的事**：不買付費資料；不用大型語言模型自動挖因子；當沖路線在成本證據出來後排除；「勝過 0050 買進持有」不是門檻。

### 3. 資料與評估框架

| 項目 | 設定 |
| --- | --- |
| 價量資料 | 上市（TWSE）與上櫃（TPEX）全部股票的日線開高低收量，2016-01-01 至 2026-10-23；資料修訂 **v1.1**：還原價只套除權息因子，有償現金增資的除權日以交易所的開盤競價基準（前一日收盤價）還原（第一版的錯誤在此修正）；不含分割、減資與面額變更（第 7.2 節） |
| 籌碼資料 | 另一個資料專案提供的 12 個官方品項（法人買賣超、融資融券、借券、當沖統計、注意與處置股、外資持股、得為融資融券與借券名單等），每列帶公布時刻，模型只用訊號日次日開盤前已公布的列 |
| 標籤 | h 日持有報酬對市場、市值與產業做橫斷面回歸後的殘差（h = 1 到 14；殘差化標籤 a）；另測殘差化標籤 b（變體 B）作對照 |
| 訓練方式 | 每季重新訓練、訓練資料從 2016 年起擴張、截止於預測季前 15 個交易日、最後 12 個月為驗證；2023 年第一季到 2026 年第四季共 16 折。每折保存 validation Rank IC 最高的模型（LightGBM 最佳迭代數的 booster、transformer 最佳 epoch 的權重），樣本外需要的折帶模型檔 |
| 樣本內 | 前段 P1 = 2023-01-02 至 2025-06-30（582 個區間）、後段 P2 = 2025-07-01 至 2026-06-30（229）；全段 811 |
| 樣本外 | P3 = 2026-07-01 至 2026-10-01（64 個開盤到開盤區間；資料到 10-23，最後 14 個交易日沒有標籤就沒有預測） |
| 宇宙 | 只含股票；前 20 日成交金額中位數上市 ≥ 3,000 萬、上櫃 ≥ 1,000 萬；收盤價 ≥ 10 元 |
| 成本 | 機構：手續費 0.03%、滑價 0.05%、借券費依規模 1.5%／3%／6%；散戶：手續費 0.0855%、滑價 0.1%／0.2%、融券手續費 0.08%；證交稅 0.3%；滑價 0／1／2 倍敏感度 |
| 資金情境 | 分數位置；機構 3 億與 10 億（單一標的名目 ≤ 20 日成交金額中位數的 5%，超出留現金） |
| 放空執行層 | 只能放空可借券名單內的股票（名單覆蓋率上市 97.6%、上櫃 92.6%）；受交易所次日限額封頂；借不到機率依規模 2%／10%／25%；召回機率 0 與 0.5%；五次隨機重複平均 |
| 統計 | 區塊 bootstrap（區塊 21 日、1,000 次、固定種子）；Sharpe 以日淨報酬年化 |

### 4. 研究歷程（第一版的七個階段，摘要）

第一版第 4 節完整記錄了 10 月 16 日到 28 日的七個階段；這裡只留每個階段的結論，細節見封存的第一版。

| 階段 | 做了什麼 | 結論 |
| --- | --- | --- |
| 4.1 微調金融時序基礎模型（10-16 以前） | Kronos + LoRA 預測美股與台股 1 到 14 日超額報酬與方向 | 三個種子都沒有穩健優勢；誤差幾乎全在同一天的橫斷面上；固定持有天數的回測全部失敗。模型有「同一天分辨相對強弱」的能力，沒有「預測明天漲跌」的能力 |
| 4.2 改成橫斷面排序（10-20 至 21） | 標籤改成同日同市場內的排名，五個基線 | Kronos 微調不優於 PatchTST；排序能力來自特徵與橫斷面建模，不是基礎模型 |
| 4.3 特徵梯度提升樹對跨股票 transformer（10-22 至 23） | 183 欄特徵的 LightGBM 對 MASTER 式 transformer，同一套 walk-forward | transformer 多 0.004 的排序能力來自三種子集成，不值 GPU 成本；每日換 10% 部位的策略被台股成本否決 |
| 4.4 策略層第一輪（10-23） | 買賣緩衝 band、台指期對沖、股票期貨當沖 | 緩衝把成本降到 3% 到 5%／年但 2025 年後失效；當沖被一個 tick 的價差否決 |
| 4.5 殘差化標籤（10-23 至 24） | 標籤改成扣除市場、市值、產業效應的殘差 | 失效是標籤帶了 β，不是模型壞了；之後所有模型都用這個標籤 |
| 4.6 暴露診斷與正交化（10-24 至 26） | 分數每天對 60 日 β、前 20 日與前 60 日報酬做橫斷面回歸取殘差；放空執行層 | 多空策略由上市 −6.3% 變成 +17.3%；這是整個專案最關鍵的一步，也是設計層的事後決定（第 7.1 節） |
| 4.7 免費籌碼資料（10-25 至 28） | 67 欄籌碼特徵 | 5 到 10 日殘差排序能力 +0.017、區間不跨 0；是獨立訊號，不只是條件化反轉 |

第一版的結果（10 月 29 日）：事前的「單一最高 Sharpe」規則選出 F2 × 週頻多頭，樣本外 +3.5% 但回撤 −25.7% 比樣本內深，未通過；事後改為門檻篩選留下三個市場中性策略。同日發現有償現金增資的還原價錯誤，三個策略的樣本外年化淨報酬在修正價下從 +34.1%／+27.5%／+30.9% 降為 +14.7%／+19.4%／+13.7%。

### 5. 本輪新增的三個階段（10 月 30 日至 11 月 2 日）

#### 5.1 資料修正與全線重訓（10-30 至 11-01）

修正只在資料準備層：有償現金增資的除權日，交易所的開盤競價基準停在前一日收盤價，但舊管線用理論參考價去還原，等於替每一筆現增製造一個不存在的上漲（2016 年起 861 筆、2023 年起 386 筆，誤差中位數約 1.9%、最大 32%），而且集中在 6 到 9 月的除權息旺季。資料修訂改為可切換（v1.1 只套除權息因子；v2 另含減資、面額變更、ETF 分割，本線不用）。清空雲端磁碟上錯誤的資料與結果後，依序重做：資料準備 → 四個特徵庫（rank 標籤、殘差化標籤 a、殘差化標籤 b、價量＋籌碼）→ 四個 LightGBM（B0、變體 A = F0、變體 B、F2）→ 三個 transformer（D0 從頭訓練、D1 Kronos-small、D1 Chronos-2-small，各三個種子）。這一輪的 16 折排程把樣本外改成由資料結束日推算：最後一個完整季（2026Q3）起算，所以 2026 年 6 月歸入樣本內後段。

#### 5.2 模型儲存與「延伸」規則（11-02）

11 月 2 日發現兩個訓練器從一開始就只保存預測、沒有任何 booster 或權重——模型無法對新日期出分數，也無法續訓。使用者裁定這是嚴重錯誤，規則改為：每個訓練都保存樣本外需要的折的模型（LightGBM 每個（折, 視野, 種子）一個 booster、transformer 每個（折, 種子）最佳 epoch 的權重），存檔後立即讀回核對；已完成的 launch 不重訓，而是「延伸」：新 launch 以符號連結沿用舊 launch 中訓練窗與預測窗都相同的折，只訓練樣本外需要而舊 launch 沒有模型的折，並寫下每折來源的 lineage；本機的登錄表記錄每個模型目前的正本 launch。七個模型各延伸一次：LightGBM 沿用 14 折、重訓 2 折（2026Q3、Q4）；transformer 沿用 13 折、重訓 3 折（2026Q2 在舊排程裡截在 6 月 1 日、與新排程的預測窗不同）。沿用折的每日 Rank IC 與舊 launch 逐位元相同，逐一核對後才下載。

#### 5.3 候選集擴大為全部模型（11-01）

使用者的理由：先前的流程是先比模型層 Rank IC、選出一個模型再做策略回測，但 Rank IC 和策略績效一樣可能在某一段很高卻不穩定；兩個 Rank IC 差不多的模型，策略報酬可能不同。於是策略層候選集改為七個模型 × 三種策略 = 21 格全部跑完整套件、用同一套門檻各自判讀，模型層的 Rank IC 只作資訊。三個 transformer 改為三個種子集成，與 LightGBM 對稱。為了讓七個模型在同一分數位置比較、也避免其餘五個模型各自在樣本內前段選參數而得到偏樂觀的 Sharpe，多日策略的 (h, K, m) 七個模型一律固定為 F0／F2 在第一版選定的 (10, 50, 3)；週頻多頭的 (K, m) 七個模型用同一規則在上市、前段各自選定。這些都在計分前登記（研究協定 §13.23 第 18 步實作註記）。

### 6. 回測結果

#### 6.1 模型層（資訊性，不參與篩選）

Rank IC（每日橫斷面 Spearman 的平均；短 = 1 日、持有 = 5 到 10 日平均、長 = 11 到 14 日平均；兩市場等權）：

| 模型 | 樣本內短 | 樣本內持有 | 樣本內長 | 樣本外短 | 樣本外持有 | 樣本外長 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 價量 LightGBM，rank 標籤 | 0.117 | 0.087 | 0.086 | 0.085 | 0.096 | 0.106 |
| F0 價量 LightGBM，殘差化標籤 a | 0.117 | 0.090 | 0.090 | 0.084 | 0.097 | 0.107 |
| 變體 B 價量 LightGBM，殘差化標籤 b | 0.113 | 0.085 | 0.083 | 0.084 | 0.094 | 0.108 |
| **F2 價量＋籌碼 LightGBM** | **0.129** | **0.109** | **0.105** | **0.092** | **0.113** | **0.124** |
| D0 跨股票 transformer（從頭訓練） | 0.121 | 0.095 | 0.096 | 0.088 | 0.072 | 0.068 |
| D1 Kronos-small 編碼器 | 0.121 | 0.094 | 0.092 | 0.092 | 0.077 | 0.074 |
| D1 Chronos-2-small 編碼器 | 0.121 | 0.093 | 0.092 | 0.089 | 0.070 | 0.065 |

對 F0 的配對差（每日差的平均與區塊 bootstrap 95% 區間，樣本內 812 日）：F2 短 +0.012 [+0.010, +0.014]、持有 +0.019 [+0.013, +0.024]（樣本外持有 +0.016 [+0.001, +0.028]）；D0 短 +0.005 [+0.002, +0.007]、持有 +0.005 [−0.001, +0.012]；Kronos 短 +0.005 [+0.002, +0.008]、持有 +0.004 [−0.002, +0.010]；Chronos 短 +0.004 [+0.002, +0.007]、持有 +0.003 [−0.004, +0.010]。三個 transformer 的持有格在樣本外都低於 F0（D0 −0.025 [−0.047, −0.005]）。結論與前兩輪相同：籌碼是明確的增量，深度學習的增量只在短天期，不足以改變 LightGBM 為主線的決定。三個 transformer 的種子穩定性：短天期三種子標準差 0.0003 到 0.0012，集成增益 +0.003 到 +0.004。變體 B 三格都低於 F0，殘差化標籤 b 不採用。

#### 6.2 策略層：21 格的主表（機構主要成本、上市上櫃各半、分數位置、V1 正交化）

多日多空（S1）與多空＋避險（S2）七個模型固定 h = 10、K = 50、m = 3；週頻多頭（S-A）的 (K, m)：B0 (50, 3)、F0 (30, 3)、變體 B (30, 2)、F2 (30, 3)、D0 (50, 3)、Kronos (30, 3)、Chronos (30, 3)。

| 模型 | 策略 | P1 Sharpe | P2 Sharpe | 全段 Sharpe | 全段年化淨報酬 | 95% 區間 | 全段最大回撤 | 入選 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2 | S2 多空＋避險 | 2.17 | 2.84 | **2.38** | +30.2% | [+18.5%, +43.0%] | −8.1% | **是** |
| F2 | S-A 週頻多頭 | 1.54 | 3.79 | 2.33 | +53.4% | [+28.1%, +82.9%] | −25.1% | **是** |
| F0 | S-A 週頻多頭 | 1.45 | 3.56 | 2.17 | +40.5% | [+20.4%, +64.1%] | −20.7% | 否（P1） |
| D1-Kronos | S-A 週頻多頭 | 1.56 | 3.36 | 2.16 | +44.2% | [+22.4%, +67.7%] | −23.9% | **是** |
| F2 | S1 多日多空 | 1.90 | 2.58 | 2.11 | +25.5% | [+13.4%, +39.2%] | −11.0% | **是** |
| D0 | S-A 週頻多頭 | 1.45 | 3.31 | 2.06 | +40.0% | [+19.0%, +64.4%] | −24.6% | 否（P1） |
| B0 | S-A 週頻多頭 | 1.48 | 3.14 | 2.05 | +39.1% | [+18.7%, +62.7%] | −21.8% | 否（P1） |
| D1-Chronos | S-A 週頻多頭 | 1.48 | 3.16 | 2.04 | +40.4% | [+17.2%, +66.4%] | −22.8% | 否（P1） |
| 變體 B | S-A 週頻多頭 | 1.44 | 2.93 | 1.95 | +36.4% | [+16.5%, +58.5%] | −18.1% | 否（P1） |
| D1-Chronos | S2 多空＋避險 | 1.69 | 2.10 | 1.81 | +24.4% | [+11.2%, +39.0%] | −9.3% | **是** |
| D0 | S2 多空＋避險 | 1.62 | 2.21 | 1.80 | +24.5% | [+11.1%, +37.8%] | −8.2% | **是** |
| F0 | S2 多空＋避險 | 1.53 | 1.91 | 1.64 | +22.3% | [+8.2%, +36.6%] | −8.6% | **是** |
| D1-Kronos | S2 多空＋避險 | 1.55 | 1.49 | 1.53 | +20.4% | [+7.7%, +33.9%] | −12.3% | 否（P2） |
| D0 | S1 多日多空 | 1.41 | 1.78 | 1.52 | +19.7% | [+6.5%, +33.4%] | −9.8% | 否（P1） |
| D1-Chronos | S1 多日多空 | 1.40 | 1.67 | 1.48 | +19.0% | [+5.6%, +33.4%] | −10.4% | 否 |
| 變體 B | S2 多空＋避險 | 1.84 | 0.57 | 1.45 | +19.1% | [+6.1%, +32.3%] | −9.3% | 否（P2） |
| D1-Kronos | S1 多日多空 | 1.34 | 1.15 | 1.28 | +16.2% | [+2.5%, +30.3%] | −14.2% | 否 |
| F0 | S1 多日多空 | 1.15 | 1.50 | 1.25 | +15.9% | [+1.6%, +30.4%] | −10.3% | 否 |
| B0 | S2 多空＋避險 | 1.38 | 0.87 | 1.24 | +16.8% | [+4.7%, +29.3%] | −8.8% | 否 |
| 變體 B | S1 多日多空 | 1.52 | −0.06 | 1.02 | +12.6% | [−1.0%, +26.6%] | −14.9% | 否 |
| B0 | S1 多日多空 | 1.00 | 0.33 | 0.80 | +10.2% | [−2.5%, +23.1%] | −11.1% | 否 |

21 格中 7 格入選。幾個讀法：(1) 七個週頻多頭的全段 Sharpe 都在 1.95 到 2.33，但只有 F2 與 Kronos 的前段過 1.5，其餘五個卡在 1.44 到 1.48——週頻多頭的高 Sharpe 幾乎全靠後段（3.1 到 3.8），正是門檻要擋的型態；(2) 多空＋避險（S2）是最穩的策略型態，七個模型裡五個全段 ≥ 1.5；(3) 純價量的 B0（rank 標籤）與變體 B 在多空上最弱，殘差化標籤 a 與籌碼特徵各自帶來一級的提升；(4) 原本的「單一最高」規則也會選到 F2 × S2，這次兩種規則一致。

#### 6.3 樣本外驗證（2026-07-01 至 10-01，64 個區間）

規則（事前）：入選格各自 (i) 年化淨報酬 > 0、(ii) 最大回撤不深於該格全段回撤的兩倍。另報第一版的「不更深」讀法作對照。

| 入選格 | 樣本外年化淨報酬 | 95% 區間 | Sharpe | 樣本外回撤 | 全段回撤 | 兩倍門檻 | (i) | (ii) | 不更深 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | **+19.6%** | [−38.2%, +87.8%] | 1.48 | −8.0% | −8.1% | −16.2% | ✓ | ✓ | ✓ | **通過** |
| F2 × S1 | +14.2% | [−42.1%, +81.4%] | 1.06 | −8.3% | −11.0% | −22.0% | ✓ | ✓ | ✓ | **通過** |
| D1-Kronos × S-A | +4.8% | [−57.6%, +63.2%] | 0.16 | −16.1% | −23.9% | −47.8% | ✓ | ✓ | ✓ | **通過** |
| D1-Chronos × S2 | +14.5% | [−49.0%, +91.8%] | 1.03 | −10.0% | −9.3% | −18.6% | ✓ | ✓ | ✗ | **通過** |
| D0 × S2 | +15.7% | [−38.2%, +77.9%] | 1.16 | −7.8% | −8.2% | −16.4% | ✓ | ✓ | ✓ | **通過** |
| F0 × S2 | +7.6% | [−33.4%, +52.6%] | 0.60 | −7.0% | −8.6% | −17.2% | ✓ | ✓ | ✓ | **通過** |
| F2 × S-A | −13.4% | [−120.3%, +75.8%] | −0.33 | −24.6% | −25.1% | −50.2% | ✗ | ✓ | ✓ | 不通過 |

分市場的樣本外：上市 F2 × S2 +47.3% [−6.2%, +105.9%]（Sharpe 2.54）、F2 × S1 +40.9%、D0 × S2 +34.6%、Kronos × S-A +34.5%、F0 × S2 +27.3%；**上櫃的多空格全部為負**（F2 × S2 −8.1%、D0 × S2 −3.1%、F0 × S2 −12.1%、F2 × S1 −12.4%），Kronos × S-A −24.8%。對照欄：等權流動性宇宙 +5.4%（回撤 −22.8%）、0050 +14.8%（−14.4%）、加權報酬指數 +14.0%。

怎麼讀：64 個區間的區間寬 ±50 到 ±120 個百分點，樣本外只能排除「策略壞了」，不能證明「策略有多好」。F2 × S2 是唯一一格四段期間（P1、P2、全段、樣本外）Sharpe 都明顯高於 1 的策略。多空格的樣本外報酬全部由上市貢獻、上櫃為負，是這一段的市場狀態（上櫃空頭腿在 7 到 9 月的反彈中受損）；確認期用兩市場各半判定。週頻多頭再一次在 7 月的廣度修正（等權宇宙 −17%）裡吃到全額回撤；Kronos × S-A 通過只因它的樣本內回撤本來就深（−23.9%），兩倍門檻很寬，判定時要記得這一點。

#### 6.4 散戶成本、資本情境、滑價與執行層

| 入選格 | 散戶全段 | 散戶樣本外 | 機構 3 億全段 | 機構 10 億全段 | 3 億樣本外 | 10 億樣本外 | 滑價 0／1／2 倍（全段） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1%（2.26） | +24.0% | +24.7%（2.20，部署 71%） | +15.6%（2.01，50%） | +36.4% | +9.8% | +31.6%／+30.2%／+28.7% |
| F2 × S1 | +24.2% | +19.8% | +19.7% | +17.2% | +32.9% | +9.8% | +26.9%／+25.5%／+24.1% |
| D1-Kronos × S-A | +41.2% | +1.5% | +31.6%（回撤 −20.2%） | +29.3%（−18.2%） | +14.3% | +19.3% | +45.2%／+44.2%／+43.2% |
| D1-Chronos × S2 | +23.3% | +3.8% | — | — | — | — | — |
| D0 × S2 | +22.2% | +19.5% | +17.8% | +11.9% | +26.6% | +25.6% | +26.1%／+24.5%／+22.8% |
| F0 × S2 | +19.5% | +3.1% | +17.4% | +13.7% | +8.5% | +16.4% | +23.9%／+22.3%／+20.7% |

放空執行層（F2 × S2，全段）：借不到倍率 × 0／× 1／× 2 與召回 0.5% 的淨報酬在 +27.8% 到 +30.9% 之間，±2 個百分點內；可借券名單幾乎不改變結果，分層借券費與借不到各吃掉 1 到 3 個百分點（與第一版相同）。

#### 6.5 逐年（機構、上市上櫃各半）

| 格 | 2023 | 2024 | 2025 | 2026（至 10-01） |
| --- | --- | --- | --- | --- |
| F2 × S2 | +17.3%（1.84） | +34.2%（2.43） | +21.8%（2.03） | +40.6%（2.94） |
| F2 × S1 | +12.5% | +26.8% | +17.0% | +37.5% |
| D0 × S2 | +16.3% | +36.8% | +8.3% | +28.5% |
| F0 × S2 | +9.3% | +32.1% | +12.2% | +25.1% |
| D1-Chronos × S2 | +12.3% | +37.0% | +9.3% | +31.1% |
| D1-Kronos × S-A | +44.0% | +28.0% | +18.7%（回撤 −23.9%） | +77.0% |

F2 × S2 沒有一年低於 +17%、沒有一年回撤深過 −8%；三個 transformer 與 F0 的多空＋避險在 2025 年都掉到 +8% 到 +12%，籌碼特徵在 2025 年的貢獻最明顯。

#### 6.6 籌碼資料在策略層的增量（資訊性）

F2 − F0 的每日淨報酬配對差（全段年化，機構、兩市場各半）：S1 **+9.6% [+1.6%, +18.1%]**、S-A **+12.9% [+4.2%, +23.1%]** → 宣稱有增量；S2 +7.8% [−0.8%, +16.3%] → 區間跨 0，不宣稱。樣本外點估計 S1 +9.2%、S2 +12.0%、S-A −3.5%。與第一版相比，多空上的增量從 +0.8% 升到 +9.6%：修正價與全資料重訓後，籌碼對空頭腿也開始有貢獻。

#### 6.7 執行層補充敏感度（2026-11-04）：處置期間不新進場、停券起日強制回補

本輪的回測在 6.4 之前都把處置股當正常交易，已持有的空頭也只有隨機召回的敏感度。11 月 4 日登記並實作兩條確定性規則（協定 §13.23 補充三）：**D** = 訊號日的次一開盤落在處置期間（公告已出）的股票，多頭與空頭都不新進場，已持有者不強制出場、不加分盤撮合的滑價；**C** = 已持有的空頭在該股被列為暫停融券賣出的次一開盤強制回補，並且停券期間不新建空單（散戶的融券規則；機構借券不因停券而必須歸還，這裡以同一旗標作上界）。暴露：處置股占宇宙股票日上市 1.2%、上櫃 2.2%，但在 D + C 下占 F2 × S2 兩腿候選的 9.1%／9.4%，在週頻多頭占 13% 到 31%；停券占股票日 3.4% 到 3.7%，F2 × S2 在 811 個區間內被強制回補約 400 次（兩個成本檔相當）。

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

散戶成本（C 是制度規則，不是上界）（年化淨報酬 / Sharpe / 最大回撤；全段 = P1 ∪ P2，樣本外 = P3 的 64 個區間；TW 兩市場各半、分數位置、主滑價）：

| 格 | 全段 基準 | 全段 D | 全段 C | 全段 D + C | 樣本外 基準 | 樣本外 D + C | D + C 讀法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1% / 2.26 / -7.3% | +24.9% / 1.85 / -7.7% | +24.7% / 1.96 / -8.6% | **+19.4% / 1.45 / -8.7%** | +24.0% / 1.73 / -7.9% | **+22.5% / 1.51 / -8.4%** | 未通過（Sharpe < 1.5） |
| F2 × S1 | +24.2% / 1.98 / -10.5% | +18.5% / 1.44 / -10.6% | +20.3% / 1.70 / -11.5% | **+13.4% / 1.06 / -11.0%** | +19.8% / 1.43 / -8.4% | **+18.3% / 1.28 / -9.8%** | 未通過（Sharpe < 1.5） |
| D0 × S2 | +22.2% / 1.63 / -9.5% | +20.7% / 1.52 / -10.4% | +19.0% / 1.41 / -9.4% | **+15.6% / 1.17 / -9.9%** | +19.5% / 1.40 / -8.0% | **-1.0% / -0.07 / -10.6%** | 未通過（Sharpe < 1.5） |
| D1-Chronos × S2 | +23.3% / 1.75 / -9.1% | +19.9% / 1.47 / -9.4% | +18.5% / 1.40 / -8.6% | **+17.0% / 1.29 / -9.7%** | +3.8% / 0.27 / -11.9% | **-0.9% / -0.06 / -11.4%** | 未通過（Sharpe < 1.5） |
| F0 × S2 | +19.5% / 1.40 / -8.5% | +18.9% / 1.36 / -8.7% | +18.4% / 1.37 / -9.5% | **+16.3% / 1.19 / -9.7%** | +3.1% / 0.24 / -8.3% | **+21.9% / 1.51 / -8.1%** | 未通過（Sharpe < 1.5） |
| D1-Kronos × S-A | +41.2% / 2.02 / -24.4% | +37.2% / 1.91 / -26.4% | +41.2% / 2.02 / -24.4% | **+37.2% / 1.91 / -26.4%** | +1.5% / 0.05 / -16.3% | **+14.6% / 0.48 / -12.0%** | 通過 |
| F2 × S-A（未進確認期） | +49.7% / 2.18 / -25.3% | +35.4% / 1.82 / -24.6% | +49.7% / 2.18 / -25.3% | **+35.4% / 1.82 / -24.6%** | -17.9% / -0.45 / -24.7% | **+1.2% / 0.05 / -13.8%** | 通過 |

怎麼讀：

- 事前預期「全段年化下降 0 到 5 個百分點、Sharpe 下降 0 到 0.3」被超過（F2 × S2 −7.5 個百分點／−0.68，F2 × S1 −8.9／−0.79），依登記的規則 **D + C 成為確認期每季讀數的保守參考**。依 D + C 的讀法（全段淨 Sharpe ≥ 1.5 且樣本外 > 0），確認期六格裡 **F2 × S2、D0 × S2、D1-Kronos × S-A 通過，F2 × S1、F0 × S2、D1-Chronos × S2 未通過**，後三格自此在報告與每季讀數旁註記「執行層敏感度未通過」。散戶只有 D1-Kronos × S-A 通過，F2 × S2 以 1.45 差 0.05。
- **D 的影響大於 C，而且在多頭與空頭兩腿都出現**：排序模型把兩端集中在隨後被處置的股票，P2 多頭報酬有相當部分來自它們（F2 × S-A 的 P2 由 +114% 降到 +65%、F2 × S1 由 +38% 降到 +22%），樣本外則反過來（7 月修正中這些股票下跌，D 讓 F2 × S-A 的樣本外由 −13.4% 變成 +5.7%）。
- 兩條規則仍偏樂觀（D 不強制出場、不加滑價；C 比真實的最後回補日晚至少一天）；確認期六格、凍結設定與 H-L 門檻不變，是否改以 D + C 為主要執行假設由使用者決定，採用即另案登記並重新起算。
- 第一次執行時機構的借券名單在停券期間仍可進場，被回補後次日又重新放空（強制回補 705 次對散戶 399 次）；修正為停券期間不新建空單後兩個成本檔相當，第一次執行的輸出保留作對照（協定「補充三實作修正」）。

### 7. 樣本的使用方式與資料限制

#### 7.1 樣本是怎麼用的：洩漏與設計偏誤的交代

**資料洩漏（未來資訊進到特徵、標籤、訓練或策略輸入）：沒有。** 每個特徵只用訊號日之前的資料；籌碼只用訊號日次日開盤前已公布的列；標籤的市值代理是過去 20 日成交金額；每一季的模型只用該季開始前 15 個交易日以前的資料訓練，提早停止只看訓練窗內最後 12 個月；策略用的 β、動能、流動性都是往回看的視窗；交易日曆與資料不一致時流程會中止。本輪的七個模型都在同一份 v1.1 資料集上重建；「延伸」沿用的折與舊 launch 逐位元相同，沒有任何折被改過。

**設計層的選擇偏誤（研究者看了結果之後決定下一步）：有，而且本輪多了一層。** 第 4 節七個階段的路徑本身就是看了結果再設計；本輪的候選集與門檻（第 5.3 節）是在看過第一版的樣本外結果之後寫下的——第一版的 P3（2026-06 至 10-02）與本輪的 P3（07-01 至 10-01）大部分重疊，所以**本輪的樣本外對「候選集與門檻」不是乾淨的**，對「重訓後的模型分數」是未看過的。這就是為什麼通過的六格不是直接進模擬交易，而是進入一段完全在資料結束日之後的確認期。

| 決定 | 日期 | 做決定時已看過的結果 | 受影響的數字 |
| --- | --- | --- | --- |
| 183 欄價量特徵、LightGBM 超參數、殘差化標籤、正交化、籌碼特徵、三種策略 | 10-22 至 10-28 | 第一版第 7.1 節逐項交代 | 全部樣本內數字 |
| 門檻篩選規則（三段 Sharpe ≥ 1.5、下界 > 0） | 10-29 | 第一版的樣本外結果 | 本輪的入選名單 |
| 候選集擴大為 21 格、模型層 Rank IC 退出門檻 | 11-01 | 第一版全部結果 | 入選名單（多了 transformer 的格） |
| 樣本外改由資料結束日推算（6 月歸入樣本內） | 11-02 | 第一版全部結果；本輪尚無任何數字 | 本輪 P2、P3 的邊界 |
| 多日參數七個模型固定為 (10, 50, 3)、週頻多頭各自在上市前段選定 | 11-02 | 本輪尚無任何策略數字 | 其餘五個模型的多日格 |
| 延伸而不重訓、保存模型 | 11-02 | 本輪的模型層數字（Rank IC）已出，策略層未出 | 無（沿用折逐位元相同） |

試過的建構：策略層六代、七個模型、三種策略、21 格候選；入選 7 格、通過 6 格。多重比較下「碰巧每一段都過 1.5」的機率比第一版的六格高，門檻沒有因此調整（調整本身就是事後決定）。**樣本內數字要打折看；確認期是唯一一次性的檢驗。**

#### 7.2 資料限制（事前聲明）

(a) 還原價只套除權息因子（含本輪修正的開盤競價基準），不含分割、減資與面額變更：2023 年起超過 25% 的單日還原開盤跳動共 100 次、91 檔，其中 98 次前面有停牌、被樣本規則整段排除，另 2 檔不在模型的樣本內；在目前的規則下不影響結果。
(b) 樣本內後段在第一版設計正交化時已看過；本輪的候選集與門檻是看過第一版樣本外之後定的；本輪的樣本外只對重訓後的分數乾淨。
(c) 樣本外只有 64 個交易日、一種市場狀態，區間寬 ±50 到 ±120 個百分點；最後 14 個交易日沒有標籤就沒有預測。
(d) 籌碼特徵的公布時刻用保守規則（次日開盤前）；外資持股比例的 z 分數欄因資料來源問題恆為缺值。
(e) 放空腿的借不到機率、召回機率與借券費分層是參數假設；主表沒有處置股與停券強制回補的約束，§6.7 的兩條規則是 11 月 4 日補登記的敏感度，而且仍偏樂觀（處置期間不強制出場、不加分盤撮合的滑價；強制回補比真實的最後回補日晚至少一天；機構的停券回補是上界）。
(f) 三個 transformer 的結果不保證逐次重現（bf16、不同 GPU）；本輪重訓的三折與舊 launch 不同，報告用的是延伸後的正本。
(g) 樣本外視窗含 2026 年 7 月修正，不含 2022 年空頭；分點與集保資料未購買。

### 8. 心得與改進方向

**流程上的心得（在第一版五點之上新增）**

6. **模型要保存，這不是細節。** 兩個訓練器跑了兩週、七輪實驗，沒有一個模型檔；發現時所有分數都沒辦法對新日期重算。修法（保存、延伸、登錄表）讓之後每季的確認期只需要訓練新增的折。
7. **候選集要一起篩，不要先選模型。** 七個模型的樣本內 Rank IC 差距 0.01 到 0.02，策略層的差距卻是 Sharpe 0.8 到 2.4；如果先用 Rank IC 選一個模型，會漏掉 transformer 的多空＋避險（1.80、1.81）這種格。
8. **門檻篩選擋住了什麼。** 七個週頻多頭的全段 Sharpe 都超過 1.95，五個被前段擋在門外；通過的兩個裡，F2 的那一個在樣本外 −13.4%。單看全段 Sharpe 會把行情依賴的策略排在最前面，這一輪再次印證。
9. **資料修正的影響方向要事前講清楚。** 修正價讓第一版的樣本外報酬少 8 到 19 個百分點，但全資料重訓後多空上的籌碼增量反而從 +0.8% 升到 +9.6%——修正不是單向地讓結果變差。

**改進方向（依序，每一項都要先登記再做）**

| 順序 | 項目 | 理由 |
| --- | --- | --- |
| 1 | 六個通過的格進入 2026-10-23 起 12 個月的確認期，各自判定（淨報酬 > 0、回撤不深於各自樣本內的兩倍）；每季以延伸方式補新折，期中（2027-04-22）只報不判 | 這是對重訓後模型第一段乾淨的樣本外 |
| 2 | 上櫃空頭腿在樣本外的虧損要做歸因（反彈期的空頭 band 持股與借券費） | 多空格的樣本外全部由上市貢獻 |
| 2b | 決定是否以「處置期間不新進場＋停券強制回補」為主要執行假設（使用者決定；採用即另案登記、重新起算），並補上處置期間的滑價加碼與真實的融券最後回補日 | §6.7 的下降超過事前預期，排序模型在兩端集中於隨後被處置的股票 |
| 3 | 事件研究：處置、注意股首次公告、當沖比跳升後的反轉 | 免費資料裡尚未使用的事件型訊號 |
| 4 | 跨股票 transformer 在籌碼特徵上重訓（使用者決定） | 籌碼增量在排序層成立，transformer 的多空＋避險已入選 |
| 5 | 購買券商分點資料（使用者決定；兩道事前門檻已通過） | 文獻與子分析都指向分點對強勢股延續／反轉的鑑別 |

### 9. 附錄

#### 9.1 本文名稱與程式庫內部代號的對照

| 本文名稱 | 程式庫代號 | 正本 launch |
| --- | --- | --- |
| B0 價量 LightGBM，rank 標籤 | `t-lgbm-alpha` × `alpha_tw_v1` | `launch-20261101T212237Z-2508230824` |
| F0 價量 LightGBM，殘差化標籤 a（變體 A） | `t-lgbm-alpha-residual_v1a` × `alpha_tw_v1` | `launch-20261101T220120Z-29216090` |
| 變體 B 價量 LightGBM，殘差化標籤 b | `t-lgbm-alpha-residual_v1b` × `alpha_tw_v1` | `launch-20261101T221954Z-1215615947` |
| F2 價量＋籌碼 LightGBM | `t-lgbm-alpha-residual_v1a` × `alpha_chip_tw_v1` | `launch-20261101T223737Z-1940816181` |
| D0 跨股票 transformer（從頭訓練） | `t-xs-scratch` | `launch-20261102T063519Z-189413204` |
| D1 Kronos-small | `t-xs-kronos-small` | `launch-20261102T071451Z-322624982` |
| D1 Chronos-2-small | `t-xs-chronos2-small` | `launch-20261102T073405Z-1322713845` |
| 多日多空 | S1 / `md_ls` | 10 日分數的 band 多空 |
| 多空＋台指期避險 | S2 / `md_ls_tx` | S1 加台指期滾動 β 疊加 |
| 週頻低換手多頭 | S-A / `wk_long` | 5 日分數的週頻 band 多頭 |
| 樣本內前段／後段 | P1 / P2 | 2023-01-02 至 2025-06-30／2025-07-01 至 2026-06-30 |
| 樣本外 | P3 | 2026-07-01 至 2026-10-01 |
| 確認期 | P4' | 2026-10-23 起 12 個月 |
| 門檻篩選、樣本外驗證、確認期判定 | §13.20 補充一、§13.23 第 5 項 H-M、§13.22 第 3 項 H-L | `docs/experiment_protocol.md` |

#### 9.2 產物與重現

- 研究協定（每個實驗的事前登記與結果）：`docs/experiment_protocol.md` §13.23 與其補充；第一版的章節 §13.1 至 §13.22。
- 模型輸出：`artifacts/lgbm/<launch>/`、`artifacts/xs/<launch>/`（含 `lineage.json` 與 `models/`）；登錄表 `registry/lgbm/*.json`、`registry/xs/*.json`。
- 回測輸出：`artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z/`（`grid.csv`、`daily.csv`、`yearly.csv`、`report.json`、`summary.md`）；篩選與樣本外驗證：`strategy-decision-p3b-21cells-20261102T082429Z/`。
- 執行層敏感度（§6.7）：`scenario-suite-p3b-disp-20261104T113932Z/`（D）、`scenario-suite-p3b-cover-20261104T114651Z/`（C）、`scenario-suite-p3b-exec-20261104T114651Z/`（D + C）與對應的 `strategy-decision-p3b-{disp,cover,exec}-…/`；設定 `backtest/config/scenario_suite_v1_p3b_{disp,cover,exec}.json`；修正前的第一次執行 `…-cover-20261104T113932Z/`、`…-exec-20261104T113932Z/` 保留。
- 重現：`cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3b.json --chip-data-root <tw_chip_data/data> --multi-day-default F0 --predictions ../artifacts/lgbm/<B0>/predictions.parquet --model t-lgbm-alpha --label B0 … --prices ../artifacts/datasets/<v1.1 資料集>/raw/market.parquet`，再 `uv run python -m twbacktest.strategy_decision --config config/strategy_decision_p3b.json --suite <套件輸出目錄>`；完整命令見協定 §13.23 第 18 步實作註記。
- 程式：`src/stock_forecasting/`（資料、特徵、模型、延伸與登錄表）、`backtest/twbacktest/`（策略與回測）、`scripts/`（雲端 CPU／GPU 工作流程與下載）。

---

## English

### 1. Summary

This report continues the first edition of 2026-10-29. That edition concluded that the strategy picked by the pre-registered rule, a weekly long, failed its out-of-sample check on drawdown and that no strategy entered paper trading; the same day an adjusted-price defect for paid rights issues was found, and every published model score carried it. On 30 October the user ordered a full retrain: corrected data (revision v1.1), rebuilt feature and label stores, all seven models retrained on the full 2016 to 2026-10-23 history, and the strategy-layer candidate set widened to "seven models × three strategies = 21 cells", all screened by one threshold rule with each shortlisted cell verified out of sample on its own. On 2 November it emerged that neither trainer had ever saved a model; a model-saving rule was added, and the finished launches were "extended" rather than retrained: the old folds were carried over by links and only the folds the out-of-sample period needs were trained and saved.

The outcome (institution primary cost, TWSE and TPEX half each, fractional capital):

- **Model level**: the price-volume + chip LightGBM (F2) is the strongest model in every period (in-sample 5-to-10-day Rank IC 0.109, +0.019 over the price-volume model with an interval clear of zero). The three cross-stock transformers (trained from scratch, Kronos-small encoder, Chronos-2 encoder) beat the price-volume LightGBM slightly at the short horizon (+0.004 to +0.005) and not at 5 to 10 days, as in the two earlier rounds.
- **Threshold shortlist**: **7 of 21 cells** reach net Sharpe ≥ 1.5 in all three periods with a lower bound above zero: F2's three strategies, F0's long-short + hedge, the two transformers' long-short + hedge, and Kronos's weekly long.
- **Out-of-sample verification (2026-07-01 to 10-01, 64 sessions)**: **6 of 7 pass** (net return > 0 and drawdown within twice the in-sample drawdown). The strongest cell is **F2 × long-short + TX hedge**: in-sample Sharpe 2.17 / 2.84 / 2.38 and +30.2% a year, out of sample **+19.6%**, Sharpe 1.48, drawdown −8.0% (in-sample −8.1%). The one failure is F2 × weekly long (−13.4% out of sample), for the reason seen in the first edition: a long-only book's drawdown is set by the market, and the July breadth correction landed on it.
- **Disposition**: the six passing cells enter a **12-month confirmation period from 2026-10-23** (rules registered in advance and frozen); F2 × weekly long stays in research. This is the first time a strategy in this project has reached a confirmation period.
- **Execution-layer sensitivity added on 4 November (§6.7)**: with the rules "no new entry during a disposition period" and "forced cover when short selling is suspended", F2 × long-short + hedge falls from Sharpe 2.38 to 1.70 and from +30.2% to +22.6% a year over the whole period while its out-of-sample +20.4% is unchanged; the fall exceeds the pre-registered expectation, so this conservative setting becomes the reference for the quarterly confirmation readings. Of the six cells, F2 × S2, D0 × S2 and D1-Kronos × S-A still clear the thresholds under it, F2 × S1, F0 × S2 and D1-Chronos × S2 do not and now carry a note; under retail costs only D1-Kronos × S-A passes. The effect comes mainly from the disposition rule: the ranking models concentrate both ends of the ranking in stocks that are subsequently placed under disposition.

Section 4 keeps the first edition's research history (seven stages) in summary, section 5 records the three stages added in this round (data fix and retrain, model saving and extension, the wider candidate set), section 6 holds all the numbers and section 7 states how the samples were used and the limitations.

### 2. Problem and goals

**Problem**: under strict point-in-time inputs, next-open execution and time-ordered evaluation, can a cross-sectional ranking model built on daily price-volume plus free chip data become a strategy fit for paper trading under Taiwan's trading costs and execution constraints?

**Success criteria (pre-registered; this round keeps the threshold rule amended on 29 October)**: under the institution primary cost, TWSE and TPEX half each and fractional capital, (a) net Sharpe **≥ 1.5 in each** of the in-sample first half, second half and the whole, and (b) the whole-period net-return 95% lower bound > 0; every passing cell is verified out of sample on its own: (i) annual net return > 0 and (ii) maximum drawdown not deeper than **twice** that cell's in-sample drawdown. Both must hold to enter the confirmation period.

**What changed against the first edition (all registered before any new number was seen)**: the candidate set grew from "two LightGBMs × three strategies" to "seven models × three strategies", and model-level Rank IC stopped being a gate into the backtest; the out-of-sample period is now derived from the data end (from the start of the last complete quarter), so June moved into the in-sample window; data revision v1.1.

**Explicitly not done**: no paid data; no large-language-model factor mining; the day-trading route was dropped once the cost evidence came in; "beating 0050 buy-and-hold" is not a gate.

### 3. Data and evaluation framework

| Item | Setting |
| --- | --- |
| Price-volume data | Daily OHLCV of every TWSE and TPEX stock, 2016-01-01 to 2026-10-23; data revision **v1.1**: adjusted prices carry ex-dividend / ex-rights factors only, with paid rights issues adjusted on the exchange's opening basis (the previous close), which fixes the first edition's defect; no splits, capital reductions or par-value changes (section 7.2) |
| Chip data | Twelve official items from a companion data project (institutional net buying, margin and short balances, securities lending, day-trade statistics, attention and disposition stocks, foreign holdings, eligibility lists), each row stamped with its publication time; the model uses only rows published before the next session's open |
| Labels | h-day holding return residualized cross-sectionally on market, size and industry (h = 1 to 14; residual label a); residual label b (variant B) tested as a comparison |
| Training | Retrained each quarter on an expanding window from 2016, cut 15 sessions before the prediction quarter, the last 12 months as validation; 16 folds from 2023Q1 to 2026Q4. Each fold keeps the model with the best validation Rank IC (LightGBM booster at the best iteration, transformer weights of the best epoch); the folds the out-of-sample period needs carry their model files |
| In sample | first half P1 = 2023-01-02 to 2025-06-30 (582 intervals), second half P2 = 2025-07-01 to 2026-06-30 (229); whole 811 |
| Out of sample | P3 = 2026-07-01 to 2026-10-01 (64 open-to-open intervals; the data run to 10-23 but the last 14 sessions have no label and so no prediction) |
| Universe | Stocks only; 20-day median dollar volume ≥ TWD 30M on TWSE and ≥ 10M on TPEX; close ≥ 10 |
| Costs | Institution: commission 0.03%, slippage 0.05%, borrow fee 1.5% / 3% / 6% by size; retail: commission 0.0855%, slippage 0.1% / 0.2%, short fee 0.08%; transaction tax 0.3%; slippage × 0 / × 1 / × 2 sensitivity |
| Capital | Fractional; institution 300M and 1B TWD (one name's notional ≤ 5% of the 20-day median dollar volume, excess held as cash) |
| Short execution layer | Only names on the borrowable list (coverage TWSE 97.6%, TPEX 92.6%); capped by the exchange's next-day limit; borrow failure 2% / 10% / 25% by size; recall 0 and 0.5%; average of five random repeats |
| Statistics | Block bootstrap (21-day blocks, 1,000 replicates, fixed seed); Sharpe from daily net returns, annualized |

### 4. Research history (the first edition's seven stages, in summary)

Section 4 of the first edition records the seven stages of 16 to 28 October in full; only each stage's conclusion is kept here.

| Stage | What was done | Conclusion |
| --- | --- | --- |
| 4.1 Fine-tuning a financial time-series foundation model (before 10-16) | Kronos + LoRA predicting 1-to-14-day excess returns and direction for US and Taiwan stocks | No robust edge across three seeds; the error sits almost entirely in the same-day cross-section; fixed-holding backtests all failed. The model can rank stocks within a day, not call tomorrow's direction |
| 4.2 Cross-sectional ranking (10-20 to 21) | Labels became same-day, same-market ranks; five baselines | Fine-tuned Kronos did not beat PatchTST; ranking power comes from features and cross-sectional modelling, not the foundation model |
| 4.3 Feature gradient boosting against a cross-stock transformer (10-22 to 23) | 183-feature LightGBM against a MASTER-style transformer on one walk-forward | The transformer's +0.004 came from the three-seed ensemble and was not worth GPU time; a strategy turning 10% of the book a day was killed by Taiwan costs |
| 4.4 Strategy layer, first round (10-23) | Hold buffers, TX hedge, stock-futures day trading | Buffers cut cost drag to 3–5% a year but the book failed after 2025; day trading was killed by one tick of spread |
| 4.5 Residualized labels (10-23 to 24) | Labels residualized on market, size and industry | The failure was beta in the label, not a broken model; every later model uses this label |
| 4.6 Exposure diagnostics and orthogonalization (10-24 to 26) | Scores residualized daily on 60-day beta and 20- and 60-day momentum; short execution layer | Long-short on TWSE went from −6.3% to +17.3%; the project's decisive step, and a design decision taken after the fact (section 7.1) |
| 4.7 Free chip data (10-25 to 28) | 67 chip features | 5-to-10-day residual Rank IC +0.017 with an interval clear of zero; a signal of its own, not only conditional reversal |

The first edition's result (29 October): the pre-registered "single highest Sharpe" rule picked F2 × weekly long, which returned +3.5% out of sample with a −25.7% drawdown deeper than the in-sample −22.6% and failed; the rule was then amended to the threshold shortlist, which left three market-neutral strategies. The same day the paid-rights price defect was found: on corrected prices the three strategies' out-of-sample annual net returns fell from +34.1% / +27.5% / +30.9% to +14.7% / +19.4% / +13.7%.

### 5. The three stages added in this round (30 October to 2 November)

#### 5.1 Data fix and full retrain (10-30 to 11-01)

The fix is confined to data preparation: on a paid rights issue's ex-date the exchange's opening basis stays at the previous close, but the old pipeline adjusted with the theoretical reference price, inventing a rise for every such issue (861 since 2016, 386 since 2023, median error about 1.9%, maximum 32%), clustered in the June-to-September ex-rights season. The data revision became switchable (v1.1 ex-dividend / ex-rights factors only; v2 also covers capital reductions, par-value changes and ETF splits and is not used on this line). After clearing the wrong data and results from the cloud volume the chain was rebuilt in order: data preparation → four stores (rank labels, residual label a, residual label b, price-volume + chip) → four LightGBMs (B0, variant A = F0, variant B, F2) → three transformers (D0 from scratch, D1 Kronos-small, D1 Chronos-2-small, three seeds each). This round's 16-fold schedule derives the out-of-sample period from the data end: it starts at the last complete quarter (2026Q3), so June 2026 belongs to the in-sample second half.

#### 5.2 Model saving and the "extension" rule (11-02)

On 2 November it emerged that both trainers had only ever saved predictions, no booster and no weights: no model could score a new date or be trained further. The user ruled this a serious defect and the rule became: every run saves the models of the folds the out-of-sample period needs (one LightGBM booster per (fold, horizon, seed), the transformer's best-epoch weights per (fold, seed)), read back right after saving; finished launches are not retrained but "extended": the new launch links to every old fold whose training and prediction windows match, trains only the folds the out-of-sample period needs and the old launch lacks models for, and writes a lineage of each fold's source; a local registry records each model's current launch. Each of the seven models was extended once: the LightGBMs reuse 14 folds and retrain 2 (2026Q3, Q4); the transformers reuse 13 and retrain 3 (2026Q2 was cut at 1 June in the old schedule, so its prediction window differs). The reused folds' daily Rank IC is byte-identical to the old launch, checked one launch at a time before downloading.

#### 5.3 The candidate set widened to every model (11-01)

The user's reasoning: the earlier flow compared model-level Rank IC first and backtested one chosen model, but Rank IC, like strategy performance, can be high in one period and unstable, and two models with near-equal Rank IC can produce different strategy returns. The strategy-layer candidate set therefore became all seven models × three strategies = 21 cells, every cell through the full suite and read against the same thresholds, with model-level Rank IC informational only. The three transformers became three-seed ensembles, symmetric with LightGBM. So that all seven models are compared at one score position, and so that the other five do not pick their own parameters on the in-sample first half and gain an optimistic Sharpe, the multi-day (h, K, m) is fixed at the first edition's F0 / F2 selection (10, 50, 3) for every model; the weekly long's (K, m) is selected per model by one rule on TWSE in the first half. All of this was registered before scoring (protocol §13.23, step 18 implementation note).

### 6. Backtest results

#### 6.1 Model level (informational, not part of the shortlist)

Rank IC (mean daily cross-sectional Spearman; short = 1 day, holding = mean of 5 to 10 days, long = mean of 11 to 14 days; two markets equal-weight):

| Model | In-sample short | In-sample holding | In-sample long | OOS short | OOS holding | OOS long |
| --- | --- | --- | --- | --- | --- | --- |
| B0 price-volume LightGBM, rank labels | 0.117 | 0.087 | 0.086 | 0.085 | 0.096 | 0.106 |
| F0 price-volume LightGBM, residual label a | 0.117 | 0.090 | 0.090 | 0.084 | 0.097 | 0.107 |
| Variant B price-volume LightGBM, residual label b | 0.113 | 0.085 | 0.083 | 0.084 | 0.094 | 0.108 |
| **F2 price-volume + chip LightGBM** | **0.129** | **0.109** | **0.105** | **0.092** | **0.113** | **0.124** |
| D0 cross-stock transformer (from scratch) | 0.121 | 0.095 | 0.096 | 0.088 | 0.072 | 0.068 |
| D1 Kronos-small encoder | 0.121 | 0.094 | 0.092 | 0.092 | 0.077 | 0.074 |
| D1 Chronos-2-small encoder | 0.121 | 0.093 | 0.092 | 0.089 | 0.070 | 0.065 |

Paired differences against F0 (mean daily difference with a block-bootstrap 95% interval, 812 in-sample days): F2 short +0.012 [+0.010, +0.014], holding +0.019 [+0.013, +0.024] (OOS holding +0.016 [+0.001, +0.028]); D0 short +0.005 [+0.002, +0.007], holding +0.005 [−0.001, +0.012]; Kronos short +0.005 [+0.002, +0.008], holding +0.004 [−0.002, +0.010]; Chronos short +0.004 [+0.002, +0.007], holding +0.003 [−0.004, +0.010]. Out of sample all three transformers' holding cells are below F0 (D0 −0.025 [−0.047, −0.005]). The conclusion matches the two earlier rounds: chip data is a clear increment, deep learning's increment is confined to the short horizon and does not change the decision that LightGBM is the main line. Transformer seed stability: short-cell seed standard deviation 0.0003 to 0.0012, ensemble gain +0.003 to +0.004. Variant B is below F0 in all three cells; residual label b is not adopted.

#### 6.2 Strategy level: the 21-cell main table (institution primary cost, TWSE and TPEX half each, fractional, V1 orthogonalized)

Long-short (S1) and long-short + hedge (S2) use h = 10, K = 50, m = 3 for all seven models; the weekly long (S-A) (K, m): B0 (50, 3), F0 (30, 3), variant B (30, 2), F2 (30, 3), D0 (50, 3), Kronos (30, 3), Chronos (30, 3).

| Model | Strategy | P1 Sharpe | P2 Sharpe | Whole Sharpe | Whole annual net | 95% interval | Whole max drawdown | Shortlisted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2 | S2 long-short + hedge | 2.17 | 2.84 | **2.38** | +30.2% | [+18.5%, +43.0%] | −8.1% | **yes** |
| F2 | S-A weekly long | 1.54 | 3.79 | 2.33 | +53.4% | [+28.1%, +82.9%] | −25.1% | **yes** |
| F0 | S-A weekly long | 1.45 | 3.56 | 2.17 | +40.5% | [+20.4%, +64.1%] | −20.7% | no (P1) |
| D1-Kronos | S-A weekly long | 1.56 | 3.36 | 2.16 | +44.2% | [+22.4%, +67.7%] | −23.9% | **yes** |
| F2 | S1 long-short | 1.90 | 2.58 | 2.11 | +25.5% | [+13.4%, +39.2%] | −11.0% | **yes** |
| D0 | S-A weekly long | 1.45 | 3.31 | 2.06 | +40.0% | [+19.0%, +64.4%] | −24.6% | no (P1) |
| B0 | S-A weekly long | 1.48 | 3.14 | 2.05 | +39.1% | [+18.7%, +62.7%] | −21.8% | no (P1) |
| D1-Chronos | S-A weekly long | 1.48 | 3.16 | 2.04 | +40.4% | [+17.2%, +66.4%] | −22.8% | no (P1) |
| Variant B | S-A weekly long | 1.44 | 2.93 | 1.95 | +36.4% | [+16.5%, +58.5%] | −18.1% | no (P1) |
| D1-Chronos | S2 long-short + hedge | 1.69 | 2.10 | 1.81 | +24.4% | [+11.2%, +39.0%] | −9.3% | **yes** |
| D0 | S2 long-short + hedge | 1.62 | 2.21 | 1.80 | +24.5% | [+11.1%, +37.8%] | −8.2% | **yes** |
| F0 | S2 long-short + hedge | 1.53 | 1.91 | 1.64 | +22.3% | [+8.2%, +36.6%] | −8.6% | **yes** |
| D1-Kronos | S2 long-short + hedge | 1.55 | 1.49 | 1.53 | +20.4% | [+7.7%, +33.9%] | −12.3% | no (P2) |
| D0 | S1 long-short | 1.41 | 1.78 | 1.52 | +19.7% | [+6.5%, +33.4%] | −9.8% | no (P1) |
| D1-Chronos | S1 long-short | 1.40 | 1.67 | 1.48 | +19.0% | [+5.6%, +33.4%] | −10.4% | no |
| Variant B | S2 long-short + hedge | 1.84 | 0.57 | 1.45 | +19.1% | [+6.1%, +32.3%] | −9.3% | no (P2) |
| D1-Kronos | S1 long-short | 1.34 | 1.15 | 1.28 | +16.2% | [+2.5%, +30.3%] | −14.2% | no |
| F0 | S1 long-short | 1.15 | 1.50 | 1.25 | +15.9% | [+1.6%, +30.4%] | −10.3% | no |
| B0 | S2 long-short + hedge | 1.38 | 0.87 | 1.24 | +16.8% | [+4.7%, +29.3%] | −8.8% | no |
| Variant B | S1 long-short | 1.52 | −0.06 | 1.02 | +12.6% | [−1.0%, +26.6%] | −14.9% | no |
| B0 | S1 long-short | 1.00 | 0.33 | 0.80 | +10.2% | [−2.5%, +23.1%] | −11.1% | no |

7 of 21 cells are shortlisted. Readings: (1) all seven weekly longs sit at a whole-period Sharpe of 1.95 to 2.33, but only F2's and Kronos's first half clears 1.5 while the other five stop at 1.44 to 1.48 — the weekly long's high Sharpe rests almost entirely on the second half (3.1 to 3.8), exactly the pattern the thresholds are meant to stop; (2) long-short + hedge (S2) is the steadiest strategy type, five of seven models reach ≥ 1.5 over the whole period; (3) the price-volume-only B0 (rank labels) and variant B are weakest on the long-short books, so residual label a and the chip features each add a step; (4) the original single-argmax rule would also pick F2 × S2; this time the two rules agree.

#### 6.3 Out-of-sample verification (2026-07-01 to 10-01, 64 intervals)

Rule (pre-registered): each shortlisted cell on its own, (i) annual net return > 0 and (ii) maximum drawdown not deeper than twice the cell's whole-period drawdown. The first edition's "not deeper" reading is shown for comparison.

| Shortlisted cell | OOS annual net | 95% interval | Sharpe | OOS drawdown | Whole-period drawdown | Twice limit | (i) | (ii) | Not deeper | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **F2 × S2** | **+19.6%** | [−38.2%, +87.8%] | 1.48 | −8.0% | −8.1% | −16.2% | ✓ | ✓ | ✓ | **pass** |
| F2 × S1 | +14.2% | [−42.1%, +81.4%] | 1.06 | −8.3% | −11.0% | −22.0% | ✓ | ✓ | ✓ | **pass** |
| D1-Kronos × S-A | +4.8% | [−57.6%, +63.2%] | 0.16 | −16.1% | −23.9% | −47.8% | ✓ | ✓ | ✓ | **pass** |
| D1-Chronos × S2 | +14.5% | [−49.0%, +91.8%] | 1.03 | −10.0% | −9.3% | −18.6% | ✓ | ✓ | ✗ | **pass** |
| D0 × S2 | +15.7% | [−38.2%, +77.9%] | 1.16 | −7.8% | −8.2% | −16.4% | ✓ | ✓ | ✓ | **pass** |
| F0 × S2 | +7.6% | [−33.4%, +52.6%] | 0.60 | −7.0% | −8.6% | −17.2% | ✓ | ✓ | ✓ | **pass** |
| F2 × S-A | −13.4% | [−120.3%, +75.8%] | −0.33 | −24.6% | −25.1% | −50.2% | ✗ | ✓ | ✓ | fail |

Out of sample by market: TWSE F2 × S2 +47.3% [−6.2%, +105.9%] (Sharpe 2.54), F2 × S1 +40.9%, D0 × S2 +34.6%, Kronos × S-A +34.5%, F0 × S2 +27.3%; **every long-short cell is negative on TPEX** (F2 × S2 −8.1%, D0 × S2 −3.1%, F0 × S2 −12.1%, F2 × S1 −12.4%), Kronos × S-A −24.8%. Comparison columns: equal-weight liquid universe +5.4% (drawdown −22.8%), 0050 +14.8% (−14.4%), TAIEX total return +14.0%.

How to read it: with 64 intervals the intervals are ±50 to ±120 points wide, so the out-of-sample period can only rule out "the strategy is broken", not show how good it is. F2 × S2 is the only cell whose Sharpe is clearly above 1 in all four periods (P1, P2, whole, out of sample). Every long-short cell's out-of-sample return comes from TWSE while TPEX is negative, a feature of this window (the TPEX short leg suffered in the July-to-September rebound); the confirmation period judges on the two markets half each. The weekly long again took the full drawdown of the July breadth correction (equal-weight universe −17%); Kronos × S-A passes only because its in-sample drawdown was already deep (−23.9%) and the two-times limit is wide, which should be remembered when reading it.

#### 6.4 Retail costs, capital, slippage and the execution layer

| Shortlisted cell | Retail whole | Retail OOS | Institution 300M whole | Institution 1B whole | 300M OOS | 1B OOS | Slippage × 0 / × 1 / × 2 (whole) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1% (2.26) | +24.0% | +24.7% (2.20, deployed 71%) | +15.6% (2.01, 50%) | +36.4% | +9.8% | +31.6% / +30.2% / +28.7% |
| F2 × S1 | +24.2% | +19.8% | +19.7% | +17.2% | +32.9% | +9.8% | +26.9% / +25.5% / +24.1% |
| D1-Kronos × S-A | +41.2% | +1.5% | +31.6% (drawdown −20.2%) | +29.3% (−18.2%) | +14.3% | +19.3% | +45.2% / +44.2% / +43.2% |
| D1-Chronos × S2 | +23.3% | +3.8% | — | — | — | — | — |
| D0 × S2 | +22.2% | +19.5% | +17.8% | +11.9% | +26.6% | +25.6% | +26.1% / +24.5% / +22.8% |
| F0 × S2 | +19.5% | +3.1% | +17.4% | +13.7% | +8.5% | +16.4% | +23.9% / +22.3% / +20.7% |

Short execution layer (F2 × S2, whole period): with borrow failure × 0 / × 1 / × 2 and 0.5% recalls the net return stays between +27.8% and +30.9%, within ±2 points; the borrowable list itself hardly moves the result, and the tiered borrow fee and the failures each cost 1 to 3 points (as in the first edition).

#### 6.5 By year (institution, TWSE and TPEX half each)

| Cell | 2023 | 2024 | 2025 | 2026 (to 10-01) |
| --- | --- | --- | --- | --- |
| F2 × S2 | +17.3% (1.84) | +34.2% (2.43) | +21.8% (2.03) | +40.6% (2.94) |
| F2 × S1 | +12.5% | +26.8% | +17.0% | +37.5% |
| D0 × S2 | +16.3% | +36.8% | +8.3% | +28.5% |
| F0 × S2 | +9.3% | +32.1% | +12.2% | +25.1% |
| D1-Chronos × S2 | +12.3% | +37.0% | +9.3% | +31.1% |
| D1-Kronos × S-A | +44.0% | +28.0% | +18.7% (drawdown −23.9%) | +77.0% |

F2 × S2 has no year below +17% and no yearly drawdown deeper than −8%; the three transformers' and F0's long-short + hedge all fall to +8% to +12% in 2025, the year in which the chip features matter most.

#### 6.6 The chip increment at the strategy level (informational)

F2 − F0 paired daily net difference (whole period annualized, institution, two markets half each): S1 **+9.6% [+1.6%, +18.1%]**, S-A **+12.9% [+4.2%, +23.1%]** → an increment is claimed; S2 +7.8% [−0.8%, +16.3%] → the interval crosses zero, not claimed. Out-of-sample point estimates S1 +9.2%, S2 +12.0%, S-A −3.5%. Against the first edition the long-short increment rose from +0.8% to +9.6%: after the price fix and the full-data retrain the chip features also help the short leg.

#### 6.7 Execution-layer sensitivity added on 2026-11-04: no new entries during a disposition period, forced cover when short selling is suspended

Up to section 6.4 the backtest treated disposition stocks as trading normally and held shorts had only the random-recall sensitivity. Two deterministic rules were registered and implemented on 4 November (protocol §13.23 supplement 3): **D** = a stock whose next open after the signal date lies inside an announced disposition period gets no new entry in either leg; held positions are not forced out and no batched-auction slippage is added; **C** = a held short is covered at the open after the stock is listed as suspended from margin short selling, and no new short is opened during the suspension (the retail margin rule; an institution's SBL borrow need not be returned at a suspension, so the same flag serves as an upper bound). Exposure: disposition stocks are 1.2% (TWSE) / 2.2% (TPEX) of universe stock-days but, under D + C, 9.1% / 9.4% of F2 × S2's candidates across both legs and 13% to 31% of the weekly long's; suspensions cover 3.4% to 3.7% of stock-days, and F2 × S2 is force-covered about 400 times over the 811 intervals (both cost profiles alike).

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

Retail cost (C is the rule here, not an upper bound) (annualized net return / Sharpe / max drawdown; whole = P1 ∪ P2, OOS = the 64 intervals of P3; TW half each, fractional, primary slippage):

| Cell | Whole baseline | Whole D | Whole C | Whole D + C | OOS baseline | OOS D + C | D + C reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 × S2 | +29.1% / 2.26 / -7.3% | +24.9% / 1.85 / -7.7% | +24.7% / 1.96 / -8.6% | **+19.4% / 1.45 / -8.7%** | +24.0% / 1.73 / -7.9% | **+22.5% / 1.51 / -8.4%** | fail (Sharpe < 1.5) |
| F2 × S1 | +24.2% / 1.98 / -10.5% | +18.5% / 1.44 / -10.6% | +20.3% / 1.70 / -11.5% | **+13.4% / 1.06 / -11.0%** | +19.8% / 1.43 / -8.4% | **+18.3% / 1.28 / -9.8%** | fail (Sharpe < 1.5) |
| D0 × S2 | +22.2% / 1.63 / -9.5% | +20.7% / 1.52 / -10.4% | +19.0% / 1.41 / -9.4% | **+15.6% / 1.17 / -9.9%** | +19.5% / 1.40 / -8.0% | **-1.0% / -0.07 / -10.6%** | fail (Sharpe < 1.5) |
| D1-Chronos × S2 | +23.3% / 1.75 / -9.1% | +19.9% / 1.47 / -9.4% | +18.5% / 1.40 / -8.6% | **+17.0% / 1.29 / -9.7%** | +3.8% / 0.27 / -11.9% | **-0.9% / -0.06 / -11.4%** | fail (Sharpe < 1.5) |
| F0 × S2 | +19.5% / 1.40 / -8.5% | +18.9% / 1.36 / -8.7% | +18.4% / 1.37 / -9.5% | **+16.3% / 1.19 / -9.7%** | +3.1% / 0.24 / -8.3% | **+21.9% / 1.51 / -8.1%** | fail (Sharpe < 1.5) |
| D1-Kronos × S-A | +41.2% / 2.02 / -24.4% | +37.2% / 1.91 / -26.4% | +41.2% / 2.02 / -24.4% | **+37.2% / 1.91 / -26.4%** | +1.5% / 0.05 / -16.3% | **+14.6% / 0.48 / -12.0%** | pass |
| F2 × S-A (not in confirmation) | +49.7% / 2.18 / -25.3% | +35.4% / 1.82 / -24.6% | +49.7% / 2.18 / -25.3% | **+35.4% / 1.82 / -24.6%** | -17.9% / -0.45 / -24.7% | **+1.2% / 0.05 / -13.8%** | pass |

How to read it:

- The pre-registered expectation ("whole-period annualized return down 0 to 5 points, Sharpe down 0 to 0.3") is exceeded (F2 × S2 −7.5 points / −0.68, F2 × S1 −8.9 / −0.79), so by the registered rule **D + C becomes the conservative reference for every quarterly confirmation reading**. Under the D + C reading (whole-period net Sharpe ≥ 1.5 and OOS > 0), of the six confirmation-period cells **F2 × S2, D0 × S2 and D1-Kronos × S-A pass, F2 × S1, F0 × S2 and D1-Chronos × S2 do not**, and the latter three now carry the note "execution-layer sensitivity not passed" in the reports and in every quarterly reading. Among retail cells only D1-Kronos × S-A passes; F2 × S2 misses by 0.05 at 1.45.
- **D matters more than C and shows in both legs**: the ranking models concentrate both ends of the ranking in stocks that are subsequently placed under disposition; a sizeable part of the P2 long-side return came from them (F2 × S-A's P2 falls from +114% to +65%, F2 × S1's from +38% to +22%), while out of sample it goes the other way (those stocks fell in the July correction, so D turns F2 × S-A's OOS from −13.4% to +5.7%).
- Both rules remain optimistic (D forces no exit and adds no slippage; C is at least one day later than the real last cover day); the six confirmation-period cells, the frozen configuration and the H-L limits are unchanged, and whether D + C becomes the primary execution assumption is the user's decision, which would mean a separate registration with a fresh clock.
- In the first run the institution SBL list still admitted suspended names, so a covered position was re-shorted the next day (705 forced covers against 399 for retail); after the correction (no new short during a suspension) the two profiles agree, and the first run's outputs are kept for comparison (protocol "supplement 3 implementation correction").

### 7. How the samples were used, and data limitations

#### 7.1 How the samples were used: leakage and design bias, stated plainly

**Data leakage (future information in features, labels, training or strategy inputs): none.** Every feature uses only data before the signal day; chip rows only if published before the next session's open; the label's size proxy is the trailing 20-day dollar volume; each quarter's model trains only on data up to 15 sessions before the quarter starts, with early stopping on the last 12 months inside the training window; the strategy's beta, momentum and liquidity are trailing windows; the pipeline halts when the trading calendar and the data disagree. All seven models were rebuilt on one v1.1 dataset; the folds carried over by extension are byte-identical to the old launches, and no fold was altered.

**Design-level selection bias (the researcher chose the next step after seeing results): present, with one more layer this round.** The seven stages of section 4 are themselves a path of designing after seeing results; this round's candidate set and thresholds (section 5.3) were written after the first edition's out-of-sample results had been seen, and the first edition's P3 (2026-06 to 10-02) overlaps most of this round's P3 (07-01 to 10-01), so **this round's out-of-sample period is not clean for the candidate set and the thresholds**, while it is unseen for the retrained model scores. That is why the six passing cells do not go straight to paper trading but into a confirmation period that lies entirely after the data end.

| Decision | Date | Results seen when deciding | Numbers affected |
| --- | --- | --- | --- |
| 183 price-volume features, LightGBM hyperparameters, residual labels, orthogonalization, chip features, the three strategies | 10-22 to 10-28 | Itemized in the first edition's section 7.1 | All in-sample numbers |
| Threshold shortlist (Sharpe ≥ 1.5 in three periods, lower bound > 0) | 10-29 | The first edition's out-of-sample results | This round's shortlist |
| Candidate set widened to 21 cells, model-level Rank IC removed as a gate | 11-01 | All first-edition results | The shortlist (adds the transformer cells) |
| Out-of-sample period derived from the data end (June moved in sample) | 11-02 | All first-edition results; no number of this round yet | This round's P2 / P3 boundaries |
| Multi-day parameters fixed at (10, 50, 3) for all seven models, weekly long selected per model on TWSE first half | 11-02 | No strategy number of this round yet | The other five models' multi-day cells |
| Extension instead of retraining, model saving | 11-02 | This round's model-level numbers (Rank IC) were out, the strategy-level ones were not | None (the reused folds are byte-identical) |

Constructions tried: six generations of strategy layer, seven models, three strategies, 21 candidate cells; 7 shortlisted, 6 passed. Under multiple comparison the chance that some cell clears 1.5 in every period by luck is higher than with the first edition's six cells, and the thresholds were not tightened for it (tightening would itself be an after-the-fact decision). **In-sample numbers must be discounted; the confirmation period is the one single-shot test.**

#### 7.2 Data limitations (stated in advance)

(a) Adjusted prices carry ex-dividend / ex-rights factors only (with this round's opening-basis fix), no splits, capital reductions or par-value changes: 100 adjusted-open jumps above 25% across 91 names since 2023, 98 of them preceded by a halt and therefore excluded wholesale by the sample rule, the other 2 names outside the model's sample; no effect under the current rules.
(b) The in-sample second half was seen when the first edition's orthogonalization was designed; this round's candidate set and thresholds were set after the first edition's out-of-sample results; this round's out-of-sample period is clean only for the retrained scores.
(c) The out-of-sample period has 64 sessions and one market state, with intervals ±50 to ±120 points wide; the last 14 sessions have no label and therefore no prediction.
(d) Chip publication times follow a conservative rule (before the next open); the foreign-holding z-score column is always missing because of a source problem.
(e) The short leg's borrow-failure probability, recall probability and fee tiers are parameter assumptions; the main tables carry no disposition-stock or forced-cover constraint, the two rules of §6.7 are a sensitivity registered on 4 November, and they remain optimistic (no forced exit and no batched-auction slippage during a disposition; the forced cover is at least one day later than the real last cover day; the institution's cover at a suspension is an upper bound).
(f) The three transformers' results are not reproducible run to run (bf16, different GPUs); the three retrained folds differ from the old launches, and the report uses the extended current launches.
(g) The out-of-sample window holds the July 2026 correction and not the 2022 bear market; broker-branch and depository data were not bought.

### 8. Lessons and improvement directions

**Process lessons (added to the first edition's five)**

6. **Saving models is not a detail.** Two trainers ran for two weeks and seven rounds without a single model file; when it was noticed, no score could be recomputed for a new date. The fix (saving, extension, registry) means each quarter of the confirmation period only trains the new folds.
7. **Screen the candidates together; do not pick a model first.** The seven models differ by 0.01 to 0.02 in in-sample Rank IC but by Sharpe 0.8 to 2.4 at the strategy level; picking one model on Rank IC would have missed cells such as the transformers' long-short + hedge (1.80, 1.81).
8. **What the thresholds stopped.** All seven weekly longs exceed a whole-period Sharpe of 1.95, five were stopped by the first half; of the two that passed, F2's returned −13.4% out of sample. Ranking by whole-period Sharpe alone puts regime-dependent strategies first, confirmed again this round.
9. **State the direction of a data fix in advance.** The corrected prices cut the first edition's out-of-sample returns by 8 to 19 points, yet after the full-data retrain the long-short chip increment rose from +0.8% to +9.6% — a fix does not move results in one direction only.

**Improvement directions (in order; each registered before it is done)**

| Order | Item | Reason |
| --- | --- | --- |
| 1 | The six passing cells enter the 12-month confirmation period from 2026-10-23, each judged on its own (net return > 0, drawdown within twice its own in-sample drawdown); new folds added each quarter by extension; the interim reading (2027-04-22) is reported, not judged | The first clean out-of-sample period for the retrained models |
| 2 | Attribute the TPEX short leg's out-of-sample loss (short-band holdings and borrow fees during the rebound) | Every long-short cell's out-of-sample return came from TWSE |
| 2b | Decide whether "no new entry during a disposition period + forced cover at a suspension" becomes the primary execution assumption (the user's decision; adopting it means a separate registration with a fresh clock), and add the disposition-period slippage and the real margin last-cover day | The fall in §6.7 exceeds the pre-registered expectation; the ranking models concentrate both ends in stocks subsequently placed under disposition |
| 3 | Event studies: disposition, first attention-stock notice, reversal after a day-trade-ratio jump | Event signals in the free data not yet used |
| 4 | Retrain the cross-stock transformer on the chip features (the user's decision) | The chip increment holds at the ranking level, and the transformers' long-short + hedge is shortlisted |
| 5 | Buy broker-branch data (the user's decision; both pre-registered gates passed) | Literature and the sub-analysis point to branch flow for telling continuation from reversal among strong stocks |

### 9. Appendix

#### 9.1 Names used in this report and the code-base identifiers

| Name here | Code-base identifier | Current launch |
| --- | --- | --- |
| B0 price-volume LightGBM, rank labels | `t-lgbm-alpha` × `alpha_tw_v1` | `launch-20261101T212237Z-2508230824` |
| F0 price-volume LightGBM, residual label a (variant A) | `t-lgbm-alpha-residual_v1a` × `alpha_tw_v1` | `launch-20261101T220120Z-29216090` |
| Variant B price-volume LightGBM, residual label b | `t-lgbm-alpha-residual_v1b` × `alpha_tw_v1` | `launch-20261101T221954Z-1215615947` |
| F2 price-volume + chip LightGBM | `t-lgbm-alpha-residual_v1a` × `alpha_chip_tw_v1` | `launch-20261101T223737Z-1940816181` |
| D0 cross-stock transformer (from scratch) | `t-xs-scratch` | `launch-20261102T063519Z-189413204` |
| D1 Kronos-small | `t-xs-kronos-small` | `launch-20261102T071451Z-322624982` |
| D1 Chronos-2-small | `t-xs-chronos2-small` | `launch-20261102T073405Z-1322713845` |
| Long-short | S1 / `md_ls` | Band long-short on the 10-day score |
| Long-short + TX hedge | S2 / `md_ls_tx` | S1 plus a rolling-beta TX overlay |
| Weekly low-turnover long | S-A / `wk_long` | Weekly band long on the 5-day score |
| In-sample first / second half | P1 / P2 | 2023-01-02 to 2025-06-30 / 2025-07-01 to 2026-06-30 |
| Out of sample | P3 | 2026-07-01 to 2026-10-01 |
| Confirmation period | P4' | 12 months from 2026-10-23 |
| Threshold shortlist, out-of-sample verification, confirmation verdict | §13.20 supplement 1, §13.23 item 5 H-M, §13.22 item 3 H-L | `docs/experiment_protocol.md` |

#### 9.2 Artifacts and reproduction

- Research protocol (every experiment's pre-registration and result): `docs/experiment_protocol.md` §13.23 and its supplements; the first edition's sections §13.1 to §13.22.
- Model outputs: `artifacts/lgbm/<launch>/`, `artifacts/xs/<launch>/` (with `lineage.json` and `models/`); registry `registry/lgbm/*.json`, `registry/xs/*.json`.
- Backtest outputs: `artifacts/diagnostics/backtest/scenario-suite-p3b-21cells-20261102T082149Z/` (`grid.csv`, `daily.csv`, `yearly.csv`, `report.json`, `summary.md`); shortlist and out-of-sample verification: `strategy-decision-p3b-21cells-20261102T082429Z/`.
- Execution-layer sensitivity (§6.7): `scenario-suite-p3b-disp-20261104T113932Z/` (D), `scenario-suite-p3b-cover-20261104T114651Z/` (C), `scenario-suite-p3b-exec-20261104T114651Z/` (D + C) with the matching `strategy-decision-p3b-{disp,cover,exec}-…/`; configurations `backtest/config/scenario_suite_v1_p3b_{disp,cover,exec}.json`; the pre-correction first run `…-cover-20261104T113932Z/`, `…-exec-20261104T113932Z/` is kept.
- Reproduction: `cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3b.json --chip-data-root <tw_chip_data/data> --multi-day-default F0 --predictions ../artifacts/lgbm/<B0>/predictions.parquet --model t-lgbm-alpha --label B0 … --prices ../artifacts/datasets/<v1.1 dataset>/raw/market.parquet`, then `uv run python -m twbacktest.strategy_decision --config config/strategy_decision_p3b.json --suite <suite output directory>`; the full command is in the protocol's §13.23 step 18 implementation note.
- Code: `src/stock_forecasting/` (data, features, models, extension and registry), `backtest/twbacktest/` (strategies and backtests), `scripts/` (cloud CPU / GPU workflows and downloads).
