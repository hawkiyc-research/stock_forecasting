# 台股量化選股策略：模型、交易策略與樣本外驗證 / A Quantitative Taiwan Stock Strategy: Model, Trading Rules and Out-of-Sample Validation

[繁體中文](#繁體中文) · [English](#english)

---

## 繁體中文

### 1. 一頁摘要

這份報告記錄一個完整的量化研究流程：用台灣上市、上櫃股票的日線價量與免費公開的籌碼資料，訓練一個橫斷面排序模型（預測每天哪些股票未來幾天相對較強），把排序轉成三種可交易的策略，在機構與散戶兩種成本下回測，並用**事前登記的規則**在一段完全未看過的樣本外期間確認結果。

| 項目 | 內容 |
| --- | --- |
| 資料 | 台灣上市（TWSE）與上櫃（TPEX）全部股票，2016 年起的日線價量；免費官方籌碼資料（三大法人買賣超、融資融券、借券、當沖統計、注意與處置股）；台指期官方日資料 |
| 模型 | LightGBM 橫斷面排序模型，183 欄價量特徵 + 67 欄籌碼特徵，標籤為扣除市場、市值與產業效應後的 5 到 10 日相對報酬；三個隨機種子集成；每季重新訓練、只用過去資料（16 個 walk-forward 折） |
| 策略 | 三種：多日多空、多空加台指期避險、週頻低換手多頭 |
| 成本 | 機構（手續費 0.03%、滑價 0.05%、借券費分層）與散戶（手續費 0.0855%、滑價 0.1% 到 0.2%）兩檔並列；放空受可借券名單、交易所限額與借不到機率限制 |
| 樣本內 | 2023-01 至 2026-05（約 790 個交易日），用來選參數 |
| 樣本外 | 2026-06 至 2026-10（85 個交易日），事前登記只看一次 |
| 選定結果 | 原本登記的規則（樣本內淨 Sharpe 最高的一格）選出價量＋籌碼模型 × 週頻低換手多頭（Sharpe 2.38、年化 +51.8%、回撤 −22.6%）。事後修正為門檻篩選（三段期間 Sharpe 各 ≥ 1.5、區間下界 > 0）後入選三個：價量＋籌碼 × 多空＋期貨避險（1.98）、價量 × 多空＋期貨避險（1.78）、價量＋籌碼 × 多日多空（1.57）；週頻多頭因前段 Sharpe 只有 1.44 落選 |
| 樣本外 | 原選定的週頻多頭：年化 +3.5%、回撤 −25.7%，確認條件「淨報酬為正」成立、「回撤不比樣本內深」**不成立**。門檻入選的三個：多空＋期貨避險（價量＋籌碼）+34.1%、Sharpe 2.19、回撤 −8.1%；多空＋期貨避險（價量）+27.5%、1.74、−6.3%；多日多空（價量＋籌碼）+30.9%、2.01、−8.6%——全部為正、回撤都在樣本內的兩倍以內，但只有價量版的多空＋避險完全不比樣本內深 |
| 結論 | **依事前登記的規則，確認未通過，沒有策略進入模擬交易。** 單一最高格選到的是靠 2025 年下半年多頭行情衝高的策略，7 月一次 −17% 的修正就把它打掉；門檻篩選留下的三個市場中性策略在同一窗口 +27% 到 +34%、回撤 −6% 到 −9%。因為門檻規則是看了樣本外之後才改的，它的結果只作資訊；三個入選策略一起進入下一段 12 個月的確認期（第 10 節） |

這份報告的價值不在「找到一個會賺錢的策略」，而在整個流程：嚴格的時間點資料、不重複使用測試資料、把成本與執行限制寫進回測、事前登記判定規則、結果不好也照實報告。

### 2. 研究問題

在台灣股市，日線價量與公開籌碼資料能不能做出一個**扣除真實交易成本後**仍有效的選股策略？三個子問題：

1. 排序能力：模型能不能穩定分辨未來幾天相對強弱的股票？
2. 可交易性：排序能力經過換手、成本、放空限制與資金規模後還剩多少？
3. 籌碼資料的增量：免費的法人、融資券、借券與當沖資料，在價量之外有沒有額外資訊？

### 3. 資料與防止偷看未來的規則

- **價量**：2016-01-01 起全部上市、上櫃股票的開高低收量；還原價只處理股息（分割與減資的調整已實作但尚未用於本輪，見第 9 節限制）。
- **籌碼**：來自另一個資料專案的 12 個官方品項。每一列都記錄「什麼時候公布」，模型在訊號日 t 只能用 t 收盤後、次日開盤前已公布的資料。
- **交易日曆**：使用固定版本的交易所日曆，臨時休市（例如 2026-07-10、2026-10-23 兩次颱風休市）核對交易所公告後補列；資料與日曆不一致時流程會自動中止並列出日期。
- **樣本規則**：每個樣本要有連續 128 個交易日的輸入與接續 14 個交易日的輸出，缺一天就排除；流動性門檻（前 60 日成交金額中位數）只用過去資料計算。
- **訓練與預測分離**：每季重新訓練一次，訓練資料截止於預測季開始前 15 個交易日（避免標籤跨期），驗證用最後 12 個月；2023 年第一季到 2026 年第四季共 16 折。模型從不接觸它要預測那一季的任何資料。

### 4. 模型

- **任務**：每天在同一市場內，預測每檔股票未來 h 天（h = 1 到 14）的**相對**表現排序，不預測絕對漲跌。
- **標籤**：先把 h 日持有報酬對市場、市值與產業做橫斷面回歸，取殘差再排名。這一步讓模型學的是個股相對優勢，而不是「大型股行情來了就全部買大型股」。
- **特徵**：價量 183 欄（報酬、波動、量能、價格位置、動能與反轉等多個視窗）；籌碼 67 欄（五類法人買賣超的 1／5／10／20 日累計比與 z 分數、融資融券與借券餘額變化、當沖比、注意與處置狀態與距離）。
- **模型**：LightGBM，每個 h 一個模型，三個隨機種子平均；超參數固定，不在測試期調整。
- **兩個版本並列**：只用價量（對照組）、價量加籌碼（實驗組），用同一組折、同一組標籤配對比較。
- **排序能力**（樣本內 791 個交易日的日 Rank IC 平均）：價量模型短天期 0.117、5 到 10 日 0.089；加籌碼後 0.128、0.107。籌碼對 5 到 10 日排序的增量 +0.017（95% 區間 +0.013 到 +0.021），在高流動性宇宙裡前後四分位的 10 日價差由 0.68% 提高到 1.08%。

### 5. 三種交易策略

所有策略都在**流動性宇宙**內運作：前 20 日成交金額中位數上市 ≥ 3,000 萬、上櫃 ≥ 1,000 萬，收盤價 ≥ 10 元，排除 ETF。訊號在收盤後產生，次一交易日開盤成交；漲停開盤不買、跌停開盤不賣。模型分數先在宇宙內取百分位、做 3 日半衰期的指數平滑，再每天對 60 日 beta、前 20 日與前 60 日報酬做橫斷面回歸取殘差——這一步把「低 beta、反動能」的隱性傾斜拿掉，是整個研究中最關鍵的一個設計。

| 策略 | 規則 | 曝險 |
| --- | --- | --- |
| **多日多空** | 10 日分數。多頭：前 50 名等額持有，掉出前 150 名才賣，每天最多換 5 檔（買賣緩衝區大幅降低換手）；空頭：同樣規則用在反向排序，只能放空可借券名單內的股票、受交易所次日限額與「借不到」機率約束，借券費依規模分層 | 金額中性 |
| **多空加台指期避險** | 同上，再用組合過去 60 日對指數的滾動 beta 掛台指期，把殘餘市場曝險對掉 | 金額中性加 beta 對沖 |
| **週頻低換手多頭** | 5 日分數。每週最後一個交易日再平衡：前 30 名等額持有，掉出前 60 名才賣，每週最多換掉 15 檔（換手上限 50%）；不放空、不用期貨 | 100% 多頭 |

K、緩衝倍數等參數在樣本內的前段（2023-01 至 2025-05、上市、機構成本）用淨 Sharpe 選定一次，之後固定。

### 6. 成本與執行假設

| 成本檔 | 手續費（每邊） | 證交稅（賣出） | 滑價（每邊） | 放空 | 台指期 |
| --- | --- | --- | --- | --- | --- |
| 機構（主要） | 0.03% | 0.3% | 0.05% | 借券費年率 1.5%／3%／6%（依規模） | 期交稅 0.002%、每口 30 元、1 tick |
| 散戶 | 0.0855% | 0.3% | 上市 0.1%、上櫃 0.2% | 融券手續費 0.08% | 每口 50 元、1 tick |

另列滑價 0 倍與 2 倍的敏感度，以及新台幣 3 億與 10 億的資金情境（單一標的名目不超過其 20 日成交金額中位數的 5%，超出部分留現金）。放空腿的借不到機率與召回機率各抽五次隨機重複取平均，並列不同倍率的敏感度。

### 7. 結果

#### 7.1 樣本內：全部六個候選

六個候選 = 兩個模型 × 三種策略，機構成本、上市上櫃各半、分數位置。Sharpe 分三段看，因為這正是策略會不會「只在某種行情賺」的證據。

| 模型 | 策略 | 2023-01 至 2025-05 Sharpe（年化淨報酬） | 2025-06 至 2026-05 Sharpe（年化淨報酬） | 全樣本內 Sharpe | 全樣本內年化淨報酬 [95% 區間] | 全樣本內最大回撤 |
| --- | --- | --- | --- | --- | --- | --- |
| 價量＋籌碼 | 週頻多頭 | 1.44（+27.6%） | 4.13（+111.2%） | 2.38 | +51.8% [+26.1%, +77.7%] | −22.6% |
| 價量 | 週頻多頭 | 1.44（+23.2%） | 3.39（+65.6%） | 2.07 | +35.5% [+17.2%, +54.7%] | −18.7% |
| 價量＋籌碼 | 多空＋期貨避險 | 1.88（+22.0%） | 2.21（+33.7%） | 1.98 | +25.4% [+13.9%, +38.0%] | −6.1% |
| 價量 | 多空＋期貨避險 | 1.57（+21.5%） | 2.29（+31.2%） | 1.78 | +24.3% [+10.4%, +38.0%] | −8.5% |
| 價量＋籌碼 | 多日多空 | 1.51（+16.7%） | 1.71（+25.1%） | 1.57 | +19.1% [+7.5%, +31.8%] | −8.4% |
| 價量 | 多日多空 | 1.32（+16.6%） | 1.74（+22.5%） | 1.44 | +18.3% [+3.6%, +32.2%] | −10.2% |

對照：同期等權持有流動性宇宙年化 +31.5%（回撤 −18.7%）、0050 +48.3%（回撤 −25.2%）、加權報酬指數 +38.3%。多空策略的 +19% 到 +25% 是市場中性的報酬，不該直接與指數相比；週頻多頭是 100% 多頭。

看這張表就看得出問題：兩個週頻多頭的全樣本內 Sharpe 最高，但幾乎全靠 2025 年 6 月之後的多頭行情（Sharpe 3.4 到 4.1），前段只有 1.44；三個多空策略在兩段期間的 Sharpe 都在 1.5 到 2.3 之間，報酬低但穩。

#### 7.2 兩種選定規則

- **原本事前登記的規則**：全樣本內淨 Sharpe 最高的一格 → 選出價量＋籌碼模型 × 週頻低換手多頭（2.38）。
- **修正後的規則**（看過樣本外結果後、由研究負責人指示修改，所以對這一段樣本外是事後規則，照實揭露；之後的確認期從一開始就用它）：三段期間的淨 Sharpe **各自 ≥ 1.5**，且全樣本內淨報酬區間下界 > 0。門檻取 1.5 的理由：業界常用的 1.0 六格全過、沒有篩選作用；2.0 沒有任何一格每段都過；1.5 是「每一段都明顯高於 1」的最低半格，不是對樣本外結果調出來的。→ 入選三格：價量＋籌碼 × 多空＋期貨避險、價量 × 多空＋期貨避險、價量＋籌碼 × 多日多空。兩個週頻多頭（前段 1.44）與價量 × 多日多空（前段 1.32）落選。

#### 7.3 樣本外（2026-06-01 至 2026-10-02，85 個交易日）：全部六格

事前登記的確認條件：(i) 樣本外年化淨報酬 > 0；(ii) 樣本外最大回撤不比樣本內深。另加一個對照讀法：(ii′) 回撤在樣本內的兩倍以內——樣本內的放空腿是五次隨機重複的平均，平均會壓低回撤，所以「不更深」偏嚴。

| 模型 | 策略 | 入選 | 樣本外年化淨報酬 [95% 區間] | Sharpe | 樣本外最大回撤 | 樣本內最大回撤 | (i) | (ii) 不更深 | (ii′) 兩倍內 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 價量＋籌碼 | 週頻多頭（原選定） | 否 | +3.5% [−110.8%, +110.6%] | 0.08 | −25.7% | −22.6% | ✓ | ✗ | ✓ |
| 價量 | 週頻多頭 | 否 | +27.5% [−38.3%, +101.1%] | 1.05 | −14.8% | −18.7% | ✓ | ✓ | ✓ |
| 價量＋籌碼 | 多空＋期貨避險 | **是** | +34.1% [−19.0%, +90.1%] | 2.19 | −8.1% | −6.1% | ✓ | ✗ | ✓ |
| 價量 | 多空＋期貨避險 | **是** | +27.5% [−16.9%, +67.1%] | 1.74 | −6.3% | −8.5% | ✓ | ✓ | ✓ |
| 價量＋籌碼 | 多日多空 | **是** | +30.9% [−25.9%, +90.0%] | 2.01 | −8.6% | −8.4% | ✓ | ✗ | ✓ |
| 價量 | 多日多空 | 否 | +23.3% [−23.4%, +64.5%] | 1.50 | −6.9% | −10.2% | ✓ | ✓ | ✓ |

各月（機構、兩市場各半）：7 月等權流動性宇宙 −17.0%、0050 −6.5%；原選定的週頻多頭 −19.5%，價量＋籌碼的多空＋期貨避險 +15.6%。

#### 7.4 怎麼讀

- **依事前規則，確認未通過，沒有策略進入模擬交易**：規則選出的週頻多頭是 100% 多頭、beta 接近 1，回撤由市場決定；85 天裡一次 −17% 的修正就讓「回撤不更深」失敗。這個結果照實記錄，不能事後換格。
- **門檻篩選留下的是另一類策略**：三個市場中性的多空版本在前後兩段的 Sharpe 都穩在 1.5 到 2.3，樣本外 +27% 到 +34%、回撤 −6% 到 −9%，而且是在多頭策略賠錢的那個 7 月賺的。它們的樣本外數字區間很寬（±50 到 ±60 個百分點），85 天分不出好壞，但「回撤在可控範圍」這件事已經和週頻多頭明顯不同。
- **報酬與穩定的取捨**：週頻多頭樣本內年化 +52%、回撤 −23%，多空＋期貨避險 +25%、回撤 −6%。前者在多頭行情裡會大贏，碰到修正會回吐四分之一；後者什麼行情都賺得慢但賠得少。哪一個「比較好」取決於能承受多大的回撤，不是一個 Sharpe 數字能決定的，這也是改用門檻篩選的原因。
- **籌碼資料的角色**：排序層有增量（+0.017）；策略層只對多頭有增量（+16.3%／年），對多空幾乎沒有（+0.8% 到 +1.0%，區間跨 0）——放空腿沒有從籌碼得到好處。所以在入選的三個候選裡，價量＋籌碼版比價量版的優勢不大。
- **下一步**：三個入選策略一起進入 2026-10-23 起 12 個月的確認期（第 10 節），各自判定。

### 8. 樣本是怎麼用的：洩漏與設計偏誤的交代

「污染」有兩種，分開交代。

**(1) 資料洩漏（未來資訊進到特徵、標籤、訓練或策略輸入）：沒有。** 每個特徵只用訊號日之前的資料；籌碼只用訊號日次日開盤前已公布的列；標籤的市值代理是過去 20 日成交金額；每一季的模型只用該季開始前 15 個交易日以前的資料訓練，提早停止只看訓練窗內最後 12 個月；策略用的 beta、動能、流動性都是往回看的視窗；交易日曆與資料不一致時流程會中止。樣本外期間（2026-06 至 10 月）的資料在所有模型、特徵、策略與判定規則凍結之後（10 月 28 日）才下載；樣本內的「全樣本內」統計明確排除樣本外。

**(2) 設計層的選擇偏誤（研究者看了結果之後決定下一步）：有，列表如下。** 這不是洩漏，而是任何「先用歷史找策略、再做樣本外」的流程都會有的性質；業界處理它的方式是把樣本內數字打折、只信樣本外、再加一段模擬交易。

| 決定 | 日期 | 做決定時已看過的結果 | 受影響的數字 |
| --- | --- | --- | --- |
| 183 欄價量特徵、LightGBM 超參數 | 10-22 | 無（事前固定） | — |
| 改用殘差化標籤（扣市場、市值、產業） | 10-23 | 原始標籤在樣本內後段（2025-06 後）的失效 | 兩個模型的樣本內排序能力 |
| 分數對 beta 與動能的正交化 | 10-25 | 多空策略在後段的虧損與暴露診斷 | 多日多空、多空＋避險的樣本內 |
| 放空執行層（名單、限額、借不到機率） | 10-26 | 樣本內全段 | 同上（方向是讓數字更保守） |
| 67 欄籌碼特徵 | 10-26 | 價量模型的樣本內結果；特徵內容依文獻設計，沒有對後段調整 | 價量＋籌碼模型的樣本內 |
| 週頻多頭的設計（週頻、5 日分數、純多頭、換手上限） | 10-25 登記、10-28 實作 | 後段的多頭行情與多空策略的結果 | 週頻多頭的樣本內，尤其後段 +65% 到 +111% |
| 各策略的 K、m（與多空的 h） | 計分時 | 只用樣本內前段（2023-01 至 2025-05）選 | 對參數而言後段未看過；對設計而言不是 |
| 選定規則與確認條件 | 10-28 | 登記時未看樣本外，也未看重訓後的樣本內格子 | — |

試過的建構：策略層六代（每日換股 → 緩衝區 → 殘差標籤 → 正交化 → 執行層 → 週頻多頭）、兩個模型、三個市場，最後六格候選。選定格的樣本內 Sharpe 2.38 是在這些嘗試之後的最大值，**樣本內數字要打折看**；樣本外那 85 天是唯一一次性的檢驗，而它沒有通過。

**(3) 樣本外的獨立性有一個例外，照實交代。** 2026-06-01 至 09-24 這段資料在 10 月 20 日對更早的基礎模型訊號做過一次前瞻檢驗（結果：排序能力為正、所有區間跨 0），之後才有本報告的模型與策略。那次檢驗沒有被用來做任何設計決策，但無法證明研究者對那段行情的印象沒有影響後續。嚴格的說法是：樣本外對本報告的模型與策略是未看過的，不是從未被任何分析碰過的。另外，前四個階段反覆使用過的固定測試集（2025-12 至 2026-05）早已失去樣本外身分，本報告只把它當作歷史。

**(4) 一個小的事後成分。** 殘差化用 2026 年 10 月的公司產業分類快照套到整段歷史；分類變更與已下市公司（歸入「未知」類）帶有輕微的事後與存活者成分，只影響產業虛擬變數。

### 9. 限制（事前聲明，不是事後找理由）

1. 樣本外只有 85 個交易日、一種市場狀態；通過或不通過都不能外推成長期結論。
2. 樣本內的後段（2025-06 之後）在設計正交化時已經看過，所以樣本內的數字只是資訊，真正的檢驗只有樣本外那 85 天。
3. 還原價只處理股息，不含分割、減資與面額變更；2023 年起超過 25% 的單日還原開盤跳動共 100 次、91 檔（樣本外期間 19 次、19 檔）。其中 98 次前面有停牌：樣本規則要求訊號日前 128 個交易日與之後 14 個交易日都沒有缺口，所以停牌前後的樣本整段排除，特徵、標籤、分數與持股報酬都碰不到跳動；另外 2 次是股息還原因子套錯日期，這 2 檔在事件前後都不在模型的樣本內。逐一核對後，這些事件沒有一次進入訓練或任一策略的損益。修正版（分割、減資、面額變更）已寫好、尚未重訓；在目前的規則下預期不會改變結果。另外在 2026-10-29 發現第二個、影響較大的還原價問題：有償現金增資的除權日，交易所的開盤競價基準停在前一日收盤價，但資料管線用理論參考價去還原，等於替每一筆現增製造一個不存在的上漲（2016 年起 861 筆、2023 年起 386 筆，誤差中位數約 1.9%、最大 32%）；這些事件沒有停牌，樣本規則擋不住，而且集中在 6 到 9 月的除權息旺季，剛好是樣本外期間（40 筆）。把修正後的價格套回同一組模型預測重算，樣本外三個入選策略的年化淨報酬從 +34.1%、+27.5%、+30.9% 降為 +14.7%、+19.4%、+13.7%，樣本內的變動在 ±4 個百分點內；仍為正、回撤仍在門檻內，但本報告樣本外的數字應視為偏高。資料管線已修正，重訓前的模型分數仍帶有這個誤差。
4. 放空腿的借不到機率、召回機率與借券費分層是參數假設，不是實際成交紀錄。
5. 籌碼只用免費官方資料；分點（券商）與集保資料沒有購買。
6. 樣本外視窗含 2025 年 4 月急跌與 2026 年 7 月修正，不含 2022 年空頭。

### 10. 下一步

1. 三個入選的市場中性策略一起進入 2026-10-23 起 12 個月的確認期（規則於 2026-10-29 定案並凍結；第 6 個月期中只讀不判）：每季把資料延到季末、重訓新增的那一季、累積每日淨報酬；期滿用事前寫好的兩條規則各自判定——淨報酬 > 0，且最大回撤不深於各自樣本內回撤的兩倍。通過的進模擬交易。
2. 週頻多頭加 β 對沖已補測（附錄 B）：白名單內的工具救不回它，不再列為候選。
3. 用修正分割與減資後的資料重訓，重算全部數字（另案）。

### 附錄 B：補充回測——幫週頻多頭加上 β 對沖（2026-10-29）

樣本外確認失敗後，做了一輪只動策略層、不動模型與資料的補充回測：在週頻低換手多頭上加一條每天重設的對沖腿，名目 = −β × 資本，β 用策略過去 60 個交易日對對沖工具的滾動回歸估計。對沖工具兩種：放空台指期，以及放空 0050（資料集白名單裡唯一同時有流動性與可借券的 ETF）。因為這個設計是看過樣本外結果之後才決定的，這一輪全部只作資訊，不能當成樣本外證據。

| 版本（價量＋籌碼模型，機構成本，兩市場各半） | 樣本內年化淨報酬 | 95% 區間 | 淨 Sharpe | 樣本內最大回撤 | 樣本外年化淨報酬 | 樣本外最大回撤 |
| --- | --- | --- | --- | --- | --- | --- |
| 週頻多頭（原版） | +51.8% | [+26.1%, +77.7%] | 2.38 | −22.6% | +3.5% | −25.7% |
| 加台指期對沖 | +25.4% | [+7.5%, +45.6%] | 1.21 | −11.4% | −15.7% | −18.8% |
| 加放空 0050 | +16.6% | [−1.2%, +33.6%] | 1.05 | −17.7% | −31.6% | −18.5% |
| 對照：多空＋期貨避險 | +25.4% | [+13.9%, +38.0%] | 1.98 | −6.1% | +34.1% | −8.1% |

結果：對沖把 Sharpe 砍掉一半、回撤只縮一部分，樣本外反而更差（7 月當月 −19.5% 變成 −13%）。原因是對沖到錯的風險：這個多頭組合接近等權、偏中小型股，對台積電主導的加權指數 β 只有 0.5 到 0.8；2026 年 7 月跌的是廣度（等權宇宙 −17%），不是權值股（0050 −6.5%）。用指數工具對沖等權組合，在權值股獨漲時付出大量成本、在廣度下跌時保護不到。這和研究歷程中「台指期對沖版失敗」、「0050 比所有策略都強」是同一件事的三個面向：這段期間台股多頭組合的風險在「等權對市值加權」的差距，不在指數 β。結論：白名單內的工具救不了純多頭策略；要控回撤，市場中性的多空策略（空頭腿本身就是等權的對沖）才是正確的主線。

### 附錄 A：本文名稱與程式庫內部代號的對照

| 本文名稱 | 程式庫代號 | 說明 |
| --- | --- | --- |
| 價量模型 | F0 / `alpha_tw_v1` | 183 欄價量特徵的 LightGBM 排序模型 |
| 價量＋籌碼模型 | F2 / `alpha_chip_tw_v1` | 183 欄價量 + 67 欄籌碼 |
| 殘差化標籤 | `residual_v1a` | 扣除市場、市值、產業效應的 h 日報酬 |
| 多日多空 | S1 / `md_ls` | 10 日分數的 band 多空組合 |
| 多空＋期貨避險 | S2 / `md_ls_tx` | S1 加台指期滾動 beta 疊加 |
| 週頻低換手多頭 | S-A / `wk_long` | 5 日分數的週頻 band 多頭 |
| 樣本內前段／後段 | P1 / P2 | 2023-01 至 2025-05／2025-06 至 2026-05 |
| 樣本外 | P3 | 2026-06-01 至 2026-10-02 |
| 樣本外確認條件 | H-J | 淨報酬 > 0 且回撤不比樣本內深 |
| 研究協定章節 | §13.20 | `docs/experiment_protocol.md` |

程式與完整表格：`backtest/`（回測）、`src/stock_forecasting/`（資料與模型）、`docs/research_full_report.md`（完整報告）。

---

## English

### 1. One-page summary

This report documents an end-to-end quantitative research cycle: a cross-sectional ranking model trained on daily price-volume data and free public "chip" data (institutional flows, margin, securities lending) for every listed stock in Taiwan, three tradeable strategies built from the ranking, backtests under institution and retail costs, and a confirmation on a fully unseen out-of-sample window under **pre-registered** rules.

| Item | Content |
| --- | --- |
| Data | All TWSE and TPEX stocks, daily OHLCV from 2016; free official chip data (institutional net buying, margin and short balances, securities lending, day-trade statistics, attention and disposition lists); official TAIEX futures daily data |
| Model | LightGBM cross-sectional ranker, 183 price-volume features + 67 chip features, labels = 5- to 10-day relative returns net of market, size and industry effects; three-seed ensemble; retrained every quarter on past data only (16 walk-forward folds) |
| Strategies | Three: multi-day long-short, long-short with a TAIEX futures hedge, weekly low-turnover long |
| Costs | Institution (commission 0.03%, slippage 0.05%, tiered borrow fees) and retail (commission 0.0855%, slippage 0.1% to 0.2%) side by side; shorts limited by the borrowable list, exchange quotas and borrow-failure probabilities |
| In-sample | 2023-01 to 2026-05 (about 790 sessions), used for parameter selection |
| Out-of-sample | 2026-06 to 2026-10 (85 sessions), pre-registered and read once |
| Selected | The originally registered rule (the single cell with the highest in-sample net Sharpe) picked the price-volume + chip model × weekly low-turnover long (Sharpe 2.38, +51.8% a year, drawdown −22.6%). The rule was amended afterwards to a threshold shortlist (Sharpe ≥ 1.5 in each of three periods, lower bound > 0), which keeps three cells: price-volume + chip × L/S + futures hedge (1.98), price-volume × L/S + futures hedge (1.78), price-volume + chip × multi-day long-short (1.57); the weekly long drops out on its first-segment Sharpe of 1.44 |
| Out-of-sample | The originally selected weekly long: +3.5% a year, drawdown −25.7%; "positive net return" holds, "drawdown not deeper than in-sample" **fails**. The three shortlisted cells: L/S + futures hedge (price-volume + chip) +34.1%, Sharpe 2.19, drawdown −8.1%; L/S + futures hedge (price-volume) +27.5%, 1.74, −6.3%; multi-day long-short (price-volume + chip) +30.9%, 2.01, −8.6% — all positive, all within twice their in-sample drawdown, but only the price-volume L/S + hedge is strictly no deeper than in-sample |
| Data correction (2026-10-29) | An adjusted-price error on paid rights issues was found, concentrated in the out-of-sample window; re-running the same predictions on corrected prices lowers the three shortlisted strategies' out-of-sample annual net return from +34.1% / +27.5% / +30.9% to +14.7% / +19.4% / +13.7%, still positive; the out-of-sample figures in this report should be read as overstated (section 9, item 3) |
| Conclusion | **Under the pre-registered rule the confirmation is not passed and no strategy enters paper trading.** The single-argmax rule picked a strategy that rode the late-2025 bull market; one −17% correction in July knocked it out. The three market-neutral strategies kept by the threshold rule made +27% to +34% with −6% to −9% drawdowns in the same window. Because the threshold rule was changed after the out-of-sample result was seen, its result is informational; the three shortlisted strategies enter the next twelve-month confirmation period together (section 10) |

The value of this work is the process rather than a winning strategy: strict point-in-time data, no reuse of test data, costs and execution limits inside the backtest, decision rules registered before scoring, and an honest report when the answer is no.

### 2. Research question

Can daily price-volume data and public chip data produce a stock-selection strategy in Taiwan that still works **after real transaction costs**? Three sub-questions:

1. Ranking skill: can a model consistently separate stocks that will be relatively strong over the next few days?
2. Tradeability: how much of that skill survives turnover, costs, short-selling constraints and capital size?
3. The chip increment: do the free institutional, margin, lending and day-trade data add information beyond prices and volumes?

### 3. Data and the rules against look-ahead

- **Prices**: OHLCV of every TWSE and TPEX stock from 2016-01-01; adjusted prices carry dividends only (split and capital-reduction adjustments are implemented but not yet used in this round, see section 9).
- **Chip data**: 12 official items from a separate data project. Every row records when it was published; on signal date t the model may only use rows published after t's close and before the next open.
- **Trading calendar**: a pinned exchange-calendar release, with unscheduled closures (for example the typhoon closures of 2026-07-10 and 2026-10-23) added after checking the exchange announcements; the pipeline stops and lists the dates whenever the data and the calendar disagree.
- **Sample rule**: every sample needs 128 consecutive sessions of input and the following 14 sessions of output; a single missing session excludes it. The liquidity screen (60-session median dollar volume) uses past data only.
- **Training separated from prediction**: retrained every quarter, with training data ending 15 sessions before the predicted quarter (no label overlap) and the last 12 months as validation; 16 folds from 2023Q1 to 2026Q4. The model never sees any data from the quarter it predicts.

### 4. The model

- **Task**: each day, within each market, rank every stock by its **relative** performance over the next h sessions (h = 1 to 14); no absolute up/down call.
- **Labels**: the h-session holding return is first regressed cross-sectionally on market, size and industry; the residual is ranked. The model therefore learns stock-specific strength rather than "buy every large cap when large caps rally".
- **Features**: 183 price-volume columns (returns, volatility, volume, price position, momentum and reversal over several windows); 67 chip columns (1 / 5 / 10 / 20-day cumulative ratios and z-scores of five institutional investor classes' net buying, margin and short-balance changes, securities lending, the day-trade ratio, attention and disposition status and distance).
- **Model**: LightGBM, one model per h, three random seeds averaged; hyperparameters fixed, never tuned on the test period.
- **Two versions side by side**: price-volume only (control) and price-volume + chip (treatment), paired on the same folds and labels.
- **Ranking skill** (mean daily Rank IC over 791 in-sample sessions): price-volume model 0.117 at short horizons and 0.089 at 5 to 10 days; with chip data 0.128 and 0.107. The chip increment at 5 to 10 days is +0.017 (95% interval +0.013 to +0.021); in the liquid universe the top-minus-bottom quartile 10-day spread rises from 0.68% to 1.08%.

### 5. The three trading strategies

Every strategy works inside a **liquidity universe**: 20-session median dollar volume ≥ NT$30M on TWSE and ≥ NT$10M on TPEX, close ≥ NT$10, no ETFs. Signals are formed after the close and executed at the next session's open; nothing is bought at a limit-up open or sold at a limit-down open. The model score is converted to a within-universe percentile, smoothed with a 3-day half-life, and then regressed each day on 60-day beta and trailing 20- and 60-day returns, keeping the residual — this removes a hidden low-beta, anti-momentum tilt and is the single most important design step in the project.

| Strategy | Rule | Exposure |
| --- | --- | --- |
| **Multi-day long-short** | 10-day score. Long book: top 50 names in equal slots, sold only when a name falls out of the top 150, at most 5 new entries a day (the hold buffer cuts turnover sharply). Short book: the same rule on the reversed ranking, limited to names on the borrowable list, to the exchange's next-day lending quota and to a borrow-failure probability, with size-tiered borrow fees | Dollar neutral |
| **Long-short + futures hedge** | As above, plus a TAIEX futures position sized by the book's trailing 60-day rolling beta to hedge the residual market exposure | Dollar neutral plus beta hedge |
| **Weekly low-turnover long** | 5-day score. Rebalanced on the last session of each week: top 30 names in equal slots, sold only when a name falls out of the top 60, at most 15 replacements a week (50% turnover cap); no shorts, no futures | 100% long |

The slot count and buffer were selected once by net Sharpe on the first in-sample segment (2023-01 to 2025-05, TWSE, institution costs) and then fixed.

### 6. Cost and execution assumptions

| Profile | Commission (per side) | Transaction tax (sale) | Slippage (per side) | Shorts | TAIEX futures |
| --- | --- | --- | --- | --- | --- |
| Institution (primary) | 0.03% | 0.3% | 0.05% | borrow fee 1.5% / 3% / 6% a year by size | tax 0.002%, NT$30 per contract, 1 tick |
| Retail | 0.0855% | 0.3% | TWSE 0.1%, TPEX 0.2% | margin short fee 0.08% | NT$50 per contract, 1 tick |

Slippage at 0× and 2× and capital scenarios of NT$300M and NT$1B (no position above 5% of the name's 20-day median dollar volume, the rest in cash) are reported as sensitivities. The short book's borrow-failure and recall draws are averaged over five random replicates, with multipliers reported as sensitivities.

### 7. Results

#### 7.1 In-sample: all six candidates

Six candidates = two models × three strategies, institution costs, TWSE and TPEX half each, fractional capital. Sharpe is shown for both in-sample segments, because that is the evidence of whether a strategy earns only in one kind of market.

| Model | Strategy | 2023-01 to 2025-05 Sharpe (net return a year) | 2025-06 to 2026-05 Sharpe (net return a year) | All in-sample Sharpe | All in-sample net return [95% interval] | All in-sample max drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| Price-volume + chip | Weekly long | 1.44 (+27.6%) | 4.13 (+111.2%) | 2.38 | +51.8% [+26.1%, +77.7%] | −22.6% |
| Price-volume | Weekly long | 1.44 (+23.2%) | 3.39 (+65.6%) | 2.07 | +35.5% [+17.2%, +54.7%] | −18.7% |
| Price-volume + chip | L/S + futures hedge | 1.88 (+22.0%) | 2.21 (+33.7%) | 1.98 | +25.4% [+13.9%, +38.0%] | −6.1% |
| Price-volume | L/S + futures hedge | 1.57 (+21.5%) | 2.29 (+31.2%) | 1.78 | +24.3% [+10.4%, +38.0%] | −8.5% |
| Price-volume + chip | Multi-day long-short | 1.51 (+16.7%) | 1.71 (+25.1%) | 1.57 | +19.1% [+7.5%, +31.8%] | −8.4% |
| Price-volume | Multi-day long-short | 1.32 (+16.6%) | 1.74 (+22.5%) | 1.44 | +18.3% [+3.6%, +32.2%] | −10.2% |

Benchmarks over the same window: equal-weight holding of the liquidity universe +31.5% a year (drawdown −18.7%), 0050 +48.3% (drawdown −25.2%), TAIEX total-return index +38.3%. The long-short strategies' +19% to +25% is market-neutral return and should not be compared with an index; the weekly long is 100% long.

The table shows the problem at a glance: the two weekly longs have the highest all-in-sample Sharpe, but almost all of it comes from the bull market after June 2025 (Sharpe 3.4 to 4.1) while the first segment is only 1.44; the three long-short strategies sit between 1.5 and 2.3 in both segments, lower return but steady.

#### 7.2 Two selection rules

- **The rule registered in advance**: the single cell with the highest all-in-sample net Sharpe → price-volume + chip model × weekly low-turnover long (2.38).
- **The amended rule** (changed at the research lead's instruction after the out-of-sample result was seen, hence post hoc for this window and disclosed as such; the next confirmation period uses it from the start): net Sharpe **≥ 1.5 in each of the three periods**, and the all-in-sample net-return lower bound > 0. Why 1.5: the common industry bar of 1.0 lets all six cells through and screens nothing; 2.0 lets none through in every segment; 1.5 is the lowest half-step that demands "clearly above 1 in every segment" and was not tuned on the out-of-sample result. → Three cells: price-volume + chip × L/S + futures hedge, price-volume × L/S + futures hedge, price-volume + chip × multi-day long-short. The two weekly longs (first segment 1.44) and price-volume × multi-day long-short (1.32) drop out.

#### 7.3 Out-of-sample (2026-06-01 to 2026-10-02, 85 sessions): all six cells

Pre-registered confirmation conditions: (i) out-of-sample annualized net return > 0; (ii) out-of-sample maximum drawdown not deeper than in-sample. A second reading is added for context: (ii′) drawdown within twice the in-sample value — the in-sample short book is the average of five random replicates, which damps drawdowns, so "not deeper" is on the strict side.

| Model | Strategy | Shortlisted | Out-of-sample net return [95% interval] | Sharpe | Out-of-sample max drawdown | In-sample max drawdown | (i) | (ii) not deeper | (ii′) within 2× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Price-volume + chip | Weekly long (original pick) | no | +3.5% [−110.8%, +110.6%] | 0.08 | −25.7% | −22.6% | ✓ | ✗ | ✓ |
| Price-volume | Weekly long | no | +27.5% [−38.3%, +101.1%] | 1.05 | −14.8% | −18.7% | ✓ | ✓ | ✓ |
| Price-volume + chip | L/S + futures hedge | **yes** | +34.1% [−19.0%, +90.1%] | 2.19 | −8.1% | −6.1% | ✓ | ✗ | ✓ |
| Price-volume | L/S + futures hedge | **yes** | +27.5% [−16.9%, +67.1%] | 1.74 | −6.3% | −8.5% | ✓ | ✓ | ✓ |
| Price-volume + chip | Multi-day long-short | **yes** | +30.9% [−25.9%, +90.0%] | 2.01 | −8.6% | −8.4% | ✓ | ✗ | ✓ |
| Price-volume | Multi-day long-short | no | +23.3% [−23.4%, +64.5%] | 1.50 | −6.9% | −10.2% | ✓ | ✓ | ✓ |

By month (institution, both markets): in July the equal-weight liquidity universe lost −17.0% and 0050 −6.5%; the originally selected weekly long lost −19.5%, the price-volume + chip L/S + futures hedge made +15.6%.

#### 7.4 How to read it

- **Under the pre-registered rule the confirmation is not passed and no strategy enters paper trading**: the selected weekly long is 100% long with a beta near one, so its drawdown is the market's, and one −17% correction in 85 days breaks "no deeper drawdown". The outcome is recorded as it is; the cell cannot be swapped after the fact.
- **The threshold rule keeps a different kind of strategy**: the three market-neutral long-short versions hold Sharpe 1.5 to 2.3 in both segments and return +27% to +34% out of sample with −6% to −9% drawdowns — earned in the very July when the long strategy lost. Their out-of-sample intervals are wide (±50 to ±60 points) and 85 days cannot rank them, but "drawdown under control" already separates them from the weekly long.
- **Return versus stability**: the weekly long makes +52% a year in-sample with a −23% drawdown; the L/S + futures hedge +25% with −6%. The first wins big in a bull market and gives a quarter back in a correction; the second earns slowly in any market and loses little. Which is "better" depends on the drawdown one can carry, not on a single Sharpe figure — which is why the selection moved to a threshold shortlist.
- **The role of chip data**: an increment at the ranking level (+0.017); at the strategy level only on the long book (+16.3% a year), almost none on the long-short books (+0.8% to +1.0%, intervals spanning zero) — the short book gains nothing from it. Among the three shortlisted cells the chip model's edge over the price-volume model is therefore small.
- **Next**: the three shortlisted strategies enter the twelve-month confirmation period from 2026-10-23 together (section 10), each judged on its own.

### 8. How the samples were used: leakage and design bias, stated plainly

"Contamination" means two different things; they are accounted for separately.

**(1) Data leakage (future information reaching features, labels, training or strategy inputs): none.** Every feature uses data before the signal date; chip rows are used only once published before the next open; the label's size proxy is trailing 20-day dollar volume; each quarter's model is trained on data ending 15 sessions before the quarter, with early stopping on the last 12 months inside the training window; the strategies' beta, momentum and liquidity are trailing windows; the pipeline stops when the trading calendar and the data disagree. The out-of-sample data (June to October 2026) was downloaded on 28 October, after every model, feature, strategy and decision rule had been frozen; the "all in-sample" statistics explicitly exclude it.

**(2) Design-level selection bias (the researcher deciding the next step after seeing results): present, listed below.** This is not leakage; it is a property of every "design on history, then test out of sample" process. The industry handles it by discounting in-sample numbers, trusting only the out-of-sample test, and adding a paper-trading period.

| Decision | Date | Results already seen when it was made | Figures affected |
| --- | --- | --- | --- |
| 183 price-volume features, LightGBM hyperparameters | 22 Oct | none (fixed in advance) | — |
| Residualized labels (net of market, size, industry) | 23 Oct | the raw labels' failure in the second in-sample segment (after June 2025) | both models' in-sample ranking skill |
| Orthogonalizing the score to beta and momentum | 25 Oct | the long-short strategies' second-segment losses and the exposure diagnostic | in-sample multi-day long-short and L/S + hedge |
| Short execution layer (lists, quotas, failure draws) | 26 Oct | the whole in-sample | same (it makes the numbers more conservative) |
| 67 chip features | 26 Oct | the price-volume model's in-sample results; feature content from the literature, not tuned to the second segment | the price-volume + chip model's in-sample |
| Design of the weekly long (weekly, 5-day score, long only, turnover cap) | registered 25 Oct, built 28 Oct | the second segment's bull market and the long-short results | the weekly long's in-sample, especially the second segment's +65% to +111% |
| Each strategy's K and m (and the long-short h) | at scoring | the first in-sample segment only (2023-01 to 2025-05) | the second segment is unseen for the parameters, not for the design |
| Selection rule and confirmation conditions | 28 Oct | neither the out-of-sample window nor the retrained in-sample cells | — |

Constructions tried: six generations of strategy (daily rotation → hold buffer → residual labels → orthogonalization → execution layer → weekly long), two models, three markets, six final candidate cells. The selected cell's in-sample Sharpe of 2.38 is the maximum after those attempts, so **in-sample figures must be discounted**; the 85 out-of-sample days are the only one-shot test, and it did not pass.

**(3) One exception to the out-of-sample window's independence, stated as it is.** The data from 2026-06-01 to 09-24 had been used once, on 20 October, for a forward test of earlier foundation-model signals (result: positive ranking skill, every interval spanning zero); this report's models and strategies came afterwards. That test informed no design decision, but it cannot be proven that the researcher's impression of that period had no influence. The precise statement is: the out-of-sample window is unseen for this report's models and strategies, not untouched by any analysis. The fixed test set reused across the first four stages (2025-12 to 2026-05) had long lost its out-of-sample status and is treated here as history only.

**(4) One small ex-post element.** The residualization applies an October 2026 snapshot of company industry classes to the whole history; reclassifications and delisted companies (assigned to an "unknown" class) carry a slight ex-post and survivorship element, affecting the industry dummies only.

### 9. Limitations (stated in advance, not found afterwards)

1. The out-of-sample window has only 85 sessions in one market regime; neither passing nor failing extrapolates to a long-run conclusion.
2. The second in-sample segment (after 2025-06) had already been seen when the orthogonalization was designed, so in-sample figures are informational and the only real test is the 85 out-of-sample days.
3. Adjusted prices carry dividends only, not splits, capital reductions or par-value changes; since 2023 there are 100 single-day adjusted-open jumps above 25% across 91 stocks (19 events, 19 stocks inside the out-of-sample window). 98 follow a trading halt: the sample rule requires 128 gap-free sessions before the signal date and 14 after it, so every sample around a halt is dropped and no feature, label, score or position return touches the jump; the other 2 are dividend factors applied on the wrong date, and neither stock was in the model's samples around its event. Checked one by one, none of these events enters training or any strategy's profit and loss. The correction (splits, capital reductions, par-value changes) is written but not retrained; under the current rules it is expected to leave the results unchanged. A second, larger adjusted-price problem was found on 2026-10-29: on the ex-day of a paid rights issue the exchange's opening basis stays at the prior close, but the pipeline adjusted with the theoretical reference price, inventing a rise for every rights issue (861 since 2016, 386 since 2023, median error about 1.9%, largest 32%); these events have no halt, so the sample rule does not catch them, and they cluster in the June-to-September ex-rights season, which is the out-of-sample window (40 events). Re-running the same model predictions on corrected prices lowers the out-of-sample annual net return of the three shortlisted strategies from +34.1%, +27.5% and +30.9% to +14.7%, +19.4% and +13.7%, with in-sample changes within ±4 points; still positive and within the drawdown limits, but the out-of-sample figures in this report should be read as overstated. The pipeline is fixed; model scores before the retrain still carry the error.
4. The short book's borrow-failure, recall and fee tiers are parameter assumptions, not actual fills.
5. Chip data is limited to free official items; broker-branch and shareholder-dispersion data were not bought.
6. The out-of-sample window contains the April 2025 crash and the July 2026 correction but not the 2022 bear market.

### 10. Next steps

1. The three shortlisted market-neutral strategies enter a twelve-month confirmation period from 2026-10-23 together (rules fixed and frozen on 2026-10-29; an interim reading at six months, not judged): each quarter the data is extended to the quarter end, the new quarter's fold is trained, and the daily net returns accumulate; at the end each cell is judged by the two pre-registered conditions — net return > 0 and maximum drawdown not deeper than twice its own in-sample drawdown. Those that pass enter paper trading.
2. Beta-hedging the weekly long was tested (appendix B): the allowlisted instruments cannot rescue it, and it is no longer a candidate.
3. Retrain on split- and capital-reduction-adjusted data and recompute every number (separately).

### Appendix B: a follow-up backtest — beta-hedging the weekly long (29 October 2026)

After the out-of-sample confirmation failed, one more round was run that touches only the strategy layer, not the model or the data: a hedge leg reset daily at −β × capital on top of the weekly low-turnover long, β estimated by a trailing 60-day rolling regression of the strategy on the hedge instrument. Two instruments: short TAIEX futures, and a short position in 0050 (the only ETF on the dataset's allowlist that is both liquid and lendable). Because this design was decided after the out-of-sample result was known, the whole round is informational and cannot count as out-of-sample evidence.

| Version (price-volume + chip model, institution costs, both markets) | In-sample net return (annual) | 95% interval | Net Sharpe | In-sample max drawdown | Out-of-sample net return (annual) | Out-of-sample max drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| Weekly long (original) | +51.8% | [+26.1%, +77.7%] | 2.38 | −22.6% | +3.5% | −25.7% |
| + TAIEX futures hedge | +25.4% | [+7.5%, +45.6%] | 1.21 | −11.4% | −15.7% | −18.8% |
| + short 0050 | +16.6% | [−1.2%, +33.6%] | 1.05 | −17.7% | −31.6% | −18.5% |
| Reference: long-short + futures hedge | +25.4% | [+13.9%, +38.0%] | 1.98 | −6.1% | +34.1% | −8.1% |

Result: the hedge halves the Sharpe, removes only part of the drawdown, and makes the out-of-sample window worse (July's −19.5% becomes about −13%). The reason is that it hedges the wrong risk: the long book is near equal-weight and tilted to small and mid caps, so its beta on the TSMC-dominated cap-weighted index is only 0.5 to 0.8; and July 2026 was a breadth sell-off (equal-weight universe −17%) rather than a mega-cap one (0050 −6.5%). Hedging an equal-weight book with an index instrument pays heavily while the mega caps rally alone and fails to protect when breadth falls. This is the same fact as two earlier findings in the research history (the futures-hedged version failing, 0050 beating every strategy): over this period the risk of a long book in Taiwan sits in the equal-weight-versus-cap-weight spread, not in index beta. Conclusion: the allowlisted instruments cannot rescue a long-only strategy; for drawdown control the market-neutral long-short line, whose short book is itself an equal-weight hedge, is the right main line.

### Appendix A: names used in this report and the code-base identifiers

| Name in this report | Identifier | Meaning |
| --- | --- | --- |
| Price-volume model | F0 / `alpha_tw_v1` | LightGBM ranker on 183 price-volume features |
| Price-volume + chip model | F2 / `alpha_chip_tw_v1` | 183 price-volume + 67 chip features |
| Residualized labels | `residual_v1a` | h-day return net of market, size and industry |
| Multi-day long-short | S1 / `md_ls` | band long-short on the 10-day score |
| L/S + futures hedge | S2 / `md_ls_tx` | S1 plus a rolling-beta TAIEX futures overlay |
| Weekly low-turnover long | S-A / `wk_long` | weekly band long on the 5-day score |
| In-sample first / second segment | P1 / P2 | 2023-01 to 2025-05 / 2025-06 to 2026-05 |
| Out-of-sample | P3 | 2026-06-01 to 2026-10-02 |
| Confirmation conditions | H-J | net return > 0 and drawdown not deeper than in-sample |
| Protocol section | §13.20 | `docs/experiment_protocol.md` |

Code and full tables: `backtest/` (backtests), `src/stock_forecasting/` (data and models), `docs/research_full_report.md` (full report).
