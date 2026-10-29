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
| 選定結果 | 價量＋籌碼模型 × 週頻低換手多頭：樣本內年化淨報酬 +51.8%（95% 區間 +26.1% 到 +77.7%）、淨 Sharpe 2.38、最大回撤 −22.6% |
| 樣本外 | 年化淨報酬 +3.5%、最大回撤 −25.7%。事前登記的確認條件有兩條：淨報酬為正（成立）、回撤不比樣本內深（**不成立**） |
| 結論 | **確認未通過，策略維持研究狀態，沒有進入模擬交易。** 原因是 2026 年 7 月一次 −17% 的市場修正直接打在 100% 多頭的策略上；同一窗口市場中性的多空策略 +23% 到 +34%、回撤 −6% 到 −9%，但依事前規則不能事後改選 |

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

#### 7.1 樣本內（2023-01 至 2026-05，機構成本，上市與上櫃各半）

| 模型 | 策略 | 淨 Sharpe | 年化淨報酬 | 95% 區間 | 最大回撤 |
| --- | --- | --- | --- | --- | --- |
| **價量＋籌碼** | **週頻多頭** | **2.38** | +51.8% | [+26.1%, +77.7%] | −22.6% |
| 價量 | 週頻多頭 | 2.07 | +35.5% | [+17.2%, +54.7%] | −18.7% |
| 價量＋籌碼 | 多空＋期貨避險 | 1.98 | +25.4% | [+13.9%, +38.0%] | −6.1% |
| 價量 | 多空＋期貨避險 | 1.78 | +24.3% | [+10.4%, +38.0%] | −8.5% |
| 價量＋籌碼 | 多日多空 | 1.57 | +19.1% | [+7.5%, +31.8%] | −8.4% |
| 價量 | 多日多空 | 1.44 | +18.3% | [+3.6%, +32.2%] | −10.2% |

對照：同期等權持有流動性宇宙年化 +31.5%（回撤 −18.7%）、0050 +48.3%（回撤 −25.2%）、加權報酬指數 +38.3%。多空策略的 +19% 到 +25% 是市場中性的報酬，不該直接與指數相比；週頻多頭是 100% 多頭，贏等權宇宙、輸 0050。

**事前登記的選定規則**（機構成本、兩市場各半、樣本內淨 Sharpe 最高）選出**價量＋籌碼模型 × 週頻低換手多頭**。它的每日換手約 6%、年成本拖累約 7%；散戶成本下年化 +47.0%、Sharpe 2.17。逐年：2023 +38.2%、2024 +35.1%、2025 +27.9%、2026 至 10 月初 +93.1%。

#### 7.2 樣本外確認（2026-06-01 至 2026-10-02，85 個交易日）

事前登記的確認條件：(i) 樣本外年化淨報酬 > 0；(ii) 樣本外最大回撤不比樣本內深。兩者都成立才進入模擬交易。

| 項目 | 選定策略（價量＋籌碼 × 週頻多頭） |
| --- | --- |
| 年化淨報酬 | +3.5%（95% 區間 −110.8% 到 +110.6%；85 天的區間必然很寬） |
| 最大回撤 | **−25.7%**（高點 7 月 3 日、低點 7 月 29 日），樣本內為 −22.6% |
| 各月 | 6 月 +9.7%、7 月 −19.5%、8 月 +8.6%、9 月 +2.4% |
| 同期市場 | 7 月等權流動性宇宙 −17.0%、0050 −6.5%；整段等權宇宙 +12.1%、0050 +25.8% |
| 判定 | (i) 成立、(ii) 不成立 → **未通過** |

**為什麼會這樣**：這個策略 100% 多頭、beta 接近 1，回撤由市場決定；85 天的確認期剛好包含一次 −17% 的修正，用「回撤不更深」當條件幾乎必然失敗。同一窗口裡，市場中性的多空加期貨避險策略（價量＋籌碼）+34.1%、回撤 −8.1%，上市市場甚至 +72.6%（區間 +27.0% 到 +120.7%）。但選定規則是事前寫死的，**不能因為結果不好就換一個格子**；這正是事前登記存在的意義。

#### 7.3 籌碼資料到底有沒有用

- 排序層：有。加入籌碼後 5 到 10 日排序能力的增量 +0.017，區間不跨 0。
- 策略層：只對多頭有用。週頻多頭加籌碼後樣本內年化多 +16.3%（區間 +5.5% 到 +27.7%）；多空策略只多 +0.8% 到 +1.0%（區間跨 0），因為放空腿沒有從籌碼得到好處。
- 樣本外：加籌碼的多頭在 7 月跌得更深（−19.5% 對 −8.5%），這是籌碼特徵偏好近期強勢股的代價。

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
3. 還原價只處理股息，不含分割、減資與面額變更；2023 年起有 109 檔出現過超過 25% 的單日跳動（樣本外期間 48 次、42 檔）。持股報酬用相鄰交易日的開盤價計算、停牌即出場，不受影響，但模型分數在事件前後會錯。修正版已寫好、尚未重訓。
4. 放空腿的借不到機率、召回機率與借券費分層是參數假設，不是實際成交紀錄。
5. 籌碼只用免費官方資料；分點（券商）與集保資料沒有購買。
6. 樣本外視窗含 2025 年 4 月急跌與 2026 年 7 月修正，不含 2022 年空頭。

### 10. 下一步

1. 週頻多頭加 beta 對沖或波動目標，讓回撤不由市場決定——這是新的策略，參數要重新在樣本內前段選定，再用新的樣本外期間確認。
2. 或者以市場中性的多空加避險為主線，重新登記一段夠長的確認期。
3. 用修正分割與減資後的資料重訓，重算全部數字。

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
| Selected | Price-volume + chip model × weekly low-turnover long: in-sample annualized net return +51.8% (95% interval +26.1% to +77.7%), net Sharpe 2.38, maximum drawdown −22.6% |
| Out-of-sample | Annualized net return +3.5%, maximum drawdown −25.7%. The two pre-registered confirmation conditions: positive net return (holds), drawdown not deeper than in-sample (**fails**) |
| Conclusion | **Confirmation not passed; the strategy stays in research and does not enter paper trading.** A −17% market correction in July 2026 landed on a 100% long book; the market-neutral strategies made +23% to +34% with −6% to −9% drawdowns in the same window, but the pre-registered rule does not allow a change of pick after the fact |

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

#### 7.1 In-sample (2023-01 to 2026-05, institution costs, TWSE and TPEX half each)

| Model | Strategy | Net Sharpe | Net return (annual) | 95% interval | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| **Price-volume + chip** | **Weekly long** | **2.38** | +51.8% | [+26.1%, +77.7%] | −22.6% |
| Price-volume | Weekly long | 2.07 | +35.5% | [+17.2%, +54.7%] | −18.7% |
| Price-volume + chip | L/S + futures hedge | 1.98 | +25.4% | [+13.9%, +38.0%] | −6.1% |
| Price-volume | L/S + futures hedge | 1.78 | +24.3% | [+10.4%, +38.0%] | −8.5% |
| Price-volume + chip | Multi-day long-short | 1.57 | +19.1% | [+7.5%, +31.8%] | −8.4% |
| Price-volume | Multi-day long-short | 1.44 | +18.3% | [+3.6%, +32.2%] | −10.2% |

Benchmarks over the same window: equal-weight holding of the liquidity universe +31.5% a year (drawdown −18.7%), 0050 +48.3% (drawdown −25.2%), TAIEX total-return index +38.3%. The long-short strategies' +19% to +25% is market-neutral return and should not be compared with an index; the weekly long is 100% long, beats the equal-weight universe and trails 0050.

The **pre-registered selection rule** (institution costs, both markets half each, highest in-sample net Sharpe) selects **the price-volume + chip model × weekly low-turnover long**. Its daily turnover is about 6% and its cost drag about 7% a year; under retail costs it returns +47.0% a year with a Sharpe of 2.17. By year: 2023 +38.2%, 2024 +35.1%, 2025 +27.9%, 2026 to early October +93.1%.

#### 7.2 Out-of-sample confirmation (2026-06-01 to 2026-10-02, 85 sessions)

Pre-registered confirmation conditions: (i) out-of-sample annualized net return > 0; (ii) out-of-sample maximum drawdown not deeper than in-sample. Both must hold for paper trading.

| Item | Selected strategy (price-volume + chip × weekly long) |
| --- | --- |
| Net return (annual) | +3.5% (95% interval −110.8% to +110.6%; an 85-day interval is necessarily wide) |
| Maximum drawdown | **−25.7%** (peak 3 July, trough 29 July) against −22.6% in-sample |
| By month | June +9.7%, July −19.5%, August +8.6%, September +2.4% |
| Market over the same window | equal-weight universe −17.0% and 0050 −6.5% in July; whole window: equal-weight universe +12.1%, 0050 +25.8% |
| Verdict | (i) holds, (ii) fails → **not passed** |

**Why it turned out this way**: the strategy is 100% long with a beta near one, so its drawdown is the market's; the 85-day window happened to contain a −17% correction, and a "no deeper drawdown" condition almost cannot survive that. In the same window the market-neutral long-short with futures hedge (price-volume + chip) made +34.1% with an −8.1% drawdown, and +72.6% on TWSE alone (interval +27.0% to +120.7%). But the selection rule was fixed in advance, and **a bad outcome is not a reason to pick a different cell** — that is exactly what pre-registration is for.

#### 7.3 Is the chip data useful?

- At the ranking level: yes. Adding chip data raises the 5- to 10-day ranking skill by +0.017 with an interval clear of zero.
- At the strategy level: only for the long book. The weekly long earns +16.3% a year more with chip data in-sample (interval +5.5% to +27.7%); the long-short books gain only +0.8% to +1.0% (intervals spanning zero), because the short book gets nothing from the chip features.
- Out of sample: the chip-driven long book fell harder in July (−19.5% vs −8.5%), the price of the chip features' preference for recent winners.

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
3. Adjusted prices carry dividends only, not splits, capital reductions or par-value changes; 109 stocks since 2023 show a single-day jump above 25% (48 events across 42 stocks inside the out-of-sample window). Position returns use adjacent sessions' opens and exit on halts, so they are unaffected, but model scores around the events are wrong. The correction is written but not yet retrained.
4. The short book's borrow-failure, recall and fee tiers are parameter assumptions, not actual fills.
5. Chip data is limited to free official items; broker-branch and shareholder-dispersion data were not bought.
6. The out-of-sample window contains the April 2025 crash and the July 2026 correction but not the 2022 bear market.

### 10. Next steps

1. The weekly long with a beta hedge or a volatility target, so the drawdown is not the market's — a new strategy whose parameters must be re-selected on the first in-sample segment and confirmed on a new out-of-sample window.
2. Or make the market-neutral long-short with futures hedge the main line and register a confirmation period long enough to judge.
3. Retrain on split- and capital-reduction-adjusted data and recompute every number.

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
