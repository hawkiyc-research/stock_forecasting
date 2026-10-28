# 台股排序模型與交易策略研究完整報告 / Taiwan Ranking Model and Trading Strategy: Full Research Report

> 狀態（2026-10-28）：實驗史、策略設計、資料限制與 roadmap 已完成；第 6 節的 P1／P2／P3 回測數字待 §13.20 的 P3 預測下載並計分後填入（標示為【待填】）。
> Status (2026-10-28): experiment history, strategy design, data limitations and roadmap are complete; the P1 / P2 / P3 backtest figures of section 6 are filled in once the §13.20 P3 predictions are downloaded and scored (marked 【TBD】).

[繁體中文](#繁體中文) · [English](#english)

---

## 繁體中文

### 1. 摘要

- **問題**：在嚴格的時間點輸入、次日開盤成交與時間順序評估下，日線價量加免費籌碼資料的橫斷面排序模型，能否在台灣的交易成本與執行限制下，變成一條可進入試運行的策略。
- **模型**：B0 LightGBM 走勢排序模型（`t-lgbm-alpha-residual_v1a`），特徵 F0 = 183 欄價量（`alpha_tw_v1`）與 F2 = F0 + 67 欄免費籌碼（`alpha_chip_tw_v1`），殘差化標籤 `residual_v1a`，三 seed 集成，季度擴張 walk-forward。
- **策略**：S1 多日多空 band（分數對 beta 與動能正交化，h = 10、K = 50、m = 3，空頭腿含可借券名單、借不到機率、召回與分層借券費）、S2 = S1 + 台指期滾動 β 疊加、S-A 週頻低換手多頭 band（h = 5 分數，K × m 於 P1 選定，單週換手上限 50%）。
- **判定**：P1 ∪ P2（2023-01 至 2026-05）為選定與資訊性門檻；**P3（2026-06-01 至 2026-10-22）為唯一的樣本外確認**，H-J = 選定格在 P3 淨報酬 > 0 且最大回撤不比 P1 ∪ P2 深。
- **結果**：【待填：選定格、P3 淨報酬與區間、H-J 結論、策略狀態】。
- **籌碼增量**：排序層已確認（§13.18 H-H 與 G1 通過：F2 − F0 持有格殘差 Rank IC +0.0173 [+0.0128, +0.0214]）；策略層的增量【待填：F2 − F0 配對淨報酬差】。

### 2. 問題定義與目標

1. **產品目標**：模型在收盤後給出個股未來 1 到 14 個交易日的相對強弱排序，供交易規則在次一交易日開盤進場；策略層把排序變成扣除成本後的報酬。
2. **成功標準（事前登記）**：機構主要成本、分數位置、P1 ∪ P2 淨 Sharpe ≥ 1.0、淨報酬 95% 區間下界 > 0、P2 淨報酬 > 0（§13.19 第 9 項）；P3 確認 H-J（§13.20 第 6 項）。散戶成本一律並列。
3. **不做的事**：不買付費資料（2026-10-28 使用者決定）；不做 LLM-in-the-loop 的因子挖掘；當沖路線（S3–S6）在 §13.19 排除；「勝過 0050 買進持有」自 2026-10-26 起不再是門檻（2023 年以來的台股是權值股單邊行情，被動持有與量化交易是不同賽道）。

### 3. 資料與評估框架

| 項目 | 設定 |
| --- | --- |
| 資料集 | TWSE、TPEX 日線 OHLCV（EODHD 加官方來源），2016-01-01 起；§13.20 的延長資料集結束日 2026-10-23（修訂 v1：`adjusted_close` 只還原股息，見第 7 節） |
| 籌碼資料 | `tw_chip_data` curated 層（法人買賣超、融資融券、借券、當沖統計、注意／處置、外資持股、得為融資融券與得為借券名單），公布時刻規則 T+1 開盤前；籌碼層到 2026-10-22 |
| 標籤 | `residual_v1a`：h 日持有報酬扣掉市場、市值、產業效應後的殘差（§13.16 變體 A） |
| Walk-forward | `tw_quarterly_v2`：2016 起擴張訓練、15 個交易日 embargo、12 個月驗證、季度折到 2026Q4，共 16 折 |
| 期間 | P1 = [2023-01-02, 2025-06-01)、P2 = [2025-06-01, 2026-06-01)、**P3 = [2026-06-01, 2026-10-23)** |
| 宇宙 | 只含股票；訊號日前 20 日成交金額中位數 TWSE ≥ 3,000 萬、TPEX ≥ 1,000 萬，收盤價 ≥ 10 元 |
| 成本檔 | 散戶：手續費 0.0855%、證交稅 0.3%、滑價 0.1%／0.2%；機構：手續費 0.03%、滑價 0.05%、借券年費 2%（分層 1.5%／3%／6%）；滑價 0／1／2 倍敏感度 |
| 資本情境 | 分數位置、機構 3 億與 10 億（每檔名目封頂於 5% × 20 日成交金額中位數） |
| 執行層（空頭腿） | 可借券名單（機構 `sbl`、散戶 `margin`）、交易所次日限額、依規模分層的借不到機率（2%／10%／25%）、召回 0 與 0.5%、五個隨機重複平均 |
| 統計 | circular block bootstrap（區塊 21、1,000 次、seed 11）；Sharpe 以日淨報酬年化 |

### 4. 實驗史（2026-10-16 至 2026-10-28）

#### 4.1 起點：Kronos 基礎模型微調（§13.1–§13.10）

- Kronos-base + LoRA 的 alpha 點預測與 ±1% 方向分類在美股與台股的 holdout 都沒有穩健優勢。誤差拆解（`docs/error_decomposition_audit.md`）顯示：alpha 的日期層級成分只占 2.6–3.8%，分類器輸在不可學的擇時；移除擇時後與 GBDT 相當。
- 第一階段回測（§13.4、§13.5）：validation 選出的 10 日設定在 test 年化淨超額 −37% 到 −44%；前 K 名的優勢只在交易時段（開盤到收盤 +18 到 +31 bps），收盤到次日開盤反轉（−5 到 −28 bps）。
- 第三階段前瞻（§13.8，2026-06-01 至 09-24）：主要假說未通過（前十分位超額 +57 bps [−10.5, +124.3]）；6 個訊號的 TWSE 10 日 Rank IC 皆為正（0.074–0.149）但沒有一個區間排除 0。
- 2026-10-21 轉向橫斷面排序任務：排序基線（PatchTST、GRU、GBDT、DLinear、均線交叉）validation → test 的 Rank IC 0.057 → 0.035 到 0.020 → 0.043；Kronos `a-rank64`（§13.11）H1 通過（台股 1 日 0.064 [0.027, 0.102]）但 H2 未通過（Kronos − PatchTST −0.002 [−0.021, +0.018]）。結論：Kronos 不是骨幹，排序能力在特徵與橫斷面建模。

#### 4.2 台股主線：特徵 LightGBM 與跨股票 DL（§13.12–§13.13）

- 路線決定（2026-10-22）：MASTER 式跨股票 transformer（D0）對同特徵 LightGBM（B0）在同一 walk-forward 下配對比較；策略層先用 Top-k dropout。
- **B0**（14 折 × 14 horizon × 3 seed = 588 個模型）P1 ∪ P2 Rank IC：短格（h = 1）0.118 [0.109, 0.127]、持有格（h = 5–10）0.086、長格 0.083；P2 持有格掉到 0.034 [−0.000, 0.073]。
- **D0** 短格 0.122、持有格 0.092；D0 − B0 配對差短格 +0.0043 [+0.0015, +0.0071]（H-A 一格通過）；消融 A1 顯示增量來自三 seed 集成，不是股票間 attention。
- Top-k dropout（每日換 5 檔於 50 檔）：TWSE P1 ∪ P2 淨超額 −6.3% [−19.0%, +7.1%]，**H-B 未通過**：每日 10% 換手的成本拖累約 17%／年，高於毛超額。

#### 4.3 策略層第一輪：band 組合與台指期（§13.14、§13.15）

- band 規則（買進前 K、掉出前 m × K 才賣、每日新進上限 K／10）把換手降到每日 2–2.5%、拖累 3.4–4.8%／年；P1 淨超額 +5.4%（區間跨 0），**P2 毛超額 −27% 到 −31%**，P1 ∪ P2 淨超額 −4.7% [−19.7%, +10.5%]，**H-D 未通過**；台指期對沖版 −12.5%／年、27–28 次追繳。絕對報酬 +23–24%／年、Sharpe 1.5–1.7，但同期 TAIEX 報酬指數 +41%／年。
- 股票期貨空方當沖（§13.15）：h = 1 分數的毛價差幾乎全在空方（後十分位盤中 −0.36% 到 −0.45%／日）；以期交所官方開盤與結算價計算毛 +61.5%／年，但一 tick 滑價每邊約 0.23%，淨 −63%／年，**H-E 未通過**——執行受限（價差），不是模型。

#### 4.4 殘差化標籤（§13.16）

- P2 的失效來自大型股集中行情的市值與產業因子報酬：對殘差目標（扣掉市場、市值、產業）的持有格 Rank IC 在 P2 仍為 0.065 [0.046, 0.086]，對原始 alpha 只剩 P1 的四到五成。
- `residual_v1a` 標籤重訓的 B0：短格對原始 alpha 不變（配對差 −0.0002）、P2 持有格 +0.0090 [+0.0055, +0.0125]，**H-F 通過**，成為之後所有模型的標籤。D0 與 D1（凍結 Kronos-small、Chronos-2-small）在殘差標籤上重訓：D0 對 B0 的增量仍只在短格（+0.004），Kronos-small 與 Chronos-2 都不優於 D0 單 seed；DL 路線不再是主線。

#### 4.5 暴露診斷與正交化（§13.17 步驟 0、§13.19 補充二至四）

- 診斷：平滑後的 h = 10 分數與 60 日 beta 相關 −0.24 到 −0.45、與前 20／60 日報酬相關 −0.1 到 −0.26（低 beta、反動能傾斜）；P2 的虧損是結構性暴露，不是排序能力消失。
- **六情境（§13.19 原建構）**：機構成本下 0／36 格達標；S1 空頭腿（P2 的大型權值股）每年虧 61–89%。
- **正交化（補充二，V1）**：同一 band 參數下 S1 的 P1 ∪ P2 機構淨報酬 B0 TWSE −6.3% → **+17.3% [−0.6%, +37.3%]**（Sharpe 1.04）、TPEX +19.4% [+4.9%, +34.4%]（1.50）、TW +18.3% [+5.5%, +33.6%]（1.46），P2 全部轉正，組合 β −0.15 到 −0.20；S2 再加 +2% 到 +9%；10／36 格達標。但 V1 是看過 P2 之後的設計，P1 ∪ P2 只在資訊層面。
- **執行層（補充四）**：加入可借券名單、限額、借不到機率與召回後，B0 TW S1 仍 +15.3% [+2.5%, +30.3%]，9／12 格達標；資本情境改為封頂而非跳過。
- **0050 核心疊加（§13.17 補充）**：所有 24 格都輸 0050 買進持有（P1 ∪ P2 年化 +46.2%）；2026-10-26 使用者決定該門檻不再適用。

#### 4.6 免費籌碼資料（§13.18）

- 67 欄籌碼特徵（法人五類累計比與 z 分數、融資融券、借券、當沖比、注意／處置狀態與距離），公布時刻保守規則，流通股以融券限額 × 4 估計。
- **H-H 通過**：F2 − F0 持有格殘差 Rank IC +0.0173 [+0.0128, +0.0214]（P1 +0.0163、P2 +0.0198）；流動性宇宙 h = 10 四分位殘差價差 0.65% → 1.04%（1.59 倍，門檻 1.5）。**G1 通過**（流動性宇宙 +0.0202、中小型股 +0.0191）。F1（只有籌碼）在 P2 的持有格殘差 IC 0.090 超過 183 欄價量的 0.065。
- 子分析：增量在近期強勢股組更大（+0.0293 對 +0.0179），但兩期、兩組都顯著——籌碼是一個與期間無關的獨立訊號，不是 P2 才出現的條件化反轉。
- 使用者決定（2026-10-28）：G1 通過但先不購買分點資料；以手上的 F0／F2 進入 §13.20。

### 5. 最終策略設計（§13.20，全部參數事前固定）

| 策略 | 建構 | 參數 | 來源 |
| --- | --- | --- | --- |
| **S1 多日多空** | 每日 V1 正交化分數（對 60 日 beta、前 20／60 日報酬），多頭 band 加反向排序的空頭 band，空頭腿走執行層 | h = 10、K = 50、m = 3、每日新進上限 5 檔 | §13.19 第 7 項選定，補充二／四建構 |
| **S2 = S1 + 台指期** | 以組合毛報酬對指數的滾動 60 日 β 掛台指期（結算價對結算價），嚴格用前期資料 | β 視窗 60、最少 30 個觀測 | §13.19 第 5 項 |
| **S-A 週頻低換手多頭** | 每個 ISO 週最後一個訊號日再平衡；V1 正交化 h = 5 分數；前 K 多頭 band、掉出前 m × K 才賣；單週最多換掉 50% × K，超過只賣分數最差者；不放空、不加期貨 | K ∈ {30, 50} × m ∈ {2, 3} 於 TWSE、P1、機構成本、分數位置淨 Sharpe 選定後固定 | §13.18 第 7 項、§13.20 第 3 項 |

- **市場**：TWSE、TPEX、TW（兩市場各半）。**成本檔**：散戶與機構並列，滑價 0／1／2 倍，資本情境分數位置、3 億、10 億。
- **選定規則（事前）**：機構主要成本、TW、分數位置，P1 ∪ P2 淨 Sharpe 最高的（模型 ∈ {F0, F2}, 策略 ∈ {S1, S2, S-A}）格為「最後選定」。
- **H-J（P3 確認）**：選定格在 P3、機構主要成本下 (i) 年化淨報酬 > 0，且 (ii) P3 最大回撤不比 P1 ∪ P2 深。兩者成立 → 「可進入試運行」；任一不成立 → 維持研究狀態，不重選參數。
- **程式**：`backtest/twbacktest/scenario_suite.py`（`config/scenario_suite_v1_p3.json`）、`backtest/twbacktest/weekly_band.py`、`backtest/twbacktest/strategy_decision.py`（`config/strategy_decision.json`）。

### 6. 回測結果（P1、P2、P1 ∪ P2、P3）

> 全部數字來自 `scenario_suite --config config/scenario_suite_v1_p3.json`（F0、F2）與 `strategy_decision` 的輸出；【待填】於 P3 預測下載並計分後填入，不改任何參數。

#### 6.1 重訓模型對既有結果的一致性檢查（§13.20 第 9 項 (f)）

| 格 | §13.13／§13.18 登記值（v1 資料集，14 折） | 新資料集 16 折 | 是否在原 95% 區間內 |
| --- | --- | --- | --- |
| F0 持有格殘差 Rank IC，P1 ∪ P2 | 【待填：自 chip_increment_results 取值】 | 【待填】 | 【待填】 |
| F2 持有格殘差 Rank IC，P1 ∪ P2 | 【待填】 | 【待填】 | 【待填】 |
| F2 − F0 配對差，P1 ∪ P2 | +0.0173 [+0.0128, +0.0214] | 【待填】 | 【待填】 |

#### 6.2 S1／S2／S-A 主表（機構主要成本、分數位置、V1）

| 模型 | 策略 | 市場 | 期間 | 年化淨報酬 | 95% 區間 | Sharpe | 最大回撤 | 換手 | 成本拖累 | 被擋買單 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | S1 | TW | P1 | 【待填】 | | | | | | |
| F0 | S1 | TW | P2 | 【待填】 | | | | | | |
| F0 | S1 | TW | P1 ∪ P2 | 【待填】 | | | | | | |
| F0 | S1 | TW | P3 | 【待填】 | | | | | | |
| … | S2、S-A；F2；TWSE、TPEX | | | 【待填：由 grid.csv 產生】 | | | | | | |

#### 6.3 散戶成本檔

【待填：同 6.2 的表，成本檔 = 散戶。】

#### 6.4 S-A 對照欄

| 模型 | 市場 | 期間 | S-A 年化淨報酬 | 等權流動性宇宙 | 0050 | TAIEX 報酬指數 | S-A 最大回撤 | 0050 回撤 | 週換手 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 【待填】 | | | | | | | | | |

#### 6.5 選定與 H-J

- 候選表（P1 ∪ P2 機構淨 Sharpe，TW）：【待填：strategy_decision 的 candidates】。
- **最後選定**：【待填：模型 × 策略】。
- **H-J**：P3 年化淨報酬【待填】[【待填】]、P3 最大回撤【待填】對 P1 ∪ P2【待填】；(i)【待填】、(ii)【待填】→ 狀態【待填：paper_trading_candidate／research】。
- **執行層敏感度**（借不到倍率 0／1／2、召回 0／0.5%）：【待填】。
- **資本情境**（3 億、10 億）：【待填】。

#### 6.6 策略層的籌碼增量（資訊性）

| 策略 | 市場 | F2 − F0，P1 ∪ P2 年化差 | 95% 區間 | P3 差 | 增量 |
| --- | --- | --- | --- | --- | --- |
| 【待填】 | | | | | |

### 7. v1 資料集的缺陷與限制（事前聲明，§13.20 第 9 項）

(a) **`adjusted_close` 只還原股息，不含分割、減資與面額變更**：2023 年起 109 檔個股有超過 25% 的單日還原開盤跳動（0050 於 2025-06-18 一拆四）；這些檔的訓練標籤與價格類特徵在事件前後是錯的。band 引擎只在相鄰訊號日之間算同一檔的開盤到開盤報酬、停牌即出場，部位報酬不受影響，但模型分數受影響。P3 期間的事件數：【待填：由 `configs/reference/tw_capital_actions_20261026.csv` 統計】。
(b) **v2 修正已實作但未經 Pod 驗證、未訓練任何模型**（分支 `claude/dataset-v2-corporate-actions-20261026`），所以不用於 P3；若決定以 v2 重訓，P1 ∪ P2 與 P3 都要重算。
(c) **P2 已看過**：S1／S2 的 V1 設計在看過 P2 之後決定，P1 ∪ P2 的數字只是資訊；P3 才是樣本外。S-A 是新策略，參數在 P1 選定，P2 與 P3 都是樣本外。
(d) **P3 只有約 95 個交易日**，且 2026 年 6 到 10 月的市場狀態單一，通過或不通過都不能外推成長期結論。
(e) 籌碼特徵的公布時刻用保守規則（T+1 開盤前）；`FDLR_Z60` 恆缺值。
(f) 新資料集上重訓的 F0 與 F2，其 P1 ∪ P2 的格子若與 §13.13／§13.18 登記的數字差異超過各自的 95% 區間，先查資料再做 P3（第 6.1 節）。
(g) 樣本外視窗 2023-01 到 2026-10 含 2025-04 的急跌但不含 2022 年空頭（2022 只在訓練期）；分點與集保未購買，籌碼只含免費官方品項。

### 8. 改進 roadmap

| 順序 | 項目 | 條件 | 依據 |
| --- | --- | --- | --- |
| 1 | 依 H-J 結果決定選定策略是否進入試運行（paper trading），定義試運行的紀錄與停損規則 | 第 6.5 節 | §13.20 第 6 項 |
| 2 | G2 事件研究（處置宣告、注意首次公告、當沖比跳升）→ 決定 S-B 反轉策略是否實作 | 事件表已交付 | §13.18 第 6 項 |
| 3 | 資料集 v2（分割、減資、面額變更調整）重訓 F0／F2 並重算 P1 ∪ P2 與 P3 | 使用者看完本報告後決定；先把 provider 快取修訂與資料集修訂拆開 | §13.17 補充 (h)、§13.20 第 9 項 |
| 4 | D0 在 F2 特徵上重訓（三 seed） | 使用者決定；增量集中在 2023–2024 且逐年遞減 | §13.18 第 3 項 |
| 5 | 分點資料購買（G1 已通過） | 使用者決定；目前不買付費資料 | §13.18 第 10 項 |
| 6 | 籌碼資料在 Pod 上自動更新（`cpu-chip`，Cloud Run relay 的櫃買路由）首次實機驗證 | 下一個 chip Pod | README「籌碼資料在 Pod 上更新」 |

### 9. 產物與重現

- 協定：`docs/experiment_protocol.md` §13.11–§13.20；各節結果報告見 `docs/*_results.md`。
- 預測：`artifacts/lgbm/<F0 launch>/predictions.parquet`、`artifacts/lgbm/<F2 launch>/predictions.parquet`（`tw_quarterly_v2`，16 折）【待填 launch id】。
- 回測：`cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/<F0>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/<F2>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/<dataset>/raw/market.parquet`，之後 `uv run python -m twbacktest.strategy_decision --suite <輸出目錄>`。
- 輸出目錄：【待填】。

---

## English

### 1. Summary

- **Question**: under strict point-in-time inputs, next-open execution and chronological evaluation, can a cross-sectional ranking model on daily OHLCV plus free chip (ownership-flow) data become a strategy fit for paper trading under Taiwan's transaction costs and execution limits?
- **Models**: the B0 LightGBM walk-forward ranker (`t-lgbm-alpha-residual_v1a`) with features F0 = 183 price-volume columns (`alpha_tw_v1`) and F2 = F0 + 67 free chip columns (`alpha_chip_tw_v1`), residualized labels `residual_v1a`, three-seed ensembles, quarterly expanding walk-forward.
- **Strategies**: S1 multi-day long-short band (score orthogonalized to beta and momentum; h = 10, K = 50, m = 3; the short book under the borrowable list, borrow-failure, recall and tiered-fee layer), S2 = S1 + a rolling-β TAIEX futures overlay, S-A weekly low-turnover long band (h = 5 score, K × m selected on P1, 50% weekly turnover cap).
- **Decision**: P1 ∪ P2 (2023-01 to 2026-05) serves selection and the informational bar; **P3 (2026-06-01 to 2026-10-22) is the only out-of-sample confirmation**, H-J = the selected cell's P3 net return > 0 and its maximum drawdown not deeper than P1 ∪ P2's.
- **Result**: 【TBD: selected cell, P3 net return and interval, H-J verdict, strategy status】.
- **Chip increment**: confirmed at the ranking level (§13.18 H-H and G1 passed: F2 − F0 holding-cell residual Rank IC +0.0173 [+0.0128, +0.0214]); at the strategy level 【TBD: paired F2 − F0 net return difference】.

### 2. Problem and goals

1. **Product goal**: after the close the model ranks stocks by relative strength over the next 1–14 sessions for a trading rule that enters at the next open; the strategy layer turns the ranking into net returns.
2. **Success criteria (pre-registered)**: institution primary cost, fractional capital, P1 ∪ P2 net Sharpe ≥ 1.0, net-return 95% lower bound > 0, P2 net return > 0 (§13.19 item 9); P3 confirmation H-J (§13.20 item 6). Retail costs are always listed alongside.
3. **Out of scope**: no paid data (user decision 2026-10-28); no LLM-in-the-loop factor mining; the day-trade routes (S3–S6) were excluded in §13.19; "beating 0050 buy-and-hold" stopped being a bar on 2026-10-26 (Taiwan since 2023 is a one-sided mega-cap rally; passive holding and quantitative trading are different races).

### 3. Data and evaluation framework

| Item | Setting |
| --- | --- |
| Dataset | TWSE and TPEX daily OHLCV (EODHD plus official sources) from 2016-01-01; the §13.20 extended dataset ends 2026-10-23 (revision v1: `adjusted_close` carries dividends only, see section 7) |
| Chip data | `tw_chip_data` curated layer (institutional flows, margin, securities lending, day-trade statistics, attention / disposition, foreign holdings, margin and SBL eligibility lists), availability rule "before the T+1 open"; chip layer through 2026-10-22 |
| Labels | `residual_v1a`: the h-session holding return net of market, size and industry effects (§13.16 variant A) |
| Walk-forward | `tw_quarterly_v2`: expanding training from 2016, 15-session embargo, 12-month validation, quarterly folds through 2026Q4, 16 folds |
| Periods | P1 = [2023-01-02, 2025-06-01), P2 = [2025-06-01, 2026-06-01), **P3 = [2026-06-01, 2026-10-23)** |
| Universe | stocks only; 20-session median dollar volume TWSE ≥ NT$30M, TPEX ≥ NT$10M, close ≥ NT$10 |
| Cost profiles | retail: commission 0.0855%, tax 0.3%, slippage 0.1% / 0.2%; institution: commission 0.03%, slippage 0.05%, borrow fee 2% a year (tiers 1.5% / 3% / 6%); slippage × 0 / 1 / 2 sensitivities |
| Capital | fractional; institution NT$300M and NT$1B (each name capped at 5% × 20-session median dollar volume) |
| Execution layer (short book) | borrowable list (institution `sbl`, retail `margin`), next-session exchange quota, borrow-failure probability by size tercile (2% / 10% / 25%), recall 0 and 0.5%, five random replicates averaged |
| Statistics | circular block bootstrap (block 21, 1,000 replicates, seed 11); Sharpe from daily net returns, annualized |

### 4. Experiment history (2026-10-16 to 2026-10-28)

#### 4.1 Starting point: fine-tuning the Kronos foundation model (§13.1–§13.10)

- Kronos-base + LoRA alpha point prediction and ±1% direction classification showed no robust holdout edge in the US or Taiwan. The error decomposition (`docs/error_decomposition_audit.md`) found the date-level component of alpha at only 2.6–3.8% and the classifier losing to unlearnable market timing; with timing removed it matched GBDT.
- Stage 1 backtests (§13.4, §13.5): the validation-selected 10-day setting lost −37% to −44% a year net of costs on test; the top-K edge existed only intraday (open to close +18 to +31 bps) and reversed overnight (−5 to −28 bps).
- Stage 3 forward test (§13.8, 2026-06-01 to 09-24): the primary hypothesis failed (top-decile excess +57 bps [−10.5, +124.3]); all six signals had positive TWSE 10-day Rank IC (0.074–0.149) but no interval excluded zero.
- Pivot to cross-sectional ranking (2026-10-21): the ranking baselines (PatchTST, GRU, GBDT, DLinear, MA crossover) went 0.057 → 0.035 down to 0.020 → 0.043 Rank IC from validation to test; Kronos `a-rank64` (§13.11) passed H1 (Taiwan 1-day 0.064 [0.027, 0.102]) but failed H2 (Kronos − PatchTST −0.002 [−0.021, +0.018]). Conclusion: Kronos is not the backbone; the ranking skill lives in the features and the cross-sectional modelling.

#### 4.2 Taiwan main line: feature LightGBM and cross-stock DL (§13.12–§13.13)

- Route decision (2026-10-22): a MASTER-style cross-stock transformer (D0) paired against LightGBM on the same features (B0) under one walk-forward; the strategy layer starts with top-k dropout.
- **B0** (14 folds × 14 horizons × 3 seeds = 588 models) P1 ∪ P2 Rank IC: short cell (h = 1) 0.118 [0.109, 0.127], holding cell (h = 5–10) 0.086, long cell 0.083; the P2 holding cell fell to 0.034 [−0.000, 0.073].
- **D0** short 0.122, holding 0.092; D0 − B0 paired difference short +0.0043 [+0.0015, +0.0071] (H-A passed in one cell); ablation A1 showed the gain came from the three-seed ensemble, not from inter-stock attention.
- Top-k dropout (5 of 50 names replaced daily): TWSE P1 ∪ P2 net excess −6.3% [−19.0%, +7.1%], **H-B failed**: 10% daily turnover costs about 17% a year, more than the gross excess.

#### 4.3 Strategy layer, first round: band portfolio and TAIEX futures (§13.14, §13.15)

- The band rule (buy into the top K, sell out of the top m × K, at most K/10 entries a day) cut turnover to 2–2.5% a day and the drag to 3.4–4.8% a year; P1 net excess +5.4% (interval spanning zero), **P2 gross excess −27% to −31%**, P1 ∪ P2 net excess −4.7% [−19.7%, +10.5%], **H-D failed**; the TX-hedged version lost −12.5% a year with 27–28 margin calls. Absolute return +23–24% a year, Sharpe 1.5–1.7, against +41% a year for the TAIEX total-return index.
- Stock-futures short day trade (§13.15): the h = 1 gross spread was almost all on the short side (bottom decile −0.36% to −0.45% intraday a day); on official TAIFEX opens and settlements gross +61.5% a year, but one tick of slippage per side (about 0.23%) made it −63% a year net, **H-E failed**: execution-limited (spread), not model-limited.

#### 4.4 Residualized labels (§13.16)

- The P2 failure came from size and industry factor returns in a concentrated large-cap rally: the holding-cell Rank IC against the residual target (net of market, size, industry) stayed at 0.065 [0.046, 0.086] in P2 while the raw-alpha IC kept only 40–50% of its P1 level.
- B0 retrained on `residual_v1a`: the short cell against raw alpha unchanged (paired difference −0.0002), the P2 holding cell +0.0090 [+0.0055, +0.0125], **H-F passed**; the label became the standard. D0 and D1 (frozen Kronos-small, Chronos-2-small) retrained on it: D0's gain over B0 stayed in the short cell (+0.004), and neither pretrained encoder beat a single-seed D0; the DL route left the main line.

#### 4.5 Exposure diagnostics and orthogonalization (§13.17 step 0, §13.19 supplements 2–4)

- Diagnosis: the smoothed h = 10 score correlated −0.24 to −0.45 with 60-session beta and −0.1 to −0.26 with trailing 20/60-session returns (a low-beta, anti-momentum tilt); the P2 loss was structural exposure, not vanished skill.
- **Six scenarios (§13.19, original construction)**: 0 of 36 cells met the institution bar; the S1 short book (P2's large-cap winners) lost 61–89% a year.
- **Orthogonalization (supplement 2, v1)**: with the same band parameters S1's P1 ∪ P2 institution net return went B0 TWSE −6.3% → **+17.3% [−0.6%, +37.3%]** (Sharpe 1.04), TPEX +19.4% [+4.9%, +34.4%] (1.50), TW +18.3% [+5.5%, +33.6%] (1.46), every P2 positive, book β −0.15 to −0.20; S2 added +2% to +9%; 10 of 36 cells met the bar. But v1 was designed after seeing P2, so P1 ∪ P2 is informational only.
- **Execution layer (supplement 4)**: with the borrowable list, quota, borrow-failure and recall draws, B0 TW S1 kept +15.3% [+2.5%, +30.3%], 9 of 12 cells at the bar; capital scenarios changed from skip to cap-and-scale.
- **0050 core overlay (§13.17 supplement)**: all 24 cells lost to 0050 buy-and-hold (P1 ∪ P2 +46.2% a year); on 2026-10-26 the user retired that bar.

#### 4.6 Free chip data (§13.18)

- 67 chip columns (five institutional investor classes' cumulative ratios and z-scores, margin, securities lending, day-trade ratio, attention / disposition status and distance), conservative availability rule, float estimated as the margin limit × 4.
- **H-H passed**: F2 − F0 holding-cell residual Rank IC +0.0173 [+0.0128, +0.0214] (P1 +0.0163, P2 +0.0198); the liquidity-universe h = 10 residual quartile spread rose 0.65% → 1.04% (1.59×, bar 1.5). **G1 passed** (liquidity universe +0.0202, small/mid caps +0.0191). F1 (chip only) reached a P2 holding-cell residual IC of 0.090, above the 183 price-volume columns' 0.065.
- Sub-analysis: the gain was larger among recent winners (+0.0293 vs +0.0179) but significant in both periods and groups: the chip features carry a period-independent signal, not a P2-only conditional reversal.
- User decision (2026-10-28): G1 passed but no broker-branch data purchase for now; §13.20 proceeds with F0 / F2 in hand.

### 5. Final strategy design (§13.20, every parameter fixed in advance)

| Strategy | Construction | Parameters | Source |
| --- | --- | --- | --- |
| **S1 multi-day long-short** | daily v1-orthogonalized score (to 60-session beta, trailing 20/60-session returns); long band plus a short band on the reversed ranking; the short book under the execution layer | h = 10, K = 50, m = 3, at most 5 entries a day | §13.19 item 7 selection; supplements 2 / 4 construction |
| **S2 = S1 + TX overlay** | TX position of −β × capital, β from the book's trailing 60-session gross returns on the index, settlement to settlement, strictly ex ante | β window 60, at least 30 observations | §13.19 item 5 |
| **S-A weekly low-turnover long** | rebalance on the last signal date of each ISO week; v1-orthogonalized h = 5 score; top-K long band, sell only out of the top m × K; at most 50% × K replacements per week, worst-ranked first; no shorts, no futures | K ∈ {30, 50} × m ∈ {2, 3} selected once on TWSE, P1, institution cost, fractional, net Sharpe | §13.18 item 7, §13.20 item 3 |

- **Markets**: TWSE, TPEX, TW (half each). **Cost profiles**: retail and institution side by side, slippage × 0 / 1 / 2, capital fractional / NT$300M / NT$1B.
- **Selection rule (pre-registered)**: institution primary cost, TW, fractional; the (model ∈ {F0, F2}, strategy ∈ {S1, S2, S-A}) cell with the highest P1 ∪ P2 net Sharpe is the final selection.
- **H-J (P3 confirmation)**: in P3 under the institution primary cost the selected cell has (i) annualized net return > 0 and (ii) a maximum drawdown not deeper than P1 ∪ P2's. Both → "paper-trading candidate"; otherwise research status, no re-selection.
- **Code**: `backtest/twbacktest/scenario_suite.py` (`config/scenario_suite_v1_p3.json`), `backtest/twbacktest/weekly_band.py`, `backtest/twbacktest/strategy_decision.py` (`config/strategy_decision.json`).

### 6. Backtest results (P1, P2, P1 ∪ P2, P3)

> Every number comes from `scenario_suite --config config/scenario_suite_v1_p3.json` (F0, F2) and `strategy_decision`; 【TBD】 cells are filled after the P3 predictions are downloaded and scored, with no parameter changed.

#### 6.1 Consistency of the retrained models with the registered results (§13.20 item 9 (f))

| Cell | Registered §13.13 / §13.18 value (v1 dataset, 14 folds) | New dataset, 16 folds | Inside the original 95% interval |
| --- | --- | --- | --- |
| F0 holding-cell residual Rank IC, P1 ∪ P2 | 【TBD: from chip_increment_results】 | 【TBD】 | 【TBD】 |
| F2 holding-cell residual Rank IC, P1 ∪ P2 | 【TBD】 | 【TBD】 | 【TBD】 |
| F2 − F0 paired difference, P1 ∪ P2 | +0.0173 [+0.0128, +0.0214] | 【TBD】 | 【TBD】 |

#### 6.2 S1 / S2 / S-A main table (institution primary cost, fractional, v1)

| Model | Strategy | Market | Period | Net return (annual) | 95% interval | Sharpe | Max drawdown | Turnover | Cost drag | Blocked buys |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | S1 | TW | P1 | 【TBD】 | | | | | | |
| F0 | S1 | TW | P2 | 【TBD】 | | | | | | |
| F0 | S1 | TW | P1 ∪ P2 | 【TBD】 | | | | | | |
| F0 | S1 | TW | P3 | 【TBD】 | | | | | | |
| … | S2, S-A; F2; TWSE, TPEX | | | 【TBD: generated from grid.csv】 | | | | | | |

#### 6.3 Retail cost profile

【TBD: the table of 6.2 under the retail profile.】

#### 6.4 S-A comparison columns

| Model | Market | Period | S-A net (annual) | Equal-weight universe | 0050 | TAIEX TR index | S-A max drawdown | 0050 drawdown | Weekly turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 【TBD】 | | | | | | | | | |

#### 6.5 Selection and H-J

- Candidate table (P1 ∪ P2 institution net Sharpe, TW): 【TBD: strategy_decision candidates】.
- **Final selection**: 【TBD: model × strategy】.
- **H-J**: P3 net return 【TBD】 [【TBD】], P3 maximum drawdown 【TBD】 against P1 ∪ P2's 【TBD】; (i) 【TBD】, (ii) 【TBD】 → status 【TBD: paper_trading_candidate / research】.
- **Execution-layer sensitivities** (failure multiplier 0 / 1 / 2, recall 0 / 0.5%): 【TBD】.
- **Capital scenarios** (NT$300M, NT$1B): 【TBD】.

#### 6.6 Chip increment at the strategy level (informational)

| Strategy | Market | F2 − F0, P1 ∪ P2 annual difference | 95% interval | P3 difference | Increment |
| --- | --- | --- | --- | --- | --- |
| 【TBD】 | | | | | |

### 7. Defects and limitations of the v1 dataset (stated in advance, §13.20 item 9)

(a) **`adjusted_close` carries dividends only, not splits, capital reductions or par-value changes**: since 2023, 109 stocks show a single-session adjusted-open jump above 25% (0050 split 1:4 on 2025-06-18); their training labels and price-type features are wrong around the events. The band engines compute each name's open-to-open return only between adjacent signal dates and exit on a halt, so position returns are unaffected, but the model scores are. P3 event count: 【TBD: from `configs/reference/tw_capital_actions_20261026.csv`】.
(b) **The v2 correction is implemented but neither Pod-verified nor used to train any model** (branch `claude/dataset-v2-corporate-actions-20261026`), so it is not used for P3; a v2 retrain would require recomputing P1 ∪ P2 and P3.
(c) **P2 has been seen**: the v1 design of S1 / S2 was decided after seeing P2, so P1 ∪ P2 figures are informational and only P3 is out of sample. S-A is new, its parameters are selected on P1, and both P2 and P3 are out of sample for it.
(d) **P3 has only about 95 sessions** in a single market regime (June to October 2026); passing or failing cannot be extrapolated into a long-run conclusion.
(e) Chip-feature availability uses the conservative rule (before the T+1 open); `FDLR_Z60` is always missing.
(f) If the P1 ∪ P2 cells of the F0 and F2 retrained on the new dataset differ from the §13.13 / §13.18 registered values by more than their 95% intervals, the data is checked before P3 is scored (section 6.1).
(g) The out-of-sample window 2023-01 to 2026-10 contains the April 2025 crash but not the 2022 bear market (training only); broker-branch and shareholder-dispersion data were not bought, so the chip features cover only the free official items.

### 8. Improvement roadmap

| Order | Item | Condition | Basis |
| --- | --- | --- | --- |
| 1 | Decide from H-J whether the selected strategy enters paper trading; define its record-keeping and stop rules | section 6.5 | §13.20 item 6 |
| 2 | G2 event study (disposition announcements, first attention listings, day-trade-ratio jumps) → decide whether the S-B reversal strategy is built | event tables delivered | §13.18 item 6 |
| 3 | Dataset v2 (split, capital-reduction and par-value adjustments): retrain F0 / F2 and recompute P1 ∪ P2 and P3 | user decision after this report; decouple the provider-cache revision from the dataset revision first | §13.17 supplement (h), §13.20 item 9 |
| 4 | Retrain D0 on the F2 features (three seeds) | user decision; the gain is concentrated in 2023–2024 and declining | §13.18 item 3 |
| 5 | Broker-branch data purchase (G1 passed) | user decision; no paid data for now | §13.18 item 10 |
| 6 | First live verification of the chip-data update on the Pod (`cpu-chip`, the Cloud Run relay's TPEx routes) | the next chip Pod | README "Updating the chip data on the Pod" |

### 9. Artifacts and reproduction

- Protocol: `docs/experiment_protocol.md` §13.11–§13.20; per-section result reports under `docs/*_results.md`.
- Predictions: `artifacts/lgbm/<F0 launch>/predictions.parquet`, `artifacts/lgbm/<F2 launch>/predictions.parquet` (`tw_quarterly_v2`, 16 folds) 【TBD launch ids】.
- Backtests: `cd backtest && uv run python -m twbacktest.scenario_suite --config config/scenario_suite_v1_p3.json --chip-data-root ../../tw_chip_data/data --predictions ../artifacts/lgbm/<F0>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F0 --predictions ../artifacts/lgbm/<F2>/predictions.parquet --model t-lgbm-alpha-residual_v1a --label F2 --prices ../artifacts/datasets/<dataset>/raw/market.parquet`, then `uv run python -m twbacktest.strategy_decision --suite <output directory>`.
- Output directories: 【TBD】.
