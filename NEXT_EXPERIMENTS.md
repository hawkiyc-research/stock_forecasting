# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-29（§13.20 計分完成）。

### 現況

§13.20 已計分（2026-10-29）：F0／F2 在延長資料集（到 2026-10-23）的 16 折 walk-forward 上重訓，一致性檢查通過；選定規則（機構主要成本、TW、分數位置、P1 ∪ P2 淨 Sharpe 最高）選出 **F2 × S-A**（Sharpe 2.38、+51.8%／年），但 P3（2026-06-01 至 10-02 的 85 個區間）確認 **H-J 未通過**——淨報酬 +3.5% > 0 成立、最大回撤 −25.7% 比 P1 ∪ P2 的 −22.6% 深不成立（7 月一次 −17% 的市場修正打在 100% 多頭上）。依登記不重選、不改建構，**沒有策略進入試運行**。市場中性的 S2（F2）同窗口 +34.1%、回撤 −8.1%，但也因回撤比 P1 ∪ P2 深而過不了 (ii)。籌碼在策略層只對多頭 band 有增量（S-A F2 − F0 +16.3% [+5.5%, +27.7%]），對多空沒有。兩份報告：`docs/research_full_report.md`、`docs/selected_strategy_report.md`。

### 執行順序

| # | 步驟 | 依據 | 前置條件 | 訓練需求 | 產出 | 誰做 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **（使用者決定方向）下一個策略實驗的登記**：(a) S-A 加 β 對沖（台指期）或波動目標，讓多頭 band 的回撤不由市場決定——新建構，參數要在 P1 選、P2 與 P3 只看；(b) 或以市場中性 S2（F2）為主線，重新登記一段 P3 之後的確認期（H-J 的回撤條件對 S2 在 7 月也過不了，確認期要夠長）。兩者都要先寫 §13.21 才能實作 | §13.20 結果 (j) | 使用者選 (a)／(b)／都不做 | 無（本機回測；S2 用現有預測） | 新章節登記、回測、報告更新 | 使用者決定；本專案登記與實作 |
| 2 | **volume 清理**：下一個 Pod 一開始先 `python3 scripts/runpod_invalid_chip_results.py`，確認後 `--apply --delete`（被取代的樣本宇宙、舊 store、失敗 launch） | README「清掉錯誤與死掉的結果」 | 下一個 CPU Pod | 無 | 釋放空間 | 使用者在 Pod 上執行 |
| 3 | **G2**：處置、注意首次公告、當沖比跳升三類事件的反轉步驟 0（事件研究模組） | §13.18 第 6 項 | 事件表已交付 | 無 | 事件研究模組、報告；決定 S-B 要不要實作 | 本專案 |
| 4 | （使用者決定，暫不執行）分點購買：G1 已通過，但 2026-10-28 決定先不買付費資料 | §13.18 第 10 項 | — | — | 購買與否的紀錄 | 使用者決定 |
| 5 | （使用者決定）資料集 v2（分割與減資調整）重訓：§13.20 的報告與 v1 缺陷聲明已出（P3 期間 48 次 >25% 的還原開盤跳動、42 檔）；程式已完成（分支 `claude/dataset-v2-corporate-actions-20261026`，需先 rebase 到 master）；v2 的 provider 快取修訂與資料集修訂綁在一起，整批重抓 API 之前要先拆開 | §13.17 補充結果 (h)、§13.20 第 9 項 | 使用者決定 | **RunPod CPU**：prepare → features F0、F2 → lgbm F0、F2 | v2 資料集與 v1 對 v2 比較 | 使用者決定；本專案計分 |
| 6 | （使用者決定）D0 在 F2 特徵上重訓 | §13.18 第 3 項 | 使用者決定 | **RunPod GPU**：`xs-train` | D0 F2 預測 | 本專案寫程式；使用者啟動 Pod |

### 訓練需求說明

第 1 項是純回測，在本機跑（S2 用現有的 F2 預測；S-A 的對沖版只加台指期腿）。第 5 項要重做資料準備、特徵庫與兩個模型的 16 折重訓（四到五個 CPU Pod）；第 6 項要一個 GPU Pod。第 3 項不需要訓練。

### 目前不做的事

- 以「勝過 0050 買進持有」作為任何階段的門檻（2026-10-26 決定；§13.17 補充到此結束，不在 P3 重算）。
- 購買集保（只在 S-C 進入實作時）；分點由第 4 項決定。
- 把 F2 × S-A 或 F2 × S2 當成可用策略：H-J 未通過，兩者都是研究狀態；不重選 band 參數（固定為 §13.19／§13.20 的選定值）。
- 推送分支：master 的 v0.3.0、v0.4.0 標籤待使用者看過 README 與報告後決定。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-29 (§13.20 scored).

### Where things stand

§13.20 is scored (2026-10-29): F0 / F2 were retrained on the extended dataset (to 2026-10-23) over the 16-fold walk-forward and the consistency check passed; the selection rule (institution primary cost, TW, fractional, highest P1 ∪ P2 net Sharpe) picks **F2 × S-A** (Sharpe 2.38, +51.8% a year), but the P3 confirmation (85 intervals, 2026-06-01 to 10-02) **fails H-J**: net return +3.5% > 0 holds, the −25.7% maximum drawdown deeper than P1 ∪ P2's −22.6% fails (July's −17% market correction landed on a 100% long book). As registered there is no re-selection and no construction change: **no strategy enters paper trading**. The market-neutral S2 (F2) made +34.1% with a −8.1% drawdown in the same window but also fails (ii) against its −6.1%. At the strategy level the chip features add only to the long band (S-A F2 − F0 +16.3% [+5.5%, +27.7%]), not to the long-short books. Reports: `docs/research_full_report.md`, `docs/selected_strategy_report.md`.

### Execution order

| # | Step | Basis | Prerequisite | Training | Output | Who |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **(User decides the direction) registration of the next strategy experiment**: (a) S-A with a beta hedge (TX) or a volatility target, so the long band's drawdown is not the market's — a new construction whose parameters are selected on P1 with P2 and P3 only read; (b) or the market-neutral S2 (F2) as the main line with a re-registered confirmation period after P3 (H-J's drawdown condition also failed S2 in July, so the period must be long enough). Either one is written as §13.21 before any implementation | §13.20 result (j) | the user picks (a) / (b) / neither | None (local backtests; S2 uses the existing predictions) | new section, backtests, report updates | the user decides; this project registers and implements |
| 2 | **Volume cleanup**: at the start of the next Pod run `python3 scripts/runpod_invalid_chip_results.py`, check, then `--apply --delete` (superseded sample universes, old stores, failed launches) | README "Removing wrong and dead results" | the next CPU Pod | None | freed space | the user, on the Pod |
| 3 | **G2**: reversal step 0 on 處置 announcements, first 注意 listings and day-trade-ratio jumps (event-study module) | §13.18 item 6 | Event tables delivered | None | Event-study module, report; decides whether S-B is built | This project |
| 4 | (User decision, on hold) broker-branch purchase: G1 passed, but on 2026-10-28 the user decided not to buy paid data for now | §13.18 item 10 | — | — | Record of the decision | The user |
| 5 | (User decision) dataset v2 (split and capital-reduction adjustments) retraining: the §13.20 reports and the v1 defect statement are out (48 adjusted-open jumps above 25% across 42 stocks inside P3); code done (branch `claude/dataset-v2-corporate-actions-20261026`, to be rebased onto master); v2 ties the provider-cache revision to the dataset revision, so decouple them before a full API re-download | §13.17 supplement result (h), §13.20 item 9 | User decision | **RunPod CPU**: prepare → features F0, F2 → lgbm F0, F2 | The v2 dataset and the v1-vs-v2 comparison | The user decides; this project scores |
| 6 | (User decision) retrain D0 on the F2 features | §13.18 item 3 | User decision | **RunPod GPU**: `xs-train` | D0 F2 predictions | This project writes the code; the user launches the Pod |

### Training notes

Step 1 is pure backtesting on the local machine (S2 uses the existing F2 predictions; the hedged S-A only adds a TX leg). Step 5 redoes data preparation, the feature stores and both models' 16-fold retraining (four to five CPU Pods); step 6 needs one GPU Pod. Step 3 needs no training.

### Not being done now

- "Beating 0050 buy-and-hold" as a bar at any stage (decided 2026-10-26; the §13.17 supplement ends there and is not recomputed on P3).
- Buying shareholder-dispersion data (only when S-C enters implementation); the broker-branch decision is step 4.
- Treating F2 × S-A or F2 × S2 as a usable strategy: H-J failed and both stay in research status; no re-selection of the band parameters (fixed at the §13.19 / §13.20 values).
- Pushing branches: the v0.3.0 and v0.4.0 tags on master wait for the user's review of the README and the reports.
