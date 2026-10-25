# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-26。

### 現況

正交化後（分數對 60 日 beta 與前 20／60 日報酬每日取殘差，V1）的現股多日多空 S1／S2，在機構成本下 P1 ∪ P2 年化淨 +17% 到 +29%、β 約 −0.2、B0 四個曆年全正，達到 §13.19 的資訊性門檻；但設計在看過 P2 之後決定、空頭腿假設全部可借券、絕對報酬低於同期 0050。籌碼資料線（`tw_chip_data`）第一階段免費資料回填中。

### 執行順序

| # | 步驟 | 依據 | 前置條件 | 訓練需求 | 產出 | 誰做 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 回填完成後 curate 全部 13 個資料集、events、calendar、manifest、validate | `tw_chip_data` DECISION_SUMMARY §10 | 回填結束（回填在終端分頁執行中） | 無 | curated 層、manifest、驗證摘要 | 資料專案；使用者在終端 commit |
| 2 | 名單品項：「平盤下得融（借）券賣出」（兩所，可回溯至 2013-09-23）、「得為融資融券」 | `tw_chip_data` §10.3；§13.19 補充三第 3 項 | 第 1 項 | 無 | 兩張每日名單表 | 資料專案 |
| 3 | **補充四**：執行不確定性與可借券宇宙，重算正交化 S1／S2——空頭腿限可借券名單、借不到的機率依規模三等分抽樣（機構 2／10／25%，散戶 5／20／50%，× 0 與 × 2 敏感度）、借券召回機率、借券費分層（1.5／3／6%）、資本情境改為封頂而非跳過；5 個隨機重複 | §13.19 補充四（已登記） | 第 2 項 | 無（本機回測；`scenario_suite` 新增 `execution` 設定） | 報告新節、協定結果註記 | 本專案 |
| 4 | **§13.17 補充**：0050 核心加正交化多空疊加，變體 A（質押借款）與 B（台指期提供 β），H-G 資訊性判讀 | §13.17 補充（已登記設計與門檻） | 第 3 項 | 無（本機回測；新模組 `core_overlay.py`） | 雙語報告、協定結果 | 本專案 |
| 5 | **§13.18 籌碼特徵庫 `chip_tw_v1`** 與 B0 的 F1／F2 | §13.18 第 1 到 4 項 | 第 1 項的 curated 層；`stock-forecasting-build-features --spec` 支援多規格 | **RunPod CPU**：`features --spec`、`lgbm --spec` 各一次（F1、F2 各一個 B0 walk-forward，三 seed） | 特徵庫、兩組 B0 預測 | 本專案寫程式；使用者啟動 Pod |
| 6 | **G1 與 H-H**：F1 − F0、F2 − F0 配對差，全宇宙與中小型股分列、分期、強勢股子分析（以 V1 分數為基線） | §13.18 第 5 項與 G1 子分析 | 第 5 項 | 無 | 協定結果、報告 | 本專案 |
| 7 | **G2**：處置、注意首次公告、當沖比跳升三類事件的反轉步驟 0 | §13.18 第 6 項 | 第 1 項（事件表） | 無 | 事件研究模組、報告 | 本專案 |
| 8 | 分點與集保購買決定 | §13.18 第 10 項 | 第 6、7 項：G1 或 G2 任一通過才買分點（Sponsor Pro 一個月）；集保只在 S-C 進入實作時買 | — | 購買與否的紀錄 | 使用者決定 |
| 9 | （若 F2 通過 H-H 且殘差幅度明顯）D0 在 F2 特徵上重訓 | §13.18 第 3 項 | 第 6 項 | **RunPod GPU**：`xs-train`（三 seed） | D0 F2 預測 | 本專案寫程式；使用者啟動 Pod |
| 10 | **P3 樣本外**：資料集延伸到 2026-06 之後（新資料集身分）、新增季度折重訓 B0（F0，必要時 F2），以 V1、無配平、固定參數重跑補充二與第 4 項 | §13.19 補充二處置、§13.17 補充第 6 項 | 2026-06 之後數個月資料；第 3、4 項 | **RunPod CPU**：資料準備與 B0 新折；D0 僅在被列為候選時（GPU） | 正式判定 | 使用者下載資料並啟動 Pod；本專案計分 |

### 訓練需求說明

第 3、4、6、7 項是純回測，用現有預測在本機跑。第 5、9、10 項需要訓練：第 5 項是 B0 LightGBM 在籌碼特徵上的 walk-forward（CPU Pod），第 9 項是 D0（GPU Pod），第 10 項是 P3 必須的新折重訓（新資料進來後舊折的預測不會自動延伸）。所有訓練都在 RunPod，Pod 由使用者啟動，每次 `--maxRuntime 24h`，啟動前先 `sync_project_to_runpod_volume.sh --apply`。

### 目前不做的事

- 購買分點或集保（等第 8 項）。
- 在 P3 之前宣稱任何策略可用；D0 的結果集中在 2023 到 2024 且逐年遞減，不列為候選。
- 重選 band 參數（固定為 §13.19 的選定值）。
- 推送分支：`claude/chip-signals-20261025` 與 master 的 v0.3.0、v0.4.0 標籤待使用者看過 README 與報告後決定。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-26.

### Where things stand

After orthogonalizing the score daily against the 60-session beta and the trailing 20 / 60-session returns (V1), the multi-day stock long-short S1 / S2 earns +17% to +29% a year net under institution costs on P1 ∪ P2 with β about −0.2 and every calendar year positive for B0, meeting the informational bar of §13.19; but the design was chosen after seeing P2, the short book assumes every name is borrowable, and the absolute return trails 0050 over the same period. The chip-data line (`tw_chip_data`) is backfilling its phase-1 free data.

### Execution order

| # | Step | Basis | Prerequisite | Training | Output | Who |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | After the backfill: curate all 13 datasets, events, calendar, manifest, validate | `tw_chip_data` DECISION_SUMMARY §10 | Backfill finished (running in a terminal tab) | None | Curated layer, manifest, validation summary | Data project; the user commits in the terminal |
| 2 | Eligibility lists: "short-sale-below-reference allowed" (both exchanges, back to 2013-09-23) and "margin-eligible" | `tw_chip_data` §10.3; §13.19 supplement 3 item 3 | Step 1 | None | Two daily list tables | Data project |
| 3 | **Supplement 4**: execution uncertainty and the borrowable universe, rerun the orthogonalized S1 / S2 — short candidates restricted to the borrowable list, borrow failure drawn per name and day by dollar-volume tercile (institution 2 / 10 / 25%, retail 5 / 20 / 50%, × 0 and × 2 sensitivities), recall probability, tiered borrow fees (1.5 / 3 / 6%), capital scenarios capped-and-scaled instead of skipped; 5 random replicates | §13.19 supplement 4 (registered) | Step 2 | None (local backtest; new `execution` block in `scenario_suite`) | New report section, protocol result note | This project |
| 4 | **§13.17 supplement**: 0050 core plus orthogonalized long-short overlay, variant A (loan secured on 0050) and B (β from TX), informational H-G | §13.17 supplement (design and bar registered) | Step 3 | None (local backtest; new module `core_overlay.py`) | Bilingual report, protocol result | This project |
| 5 | **§13.18 chip feature library `chip_tw_v1`** and B0's F1 / F2 | §13.18 items 1–4 | Curated layer of step 1; multi-spec support in `stock-forecasting-build-features --spec` | **RunPod CPU**: one `features --spec` and one `lgbm --spec` each (a B0 walk-forward for F1 and for F2, three seeds) | Feature store, two sets of B0 predictions | This project writes the code; the user launches the Pods |
| 6 | **G1 and H-H**: paired F1 − F0 and F2 − F0, full universe and small / mid caps, by period, recent-winner sub-analysis (V1 score as the baseline) | §13.18 item 5 and the G1 sub-analysis | Step 5 | None | Protocol result, report | This project |
| 7 | **G2**: reversal step 0 on 處置 announcements, first 注意 listings and day-trade-ratio jumps | §13.18 item 6 | Step 1 (event tables) | None | Event-study module, report | This project |
| 8 | Decision on buying 分點 and 集保 | §13.18 item 10 | Steps 6 and 7: 分點 (Sponsor Pro, one month) only if G1 or G2 passes; 集保 only when S-C enters implementation | — | Record of the decision | The user |
| 9 | (Only if F2 passes H-H with a clear residual spread) retrain D0 on the F2 features | §13.18 item 3 | Step 6 | **RunPod GPU**: `xs-train` (three seeds) | D0 F2 predictions | This project writes the code; the user launches the Pod |
| 10 | **P3 out of sample**: extend the dataset past 2026-06 (new dataset identity), retrain B0 on the new quarterly folds (F0, F2 if needed), rerun supplement 2 and step 4 with V1, no matching, fixed parameters | §13.19 supplement 2 disposition, §13.17 supplement item 6 | Several months of data after 2026-06; steps 3 and 4 | **RunPod CPU** for data preparation and the new B0 folds; D0 only if it is a candidate (GPU) | The formal decision | The user downloads data and launches the Pods; this project scores |

### Training notes

Steps 3, 4, 6 and 7 are pure backtests on existing predictions, run locally. Steps 5, 9 and 10 need training: step 5 is B0 LightGBM walk-forwards on the chip features (CPU Pod), step 9 is D0 (GPU Pod), step 10 is the new-fold retraining that P3 requires (existing folds do not extend themselves when new data arrives). All training runs on RunPod, Pods are launched by the user with `--maxRuntime 24h`, after `sync_project_to_runpod_volume.sh --apply`.

### Not being done now

- Buying 分點 or 集保 (wait for step 8).
- Calling any strategy usable before P3; D0's results concentrate in 2023–2024 and decay, so D0 is not a candidate.
- Re-selecting the band parameters (fixed at the §13.19 selection).
- Pushing: branch `claude/chip-signals-20261025` and master's v0.3.0 / v0.4.0 tags wait for the user's review of the README and reports.
