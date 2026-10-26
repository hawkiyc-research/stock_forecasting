# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-26。

### 現況

正交化後（分數對 60 日 beta 與前 20／60 日報酬每日取殘差，V1）的現股多日多空 S1／S2，在機構成本下達到 §13.19 的資訊性門檻；2026-10-26 補充四把空頭腿的「全部可借券」換成 `tw_chip_data` 的可借券名單、交易所次日限額、依規模分層的借不到機率、召回與分層借券費（五個隨機重複平均）之後，B0 TW 的 S1 仍 +15.3% [+2.5%, +30.3%]（Sharpe 1.24）、三市場 P2 全正，九格達機構門檻（原十格），事前判讀「補充二的結論在可得性修正後成立」；名單本身幾乎不改變結果，分層費與抽樣各吃 1 到 3 個百分點，借不到倍率 0 到 2 的差異在五個重複的極差之內。設計仍在看過 P2 之後決定、絕對報酬低於同期 0050，正式判定留給 P3。籌碼資料線（`tw_chip_data`）第一階段已完成（2026-10-26，head f221e30）：16 個免費官方資料集 2012-01-02 至 2026-10-22，含 `eligibility`、`sbl` 與兩張事件表；資料契約在該專案 `docs/INTEGRATION.md`。

### 執行順序

| # | 步驟 | 依據 | 前置條件 | 訓練需求 | 產出 | 誰做 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.17 補充**：0050 核心加正交化多空疊加，變體 A（質押借款）與 B（台指期提供 β），H-G 資訊性判讀；空頭腿沿用補充四的主要假設（名單、限額、× 1、分層費、五個重複） | §13.17 補充（已登記設計與門檻） | 無 | 無（本機回測；新模組 `core_overlay.py`） | 雙語報告、協定結果 | 本專案 |
| 2 | **§13.18 籌碼特徵庫 `chip_tw_v1`** 與 B0 的 F1／F2 | §13.18 第 1 到 4 項 | curated 層已交付；`stock-forecasting-build-features --spec` 支援多規格 | **RunPod CPU**：`features --spec`、`lgbm --spec` 各一次（F1、F2 各一個 B0 walk-forward，三 seed） | 特徵庫、兩組 B0 預測 | 本專案寫程式；使用者啟動 Pod |
| 3 | **G1 與 H-H**：F1 − F0、F2 − F0 配對差，全宇宙與中小型股分列、分期、強勢股子分析（以 V1 分數為基線） | §13.18 第 5 項與 G1 子分析 | 第 2 項 | 無 | 協定結果、報告 | 本專案 |
| 4 | **G2**：處置、注意首次公告、當沖比跳升三類事件的反轉步驟 0 | §13.18 第 6 項 | 事件表已交付 | 無 | 事件研究模組、報告 | 本專案 |
| 5 | 分點與集保購買決定 | §13.18 第 10 項 | 第 3、4 項：G1 或 G2 任一通過才買分點（Sponsor Pro 一個月）；集保只在 S-C 進入實作時買 | — | 購買與否的紀錄 | 使用者決定 |
| 6 | （若 F2 通過 H-H 且殘差幅度明顯）D0 在 F2 特徵上重訓 | §13.18 第 3 項 | 第 3 項 | **RunPod GPU**：`xs-train`（三 seed） | D0 F2 預測 | 本專案寫程式；使用者啟動 Pod |
| 7 | **P3 樣本外**：資料集延伸到 2026-06 之後（新資料集身分）、新增季度折重訓 B0（F0，必要時 F2），以 V1、無配平、固定參數、補充四的執行層重跑補充二與第 3 項 | §13.19 補充二處置、補充四第 (f) 項、§13.17 補充第 6 項 | 2026-06 之後數個月資料；第 1 項 | **RunPod CPU**：資料準備與 B0 新折；D0 僅在被列為候選時（GPU） | 正式判定 | 使用者下載資料並啟動 Pod；本專案計分 |

### 訓練需求說明

第 1、3、4 項是純回測，用現有預測在本機跑。第 2、6、7 項需要訓練：第 2 項是 B0 LightGBM 在籌碼特徵上的 walk-forward（CPU Pod），第 6 項是 D0（GPU Pod），第 7 項是 P3 必須的新折重訓（新資料進來後舊折的預測不會自動延伸）。所有訓練都在 RunPod，Pod 由使用者啟動，每次 `--maxRuntime 24h`，啟動前先 `sync_project_to_runpod_volume.sh --apply`。

### 目前不做的事

- 購買分點或集保（等第 6 項）。
- 在 P3 之前宣稱任何策略可用；D0 的結果集中在 2023 到 2024 且逐年遞減，不列為候選。
- 重選 band 參數（固定為 §13.19 的選定值）。
- 推送分支：`claude/chip-signals-20261025` 與 master 的 v0.3.0、v0.4.0 標籤待使用者看過 README 與報告後決定。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-26.

### Where things stand

After orthogonalizing the score daily against the 60-session beta and the trailing 20 / 60-session returns (V1), the multi-day stock long-short S1 / S2 meets the informational bar of §13.19 under institution costs. On 2026-10-26 supplement 4 replaced the short book's "every name is borrowable" assumption with the `tw_chip_data` borrowable lists, the exchange's next-session quota, borrow-failure probabilities by size tercile, recalls and tiered borrow fees (five random replicates averaged): B0's S1 on TW stays at +15.3% [+2.5%, +30.3%] (Sharpe 1.24) with every P2 positive on the three markets, nine cells clear the institution bar (ten before), and the pre-registered reading "supplement 2's conclusion survives the availability correction" holds; the list itself barely moves the result, tiered fees and the failure draws each cost 1 to 3 points, and failure multipliers 0 to 2 differ by less than the range across replicates. The design was still chosen after seeing P2 and the absolute return trails 0050 over the same period, so the formal decision waits for P3. The chip-data line (`tw_chip_data`) finished phase 1 (2026-10-26, head f221e30): 16 free official datasets from 2012-01-02 to 2026-10-22, including `eligibility`, `sbl` and the two event tables; the data contract is that project's `docs/INTEGRATION.md`.

### Execution order

| # | Step | Basis | Prerequisite | Training | Output | Who |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.17 supplement**: 0050 core plus orthogonalized long-short overlay, variant A (loan secured on 0050) and B (β from TX), informational H-G; the short book uses supplement 4's primary assumptions (lists, quota, × 1, tiered fees, five replicates) | §13.17 supplement (design and bar registered) | None | None (local backtest; new module `core_overlay.py`) | Bilingual report, protocol result | This project |
| 2 | **§13.18 chip feature library `chip_tw_v1`** and B0's F1 / F2 | §13.18 items 1–4 | Curated layer delivered; multi-spec support in `stock-forecasting-build-features --spec` | **RunPod CPU**: one `features --spec` and one `lgbm --spec` each (a B0 walk-forward for F1 and for F2, three seeds) | Feature store, two sets of B0 predictions | This project writes the code; the user launches the Pods |
| 3 | **G1 and H-H**: paired F1 − F0 and F2 − F0, full universe and small / mid caps, by period, recent-winner sub-analysis (V1 score as the baseline) | §13.18 item 5 and the G1 sub-analysis | Step 2 | None | Protocol result, report | This project |
| 4 | **G2**: reversal step 0 on 處置 announcements, first 注意 listings and day-trade-ratio jumps | §13.18 item 6 | Event tables delivered | None | Event-study module, report | This project |
| 5 | Decision on buying 分點 and 集保 | §13.18 item 10 | Steps 3 and 4: 分點 (Sponsor Pro, one month) only if G1 or G2 passes; 集保 only when S-C enters implementation | — | Record of the decision | The user |
| 6 | (Only if F2 passes H-H with a clear residual spread) retrain D0 on the F2 features | §13.18 item 3 | Step 3 | **RunPod GPU**: `xs-train` (three seeds) | D0 F2 predictions | This project writes the code; the user launches the Pod |
| 7 | **P3 out of sample**: extend the dataset past 2026-06 (new dataset identity), retrain B0 on the new quarterly folds (F0, F2 if needed), rerun supplement 2 and step 3 with V1, no matching, fixed parameters and supplement 4's execution layer | §13.19 supplement 2 disposition, supplement 4 item (f), §13.17 supplement item 6 | Several months of data after 2026-06; step 1 | **RunPod CPU** for data preparation and the new B0 folds; D0 only if it is a candidate (GPU) | The formal decision | The user downloads data and launches the Pods; this project scores |

### Training notes

Steps 1, 3 and 4 are pure backtests on existing predictions, run locally. Steps 2, 6 and 7 need training: step 2 is B0 LightGBM walk-forwards on the chip features (CPU Pod), step 6 is D0 (GPU Pod), step 7 is the new-fold retraining that P3 requires (existing folds do not extend themselves when new data arrives). All training runs on RunPod, Pods are launched by the user with `--maxRuntime 24h`, after `sync_project_to_runpod_volume.sh --apply`.

### Not being done now

- Buying 分點 or 集保 (wait for step 6).
- Calling any strategy usable before P3; D0's results concentrate in 2023–2024 and decay, so D0 is not a candidate.
- Re-selecting the band parameters (fixed at the §13.19 selection).
- Pushing: branch `claude/chip-signals-20261025` and master's v0.3.0 / v0.4.0 tags wait for the user's review of the README and reports.
