# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-26。

### 現況

正交化後（分數對 60 日 beta 與前 20／60 日報酬每日取殘差，V1）的現股多日多空 S1／S2，在機構成本下達到 §13.19 的資訊性門檻；2026-10-26 補充四把空頭腿的「全部可借券」換成 `tw_chip_data` 的可借券名單、交易所次日限額、依規模分層的借不到機率、召回與分層借券費（五個隨機重複平均）之後，B0 TW 的 S1 仍 +15.3% [+2.5%, +30.3%]（Sharpe 1.24）、三市場 P2 全正，九格達機構門檻（原十格），事前判讀「補充二的結論在可得性修正後成立」；名單本身幾乎不改變結果，分層費與抽樣各吃 1 到 3 個百分點，借不到倍率 0 到 2 的差異在五個重複的極差之內。設計仍在看過 P2 之後決定、絕對報酬低於同期 0050，正式判定留給 P3。§13.17 補充（0050 核心加疊加，2026-10-26）：變體 A（質押借款）與 B（台指期提供 β）在兩個模型、三個市場、兩個成本檔共 24 格的 H-G 全部未通過（0050 買進持有年化 +46.2%，疊加打平或小贏、區間跨 0），選到的 w = 0.7、L = 1.0 在 60% 質押成數下不可融資，疊加改善的是回撤（−25% → −13% 到 −20%）不是絕對報酬；結論維持「沒有勝過 0050 的可部署策略」。**使用者決定（2026-10-26）：勝過 0050 不再是任何階段的門檻**——被動持有 0050 與量化交易是不同的賽道，本專案回到量化交易的賽道，判定一律用 §13.19 第 9 項的業界門檻（機構成本、P1 ∪ P2 淨 Sharpe ≥ 1.0、區間下界 > 0、P2 > 0，資本情境並列，補充四執行層），0050 只作對照。資料發現：資料集 `adjusted_close` 不含分割與減資（0050 一拆四已在 `core.splits` 修正；band 部位報酬未受影響）。籌碼資料線（`tw_chip_data`）第一階段已完成（2026-10-26，head f221e30）：16 個免費官方資料集 2012-01-02 至 2026-10-22，含 `eligibility`、`sbl` 與兩張事件表；資料契約在該專案 `docs/INTEGRATION.md`。

### 執行順序

| # | 步驟 | 依據 | 前置條件 | 訓練需求 | 產出 | 誰做 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.18 籌碼特徵庫 `chip_tw_v1`** 與 B0 的 F1／F2 | §13.18 第 1 到 4 項 | curated 層已交付；`stock-forecasting-build-features --spec` 支援多規格 | **RunPod CPU**：`features --spec`、`lgbm --spec` 各一次（F1、F2 各一個 B0 walk-forward，三 seed） | 特徵庫、兩組 B0 預測 | 本專案寫程式；使用者啟動 Pod |
| 2 | **G1 與 H-H**：F1 − F0、F2 − F0 配對差，全宇宙與中小型股分列、分期、強勢股子分析（以 V1 分數為基線） | §13.18 第 5 項與 G1 子分析 | 第 1 項 | 無 | 協定結果、報告 | 本專案 |
| 3 | **G2**：處置、注意首次公告、當沖比跳升三類事件的反轉步驟 0 | §13.18 第 6 項 | 事件表已交付 | 無 | 事件研究模組、報告 | 本專案 |
| 4 | 分點與集保購買決定 | §13.18 第 10 項 | 第 2、3 項：G1 或 G2 任一通過才買分點（Sponsor Pro 一個月）；集保只在 S-C 進入實作時買 | — | 購買與否的紀錄 | 使用者決定 |
| 5 | （若 F2 通過 H-H 且殘差幅度明顯）D0 在 F2 特徵上重訓 | §13.18 第 3 項 | 第 2 項 | **RunPod GPU**：`xs-train`（三 seed） | D0 F2 預測 | 本專案寫程式；使用者啟動 Pod |
| 6 | **資料管線**：資料集的 `adjusted_close` 加入分割與減資調整（2023 年起 109 檔個股、0050 等 ETF），重建資料集身分；P3 的新資料集一併採用 | §13.17 補充結果 (h) | 資料來源的除權息與減資參考表 | **RunPod CPU**（資料準備） | 新資料集、驗證報告 | 本專案寫程式；使用者啟動 Pod |
| 7 | **P3 樣本外**：資料集延伸到 2026-06 之後（新資料集身分，含第 6 項的調整）、新增季度折重訓 B0（F0，必要時 F2），以 V1、無配平、固定參數、補充四的執行層重跑補充二與第 2 項，門檻為 §13.19 第 9 項（不再比對 0050） | §13.19 補充二處置、補充四第 (f) 項、2026-10-26 的門檻決定 | 2026-06 之後數個月資料；第 6 項 | **RunPod CPU**：資料準備與 B0 新折；D0 僅在被列為候選時（GPU） | 正式判定 | 使用者下載資料並啟動 Pod；本專案計分 |

### 訓練需求說明

第 2、3 項是純回測，用現有預測在本機跑。第 1、5、6、7 項需要 Pod：第 1 項是 B0 LightGBM 在籌碼特徵上的 walk-forward（CPU Pod），第 5 項是 D0（GPU Pod），第 6 項是資料準備（CPU Pod），第 7 項是 P3 必須的新折重訓（新資料進來後舊折的預測不會自動延伸）。所有訓練都在 RunPod，Pod 由使用者啟動，每次 `--maxRuntime 24h`，啟動前先 `sync_project_to_runpod_volume.sh --apply`。

### 目前不做的事

- 以「勝過 0050 買進持有」作為任何階段的門檻（2026-10-26 決定；§13.17 補充到此結束，不在 P3 重算）。
- 購買分點或集保（等第 4 項）。
- 在 P3 之前宣稱任何策略可用；D0 的結果集中在 2023 到 2024 且逐年遞減，不列為候選。
- 重選 band 參數（固定為 §13.19 的選定值）。
- 推送分支：`claude/chip-signals-20261025` 與 master 的 v0.3.0、v0.4.0 標籤待使用者看過 README 與報告後決定。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-26.

### Where things stand

After orthogonalizing the score daily against the 60-session beta and the trailing 20 / 60-session returns (V1), the multi-day stock long-short S1 / S2 meets the informational bar of §13.19 under institution costs. On 2026-10-26 supplement 4 replaced the short book's "every name is borrowable" assumption with the `tw_chip_data` borrowable lists, the exchange's next-session quota, borrow-failure probabilities by size tercile, recalls and tiered borrow fees (five random replicates averaged): B0's S1 on TW stays at +15.3% [+2.5%, +30.3%] (Sharpe 1.24) with every P2 positive on the three markets, nine cells clear the institution bar (ten before), and the pre-registered reading "supplement 2's conclusion survives the availability correction" holds; the list itself barely moves the result, tiered fees and the failure draws each cost 1 to 3 points, and failure multipliers 0 to 2 differ by less than the range across replicates. The design was still chosen after seeing P2 and the absolute return trails 0050 over the same period, so the formal decision waits for P3. §13.17 supplement (0050 core plus overlay, 2026-10-26): variants A (secured loan) and B (β from TX) fail H-G in all 24 cells (two models × three markets × two cost profiles; 0050 buy and hold earns +46.2% a year, the overlays tie or win slightly with intervals spanning 0), the selected w = 0.7, L = 1.0 cannot be financed at a 60% loan-to-value, and the overlay improves the drawdown (−25% → −13% to −20%), not the absolute return; the conclusion stays "no deployable strategy beats 0050". **User decision (2026-10-26): beating 0050 is no longer a gate at any stage** — holding 0050 passively and quantitative trading are different tracks, this project stays on the quantitative track, and every verdict uses the industry bar of §13.19 item 9 (institution costs, P1 ∪ P2 net Sharpe ≥ 1.0, lower bound > 0, P2 > 0, capital scenarios alongside, supplement 4's execution layer), with 0050 as a reference column only. Data finding: the dataset's `adjusted_close` ignores splits and capital reductions (0050's 1:4 split is corrected through `core.splits`; the band position returns are unaffected). The chip-data line (`tw_chip_data`) finished phase 1 (2026-10-26, head f221e30): 16 free official datasets from 2012-01-02 to 2026-10-22, including `eligibility`, `sbl` and the two event tables; the data contract is that project's `docs/INTEGRATION.md`.

### Execution order

| # | Step | Basis | Prerequisite | Training | Output | Who |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.18 chip feature library `chip_tw_v1`** and B0's F1 / F2 | §13.18 items 1–4 | Curated layer delivered; multi-spec support in `stock-forecasting-build-features --spec` | **RunPod CPU**: one `features --spec` and one `lgbm --spec` each (a B0 walk-forward for F1 and for F2, three seeds) | Feature store, two sets of B0 predictions | This project writes the code; the user launches the Pods |
| 2 | **G1 and H-H**: paired F1 − F0 and F2 − F0, full universe and small / mid caps, by period, recent-winner sub-analysis (V1 score as the baseline) | §13.18 item 5 and the G1 sub-analysis | Step 1 | None | Protocol result, report | This project |
| 3 | **G2**: reversal step 0 on 處置 announcements, first 注意 listings and day-trade-ratio jumps | §13.18 item 6 | Event tables delivered | None | Event-study module, report | This project |
| 4 | Decision on buying 分點 and 集保 | §13.18 item 10 | Steps 2 and 3: 分點 (Sponsor Pro, one month) only if G1 or G2 passes; 集保 only when S-C enters implementation | — | Record of the decision | The user |
| 5 | (Only if F2 passes H-H with a clear residual spread) retrain D0 on the F2 features | §13.18 item 3 | Step 2 | **RunPod GPU**: `xs-train` (three seeds) | D0 F2 predictions | This project writes the code; the user launches the Pod |
| 6 | **Data pipeline**: add split and capital-reduction adjustments to the dataset's `adjusted_close` (109 stocks since 2023, 0050 and other ETFs), rebuild the dataset identity; P3's new dataset adopts it | §13.17 supplement result (h) | Corporate-action reference tables from the data source | **RunPod CPU** (data preparation) | New dataset, validation report | This project writes the code; the user launches the Pod |
| 7 | **P3 out of sample**: extend the dataset past 2026-06 (new dataset identity, with step 6's adjustments), retrain B0 on the new quarterly folds (F0, F2 if needed), rerun supplement 2 and step 2 with V1, no matching, fixed parameters and supplement 4's execution layer; the bar is §13.19 item 9 (no comparison with 0050) | §13.19 supplement 2 disposition, supplement 4 item (f), the 2026-10-26 bar decision | Several months of data after 2026-06; step 6 | **RunPod CPU** for data preparation and the new B0 folds; D0 only if it is a candidate (GPU) | The formal decision | The user downloads data and launches the Pods; this project scores |

### Training notes

Steps 2 and 3 are pure backtests on existing predictions, run locally. Steps 1, 5, 6 and 7 need Pods: step 1 is B0 LightGBM walk-forwards on the chip features (CPU Pod), step 5 is D0 (GPU Pod), step 6 is data preparation (CPU Pod), step 7 is the new-fold retraining that P3 requires (existing folds do not extend themselves when new data arrives). All training runs on RunPod, Pods are launched by the user with `--maxRuntime 24h`, after `sync_project_to_runpod_volume.sh --apply`.

### Not being done now

- Using "beats 0050 buy and hold" as a gate at any stage (decided 2026-10-26; the §13.17 supplement ends here and is not rerun on P3).
- Buying 分點 or 集保 (wait for step 4).
- Calling any strategy usable before P3; D0's results concentrate in 2023–2024 and decay, so D0 is not a candidate.
- Re-selecting the band parameters (fixed at the §13.19 selection).
- Pushing: branch `claude/chip-signals-20261025` and master's v0.3.0 / v0.4.0 tags wait for the user's review of the README and reports.
