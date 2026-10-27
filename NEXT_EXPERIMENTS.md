# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-28（§13.20 登記後）。

### 現況

正交化後（分數對 60 日 beta 與前 20／60 日報酬每日取殘差，V1）的現股多日多空 S1／S2，在機構成本下達到 §13.19 的資訊性門檻；2026-10-26 補充四把空頭腿的「全部可借券」換成 `tw_chip_data` 的可借券名單、交易所次日限額、依規模分層的借不到機率、召回與分層借券費（五個隨機重複平均）之後，B0 TW 的 S1 仍 +15.3% [+2.5%, +30.3%]（Sharpe 1.24）、三市場 P2 全正，九格達機構門檻（原十格），事前判讀「補充二的結論在可得性修正後成立」；名單本身幾乎不改變結果，分層費與抽樣各吃 1 到 3 個百分點，借不到倍率 0 到 2 的差異在五個重複的極差之內。設計仍在看過 P2 之後決定、絕對報酬低於同期 0050，正式判定留給 P3。§13.17 補充（0050 核心加疊加，2026-10-26）：變體 A（質押借款）與 B（台指期提供 β）在兩個模型、三個市場、兩個成本檔共 24 格的 H-G 全部未通過（0050 買進持有年化 +46.2%，疊加打平或小贏、區間跨 0），選到的 w = 0.7、L = 1.0 在 60% 質押成數下不可融資，疊加改善的是回撤（−25% → −13% 到 −20%）不是絕對報酬；結論維持「沒有勝過 0050 的可部署策略」。**使用者決定（2026-10-26）：勝過 0050 不再是任何階段的門檻**——被動持有 0050 與量化交易是不同的賽道，本專案回到量化交易的賽道，判定一律用 §13.19 第 9 項的業界門檻（機構成本、P1 ∪ P2 淨 Sharpe ≥ 1.0、區間下界 > 0、P2 > 0，資本情境並列，補充四執行層），0050 只作對照。資料發現：資料集 `adjusted_close` 不含分割與減資（0050 一拆四已在 `core.splits` 修正；band 部位報酬未受影響）。籌碼資料線（`tw_chip_data`）第一階段已完成（2026-10-26，head f221e30）：16 個免費官方資料集 2012-01-02 至 2026-10-22，含 `eligibility`、`sbl` 與兩張事件表；資料契約在該專案 `docs/INTEGRATION.md`。**§13.18 F1／F2 已完成（2026-10-28，`docs/chip_increment_results.md`）：H-H 通過**（F2 − F0 全宇宙持有格殘差 ΔIC +0.017 [+0.013, +0.021]；流動性宇宙 h = 10 殘差四分位價差 0.65% → 1.04%，1.59 倍，餘裕小），**G1 通過**（流動性宇宙與中小型股的 ΔIC 下界都 > 0），分點購買的第一道門已開；籌碼單獨（F1）P1 ∪ P2 與價量打平、P2 勝出（ΔIC +0.025）；增量在每一組都有、在近期強勢股更大，是獨立訊號加對強勢股更敏感，不是只在 P2 的條件化反轉；幅度仍小（流動性宇宙產業中性多空毛年化 +8.4% → +12.9%），可不可交易由策略層決定。

### 執行順序

| # | 步驟 | 依據 | 前置條件 | 訓練需求 | 產出 | 誰做 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.20 回測通道的 Pod**：新選擇（`--end 2026-10-23 --test-end 2026-10-23`，修訂 v1）→ `cpu prepare` → `features` F0、F2 → `lgbm` F0、F2（`--schedule tw_quarterly_v2`，16 折，含 P3） | §13.20 第 1、2 項；README「回測通道」節 | 程式已完成（2026-10-28） | **RunPod CPU** 五個 Pod | P3 的 F0／F2 預測 | 使用者啟動 Pod |
| 2 | **§13.20 策略層與報告**：S-A 模組（週頻多頭 band，K、m 於 P1 選定）、§13.19 設定檔加入 P3、S1／S2／S-A 在 F0 與 F2 上的 P1／P2／P3 回測（兩成本檔、補充四執行層）、選定規則與 H-J、`docs/research_full_report.md` 與 `docs/selected_strategy_report.md` | §13.20 第 3 到 9 項 | 第 1 項 | 無（本機回測） | 兩份報告、協定結果 | 本專案 |
| 3 | **G2**：處置、注意首次公告、當沖比跳升三類事件的反轉步驟 0（事件研究模組） | §13.18 第 6 項 | 事件表已交付 | 無 | 事件研究模組、報告；決定 S-B 要不要實作 | 本專案 |
| 4 | （使用者決定，暫不執行）分點購買：G1 已通過，但 2026-10-28 決定先不買付費資料 | §13.18 第 10 項 | — | — | 購買與否的紀錄 | 使用者決定 |
| 5 | （使用者決定）資料集 v2（分割與減資調整）重訓：看完 §13.20 的報告與 v1 缺陷聲明後決定；程式已完成（分支 `claude/dataset-v2-corporate-actions-20261026`，需先 rebase 到 master）；v2 的 provider 快取修訂與資料集修訂綁在一起，整批重抓 API 之前要先拆開 | §13.17 補充結果 (h)、§13.20 第 9 項 | 第 2 項的報告 | **RunPod CPU**：prepare → features F0、F2 → lgbm F0、F2 | v2 資料集與 v1 對 v2 比較 | 使用者決定；本專案計分 |
| 6 | （使用者決定）D0 在 F2 特徵上重訓 | §13.18 第 3 項 | 使用者決定 | **RunPod GPU**：`xs-train` | D0 F2 預測 | 本專案寫程式；使用者啟動 Pod |

### 訓練需求說明

第 2、3 項是純回測，在本機跑。第 1 項是 P3 必須的資料準備、特徵庫與 B0 新折重訓（五個 CPU Pod；新資料進來後舊折的預測不會自動延伸，§13.20 的回測通道就是為此而設）；第 5 項（v2）與第 6 項（D0）只在使用者決定後才開 Pod。所有訓練都在 RunPod，Pod 由使用者啟動，每次 `--maxRuntime 24h`，啟動前先 `sync_project_to_runpod_volume.sh --apply`。

### 目前不做的事

- 以「勝過 0050 買進持有」作為任何階段的門檻（2026-10-26 決定；§13.17 補充到此結束，不在 P3 重算）。
- 購買集保（只在 S-C 進入實作時）；分點由第 2 項決定。
- 在 P3 之前宣稱任何策略可用；D0 的結果集中在 2023 到 2024 且逐年遞減，不列為候選。
- 重選 band 參數（固定為 §13.19 的選定值）。
- 推送分支：`claude/chip-signals-20261025` 與 master 的 v0.3.0、v0.4.0 標籤待使用者看過 README 與報告後決定。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-28 (after the §13.20 registration).

### Where things stand

After orthogonalizing the score daily against the 60-session beta and the trailing 20 / 60-session returns (V1), the multi-day stock long-short S1 / S2 meets the informational bar of §13.19 under institution costs. On 2026-10-26 supplement 4 replaced the short book's "every name is borrowable" assumption with the `tw_chip_data` borrowable lists, the exchange's next-session quota, borrow-failure probabilities by size tercile, recalls and tiered borrow fees (five random replicates averaged): B0's S1 on TW stays at +15.3% [+2.5%, +30.3%] (Sharpe 1.24) with every P2 positive on the three markets, nine cells clear the institution bar (ten before), and the pre-registered reading "supplement 2's conclusion survives the availability correction" holds; the list itself barely moves the result, tiered fees and the failure draws each cost 1 to 3 points, and failure multipliers 0 to 2 differ by less than the range across replicates. The design was still chosen after seeing P2 and the absolute return trails 0050 over the same period, so the formal decision waits for P3. §13.17 supplement (0050 core plus overlay, 2026-10-26): variants A (secured loan) and B (β from TX) fail H-G in all 24 cells (two models × three markets × two cost profiles; 0050 buy and hold earns +46.2% a year, the overlays tie or win slightly with intervals spanning 0), the selected w = 0.7, L = 1.0 cannot be financed at a 60% loan-to-value, and the overlay improves the drawdown (−25% → −13% to −20%), not the absolute return; the conclusion stays "no deployable strategy beats 0050". **User decision (2026-10-26): beating 0050 is no longer a gate at any stage** — holding 0050 passively and quantitative trading are different tracks, this project stays on the quantitative track, and every verdict uses the industry bar of §13.19 item 9 (institution costs, P1 ∪ P2 net Sharpe ≥ 1.0, lower bound > 0, P2 > 0, capital scenarios alongside, supplement 4's execution layer), with 0050 as a reference column only. Data finding: the dataset's `adjusted_close` ignores splits and capital reductions (0050's 1:4 split is corrected through `core.splits`; the band position returns are unaffected). The chip-data line (`tw_chip_data`) finished phase 1 (2026-10-26, head f221e30): 16 free official datasets from 2012-01-02 to 2026-10-22, including `eligibility`, `sbl` and the two event tables; the data contract is that project's `docs/INTEGRATION.md`. **§13.18 F1 / F2 are done (2026-10-28, `docs/chip_increment_results.md`): H-H passes** (F2 − F0 full-universe holding-cell residual ΔIC +0.017 [+0.013, +0.021]; liquidity-universe h = 10 residual quartile spread 0.65% → 1.04%, 1.59×, a thin margin) and **G1 passes** (ΔIC lower bounds above 0 in both the liquidity universe and the small / mid caps), so the first gate for buying broker-branch data is open; chip data alone (F1) ties price / volume on P1 ∪ P2 and wins in P2 (ΔIC +0.025); the increment shows in every segment and is larger among recent winners — an independent signal plus sharper discrimination among winners, not a P2-only conditional reversal; the magnitude stays small (liquidity-universe industry-neutral long-short gross annual +8.4% → +12.9%), and tradeability is for the strategy layer to decide.

### Execution order

| # | Step | Basis | Prerequisite | Training | Output | Who |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **§13.20 backtest-channel Pods**: a new selection (`--end 2026-10-23 --test-end 2026-10-23`, revision v1) → `cpu prepare` → `features` F0, F2 → `lgbm` F0, F2 (`--schedule tw_quarterly_v2`, 16 folds, with P3) | §13.20 items 1–2; README section "backtest channel" | Code done (2026-10-28) | **RunPod CPU**, five Pods | P3 predictions of F0 / F2 | The user launches the Pods |
| 2 | **§13.20 strategy layer and reports**: the S-A module (weekly long band, K and m selected on P1), P3 in the §13.19 configs, S1 / S2 / S-A on F0 and F2 over P1 / P2 / P3 (both cost profiles, supplement 4's execution layer), the selection rule and H-J, `docs/research_full_report.md` and `docs/selected_strategy_report.md` | §13.20 items 3–9 | Step 1 | None (local backtests) | The two reports, protocol results | This project |
| 3 | **G2**: reversal step 0 on 處置 announcements, first 注意 listings and day-trade-ratio jumps (event-study module) | §13.18 item 6 | Event tables delivered | None | Event-study module, report; decides whether S-B is built | This project |
| 4 | (User decision, on hold) broker-branch purchase: G1 passed, but on 2026-10-28 the user decided not to buy paid data for now | §13.18 item 10 | — | — | Record of the decision | The user |
| 5 | (User decision) dataset v2 (split and capital-reduction adjustments) retraining: decided after the §13.20 reports and the v1 defect statement; code done (branch `claude/dataset-v2-corporate-actions-20261026`, to be rebased onto master); v2 ties the provider-cache revision to the dataset revision, so decouple them before a full API re-download | §13.17 supplement result (h), §13.20 item 9 | The step 2 reports | **RunPod CPU**: prepare → features F0, F2 → lgbm F0, F2 | The v2 dataset and the v1-vs-v2 comparison | The user decides; this project scores |
| 6 | (User decision) retrain D0 on the F2 features | §13.18 item 3 | User decision | **RunPod GPU**: `xs-train` | D0 F2 predictions | This project writes the code; the user launches the Pod |

### Training notes

Steps 2 and 3 are pure backtests, run locally. Step 1 is the data preparation, feature stores and B0 new-fold retraining that P3 requires (five CPU Pods; existing folds do not extend themselves when new data arrives, which is what the §13.20 backtest channel is for); steps 5 (v2) and 6 (D0) get Pods only after the user decides. All training runs on RunPod, Pods are launched by the user with `--maxRuntime 24h`, after `sync_project_to_runpod_volume.sh --apply`.

### Not being done now

- Using "beats 0050 buy and hold" as a gate at any stage (decided 2026-10-26; the §13.17 supplement ends here and is not rerun on P3).
- Buying 集保 (only when S-C enters implementation); 分點 is decided in step 2.
- Calling any strategy usable before P3; D0's results concentrate in 2023–2024 and decay, so D0 is not a candidate.
- Re-selecting the band parameters (fixed at the §13.19 selection).
- Pushing: branch `claude/chip-signals-20261025` and master's v0.3.0 / v0.4.0 tags wait for the user's review of the README and reports.
