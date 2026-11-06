# 舊版模型報告歸檔

## 繁體中文

索引更新日期：2026-11-07（首版 2026-10-16）。所有舊 DOCX 均保留原始位元組，只供版本追溯，不能拿不同架構的指標名稱直接解讀目前模型，也不能作為正式策略績效的證明。

| 版本群 | 保留內容 | DOCX 數 |
| --- | --- | ---: |
| [legacy-202609](legacy-202609/README.md) | 2026-09-03 至 2026-09-23 的 7 份歷史報告，以及更早的模板；2026-11-06 由本層根目錄移入子目錄，位元組未改 | 8 |
| [pre-20261009](pre-20261009/) | 2026-10-07 A-LoRA32、A-LoRA64、A-partial 在 10 月 9 日修訂前的原版；本次未移動或修改 | 3 |
| [pre-a-signal64-20261016](pre-a-signal64-20261016/README.md) | 此次改版前的 10 份正式報告、舊模板及舊手冊；依實驗輪次與文件用途再分 5 個子資料夾 | 12 |
| [pre-rank-pivot-20261021](pre-rank-pivot-20261021/README.md) | 2026-10-21 改為橫斷面排名任務之前的最後一代架構（`a-signal64`／`signal-1.0`）正式文件：seed 42／347／5067 三份效能報告、指標手冊、報告模板；2026-11-06 由上層 `reports/` 移入，位元組未改 | 5 |
| [pre-1323-20261102](pre-1323-20261102/) | 2026-10-29 的第一版完整研究報告與入選策略報告（Word 與 Markdown 各一份），2026-11-02 由 §13.23 全線重訓的第二版取代；第一版的模型分數帶有有償現金增資的還原價錯誤，其樣本外數字偏高、選定規則已修訂，只供追溯。SHA-256：`research_full_report_20261029.docx` 72573b44…135a、`.md` 79323882…d4f9；`selected_strategy_report_20261029.docx` 012d1030…18ee、`.md` b65ff4bf…8330 | 2 |
| [pre-1323s3-20261104](pre-1323s3-20261104/) | 2026-11-02 的第二版完整研究報告與入選策略報告在 11-04 補充執行層敏感度（§6.7／§7.5）之前的版本（Word 與 Markdown 各一份），11-04 由加註後的同名 Markdown 與 `*_20261104.docx` 取代；數字未變，只少了處置與停券規則的敏感度與三格的註記，只供追溯。SHA-256：`research_full_report_20261102.docx` d0ac8caf…5f4a、`.md` 96bf97ea…a1b3；`selected_strategy_report_20261102.docx` f1480d15…40be、`.md` d84afba8…6963 | 2 |
| [pre-1323s4-20261106](pre-1323s4-20261106/) | 2026-11-04 版的完整研究報告與入選策略報告在 11-06 補充成交價模型與追蹤停損（§6.8／§7.6）之前的版本（Word 與 Markdown 各一份），11-06 由加註後的同名 Markdown 與 `*_20261106.docx` 取代；數字未變，只少了成交價模型、停損與資本情境的並列讀數，只供追溯。SHA-256：`research_full_report_20261104.docx` b48f3214…b0f8、`.md` d1de26c8…bac5；`selected_strategy_report_20261104.docx` 78287d84…478d、`.md` 2acbc0df…e8fe | 2 |
| [pre-1323s5-20261106](pre-1323s5-20261106/) | 2026-11-06 的五次中間版：`a` = 含 §6.8／§7.6、不含方式 B；`b` = 含方式 B，成交價模型仍以「基準 vs 修正」呈現；`c` = 依裁定改寫，但 §6.4／§6.8／§7.3／§7.6 與方式 B 對照欄的表格因數字檔鍵值錯誤而全是「—」；`d` = 數字正確，但資本情境仍同時扣固定滑價與自身衝擊（補充四修正二之前）。；`e` = 滑價歸零後、稽核揭露（選樣前瞻、2 倍曝險、資本情境口徑）之前。同日最後由含稽核揭露的 `*_20261106.docx` 取代；只供追溯。SHA-256：research_full_report a 37d0fad8…04de／6224e7c6…0c1d、b e9f7b4b1…c89a／386f35c5…1ab6、c 545ffda4…c2ff／00aa1d78…b48a；selected_strategy_report a 7cb411d9…a1e3／29eeb80a…55b0、b e4f567fc…b32f／9b0a911d…0507、c 2dfe491a…88f3／31e810eb…795c、d 6cb39623…fd5b／c052a9ea…ae6f；selected d 0bf339e6…7fd2／c6cefa6b…712d、e 6986d87d…f544／9aed0d40…aeae；selected e d2956e6f…f214／85e623ec…a6a1（各 docx／md） | 10 |
| [pre-1323s6-20261107](pre-1323s6-20261107/) | 2026-11-06 含稽核揭露的第二版最終版完整研究報告與入選策略報告（Word 與 Markdown 各一份），2026-11-07 由第三版（§13.23 補充六：多空書每 1 元資金 1 元毛曝險、資本情境為兩市場總額）取代；第二版的多空格年化報酬與回撤是 2 倍曝險下的數字、資本列是每腿每市場各算一次，不能與第三版直接比較。SHA-256：research_full_report 0d7c3911…3f46／6b5d24e1…789f、selected_strategy_report 173285e4…e1bc／da08a7d3…0364（docx／md） | 2 |

新版本群與較早版本群分開保存，不合併、不覆蓋同名舊檔。此次搬移的完整清單及 SHA-256 見 [archive-manifest.json](pre-a-signal64-20261016/archive-manifest.json)。本索引更新前的原始內容亦已 [完整保存](pre-a-signal64-20261016/archive-index-before-20261016.md)。

現行文件位於上層 `reports/`，適用 2026-10-21 之後的橫斷面排名架構（§13.23 全線重訓之後的第三版）：

- [完整研究報告（第三版，2026-11-07）](../research_full_report_20261107.docx)，Markdown 原稿 `docs/research_full_report.md`
- [入選策略報告（第三版，2026-11-07）](../selected_strategy_report_20261107.docx)，Markdown 原稿 `docs/selected_strategy_report.md`
- [通過門檻各格的各期摘要](../passing_cells_summary_20261102.docx)，Markdown 原稿 `docs/passing_cells_summary.md`
- [台股排名管線成果報告（2026-10-25）](../taiwan_rank_pipeline_results_20261025.docx)、[六情境報告（2026-10-25）](../institution_cost_scenarios_20261025.docx)、[0050 核心加疊加結果（2026-10-26）](../core_overlay_results_20261026.docx)：排名架構早期（§13.14–§13.19）的分析文件，數字建立在 §13.23 重訓前的 v1 資料集，只作研究歷程參考。

目前的架構以橫斷面排序分數、Rank IC 與策略層回測評估，不沿用舊架構的分類指標解讀；每一輪新報告都必須重建全部身分、數值、來源及結論。

歷史限制仍須保留：2026-10-07 三組舊評估的成本後報酬、Sharpe 與最大回撤存在持倉／報酬／成本尺度不一致，不能當作可靠策略績效，也不能直接除以二修正。請查閱本次歸檔的已修訂報告。新架構目前只評估預測訊號，未執行正式交易回測；兩者不可混用。

## English

Index updated: 2026-11-07 (first version 2026-10-16). All historical DOCX files retain their original bytes and are kept for traceability. Metric names from different architectures must not be carried into current-model interpretation or treated as proof of formal strategy performance.

| Version group | Retained contents | DOCX count |
| --- | --- | ---: |
| [legacy-202609](legacy-202609/README.md) | Seven historical reports dated 2026-09-03 through 2026-09-23 and the earlier template; moved from this level into the subdirectory on 2026-11-06, bytes unchanged | 8 |
| [pre-20261009](pre-20261009/) | Original 2026-10-07 A-LoRA32, A-LoRA64 and A-partial reports before the October 9 revision, unchanged here | 3 |
| [pre-a-signal64-20261016](pre-a-signal64-20261016/README.md) | Ten previously current reports, the previous template and handbook, further divided into five experiment/purpose subdirectories | 12 |
| [pre-rank-pivot-20261021](pre-rank-pivot-20261021/README.md) | The last architecture's official documents before the switch to the cross-sectional ranking task on 2026-10-21 (`a-signal64` / `signal-1.0`): the seed 42 / 347 / 5067 performance reports, the metrics handbook and the report template; moved from the parent `reports/` on 2026-11-06, bytes unchanged | 5 |
| [pre-1323-20261102](pre-1323-20261102/) | The first-edition full research report and selected-strategy report of 2026-10-29 (Word and Markdown each), superseded on 2026-11-02 by the second edition after the §13.23 full retrain; the first edition's model scores carry the paid-rights adjusted-price defect, its out-of-sample numbers are overstated and its selection rule has since been amended, so it is kept for traceability only. SHA-256: `research_full_report_20261029.docx` 72573b44…135a, `.md` 79323882…d4f9; `selected_strategy_report_20261029.docx` 012d1030…18ee, `.md` b65ff4bf…8330 | 2 |
| [pre-1323s3-20261104](pre-1323s3-20261104/) | The second-edition full research report and selected-strategy report of 2026-11-02 as they stood before the execution-layer sensitivity (§6.7 / §7.5) was added on 11-04 (Word and Markdown each), superseded on 11-04 by the annotated Markdown of the same name and `*_20261104.docx`; the numbers are unchanged, only the disposition / suspension sensitivity and the notes on three cells are missing, so they are kept for traceability only. SHA-256: `research_full_report_20261102.docx` d0ac8caf…5f4a, `.md` 96bf97ea…a1b3; `selected_strategy_report_20261102.docx` f1480d15…40be, `.md` d84afba8…6963 | 2 |
| [pre-1323s4-20261106](pre-1323s4-20261106/) | The 2026-11-04 full research report and selected-strategy report as they stood before the fill model and trailing stop (§6.8 / §7.6) were added on 11-06 (Word and Markdown each), superseded on 11-06 by the annotated Markdown of the same name and `*_20261106.docx`; the numbers are unchanged, only the fill-model, stop and capital-scenario parallel readings are missing, so they are kept for traceability only. SHA-256: `research_full_report_20261104.docx` b48f3214…b0f8, `.md` d1de26c8…bac5; `selected_strategy_report_20261104.docx` 78287d84…478d, `.md` 2acbc0df…e8fe | 2 |
| [pre-1323s5-20261106](pre-1323s5-20261106/) | The five intermediate 2026-11-06 builds: `a` = with §6.8 / §7.6, without policy B; `b` = with policy B, the fill model still shown as "baseline vs corrected"; `c` = rewritten per the ruling but with the tables of §6.4 / §6.8 / §7.3 / §7.6 and the policy-B comparison column all "—" because of a key defect in the numbers file. `d` = correct numbers but the capital rows still paying both the flat slippage and own impact (before supplement 4 correction 2). `e` = after the slippage removal but before the audit disclosures (sample-selection look-ahead, 2× exposure, capital-scenario convention). Superseded the same day by `*_20261106.docx` with the audit disclosures; kept for traceability only. SHA-256: research_full_report a 37d0fad8…04de / 6224e7c6…0c1d, b e9f7b4b1…c89a / 386f35c5…1ab6, c 545ffda4…c2ff / 00aa1d78…b48a; selected_strategy_report a 7cb411d9…a1e3 / 29eeb80a…55b0, b e4f567fc…b32f / 9b0a911d…0507, c 2dfe491a…88f3 / 31e810eb…795c , d 6cb39623…fd5b / c052a9ea…ae6f; selected d 0bf339e6…7fd2 / c6cefa6b…712d , e 6986d87d…f544 / 9aed0d40…aeae; selected e d2956e6f…f214 / 85e623ec…a6a1 (docx / md each) | 10 |
| [pre-1323s6-20261107](pre-1323s6-20261107/) | The final second-edition full research report and selected-strategy report of 2026-11-06 with the audit disclosures (Word and Markdown each), superseded on 2026-11-07 by the third edition (§13.23 supplement 6: NT$1 of gross exposure per NT$1 of capital in the long-short book, capital scenarios as totals across both markets); the second edition's long-short returns and drawdowns are 2×-exposure figures and its capital rows count each leg and market separately, so they are not comparable with the third edition. SHA-256: research_full_report 0d7c3911…3f46 / 6b5d24e1…789f, selected_strategy_report 173285e4…e1bc / da08a7d3…0364 (docx / md) | 2 |

The new group remains separate from older groups. No groups were merged and no older same-named file was overwritten. See [archive-manifest.json](pre-a-signal64-20261016/archive-manifest.json) for exact paths and SHA-256 hashes. The original text of this index has also been [preserved in full](pre-a-signal64-20261016/archive-index-before-20261016.md).

The current documents are in the parent `reports/` and belong to the cross-sectional ranking architecture adopted on 2026-10-21 (third edition after the §13.23 full retrain):

- [Full research report (third edition, 2026-11-07)](../research_full_report_20261107.docx), Markdown source `docs/research_full_report.md`
- [Selected-strategy report (third edition, 2026-11-07)](../selected_strategy_report_20261107.docx), Markdown source `docs/selected_strategy_report.md`
- [Per-period summary of the cells that pass the thresholds](../passing_cells_summary_20261102.docx), Markdown source `docs/passing_cells_summary.md`
- [Taiwan rank pipeline results (2026-10-25)](../taiwan_rank_pipeline_results_20261025.docx), [six-scenario report (2026-10-25)](../institution_cost_scenarios_20261025.docx), [0050 core plus overlay results (2026-10-26)](../core_overlay_results_20261026.docx): analyses from the ranking architecture's early rounds (§13.14–§13.19), whose numbers rest on the v1 dataset before the §13.23 retrain; research-history reference only.

The current architecture is evaluated by cross-sectional ranking scores, Rank IC and strategy-level backtests; the old architecture's classification-metric interpretations do not carry over, and every new report must rebuild all identities, values, sources and conclusions.

Historical limits remain important: the legacy 2026-10-07 net returns, Sharpe and maximum drawdown use inconsistent position/return/cost scales. They are not reliable strategy results and cannot be corrected by simply dividing by two. Consult the revised reports archived in the new group. The current architecture evaluates predictive signals, not a formal trading backtest; these forms of evidence are not interchangeable.
