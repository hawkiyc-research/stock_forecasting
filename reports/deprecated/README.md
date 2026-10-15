# 舊版模型報告歸檔

## 繁體中文

索引更新日期：2026-10-16。所有舊 DOCX 均保留原始位元組，只供版本追溯，不能拿不同架構的指標名稱直接解讀目前模型，也不能作為正式策略績效的證明。

| 版本群 | 保留內容 | DOCX 數 |
| --- | --- | ---: |
| 本層較早文件 | 2026-09-03 至 2026-09-23 的 7 份歷史報告，以及更早的模板；本次未移動或修改 | 8 |
| [pre-20261009](pre-20261009/) | 2026-10-07 A-LoRA32、A-LoRA64、A-partial 在 10 月 9 日修訂前的原版；本次未移動或修改 | 3 |
| [pre-a-signal64-20261016](pre-a-signal64-20261016/README.md) | 此次改版前的 10 份正式報告、舊模板及舊手冊；依實驗輪次與文件用途再分 5 個子資料夾 | 12 |

新版本群與較早版本群分開保存，不合併、不覆蓋同名舊檔。此次搬移的完整清單及 SHA-256 見 [archive-manifest.json](pre-a-signal64-20261016/archive-manifest.json)。本索引更新前的原始內容亦已 [完整保存](pre-a-signal64-20261016/archive-index-before-20261016.md)。

現行文件位於上層 `reports/`，適用 `a-signal64`／`signal-1.0`：

- [Seed 42 模型效能分析報告](../training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx)
- [Seed 347 模型效能分析報告](../training_result_analysis_run-20261015T064559Z-624f212a27894308.docx)
- [模型指標與參數說明手冊](../evaluation_metrics_reference_guide.docx)
- [雙模型評估報告模板](../analysis_report_template.docx)

現行架構分別訓練 alpha 點估計與個股三分類兩套獨立模型；不沿用舊分位數、區間校準或獨立 ranking head 的解讀。模板以 seed 42 的已完成 test 為完整範例，另列 seed 347 對照；新一輪必須重建全部身分、數值、來源及結論。

歷史限制仍須保留：2026-10-07 三組舊評估的成本後報酬、Sharpe 與最大回撤存在持倉／報酬／成本尺度不一致，不能當作可靠策略績效，也不能直接除以二修正。請查閱本次歸檔的已修訂報告。新架構目前只評估預測訊號，未執行正式交易回測；兩者不可混用。

## English

Index updated: 2026-10-16. All historical DOCX files retain their original bytes and are kept for traceability. Metric names from different architectures must not be carried into current-model interpretation or treated as proof of formal strategy performance.

| Version group | Retained contents | DOCX count |
| --- | --- | ---: |
| Earlier files at this level | Seven historical reports dated 2026-09-03 through 2026-09-23 and the earlier template, neither moved nor modified in this update | 8 |
| [pre-20261009](pre-20261009/) | Original 2026-10-07 A-LoRA32, A-LoRA64 and A-partial reports before the October 9 revision, unchanged here | 3 |
| [pre-a-signal64-20261016](pre-a-signal64-20261016/README.md) | Ten previously current reports, the previous template and handbook, further divided into five experiment/purpose subdirectories | 12 |

The new group remains separate from older groups. No groups were merged and no older same-named file was overwritten. See [archive-manifest.json](pre-a-signal64-20261016/archive-manifest.json) for exact paths and SHA-256 hashes. The original text of this index has also been [preserved in full](pre-a-signal64-20261016/archive-index-before-20261016.md).

Current `a-signal64` / `signal-1.0` documents are in the parent `reports/` directory:

- [Seed 42 performance analysis](../training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx)
- [Seed 347 performance analysis](../training_result_analysis_run-20261015T064559Z-624f212a27894308.docx)
- [Model metrics and parameter handbook](../evaluation_metrics_reference_guide.docx)
- [Independent-model evaluation report template](../analysis_report_template.docx)

The current architecture independently trains alpha point estimation and stock three-class classification. Legacy quantile, interval-calibration and independent ranking-head interpretations do not apply. The template contains the completed seed-42 test as a full worked example and an identified seed-347 comparison. Every new report must rebuild all identities, values, sources and conclusions.

Historical limits remain important: the legacy 2026-10-07 net returns, Sharpe and maximum drawdown use inconsistent position/return/cost scales. They are not reliable strategy results and cannot be corrected by simply dividing by two. Consult the revised reports archived in the new group. The current architecture evaluates predictive signals, not a formal trading backtest; these forms of evidence are not interchangeable.
