# 排名改版前的模型報告（a-signal64／signal-1.0 架構）

## 繁體中文

歸檔日期：2026-11-06。本群組是 2026-10-21 專案改為橫斷面排名任務（commit `cb6901e`「add cross-sectional rank task to the full baseline suite」）之前、最後一代架構的正式文件。當時的架構分別訓練 alpha 點估計與個股三分類兩套獨立模型（`a-signal64`／`signal-1.0`），評估的是預測訊號本身，沒有正式交易回測；改版後模型輸出變成橫斷面排序分數、評估改為 Rank IC 與策略層回測，這些文件的指標名稱（balanced accuracy、up precision、pinball 等）與目前的排名模型沒有對應關係。

| 檔案 | 內容 |
| --- | --- |
| `training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx` | seed 42 的模型效能分析報告（2026-10-16） |
| `training_result_analysis_run-20261015T064559Z-624f212a27894308.docx` | seed 347 的模型效能分析報告（2026-10-16） |
| `training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx` | seed 5067 的模型效能分析報告（2026-10-16） |
| `evaluation_metrics_reference_guide.docx` | 該架構的模型指標與參數說明手冊 |
| `analysis_report_template.docx` | 該架構的雙模型評估報告模板（以 seed 42 為完整範例） |

所有檔案保留原始位元組，只供版本追溯；不得把這些指標直接解讀為目前模型的表現，也不得作為策略績效的證明。目前的正式文件在上層 `reports/`（見 `reports/deprecated/README.md` 的「現行文件」）。

| 檔案 | SHA-256 |
| --- | --- |
| `analysis_report_template.docx` | 7d7c4a316487adf929f86d0b115ee29aa6d920e81a1a7de82143f58ace90df1e |
| `evaluation_metrics_reference_guide.docx` | bd2095cc70b05ee3a1e9713c6e25e789f1b4ae00516950b28455997f21787ca8 |
| `training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx` | 7c40cb42b7ea11ea9e62b259512754b2b0ee9db24be5f466ae06c65a2c4459c1 |
| `training_result_analysis_run-20261015T064559Z-624f212a27894308.docx` | 71f5de29b4b2637c3219c91fbc54f273c2e4fd5e3b7f92960ad2e57909bc75fc |
| `training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx` | b92bc31d205bdca7613a68441eb82d28c02972d6899072324474899e05dc5dea |

## English

Archived: 2026-11-06. This group holds the official documents of the last architecture before the project switched to the cross-sectional ranking task on 2026-10-21 (commit `cb6901e`, "add cross-sectional rank task to the full baseline suite"). That architecture trained two independent models, an alpha point estimate and a per-stock three-class classifier (`a-signal64` / `signal-1.0`), and evaluated the predictive signal only, with no formal trading backtest; after the switch the models output cross-sectional ranking scores and are evaluated by Rank IC and strategy-level backtests, so the metric names in these documents (balanced accuracy, up precision, pinball and so on) have no counterpart in the current ranking models.

| File | Contents |
| --- | --- |
| `training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx` | Seed 42 performance analysis (2026-10-16) |
| `training_result_analysis_run-20261015T064559Z-624f212a27894308.docx` | Seed 347 performance analysis (2026-10-16) |
| `training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx` | Seed 5067 performance analysis (2026-10-16) |
| `evaluation_metrics_reference_guide.docx` | That architecture's metrics and parameter handbook |
| `analysis_report_template.docx` | That architecture's two-model evaluation report template (seed 42 as the worked example) |

All files keep their original bytes and are kept for traceability only; their metrics must not be read as the current models' performance or as evidence of strategy results. The current documents are in the parent `reports/` (see "current documents" in `reports/deprecated/README.md`).

| File | SHA-256 |
| --- | --- |
| `analysis_report_template.docx` | 7d7c4a316487adf929f86d0b115ee29aa6d920e81a1a7de82143f58ace90df1e |
| `evaluation_metrics_reference_guide.docx` | bd2095cc70b05ee3a1e9713c6e25e789f1b4ae00516950b28455997f21787ca8 |
| `training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx` | 7c40cb42b7ea11ea9e62b259512754b2b0ee9db24be5f466ae06c65a2c4459c1 |
| `training_result_analysis_run-20261015T064559Z-624f212a27894308.docx` | 71f5de29b4b2637c3219c91fbc54f273c2e4fd5e3b7f92960ad2e57909bc75fc |
| `training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx` | b92bc31d205bdca7613a68441eb82d28c02972d6899072324474899e05dc5dea |
