# 舊版模型報告歸檔

## 繁體中文

歸檔日期：2026-10-09。此資料夾保留歷史報告、過時模板與本次修訂前的原版文件，僅供版本追溯，不作目前模型或正式策略績效的判斷依據。所有歸檔 DOCX 均保留搬移前的原始位元組，未重寫內容。

- 本層：7 份 2026-09-03 至 2026-09-23 的歷史訓練報告，以及舊版 `analysis_report_template.docx`。
- [pre-20261009](pre-20261009/)：A-LoRA32、A-LoRA64、A-partial 三份 2026-10-07 實驗報告在本次修訂前的原版。
- 最新模板：[analysis_report_template.docx](../analysis_report_template.docx)。模板以已完成的 A-LoRA64 holdout test 作為數值範例；新一輪報告必須替換全部身分、數值、來源與結論。

本次修訂後的正式報告位於上層 `reports/`：

- [A-LoRA32](../training_result_analysis_run-20261007T085635Z-bc59d5bbd2d84ad4.docx)
- [A-LoRA64](../training_result_analysis_run-20261007T085627Z-cc5f9ef13b484d90.docx)
- [A-partial](../training_result_analysis_run-20261007T085308Z-2debcc6e669b4522.docx)

修訂版保留完整 1～14 日結果，補齊 raw／calibrated、市場別校準、training 固定分布方向 baseline 與獨立 ranking head 的比較，並記錄資料來源及缺口。

重要限制：本次沒有修正評估程式或重算投資組合序列。2026-10-07 這三組既有的成本後報酬、Sharpe 與最大回撤存在持倉、報酬及成本資金尺度不一致，只能作舊版簡化診斷的追溯，不能支持可部署、穩定獲利或策略優劣的結論；也不能直接除以二當成修正值。完整說明請以修訂版報告為準。模型指標手冊未修改。

## English

Archive date: 2026-10-09. This directory retains historical reports, the obsolete template and the originals before this revision for version traceability only. They are not evidence of current model readiness or formal strategy performance. All archived DOCX files retain their original bytes; their contents were not rewritten.

- This level: seven historical training reports dated 2026-09-03 through 2026-09-23 and the old `analysis_report_template.docx`.
- [pre-20261009](pre-20261009/): the original A-LoRA32, A-LoRA64 and A-partial reports for the 2026-10-07 experiments before this revision.
- Current template: [analysis_report_template.docx](../analysis_report_template.docx). It uses the completed A-LoRA64 holdout test as a worked numerical example. Future reports must replace every identity, value, source and conclusion.

The revised current reports are in the parent `reports/` directory:

- [A-LoRA32](../training_result_analysis_run-20261007T085635Z-bc59d5bbd2d84ad4.docx)
- [A-LoRA64](../training_result_analysis_run-20261007T085627Z-cc5f9ef13b484d90.docx)
- [A-partial](../training_result_analysis_run-20261007T085308Z-2debcc6e669b4522.docx)

The revised reports retain complete day-1-to-day-14 results and add raw/calibrated distributions, market-specific calibration, direction versus the training-fixed distribution baseline, and independent ranking-head comparisons, with source provenance and evidence gaps.

Important limitation: this revision does not correct the evaluator or recompute portfolio sequences. The existing net returns, Sharpe ratios and maximum drawdowns for these three 2026-10-07 experiments use inconsistent capital scales for positions, returns and costs. They are retained only as legacy simplified diagnostics, not evidence of deployment readiness, stable profit or strategy superiority. Simply dividing old values by two is not a valid correction. Refer to the revised reports for the complete limitations. The metric handbook remains unchanged.
