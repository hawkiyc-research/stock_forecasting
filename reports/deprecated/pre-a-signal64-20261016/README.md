# a-signal64 改版前文件歸檔

## 繁體中文

歸檔日期：2026-10-16。本版本群保留改用 `signal-1.0`、兩套獨立 `alpha_point`／`absolute_direction` 模型前，原位於 `reports/` 的 10 份分析報告、1 份模板及 1 份指標手冊。12 份 DOCX 都是原檔搬移，檔名與原始位元組不變；沒有重寫舊數據或刪除文件。

本版本群與更早的 `deprecated/` 本層及 `pre-20261009/` 分開；更早歸檔的 11 份 DOCX 沒有移動或修改。本群內再依實驗輪次與文件用途分類：

| 子資料夾 | 內容 | DOCX 數 |
| --- | --- | ---: |
| [20261007-a-group](20261007-a-group/) | A-LoRA32、A-LoRA64、A-partial 的已修訂報告；不是 `pre-20261009/` 的修訂前原版 | 3 |
| [20261008-adaptive64](20261008-adaptive64/) | A-adaptive64，seed 42、43、44 | 3 |
| [20261010-direction64](20261010-direction64/) | A-direction64，seed 42、43、44 | 3 |
| [20261011-direction64-v2](20261011-direction64-v2/) | A-direction64-v2，seed 42 | 1 |
| [templates-and-guide](templates-and-guide/) | 舊 A-LoRA64 數值範例模板及舊 schema 5.0 指標手冊 | 2 |

完整檔名、搬移前路徑、搬移後路徑、檔案大小與 SHA-256 見 [archive-manifest.json](archive-manifest.json)。搬移前後逐檔核對雜湊相同。更新前的歸檔總索引另以原始位元組保存為 [archive-index-before-20261016.md](archive-index-before-20261016.md)；該索引中的相對連結保留當時原文，請用本索引或 manifest 尋找目前位置。

這些文件只供歷史追溯。舊分位數、區間校準、共享多頭或方向 V1/V2 等架構不能用來解釋目前獨立雙模型的輸出。舊版成本後報酬、Sharpe 與最大回撤亦不可視為本輪策略績效。現行兩份 seed 42／347 分析報告、指標手冊與模板位於上兩層 [reports/](../../)，以 `signal-1.0` 的最終 holdout test 為準。

## English

Archive date: 2026-10-16. This version group preserves the ten analysis reports, one template and one metric handbook previously located directly in `reports/`, before the transition to `signal-1.0` and two independent `alpha_point` / `absolute_direction` models. All twelve DOCX files were moved without rewriting their contents. Original filenames and bytes are retained; no historical data or document was deleted.

This group is separate from the older files directly under `deprecated/` and the `pre-20261009/` group. The eleven earlier archived DOCX files were neither moved nor modified. This group is further organized by experiment round and document purpose:

| Subdirectory | Contents | DOCX count |
| --- | --- | ---: |
| [20261007-a-group](20261007-a-group/) | Revised A-LoRA32, A-LoRA64 and A-partial reports, distinct from the pre-revision originals in `pre-20261009/` | 3 |
| [20261008-adaptive64](20261008-adaptive64/) | A-adaptive64, seeds 42, 43 and 44 | 3 |
| [20261010-direction64](20261010-direction64/) | A-direction64, seeds 42, 43 and 44 | 3 |
| [20261011-direction64-v2](20261011-direction64-v2/) | A-direction64-v2, seed 42 | 1 |
| [templates-and-guide](templates-and-guide/) | Previous A-LoRA64 worked-example template and schema-5.0 metric handbook | 2 |

See [archive-manifest.json](archive-manifest.json) for every filename, original and archived path, byte size and SHA-256. Hashes were verified before and after each move. The original archive index was separately preserved byte-for-byte as [archive-index-before-20261016.md](archive-index-before-20261016.md). Its relative links retain the historical text; use this index or the manifest for current locations.

These files are historical evidence only. Legacy quantile, interval-calibration, shared-head and direction-V1/V2 architectures must not define the outputs of the current independent models. Legacy net returns, Sharpe and maximum drawdown are not strategy results for this round. The current seed-42/347 reports, handbook and template are two levels above in [reports/](../../), based on the final `signal-1.0` holdout test.
