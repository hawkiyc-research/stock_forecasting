# 下一步實驗佇列 / Next Experiments Queue

## 中文

本檔只列**尚未完成**的實驗與前置工作，依執行順序排列；每完成一項就從本檔移除，結果寫入 `docs/experiment_protocol.md` 對應章節與結果報告。歷史結論見 `MODEL_DEVELOPMENT_ROADMAP.md` 第 1 節與第 9 節。最後更新：2026-10-30（§13.23 全線重訓登記）。

### 現況

2026-10-29 發現 v1 還原價對有償現金增資套錯（§13.20 補充三）：同一組預測改用修正價重算，三個入選格的樣本外年化淨報酬從 +34.1%／+27.5%／+30.9% 降為 +14.7%／+19.4%／+13.7%；所有已出的模型分數仍帶著這個錯誤。2026-10-30 使用者決定全線重訓（§13.23）：v2 分支已合併、資料修訂可用 `--dataset-revision v1*`／`v2*` 切換（共用 provider 快取）、除權息因子修正已在 master；資料集用 **v1.1**（只套除權息因子）、停牌排除規則不變、P3 仍為樣本外；§13.22 的確認期在任何讀數出現前作廢。下面是 Pod 與本機的執行順序，每一步都要等前一步結束（features 與 lgbm Pod 在 volume 上互斥）。

### 執行順序

| # | 步驟 | 命令 | 誰做 |
| --- | --- | --- | --- |
| 1 | **同步程式到 volume**（volume 上沒有 Pod 時） | `bash scripts/runpod_workflow.sh sync --apply` | 使用者 |
| 2 | **新的選擇**（資料修訂 v1.1，結束日與 holdout 結束日 2026-10-23） | `bash scripts/runpod_workflow.sh configure --stage stage1 --data-profile tw_only --dataset-revision v1.1 --start 2016-01-01 --end 2026-10-23 --test-end 2026-10-23 --universe all` | 使用者 |
| 3 | **建立資料準備 Pod**，SSH 進去後**先清空 volume**（先看計畫，再刪；保留 api-cache、tw-chip-data 與各 launch 的結果檔），再啟動準備工作 | `bash scripts/runpod_workflow.sh cpu prepare --max-api-calls 1 --taiwan-qps 0.5 --maxRuntime 24h --prepareReserve 2h --maxBackoff 1m` → Pod 內：`cd /runpod-volume/stock_forecasting && python3 scripts/runpod_volume_reset.py` → 確認列表後 `cd /runpod-volume/stock_forecasting && python3 scripts/runpod_volume_reset.py --apply --confirm RESET` → `bash scripts/runpod_tmux_launch.sh cpu-prepare` | 使用者 |
| 4 | **特徵庫 (a)**：rank 標籤（B0 用） | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --spec configs/features/alpha_tw_v1.json` | 使用者 |
| 5 | **特徵庫 (b)**：變體 A（F0、D0、D1 用） | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_tw_v1.json` | 使用者 |
| 6 | **特徵庫 (c)**：變體 B | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1b.json --spec configs/features/alpha_tw_v1.json` | 使用者 |
| 7 | **特徵庫 (d)**：F2（價量＋籌碼） | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_chip_tw_v1.json --chipDataRoot /runpod-volume/datasets/tw-chip-data` | 使用者 |
| 8 | **B0**（rank 標籤） | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | 使用者 |
| 9 | **變體 A = F0** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | 使用者 |
| 10 | **變體 B** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1b.json --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | 使用者 |
| 11 | **F2** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_chip_tw_v1.json --chipDataRoot /runpod-volume/datasets/tw-chip-data --schedule configs/walk_forward/tw_quarterly_v2.json` | 使用者 |
| 12 | **D0**（GPU；配對基線自動取第 9 步的 launch） | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-scratch --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | 使用者 |
| 13 | **D1 Kronos-small**（GPU） | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-kronos-small --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | 使用者 |
| 14 | **D1 Chronos-2-small**（GPU） | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-chronos2-small --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | 使用者 |
| 15 | **下載**：每個 lgbm launch 一次 `download-lgbm`、每個 xs launch 一次 `download-research <launch-id>` | `bash scripts/runpod_workflow.sh download-lgbm`、`bash scripts/runpod_workflow.sh download-research launch-...` | 使用者 |
| 16 | **本機回測與判定**（§13.23 第 4、5 項）：模型層表、`scenario_suite`（F0、F2 全套件；B0、變體 B、D0、D1 對照）、門檻篩選、P3 驗證 H-M、F2 − F0 增量 | 本機 `backtest/.venv`，命令由本專案給 | 本專案 |
| 17 | **報告**：新的完整研究報告與入選策略報告（Markdown + Word），舊報告封存到 `reports/deprecated/` 並建索引 | — | 本專案 |
| 18 | 通過 H-M 的格進入確認期 P4'（§13.23 第 5 項；季度延伸需先建 `configs/walk_forward/tw_quarterly_v3.json`） | — | 本專案登記、使用者啟動 Pod |

### 注意

- 第 3 步的清空腳本只在使用者確認列表後才刪；`--apply` 沒有 `--confirm RESET` 不會執行。刪除後 `lifecycle/` 的舊指標會指向不存在的路徑，腳本會列出；第 2 步的新選擇與第 3 步的準備會重新發布。
- 第 4 到 11 步的 Pod 在 volume 上互斥，一次一個；第 12 到 14 步各一台 GPU Pod。所有 Pod 一律 `--maxRuntime 24h`。
- 沒有 Pod 會重抓 API：v1.1 與舊的 v1 共用同一個 provider 快取（快取修訂固定 `v1`）；`cpu prepare` 的 `--max-api-calls 1` 只限制 EODHD，台股 profile 不用它。

### 目前不做的事

- 消融 A1–A4 不重做（使用者未列）；`v2*`（減資、面額變更、ETF 分割）不在這條線使用，只保留切換能力。
- 以「勝過 0050 買進持有」作為任何階段的門檻；購買集保或分點資料（G1 通過但 2026-10-28 決定不買）。
- 推送分支：待使用者核准。

## English

This file lists only the experiments and prerequisites **not yet done**, in execution order; a finished item is removed and its result goes into the matching section of `docs/experiment_protocol.md` and the results report. Past conclusions are in `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9. Last update: 2026-10-30 (§13.23 full retrain registered).

### Where things stand

On 2026-10-29 v1's adjusted prices were found wrong for paid rights issues (§13.20 supplement 3): re-running the same predictions on corrected prices lowers the three shortlisted cells' out-of-sample annual net return from +34.1% / +27.5% / +30.9% to +14.7% / +19.4% / +13.7%, and every published model score still carries the error. On 2026-10-30 the user ordered a full retrain (§13.23): the v2 branch is merged, the data revision is selectable with `--dataset-revision v1*` / `v2*` (one shared provider cache), and the ex-rights fix is on master; the dataset is **v1.1** (ex-rights factors only), the halt-exclusion rule is unchanged, P3 stays out of sample, and §13.22's confirmation period is void before any reading. Below is the Pod and local order; every step waits for the previous one (features and lgbm Pods are exclusive on the volume).

### Execution order

| # | Step | Command | Who |
| --- | --- | --- | --- |
| 1 | **Sync the code to the volume** (no Pod on the volume) | `bash scripts/runpod_workflow.sh sync --apply` | the user |
| 2 | **New selection** (data revision v1.1, data and holdout end 2026-10-23) | `bash scripts/runpod_workflow.sh configure --stage stage1 --data-profile tw_only --dataset-revision v1.1 --start 2016-01-01 --end 2026-10-23 --test-end 2026-10-23 --universe all` | the user |
| 3 | **Create the preparation Pod**, SSH in, **clear the volume first** (plan, then delete; api-cache, tw-chip-data and every launch's result files stay), then start the preparation job | `bash scripts/runpod_workflow.sh cpu prepare --max-api-calls 1 --taiwan-qps 0.5 --maxRuntime 24h --prepareReserve 2h --maxBackoff 1m` → inside the Pod: `cd /runpod-volume/stock_forecasting && python3 scripts/runpod_volume_reset.py` → after checking the list `cd /runpod-volume/stock_forecasting && python3 scripts/runpod_volume_reset.py --apply --confirm RESET` → `bash scripts/runpod_tmux_launch.sh cpu-prepare` | the user |
| 4 | **Feature store (a)**: rank labels (for B0) | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --spec configs/features/alpha_tw_v1.json` | the user |
| 5 | **Feature store (b)**: variant A (F0, D0, D1) | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_tw_v1.json` | the user |
| 6 | **Feature store (c)**: variant B | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1b.json --spec configs/features/alpha_tw_v1.json` | the user |
| 7 | **Feature store (d)**: F2 (price-volume + chip) | `bash scripts/runpod_workflow.sh features --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_chip_tw_v1.json --chipDataRoot /runpod-volume/datasets/tw-chip-data` | the user |
| 8 | **B0** (rank labels) | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | the user |
| 9 | **Variant A = F0** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | the user |
| 10 | **Variant B** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1b.json --spec configs/features/alpha_tw_v1.json --schedule configs/walk_forward/tw_quarterly_v2.json` | the user |
| 11 | **F2** | `bash scripts/runpod_workflow.sh lgbm --maxRuntime 24h --labelVariant configs/labels/residual_v1a.json --spec configs/features/alpha_chip_tw_v1.json --chipDataRoot /runpod-volume/datasets/tw-chip-data --schedule configs/walk_forward/tw_quarterly_v2.json` | the user |
| 12 | **D0** (GPU; the paired baseline is step 9's launch, picked automatically) | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-scratch --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | the user |
| 13 | **D1 Kronos-small** (GPU) | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-kronos-small --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | the user |
| 14 | **D1 Chronos-2-small** (GPU) | `bash scripts/runpod_workflow.sh xs-train --experiment t-xs-chronos2-small --labelVariant configs/labels/residual_v1a.json --seed 42 --maxRuntime 24h` | the user |
| 15 | **Download**: one `download-lgbm` per lgbm launch, one `download-research <launch-id>` per xs launch | `bash scripts/runpod_workflow.sh download-lgbm`, `bash scripts/runpod_workflow.sh download-research launch-...` | the user |
| 16 | **Local backtests and verdicts** (§13.23 items 4 and 5): model-level tables, `scenario_suite` (full suite on F0 and F2; B0, variant B, D0, D1 as references), threshold shortlist, P3 verification H-M, F2 − F0 increment | local `backtest/.venv`, commands given by this project | this project |
| 17 | **Reports**: new full research report and selected-strategy report (Markdown + Word); old reports archived under `reports/deprecated/` with an index | — | this project |
| 18 | Cells passing H-M enter the confirmation period P4' (§13.23 item 5; the quarterly extension needs `configs/walk_forward/tw_quarterly_v3.json` first) | — | registered by this project, Pods launched by the user |

### Notes

- Step 3's reset script deletes only after the user has checked the list; `--apply` without `--confirm RESET` does nothing. Afterwards old `lifecycle/` markers point at deleted paths, which the script lists; step 2's selection and step 3's preparation republish them.
- The Pods of steps 4 to 11 are exclusive on the volume, one at a time; steps 12 to 14 are one GPU Pod each. Every Pod runs with `--maxRuntime 24h`.
- No Pod re-downloads the API: v1.1 shares the provider cache with the old v1 (cache revision fixed at `v1`); `--max-api-calls 1` on `cpu prepare` limits EODHD only, which the Taiwan profile does not use.

### Not being done now

- The ablations A1–A4 are not redone (not on the user's list); `v2*` (capital reductions, par-value changes, ETF splits) is not used on this line, only the switch is kept.
- "Beating 0050 buy-and-hold" as a bar at any stage; buying shareholder-dispersion or broker-branch data (G1 passed, but on 2026-10-28 the user decided not to buy).
- Pushing branches: awaits the user's approval.
