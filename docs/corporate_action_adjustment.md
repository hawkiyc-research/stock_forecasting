# 台股分割、減資與面額變更的還原價調整 / Split, capital-reduction and par-value adjustment of Taiwan prices

## 中文

### 1. 問題

資料集 `artifacts/datasets/<id>/raw/market.parquet` 的 `adjusted_close` 由官方除權息表
（TWSE `TWT49U`、TPEx `exDailyQ`）累乘得到，只含現金股利與無償配股。股票分割、減資（現金減資、
彌補虧損減資）與變更面額在台灣都是「停止買賣數個交易日，之後以參考價恢復買賣」，不在除權息表內，
因此 2026-10-26 的 §13.17 補充發現 0050 在 2025-06-18 一拆四沒有調整（停牌五個交易日後價格變成四分之一），
2023 年起全檔有 109 檔個股出現超過 25% 的單日還原開盤跳動。回測的 band 引擎只在同一檔相鄰訊號日之間
計算開盤到開盤報酬、停牌即出場，所以 §13.14 到 §13.19 的部位報酬未受影響；但訓練標籤（h 日收盤報酬）
與特徵（前 20／60 日報酬等）會含這些跳動。

### 2. 資料來源

| 事件 | 來源 | 欄位 | 2016–2026 筆數 |
| --- | --- | --- | --- |
| TWSE 減資恢復買賣（退還股款、彌補虧損） | `https://www.twse.com.tw/rwd/zh/reducation/TWTAUU` | 恢復買賣日期、股票代號、停止買賣前收盤價格、恢復買賣參考價、減資原因 | 290（另 2 筆尚未公告價格） |
| TWSE 變更股票面額 | `https://www.twse.com.tw/rwd/zh/change/TWTB8U` | 同上 | 10 |
| TWSE ETF 分割／反分割 | `https://www.twse.com.tw/rwd/zh/split/TWTCAU` | 恢復買賣日期、ETF代號、停止買賣前收盤價格、恢復買賣參考價 | 2（0050、0052；9 檔槓桿／反向 ETF 不在白名單） |
| TPEx 減資恢復買賣 | `https://www.tpex.org.tw/www/zh-tw/bulletin/revivt` | 恢復買賣日期、股票代號、最後交易日之收盤價格、減資恢復買賣開始日參考價格、詳細資料（含每壹仟股換發新股票、每股退還股款） | 229 |
| TPEx 變更股票面額 | 櫃買中心沒有可查詢的表，只有個別公司公告；以行情跳動偵測（停牌 5–7 日後的第一個交易日）加上櫃買公司基本資料的現行面額（`configs/reference/tw_company_reference_20261022.csv`）整理成 `taiwan.py` 內的 `_TPEX_PAR_VALUE_CHANGES` | 12（10 檔） |

四個官方端點都以一次請求回傳整段日期範圍；每個 provider 的 `fetch_actions` 在除權息表之後呼叫
`fetch_resumption_actions`，把這些列併入同一張公司行動表。端點、解析函式與面額表都列入資料內容識別碼。

### 3. 調整方法

- **事件日**：恢復買賣日（新股第一個交易日）。`apply_cumulative_adjustments` 把該日之前所有列的
  `adjusted_close` 乘上 `price_factor`，與既有股息因子相乘；同日多個事件（例如分割加除息）相乘成一個因子，
  `source` 以 `+` 串接。
- **價格因子**：`price_factor = 恢復買賣參考價 / 停止買賣前收盤價`，與除權息表的定義相同。官方公式：退還股款
  `參考價 = (停止買賣前收盤價 − 息值 − 每股退還股款) / 減資換股率`；彌補虧損與變更面額 `參考價 = 停止買賣前收盤價 / 換股率`；
  ETF `參考價 = 停止買賣前最後收盤價 / 分割比率`。
- **股數乘數**（只影響 `split_adjusted_volume`，不影響價格）：彌補虧損減資、變更面額與 ETF 分割用
  `停止買賣前收盤價 / 參考價`（即換股率的倒數）；TPEx 減資用詳細資料的「每壹仟股換發新股票 / 1000」，缺值時退回價格比並在
  `source` 加 `_share_rate_from_price`；TWSE 退還股款沒有明細頁可查，也用價格比並標示
  `twse_twtauu_share_rate_from_price`（2016–2026 共 162 筆），誤差為退還股款佔股價的比例。
- **原始欄位不動**：`open/high/low/close/volume` 維持官方原值；`adjustment_source` 改為
  `twse_twt49u+twtauu+twtb8u+twtcau` 與 `tpex_exdailyq+revivt`。
- **識別碼**：`configs/data_content_identity_snapshot.json` 的 `twse_official`、`tpex_official` 與
  `data_pipeline_digest` 已更新（前值記在 `previous_data_pipeline_digest`）。舊資料集（修訂 `v1`，只含股息）對已完成的
  實驗仍有效；新的資料準備必須用 `--dataset-revision v2`。EODHD 美股不受影響。

### 4. 偵測清單與驗證

`scripts/fetch_tw_capital_actions.py` 把四個端點與面額表寫成
`configs/reference/tw_capital_actions_20261026.csv`（543 筆；JSON 附檔記錄來源 URL、各表筆數與 sha256）。
`scripts/detect_tw_price_discontinuities.py` 掃描本機的舊資料集（2016-01-04 至 2026-05-29，2,157 檔）：
以 `open × adjusted_close / close` 為還原開盤，相鄰兩列比值 > 1.25 或 < 0.75 列為事件，`halt_sessions` 為兩列之間
該市場的交易日數（以 TAIEX／櫃買指數列為日曆），再與官方清單依（symbol, 恢復買賣日）比對，並把官方因子套到整張表重算。
結果在 `configs/reference/tw_price_discontinuities_20261026.csv` 與 `.json`。

| 範圍 | 事件 | 檔數 | 停牌 ≥ 2 日的事件 | 其中對到官方公告 | 套用官方因子後剩餘事件 | 剩餘且停牌 ≥ 2 日 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 年起 | 568 | 320 | 335 | 296（88.4%） | 260 | 28 |
| 2023 年起 | 124 | 109 | 82 | 75（91.5%） | 47 | 5 |

- **0050**：偵測到 2025-06-18（前一列 2025-06-10，停牌 5 個交易日）比值 0.2569，對到 TWTCAU 的官方因子 0.2500
  （188.65 → 47.16），套用後殘餘比值 1.028（首日漲幅），即正確偵測並調整；0052 的 2025-11-26 一拆七同樣對到（因子 0.1428）。
- **109 檔個股清單**：124 件中 82 件在停牌後發生，75 件（91.5%）對到同日的官方減資、面額變更或分割公告；
  另 7 件停牌後事件中 2 件是官方恢復買賣日與資料集第一個成交日差一個交易日（1538 2024-12-31、4806 2025-10-07，
  管線依日期套用因子後跳動消失），5 件不是公司行動（6103 2023-05-23、8101 2024-11-19、2740 2024-12-06、2924 2025-11-20、
  6236 2025-12-26，為其他原因停牌後的價格變動，保留原樣）。42 件沒有停牌（0–1 日）的事件是正常價格變動，不調整。
- 官方清單中 2016 至 2026-05 有 499 筆，其中 215 筆因子小於門檻沒有被偵測（中位數 1.09），仍會套用。

### 5. 尚未涵蓋與注意事項

- TWSE 退還股款的股數乘數為近似值（見第 3 節），只影響成交量正規化。
- TPEx 變更面額為整理表而非官方表；新增事件需更新 `_TPEX_PAR_VALUE_CHANGES`（會改變識別碼）。
- 彌補虧損減資同日辦理現金增資時，現金增資的除權因子由 `TWT49U` 另行提供，本表只套減資因子。
- 資料集結束日之後公告但尚未定價的列（`pending_rows`）會略過。

### 6. 執行（RunPod CPU Pod，使用者啟動）

本機不建環境、不跑完整資料準備。流程與一般 CPU prepare 相同，只多了新的資料修訂：

```bash
bash scripts/sync_project_to_runpod_volume.sh --apply
```

```bash
bash scripts/runpod_workflow.sh configure --stage stage1 --data-profile tw_only --dataset-revision v2 --start 2016-01-01 --end 2026-06-01 --universe all
```

```bash
bash scripts/runpod_workflow.sh cpu prepare --max-api-calls 1 --taiwan-qps 0.5 --maxRuntime 24h --prepareReserve 2h --maxBackoff 1m --cpuNumber 16 --cpuFlavor cpu5g
```

`--max-api-calls` 只限制 EODHD 請求，`tw_only` 不用 EODHD，填最小正整數即可。SSH 進 Pod 後 `bash scripts/runpod_tmux_launch.sh cpu-prepare`。Pod 的 gate 會先跑 `ruff check .` 與完整 pytest
（含更新後的識別碼快照比對）。P3 延長到 2026-06 之後時，把 `--end` 改成新的結束日即可，同一修訂 `v2`。完成後依既有流程把 Pod 上的 `raw/market.parquet` 下載到本機 `artifacts/datasets/<id>/raw/`，再執行：

```bash
python scripts/detect_tw_price_discontinuities.py --market-parquet artifacts/datasets/<id>/raw/market.parquet --actions configs/reference/tw_capital_actions_20261026.csv --output configs/reference/tw_price_discontinuities_v2.csv
```

預期 2023 年起停牌後的事件剩 5 件（第 4 節），0050 不再出現。

## English

### 1. Problem

`adjusted_close` in `artifacts/datasets/<id>/raw/market.parquet` is the cumulative product of the
official ex-dividend tables (TWSE `TWT49U`, TPEx `exDailyQ`) and therefore carries cash dividends and
stock dividends only. Stock splits, capital reductions (cash refunds and loss-covering reductions) and
par-value changes are all "halt for several sessions, then resume at a reference price" events in
Taiwan and are not in those tables. The §13.17 supplement of 2026-10-26 found that 0050's 1:4 split of
2025-06-18 was unadjusted (the price became a quarter after a five-session halt) and that 109 stocks show
single-session jumps of more than 25% in the adjusted open since 2023. The band engine only computes
open-to-open returns between adjacent signal dates of one name and exits on a halt, so the position returns
of §13.14–§13.19 are unaffected; the training labels (h-day close returns) and features (trailing 20 / 60
day returns) do carry the jumps.

### 2. Sources

| Event | Source | Fields | Rows 2016–2026 |
| --- | --- | --- | --- |
| TWSE capital reductions (cash refund, loss covering) | `https://www.twse.com.tw/rwd/zh/reducation/TWTAUU` | resumption date, code, last close before the halt, resumption reference price, reason | 290 (2 more announced without prices) |
| TWSE par-value changes | `https://www.twse.com.tw/rwd/zh/change/TWTB8U` | as above | 10 |
| TWSE ETF splits / reverse splits | `https://www.twse.com.tw/rwd/zh/split/TWTCAU` | resumption date, ETF code, last close, reference price | 2 (0050, 0052; 9 leveraged / inverse ETFs are outside the allowlist) |
| TPEx capital reductions | `https://www.tpex.org.tw/www/zh-tw/bulletin/revivt` | resumption date, code, last close, reference price, detail cell (shares per 1,000 old shares, cash refund per share) | 229 |
| TPEx par-value changes | TPEx has no queryable table, only per-company announcements; curated from the price-discontinuity scan (first session after the 5–7 session exchange halt) and the current par values in the TPEx company data (`configs/reference/tw_company_reference_20261022.csv`) into `_TPEX_PAR_VALUE_CHANGES` in `taiwan.py` | 12 (10 names) |

Each official endpoint returns the whole date range in one request. Each provider's `fetch_actions`
calls `fetch_resumption_actions` after the ex-dividend table and appends the rows to the same
corporate-action frame. The endpoints, parsers and the par-value table are part of the data content identity.

### 3. Method

- **Event date**: the resumption date (first session of the new shares). `apply_cumulative_adjustments`
  multiplies `adjusted_close` of every earlier row by `price_factor`, on top of the dividend factors;
  several events on one date (a split plus an ex-dividend) multiply into one factor and `source` is joined
  with `+`.
- **Price factor**: `price_factor = reference price / last close before the halt`, the same definition
  as the ex-dividend tables. Official formulas: cash refund `reference = (last close − dividend − refund
  per share) / share rate`; loss covering and par-value change `reference = last close / share rate`;
  ETF `reference = last close / split ratio`.
- **Share multiplier** (scales `split_adjusted_volume` only, never prices): loss-covering reductions,
  par-value changes and ETF splits use `last close / reference` (the inverse share rate); TPEx reductions
  use the detail cell's shares per 1,000 / 1,000 and fall back to the price ratio with
  `_share_rate_from_price` appended to `source`; TWSE cash refunds have no detail page, so they use the
  price ratio and are tagged `twse_twtauu_share_rate_from_price` (162 rows in 2016–2026), with an error
  equal to the refund's share of the price.
- **Raw fields unchanged**: `open/high/low/close/volume` keep the official values; `adjustment_source`
  becomes `twse_twt49u+twtauu+twtb8u+twtcau` and `tpex_exdailyq+revivt`.
- **Identity**: `twse_official`, `tpex_official` and `data_pipeline_digest` in
  `configs/data_content_identity_snapshot.json` are updated (the previous value is kept as
  `previous_data_pipeline_digest`). Existing datasets (revision `v1`, dividend-only) stay valid for the
  experiments that used them; new preparations must pass `--dataset-revision v2`. EODHD US data is unaffected.

### 4. Detection list and verification

`scripts/fetch_tw_capital_actions.py` writes the four endpoints plus the par-value table to
`configs/reference/tw_capital_actions_20261026.csv` (543 rows; the JSON sidecar records the URLs, per-table
counts and the sha256). `scripts/detect_tw_price_discontinuities.py` scans the local old dataset
(2016-01-04 to 2026-05-29, 2,157 names): the adjusted open is `open × adjusted_close / close`, adjacent rows
with a ratio above 1.25 or below 0.75 are events, `halt_sessions` is the number of that market's sessions
between the two rows (calendar from the TAIEX / TPEx index rows), every event is matched to the official
list on (symbol, resumption date), and the official factors are applied to the whole table and the scan
repeated. Results: `configs/reference/tw_price_discontinuities_20261026.csv` and `.json`.

| Range | Events | Names | Events after a halt of ≥ 2 sessions | Of which matched to an official announcement | Events left after applying the official factors | Left and after a halt |
| --- | --- | --- | --- | --- | --- | --- |
| Since 2016 | 568 | 320 | 335 | 296 (88.4%) | 260 | 28 |
| Since 2023 | 124 | 109 | 82 | 75 (91.5%) | 47 | 5 |

- **0050**: detected on 2025-06-18 (previous row 2025-06-10, five-session halt) with ratio 0.2569, matched to
  the TWTCAU factor 0.2500 (188.65 → 47.16); the residual after adjustment is 1.028 (the first-day move),
  so the split is detected and adjusted correctly. 0052's 1:7 split of 2025-11-26 is matched as well
  (factor 0.1428).
- **The 109-name list**: 82 of the 124 events follow a halt and 75 of them (91.5%) match a same-day official
  capital reduction, par-value change or split. Of the remaining 7 post-halt events, 2 are cases where the
  official resumption date is one session before the dataset's first traded bar (1538 on 2024-12-31, 4806 on
  2025-10-07; the pipeline applies the factor by date and the jump disappears) and 5 are not corporate
  actions (6103 2023-05-23, 8101 2024-11-19, 2740 2024-12-06, 2924 2025-11-20, 6236 2025-12-26: price moves
  after halts of other causes, left as they are). The 42 events without a halt (0–1 sessions) are ordinary
  price moves and are not adjusted.
- The official list has 499 rows between 2016 and 2026-05; 215 of them have factors below the threshold
  (median 1.09) and were not detected, but they are applied all the same.

### 5. Not covered, and caveats

- The share multiplier of TWSE cash refunds is approximate (section 3) and only affects volume normalization.
- TPEx par-value changes are a curated table, not an official one; new events need an entry in
  `_TPEX_PAR_VALUE_CHANGES` (which changes the identity).
- When a loss-covering reduction and a cash capital increase share one date, the ex-rights factor of the
  increase comes from `TWT49U`; this table applies the reduction factor only.
- Rows announced after the dataset's end without prices yet (`pending_rows`) are skipped.

### 6. Running it (RunPod CPU Pod, launched by the user)

No local environment and no full local preparation. The flow is the usual CPU prepare with a new
dataset revision:

```bash
bash scripts/sync_project_to_runpod_volume.sh --apply
```

```bash
bash scripts/runpod_workflow.sh configure --stage stage1 --data-profile tw_only --dataset-revision v2 --start 2016-01-01 --end 2026-06-01 --universe all
```

```bash
bash scripts/runpod_workflow.sh cpu prepare --max-api-calls 1 --taiwan-qps 0.5 --maxRuntime 24h --prepareReserve 2h --maxBackoff 1m --cpuNumber 16 --cpuFlavor cpu5g
```

`--max-api-calls` caps EODHD requests only; `tw_only` does not use EODHD, so the smallest positive integer is enough. After SSH login run `bash scripts/runpod_tmux_launch.sh cpu-prepare`. The Pod gate runs `ruff check .`
and the full pytest first (including the updated identity snapshot comparison). For P3 beyond 2026-06,
change `--end` to the new end date under the same revision `v2`. Once the new `raw/market.parquet` is
downloaded from the Pod into the local `artifacts/datasets/<id>/raw/` with the existing flow, run:

```bash
python scripts/detect_tw_price_discontinuities.py --market-parquet artifacts/datasets/<id>/raw/market.parquet --actions configs/reference/tw_capital_actions_20261026.csv --output configs/reference/tw_price_discontinuities_v2.csv
```

Expected: 5 post-halt events since 2023 remain (section 4) and 0050 no longer appears.
