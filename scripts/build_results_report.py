"""Build the bilingual Word results report with its own package (no template).

The narrative is written here; every results table is taken verbatim from the project's
Markdown result reports (Chinese half for the Chinese part, English half for the English part),
so the Word file carries the complete backtest tables rather than a summary of them.

Usage: python3 scripts/build_results_report.py <repo-root> <output.docx>
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx_report import markdown_to_xml, page_break, write_docx

ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
DOCS = ROOT / "docs"

def section(doc_name: str, language: str, start: str, end: str | None) -> str:
    """Slice one heading-delimited section of a report's Chinese or English half."""
    text = (DOCS / f"{doc_name}.md").read_text(encoding="utf-8")
    zh, en = text.split("\n## English\n", 1)
    half = zh if language == "zh" else en
    begin = half.index(start)
    stop = half.index(end, begin + len(start)) if end else len(half)
    return half[begin:stop]


# ------------------------------------------------------------------------------------------
# Narrative (Chinese, then English)
# ------------------------------------------------------------------------------------------

ZH_FRONT = """
# 台股橫斷面排序模型結果報告

B0 LightGBM 基線與殘差標籤變體｜D0 跨股票 transformer 與消融 A1｜D1 凍結編碼器｜策略層回測與判定

報告版本：2026-10-25｜實驗協定 §13.12–§13.17｜資料集 `53e3915b…`｜評估期間 P1 2023-01 至 2025-05、P2 2025-06 至 2026-05｜分支 `claude/model-performance-20261016`

三個模型都有真實的排序能力：h = 1 的 Rank IC 0.12（ICIR 0.9），殘差化標籤後多日排序在 P2 回到 0.065–0.071 且區間不跨 0。但在台灣現有工具與成本下，六條事先登記的策略判定沒有一條通過：Top-k dropout、band 組合、台指期對沖、股票期貨空方當沖都付不起成本，暴露中性多空在步驟 0 就被停止規則擋下。結論是：模型線收斂為 B0 基線加 D0 集成候選，沒有勝過 0050 買進持有的可部署策略。本報告第 3 至 7 章的結果表格全部取自各節的雙語結果報告，數字與 `artifacts/` 下的輸出一致。

## 1. 執行摘要與評估口徑

本輪的研究問題是：在只用台股、季度 walk-forward 重訓、同一組特徵的條件下，跨股票深度模型能否勝過特徵 LightGBM 基線，以及任一模型的排序能力能否在台灣的交易成本下變成報酬。所有判定在實作與計分前寫入實驗協定；判定用的區間是同一訊號日配對相減後的 circular moving-block bootstrap（1,000 次、seed 11），不以單一數字或單一期間下結論。

成立的三件事：（1）h = 1 的橫斷面排序能力真實且穩定，B0 0.118、D0 集成 0.122，三個 seed 的極差只有 0.002；（2）跨股票 DL 對特徵 GBDT 的增量只在 h = 1，幅度 0.003–0.004，且來自三 seed 集成而不是架構（消融 A1）；（3）多日排序在 P2 的崩潰來自標籤裡的市值與產業暴露，殘差化之後對殘差目標的 Rank IC 回到 0.065–0.071。不成立的三件事：所有多頭與多空策略的毛報酬都付不起台灣的成本；凍結的時序基礎模型編碼器（Kronos-small、Chronos-2-small）沒有增量；殘差 alpha 在約 130 檔可放空的大型股裡 P2 為 0。

### 實驗身分

| 實驗 | launch ID | seeds | fit 數 | 硬體 |
| --- | --- | --- | --- | --- |
| B0 `t-lgbm-alpha`（標準標籤） | launch-20261022T234922Z-22048038 | 42／43／44 | 588 | RunPod CPU |
| B0 變體 A `t-lgbm-alpha-residual_v1a` | launch-20261023T103805Z-2637116736 | 42／43／44 | 588 | RunPod CPU |
| B0 變體 B `t-lgbm-alpha-residual_v1b` | launch-20261023T130541Z-2860518787 | 42／43／44 | 588 | RunPod CPU |
| D0 `t-xs-scratch`（標準標籤） | launch-20261023T015746Z-224647797 | 42／43／44 | 42 | RTX PRO 4500 Blackwell |
| A1 `t-xs-a1-no-cross`（標準標籤） | launch-20261023T075624Z-2596926448 | 42 | 14 | RTX PRO 4500 Blackwell |
| D0 `t-xs-scratch`（變體 A 標籤） | launch-20261023T162203Z-877225215 | 42／43／44 | 42 | RTX PRO 4500 Blackwell |
| D1 `t-xs-kronos-small`（變體 A） | launch-20261024T074202Z-2401818372 | 42 | 14 | RTX 5090 |
| D1 `t-xs-chronos2-small`（變體 A） | launch-20261024T104417Z-980120111 | 42 | 14 | RTX 5090 |

### 事先登記的判定與結果

| 判定 | 規則 | 結果 | 解讀 |
| --- | --- | --- | --- |
| H-A（§13.13） | D0 − B0 配對 Rank IC，短格與持有格下界 > 0 | 只有短格通過：標準標籤 +0.0043 [+0.0015, +0.0071]；變體 A +0.0037 [+0.0011, +0.0065]；持有格區間跨 0 | D0 為候選，DL 增量只在 h = 1 |
| H-B（§13.13） | Top-k dropout 回測，TWSE P1 ∪ P2 年化淨超額下界 > 0 | 未通過：B0 −6.3% [−19.0%, +7.1%]；D0 −7.3% [−17.9%, +3.5%] | 每日換 5／50 檔的成本拖累約 17%／年 |
| H-D（§13.14） | band 組合 TWSE P1 ∪ P2 年化淨超額下界 > 0 | 未通過：B0 −4.7% [−19.7%, +10.5%]；D0 −5.4% [−18.2%, +7.7%]；所有資本情境皆然 | 換手降到 2%／日，但 P2 多日訊號失效 |
| H-D-hedged（§13.14） | 台指期對沖版 P1 ∪ P2 年化淨報酬下界 > 0 | 未通過：−12.5%／−13.3%，追繳 28／27 次 | 等權組合對市值加權指數的 β 配不上 |
| H-E（§13.15） | 股票期貨空方當沖 P1 ∪ P2 年化淨報酬下界 > 0 且 P2 > 0 | 未通過：毛 +61.5%／+64.5%，淨 −63.4%／−60.5% | tick 滑價 0.23%／邊；損益兩平 0.12% |
| H-F（§13.16） | 變體 A 的 B0：P2 持有格殘差目標下界 > 0，且 h = 1 配對差下界 > −0.01 | 通過：0.0646 [0.0457, 0.0856]；−0.0002 [−0.0013, +0.0009] | 崩掉的是標籤裡的暴露，不是模型 |
| §13.17 步驟 0 停止規則 | 股票期貨宇宙 h = 10 產業中性多空毛年化 ≥ 2 × 成本拖累 | 觸發停止：B0 −0.6%、D0 +0.4% 對 18.3% | 殘差 alpha 幅度太小，且不在可放空宇宙內 |

## 2. 資料、標籤與特徵

| 項目 | 內容 |
| --- | --- |
| 資料集 | `53e3915b…`：台股上市與上櫃普通股（ETF 與指數不作為樣本），2016-01-01 起，以凍結的 CPU prepare 腳本準備，還原權息 |
| 樣本與標籤 | 每個（股票, 訊號日 t）一筆；標籤為 t+1 開盤到第 h 日收盤的 benchmark 相對 log 報酬在同日同市場的 gauss-rank（Φ⁻¹((r − 0.5)/n)），h = 1…14；群組少於 20 檔無標籤 |
| 殘差化標籤（§13.16） | 每日每市場把原始 alpha 目標對截距、市值代理、產業虛擬變數做 OLS，殘差再 gauss-rank。變體 A：20 日成交金額中位數的對數；變體 B：log（原始收盤 × 快照股數）。公司參考表來自證交所與櫃買 OpenAPI 2026-10-22 快照 |
| 特徵庫 | `alpha_tw_v1`：183 欄 Alpha158 型特徵（K 線形態、開高低與 VWAP 滯後、成交量滯後、29 個滾動算子 × 5 個視窗、相對大盤報酬、成交值百分位、市場指標），只用訊號日之前的資料，原始值不正規化 |
| walk-forward | `tw_quarterly_v1`：2023Q1–2026Q2 共 14 折，每折最後 12 個月為 validation，15 個交易日 embargo，每折從頭重訓；P1 = 2023-01 至 2025-05、P2 = 2025-06 至 2026-05 |
| 指標 | 每（模型, 市場, h, 訊號日）的 Spearman Rank IC；cell short（h = 1）／holding（5–10）／long（11–14）；兩市場等權；circular moving-block bootstrap 1,000 次、seed 11、區塊 5／20／28；配對差在同一訊號日相減 |
| 成本（現股） | 手續費 0.1425% × 折扣 0.6、證交稅 0.3%（ETF 0.1%、當沖 0.15%）、滑價 TWSE 0.1%／TPEX 0.2%；另報折扣 1.0／0.28 與滑價 0／2 倍 |
| 成本（期貨） | 股票期貨每邊期交稅 0.002%、手續費每口 30 元（另報 60）、滑價 1 tick（另報 2）；台指期每邊期交稅 0.002%、每口 50 元、1 點；期交所官方日行情，結算價為 mark |

## 3. B0：LightGBM walk-forward 基線

每個 fold、每個 horizon、每個 seed 各一個 LightGBM 迴歸器（目標為 gauss-rank 標籤）：num_leaves 31、learning_rate 0.05、feature_fraction 與 bagging_fraction 0.8、min_data_in_leaf 200、最多 1,000 輪，以 fold validation 的每日 Rank IC 早停 50 輪；超參數在訓練前固定，三 seed 的分數平均為集成。以下結果表取自 `docs/lgbm_baseline_results.md`。
"""

ZH_D0 = """
## 4. D0：跨股票 transformer 與消融 A1

| 項目 | 內容 |
| --- | --- |
| 輸入 | 每檔股票 128 個交易日的還原 OHLCV 視窗（scale-free：相對最後收盤的 log 價格、相對中位數的 log 成交量，裁剪 ±5）與 183 欄特徵（batch 內橫斷面 z-score，裁剪 ±5） |
| 股票內編碼器 | patch 8 的 patch transformer，d_model 128、4 heads、2 層、feedforward 256、dropout 0.1，mean pooling |
| 跨股票 | 同一訊號日、同一市場的全部股票為一個 batch；特徵側輸入與編碼融合後，經 2 層 inter-stock attention；市場狀態門控（該日所有 token 的平均經 MLP 產生 sigmoid 門） |
| 輸出與損失 | 14 個 horizon 分數；loss = gauss-rank 標籤的 MSE + 1 ×（1 − 當日橫斷面 Pearson 相關）；645,518 參數 |
| 訓練 | AdamW lr 5e-4、weight decay 1e-4、最多 20 epoch、以 fold validation 的每日 Rank IC 早停（patience 4）、gradient clip 1.0、bf16 autocast；每折每 seed 從頭訓練；seed 42／43／44 的分數平均為集成 |
| 消融 A1–A4 | A1 關閉 inter-stock attention；A2 λ = 0；A3 不用特徵側輸入；A4 關閉市場門控。只有 A1 執行（seed 42）；A2–A4 在 A1 結果出來後取消 |

以下結果表取自 `docs/cross_stock_results.md`（標準標籤）。
"""

ZH_RESIDUAL = """
## 5. 殘差化標籤、D0 重訓與 D1 凍結編碼器

| 項目 | 內容 |
| --- | --- |
| D1 編碼器 | 凍結編碼器取代 patch encoder，embedding 以 (symbol, cutoff) 快取一次：Kronos-small（`NeoQuasar/Kronos-small` 901c26c、Tokenizer-base 0e01173，最後位置 hidden state，512 維）、Chronos-2-small（`autogluon/chronos-2-small` ddec013，五個 OHLCV 通道為多變量序列，對 variates 與 patches 平均池化，512 維）；跨股票模組 438,670 參數；依事前 seed 規則只跑 seed 42 |
| 配對 | 每個實驗對同標籤 B0 配對（H-A）；D1 另以 `scripts/compare_xs_runs.py` 對 D0 三 seed 集成與 D0 單 seed 配對 |

以下結果表取自 `docs/residual_labels_results.md`。
"""

ZH_STRATEGY = """
## 6. 策略層回測

每條策略都在實作與計分前登記（§13.13 第 3 項、§13.14、§13.15、§13.17），參數格子只在 P1 選一組，選定後固定用於 P2 與 P1 ∪ P2；報告一律同時列出對基準的超額與絕對報酬。本章四節的表格分別取自 `docs/lgbm_baseline_results.md`、`docs/cross_stock_results.md`（Top-k dropout 與持有對齊回測）、`docs/strategy_layer_results.md`（band 組合、資本情境、台指期疊加、股票期貨宇宙診斷）、`docs/futures_daytrade_results.md`（空方當沖）與 `docs/exposure_diagnostics_results.md`（步驟 0）。

### 6.1 Top-k dropout 與持有對齊回測（H-B）

Top-k dropout：TWSE 前 50 名持有、每日換 5 檔，TPEX 30／3，horizon 依 TWSE P1 淨超額 IR 選定；持有對齊回測以標籤的方式（次一開盤買進、h 日後收盤賣出）持有前 K 名，只作次要分析。B0 的表在第 3 章，D0 的表在第 4 章；兩者的 TWSE P1 ∪ P2 年化淨超額都為負（B0 −6.3% [−19.0%, +7.1%]、D0 −7.3% [−17.9%, +3.5%]），H-B 未通過；每日換手約 10% 的成本拖累約 17%／年，高於 P1 的毛超額。

### 6.2 成本感知 band 組合、資本情境與台指期疊加（H-D）
"""

ZH_FUTURES = """
### 6.3 股票期貨空方當沖（H-E）
"""

ZH_STEP0 = """
### 6.4 暴露中性策略的步驟 0 診斷（§13.17）

依使用者 2026-10-24 設定的門檻，§13.17 登記的策略（0050 核心加股票期貨多空疊加、0050 核心加多頭衛星）必須勝過 0050 買進持有；建策略前先以描述性診斷量殘差 alpha 的可交易幅度，並事先寫好停止規則。
"""

ZH_END = """
## 7. 結論、限制與後續

- 模型線收斂：B0 為策略基線、D0 三 seed 集成並列候選；D1 兩個編碼器與剩餘消融不再投入；`t-xs-kronos-base-lora` 以前者有效為前提，不執行。
- 在台灣現有工具與成本結構下，這條模型線沒有勝過 0050 買進持有的可部署策略。有證據的能力是 h = 1 排序（執行成本下不可交易）與殘差多日排序（幅度太小、不在可放空宇宙內）。
- 三個結構性原因：Rank IC 0.09 對應的前後四分位殘差價差每 10 日不到 1%；現股證交稅 0.3% 與股票期貨半價差 0.17–0.24%／邊使持有 10 日的年化拖累達 12–18%；股票期貨流動宇宙內 P2 的殘差 Rank IC 為 0，多日排序能力集中在只能做多、成本更高的中小型股。
- 限制：P2 已被看過，殘差化與中性化都是看到失效後設計的修補，真正的確認是 P3 前瞻；產業與股數為 2026-10-22 快照（存活者偏誤）；股票期貨契約代碼依 2026-10 清單對應（38 個已下市代碼無法對應）；D1 只有 seed 42；步驟 0 以現股價格代表期貨並用 tick 模型估滑價；回測不含股利稅與借券。
- 能改變結論的只有三類新輸入，不是調參：成本結構（法人等級手續費、借券放空中小型股、盤中執行把價差壓到 0.1% 以下）、新的訊號來源（籌碼、分點、盤中資料）、或 P3 期間的不同市場風格。
- 可重現的交付物：整套 walk-forward pipeline（特徵庫、標籤庫、B0／D0／D1 訓練、配對比較）、事先登記的判定、本機回測模組與六份雙語結果報告。

## 附錄 A｜來源、欄位與追溯

| 類別 | 位置 |
| --- | --- |
| 實驗協定 | `docs/experiment_protocol.md` §13.12–§13.17（含每節的結果註記） |
| 結果報告 | `docs/lgbm_baseline_results.md`、`docs/cross_stock_results.md`、`docs/strategy_layer_results.md`、`docs/futures_daytrade_results.md`、`docs/residual_labels_results.md`、`docs/exposure_diagnostics_results.md`；`MODEL_DEVELOPMENT_ROADMAP.md` 第 1、9 節 |
| 模型輸出 | `artifacts/lgbm/<launch>/{results.json, predictions.parquet, targets.parquet, daily_rank_ic.parquet, fits.parquet}`；`artifacts/xs/<launch>/` 同格式，另有 `daily_rank_ic_residual.parquet`、`paired-vs-*.json`；變體評分 `label-variant-score.json` |
| 回測輸出 | `artifacts/diagnostics/backtest/{topk-dropout-*, holding-wf-*, tw-strategy-capital-20261023T080816Z, futures-universe-b0-d0-20261023T081723Z, futures-daytrade-b0-d0-20261023T090019Z, exposure-step0-b0-d0-20261024T120000Z}` |
| 外部資料 | 期交所每日行情下載（TX 與全契約，`artifacts/diagnostics/taifex-*`）；證交所與櫃買 OpenAPI 公司基本資料（`configs/reference/tw_company_reference_20261022.csv`） |
| 程式與設定 | `src/stock_forecasting/{walk_forward_lgbm.py, cross_stock_training.py, models/cross_stock.py, encoders.py, data/{feature_library,rank_labels,residual_labels,xs_cache,xs_label_variant}.py}`；`configs/{features/alpha_tw_v1.json, walk_forward/tw_quarterly_v1.json, labels/residual_v1a.json, residual_v1b.json}`；`backtest/twbacktest/*.py` 與 `backtest/config/*.json` |
"""

EN_FRONT = """
# Taiwan Cross-Sectional Rank Models: Results Report

B0 LightGBM baseline and residualized label variants | D0 cross-stock transformer and ablation A1 | D1 frozen encoders | strategy-layer backtests and decisions

Report version 2026-10-25 | experiment protocol §13.12–§13.17 | dataset `53e3915b…` | evaluation periods P1 2023-01 to 2025-05, P2 2025-06 to 2026-05 | branch `claude/model-performance-20261016`

All three models rank: the h = 1 Rank IC is 0.12 (ICIR 0.9), and with residualized labels the multi-day ranking returns to 0.065–0.071 in P2 with intervals clear of 0. Yet under Taiwan's instruments and costs none of the six pre-registered strategy decisions passes: top-k dropout, the band portfolio, the TX hedge and the stock-futures short day trade cannot pay for costs, and the exposure-neutral long-short is stopped at step 0. The model line converges on the B0 baseline with the D0 ensemble as co-candidate, and there is no deployable strategy that beats 0050 buy-and-hold. The result tables in chapters 3–7 are taken verbatim from the bilingual result reports and match the outputs under `artifacts/`.

## 1. Executive summary and evaluation convention

The research question: with Taiwan-only data, quarterly walk-forward retraining and one shared feature set, can a cross-stock deep model beat a feature LightGBM baseline, and can either model's ranking skill become returns under Taiwan trading costs? Every decision was written into the experiment protocol before implementation and scoring; the decision intervals are circular moving-block bootstraps (1,000 replicates, seed 11) of differences paired on common signal dates, never a single number or a single period.

Three things hold: (1) h = 1 cross-sectional ranking skill is real and stable, B0 0.118 and the D0 ensemble 0.122, with a seed range of 0.002; (2) the cross-stock DL's increment over the feature GBDT exists only at h = 1, is 0.003–0.004 in size, and comes from the three-seed ensemble rather than the architecture (ablation A1); (3) the P2 collapse of multi-day ranking came from size and industry exposure inside the labels, and after residualization the Rank IC against the residual target returns to 0.065–0.071. Three things fail: no long or long-short strategy's gross return pays Taiwan's costs; frozen time-series foundation encoders (Kronos-small, Chronos-2-small) add nothing; and the residual alpha is zero in P2 among the roughly 130 shortable large caps.

### Experiment identity

| Experiment | Launch ID | Seeds | Fits | Hardware |
| --- | --- | --- | --- | --- |
| B0 `t-lgbm-alpha` (standard labels) | launch-20261022T234922Z-22048038 | 42/43/44 | 588 | RunPod CPU |
| B0 variant A `t-lgbm-alpha-residual_v1a` | launch-20261023T103805Z-2637116736 | 42/43/44 | 588 | RunPod CPU |
| B0 variant B `t-lgbm-alpha-residual_v1b` | launch-20261023T130541Z-2860518787 | 42/43/44 | 588 | RunPod CPU |
| D0 `t-xs-scratch` (standard labels) | launch-20261023T015746Z-224647797 | 42/43/44 | 42 | RTX PRO 4500 Blackwell |
| A1 `t-xs-a1-no-cross` (standard labels) | launch-20261023T075624Z-2596926448 | 42 | 14 | RTX PRO 4500 Blackwell |
| D0 `t-xs-scratch` (variant A labels) | launch-20261023T162203Z-877225215 | 42/43/44 | 42 | RTX PRO 4500 Blackwell |
| D1 `t-xs-kronos-small` (variant A) | launch-20261024T074202Z-2401818372 | 42 | 14 | RTX 5090 |
| D1 `t-xs-chronos2-small` (variant A) | launch-20261024T104417Z-980120111 | 42 | 14 | RTX 5090 |

### Pre-registered decisions and outcomes

| Decision | Rule | Outcome | Reading |
| --- | --- | --- | --- |
| H-A (§13.13) | D0 − B0 paired Rank IC, short and holding cells' lower bounds > 0 | Short cell only: standard labels +0.0043 [+0.0015, +0.0071]; variant A +0.0037 [+0.0011, +0.0065]; holding intervals span 0 | D0 is the candidate; the DL increment is h = 1 only |
| H-B (§13.13) | Top-k dropout backtest, TWSE P1 ∪ P2 annual net excess lower bound > 0 | Failed: B0 −6.3% [−19.0%, +7.1%]; D0 −7.3% [−17.9%, +3.5%] | Replacing 5 of 50 names daily costs about 17% a year |
| H-D (§13.14) | Band portfolio, TWSE P1 ∪ P2 annual net excess lower bound > 0 | Failed: B0 −4.7% [−19.7%, +10.5%]; D0 −5.4% [−18.2%, +7.7%]; every capital scenario | Turnover down to 2% a day, but the P2 multi-day signal is dead |
| H-D-hedged (§13.14) | TX-hedged version, P1 ∪ P2 annual net return lower bound > 0 | Failed: −12.5% / −13.3%, 28 / 27 margin calls | An equal-weight book cannot be hedged with a cap-weighted index |
| H-E (§13.15) | Stock-futures short day trade, P1 ∪ P2 net lower bound > 0 and P2 > 0 | Failed: gross +61.5% / +64.5%, net −63.4% / −60.5% | One tick is 0.23% per side; break-even 0.12% |
| H-F (§13.16) | Variant A B0: P2 holding cell on the residual target lower bound > 0, and h = 1 paired difference lower bound > −0.01 | Passed: 0.0646 [0.0457, 0.0856]; −0.0002 [−0.0013, +0.0009] | The collapse was exposure inside the labels, not the model |
| §13.17 step 0 stop rule | Stock-futures universe, h = 10, industry-neutral long-short gross ≥ 2 × cost drag | Stopped: B0 −0.6%, D0 +0.4% against 18.3% | Residual alpha is too small and absent from the shortable universe |

## 2. Data, labels and features

| Item | Content |
| --- | --- |
| Dataset | `53e3915b…`: TWSE and TPEX common stocks (ETFs and indexes are not samples) from 2016-01-01, prepared by the frozen CPU-prepare scripts, dividend- and split-adjusted |
| Samples and labels | One row per (stock, signal date t); the label is the gauss-rank within the same date and market (Φ⁻¹((r − 0.5)/n)) of the benchmark-relative log return from the t+1 open to the close of session h, h = 1…14; groups below 20 names carry no label |
| Residualized labels (§13.16) | Per date and market, the raw alpha target is regressed on an intercept, a size proxy and industry dummies; the residual is gauss-ranked. Variant A: log of the 20-session median dollar volume; variant B: log(raw close × snapshot shares). Company reference from the TWSE and TPEX OpenAPIs, 2026-10-22 snapshot |
| Feature library | `alpha_tw_v1`: 183 Alpha158-style columns (candle shapes, open/high/low/VWAP lags, volume lags, 29 rolling operators × 5 windows, market-relative returns, dollar-volume percentile, market indicator), computed from data up to the signal date, stored raw |
| Walk-forward | `tw_quarterly_v1`: 14 folds 2023Q1–2026Q2, the last 12 months of each fold as validation, a 15-session embargo, retrained from scratch per fold; P1 = 2023-01 to 2025-05, P2 = 2025-06 to 2026-05 |
| Metrics | Spearman Rank IC per (model, market, h, signal date); cells short (h = 1) / holding (5–10) / long (11–14); markets equally weighted; circular moving-block bootstrap, 1,000 replicates, seed 11, blocks 5 / 20 / 28; paired differences on common signal dates |
| Costs (stocks) | Commission 0.1425% × 0.6 discount, transaction tax 0.3% (ETF 0.1%, day trade 0.15%), slippage TWSE 0.1% / TPEX 0.2%; discounts 1.0 / 0.28 and slippage 0 / 2× also reported |
| Costs (futures) | Stock futures per side: 0.002% tax, NT$30 commission per contract (60 also reported), one tick of slippage (two also reported); TX per side: 0.002% tax, NT$50 per contract, one point; official TAIFEX daily data, settlement price as the mark |

## 3. B0: LightGBM walk-forward baseline

One LightGBM regressor per fold, horizon and seed (target: the gauss-rank label): num_leaves 31, learning_rate 0.05, feature_fraction and bagging_fraction 0.8, min_data_in_leaf 200, at most 1,000 rounds, early stopping after 50 rounds on the fold validation's daily Rank IC; hyper-parameters fixed before training, the three seeds' scores averaged into the ensemble. The result tables below come from `docs/lgbm_baseline_results.md`.
"""

EN_D0 = """
## 4. D0: cross-stock transformer and ablation A1

| Item | Content |
| --- | --- |
| Input | Per stock, a 128-session adjusted OHLCV window (scale-free: log price relative to the last close, log volume relative to its median, clipped at ±5) and the 183 features (cross-sectional z-score inside the batch, clipped at ±5) |
| Intra-stock encoder | Patch transformer with patch 8, d_model 128, 4 heads, 2 layers, feedforward 256, dropout 0.1, mean pooling |
| Cross-stock | All stocks of one signal date and market form one batch; the feature side input is fused with the encoding, then 2 inter-stock attention layers; market-state gating (a sigmoid gate from the MLP of the day's mean token) |
| Output and loss | 14 horizon scores; loss = MSE on the gauss-rank labels + 1 × (1 − the day's cross-sectional Pearson correlation); 645,518 parameters |
| Training | AdamW lr 5e-4, weight decay 1e-4, at most 20 epochs, early stopping on the fold validation's daily Rank IC (patience 4), gradient clip 1.0, bf16 autocast; trained from scratch per fold and seed; the ensemble of seeds 42 / 43 / 44 averages the scores |
| Ablations A1–A4 | A1 inter-stock attention off; A2 λ = 0; A3 no feature side input; A4 market gating off. Only A1 ran (seed 42); A2–A4 were dropped after A1 |

The result tables below come from `docs/cross_stock_results.md` (standard labels).
"""

EN_RESIDUAL = """
## 5. Residualized labels, D0 retraining and D1 frozen encoders

| Item | Content |
| --- | --- |
| D1 encoders | A frozen encoder replaces the patch encoder; embeddings are cached once per (symbol, cutoff): Kronos-small (`NeoQuasar/Kronos-small` 901c26c, Tokenizer-base 0e01173, last-position hidden state, 512 dims) and Chronos-2-small (`autogluon/chronos-2-small` ddec013, the five OHLCV channels as one multivariate series, mean-pooled over variates and patches, 512 dims); cross-stock module 438,670 parameters; seed 42 only under the pre-registered seed rule |
| Pairing | Every experiment is paired against the same-label B0 (H-A); D1 is also paired against the D0 three-seed ensemble and the single D0 seed with `scripts/compare_xs_runs.py` |

The result tables below come from `docs/residual_labels_results.md`.
"""

EN_STRATEGY = """
## 6. Strategy-layer backtests

Every strategy was registered before implementation and scoring (§13.13 item 3, §13.14, §13.15, §13.17); parameter grids are selected once on P1 and then fixed for P2 and P1 ∪ P2; excess over the benchmark and absolute returns are always reported together. The tables of the four sections come from `docs/lgbm_baseline_results.md` and `docs/cross_stock_results.md` (top-k dropout and holding-aligned backtests), `docs/strategy_layer_results.md` (band portfolio, capital scenarios, TX overlay, stock-futures universe diagnostic), `docs/futures_daytrade_results.md` (short day trade) and `docs/exposure_diagnostics_results.md` (step 0).

### 6.1 Top-k dropout and holding-aligned backtests (H-B)

Top-k dropout: hold the top 50 names on TWSE and replace 5 a day (TPEX 30 / 3), the horizon chosen by the TWSE P1 net-excess IR; the holding-aligned backtest holds the top K the way the label does (buy at the next open, sell at the close h sessions later) and is secondary. B0's tables are in chapter 3 and D0's in chapter 4; both TWSE P1 ∪ P2 annual net excesses are negative (B0 −6.3% [−19.0%, +7.1%], D0 −7.3% [−17.9%, +3.5%]), so H-B fails; a daily turnover of about 10% costs about 17% a year, more than P1's gross excess.

### 6.2 Cost-aware band portfolio, capital scenarios and TX overlay (H-D)
"""

EN_FUTURES = """
### 6.3 Stock-futures short day trade (H-E)
"""

EN_STEP0 = """
### 6.4 Step 0 diagnostic of the exposure-neutral strategy (§13.17)

Under the bar the user set on 2026-10-24, the §13.17 strategies (0050 core plus a stock-futures long-short overlay; 0050 core plus a long-only satellite) must beat 0050 buy-and-hold; before any strategy was built, a descriptive diagnostic measured the tradable size of the residual alpha under a pre-registered stop rule.
"""

EN_END = """
## 7. Conclusions, limits and next steps

- The model line converges: B0 is the strategy baseline and the three-seed D0 ensemble the co-candidate; the two D1 encoders and the remaining ablations receive no further spend; `t-xs-kronos-base-lora`, conditional on the frozen encoder helping, was not run.
- Under Taiwan's instruments and cost structure this model line has no deployable strategy that beats 0050 buy-and-hold. The evidenced abilities are h = 1 ranking (untradable at execution costs) and residual multi-day ranking (too small, absent from the shortable universe).
- Three structural reasons: a Rank IC of 0.09 is a top-minus-bottom quartile residual spread below 1% per 10 sessions; the 0.3% stock transaction tax and the 0.17–0.24% per-side stock-futures half-spread make a 10-session hold cost 12–18% a year; the P2 residual Rank IC inside the liquid stock-futures universe is zero, with the multi-day skill concentrated in small and mid caps that can only be bought and cost more.
- Limits: P2 had been seen, and both residualization and neutralization were designed after observing the failure, so P3 is the real confirmation; industry and share counts are a 2026-10-22 snapshot (survivorship bias); stock-futures codes follow the 2026-10 list (38 delisted codes unmatched); D1 has seed 42 only; step 0 uses stock prices for futures and a tick model for slippage; the backtests omit dividend tax and securities lending.
- Only three kinds of new input can change the conclusion, none of them re-tuning: the cost structure (institutional commissions, securities lending to short small and mid caps, intraday execution below a 0.1% spread), a new signal source (flows, broker-branch or intraday data), or a different market regime in P3.
- Reproducible deliverables: the complete walk-forward pipeline (feature and label stores, B0 / D0 / D1 training, paired comparisons), the pre-registered decisions, the local backtest modules and six bilingual result reports.

## Appendix A | Sources, fields and traceability

| Category | Location |
| --- | --- |
| Experiment protocol | `docs/experiment_protocol.md` §13.12–§13.17 (with each section's result notes) |
| Result reports | `docs/lgbm_baseline_results.md`, `docs/cross_stock_results.md`, `docs/strategy_layer_results.md`, `docs/futures_daytrade_results.md`, `docs/residual_labels_results.md`, `docs/exposure_diagnostics_results.md`; `MODEL_DEVELOPMENT_ROADMAP.md` sections 1 and 9 |
| Model outputs | `artifacts/lgbm/<launch>/{results.json, predictions.parquet, targets.parquet, daily_rank_ic.parquet, fits.parquet}`; `artifacts/xs/<launch>/` in the same format plus `daily_rank_ic_residual.parquet` and `paired-vs-*.json`; variant scores in `label-variant-score.json` |
| Backtest outputs | `artifacts/diagnostics/backtest/{topk-dropout-*, holding-wf-*, tw-strategy-capital-20261023T080816Z, futures-universe-b0-d0-20261023T081723Z, futures-daytrade-b0-d0-20261023T090019Z, exposure-step0-b0-d0-20261024T120000Z}` |
| External data | TAIFEX daily downloads (TX and all contracts, `artifacts/diagnostics/taifex-*`); TWSE and TPEX OpenAPI company data (`configs/reference/tw_company_reference_20261022.csv`) |
| Code and configuration | `src/stock_forecasting/{walk_forward_lgbm.py, cross_stock_training.py, models/cross_stock.py, encoders.py, data/{feature_library,rank_labels,residual_labels,xs_cache,xs_label_variant}.py}`; `configs/{features/alpha_tw_v1.json, walk_forward/tw_quarterly_v1.json, labels/residual_v1a.json, residual_v1b.json}`; `backtest/twbacktest/*.py` and `backtest/config/*.json` |
"""


def drop_first_heading(markdown: str) -> str:
    """The sliced section starts with its own '### 3. Results' line; the chapter heading in
    this report already says that, so only the subsections are kept."""
    lines = markdown.splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines)


def results(doc: str, language: str, start_zh: str, start_en: str, end_zh: str, end_en: str) -> str:
    start, end = (start_zh, end_zh) if language == "zh" else (start_en, end_en)
    text = drop_first_heading(section(doc, language, start, end))
    return markdown_to_xml(text, demote=-1, strip_numbers=True)


def body(language: str) -> str:
    zh = language == "zh"
    front, d0, residual, strategy, futures, step0, end = (
        (ZH_FRONT, ZH_D0, ZH_RESIDUAL, ZH_STRATEGY, ZH_FUTURES, ZH_STEP0, ZH_END)
        if zh
        else (EN_FRONT, EN_D0, EN_RESIDUAL, EN_STRATEGY, EN_FUTURES, EN_STEP0, EN_END)
    )
    parts = [markdown_to_xml(front)]
    parts.append(results("lgbm_baseline_results", language, "### 3. 結果", "### 3. Results", "### 4. 意涵", "### 4. Implications"))
    parts.append(markdown_to_xml(d0))
    parts.append(results("cross_stock_results", language, "### 3. 結果", "### 3. Results", "### 4. 意涵", "### 4. Implications"))
    parts.append(markdown_to_xml(residual))
    parts.append(results("residual_labels_results", language, "### 3. 結果", "### 3. Results", "### 4. 判讀限制", "### 4. Reading limits"))
    parts.append(markdown_to_xml(strategy))
    parts.append(results("strategy_layer_results", language, "### 3. 結果", "### 3. Results", "### 4. 意涵", "### 4. Implications"))
    appendix = section("strategy_layer_results", language, "### 7. 附錄" if zh else "### 7. Appendix", None)
    appendix = appendix.replace("### 7. 附錄：", "### ").replace("### 7. Appendix: ", "### ", 1)
    parts.append(markdown_to_xml(appendix, demote=0, strip_numbers=True))
    parts.append(markdown_to_xml(futures))
    parts.append(results("futures_daytrade_results", language, "### 3. 結果", "### 3. Results", "### 4. 意涵", "### 4. Implications"))
    parts.append(markdown_to_xml(step0))
    parts.append(results("exposure_diagnostics_results", language, "### 3. 結果", "### 3. Results", "### 4. 判讀", "### 4. Reading"))
    parts.append(markdown_to_xml(end))
    return "".join(parts)


# ------------------------------------------------------------------------------------------
# Package parts
# ------------------------------------------------------------------------------------------

def main() -> None:
    body_xml = body("zh") + page_break() + body("en")
    counts = write_docx(
        OUTPUT, body_xml,
        title="台股橫斷面排序模型結果報告 / Taiwan Cross-Sectional Rank Models: Results Report",
        footer_prefix=(
            "stock_forecasting · 台股橫斷面排序模型結果報告 / Taiwan cross-sectional rank results"
            " · 2026-10-25"
        ),
    )
    print("written", OUTPUT, OUTPUT.stat().st_size, "bytes; tables", counts["tables"],
          "paragraphs", counts["paragraphs"])


if __name__ == "__main__":
    main()
