# 條件式 Alpha 預測架構

## 中文

### 1. 系統邊界

本版本是單一金融商品的純數值預測模型。訓練 target 限定為普通股、ADR／TDR，
以及經稽核且可映射的非槓桿股票型 ETF，並使用其日線 OHLCV。
不載入 LLM、不生成文字、不重建 fact，也不包含交易規則、技術指標組合、投資組合
配置或下單功能。傳統技術指標與完整交易策略屬於其他專案；本模型輸出可作為其中
一項連續、帶不確定性的訊號來源。

### 2. 時間與標籤契約

對每個決策日 `t`：

1. 模型只讀取 `t` 收盤前已知的商品與 benchmark 歷史。
2. 訊號在 `t` 收盤後產生。
3. 下一個市場交易日 `t+1` 的 raw regular-session open 進場，不延後至個股下一筆資料。
4. 進場日算第 1 個持有交易日。
5. horizon `h` 在第 `h` 個市場交易日的 raw close 出場，
   `h=h_start,...,14`，且 `h_start ∈ {1,2,3}`。

令商品與 benchmark 在同一 entry/exit timestamp 的 total-return gross factors 分別為
`G_asset(h)` 與 `G_benchmark(h)`，訓練 label 是：

```text
alpha_h = log(G_asset(h)) - log(G_benchmark(h))
```

這是 benchmark-relative adjusted execution log return。CAPM abnormal return 只保留
為未來 diagnostic/ablation 欄位，不是目前的輸入或 loss target。

正式訓練使用固定日期：train < 2025-06-01，validation 為
[2025-06-01, 2025-12-01)，test 為 [2025-12-01, 2026-06-01)。每筆最晚
`label.end_at` 必須嚴格早於 split 右界。三個 split 共同遵守 README「資料完整性、
連續性與最低流動性」；以市場日曆檢查完整 128 + 14 sessions，不壓縮缺成交日。
真實極端報酬保留，validation/test 全量循序遍歷共用來源。

### 3. Benchmark policy

- 美國普通股、ADR 與 allowlist 內的美國股票型 ETF：`VTI.US`。
- TWSE 普通股、TDR 與 allowlist 內的台灣股票型 ETF：`TAIEX.TW`。
- TPEx 股票：`TPEX.TWO`。
- 槓桿、反向、債券、商品、波動率與未稽核 ETF 一律排除；
  `benchmark_mapping_path` 只允許替已在 allowlist 的 ETF 改用另一個可稽核
  benchmark，不能把被排除的 ETF 加入訓練。
- benchmark index 只作條件輸入與 label construction，不是訓練 target symbol。

未來 benchmark 資料只在離線 label construction 出現。序列化的 `context` 與
`benchmark_context` 都在 `cutoff_at` 結束。

### 4. Corporate-action adjustment

raw O/H/L/C 永久保留在 canonical Parquet，台股 volume 也保留官方 raw field。
EODHD EOD volume 本身已做 split adjustment，因此 canonical `volume` 由完整
Historical Splits response 反推，vendor 值保存為 `split_adjusted_volume`。模型視窗使用
point-in-time 調整：

- O/H/L/C 使用 total-return adjustment factor，並除以 `cutoff_at` 當日 factor；
  因此 vendor 對整段歷史的共同 back-adjustment scale 會抵消，未來事件也不會改寫
  既有推論視窗。
- volume 只使用 split/share-change factor，不使用現金股利 factor。
- label 的 entry 是 raw open 乘以當日 total-return factor，exit 是 adjusted close；
  商品與 benchmark 使用完全相同日期。

EODHD 保留 `adjusted_close`，並對每個 symbol 取得不帶日期裁切的完整 Historical
Splits response，以反推未調整 volume；不會對 vendor 已調整 volume 再乘一次 split
factor。TWSE/TPEx 使用官方除權息資料，並由既有月度 benchmark rows 取得實際交易日；
TAIEX/TPEx benchmark 另以官方 price index OHLC 與 total-return index 對齊。這個設計避免分割
與除權息造成的非經濟跳空，同時保留下一日開盤進場的可交易語意。

### 5. 模型資料流

```text
asset adjusted OHLCV through close t ─────┐
                                          ├─ shared Kronos tokenizer/predictor
benchmark adjusted OHLCV through close t ─┘       (LoRA / partial adaptation)
                                                         │
                                     per-bar causal hidden states
                                                         │
                              shared Causal Perceiver Resampler
                                                         │
                         asset latents + historical benchmark latents
                                                         │
                              GatedBenchmarkConditioner
                                 cross-attention + gate
                                                         │
                              conditioned numeric latent tokens
                                                         │
                               MultiHorizonAlphaHead
                                                         │
                        alpha_quantiles [B,15-h_start,3]
```

資產與 benchmark 共用同一個 Kronos encoder 與 resampler 權重。這能避免兩條分支
學到不必要的不同座標系，並控制 side-project 的參數量。`GatedBenchmarkConditioner`
以資產 latent 作 query、歷史 benchmark latent 作 key/value，再用 learnable sigmoid
gate 控制 benchmark 注入強度。gate 初始 bias 為負值，讓模型從接近資產自身表示開始，
再由訓練資料決定需要多少市場條件。

這是「動態條件耦合」：alpha head 直接預測條件 alpha 分布。它不是兩個互相獨立的
raw-return/alpha heads，也不是把兩個 q50 做算術相減。

Alpha head 同時接收 20 維截至 `t` 的歷史數值特徵與 benchmark latent 的直接路徑。
`adaptive64`／`adaptive128` 在共享 head trunk 之外，加入零初始化的市場 residual
輸出，讓 US、TWSE、TPEx 分別調整 location 與上下寬度參數；不複製 Kronos。
數值分支的 48 維編碼另經獨立 residual 接入 ranking score，排序輸出層不直接改寫 q50。

### 6. 輸入與輸出 schema

模型輸入：

```text
asset_ohlcv:               float [B,T,5]
benchmark_ohlcv:           float [B,T,5]
asset_attention_mask:       bool [B,T]
benchmark_attention_mask:   bool [B,T]
asset_timestamps:            int [B,T,5]
benchmark_timestamps:        int [B,T,5]
```

報酬分布輸出：

```text
alpha_quantiles: float [B,15-h_start,3]
```

啟用獨立排序目標時另輸出 `ranking_scores: float [B,15-h_start]`；它是無量綱
score，不是中位數報酬或機率。診斷欄位 `scale_features` 為 past-only 數值統計；
`pinball_loss`、`weighted_pinball_loss` 與 `ranking_loss` 分別保留各 loss component。

其中 `h_start` 只能是 1、2 或 3，horizons 是連續的
`[h_start,h_start+1,...,14]`；quantiles 固定為 `[0.1,0.5,0.9]`，並由
參數化方式保證 `q10 <= q50 <= q90`。輸出同時暴露以下數值表示，供稽核與未來
多模態銜接：

- `asset_last_hidden_state`
- `benchmark_last_hidden_state`
- `asset_latent_tokens`
- `benchmark_latent_tokens`
- `conditioned_latent_tokens`
- `conditioning_gate`

沒有 logits、class probabilities、文字 token、fact output 或 language-model state。

### 7. Loss 與訊號後處理

主要預測 objective 是所有 horizon 與 q10/q50/q90 的 pinball loss。每個 horizon
先除以只由 train split 估計的 robust scale：

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
loss = mean(pinball(alpha_h / scale_h))
```

模型輸出維持原始 log-return 單位。q50 location 與正值區間寬度分離；歷史尺度只控制
上下寬度，不同步放大 q50。training total loss 為 normalized pinball 加上權重 0.05
的獨立 ranking-head loss；checkpoint selection 只使用未校準的 validation pinball。
整合設定對 training pinball 使用 US/TWSE/TPEx/unknown = `[1,2,2,1]` 的固定權重，
不以單一 microbatch 的權重總和抵消市場倍率。未加權 pinball 仍獨立記錄。

最後選定的 checkpoint 以 validation 擬合上下尾校準，test 同時保留原始與校準結果。
整合設定增加市場／horizon／歷史波動狀態分組、日期等權後的新近衰減及向上層係數
收縮；凍結係數只讀 validation。`calibrated_online` 是另外標記的逐日診斷，只使用
`t+h` 出場日已嚴格早於當次預測日的標籤，且不修改 q50、模型或凍結係數。原靜態
校準結果另存 `calibrated_static`。單筆推論使用凍結的波動狀態校準。

近期加權與延遲回饋的設計參考
[Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415) 與
[Adaptive Conformal Inference](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)。
本專案的 bounded log-width 更新不是這些論文定理的直接實作，不宣稱保證未來或各市場
80% coverage；需同時比較 coverage、寬度與 pinball。

五級方向訊號完全是推論後處理。對門檻 `tau >= 0`：

- `q90 < -tau`：strong bearish
- 否則 `q50 < -tau`：bearish
- `-tau <= q50 <= tau`：neutral
- 否則 `q10 <= tau`：bullish
- `q10 > tau`：strong bullish

後處理規則不回傳機率，也不參與訓練或 checkpoint selection。

### 8. Trainability 與 checkpoint

| 元件 | 狀態 |
|---|---|
| Kronos tokenizer | frozen |
| Kronos base predictor | LoRA-32／64／128 實驗凍結；partial 實驗解凍最後 2 層與 final norm |
| Kronos 指定 projection/MLP LoRA | trainable |
| shared resampler | trainable |
| benchmark conditioner | trainable |
| multi-horizon alpha head | trainable |
| 獨立 ranking head | trainable |

partial 實驗不在已解凍層重複加入 LoRA；前面各層仍使用 LoRA-32。原六份設定保留；
整合設定提供 A/B 各兩份 LoRA-64／128，部分解凍保留為次要研究線。十份設定與
`train --experiment` 用法見 README。只有 training 採年度遞減動態抽樣。
整合設定的排程按實際已呈現 windows 計數，在 100 萬至 400 萬間衰減 LR，並保留
validation plateau、最低 LR 訓練間隔與 early stopping；checkpoint 保存樣本計數，
換 GPU 重新調校 batch 不會把排程歸零。

`adapter.safetensors` 僅保存上述可訓練參數。trainer state 必須包含
`model_output_schema_version=5.0`、包含 `h_start` 的 architecture digest、
dataset/training contract、
固定的 Kronos source/model/tokenizer revisions 與 artifact hashes。舊版 LLM/fact/
三分類或固定 3–14 horizon checkpoint 缺少 schema 5.0 或正確 trainable-key union，
必須 fail closed。

checkpoint artifact 外層 schema 仍保留既有 RunPod lifecycle 所需版本；這只是部署
相容層，不代表舊模型輸出仍存在。

### 9. RunPod 相容邊界

Pod 建立、S3 source sync、network volume、readiness、W&B identity、checkpoint
ranking、validation、自動終止與 artifact download 流程保持不變。穩定 CPU prepare
script 固定傳入 `--stride 5` 與 `--embargo-bars 5`，因此它們保留為 readiness
sentinel；真正的 dataset 值記錄為 `effective_sample_stride=1` 與
`effective_embargo_bars=14`。訓練與 manifest validator 同時驗證 sentinel 與有效值，
避免部署相容參數誤改新標籤語意。

### 10. 未來多模態能力

本次不保留 LLM 或 fact head，但保留 `encode_ohlcv(...)` 與 conditioned numeric latent
tokens。未來可在這些表示之後加入 point-in-time 文字、財報或事件 encoder；不得把
未來文字、未來 benchmark 或 label diagnostics 當作模型輸入。新的跨模態 objective
與 checkpoint schema 應另行設計，不應靜默改寫目前 alpha contract。

### 11. 依據

- [Kronos paper](https://arxiv.org/abs/2508.02739)
- [Kronos official repository](https://github.com/shiyu-coder/Kronos)
- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [TWSE ex-right/ex-dividend calculation](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)

---

## English

### 1. System boundary

This version is a strictly numerical, single-instrument forecasting model. Its
training targets are common stocks, ADRs/TDRs, and audited benchmark-mappable
unleveraged equity ETFs using daily OHLCV. It does not load an LLM, generate text,
reconstruct facts, combine technical-analysis rules, allocate a portfolio, or
place orders. A separate trading system may consume this model as one
continuous, uncertainty-aware indicator.

### 2. Timing and label contract

For each decision date `t`:

1. The model reads only instrument and benchmark history known by close `t`.
2. It emits a signal after close `t`.
3. Entry is the next market session's raw regular-session open, never a later available asset row.
4. The entry day counts as holding day one.
5. Horizon `h` exits at the raw close of the `h`th market session, for
   `h=h_start,...,14` and `h_start in {1,2,3}`.

If the instrument and benchmark total-return gross factors over the identical
entry/exit timestamps are `G_asset(h)` and `G_benchmark(h)`, then:

```text
alpha_h = log(G_asset(h)) - log(G_benchmark(h))
```

This is benchmark-relative adjusted execution log return. CAPM abnormal return
is reserved for a future diagnostic/ablation and is not an input or current
loss target.

Production uses train < 2025-06-01, validation [2025-06-01, 2025-12-01), and
test [2025-12-01, 2026-06-01). The latest `label.end_at` must precede its split's
upper boundary. All splits follow the README completeness, continuity, and
minimum-liquidity rules: exact 128 + 14 market sessions, no compressed missing
days, and no removal of genuine extreme returns. Validation/test traverse the
complete eligible population from one shared source.

### 3. Benchmark policy

- US common stocks, ADRs, and allowlisted equity ETFs: `VTI.US`.
- TWSE common stocks, TDRs, and allowlisted Taiwan-equity ETFs: `TAIEX.TW`.
- TPEx stocks: `TPEX.TWO`.
- Leveraged, inverse, bond, commodity, volatility, and unaudited ETFs fail
  closed. `benchmark_mapping_path` may change the benchmark only for an ETF
  already in the allowlist; it cannot add an excluded ETF to training.
- Benchmark indices are conditioning/label series, never target instruments.

Future benchmark data appears only during offline label construction. Both
serialized contexts stop at `cutoff_at`.

### 4. Corporate-action adjustment

Canonical Parquet permanently retains raw O/H/L/C, plus official raw Taiwan
volume. EODHD EOD volume is already split-adjusted, so canonical `volume` is
reconstructed with the complete Historical Splits response and the vendor value
is retained as `split_adjusted_volume`. Model windows use a
point-in-time adjustment:

- O/H/L/C uses the total-return factor normalized by the factor at `cutoff_at`.
  A vendor's common back-adjustment scale therefore cancels, and future actions
  cannot rewrite an existing inference window.
- Volume uses only split/share-change factors, never cash-dividend factors.
- Label entry is raw open times its contemporaneous total-return factor; exit
  is adjusted close. Instrument and benchmark use identical dates.

EODHD retains `adjusted_close` and fetches the complete, non-date-truncated
Historical Splits response per symbol to reconstruct unadjusted volume. It never
multiplies vendor-adjusted volume by a split factor again. TWSE/TPEx use official
corporate-action records and derive actual sessions from existing monthly
benchmark rows; their benchmarks align official price-index OHLC with official
total-return indices. This removes
artificial corporate-action gaps while preserving tradable next-open entry
semantics.

### 5. Model flow

```text
asset adjusted OHLCV through close t ─────┐
                                          ├─ shared Kronos tokenizer/predictor
benchmark adjusted OHLCV through close t ─┘       (LoRA / partial adaptation)
                                                         │
                                     per-bar causal hidden states
                                                         │
                              shared Causal Perceiver Resampler
                                                         │
                         asset latents + historical benchmark latents
                                                         │
                              GatedBenchmarkConditioner
                                 cross-attention + gate
                                                         │
                              conditioned numeric latent tokens
                                                         │
                               MultiHorizonAlphaHead
                                                         │
                        alpha_quantiles [B,15-h_start,3]
```

Both streams share Kronos and resampler weights. The conditioner uses asset
latents as queries and historical benchmark latents as keys/values, then applies
a learned sigmoid gate. Its negative initial gate bias starts near the
instrument-only representation and lets training learn the useful market
contribution. This is dynamic conditional coupling: the model predicts alpha
directly, rather than training independent raw-return and alpha heads or
subtracting two q50 values.

The alpha head also receives 20 historical numerical features through `t` and a
direct benchmark-latent path. `adaptive64`/`adaptive128` add zero-initialized market
residual outputs for separate US/TWSE/TPEx location and width adjustments without
copying Kronos. The numerical branch's 48-dimensional encoding additionally feeds
an independent ranking residual; ranking output layers do not directly overwrite q50.

### 6. Input and output schemas

Inputs:

```text
asset_ohlcv:               float [B,T,5]
benchmark_ohlcv:           float [B,T,5]
asset_attention_mask:       bool [B,T]
benchmark_attention_mask:   bool [B,T]
asset_timestamps:            int [B,T,5]
benchmark_timestamps:        int [B,T,5]
```

The return-distribution prediction is:

```text
alpha_quantiles: float [B,15-h_start,3]
```

An enabled independent ranking objective also exposes `ranking_scores: float [B,15-h_start]`,
dimensionless scores rather than median returns or probabilities. Diagnostic
`scale_features` contains past-only statistics; `pinball_loss`, `weighted_pinball_loss`
and `ranking_loss` preserve distinct loss components.

`h_start` is restricted to 1, 2, or 3, and horizons are the contiguous sequence
`[h_start,h_start+1,...,14]`; quantiles are fixed to `[0.1,0.5,0.9]`.
The parameterization guarantees `q10 <= q50 <= q90`. The output also exposes
asset/benchmark hidden states, both latent streams, conditioned latent tokens,
and the conditioning gate for auditability and future multimodal integration.
It contains no logits, class probabilities, text tokens, facts, or language
model state.

### 7. Loss and signal post-processing

The primary prediction objective is pinball loss across every horizon and quantile. Each
horizon is normalized by a train-only robust scale:

```text
scale_h = max(IQR_h, 1.4826 * MAD_h, 1e-4)
loss = mean(pinball(alpha_h / scale_h))
```

Predictions remain in original log-return units. The q50 location is separate from
positive interval widths; historical scale controls the widths without jointly
rescaling q50. Training adds an independent ranking-head loss with weight 0.05 to
normalized pinball; checkpoint selection uses only uncalibrated validation pinball.
Integrated presets weight training pinball by US/TWSE/TPEx/unknown = `[1,2,2,1]`;
normalization does not cancel the market multiplier inside a single-market microbatch.
Unweighted pinball remains separately logged.

The selected checkpoint fits tail calibration on validation, with both raw and
calibrated test results retained. Integrated presets add market/horizon/historical-volatility
groups, equal date mass followed by recency decay, and shrinkage toward parent factors.
Frozen coefficients use validation only. Separate `calibrated_online` daily diagnostics
use labels only when their `t+h` exit is strictly before the current forecast date; they
never change q50, model weights or frozen factors. Original static calibration is retained
as `calibrated_static`; single-point inference uses frozen regime calibration.

Recency weighting and delayed feedback are informed by
[Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415) and
[Adaptive Conformal Inference](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
The project's bounded log-width controller is not a direct implementation of their
theorems and provides no future or per-market 80% coverage guarantee. Evaluate coverage
together with width and pinball.

Five-level signals are deterministic post-processing:

- `q90 < -tau`: strong bearish
- otherwise `q50 < -tau`: bearish
- `-tau <= q50 <= tau`: neutral
- otherwise `q10 <= tau`: bullish
- `q10 > tau`: strong bullish

Post-processing returns no probabilities and never affects training or
checkpoint selection.

### 8. Trainability and checkpoints

| Component | State |
|---|---|
| Kronos tokenizer | frozen |
| Kronos base predictor | Frozen for LoRA-32/64/128; final 2 blocks and final norm unfrozen for partial |
| selected Kronos projection/MLP LoRA | trainable |
| shared resampler | trainable |
| benchmark conditioner | trainable |
| multi-horizon alpha head | trainable |
| independent ranking head | trainable |

The partial experiment does not add redundant LoRA to unfrozen blocks; preceding
blocks use LoRA-32. The original six presets remain available; integrated A/B presets
each provide LoRA-64/128, with partial unfreezing a secondary research line. See README
for all ten presets and `train --experiment`. Only training uses annual-decay dynamic sampling.
Integrated scheduling decays LR between one and four million actual window presentations
while retaining validation plateau reductions, minimum-LR intervals and early stopping.
Checkpoints persist the sample counter; changing GPU and retuning batches cannot reset it.

`adapter.safetensors` stores only the trainable union. Trainer state binds
`model_output_schema_version=5.0`, the `h_start`-aware architecture digest,
dataset/training contracts,
pinned source/model/tokenizer revisions, and artifact hashes. Old LLM/fact/
three-class checkpoints fail closed. The outer checkpoint artifact version
retains the stable RunPod lifecycle compatibility layer; it does not imply that
the old outputs remain.

### 9. RunPod compatibility boundary

Pod creation, S3 source sync, network volumes, readiness, W&B identity,
checkpoint ranking, validation, automatic termination, and artifact download
remain unchanged. Stable CPU preparation passes `--stride 5` and
`--embargo-bars 5`; these remain readiness sentinels. Effective dataset values
are separately recorded as `effective_sample_stride=1` and
`effective_embargo_bars=14`. Validators bind both layers.

### 10. Future multimodal capability

This version retains `encode_ohlcv(...)` and conditioned numeric latent tokens,
not the LLM or fact head. A later system may attach point-in-time text,
fundamental, or event encoders after these representations. Future text,
future benchmark values, and label diagnostics must never become inputs. A new
cross-modal objective requires an explicit new checkpoint contract.

### 11. Evidence

- [Kronos paper](https://arxiv.org/abs/2508.02739)
- [Kronos official repository](https://github.com/shiyu-coder/Kronos)
- [EODHD historical EOD API](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes)
- [EODHD split calendar API](https://eodhd.com/financial-apis/calendar-upcoming-earnings-ipos-and-splits)
- [EODHD Historical Splits API](https://eodhd.com/financial-apis/api-splits-dividends)
- [TWSE ex-right/ex-dividend calculation](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html)
- [TPEx return index](https://www.tpex.org.tw/web/stock/iNdex_info/reward_index/ROE.php?l=en-us)
