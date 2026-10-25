# stock_forecasting 模型開發沿革與改善任務交接

## 繁體中文

### 1 文件用途與目前結論

本文件將本次對話的需求與決策、Git 歷史、已下載訓練產物、歷代模型效能報告，以及未完成實驗的保存紀錄，整理為可交給其他 session 或 AI 工具的開發 roadmap。範圍截至 2026-10-16；編寫時的 repository HEAD 為 `9582978`。本次只整理既有證據，沒有重訓、重新評估、啟動 Pod、改動模型或清理資料。

**目前仍未建立穩健、跨市場、跨 seed 的預測優勢。** 最新 `a-signal64` 的三個 seed 已全部有 alpha 與個股分類的最終 test 結果；先前「seed 5067 分類器尚在訓練」是較早快照，已被本次檔案核對更新。Alpha seed 42 的單一整體 loss 稍優於 GRU，但逐日配對區間跨零；另外兩個 seed 的 alpha loss 落後可訓練 baseline。分類器的整體 accuracy 約 42.36%–44.59%，CE 都落後主要可訓練 baseline，台股長持有期 neutral 漏判與市場間不穩定仍然存在。[S1–S3]

**2026-10-17 增補（分支 `claude/model-performance-20261016`）：** A9 date-mixing 單因子對照（R24，seed 42）未通過事先固定的 validation 判準。只改訓練順序、讓每次更新涵蓋的日期×市場數由中位數 2 增為 166 後，train loss 降得更快，alpha 最佳 val loss 0.46363 卻差於歷史三 seed 的 0.45311–0.45476；分類器在 test 的 10–14 日約九成預測為「跌」，CE 1.118239。退化集中在日期層級的共同成分（點估計與市場方向機率），選中 checkpoint 的橫斷面排序沒有明顯變差。詳見第 5.6 節；後續候選見第 9 節。[S11]

**同日誤差拆解診斷（不需 GPU）：** alpha 的日期層級成分只占標準化 MSE 的 2.6%–3.8%，上市市場例外；loss 由約 24%–31% 落在 Smooth L1 線性區間的尾端樣本主導（占 81%–84%）。分類的日期層級資訊可降低 CE 5.4%–5.9%，但 GRU／GBDT 的日期層級預測都比 train 先驗差；可學到的部分是兩個任務共有的同日鑑別力。詳見[誤差拆解報告](docs/error_decomposition_audit.md)。[S12]

**P0 Kronos 逐樣本拆解（2026-10-18）：** 對 R21–R24 共 10 個 checkpoint 匯出逐樣本預測，並以同一方法拆解。主要結果：

- **同日鑑別力最佳**：Kronos 在兩個任務、兩個期間都是所有模型中最好的。
- **分類器輸在擇時**：分類器落後 baseline，原因是日期層級的擇時。Validation 擇時能力 −0.009～−0.024，test −0.041～−0.053；GBDT 約 −0.009。
- **移除擇時後追平或超越 baseline**：以不使用標籤的同日先驗正規化移除擇時後，分類器 validation CE 三 seed 平均 1.0048，與 GBDT 的 1.0047 相當；test 為 0.986，優於 GBDT 的 0.990。
- **Alpha**：US 排序訊號可以延續到 test；台股在 test 對所有模型都崩落。
- **P1 啟動門檻 S0 成立。**

目前「訓練不起來」的主因是目標與評估讓擇時錯誤蓋掉了選股能力，不是模型容量不足。詳見[誤差拆解報告第 7 節](docs/error_decomposition_audit.md)。[S13]

**2026-10-18 方向調整（第 1 階段）：**

- **產品目標**：模型輸出供交易規則參考。規則在收盤後計算訊號、隔日開盤進場，持有數天到兩週；
  需要的是絕對 alpha 與絕對漲／平／跌，不是只做排序。現有標籤已是隔日開盤進場
  （`execution_total_return`），進場點不需修改。
- **補充拆解**：分類器相對先驗的改善全部來自「會不會超出 ±1%」；「漲還是跌」在每個天期都不優於先驗，
  Kronos 與 GBDT 皆然。台股 alpha 的 test Rank IC 只在 1–3 日延續（1 日 0.063，14 日 −0.043）。
- **擱置**：推論期正規化與 P1 日期固定效果先擱置，因為兩者都移除了方向判斷。
- **第 1 階段**：只用 CPU 的資訊測試，以 GBDT 檢驗免費的台股籌碼、期貨與美股隔夜資料，
  能否讓「漲還是跌」優於先驗。2026-10-18 已執行，未通過（見下方 2026-10-19 增補）。
  判準見[實驗協定 §13.3](docs/experiment_protocol.md)，操作見 README「台股資訊測試」。[S13]

**2026-10-19 增補（第 1 階段結果與回測）：**

- **第 1 階段資訊測試未通過**（`information_found=false`）：8 個新增特徵組都沒有在兩個以上的
  決策天期通過。最佳為 `all_pre_open` 1 日，「漲還是跌」CE 比先驗改善 1.34%，但相對 `F0`
  的 95% 區間跨 0；`us_pre_open` 1 日改善 0.94%。TAIFEX 期貨與 Put/Call 資料下載失敗，
  因為免費下載只提供近 3 年。[S14]
- **回測第一階段與 1b 都未通過**（[回測結果報告](docs/backtest_stage1_results.md)）：
  - 固定持有天數（[§13.4](docs/experiment_protocol.md)）：validation 選出的 10 日設定，
    test 年化淨超額 −37% 至 −44%。
  - 持有緩衝、逐部位記帳（[§13.5](docs/experiment_protocol.md)）：test −13% 至 −40%（探索性）。
  - 前 K 名的優勢只在交易時段：次一交易日開盤到收盤領先等權約 20–31 bps（上市），收盤到
    下一個開盤又落後約 5–30 bps。持有過夜會抵銷優勢；只在交易時段持有則是當沖，來回成本
    0.52% 高於優勢。
  - 長天期的整體 Rank IC 為正時，test 的前 K 名仍可能反轉；validation 選出的設定在 test 反轉。
  - 依事先登記，停止「降低換手」路線，回到模型本身。[S15]

**2026-10-20 增補（第三階段前瞻檢驗）：**

- **主要假說未通過**（[前瞻檢驗結果報告](docs/stage3_forward_results.md)）：上市 10 日 R22-IR 在
  2026-06-01 之後的未見資料上，前十分位超額 +57.1 bps [−10.5, +124.3]，固定組合年化淨超額
  +7.4% [−36.6%, +56.0%]。依 §13.8 不採用，回到模型本身的改善。
- 點估計與 validation 相近，未通過主要是期間太短（約 7 段互不重疊的 10 日窗口）。
- 上市 6 個訊號的 Rank IC 都為正（0.074–0.149），但沒有一個前後十分位差或上市組合的區間
  排除 0；以持有指標選點在前瞻期間沒有顯現優勢；上櫃仍偏弱。[S17]

**2026-10-21 增補（改為預測橫斷面排序）：**

- **任務改為 `cross_sectional_rank`**（[§13.10](docs/experiment_protocol.md)）：只有同日同市場的相對排序
  有證據，因此改為預測名次；標籤區間不變（t+1 開盤到第 h 日收盤）。rank baseline 的市場等權
  Rank IC（validation → test）：PatchTST 0.057 → 0.035、均線交叉 0.020 → 0.043；台股除 1 日外在 test 接近 0。
- **Kronos `a-rank64` 試行**（[§13.11](docs/experiment_protocol.md)，[排序模型試行結果報告](docs/rank_pilot_results.md)）：
  台股 1 日 Rank IC 0.064 [0.027, 0.102]，通過；Kronos 減 PatchTST −0.002 [−0.021, 0.018]，沒有勝過；
  台股 6–14 日 0.008 [−0.044, 0.064]，未通過。Kronos 與 PatchTST 在 test 六格的配對差都跨 0。
- 台股 1 日排序是所有學習型模型共有的訊號，但 1 日持有就是當沖，回測第一階段的毛優勢低於成本。
  台股長天期在 test 期間對所有學習型模型失效，均線交叉反而變強。[S18]

**2026-10-22 增補（主線改為台股跨股票 DL pipeline）：**

- 文獻查核：混合 vs 只用本地市場訓練沒有一致結論（Jiang/Kelly/Xiu 2023：美國訓練模型只在小市場
  明顯較好；Choi/Jiang/Zhang 2025：分開訓練較好；Cakici/Zaremba 2026：兩者相當）。台股 1 日 Rank IC
  已達 Qlib 公開基準的水準，差的是 walk-forward 重訓、同一組特徵的 GBDT 基線，以及與持有方式對齊
  的 Top-k dropout 回測。
- 決定：只用台股 2016 起、季度 walk-forward；最終模型為 MASTER 型跨股票 transformer，編碼器可抽換
  （scratch／Kronos／Chronos-2）；必須勝過同一組特徵的 LightGBM；LLM 迴圈挖 alpha 因成本排除；
  Kronos 不再是 backbone。登記見 [§13.12、§13.13](docs/experiment_protocol.md)。

**2026-10-23／24 增補（台股主線完成；模型線與策略線的結論）：**

- **B0／D0／消融**（[§13.13](docs/experiment_protocol.md)，[LightGBM 基線報告](docs/lgbm_baseline_results.md)、[跨股票報告](docs/cross_stock_results.md)）：B0 `t-lgbm-alpha` 短格 Rank IC 0.118、D0 三 seed 集成 0.122；D0 − B0 配對差只有短格 +0.0043 [+0.0015, +0.0071] 不跨 0；A1 顯示股票間 attention 無增量，D0 的優勢是集成。多日 horizon 在 P2 崩掉（持有格 0.034／0.047）。
- **策略層**（[§13.14、§13.15](docs/experiment_protocol.md)，[策略層報告](docs/strategy_layer_results.md)、[空方當沖報告](docs/futures_daytrade_results.md)）：多頭 band 組合把換手壓到 2%／日、成本拖累 3–5%／年，但 P2 訊號失效，H-D 未通過，所有資本情境皆然；台指期對沖版追繳 27–28 次。h = 1 空方股票期貨當沖毛年化 +61–65% 但 tick 滑價 0.23%／邊使淨年化 −60%，H-E 未通過。
- **殘差化標籤**（[§13.16](docs/experiment_protocol.md)，[殘差化標籤報告](docs/residual_labels_results.md)）：對市值代理與產業取殘差後的標籤讓 B0 的 P2 持有格對殘差目標回到 0.0646 [0.0457, 0.0856]（H-F 通過），h = 1 不受損；崩掉的是標籤裡的暴露，不是模型。D0 在新標籤上 H-A 仍只有短格通過（+0.0037）；凍結 Kronos-small 等於單 seed D0，Chronos-2 略差；三 seed 極差 0.002–0.008。模型線收斂為 **B0 基線、D0 集成並列候選**，D1 不進入策略層。
- **§13.17 步驟 0**（[結果報告](docs/exposure_diagnostics_results.md)）：殘差 alpha 的可交易幅度太小（流動性宇宙 P1 的前後四分位殘差價差每 10 日 0.9–1.0%，產業中性多空毛年化 9–10%，P1 ∪ P2 5–8%），而且在約 130 檔有流動股票期貨的大型股裡 P2 的殘差 Rank IC 為 0；股票期貨多空毛年化 B0 −0.6%／D0 +0.4% 對成本拖累 18%，事前停止規則觸發，不建策略。
- **§13.19 機構成本六情境**（[結果報告](docs/institution_cost_scenarios_results.md)）：把成本換成機構條件（手續費 0.03%、滑價 0.05%、借券 2%），六個多空情境、兩個模型、三個市場共 36 格沒有一格達到事前寫死的門檻（P1 ∪ P2 淨 Sharpe ≥ 1、區間下界 > 0、P2 > 0）。多日多空的拖累只降 2–5 個百分點，瓶頸是 P2 的空頭腿（−61% 到 −89%；多日排序本質上是短期反轉訊號，放空的是近 20–60 日漲幅最大、規模與宇宙相同的股票，P2 動能延續使兩腿同時逆向）；台指期疊加因 β ≈ −0.5 而做多，救回的是 beta；當沖毛年化 +123–179% 對機構往返 0.31%／腿仍為負，只有零滑價才成立；3 億以上容量把 TPEX 的正值抹掉。補充的反轉暴露診斷：分數對 beta（相關 −0.24 到 −0.45）與前 20／60 日報酬正交化並在動能五分位內配對後，P2 的每批價差由 −1.8% 到 −0.5% 回到 +0.1% 到 +0.9%，P1 ∪ P2 四個（模型, 市場）全部為正且區間不跨 0；P2 虧損的主因是結構性暴露，不是排序能力消失；週轉率條件化無一致效果；純化後的年化毛價差 TWSE +12%、TPEX +25–28%，仍付不起每 10 日全換手的成本。把正交化分數放進 band 建構（補充二）後，同一組參數的 S1 機構淨報酬由 −6% 到 +8% 變成 +11% 到 +29%，P2 全部轉正、最大回撤 −11% 到 −18%、β −0.15 到 −0.20；機構門檻 10 格達標（B0 與 D0 的 S2 三市場、S1 的 TPEX 與 TW）；動能五分位配平因換手 10%／日不可行；容量約 3 億元以內。設計在看過 P2 後決定，達標只在資訊層面，需 P3 確認。穩健性檢查（補充三）：B0 的 V1 S1 四個曆年全部為正、不靠單一年份，D0 的結果集中在 2023–2024 且逐年遞減；V1 分數下 8 組 (h, K, m) 在四個（模型, 市場）全部為正，選定參數居中而非最佳；可借券名單已請資料專案加入第一階段。
- **結論**：在台灣現有工具與成本下，這條模型線沒有勝過 0050 買進持有的可部署策略。有證據的能力是 h = 1 的排序（執行成本下不可交易）與殘差多日排序（幅度太小、且不在可放空宇宙內）。可重現的交付物是整套 walk-forward pipeline、事先登記的判定與雙語報告。

現行架構是兩套獨立模型，不是方向 V1／V2／V3 的新名稱，也不是共享 backbone 的雙頭多任務模型。歷史中曾有 **validation 已顯示品質不佳、使用者要求直接停止訓練** 的實驗；第 6 節特別保留，不把它們包裝成完成的 holdout 實驗，也不與硬體、OOM 或啟動失敗混為一談。

本文件的三個用途：

1. 讓接手者知道每次改版試圖解決什麼、實際改了什麼、有哪些正式實驗。
2. 以 run ID、checkpoint、來源與 commit 回到證據，不依賴對話中的口頭保證。
3. 清楚區分已觀察結果、原因假說、工程驗收，以及尚未執行的改善候選。

### 2 必須保留的專案邊界

- **研究主體保留 Kronos／LoRA。** 不因其他模型表現較好，就擅自廢除 Kronos 或以技術指標／tree model 取代主專案；baseline 用於量測優勢，不代表已獲授權轉換研究主線。
- **模型是 predictive-signal layer，不是交易引擎。** 未來由 Hermes agent 結合模型輸出、傳統指標、交易規則與回測。現行標籤不得自行加入券商費用、股數、倉位或淨收益門檻。
- 現行需求是 **alpha 點估計＋個股漲／平／跌**。`alpha_point` 的正負可描述預測相對 benchmark 是否為正，不需要另一個 alpha 方向分類器。
- 使用者已要求未經授權不得移除既有 feature。歷史 preset 保留重現能力；不能以修復新功能為理由拔掉 readiness、baseline 完成檢查、資料清理檢查或既有 CLI 能力。
- Training 保留動態 sampling；validation／test 必須動態完整遍歷有效 windows，不抽樣、不 drop 尾 batch、不預先展開巨大資料集。
- 不因模型或部署改版無故重新抓行情、重做 CPU prepare，或重建數值契約未變的 baseline。
- 訓練／CUDA 驗收在相符的雲端環境執行；不能用本機缺乏訓練依賴的測試冒充雲端驗收。不得擅自建立／檢查本機訓練環境或變更既有套件管理方式。
- 不影響其他正在執行的 Pod；接續、監控、停止與清理都必須核對精確 run／Pod 身分。新付費實驗須遵守當次授權的預算。
- 所有持有期目前仍等權；使用者選擇先保留 1–14 日設定，主要目標是行情訊號，不是快速交易。
- 使用者將歷史名目 80% coverage 的 ±5 個百分點視為可先接受的研究範圍；coverage 不是當前核心瓶頸，不能靠加寬區間宣稱方向能力改善。現行模型已不輸出區間。

### 3 如何閱讀證據與 commit 對應

#### 3.1 狀態標籤

| 標籤 | 本文件中的定義 |
| --- | --- |
| `TEST_COMPLETE` | 有相符的訓練完成與 `state=ready`、`evaluation_split=test` 最終報告；正常 early stopping 仍屬完成 |
| `VALIDATION_ONLY` | 舊版已保存 checkpoint／validation benchmark，但 test 未解鎖或沒有明確 test 證據 |
| `STOPPED_POOR_VALIDATION` | 已有 validation 品質不佳／退化證據，依使用者決定主動終止；不是正常完成 |
| `STOPPED_INFRA` | 因硬體／執行效率或營運原因停止；不據此判定模型預測能力 |
| `FAILED_RUNTIME` | 啟動、CUDA 記憶體邊界、程式或流程錯誤；不是效能消融結果 |
| `STOPPED_WRONG_PRESET` | 原本要跑新任務，實際卻啟動舊 preset；有完成的舊結果也不算新架構驗收 |
| `INCOMPLETE_UNRESOLVED` | 沒有完成證據，保存資料不足以確定單一中止原因；不得補寫成品質失敗 |
| `DEFINED_NOT_VERIFIED_RUN` | 程式／YAML 已定義，但本次盤點未找到對應正式完成結果 |

W&B 的 `crashed`、舊 lifecycle 的 `preparing` 或 Pod 消失，都不是獨立的停止原因。必須結合訓練紀錄、當時決策與最終產物。

#### 3.2 三種不同的 commit

1. **架構導入 commit**：引入模型、資料規則或目標的程式變更。
2. **數值程式相符快照**：以 run manifest 的 `training_resume_contract.training_implementation.files` 逐檔 SHA-256，比對 Git 中相同程式內容。本次對有紀錄的 completed／部分 stopped runs 做了這項比對。
3. **報告 commit**：將分析報告新增、修訂或歸檔的提交；它不代表模型是在該 commit 上訓練。

表中的「快照」是 **manifest 所列數值檔案相符**，不是宣稱 run 有保存完整 `git HEAD`、整個 worktree 無差異，或所有 shell／部署腳本都等同該 commit。選取 run 初始化前的候選 Git 快照後，已核對全部列載檔案；不能只靠日期相近就當作相符。上游 `time_series_model_revision` 是 Hugging Face 權重 revision，不是本 repository commit。

#### 3.3 可比較的評估世代

| 世代 | 實際評估 | 主要限制 |
| --- | --- | --- |
| G0 原始 Stage 1／2 | 舊 chronological split，benchmark 通常限量 20,000 validation windows | test 未解鎖；長歷史也改變 split，不是共同 holdout |
| G1 固定日期、抽樣 benchmark | validation 2025-06 至 11 月；test 2025-12 至 2026-05，最終 benchmark 20,000 | A／B 有序股票—日期指紋不同；baseline 訓練亦有舊上限 |
| G2 全量但仍讀不同快照 | A test 1,261,524；B test 1,264,861 | 差 3,337 windows；不能視為完全配對的資料跨度消融 |
| G3 清理後共用評估 | train／validation／test 同一完整性規則；test 534,327 windows、112 cutoff dates | 僅一個相依時間區間；市場權重不均；任務／normalization／Rank IC 實作仍須逐版本核對 |

目前六個 signal 子模型的 test membership 相同：`93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`。名義 test 是 `[2025-12-01, 2026-06-01)`；實際 cutoff 為 2025-12-01 至 2026-05-11，因每筆都要有完整後續 14 日 labels。112 個相依日期不能當成 534,327 個獨立時間觀測。[S1–S3]

### 4 架構開發 roadmap

本節的 A0–A9 是本文件的沿革編號，不是既有 CLI／schema。Git release `v0.x`、方向系列 `V1/V2/V3`、Stage 1／2 與 A／B 資料組別是不同概念。

| 沿革 | 要解決的問題與實際變更 | 主要 commit | 已觀察結果 |
| --- | --- | --- | --- |
| A0 原始 quant-only alpha | OHLCV＋benchmark → Kronos tokenizer／backbone＋LoRA → resampler／benchmark conditioning → 1–14 日 alpha q10/q50/q90；rank 8／alpha 16 | `04e697b`；bar-store `abf8f7c`；GPU 調校 `3af6088` | Stage 2 對 Stage 1 normalized pinball 小幅改善，但方向／相關性弱，長歷史沒有穩定優勢；G0 不能作共同 test 結論 |
| A1 尺度可讀性診斷 | 凍結既有 checkpoint，ridge probes 讀歷史尺度；有 shuffled-label controls，不重新訓練 Kronos | `2562e86`；tmux／本機 guard `42d531c`、`91e3e34` | 個股尺度可被讀出，benchmark 尺度泛化較差；不是「所有尺度資訊消失」，更不是未來 alpha 已改善 |
| A2 固定日期＋小型數值分支 | Train 到 2025-05；validation 2025-06–11；test 2025-12–2026-05；8 維歷史尺度＋benchmark latent 直連 head | `642217c`；快照 tag `v0.1.0` → `fb8482e` | A／B 小幅、混合結果；仍有抽樣與不同 membership，不足以證明擴大資料一定有效 |
| A3 Full-data workflow | 完整 lazy validation／test；獨立 full-data baseline 前置建置與重用；20 維尺度、market-aware／顯式輸出尺度、同日同市場 ranking loss；LoRA rank32／alpha64 | `a51da94`、`8e7f5d1`；tag `v0.2.0` → `44e7139` | 工程與資源公平性改善，但 G2 的 A／B 仍不是相同股票—日期集合；最佳點仍很早 |
| A4 完整清理＋受控容量 | 128 個連續 input＋14 個連續 output；最低流動性；共用評估來源；年度 0.8 sampling decay；位置／寬度分離、靜態校準、獨立 ranking head；A/B 各 LoRA32／LoRA64／partial | `3ddaad7`；多 Pod `5e18f88`；大 batch 防護 `39cef62` | A-LoRA64 是該歷史 quantile 組較好的 alpha 參考；partial 未勝出。較佳 checkpoint 可到 0.6–0.8 epoch，但不代表泛化根因已解決 |
| A5 Adaptive64／128 | 市場 residual、ranking 數值直連；regime／recency 校準及延遲回饋 online 診斷；dropout 0.10、weight decay 0.03、較低 head LR 與 sample-plateau | `0d47732`；多 seed `ead2351` | Adaptive64 三 seed raw alpha 誤差都未勝 A-LoRA64；coverage 改善不等於 alpha／方向改善。Adaptive128 有 preset，未找到本次正式完成結果 |
| A6 方向 V1 `direction64` | 回到 A-LoRA64 主體，加個股絕對與 alpha 的成本帶三分類、十技術通道、horizon-query readout、表徵正則化；保留 quantiles／ranking；加權多任務 loss | `04f692b`；baseline 對齊 `39f337e`；成本邊界修復 `5a55668` | 三 seed 類別塌縮、市場失衡；方向 head 未直接接尺度；loss reference 受初始化影響，跨 seed 任務權重實際不同；未建立方向優勢 |
| A7 方向 V2 | 方向尺度直連；市場 prior 與個股 residual 分離；固定、seed 無關 train reference；共享 LoRA PCGrad；BA 導向選模＋pinball guard | `d2238a7`；CUDA 驗收快照 `f6271e7` | 部分指標改善但未超越歷史 A-LoRA64 的 alpha；選模可能犧牲 alpha 最佳點。有限梯度／PCGrad 執行不證明泛化有效 |
| A8a 方向 V3 | 有序 conditional-chain 機率；benchmark 狀態 gate 選 token／尺度特徵；CE／Brier prior skill、macro-F1、反向錯誤與塌縮懲罰選模 | `346cae0` | **Validation 已顯示退化、方向偏向與 tiny prior skill，依使用者要求直接停訓；無最終 test** |
| A8b 現行 signal64 | 回歸改 normalized Smooth L1 alpha 點估計；分類改固定 ±1% 個股三分類；兩套 Kronos／LoRA64 完全獨立，移除主線中跨任務梯度干擾與成本標籤；沿用尺度與 benchmark 路徑 | 數值架構 `bc84d49`；成對流程 `8bae2db`；selection／Python 3.12 修復 `3b7cccb` | 三 seed 全部完成；仍未建立跨 seed、跨市場且勝過 learned baseline 的穩健優勢。這是簡化後的結果，不是沿用 V3 頭 |
| A9 date-mixing 對照（未採用） | 假說：歷史 `DateMarketSampler` 以同日同市場的 128 筆區塊輸出，每次更新約只涵蓋 2 個日期×市場，使模型追逐日期共同衝擊。以有界局部混洗 `training.date_mixing_buffer_samples=16384` 讓每次更新混入多個日期；每 epoch 樣本與年度配額不變，其餘與 signal64 相同、baseline 重用 | 實作 `31ef7c1`；文件 `64a7b3f` | **Validation 未通過**：train loss 降得更快，validation 更差；分類器長天期「跌」偏誤加劇。反證原假說，混洗預設關閉、保留作重現 |

#### 4.1 尺度診斷到數值分支的因果邊界

診斷使用 `run-20260905T203327Z-270337978/checkpoint-018145`，主要可追溯結果為 `probe-20260914T101527Z-105a4d66`，另有 `probe-20260914T091728Z-18045a54`。抽樣 train 16,384、validation 4,096，只預測歷史尺度，不建 test loader、不評估未來 alpha。

`conditioned_mean` 的個股 20／60 日尺度 validation R² 約 0.552／0.605；benchmark 20／60 日卻約 −0.322／−0.548，benchmark context CV 約 0.00073。這支持排查 benchmark 尺度的讀取／分布問題，並試驗標準化前的顯式尺度分支與 benchmark 直接連接；**不支持「Kronos 完全沒有尺度資訊」或「分支必然提升方向率」的結論**。Pooling／線性探針失敗也不等於資訊不存在。[S6]

#### 4.2 資料清理的真正變更

三個 split 都要求 128＋14 個連續市場交易日、個股／benchmark 對齊、有效 OHLC／adjusted price。個股每日 volume 必須大於零；非交易型 benchmark 指數可為零。最近 60 日 raw close × raw volume 中位數須達 US$1m／TWD10m。缺日就排除 window，不往前跨月／年湊 128 筆，不壓縮停牌；有效真實極端報酬保留。

主模型 training 是年度遞減的動態 sampling；baseline 神經模型完整 train traversal，不是完全相同的 sampling 權重／FLOPs。共用評估消除 A／B 分別下載快照的集合差異，但這次盤點中 **沒有新版 B-signal64 完成實驗**，不能聲稱長歷史退化已被驗證解決。歷史大回撤分組是診斷，不是已實施的剔除規則。

### 5 已完成或已有正式 validation 報告的實驗帳本

#### 5.1 原始 Stage 1／2 與固定日期 A／B

本表 loss 是全 1–14 日 **normalized pinball**。`VALIDATION_ONLY` 不得改寫成 holdout。日期跨度與 split 以該 run 保存的 manifest／config 為準，不依對話中曾出現的日期誤記。

| ID | Run ID／實驗 | 最佳 checkpoint | 狀態、摘要 | 數值相符快照 |
| --- | --- | --- | --- | --- |
| R01 | `run-20260903T092249Z-1771920450`；早期 Stage 1 | `checkpoint-018750` | `VALIDATION_ONLY`；舊 20k 評估，loss 0.317998；僅本機歷史 audit 有其完整 metadata，無獨立 DOCX | `6e7480f`，22 檔 |
| R02 | `run-20260903T140429Z-1247813592`；Stage 1，2021 起 | `checkpoint-003908` | `VALIDATION_ONLY`；train_fraction 0.05／cap 500k；loss 0.314759，5d q50 correlation 0.078797 | `9bf84f9`，22 檔 |
| R03 | `run-20260903T184811Z-81563229`；Stage 2，2021 起 | `checkpoint-028982` | `VALIDATION_ONLY`；loss 0.307166，較 R02 約低 2.41%；5d correlation 0.107818；舊報告另記 lifecycle exit 1，不能稱全流程無異常 | `3d3831e`，22 檔 |
| R04 | `run-20260905T203327Z-270337978`；Stage 2，2016 起 | `checkpoint-018145` | `VALIDATION_ONLY`；loss 0.314283，5d correlation 0.044618；長歷史同時改變 chronological split，非純資料量消融 | `ccb973c`，22 檔 |
| R05 | `run-20260915T091153Z-918316277`；固定日期 A，2021 起 | `checkpoint-043773` | `TEST_COMPLETE`／G1；20k test，loss 0.306784；rank8＋8 維尺度／benchmark 直連 | `e3f8086`，26 檔 |
| R06 | `run-20260917T110811Z-1986920595`；固定日期 B，2016 起 | `checkpoint-094452` | `TEST_COMPLETE`／G1；20k test，loss 0.313541；raw pinball／MAE 部分較好，normalized loss 較差；與 A membership／train scales 不同 | `a37aaf8`，26 檔 |
| R07 | `run-20260922T182053Z-1388222581`；full-data A | `checkpoint-010944` | `TEST_COMPLETE`／G2；test 1,261,524；loss 0.321509；5d correlation 0.051991；最佳仍是首次驗證 | `feff226`，31 檔 |
| R08 | `run-20260923T095302Z-311336160`；full-data B | `checkpoint-023613` | `TEST_COMPLETE`／G2；test 1,264,861；loss 0.333552；5d correlation 0.068226；多資料未全面改善 | `0c6b627`，29 檔 |

R02–R08 的 DOCX 在 [早期報告歸檔](reports/deprecated/README.md)；依 run ID 對應同名檔。R01 的來源是本機 cleanup 前 metadata 快照。[S4、S7]

報告提交：R02／R03 起始報告 `a80c85f`，R03 更新 `a433e5c`，R04 `3cf3043`，R05 `a37aaf8`，R06 `28c6a1e`；R07／R08 均在 `4b33d1f` 新增，後續由 `fb841a7` 修訂／歸檔。應用 `git log --follow -- reports/...` 核對各檔案的新增與修訂，不能把報告提交當訓練版本。完整名稱索引以 S4 為準。

#### 5.2 清理後容量實驗

共同 G3 holdout、seed 42；仍是 quantile＋獨立 ranking 架構，非現行點估計。三者的全部正式流程均已完成正常 early stopping 與 test。

| ID | Run ID／實驗 | 容量與最佳點 | Raw test normalized pinball／5d q50 correlation | 結論與快照 |
| --- | --- | --- | --- | --- |
| R09 | `run-20261007T085635Z-bc59d5bbd2d84ad4`；A-LoRA32 | rank32，凍結 backbone；`checkpoint-014720`，約 0.8 epoch | 0.293694／0.081995 | 未勝 LoRA64；`39cef62`，31 檔相符 |
| R10 | `run-20261007T085627Z-cc5f9ef13b484d90`；A-LoRA64 | rank64，凍結 backbone；`checkpoint-014720`，約 0.8 epoch | **0.291608／0.098657** | 該歷史 quantile 組的 alpha 參考，不是跨任務總冠軍或已證明穩定；`39cef62`，31 檔 |
| R11 | `run-20261007T085308Z-2debcc6e669b4522`；A-partial | rank32＋最後 2 個 blocks 解凍；`checkpoint-011040`，約 0.6 epoch | 0.293823／0.083260 | 未勝 LoRA64；不能據此認定模型容量不足；`39cef62`，31 檔 |

架構導入 `3ddaad7`；初版報告 `5909b4e`，校準／策略尺度限制修訂 `fb841a7`，目前歸檔 `9582978`。請讀 [已修訂版本](reports/deprecated/pre-a-signal64-20261016/20261007-a-group/)，不要用 `pre-20261009/` 的修訂前報告作為最新解讀。[S5]

#### 5.3 Adaptive64 三 seed

| ID | Run ID／seed | 最佳 checkpoint／約 epoch | Raw test normalized pinball／5d q50 correlation | 結論 |
| --- | --- | --- | --- | --- |
| R12 | `run-20261008T084331Z-3e990331418d46d0`；42 | `checkpoint-025759`／1.4 | 0.296831／0.070310 | 選模點後移，但 alpha 未勝 R10；不能把延長訓練視為品質改善 |
| R13 | `run-20261008T084336Z-993a064dae124ab0`；43 | `checkpoint-007360`／0.4 | 0.296756／0.073966 | Seed 敏感，沒有共同改善 |
| R14 | `run-20261008T084341Z-d677229bb2c44a32`；44 | `checkpoint-014720`／0.8 | 0.296555／0.065017 | 三者 raw loss 較好，但仍差於 R10 |

三輪 `TEST_COMPLETE`；33 個 manifest 數值檔案均與 `ead2351` 相符，架構導入是 `0d47732`。報告提交 `13cf01a`；目前在 [Adaptive64 歸檔](reports/deprecated/pre-a-signal64-20261016/20261008-adaptive64/)。凍結／static／online 校準須分開，online 延遲回饋診斷不能冒充純凍結 unseen holdout。[S5]

#### 5.4 方向 V1／V2 與誤啟動的舊架構

本表 loss 仍是 **raw alpha normalized pinball**，不是多任務／方向 selection loss。不同 V1/V2/V3 選模公式不可直接比數字大小。

| ID | Run ID／實驗 | 最佳 checkpoint | Alpha loss／5d q50 correlation | 狀態、摘要與數值快照 |
| --- | --- | --- | --- | --- |
| R15 | `run-20261010T112317Z-a2128baabec94c55`；V1 seed42 | `checkpoint-003680` | 0.297410／0.044963 | `TEST_COMPLETE`；首次驗證最佳、台股 alpha 類別嚴重塌縮；`3ead8d8`，38 檔 |
| R16 | `run-20261010T112320Z-1531ef3bf22949ad`；V1 seed43 | `checkpoint-007360` | 0.294903／0.086873 | `TEST_COMPLETE`；較 seed42 好但未勝 R10；`3ead8d8`，38 檔 |
| R17 | `run-20261010T112319Z-bb81bfdd1b4545f5`；V1 seed44 | `checkpoint-003680` | 0.294654／0.083842 | `TEST_COMPLETE`；仍早期最佳、跨 seed reference 不同；`3ead8d8`，38 檔 |
| R18 | `run-20261011T200040Z-6a76903c49f1459f`；V2 seed42 | `checkpoint-007360` | 0.293857／0.084676 | `TEST_COMPLETE`；部分回升，仍未超越 R10；`f6271e7`，40 檔 |
| R19 | `run-20261014T080044Z-06239560501f47a9`；誤啟動 V1 seed42 | `checkpoint-003680` | 0.297410／0.044963 | 舊 V1 `TEST_COMPLETE`，**不是新 signal 架構的 V1**；使用者原意是新任務，CLI／文件導引失誤；`7d4130c`，42 檔 |
| R20 | `run-20261014T084311Z-5ba50929af254002`；誤啟動 V1 seed348 | `checkpoint-003680` | 0.296465／未在本表重新展開 | 舊 V1 `TEST_COMPLETE`；僅 cleanup 前 metadata 留存，本機無獨立下載報告；`7d4130c`，42 檔 |

V1 架構導入 `04f692b`，V2 `d2238a7`。R15–R18 的報告提交為 `d37fa14`，現於 [方向 V1 歸檔](reports/deprecated/pre-a-signal64-20261016/20261010-direction64/) 與 [V2 歸檔](reports/deprecated/pre-a-signal64-20261016/20261011-direction64-v2/)。R19／R20 必須保留為 **版本／操作錯置的成本紀錄**，不能把舊模型的完成當成新模型已驗收。[S5、S7]

V1 在 5d alpha 分類的 balanced accuracy 約 33.62%／34.84%／34.09%，即使 raw accuracy 約 51% 仍不代表有用。Seed42 的 TPEX alpha up 預測占比約 0.0224%、down recall 約 99.9636%，屬類別塌縮。V2 修復了尺度路徑與 reference／梯度管理，不等於已解決泛化；方向選模選到 `checkpoint-007360`，但 alpha validation pinball 的最佳點可在其他 checkpoint，需分任務檢視。[S5、S9]

#### 5.5 現行 signal64 三個成對實驗

六個子模型均為 `TEST_COMPLETE`，獨立正常 early stopping；共同 test 534,327 windows。數值實作 45 檔均與 `3b7cccb` 相符；數值架構導入 `bc84d49`、成對流程 `8bae2db`，正式報告 commit `9582978`。[S1–S3]

| ID | 父 run ID／seed | Alpha 子 run／最佳 checkpoint | 分類子 run／最佳 checkpoint |
| --- | --- | --- | --- |
| R21 | `run-20261015T064558Z-f04e155d4299416f`／42 | `run-20261015T064558Z-f04e155d4299416f-alpha`／`checkpoint-003680`，第 1 次／7 次 validation，約 0.2 epoch | `run-20261015T064558Z-f04e155d4299416f-direction`／`checkpoint-007360`，第 2 次／8 次，約 0.4 epoch |
| R22 | `run-20261015T064559Z-624f212a27894308`／347 | `run-20261015T064559Z-624f212a27894308-alpha`／`checkpoint-011040`，第 3 次／9 次，約 0.6 epoch | `run-20261015T064559Z-624f212a27894308-direction`／`checkpoint-007360`，第 2 次／8 次，約 0.4 epoch |
| R23 | `run-20261015T064600Z-13b8b4376ea5413b`／5067 | `run-20261015T064600Z-13b8b4376ea5413b-alpha`／`checkpoint-018399`，第 5 次／10 次，約 1.0 epoch | `run-20261015T064600Z-13b8b4376ea5413b-direction`／`checkpoint-011040`，第 3 次／9 次，約 0.6 epoch |

下表均為 **1–14 日指標等權平均**。Correlation／accuracy／RMSE 不是把所有 horizon 混在一起重新計算。

| Seed | Alpha normalized Smooth L1 ↓ | Alpha MAE ↓ | Alpha Pearson ↑ | Alpha 符號一致率 ↑ | 分類 CE ↓ | 分類 accuracy ↑ | Balanced accuracy ↑ | Macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.528707 | 0.060582 | 0.102801 | 54.30% | 1.039206 | 42.36% | 36.39% | 0.338556 |
| 347 | 0.540340 | 0.061778 | 0.074525 | 51.37% | 1.032702 | 44.13% | 37.67% | 0.355016 |
| 5067 | 0.536003 | 0.061239 | 0.092305 | 53.87% | 1.027663 | 44.59% | 38.05% | 0.364459 |

逐日配對 loss 差先在同一日期內平均股票與全部持有期，再對日期等權；因此不一定等於樣本整體 loss 相減。95% 區間使用 14 日期 circular moving-block bootstrap、1,000 次、seed42，未作多重比較校正；不是 accuracy／Rank IC 或跨 seed 的信賴區間。區間跨零不等於證明兩模型等效。

**實際暴露的問題：**

- Alpha learned baseline：GRU loss 0.529025，GBDT 0.531839，zero-return 0.536328。Seed42 整體 loss 排第 1／13，但主模型減 GRU 的日均 loss 差為 −0.000328，95% 區間 [−0.004133, +0.003755]，不能宣稱穩定勝出。Seed347 與 5067 的 alpha loss 排在四個 learned baseline 後面。
- 分類的事後 test-majority 參考為 42.90%，不是可部署的 test-fitted baseline。Seed347／5067 超過該參考，不代表機率品質勝出；三個 seed 的 CE 都落後 GBDT 0.999028 等 learned baselines。
- Seed5067 的樣本整體分類比前兩輪好，但市場等權 macro-F1 為 0.312710，反而低於 seed42 的 0.330799、seed347 的 0.345726。US 占 82.30% windows，不能只看樣本整體結果。
- 14d alpha Rank IC：US 為 +0.045486／+0.072947／+0.083138；TWSE 為 −0.063087／−0.037710／+0.031969；TPEX 為 −0.083191／−0.090545／−0.016489。Seed5067 的 TWSE 全 horizon 正排序是有限正面證據，**不能再照抄「所有 seed 台股長天期皆負」**；但 TPEX 11–14 日仍負。
- Seed5067 的 14d neutral recall 僅 6.35%，TPEX 0.18%、TWSE 1.88%；整體改善不能掩蓋漏判。較佳 seed 的 accuracy 也遠未到使用者期待的方向能力。
- 訓練沒有全面改善 early peak：alpha 的最佳點隨 seed 後移到 0.2／0.6／1.0 epoch，但分類仍在 0.4／0.4／0.6 epoch；不能把「後移」當作泛化退化根因已解決。
- Seed42 micro-batch 為 32×8 accumulation；347／5067 為 64×4。有效 batch 均 256，仍不是控制全部執行條件的純 seed 消融。Baseline learned results 固定 seed42，也不是每個主模型 seed 的同 seed baseline 重訓。

#### 5.6 A9 date-mixing 單因子對照

R24 與 R21 使用相同 seed、資料、標籤、loss、模型架構與 baseline；唯一數值差異是訓練順序。兩個子 run 的 45 個 manifest 數值檔均與 `31ef7c1` 相符；與 `c7b5550` 相比只差 `config.py`、`training.py`、`date_market_sampler.py`。RTX 5090 四次建立請求被拒，實際使用 RTX 4090；自動 batch 規劃仍為 32×8、有效 batch 256，與 R21 相同。判準事先固定於[實驗協定 13.1](docs/experiment_protocol.md)，只以 validation 決策。[S11]

| ID | 父 run ID／seed | Alpha 子 run／最佳 checkpoint | 分類子 run／最佳 checkpoint | 狀態 |
| --- | --- | --- | --- | --- |
| R24 | `run-20261016T165022Z-32d734a0cb644e44`／42 | `run-20261016T165022Z-32d734a0cb644e44-alpha`／`checkpoint-007360`，第 2 次／8 次 validation，約 0.4 epoch | `run-20261016T165022Z-32d734a0cb644e44-direction`／`checkpoint-003680`，第 1 次／7 次，約 0.2 epoch | `TEST_COMPLETE`；兩者正常 early stop；**validation 判準未通過** |

**Batch 組成**（`summary.json` 的 `training_batch_composition`，前 256 個 optimizer steps）：未混洗參考順序每次更新的相異日期×市場數中位數為 2（每個 micro-batch 為 1）；混洗後為 166（每個 micro-batch 為 30）。這是以真實訓練索引確認的歷史 batch 組成，不是推論。

**Validation（決策依據）：**

| 指標 | R24 datemix | R21 seed42 | R21–R23 三 seed 範圍 |
| --- | ---: | ---: | ---: |
| Alpha 最佳 normalized Smooth L1 ↓ | 0.46363 | 0.45476 | 0.45311–0.45476 |
| Alpha 最後一次 validation ↓ | 0.47878 | 0.45883 | 0.45883–0.46297 |
| 分類最佳 CE ↓ | 1.02728 | 1.02844 | 1.01311–1.02844 |
| 分類最佳點 12／14 日 ECE ↓ | 0.1026／0.1380 | 0.0949／0.1037 | — |
| 分類最佳點 market-macro CE ↓／macro-F1 ↑ | 1.0430／0.3299 | 1.0197／0.3592 | — |
| Alpha 最佳點美股 14 日 Rank IC ↑ | 0.0161 | 0.0048 | 0.0048–0.0413 |

Alpha 判準（最佳值低於 0.45476 且不是首次評估）與分類判準（CE、10–14 日 ECE、market-macro CE／F1 同時不劣化）皆未成立。鄰近 train batch loss 在混洗下降得更快：alpha 約 1.4–1.6 epoch 時約 0.396，R21 約 0.42；分類約 1.2 epoch 時約 0.864，R21 約 0.894。這是鄰近 batch loss，不是全量 train loss 或無偏的泛化差距，但方向與 validation 退化一致。

**Test（只記錄，不用於選擇方案）：**

| Run | Alpha normalized Smooth L1 ↓ | Alpha MAE ↓ | Alpha Pearson ↑ | Alpha 符號一致率 ↑ | 分類 CE ↓ | 分類 accuracy ↑ | Balanced accuracy ↑ | Macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R24 datemix 42 | 0.535295 | 0.061161 | 0.095170 | 52.67% | 1.118239 | 42.84% | 36.51% | 0.300239 |

- 逐日配對 loss 差（主模型減 baseline，負值有利主模型；14 日 block bootstrap、1,000 次）：alpha 對 GRU +0.006578［−0.002895, +0.016329］、對 GBDT +0.003796［−0.008078, +0.015085］；分類對 GBDT +0.116751［+0.053923, +0.183887］、對 GRU +0.115948［+0.045455, +0.191853］。分類 CE 明顯劣於 learned baselines。
- 分類器 10／14 日預測「跌」占 90.2%／91.8%，真實為 42.7%；ECE 0.212／0.227；逐日 CE 標準差 0.152，R21 為 0.087。
- Alpha 14 日 test Rank IC：US +0.0718、TWSE −0.0385、TPEX −0.0737；美股落在歷史 seed 範圍內，台股仍為負。

**解讀（推論，待驗證）：** 結果與原假說相反。混洗後模型更有效率地擬合訓練期，但 validation 變差；退化集中在點估計與市場方向機率等日期層級共同成分，選中 checkpoint 的橫斷面排序沒有明顯變差。同一 micro-batch 內 benchmark 完全相同時，日期共同成分的梯度在相鄰 step 間方向不一致而互相抵消，歷史同日 batch 因此可能對訓練期日期／市場狀態的擬合有隱性抑制作用；混洗讓這類模式變成一致、可學的訊號，但不轉移到新期間。依協定不補跑其他 datemix seed，也不以 R24 test 選擇方案；`date_mixing_buffer_samples` 預設 `null`，歷史 preset 的訓練順序不受影響。

### 6 提前終止與失敗的實驗帳本

#### 6.1 因 validation 已顯示品質差而直接停訓

**Q01 — `run-20261012T211457Z-09c0c46c20444122`，A-direction64-v3、seed42：`STOPPED_POOR_VALIDATION`。**

這輪在完整 validation 上已看出品質不佳、後續反彈與方向偏向，使用者明確要求「直接終止 pod，不要再浪費 GPU 運算費用」，因此未等訓練自然結束就停止。不能稱為正常 early-stop 完成，不能補造最終 test，也不能列入 completed alpha leaderboard。該 run 的 42 個已記錄數值程式檔與 `346cae0` 相符。[S7、S9]

當時對話保存的四次完整 validation 診斷如下，每次 528,315 windows；此表是當時診斷紀錄，不是本次重新執行的評估：

| Checkpoint | 約 epoch | 5d q50 alpha 符號一致率 | 5d q50 correlation | 該版方向 selection loss |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-003680` | 0.2 | 52.77% | 0.08829 | 1.10185 |
| `checkpoint-007360` | 0.4 | 55.43% | 0.09397 | **1.05147** |
| `checkpoint-011040` | 0.6 | 54.98% | 0.09188 | 1.05635 |
| `checkpoint-014720` | 0.8 | 53.74% | 0.08726 | 1.08340 |

在最佳點之後退化；最新 alpha 分類 down 預測占比約 TWSE 96.25%、TPEX 96.43%、US 92.11%。對 train prior 的機率 skill 很小，最新 absolute skill 轉負；共享 LoRA 梯度有限、有 PCGrad 衝突投影只證明訓練路徑有更新，沒有證明更新能泛化。

Pod `cn0bxripyi5mp8` 的停止有當時刪除／不存在核對；後來使用者授權清理未完成 runs，該 run 也在 2026-10-15 cleanup 的 retired 清單內。因此今日不能假設 network volume／W&B 還留著原 checkpoint；主要可攜證據是這份沿革、Git 架構、當時對話診斷與本機 cleanup 前 metadata。**這一類因品質主動停訓的紀錄不可再因 run 清單清理而從研究沿革消失。**

#### 6.2 硬體／程式失敗／誤用 preset 必須分開

| ID | Run ID | 原因與可證實範圍 | 對應 commit／後續 |
| --- | --- | --- | --- |
| F01 | `run-20260903T134249Z-105084056` | `INCOMPLETE_UNRESOLVED`；僅留下 W&B failed 等 metadata，無完成模型；不足以判定是品質差而停 | 未建立數值快照匹配 |
| F02 | `run-20260922T074951Z-141376769` | `INCOMPLETE_UNRESOLVED`；當時主模型吞吐調查與重啟期間的 A 組嘗試，無完成 test；不能歸因為模型品質 | 31 檔與 `21990dc` 相符 |
| F03 | `run-20260922T145223Z-218655310` | `INCOMPLETE_UNRESOLVED`；同輪效能修復期間的 A 組嘗試，原始單一停止原因未完整保存 | 31 檔與 `e5d92c9` 相符；後續正式完成為 R07 |
| F04 | `run-20261007T070237Z-767e3100b05146c8` | `FAILED_RUNTIME`；A-LoRA32，lifecycle exit1，無模型初始化／最終 test | 大 batch／記憶體與 worker 清理問題修復 `39cef62`；成功重跑 R09 |
| F05 | `run-20261007T070239Z-2dd9f4ffc9634620` | `FAILED_RUNTIME`；A-LoRA64，exit1，非預測效能結論 | 同上；成功重跑 R10 |
| F06 | `run-20261007T070243Z-8c691a7c9ba14a27` | `FAILED_RUNTIME`；A-partial，exit1，非品質停訓 | 同上；成功重跑 R11 |
| F07 | `run-20261011T184451Z-956d8946164f40b3` | `STOPPED_INFRA`；V2 seed42 的 RTX PRO 6000 WK 持續 SW Power Cap；43 分鐘後按要求停止，僅約 step368／94,208 samples，未找到 checkpoint，沒有 validation／test 結論 | 40 檔與 `f6271e7` 相符；完整 incident 已保存；後續 R18 完成 [S8] |
| F08 | `run-20261014T090108Z-338fadabfeb848f4` | `STOPPED_WRONG_PRESET`；本應跑新任務，卻是舊 `a-direction64`、seed147；使用者要求終止所有 Pods；未完成 | 42 檔與 `7d4130c` 相符；成對導引與流程修復 `8bae2db` |

F02／F03 的效能問題是對話背景，保存 metadata 沒有足夠的精確最後錯誤／停止標記，不應寫成已證實單一根因。F04–F06 可證實 failed；`39cef62` 是該批失敗後的修復提交，不是三輪執行時已被證實的完整 Git HEAD。

另有多次 baseline／驗證嘗試，非正式模型品質實驗：`baseline-run-20260920T124808Z-20921`、`baseline-run-20260920T152758Z-32557`、`baseline-run-20260920T175140Z-32301`、`baseline-run-20260921T032356Z-2164`、`baseline-run-20260921T043022Z-3705`、`baseline-run-20260921T061805Z-16783`、`baseline-run-20260921T073833Z-10520`、`baseline-run-20261010T052256Z-15323`。它們列在歷史 retired attempts；不能全部稱為 validation 品質失敗。最後一筆關聯成本比例超過 notional 的失敗 baseline cache `baseline-0cb2d490fa3d450dbfdaa208fc90e3c4355f8b27a9a61e2354f48c859921a497`，修復為 `5a55668`。[S7]

### 7 模型主線之外但影響可重現性的工程沿革

| 問題 | 主要提交 | 交接時要保留的能力／教訓 |
| --- | --- | --- |
| Baseline 每次重訓、20k 上限與評估抽樣 | `a51da94`、`8e7f5d1`、`44e7139` | 前置完整 baseline，保存權重／結果，主模型 test 只讀相同契約的快取；training 仍動態 sampling |
| Baseline CPU bottleneck、GPU 利用率／續訓 | `9ebff63`、`9e1af2c`、`49e68b4`、`bea8614` | 有界向量化、硬體自調 batch／prefetch、durable cursor；工程吞吐改善不是準確率改善 |
| Baseline 錯綁主模型／共用 manifest | `7d7bcd5`、`556c0d8` | 依 baseline 數值參數／資料規則與來源追溯，不因主模型名稱或無關腳本改動重訓；A/B 必須均可切換 |
| 專案與 Pod 命名回歸 | `eae18bc`、`900000c` | 新資源使用 stock_forecasting／stock-forecasting，不搬舊結果、不變數值身分；README 不寫內部對話 |
| 主模型吞吐、過於保守的 batch 與 hot path | `f66af29`、`e5d92c9`、`feff226` | 實測 compute／input wait／記憶體，不能只憑 VRAM 占比判定利用率；不干擾執行中訓練 |
| membership／targets 在 baseline 結果重複存放 | `f2cdacb`、`0c6b627` | 共用評估資料，避免每份模型結果重複儲存；testing 讀取路徑與 resume 都要覆蓋驗收 |
| Data cleaning 更改但 readiness 仍顯示舊候選筆數 | `736f1ae`、`adb521a` | 同時保留 current-rule 檢查及 baseline 完成檢查，不能修一個拔掉另一個 |
| Training／validation loss 診斷不足 | `3ddaad7`；現行獨立任務 `bc84d49` | 保存所有 validation 點與各任務 train/loss、validation/loss；鄰近 train batch loss 不是全量 train loss，不能直接當無偏泛化差距 |
| 搶 GPU 時重複 readiness、錯誤訊息失真 | `fc98524` | 保留 provider 詳細錯誤，重用有效 readiness；不能讓控制面檢查浪費搶卡窗口 |
| 多 Pod／seed 身分、查 run 不清楚 | `5e18f88`、`838d8bf`、`ead2351` | run-scoped 狀態／鎖／guard、不可變 seed/config；查詢需列 run ID、設定、時間與真正完成度 |
| GPU 時脈與 orphan guards | `90acf1d`、`39f337e`、`3ead8d8`、`8bae2db` | 本機 guard 監測持續低時脈；以監測時段超過一半時間低於運作基準 80% 才警示；終止後身分限定清理，不殺其他 run |
| 新任務拆成兩個需手動啟動／Pod 未建就寫 run JSON | `8bae2db` | 父 run 一次管理兩個獨立子模型；先成功建立 Pod 再發布 run 紀錄；接續略過已完成子模型 |
| 成對 child selection 與 Python3.12 路由錯誤 | `3b7cccb` | 不能用修改 SHA 例外繞過真正身分錯誤；測試需覆蓋實際 mounted selection 與 CLI 參數順序 |
| 現行報告與舊架構混淆 | `7d4130c`、`8d43d03`、`9582978` | 現行與歷史分區；保留歷史 CLI、架構圖、人工特徵細節與歸檔，不能將舊標籤套進新報告 |

版本 tag：`v0.1.0` → `fb8482e`；`v0.2.0` → `44e7139`；`v0.2.1` → `c03e416`；`v0.2.2` → `bea8614`。這些是 tag 解引用後的 commit，不是 annotated tag 物件 SHA；tag 不表示後續正式模型有效。

W&B 另有尚未完成的需求／風險需要交接：使用者詢問只保留超參數與訓練指標、不存 artifacts，這是討論，不是已實作。編寫時現行 preset 的 `log_model_artifact` 仍為 `true`，最終 benchmark 另有 artifact 上傳；不能宣稱已全面關閉。`tracking.py` 的 backend log／summary／artifact／finish 例外會記錄狀態後重新 raise，因此也不能保證配額或上傳錯誤永遠不影響訓練。此項應獨立評估非致命追蹤與本地保存契約，不是本次已修復的功能；本次不刪 W&B 資料，也不改其設定。

### 8 現行架構的精確交接規格

```text
同資料組別＋seed → 一個父 run／一台 Pod
  ├─ alpha 子模型：獨立 Kronos-base／LoRA64 → alpha_point[batch,14]
  │    normalized Smooth L1 → 自己的 optimizer／scheduler／checkpoint／early stop
  └─ direction 子模型：另一套獨立 Kronos-base／LoRA64 → logits[batch,14,3]
       固定 ±1% 個股標籤 CE → 自己的 optimizer／scheduler／checkpoint／early stop
```

每套內部：個股／benchmark 各 128 日 PIT OHLCV＋推導 amount → 視窗標準化 → 固定 tokenizer → Kronos／LoRA → resampler → benchmark conditioning → mean pooling＋market/horizon embeddings → 任務 head。20 維歷史尺度統計與 benchmark latent 直接接數值 residual。Tokenizer 固定；無 partial unfreeze。

- A train：2021-01-01 起、2025-06-01 前，4,710,117 有效 windows；validation `[2025-06-01,2025-12-01)`，528,315；test `[2025-12-01,2026-06-01)`，534,327。Label 的最晚日期必須在各 split 右界前，train／validation／test 不共用未來真值。
- 預測在 t 收盤後形成，下一個市場交易日 open 為 entry，該日算持有第1日，第h個市場交易日 close 為 exit；不能因個股缺資料就延後 entry／exit。US benchmark 為 `VTI.US`，TWSE 為 `TAIEX.TW`，TPEx 為 `TPEX.TWO`。
- Alpha label：同 entry／exit 的個股 total-return log return 減 benchmark total-return log return。不是 CAPM residual，也不是 simple-return 百分點差。
- Alpha loss：`SmoothL1((prediction-target)/s_h, beta=1)`；`s_h=max(train IQR,1.4826×train MAD,1e-4)`。只用 train 校準，50k samples、seed59；不是 q50 或保證的條件平均。
- 分類：個股絕對 simple return `<−1% / [−1%,+1%] / >+1%`；在 log 座標用 `log1p(−0.01)`／`log1p(0.01)`。順序 down／neutral／up，CE 讀 logits，不是 BCE，不扣成本。
- `direction_agreement` 只指 alpha 的符號一致率，不是分類 accuracy。正 alpha 可以發生在股票下跌但跌得比 benchmark 少的情況。
- `primary_5d/selection_score` 是相容鍵，對這兩任務分別為全 1–14H 平均 normalized Smooth L1／CE；其他 `primary_5d` 欄位仍是 5d。`comparison.rows` 也混有全 H selection loss 與 5d 品質欄位，不能一律當全 H。
- 預設 LoRA rank64／alpha128、dropout0.05；head LR3e-5、LoRA LR5e-6、weight decay0.01。最多5 epochs，每 epoch5次完整 validation，patience5次；plateau2次未改善 LR×0.3，最低ratio0.09，再經2個驗證間隔才允許 early stop。
- 最終 test 用 validation 最佳 checkpoint。`completion-result/` 是停止當下權重，不保證等同最佳 checkpoint；不能拿最後權重解釋另一份 test。
- **未啟用** quantiles、interval calibration、獨立 ranking、雙方向／alpha classifier、十技術通道、horizon-query、prior、market-state gating、PCGrad、表徵擾動與成本特徵。Config 存在舊預設欄位不等於啟用。

現行基準 cache：

```text
alpha_point:
baseline-3befc8e902e5c641b24ddf80c4ab7e11769529fa307e391de47e4416f1d885a3
absolute_direction:
baseline-b21f46ea4963562f03c648e9b0394f8a069767bca2a15a9f82e7c8ed69e04cc8
```

Baseline 包含 always_buy、zero_return、momentum_5d、reversal_5d、ma_crossover、rsi、macd、volatility_scaled、GBDT、GRU、DLinear、PatchTST。神經 alpha baseline 用相同 normalized Smooth L1；GBDT fitting 用 MAE、選模用相同評估 loss。分類模型用固定標籤 CE／log loss。固定規則沒有可訓練 epoch／early stop；其分類 logit 0／−30 不是校準過的機率模型。

### 9 未解決問題與交接後的研究順序

本節是依既有結果整理的 **候選工作與驗收要求，不是已實作或已授權的新改版**。不能再把一次加入很多 feature 後的結果，歸因於其中單一 feature。

| 優先 | 待解問題／候選工作 | 既有證據 | 最低驗收要求 |
| --- | --- | --- | --- |
| 完成 | **日期層級與同日橫斷面的誤差分解（不需 GPU）**：見[誤差拆解報告](docs/error_decomposition_audit.md) | Alpha 日期層級只占標準化 MSE 2.6%–3.8%（上市例外）；loss 由尾端樣本主導；分類日期層級資訊 5.4%–5.9% CE，但 baseline 擇時皆為負，可學部分是同日鑑別力（+0.023–0.029 CE） | 已用 baseline 記錄值核對到 5×10⁻⁷；主模型逐樣本預測未保存，主模型只做逐日輔助分析 |
| 完成 | **保存逐樣本預測並對 R21–R24 checkpoint 重跑 validation／test（GPU 推論）**：含依[實驗協定 §13.2](docs/experiment_protocol.md) 重選的 R21／R22 alpha checkpoint；結果見[誤差拆解報告第 7 節](docs/error_decomposition_audit.md) | 匯出與訓練紀錄完全一致（R21–R23 差距 0）。分類：Kronos 同日鑑別力最佳，落後來自擇時；移除擇時後 validation 1.0034–1.0065（GBDT 1.0047）、test 0.986（GBDT 0.990）。Alpha：Kronos 同日相關 0.10–0.12，US Rank IC 延續到 test，台股對所有模型都崩落。三 seed 集成在 loss 類指標最好 | 10 個 checkpoint，RTX 5090 約 3.5 小時；匯出 Pod 因 lifecycle bug 多閒置約 4.4 小時，已於 `e8d29a8` 修正 |
| 完成 | **第 1 階段：台股免費資料的方向資訊測試（只用 CPU）**：見[實驗協定 §13.3](docs/experiment_protocol.md) | 未通過：8 個特徵組都沒有在兩個以上的決策天期通過；最佳 `all_pre_open` 1 日比先驗改善 1.34%，相對 `F0` 的區間跨 0。TAIFEX 下載失敗（免費資料只有近 3 年） | 事先固定判準見 §13.3；結果檔見 S14 |
| 完成 | **回測第一階段：固定持有天數（本機）**：見[實驗協定 §13.4](docs/experiment_protocol.md) 與[回測結果報告](docs/backtest_stage1_results.md) | 三個市場未通過，test 年化淨超額 −37% 至 −44%；1 日毛優勢被成本吃掉；長天期 test 尾端反轉；大型權值股主導期間，指數避險版全部為負 | 事先登記 `5ddeff6`、程式 `a292a3b`；輸出見 S15 |
| 完成 | **回測 1b：持有緩衝與逐部位記帳（本機）**：見[實驗協定 §13.5](docs/experiment_protocol.md) 與[回測結果報告](docs/backtest_stage1_results.md) | 三個市場未通過，test −13% 至 −40%（探索性）；前 K 名優勢只在交易時段、隔夜反轉；降低換手省下約四分之三的成本，但沒有毛優勢可保留 | 事先登記 `c4ee514`、程式 `4a5bdbb`；依協定停止降低換手路線 |
| 完成 | **A：與持有方式對齊的評估（本機）**：見[實驗協定 §13.6](docs/experiment_protocol.md) 與[評估結果報告](docs/holding_eval_results.md) | 多數訊號的前十分位在隔夜大幅落後；以 Rank IC-IR 重選的 checkpoint 與 R23 隔夜拖累小，上市 10 日 R22-IR 前十分位在 validation（+69.6 bps）與 test（+103.2 bps）都為正；上櫃幾乎全為負。只作描述 | 事先登記 `3949fb6`、程式 `7e7d925` |
| 完成 | **第三階段：前瞻檢驗（2026-06-01 之後的未見資料）**：見[實驗協定 §13.8 與修訂、修訂補充](docs/experiment_protocol.md)與[前瞻檢驗結果報告](docs/stage3_forward_results.md) | 所有資料與一致性檢查通過；主要假說（R22-IR、上市、10 日）未通過：前十分位 +57.1 bps [−10.5, +124.3]，固定組合年化淨超額 +7.4% [−36.6%, +56.0%]。依協定不採用，回到模型本身的改善。上市 6 個訊號 Rank IC 都為正（0.074–0.149），但沒有一個前後差或上市組合的區間排除 0；選點在前瞻期間沒有顯現優勢；上櫃仍偏弱 | 事先登記 `e4d823c`、`c19bedc`、`5afcae9`；程式 `65bba60`、`b7d8561`、`cc57971`、`41278bc`；輸出見 S17 |
| 完成 | **P2：以持有指標選 checkpoint（推論在 RunPod，評分在本機）**：見[實驗協定 §13.7 與修訂](docs/experiment_protocol.md)與[選點結果報告](docs/checkpoint_selection_results.md) | 15 份匯出全部通過完整性檢查；R21–R23 都選中 checkpoint-018399，R22 選中的就是 R22-IR，§13.8 主要假說不變。選出者的上市 10 日 test 前十分位：R22 +103.2、R23 +86.4、R21 −63.9 bps；test 前、後十分位都高於等權平均，前後差為負。只作描述 | 事先登記 `e4d823c`、`1197b60`；程式 `2d84ff9`、`e93705c`；輸出見 S16 |
| 完成 | **橫斷面排序任務 `cross_sectional_rank`：rank 標籤與 baseline**：見[實驗協定 §13.10](docs/experiment_protocol.md) | 只有同日同市場的相對排序有證據（Kronos alpha 同日相關 validation 0.099–0.119、test 0.091–0.124）；數值 alpha 與絕對方向都沒有穩健勝過 baseline。標籤區間不變（t+1 開盤到第 h 日收盤），進出場規則由 hermes agent 回測決定 | baseline 已完成（`baseline-5fb3473a…`，結果見 §13.10）；Kronos 試行見下一列 |
| 完成 | **Kronos 排序模型 `a-rank64` 試行（訓練與匯出在 RunPod，計分在本機）**：見[實驗協定 §13.11](docs/experiment_protocol.md) 與[排序模型試行結果報告](docs/rank_pilot_results.md) | H1 通過：台股 1 日 0.064 [0.027, 0.102]；H2 未通過：Kronos 減 PatchTST −0.002 [−0.021, 0.018]；H3 未通過：台股 6–14 日 0.008 [−0.044, 0.064]。Kronos 與 PatchTST 在 test 六格的配對差都跨 0；台股長天期在 test 對所有學習型模型失效，均線交叉 0.063 | 事先登記 `64df916`；程式 `f5bdf1d`、`aad2d63`；輸出見 S18。依協定不加跑 seed；2026-10-22 決定主線改為台股跨股票 DL（下一列），Kronos 不再是 backbone，日期混洗不再登記 |
| 完成 | **台股主線：特徵 LightGBM 基線 B0、跨股票 DL D0、消融 A1、Top-k dropout 回測**：見[實驗協定 §13.12、§13.13](docs/experiment_protocol.md)、[LightGBM 基線報告](docs/lgbm_baseline_results.md)與[跨股票報告](docs/cross_stock_results.md) | B0 短格 0.118、D0 集成 0.122；H-A 只有短格通過（+0.0043 [+0.0015, +0.0071]）；H-B 未通過（Top-k dropout 成本拖累 17%／年）；A1 顯示股票間 attention 無增量；P2 多日 horizon 崩掉 | 事先登記；三 seed；A2–A4 延後後取消（D0 的優勢來自集成，架構消融無進一步資訊） |
| 完成 | **策略層：多頭 band 組合與台指期疊加（§13.14）、股票期貨空方當沖（§13.15）**：見[策略層報告](docs/strategy_layer_results.md)與[空方當沖報告](docs/futures_daytrade_results.md) | H-D 未通過（P1 ∪ P2 淨超額 −4.7%／−5.4%，所有資本情境）；H-D-hedged 未通過（追繳 27–28 次）；H-E 未通過（毛 +61–65%、淨 −60%，tick 滑價 0.23%／邊，損益兩平 0.12%） | 事先登記；期交所官方資料；不再調參重跑 |
| 完成 | **殘差化標籤與可抽換編碼器（§13.16）**：見[殘差化標籤報告](docs/residual_labels_results.md) | H-F 通過（變體 A：P2 持有格殘差目標 0.0646 [0.0457, 0.0856]，h = 1 不受損）；變體 B 劣於 A；D0 新標籤 H-A 仍只短格通過；Kronos-small = 單 seed D0；Chronos-2 略差；seed 穩定 | 模型線收斂：B0 基線、D0 集成候選；D1 與 kronos-base-lora 不再投入 |
| 停止 | **暴露中性策略：0050 核心加股票期貨多空疊加（§13.17）**：見[步驟 0 報告](docs/exposure_diagnostics_results.md) | 步驟 0 停止規則觸發：股票期貨宇宙 h = 10 產業中性多空毛年化 −0.6%／+0.4% 對成本拖累 18.3%；該宇宙 P2 殘差 Rank IC 為 0；流動性宇宙 P1 殘差四分位價差每 10 日 0.9–1.0% | S1／S2／S3 不實作；結論為沒有勝過 0050 買進持有的可部署策略；若重啟需新資料（P3）或新訊號來源，不得調參 |
| 完成 | **機構成本下的六情境回測（§13.19，資訊性）**：見[結果報告](docs/institution_cost_scenarios_results.md)與 Word 版 `reports/institution_cost_scenarios_20261025.docx` | 36 格無一達標；最接近 D0 S2（多日多空加台指期）TW：淨 +16.8% [+1.4%, +33.7%]、Sharpe 0.83、P2 −6.5%；S1 的 P2 空頭腿 −61% 到 −89%；當沖機構淨 TWSE −17% 到 −33%、TPEX +7–12%（區間跨 0）；散戶當沖拖累 263–363%／年 | 事先登記；不觸發前瞻；結論為「機構成本下仍無對業界有吸引力的策略」；瓶頸在空頭腿建構、當沖執行成本與容量 |
| 取消 | **B：GBDT 改用開盤到開盤標籤，以及以持有指標選 checkpoint** | 2026-10-20 取消：原設計的標籤區間本來就是 t+1 開盤到第 h 日收盤，出場時點由 hermes agent 決定，不需要改成開盤到開盤；改由排序任務取代 | — |
| 擱置 | **分類器推論期的同日先驗正規化（不需 GPU 訓練）**：做成正式的推論與評估選項。2026-10-18 擱置：移除擇時也移除了交易規則需要的方向判斷 | 移除擇時是無標籤、可部署的後處理；validation 追平 GBDT，market-macro 三個 seed 都優於 GBDT；test 優於 GBDT | 先在協定寫好 validation 驗收判準；對所有模型一致套用；同一日期與市場的股票須一起推論；需實作授權 |
| 擱置 | **訓練時的日期固定效果**（分類優先）：設計草稿與事先固定的判準見 [P1 設計](docs/p1_date_market_logit_adjustment_design.md)。2026-10-18 擱置，理由同上 | **S0 成立**：移除擇時讓 R21–R23 validation CE 平均改善 0.0146（門檻 0.005）；參照組數值已量測（誤差拆解報告 §7.6） | 單因子新 preset；輸出格式不變；三 seed、事先固定 validation 判準；必須勝過免費的推論期正規化（判準 C1） |
| P2 | 同日去均值的 alpha 目標（降級） | Alpha 日期層級成分小，預期改善有限；上市市場例外 | 只作次要研究，不視為 alpha 主要解法 |
| P0 | 先確認模型實際從哪些路徑取得增益：Kronos latent、尺度分支、benchmark、market/horizon embeddings 的容量匹配消融；必要時以同訓練預算比較凍結 readout／LoRA | Signal alpha 僅 seed42 微幅領先 GRU；多次複雜化失敗 | 同一有效集合與 train-only scales；固定 batch執行條件；不能為做消融刪除原 feature；不預設 Kronos 無效 |
| P0 | 針對 early peak／validation反彈研究正則化、更新量、取樣分布與 learning curves；查每市場、年份、類別的梯度／誤差 | 現行分類仍在0.4–0.6epoch最佳；train batch loss下降而full val無相同改善；**R24：混洗日期使 train 擬合加快、validation 變差，取樣不是越 iid 越好** | 保存所有 validation 點與明確 loss 定義；同分布的 train診斷不得冒充已存在；不能只增加驗證頻率或延長訓練宣稱解決 |
| P1 | 固定±1%三分類的 horizon／市場不平衡與neutral漏判；先稽核support、預測機率、decision rule與loss，再提出最小變更 | 14d台股neutral recall極低；pooled與market-macro結論相反 | 只用train/validation定案；不把成本帶或alpha classifier自行加回；保留per-market/per-H confusion與CE／F1 |
| P1 | 數值幅度、alpha sign、每日橫截面排序分開處理；檢查台股長H負RankIC與seed敏感性 | 5067排序改善但alpha loss不最佳；不同市場有反向結果。**P0**：台股 validation Rank IC 在 test 對所有模型崩落（多個 Kronos TPEX 為負），US 可延續；loss 選模在 seed 42 選到 validation US IC 只有 0.017 的點 | 保存可重算逐日IC／預測證據；核對ties方法、有效日期，不以單一總分遮掩市場退化；正式改用 Rank IC 選模前，先做各年、各市場的 train Rank IC 診斷 |
| P1 | 修復證據保存缺口與可攜性：輕量逐日／分組證據、精確版本關聯及停止原因 | 舊未完成runs已清理；本機主要是彙總JSON；部分baseline只有seed42 | 不預設永久保存巨大逐筆資料；設定容量上限；清理前保留小型研究帳本；不能要求重抓行情補metadata |
| P2 | 如要重測長歷史，用清理後共用評估的B模型，或先以既有bar-store預先固定rolling windows | 舊A/B受不同membership／尺度影響；現行只有A的三seed完成證據 | 所有跨度變更須列清train/validation/test與尺度；不以既有test反覆選方案；付費工作另授權 |
| P2 | 容量仍可研究LoRA，partial作次要對照；暫不把rank128／Kronos-large當保證解法 | LoRA64在舊組較好；partial失敗，Adaptive128僅定義 | 清楚區分已定義／已跑；控制訓練預算與多seed；不再只報工程驗收數量 |

**不得再次直接複製的失敗路徑：** 為提高coverage而偏離方向問題；把成本／交易規則揉入預測標籤；只看accuracy掩蓋類別塌縮；以PCGrad有梯度或CUDA smoke通過當品質提升；只換seed選好看的test；把B較大但少數指標較差的現象直接斷言為容量不足；假設跨日期混洗的 batch 必然改善泛化（R24 已反證）；以 1 日開盤到收盤的 Rank IC 推論多日持有的可交易性（回測 1b 已反證）。

### 10 證據路徑與接手檢查表

#### 10.1 主要來源

- **S1** [現行 seed42 報告](reports/training_result_analysis_run-20261015T064558Z-f04e155d4299416f.docx)。
- **S2** [現行 seed347 報告](reports/training_result_analysis_run-20261015T064559Z-624f212a27894308.docx)。
- **S3** [現行 seed5067 報告](reports/training_result_analysis_run-20261015T064600Z-13b8b4376ea5413b.docx)。
- **S4** [早期報告索引](reports/deprecated/README.md)：R02–R08 在該目錄本層。
- **S5** [改版前報告分類索引](reports/deprecated/pre-a-signal64-20261016/README.md) 與 [歸檔 manifest](reports/deprecated/pre-a-signal64-20261016/archive-manifest.json)：R09–R18 的修訂後報告與原始 bytes/hash。
- **S6** [主要歷史尺度診斷](artifacts/diagnostics/representation-scales/run-20260905T203327Z-270337978/checkpoint-018145/probe-20260914T101527Z-105a4d66/summary.md)；同目錄 `report.json` 是逐目標／市場數值。
- **S7** 本機已保存的清理前 snapshots：`.runpod/storage-audit/20261015/metadata-before.json`、`runs-before.json`、`authorized-cleanup-plan.json`、`cleanup-result.json`。這是歷史證據，不是目前線上檔案仍存在的保證。
- **S8** [V2 GPU降頻 incident](artifacts/runpod_incidents/20261012T032735+0800-9apjlhs763hd72/INCIDENT_REPORT.txt) 與同目錄 `incident-summary.json`。
- **S9** 本對話及當時保存的V3診斷／授權停訓紀錄；run身分由S7再核對，數值屬當時validation診斷，無今日重評估或holdout。
- **S10** [README現行架構及歷史附錄](README.md)、[架構文件](docs/architecture.md)、[實驗協定](docs/experiment_protocol.md)、[版本tag說明](RELEASES.md)；若與run不一致，以該run的保存設定／程式身分解讀歷史。
- **S11** R24 產物：分支 `claude/model-performance-20261016` worktree 的 `artifacts/runpod/run-20261016T165022Z-32d734a0cb644e44/`（`pair.json`、`download-result.json`），以及 `-alpha`／`-direction` 子目錄的 manifest、resolved config、`metrics.jsonl`、`summary.json`、leaderboard 與最終 `validation-benchmark.json`。`artifacts/` 不受 Git 追蹤，移交時需另行攜帶。
- **S12** [誤差拆解報告](docs/error_decomposition_audit.md)，以及本分支 worktree 的 `artifacts/audit/`（`decomposition_audit.py`、`decomposition-results.json`、下載的 baseline 評估陣列）；同樣不受 Git 追蹤。
- **S13** P0 逐樣本匯出與拆解（[誤差拆解報告第 7 節](docs/error_decomposition_audit.md)）：本分支 worktree 的 `artifacts/diagnostics/signal-predictions/`（10 個 checkpoint 的 validation／test 逐樣本預測、`export.json` 與 `split.json`），以及 `artifacts/diagnostics/signal-decomposition/`（`results-p0.json`、`rank-ic-p0.json` 與執行 log）；不受 Git 追蹤。
- **S14** 第 1 階段資訊測試輸出：本分支 worktree 的 `artifacts/diagnostics/tw-information-test/launch-20261018T132744Z-378616789/`（`report.json`、`summary.md`、`auxiliary-manifest.json`）；不受 Git 追蹤。
- **S15** [回測結果報告](docs/backtest_stage1_results.md)，以及本分支 worktree 的 `artifacts/diagnostics/backtest/stage1-20261018T221138Z-a2a9d283/` 與 `stage1b-20261018T224557Z-f9cd85b4/`（`report.json`、`grid.csv`、`summary.md`、1b 的 `daily.csv`）；不受 Git 追蹤。
- **S16** [選點結果報告](docs/checkpoint_selection_results.md)，以及本分支 worktree 的 `artifacts/diagnostics/backtest/selection-20261020T042715Z-2e0ff319/`（`report.json`、`ranking.csv`、`portfolio.csv`、`summary.md`）與 `artifacts/diagnostics/signal-predictions/` 下 15 個 checkpoint 的 2026-10-19 匯出；不受 Git 追蹤。
- **S17** [前瞻檢驗結果報告](docs/stage3_forward_results.md)，以及本分支 worktree 的 `artifacts/forward/2026-10-20/`（輸入包、標籤、資料檢查）、`artifacts/diagnostics/forward-predictions/2026-10-20/`（6 個 checkpoint 的前瞻匯出）與 `artifacts/diagnostics/backtest/stage3-20261020T134953Z-6f781e13/`；不受 Git 追蹤。
- **S18** [排序模型試行結果報告](docs/rank_pilot_results.md)，以及本分支 worktree 的 `artifacts/audit/baselines/baseline-5fb3473add4ee32a10fa563c53757d78bc30a527ef50d803b17360e313212a4d/`（rank baseline 評估陣列、預測與 `complete.json`）、`artifacts/diagnostics/signal-predictions/run-20261021T064937Z-c1cf693ab51744a6/`（checkpoint-003680 的匯出）與 `artifacts/diagnostics/backtest/rank-pilot-20261021T130741Z-937a2631/`；不受 Git 追蹤。

Git 已追蹤文件會隨 repository 移交；`artifacts/` 與 `.runpod/` 通常被忽略，不會因 clone 自動取得。移交其他工具時，至少附本文件、三份最新DOCX與必要run的metadata／metrics；不要把私鑰、`.env`、API token或整個W&B原始環境一起打包。已被清理的run不再要求從已不存在的路徑讀取。

#### 10.2 每個完成 run 的讀取順序

```text
artifacts/runpod/<run-id>/
  run-manifest.json            # Actual model/data identity and numerical source hashes
  resolved-config.yaml         # Actual training seed and enabled settings
  training-completed.json      # Durable completion, not merely a stopped Pod
  metrics.jsonl                # All recorded training and validation points
  summary.json
  best-checkpoint.json
  checkpoint-leaderboard.json
  validation-benchmark.json    # Inspect evaluation_split; filename alone is misleading
  checkpoint-*/trainer-state.json
```

對父signal run，`pair.json`／`download-result.json`在父目錄，實際模型結果在 `-alpha`／`-direction` 子run。JSON品質值以 `models.kronos_full.metrics`為準；baseline可在 `models.<NAME>.metrics`或`seed_results.42`。頂層`seeds=[42]`是評估／baseline設定，不是主模型training seed。

查完成用既有 `bash scripts/runpod_workflow.sh status RUN_ID --output json`；查run用`bash scripts/runpod_workflow.sh runs`。這些是唯讀控制操作；不要為寫報告啟動訓練、sync或重建baseline。先讀目前`--help`與README，歷史CLI標籤不等於模型架構。

#### 10.3 接手時必須逐項確認

- [ ] 正在研究哪一代架構、哪個task，不能只憑V1／V2／V3名稱。
- [ ] 每個數值來自validation還是test、raw還是calibrated、5d還是14H平均。
- [ ] 股票—日期membership、train scales與RankIC方法能否直接比較。
- [ ] Early stop完成、手動品質停訓、infra失敗分開標示；沒有最終test就不補造。
- [ ] Baseline不是每個主模型seed重新訓練；資料全集相同不等於sampling／FLOPs相同。
- [ ] 既有Sharpe／成本後報酬／策略最大回撤有尺度限制，不作本輪品質冠軍依據。
- [ ] 最新三seed有完整結果，5067不再被誤標為未完成；也不把較早的五子run hash核對紀錄當作六子run全部重新驗證。
- [ ] 新改善須先提出可驗證假說、比較項與預算，並取得實作／付費授權；不將這份roadmap當成自動開Pod授權。

### 11 尚缺或不可過度延伸的證據

1. 本地有23份歷代效能DOCX，其中包含同run的修訂前／後版本，不代表23種獨立架構。另有僅在歷史metadata留存的completed runs與多次未完成嘗試。
2. 已匹配的數值程式快照不等於已保存完整runtime Git HEAD；本文件明確分開架構導入、相符快照、報告commit。
3. V3的原始未完成產物後來經授權清理；其validation停訓結論有當時對話與身分紀錄，但不是可在今日直接重算的完整test。
4. 現行基準使用同一個112日期holdout與三seed，尚未證明跨年度／regime泛化。這個區間已反覆用於研究比較，未來不能再將新調整對它的改善描述成完全未見資料的確認性證據。
5. 刪除W&B artifact／失敗run是儲存管理，不等於原實驗沒發生；工程測試通過與高GPU utilization也不等於模型有效。
6. 本文件不改既有授權：專案個人非商業條款見 [LICENSE](LICENSE)，上游Kronos權利與資料供應商條款仍各自適用。

## English

### 1 Purpose and present conclusion

This handoff roadmap combines the development decisions in this conversation, Git history, downloaded experiment artifacts, historical performance reports, and records of unfinished attempts. Its evidence cutoff is 2026-10-16; repository HEAD when compiled was `9582978`. Compilation did not train or evaluate a model, launch a Pod, change the architecture, or delete data.

**A robust cross-market, cross-seed forecasting advantage has not been established.** All three current `a-signal64` parents now have completed alpha and stock-classification final tests. The earlier observation that seed 5067 classification was still running is superseded by the downloaded final artifacts. Alpha seed 42 narrowly leads GRU on the aggregate loss, but its date-paired interval crosses zero. The other two seeds trail the learned alpha baselines. Classification accuracy is approximately 42.36%–44.59%; all three have worse CE than the main learned baselines, with persistent long-horizon Taiwan neutral-class failures.[S1–S3]

**2026-10-17 addendum (branch `claude/model-performance-20261016`):** the A9 date-mixing single-factor control (R24, seed 42) failed its pre-registered validation criteria. Changing only the training order, which raised the median date/market groups per update from 2 to 166, made training loss fall faster, yet the best alpha validation loss of 0.46363 trails the three historical seeds' 0.45311–0.45476. On test, the classifier predicts "down" for about nine tenths of 10–14 day windows, with CE 1.118239. The degradation concentrates in date-level common components (point estimates and market-direction probabilities); cross-sectional ranking at the selected checkpoint did not clearly worsen. See section 5.6 and the follow-up candidates in section 9.[S11]

**Within-date error decomposition (no GPU):** alpha's date-level component is only 2.6%–3.8% of normalized MSE, TWSE excepted; the loss is dominated by the roughly 24%–31% of tail samples in the linear Smooth L1 regime (81%–84% of it). Date-level information could cut classification CE by 5.4%–5.9%, yet GRU/GBDT date-level forecasts are worse than the train prior; the learnable part is within-date discrimination shared by both tasks. See the [error decomposition report](docs/error_decomposition_audit.md).[S12]

**P0 Kronos per-sample decomposition (2026-10-18):** per-sample predictions for 10 R21–R24 checkpoints were exported and decomposed with the same method. Main results:

- **Best within-date discrimination:** Kronos is the best of all models in both tasks and both periods.
- **The classifier loses on timing:** it trails the baselines because of date-level timing. Validation timing skill is −0.009 to −0.024 and test −0.041 to −0.053, against about −0.009 for GBDT.
- **With timing removed it matches or beats the baselines:** after label-free same-day prior normalization, classifier validation CE averages 1.0048 over three seeds, matching GBDT's 1.0047; test CE is 0.986, better than GBDT's 0.990.
- **Alpha:** the US ranking signal carries forward to test; Taiwan collapses on test for every model.
- **P1 start gate S0 passes.**

The main reason training "does not work" is that the objective and evaluation let timing errors hide stock-selection skill, not insufficient model capacity. See [error decomposition report section 7](docs/error_decomposition_audit.md).[S13]

**2026-10-18 change of direction (Phase 1):**

- **Product goal**: model outputs feed trading rules that compute signals after the close, enter
  at the next open and hold for days to two weeks. They need absolute alpha and absolute
  up/flat/down, not ranking alone. The existing labels already enter at the next open
  (`execution_total_return`), so the entry point stays.
- **Further decomposition**: all of the classifiers' gain over the prior comes from "does it move
  beyond ±1%"; "up or down" is no better than the prior at any horizon, for Kronos and GBDT alike.
  Taiwan alpha test Rank IC carries forward only at 1–3 sessions (0.063 at 1 day, −0.043 at
  14 days).
- **Shelved**: the inference-time normalization and the P1 date fixed effects, because both remove
  the direction judgment.
- **Phase 1**: a CPU-only information test in which GBDT checks whether free Taiwan flow, futures
  and US overnight data beat the prior on "up or down". It ran on 2026-10-18 and did not pass (see
  the 2026-10-19 addendum below). Criteria are in [experiment protocol §13.3](docs/experiment_protocol.md);
  operation is in the README's "Taiwan information test".[S13]

**2026-10-19 addendum (Phase 1 result and backtests):**

- **The Phase 1 information test did not pass** (`information_found=false`): none of the 8 added
  feature sets passed on two or more decision horizons. The best, `all_pre_open` at 1 session,
  improved "up or down" CE on the prior by 1.34%, but its 95% interval against `F0` crosses 0;
  `us_pre_open` at 1 session improved by 0.94%. TAIFEX futures and put/call downloads failed
  because the free download covers only the last three years.[S14]
- **Backtest stage 1 and stage 1b both failed** ([backtest results report](docs/backtest_stage1_results.md)):
  - Fixed holding periods ([§13.4](docs/experiment_protocol.md)): the 10-session configurations
    chosen on validation lost 37% to 44% a year in net excess on test.
  - Hold buffer with per-position accounting ([§13.5](docs/experiment_protocol.md)): test −13% to
    −40% (exploratory).
  - The top-K edge exists only during sessions: from the next session's open to its close the top
    names lead equal weight by about 20–31 bps on TWSE, then trail by about 5–30 bps from that
    close to the following open. Holding overnight cancels the edge; holding only during the
    session is a day trade whose 0.52% round trip exceeds the edge.
  - Even when overall long-horizon Rank IC is positive, the test top K can invert; configurations
    chosen on validation reversed on test.
  - As pre-registered, the lower-turnover route stops and work returns to the models.[S15]

**2026-10-20 addendum (stage 3 forward test):**

- **The primary hypothesis failed** ([forward test results report](docs/stage3_forward_results.md)):
  on unseen data from 2026-06-01, TWSE 10-session R22-IR has a top-decile excess of +57.1 bps
  [−10.5, +124.3] and a fixed-portfolio annual net excess of +7.4% [−36.6%, +56.0%]. Under §13.8 it
  is not adopted, and work returns to improving the model itself.
- The point estimate is close to validation; the failure is mainly a short period (about 7
  non-overlapping 10-session windows).
- All six TWSE signals have positive Rank IC (0.074–0.149), but no top-minus-bottom spread or TWSE
  portfolio interval excludes 0; checkpoint selection by a holding metric shows no advantage in
  the forward period; TPEX remains weak.[S17]

**2026-10-21 addendum (switch to cross-sectional rank):**

- **The task becomes `cross_sectional_rank`** ([§13.10](docs/experiment_protocol.md)): only relative
  ranking within a date and market has evidence, so the models now predict ranks; the label window
  is unchanged (t+1 open to the close of session h). Rank baselines' market-equal Rank IC
  (validation → test): PatchTST 0.057 → 0.035, moving-average crossover 0.020 → 0.043; Taiwan is
  near 0 on test except at 1 session.
- **Kronos `a-rank64` pilot** ([§13.11](docs/experiment_protocol.md), [rank model pilot results report](docs/rank_pilot_results.md)):
  Taiwan 1-session Rank IC 0.064 [0.027, 0.102], pass; Kronos minus PatchTST −0.002 [−0.021,
  0.018], no gain; Taiwan 6–14 sessions 0.008 [−0.044, 0.064], fail. Every paired difference
  between Kronos and PatchTST in the six test cells crosses 0.
- Taiwan 1-session ranking is shared by every learned model, but holding for 1 session is a day
  trade, and backtest stage 1 found the gross edge below costs. Taiwan long horizons failed for
  every learned model during test, while the moving-average crossover grew stronger.[S18]

**2026-10-22 addendum (the main line becomes the Taiwan cross-stock DL pipeline):**

- Literature check: pooled versus local training has no settled answer (Jiang/Kelly/Xiu 2023: a
  US-trained model is clearly better only in small markets; Choi/Jiang/Zhang 2025: separate
  training is better; Cakici/Zaremba 2026: comparable). Taiwan 1-session Rank IC already matches
  the public Qlib benchmark; what is missing is walk-forward retraining, a GBDT baseline on the
  same features, and a holding-aligned top-k dropout backtest.
- Decision: Taiwan only from 2016 with quarterly walk-forward; the final model is a MASTER-style
  cross-stock transformer with a pluggable encoder (scratch / Kronos / Chronos-2); it must beat
  LightGBM on the same features; LLM-in-the-loop alpha mining is excluded on cost; Kronos is no
  longer the backbone. Registered in [§13.12 and §13.13](docs/experiment_protocol.md).

**2026-10-23/24 addendum (the Taiwan main line is complete; conclusions of the model and strategy lines):**

- **B0 / D0 / ablation** ([§13.13](docs/experiment_protocol.md); [LightGBM baseline report](docs/lgbm_baseline_results.md), [cross-stock report](docs/cross_stock_results.md)): B0 `t-lgbm-alpha` short-cell Rank IC 0.118, the three-seed D0 ensemble 0.122; the only D0 − B0 paired difference clear of 0 is the short cell, +0.0043 [+0.0015, +0.0071]; A1 shows inter-stock attention adds nothing and D0's edge is the ensemble. Multi-day horizons collapse in P2 (holding cells 0.034 / 0.047).
- **Strategy layer** ([§13.14, §13.15](docs/experiment_protocol.md); [strategy layer report](docs/strategy_layer_results.md), [short day-trade report](docs/futures_daytrade_results.md)): the long-only band portfolio cuts turnover to 2% a day and the cost drag to 3–5% a year, but the P2 signal is dead and H-D fails in every capital scenario; the TX-hedged version takes 27–28 margin calls. The h = 1 short stock-futures day trade earns +61–65% gross a year but one tick of slippage (0.23% per side) makes it −60% net; H-E fails.
- **Residualized labels** ([§13.16](docs/experiment_protocol.md); [residualized labels report](docs/residual_labels_results.md)): labels residualized on a size proxy and industry restore B0's P2 holding cell against the residual target to 0.0646 [0.0457, 0.0856] (H-F passes) without hurting h = 1; what collapsed was the exposure inside the labels, not the model. D0 on the new labels still passes H-A only in the short cell (+0.0037); frozen Kronos-small equals a single D0 seed and Chronos-2 is slightly worse; the three-seed range is 0.002–0.008. The model line converges on **B0 as baseline with the D0 ensemble as co-candidate**; D1 does not enter the strategy layer.
- **§13.17 step 0** ([results report](docs/exposure_diagnostics_results.md)): the tradable residual alpha is too small (top-minus-bottom quartile residual spread 0.9–1.0% per 10 sessions in the liquidity universe during P1, industry-neutral long-short gross 9–10% a year, 5–8% on P1 ∪ P2), and inside the roughly 130 large caps with liquid stock futures the P2 residual Rank IC is zero; the stock-futures long-short earns −0.6% (B0) / +0.4% (D0) gross against an 18% cost drag, so the pre-registered stop rule fires and no strategy is built.
- **§13.19 six scenarios under institution costs** ([results report](docs/institution_cost_scenarios_results.md)): with institution costs (commission 0.03%, slippage 0.05%, borrow 2%) none of the 36 cells (six long-short scenarios × two models × three markets) clears the pre-registered bar (P1 ∪ P2 net Sharpe ≥ 1, lower bound > 0, P2 > 0). The multi-day long-short drag falls only 2–5 points and its bottleneck is the P2 short book (−61% to −89%; the multi-day ranking is in substance a short-term reversal signal that shorts the names with the largest 20–60-session gains, of the universe's size, and P2's persistent momentum turned both legs against the book); the TX overlay goes long because β ≈ −0.5 and recovers beta, not alpha; day trading earns +123–179% gross against a 0.31% institution round trip per leg and stays negative unless slippage is zero; capacity above NT$300M erases the TPEX positives. The supplementary reversal-exposure diagnostic: after orthogonalizing the score to beta (correlation −0.24 to −0.45) and the trailing 20 / 60-session returns and pairing inside momentum quintiles, the P2 spread per cohort moves from −1.8% to −0.5% back to +0.1% to +0.9%, and P1 ∪ P2 is positive with intervals clear of 0 for all four (model, market) pairs; the P2 loss is structural exposure, not vanished skill; turnover conditioning has no consistent effect; the purified gross spread (TWSE +12%, TPEX +25–28% a year) still cannot pay for full turnover every 10 sessions. Inside the band construction (supplement 2) the orthogonalized score turns S1's institution net from −6% to +8% into +11% to +29% with the same parameters, every P2 positive, maximum drawdown −11% to −18%, β −0.15 to −0.20; 10 cells clear the institution bar (B0 and D0 S2 on all three markets, S1 on TPEX and TW); momentum-quintile matching is not viable (10% daily turnover); capacity about NT$300M or less. The design was chosen after seeing P2, so the bar is met informationally only and needs P3. Robustness checks (supplement 3): B0's V1 S1 is positive in all four calendar years and does not rest on one year, while D0's results concentrate in 2023–2024 and decay year by year; on the V1 score all 8 (h, K, m) combinations are positive in all four (model, market) pairs and the selected parameters sit mid-grid; the borrowable-universe lists have been requested from the data project for phase 1.
- **Conclusion**: under Taiwan's instruments and costs this model line has no deployable strategy that beats 0050 buy-and-hold. The evidenced abilities are h = 1 ranking (untradable at execution costs) and residual multi-day ranking (too small, and absent from the shortable universe). The reproducible deliverables are the walk-forward pipeline, the pre-registered decisions and the bilingual reports.

The current pair consists of two independent models. It is not a renamed direction V1/V2/V3 shared-backbone multitask model. Section 6 explicitly retains the experiment stopped because **validation already demonstrated poor and deteriorating quality**, separately from GPU throttling, memory-boundary failures, and wrong-preset launches.

### 2 Project boundaries that must survive the handoff

- Retain Kronos/LoRA as the research subject unless the user authorizes a change. A stronger baseline does not authorize replacement with indicator rules or a tree model.
- The model is a predictive-signal layer. Hermes will later combine forecasts, indicators, trading rules, and backtests. Do not embed fees, order size, portfolio rules, or net-of-cost targets in the current model without approval.
- The current tasks are alpha point estimation and absolute-stock down/neutral/up classification. There is no additional alpha-direction classifier.
- Do not remove existing features without authorization. Preserve historical presets, readiness capabilities, data-policy checks, baseline-completion checks, and CLI behavior.
- Keep dynamic training sampling and exhaustive lazy validation/test. Do not drop final batches or materialize a giant window dataset.
- Do not redownload market data, rerun CPU prepare, or rebuild numerically unchanged baselines merely because unrelated code or naming changes.
- Validate training/CUDA behavior in the intended cloud environment. Do not inspect/create a local training environment or silently migrate dependency management.
- Scope all lifecycle operations to exact run/Pod identities and protect concurrent work. New paid experiments require the applicable budget authorization.
- All 1–14 horizons remain equally weighted. The user prioritizes market-move signals over rapid trading.
- The earlier nominal 80% coverage within ±5 percentage points was considered an acceptable provisional range. Widening intervals is not directional improvement; the current models no longer output intervals.

### 3 Evidence and commit semantics

`TEST_COMPLETE` means matching durable training and ready final `split=test` evidence; normal early stopping is completion. `VALIDATION_ONLY` is not a holdout result. `STOPPED_POOR_VALIDATION` means manual termination for observed weak validation, not successful workflow completion. `STOPPED_INFRA`, `FAILED_RUNTIME`, `STOPPED_WRONG_PRESET`, and `INCOMPLETE_UNRESOLVED` remain distinct. `DEFINED_NOT_VERIFIED_RUN` marks a preset without a verified completed formal run in this inventory.

Three commits must not be conflated: the **architecture-introduction commit**, a **matching numerical-source snapshot**, and the **report commit**. A snapshot in the ledgers was checked against every file hash recorded in `training_resume_contract.training_implementation.files`. It is not proof of a recorded runtime Git HEAD, a clean entire worktree, or matching deployment scripts. Candidates were selected before run initialization and then verified by content, not assigned from dates alone. Hugging Face weight revisions are not repository commits.

Evaluation cohorts:

| Cohort | Actual population | Comparison boundary |
| --- | --- | --- |
| G0 | Older chronological splits, generally 20k validation benchmarks | Test remained locked; extending history also changed split periods |
| G1 | Fixed validation June–November 2025 and test December 2025–May 2026, 20k benchmark | A/B membership and train scales differed; old baselines were capped |
| G2 | Full test: A 1,261,524; B 1,264,861 | A 3,337-window difference prevents an exact paired history-span ablation |
| G3 | Common cleaned evaluation: 534,327 windows, 112 cutoff dates | One dependent period, unequal market weights, and version-specific metrics remain |

All six current signal models have membership `93e2e6a0b74edbdce616b82628aed180dad4d004a9bcd24292a78896c240bc0d`. The nominal test is `[2025-12-01,2026-06-01)`; actual cutoffs end on 2026-05-11 because each window requires all subsequent 14 sessions. Windows are not independent temporal observations.

### 4 Architecture roadmap

A0–A9 are labels used in this document, not existing CLI/schema identifiers. Direction V1/V2/V3, Git v0.x tags, Stage 1/2, and A/B history groups are separate concepts.

| Generation | Changes and intended problem | Introduction/support commits | Observed outcome |
| --- | --- | --- | --- |
| A0 quant-only alpha | Kronos tokenizer/backbone, LoRA 8/16, resampler, benchmark conditioning, 14-horizon q10/q50/q90 | `04e697b`, bar-store `abf8f7c`, tuning `3af6088` | Stage 2 modestly improved normalized pinball; direction/correlation and longer-history stability remained weak under G0 |
| A1 scale probes | Frozen checkpoint, ridge historical-scale readouts and shuffled controls | `2562e86`, tmux/guard `42d531c`, `91e3e34` | Asset scales were readable; benchmark-scale generalization was poor. Not evidence of total information loss or future-alpha improvement |
| A2 fixed splits/numeric branch | Train through May 2025; June–November validation; December–May test; 8 scale features and direct benchmark latents | `642217c`; v0.1.0=`fb8482e` | Mixed A/B results, sampled evaluation and unequal membership |
| A3 full-data workflow | Exhaustive lazy evaluation, reusable prebuilt full-data baselines, 20 scale descriptors, market/output-scale awareness, same-date/market ranking, larger LoRA | `a51da94`, `8e7f5d1`; v0.2.0=`44e7139` | Better engineering/comparison resources, but G2 A/B still differed and early peaks persisted |
| A4 cleaned capacity controls | Continuous 128+14 sessions, liquidity gate, common evaluation, 0.8 annual sampling decay, separated quantile position/width, static calibration, independent ranking; LoRA 32/64/partial presets | `3ddaad7`, parallel Pods `5e18f88`, memory fix `39cef62` | A-LoRA 64 became the historical quantile alpha reference; partial unfreezing did not win |
| A5 adaptive | Market residuals, numeric ranking path, regime/recency calibration and delayed-feedback diagnostics; stronger dropout/decay and sample schedule | `0d47732`, `ead2351` | All three Adaptive64 raw losses trailed A-LoRA 64; coverage was not alpha skill. Adaptive128 was defined but no formal completed run was found |
| A6 direction V1 | Cost-band stock and alpha classifiers, ten technical channels, horizon-query readout, representation regularization, quantile/ranking retention | `04f692b`, `39f337e`, `5a55668` | Class collapse, missing direct direction-scale path, seed-dependent reference weights and weak generalization |
| A7 direction V2 | Direct scales, market-prior/signal separation, seed-invariant train references, shared-LoRA PCGrad, BA-led selection with pinball guard | `d2238a7`, acceptance snapshot `f6271e7` | Some improvement, no alpha breakthrough; direction selection could sacrifice alpha-optimal checkpoints |
| A8a direction V3 | Ordered conditional probabilities, benchmark-state token/scale gates, prior-relative probability skill and decision penalties | `346cae0` | **Manually stopped for poor and deteriorating validation; no final test** |
| A8b current signal64 | Two independent LoRA 64 models: robust alpha point loss and fixed ±1% stock CE; no shared gradients or cost labels | `bc84d49`, pairing `8bae2db`, repair `3b7cccb` | Three completed seeds; no stable cross-seed/cross-market learned-baseline advantage |
| A9 date-mixing control (not adopted) | Hypothesis: the historical `DateMarketSampler` emits 128-sample same-date/market blocks, so each update covers about two date/market groups and the model chases date-level common shocks. A bounded local shuffle, `training.date_mixing_buffer_samples=16384`, mixes many dates per update while keeping each epoch's samples and annual quotas; everything else equals signal64 and baselines are reused | Implementation `31ef7c1`; docs `64a7b3f` | **Validation failed**: faster training-loss decline, worse validation and a stronger long-horizon "down" classifier bias. It contradicts the hypothesis; mixing stays off by default and is kept for reproducibility |

The main historical probe used R04/checkpoint 018145, `probe-20260914T101527Z-105a4d66`; another retained probe is `probe-20260914T091728Z-18045a54`. It used 16,384 train and 4,096 validation samples without future-alpha targets or test. Conditioned readout asset 20/60d validation R²≈0.552/0.605, benchmark≈−0.322/−0.548, and benchmark context-CV≈0.00073. This motivated explicit scales and benchmark access, not a claim that all scale information had vanished.[S6]

All splits now require aligned 128 input and 14 output exchange sessions, valid OHLC/adjusted prices, positive stock volume, and historical 60-session median raw turnover of US$1m/TWD 10m. No cross-suspension backfilling, date compression, or removal of valid extreme returns. Current maximum-drawdown buckets are diagnostics, not filters. Main-model annual sampling differs from baseline full traversal; matching populations do not imply matching weights/FLOPs. No completed B-signal64 run was verified here.

### 5 Completed and validation-only experiment ledger

#### 5.1 Original and fixed-date experiments

Loss below is 14-horizon normalized pinball, not current Smooth L1. Historical validation-only results must remain distinct from test.

| ID | Run ID and experiment | Best checkpoint | State and brief result | Matching numerical snapshot |
| --- | --- | --- | --- | --- |
| R01 | `run-20260903T092249Z-1771920450`, early Stage 1 | `checkpoint-018750` | Validation only; 20k, loss 0.317998; metadata snapshot only, no standalone DOCX | `6e7480f`, 22 files |
| R02 | `run-20260903T140429Z-1247813592`, Stage 1 from 2021 | `checkpoint-003908` | Validation only; train fraction 0.05/cap 500k; loss 0.314759, 5d correlation 0.078797 | `9bf84f9`, 22 |
| R03 | `run-20260903T184811Z-81563229`, Stage 2 from 2021 | `checkpoint-028982` | Validation only; loss 0.307166, 2.41% below R02; 5d correlation 0.107818; historical lifecycle exit 1 is an operational caveat | `3d3831e`, 22 |
| R04 | `run-20260905T203327Z-270337978`, Stage 2 from 2016 | `checkpoint-018145` | Validation only; loss 0.314283, 5d correlation 0.044618; not a history-only ablation | `ccb973c`, 22 |
| R05 | `run-20260915T091153Z-918316277`, fixed A | `checkpoint-043773` | Complete G1 test, 20k; loss 0.306784; LoRA 8 and 8-feature branch | `e3f8086`, 26 |
| R06 | `run-20260917T110811Z-1986920595`, fixed B | `checkpoint-094452` | Complete G1 test, 20k; loss 0.313541; some better raw errors but different membership/scales | `a37aaf8`, 26 |
| R07 | `run-20260922T182053Z-1388222581`, full-data A | `checkpoint-010944` | Complete G2; 1,261,524 test; loss 0.321509; 5d correlation 0.051991; first validation best | `feff226`, 31 |
| R08 | `run-20260923T095302Z-311336160`, full-data B | `checkpoint-023613` | Complete G2; 1,264,861 test; loss 0.333552; 5d correlation 0.068226 | `0c6b627`, 29 |

R02–R08 reports are indexed in S4; R01 survives in S7. Reporting milestones include `a80c85f`, `a433e5c`, `3cf3043`, `a37aaf8`, `28c6a1e`, `4b33d1f`; use each actual file's `git log --follow` for addition/revision ancestry, not as a training-version claim.

#### 5.2 Cleaned capacity experiments

All three are complete G3 tests, seed 42, still quantile/ranking models. Architecture=`3ddaad7`; all 31 recorded numerical files match `39cef62`. Reports were added at `5909b4e`, revised at `fb841a7`, and archived at `9582978`.[S5]

| ID | Run ID | Capacity and best point | Raw normalized pinball /5d q50 correlation | Conclusion |
| --- | --- | --- | --- | --- |
| R09 | `run-20261007T085635Z-bc59d5bbd2d84ad4` | LoRA 32, frozen; `checkpoint-014720`, ~0.8 epoch | 0.293694 /0.081995 | Did not beat LoRA 64 |
| R10 | `run-20261007T085627Z-cc5f9ef13b484d90` | LoRA 64, frozen; `checkpoint-014720`, ~0.8 epoch | **0.291608 /0.098657** | Historical quantile alpha reference, not a cross-task or stable champion |
| R11 | `run-20261007T085308Z-2debcc6e669b4522` | LoRA 32, last 2 blocks unfrozen; `checkpoint-011040`, ~0.6 epoch | 0.293823 /0.083260 | Partial unfreezing did not win |

Use revised reports in `pre-a-signal64-20261016/20261007-a-group`, not the older `pre-20261009` originals.

#### 5.3 Adaptive64

All are complete tests; 33 recorded files match `ead2351`, architecture introduction=`0d47732`, reports=`13cf01a`.[S5]

| ID | Run ID /seed | Best checkpoint/~epoch | Raw normalized pinball /5d q50 correlation | Brief result |
| --- | --- | --- | --- | --- |
| R12 | `run-20261008T084331Z-3e990331418d46d0` /42 | `checkpoint-025759` /1.4 | 0.296831 /0.070310 | Later best point without better alpha than R10 |
| R13 | `run-20261008T084336Z-993a064dae124ab0` /43 | `checkpoint-007360` /0.4 | 0.296756 /0.073966 | Seed-sensitive outcome |
| R14 | `run-20261008T084341Z-d677229bb2c44a32` /44 | `checkpoint-014720` /0.8 | 0.296555 /0.065017 | Best raw loss of these three, still behind R10 |

Frozen, static and delayed-feedback online calibration are different evidence. Do not call online diagnostics a purely frozen unseen-holdout improvement.

#### 5.4 Direction V1/V2 and wrong-preset runs

Loss here remains raw alpha normalized pinball. Direction selection objectives differ across versions and are not directly rankable.

| ID | Run ID /experiment | Best checkpoint | Alpha loss /5d q50 correlation | State and snapshot |
| --- | --- | --- | --- | --- |
| R15 | `run-20261010T112317Z-a2128baabec94c55`, V1 seed 42 | `checkpoint-003680` | 0.297410 /0.044963 | Complete; severe Taiwan alpha-class collapse; `3ead8d8`, 38 files |
| R16 | `run-20261010T112320Z-1531ef3bf22949ad`, V1 seed 43 | `checkpoint-007360` | 0.294903 /0.086873 | Complete; behind R10; `3ead8d8`, 38 |
| R17 | `run-20261010T112319Z-bb81bfdd1b4545f5`, V1 seed 44 | `checkpoint-003680` | 0.294654 /0.083842 | Complete; early peak/reference differences; `3ead8d8`, 38 |
| R18 | `run-20261011T200040Z-6a76903c49f1459f`, V2 seed 42 | `checkpoint-007360` | 0.293857 /0.084676 | Complete; no alpha breakthrough; `f6271e7`, 40 |
| R19 | `run-20261014T080044Z-06239560501f47a9`, wrong V1 seed 42 | `checkpoint-003680` | 0.297410 /0.044963 | Complete **old** task, not the intended new signal model; `7d4130c`, 42 |
| R20 | `run-20261014T084311Z-5ba50929af254002`, wrong V1 seed 348 | `checkpoint-003680` | 0.296465 /not expanded here | Complete old task; historical metadata only; `7d4130c`, 42 |

V1 introduction=`04f692b`; V2=`d2238a7`; R15–R18 reports=`d37fa14`, now in S5. R19/R20 preserve the cost of misleading version/CLI guidance rather than demonstrate acceptance of the new model.

V1's 5d alpha balanced accuracy was about 33.62%/34.84%/34.09% despite roughly 51% raw accuracy. Seed 42 TPEX predicted alpha-up only≈0.0224% with≈99.9636% down recall. V2 repaired scale access and references/gradient handling, but direction-led selection could retain a checkpoint different from the best alpha-validation point.[S5,S9]

#### 5.5 Current paired signal64 runs

All six children have completed training and ready final tests.45 recorded numerical files match `3b7cccb`; architecture=`bc84d49`, pairing=`8bae2db`, reports=`9582978`.[S1–S3]

| ID | Parent run /seed | Alpha child and best checkpoint | Stock-classification child and best checkpoint |
| --- | --- | --- | --- |
| R21 | `run-20261015T064558Z-f04e155d4299416f` /42 | `run-20261015T064558Z-f04e155d4299416f-alpha`; `checkpoint-003680`, 1st/7 validations, ~0.2 epoch | `run-20261015T064558Z-f04e155d4299416f-direction`; `checkpoint-007360`, 2nd/8, ~0.4 epoch |
| R22 | `run-20261015T064559Z-624f212a27894308` /347 | `run-20261015T064559Z-624f212a27894308-alpha`; `checkpoint-011040`, 3rd/9, ~0.6 epoch | `run-20261015T064559Z-624f212a27894308-direction`; `checkpoint-007360`, 2nd/8, ~0.4 epoch |
| R23 | `run-20261015T064600Z-13b8b4376ea5413b` /5067 | `run-20261015T064600Z-13b8b4376ea5413b-alpha`; `checkpoint-018399`, 5th/10, ~1.0 epoch | `run-20261015T064600Z-13b8b4376ea5413b-direction`; `checkpoint-011040`, 3rd/9, ~0.6 epoch |

Every aggregate below equally averages the 1–14 horizon metrics rather than pooling horizons and recomputing them.

| Seed | Alpha normalized Smooth L1↓ | Alpha MAE↓ | Alpha Pearson↑ | Alpha sign↑ | Stock CE↓ | Accuracy↑ | Balanced accuracy↑ | Macro-F1↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.528707 | 0.060582 | 0.102801 | 54.30% | 1.039206 | 42.36% | 36.39% | 0.338556 |
| 347 | 0.540340 | 0.061778 | 0.074525 | 51.37% | 1.032702 | 44.13% | 37.67% | 0.355016 |
| 5067 | 0.536003 | 0.061239 | 0.092305 | 53.87% | 1.027663 | 44.59% | 38.05% | 0.364459 |

Date-paired loss differences first average stocks and horizons within a date, then equally average dates; they need not equal the difference of population-weighted aggregate losses. The 95% intervals use a 14-date circular moving-block bootstrap, 1,000 replicates and seed 42, without multiple-comparison correction. They are not accuracy, Rank IC or between-seed intervals. Crossing zero does not establish equivalence.

Key unresolved findings:

- GRU alpha loss = 0.529025, GBDT = 0.531839, zero-return = 0.536328. Seed 42 ranks 1st/13, but daily main-minus-GRU gap−0.000328 has 95% interval[−0.004133, +0.003755]. Seeds 347/5067 trail all four learned baselines.
- Test-majority accuracy 42.90% is an ex-post imbalance reference, not a deployable test-fitted model. Exceeding it does not establish probability skill: all three CE values trail GBDT 0.999028 and the other learned baselines.
- Seed 5067 improves sample-weighted classification but its equal-market macro-F1=0.312710 is below 42=0.330799 and 347=0.345726. US contributes 82.30% of windows.
- Day 14 alpha RankIC for 42/347/5067: US +0.045486/+0.072947/+0.083138; TWSE−0.063087/−0.037710/+0.031969; TPEX−0.083191/−0.090545/−0.016489. Seed 5067's all-horizon-positive TWSE ranks are limited positive evidence; do not repeat the obsolete claim that every seed has negative long-horizon Taiwan ranks. TPEX remains negative over 11–14d.
- Seed 5067 day 14 neutral recall is only 6.35%, TPEX 0.18%, TWSE 1.88%. Pooled progress does not hide these failures.
- Alpha best points move to 0.2/0.6/1.0 epoch, while classifiers remain at 0.4/0.4/0.6. A later peak is not proof the generalization problem is solved.
- Runtime micro-batches differ:32×8 accumulation for 42 versus 64×4 for 347/5067. Effective batch 256 does not make this a fully controlled seed-only ablation. Learned baselines remain persisted seed 42 results.

#### 5.6 A9 date-mixing single-factor control

R24 shares R21's seed, data, labels, loss, model architecture and baselines; the only numerical difference is the training order. All 45 recorded manifest numerical files of both children match `31ef7c1`; relative to `c7b5550` only `config.py`, `training.py` and `date_market_sampler.py` differ. Four RTX 5090 creation requests were rejected and an RTX 4090 ran instead; the automatic batch plan remained 32×8 with effective batch 256, as in R21. Criteria were fixed beforehand in [experiment protocol 13.1](docs/experiment_protocol.md) and decided on validation only.[S11]

| ID | Parent run ID /seed | Alpha child /best checkpoint | Classification child /best checkpoint | State |
| --- | --- | --- | --- | --- |
| R24 | `run-20261016T165022Z-32d734a0cb644e44` /42 | `run-20261016T165022Z-32d734a0cb644e44-alpha` /`checkpoint-007360`, 2nd/8 validations, ~0.4 epoch | `run-20261016T165022Z-32d734a0cb644e44-direction` /`checkpoint-003680`, 1st/7, ~0.2 epoch | `TEST_COMPLETE`; both stopped early normally; **validation criteria failed** |

**Batch composition** (`training_batch_composition` in `summary.json`, first 256 optimizer steps): the unmixed reference order has a median of 2 distinct date/market groups per update (1 per micro-batch); the mixed order has 166 (30 per micro-batch). This confirms the historical batch composition on the real training index; it is not an inference.

**Validation (decision basis):**

| Metric | R24 datemix | R21 seed 42 | R21–R23 three-seed range |
| --- | ---: | ---: | ---: |
| Best alpha normalized Smooth L1 ↓ | 0.46363 | 0.45476 | 0.45311–0.45476 |
| Last alpha validation ↓ | 0.47878 | 0.45883 | 0.45883–0.46297 |
| Best classification CE ↓ | 1.02728 | 1.02844 | 1.01311–1.02844 |
| 12/14 day ECE at the best classifier ↓ | 0.1026/0.1380 | 0.0949/0.1037 | — |
| Market-macro CE ↓/macro-F1 ↑ at the best classifier | 1.0430/0.3299 | 1.0197/0.3592 | — |
| US 14 day Rank IC at the best alpha point ↑ | 0.0161 | 0.0048 | 0.0048–0.0413 |

Neither the alpha criterion (best value below 0.45476 and not the first evaluation) nor the classification criterion (no deterioration in CE, 10–14 day ECE, market-macro CE and F1) held. Nearby training-batch loss fell faster under mixing: alpha about 0.396 at 1.4–1.6 epochs versus about 0.42 for R21; classification about 0.864 at 1.2 epochs versus about 0.894. These are nearby-batch losses, not full-train losses or an unbiased generalization gap, but they move consistently with the validation deterioration.

**Test (recorded only, not used for method selection):**

| Run | Alpha normalized Smooth L1 ↓ | Alpha MAE ↓ | Alpha Pearson ↑ | Alpha sign ↑ | Classification CE ↓ | Accuracy ↑ | Balanced accuracy ↑ | Macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R24 datemix 42 | 0.535295 | 0.061161 | 0.095170 | 52.67% | 1.118239 | 42.84% | 36.51% | 0.300239 |

- Date-paired loss differences (main minus baseline, negative favors the main model; 14-date block bootstrap, 1,000 replicates): alpha versus GRU +0.006578 [−0.002895, +0.016329], versus GBDT +0.003796 [−0.008078, +0.015085]; classification versus GBDT +0.116751 [+0.053923, +0.183887], versus GRU +0.115948 [+0.045455, +0.191853]. Classification CE is clearly worse than the learned baselines.
- The classifier predicts "down" for 90.2%/91.8% of 10/14 day windows against a true 42.7%, with ECE 0.212/0.227; the per-date CE standard deviation is 0.152 versus 0.087 for R21.
- Day 14 alpha test Rank IC: US +0.0718, TWSE −0.0385, TPEX −0.0737. The US value lies inside the historical seed range; Taiwan remains negative.

**Interpretation (inference, not yet verified):** the result contradicts the hypothesis. With mixing, the model fits the training period more efficiently while validation worsens; the deterioration concentrates in date-level common components such as point estimates and market-direction probabilities, while cross-sectional ranking at the selected checkpoint did not clearly worsen. When every sample in a micro-batch shares the same benchmark, gradients for the date-common component point in inconsistent directions across neighboring steps and largely cancel, so the historical same-date batches may implicitly suppress fitting of training-period date/market states. Mixing turns those patterns into consistent, learnable signals that do not transfer to a new period. Per protocol, no further datemix seeds run and R24's test selects nothing; `date_mixing_buffer_samples` defaults to `null`, so historical presets keep their training order.

### 6 Manually stopped and failed experiments

#### 6.1 Explicit stop for poor validation

**Q01: `run-20261012T211457Z-09c0c46c20444122`, A-direction64-v3 seed 42, `STOPPED_POOR_VALIDATION`.**

The user requested immediate Pod termination after full validation exposed weak quality, rebound, and directional bias. It was stopped before natural training completion, has no final test, and must not enter a completed-alpha leaderboard. Its 42 recorded numerical files match `346cae0`.[S7,S9]

Contemporaneous conversation diagnostics, each covering 528,315 validation windows:

| Checkpoint | Approx.epoch | 5d q50 alpha-sign agreement | 5d q50 correlation | Version-specific selection loss |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-003680` | 0.2 | 52.77% | 0.08829 | 1.10185 |
| `checkpoint-007360` | 0.4 | 55.43% | 0.09397 | **1.05147** |
| `checkpoint-011040` | 0.6 | 54.98% | 0.09188 | 1.05635 |
| `checkpoint-014720` | 0.8 | 53.74% | 0.08726 | 1.08340 |

The model degraded after its best point. Latest alpha-down prediction shares were approximatelyTWSE 96.25%, TPEX 96.43%, US 92.11%. Prior-relative probability skill was tiny and the latest absolute skill negative. Finite shared-LoRA gradients/PCGrad execution proved an update path, not useful generalization.

Pod `cn0bxripyi5mp8` deletion/absence was verified then. Later authorized unfinished-run cleanup included this run; do not assume its volume/W&B checkpoints still exist. This document, architecture commits, contemporaneous diagnostics and pre-cleanup metadata preserve the research event. Do not erase poor-validation stops from research history when removing operational run clutter.

#### 6.2 Infrastructure, runtime, and wrong-preset attempts

| ID | Run ID | Cause/evidence boundary | Snapshot or repair |
| --- | --- | --- | --- |
| F01 | `run-20260903T134249Z-105084056` | Incomplete/unresolved; failed W&B metadata, no completed model; no quality-stop claim | No recorded numerical match |
| F02 | `run-20260922T074951Z-141376769` | Unfinished during main-throughput investigation/restarts; not a forecast-quality outcome | `21990dc`, 31 files |
| F03 | `run-20260922T145223Z-218655310` | Unfinished during the same infrastructure-repair period; exact final cause not fully retained | `e5d92c9`, 31 files; later R07 |
| F04 | `run-20261007T070237Z-767e3100b05146c8` | LoRA 32 runtime failure, exit 1, no model initialization/final test | Later memory/probe/worker cleanup repair `39cef62`; successful R09 |
| F05 | `run-20261007T070239Z-2dd9f4ffc9634620` | LoRA 64 runtime failure, exit 1, not a quality ablation | Same repair; successful R10 |
| F06 | `run-20261007T070243Z-8c691a7c9ba14a27` | Partial runtime failure, exit 1, not a poor-validation stop | Same repair; successful R11 |
| F07 | `run-20261011T184451Z-956d8946164f40b3` | V2 seed 42 sustained SW Power Cap on RTX PRO 6000 WK; stopped after 43 minutes, ~step 368/94,208 samples, no checkpoint/validation/test | `f6271e7`, 40 files; incident S8; later R18 |
| F08 | `run-20261014T090108Z-338fadabfeb848f4` | Intended new task but old `a-direction64` seed 147 actually ran; terminated on request, incomplete | `7d4130c`, 42 files; pairing/guidance repaired by `8bae2db` |

F02/F03 retain the infrastructure context but not enough terminal evidence for a unique root cause. F04–F06 are verified failed attempts; `39cef62` is the subsequent repair, not a proven original runtime HEAD.

Retired baseline/verification attempt IDs also include `baseline-run-20260920T124808Z-20921`, `baseline-run-20260920T152758Z-32557`, `baseline-run-20260920T175140Z-32301`, `baseline-run-20260921T032356Z-2164`, `baseline-run-20260921T043022Z-3705`, `baseline-run-20260921T061805Z-16783`, `baseline-run-20260921T073833Z-10520`, and `baseline-run-20261010T052256Z-15323`. They are not all quality failures. The last was associated with the cost-ratio-over-notional baseline cache `baseline-0cb2d490fa3d450dbfdaa208fc90e3c4355f8b27a9a61e2354f48c859921a497`, repaired by `5a55668`.[S7]

### 7 Operational changes that matter to reproducibility

| Area | Main commits | Capability/lesson to preserve |
| --- | --- | --- |
| Full baseline and evaluation | `a51da94`, `8e7f5d1`, `44e7139` | Prebuild full-data baselines, save models/results, reuse matching test metrics; retain dynamic training |
| Baseline throughput/resume | `9ebff63`, `9e1af2c`, `49e68b4`, `bea8614` | Bounded vectorization, hardware tuning, durable cursors; throughput is not quality |
| Dataset/baseline identity | `7d7bcd5`, `556c0d8` | A/B switching and numerical contracts, not unrelated names/scripts or a shared main-model manifest |
| Canonical naming | `eae18bc`, `900000c` | New project names without moving historical results/invalidation; README is public documentation |
| Main training performance | `f66af29`, `e5d92c9`, `feff226` | Measure compute/input wait/memory rather than VRAM percentage alone; protect active training |
| Duplicate membership/targets | `f2cdacb`, `0c6b627` | Shared evaluation backing, correct testing/resume paths, no per-model duplication |
| Readiness regressions | `736f1ae`, `adb521a` | Keep both current-policy admission and baseline-completion checks |
| Training/validation loss visibility | `3ddaad7`, current single tasks `bc84d49` | Retain all validation points and task losses; a nearby training-batch loss is not a full-train loss or an unbiased generalization-gap estimate |
| Error/repeated preflight | `fc98524` | Preserve provider errors, reuse valid readiness during GPU competition |
| Parallel runs/seeds/inventory | `5e18f88`, `838d8bf`, `ead2351` | Run-scoped locks/status/guards, immutable configuration, clear IDs/times/completion |
| Clock and orphan monitoring | `90acf1d`, `39f337e`, `3ead8d8`, `8bae2db` | Local sustained-clock checks, more than half the monitoring period below 80% reference before alert; identity-scoped cleanup |
| Paired workflows/publication | `8bae2db` | One parent/Pod for two independent children, publish only after Pod creation, resume unfinished child only |
| Selection/Python 3.12 | `3b7cccb` | Correct mounted identity and parser routing, not SHA-exception bypasses |
| Documentation/report history | `7d4130c`, `8d43d03`, `9582978` | Separate current/legacy architecture, retain diagrams/features/historical CLI and original reports |

Dereferenced release commits: v0.1.0=`fb8482e`, v0.2.0=`44e7139`, v0.2.1=`c03e416`, v0.2.2=`bea8614`. These are commits, not annotated-tag object IDs, and not guarantees of forecast quality.

Outstanding W&B request/risk: metrics-and-hyperparameters-only tracking without artifacts was discussed, not implemented. Current presets still set `log_model_artifact=true`, and final benchmarking separately uploads an artifact. Backend log/summary/artifact/finish exceptions in `tracking.py` are recorded and re-raised, so quota/upload failures cannot be guaranteed harmless to training. Nonfatal tracking with durable local evidence needs a separate design/acceptance task. This roadmap changes no W&B settings or records.

### 8 Exact current handoff specification

Each parent sequentially runs independent alpha and stock-classification models, sharing only read-only data. Each has its own Kronos-base/LoRA 64, resampler, conditioner, numeric branch, optimizer, scheduler, checkpoints and stopping state.

Each internal path uses 128 past PIT-adjusted asset/benchmark OHLCV bars, derived amount, window normalization, frozen tokenizer, Kronos/LoRA, resampling, benchmark conditioning, mean pooling, market/horizon embeddings and one task head. Twenty train-standardized statistics and direct benchmark latents supply numerical residuals. Alpha output is `[batch,14]`; stock logits are `[batch,14,3]`.

- Group A train starts on 2021-01-01 and ends before 2025-06-01: 4,710,117 eligible windows. Validation is `[2025-06-01,2025-12-01)` with 528,315; test is `[2025-12-01,2026-06-01)` with 534,327. Every latest label must precede its split's right boundary.
- Signals form after close t; entry is the next market-session open, counted as holding day 1, and exit is holding day h close. Missing stock data cannot postpone entry/exit. Benchmarks are `VTI.US`, `TAIEX.TW` and `TPEX.TWO` for US, TWSE and TPEx respectively.
- Alpha=`log(asset total-return gross)−log(benchmark gross)` over identical next-open-to-horizon-close times; not a CAPM residual or simple-return percentage-point difference.
- Loss=`SmoothL1((prediction−target)/s_h,beta=1)` with `s_h=max(train IQR,1.4826×train MAD,1e-4)`, fit on at most 50k train samples, seed 59. The output is not q50 or a guaranteed conditional mean.
- Stock labels use simple returns below −1%, inclusive [−1%, +1%], above +1%; compare in log coordinates using `log1p(−0.01)`/`log1p(0.01)`. Class order down/neutral/up; CE reads logits, not BCE; no costs.
- Alpha sign agreement is not stock accuracy. Positive alpha can coexist with a falling stock.
- `primary_5d/selection_score` is the 14H task loss despite its name; other fields in `primary_5d` are 5d. `comparison.rows` similarly mixes 14H selection loss with 5d quality fields.
- LoRA rank 64 and alpha 128, dropout 0.05, head LR 3e-5, LoRA LR 5e-6, weight decay 0.01, max 5 epochs, five full validations/epoch, patience 5 evaluations; plateau factor 0.3 after 2 stale evaluations, min ratio 0.09 and 2 low-LR intervals before stopping.
- Final test uses best-validation weights, not necessarily `completion-result` stopping weights.
- Quantiles, interval calibration, independent ranking, alpha classifier, ten technical channels, horizon queries, prior/state gates, PCGrad, representation perturbation, cost features and partial unfreezing are **not enabled**. Legacy config defaults do not establish active features.

Current baseline IDs are `baseline-3befc8e902e5c641b24ddf80c4ab7e11769529fa307e391de47e4416f1d885a3` for alpha and `baseline-b21f46ea4963562f03c648e9b0394f8a069767bca2a15a9f82e7c8ed69e04cc8` for stock direction. Models: always_buy, zero_return, momentum_5d, reversal_5d, ma_crossover, rsi, macd, volatility_scaled, GBDT, GRU, DLinear, PatchTST. Neural alpha uses normalized Smooth L1; GBDT fits MAE but selects using the shared evaluation loss. Stock classifiers use CE/log-loss. Fixed rules have no trainable epochs/early stopping and their 0/−30 logit outputs are not calibrated probability models.

### 9 Open problems and proposed research order

These are **candidate tasks and acceptance requirements, not implemented changes or new spending authorization**.

| Priority | Candidate investigation | Existing evidence | Acceptance boundary |
| --- | --- | --- | --- |
| Done | **Date-level versus within-date error decomposition (no GPU):** see the [error decomposition report](docs/error_decomposition_audit.md) | Alpha date-level share of normalized MSE is only 2.6%–3.8% (TWSE excepted); tail samples dominate the loss; date-level classification information is worth 5.4%–5.9% CE, yet baseline timing is negative and the learnable part is within-date discrimination (+0.023–0.029 CE) | Verified against recorded baseline values within 5×10⁻⁷; main-model per-sample predictions were not persisted, so the main model only gets a supporting per-date analysis |
| Done | **Persist per-sample predictions and rerun validation/test for R21–R24 checkpoints (GPU inference):** including the R21/R22 alpha checkpoints re-selected under [experiment protocol §13.2](docs/experiment_protocol.md); results in [error decomposition report section 7](docs/error_decomposition_audit.md) | Exports match the training records exactly (R21–R23 difference 0). Classification: Kronos has the best within-date discrimination and loses on timing; with timing removed, validation 1.0034–1.0065 (GBDT 1.0047) and test 0.986 (GBDT 0.990). Alpha: Kronos within-date correlation 0.10–0.12; US Rank IC carries forward to test while Taiwan collapses for every model. Three-seed ensembles are best on loss-type metrics | 10 checkpoints in about 3.5 hours on an RTX 5090; a lifecycle bug left the export Pod idle about 4.4 extra hours, fixed in `e8d29a8` |
| Done | **Phase 1: direction information in free Taiwan data (CPU only):** see [experiment protocol §13.3](docs/experiment_protocol.md) | Failed: none of the 8 feature sets passed on two or more decision horizons; the best, `all_pre_open` at 1 session, improved on the prior by 1.34%, with its interval against `F0` crossing 0. TAIFEX downloads failed (free data cover only the last three years) | Pre-registered criteria in §13.3; result files in S14 |
| Done | **Backtest stage 1: fixed holding periods (local):** see [experiment protocol §13.4](docs/experiment_protocol.md) and the [backtest results report](docs/backtest_stage1_results.md) | All three markets failed, with test annual net excess of −37% to −44%; costs consume the one-day gross edge; long-horizon test tails invert; large caps dominated the period, so every index-hedged version is negative | Pre-registration `5ddeff6`, code `a292a3b`; outputs in S15 |
| Done | **Backtest stage 1b: hold buffer with per-position accounting (local):** see [experiment protocol §13.5](docs/experiment_protocol.md) and the [backtest results report](docs/backtest_stage1_results.md) | All three markets failed, test −13% to −40% (exploratory); the top-K edge exists only during sessions and reverses overnight; lower turnover saves about three quarters of the costs, but no gross edge is left to keep | Pre-registration `c4ee514`, code `4a5bdbb`; the lower-turnover route stops as pre-registered |
| Done | **A: holding-aligned evaluation (local):** see [experiment protocol §13.6](docs/experiment_protocol.md) and the [evaluation results report](docs/holding_eval_results.md) | Most signals' top deciles lag heavily overnight; the Rank IC-IR re-selected checkpoints and R23 have little overnight drag, and TWSE 10-session R22-IR's top decile is positive on validation (+69.6 bps) and test (+103.2 bps); TPEX is negative almost everywhere. Descriptive only | Pre-registration `3949fb6`, code `7e7d925` |
| Done | **Stage 3: forward test (unseen data from 2026-06-01):** see [experiment protocol §13.8, its amendment and supplement](docs/experiment_protocol.md) and the [forward test results report](docs/stage3_forward_results.md) | All data and consistency checks passed; the primary hypothesis (R22-IR, TWSE, 10 sessions) failed: top decile +57.1 bps [−10.5, +124.3], fixed-portfolio annual net excess +7.4% [−36.6%, +56.0%]. Not adopted under the protocol; work returns to improving the model itself. All six TWSE signals have positive Rank IC (0.074–0.149), but no spread or TWSE portfolio interval excludes 0; checkpoint selection shows no forward advantage; TPEX remains weak | Pre-registration `e4d823c`, `c19bedc`, `5afcae9`; code `65bba60`, `b7d8561`, `cc57971`, `41278bc`; outputs in S17 |
| Done | **P2: checkpoint selection by a holding metric (inference on RunPod, scoring local):** see [experiment protocol §13.7 and its amendment](docs/experiment_protocol.md) and the [checkpoint selection report](docs/checkpoint_selection_results.md) | All 15 exports passed the integrity checks; R21–R23 all selected checkpoint-018399, and R22's selection is R22-IR, so the §13.8 primary hypothesis is unchanged. Selected checkpoints' TWSE 10-session test top decile: R22 +103.2, R23 +86.4, R21 −63.9 bps; on test both deciles beat the equal-weight average and the spread is negative. Descriptive only | Pre-registration `e4d823c`, `1197b60`; code `2d84ff9`, `e93705c`; outputs in S16 |
| Done | **Cross-sectional rank task `cross_sectional_rank`: rank labels and baselines:** see [experiment protocol §13.10](docs/experiment_protocol.md) | Only relative ranking within a date and market has evidence (Kronos alpha within-date correlation: validation 0.099–0.119, test 0.091–0.124); neither the numeric alpha nor the absolute direction robustly beat the baselines. The label window is unchanged (t+1 open to the close of session h); entry and exit rules are backtested by the hermes agent | Baselines are complete (`baseline-5fb3473a…`, results in §13.10); the Kronos pilot is in the next row |
| Done | **Kronos rank model `a-rank64` pilot (training and export on RunPod, scoring local):** see [experiment protocol §13.11](docs/experiment_protocol.md) and the [rank model pilot results report](docs/rank_pilot_results.md) | H1 passes: Taiwan 1 session 0.064 [0.027, 0.102]; H2 fails: Kronos minus PatchTST −0.002 [−0.021, 0.018]; H3 fails: Taiwan 6–14 sessions 0.008 [−0.044, 0.064]. Every paired difference between Kronos and PatchTST in the six test cells crosses 0; Taiwan long horizons failed on test for every learned model, while the moving-average crossover reached 0.063 | Pre-registration `64df916`; code `f5bdf1d`, `aad2d63`; outputs in S18. No further seeds under the protocol; on 2026-10-22 the main line became the Taiwan cross-stock DL pipeline (next row), Kronos is no longer the backbone, and date mixing is no longer registered |
| Done | **Taiwan main line: feature LightGBM baseline B0, cross-stock DL D0, ablation A1, top-k dropout backtest:** see [experiment protocol §13.12 and §13.13](docs/experiment_protocol.md), the [LightGBM baseline report](docs/lgbm_baseline_results.md) and the [cross-stock report](docs/cross_stock_results.md) | B0 short cell 0.118, D0 ensemble 0.122; H-A passes only the short cell (+0.0043 [+0.0015, +0.0071]); H-B fails (top-k dropout cost drag 17% a year); A1 shows inter-stock attention adds nothing; multi-day horizons collapse in P2 | Pre-registered; three seeds; A2–A4 deferred and then dropped (D0's edge is the ensemble, so architecture ablations carry no further information) |
| Done | **Strategy layer: long-only band portfolio with TX overlay (§13.14), stock-futures short day trade (§13.15):** see the [strategy layer report](docs/strategy_layer_results.md) and the [short day-trade report](docs/futures_daytrade_results.md) | H-D fails (P1 ∪ P2 net excess −4.7% / −5.4% in every capital scenario); H-D-hedged fails (27–28 margin calls); H-E fails (gross +61–65%, net −60%, one tick 0.23% per side, break-even 0.12%) | Pre-registered; official TAIFEX data; no re-tuning |
| Done | **Residualized labels and pluggable encoders (§13.16):** see the [residualized labels report](docs/residual_labels_results.md) | H-F passes (variant A: P2 holding cell against the residual target 0.0646 [0.0457, 0.0856], h = 1 unharmed); variant B trails A; D0 on the new labels still passes H-A only in the short cell; Kronos-small = a single D0 seed; Chronos-2 slightly worse; seeds stable | Model line converges: B0 baseline, D0 ensemble co-candidate; D1 and kronos-base-lora receive no further spend |
| Stopped | **Exposure-neutral strategy: 0050 core plus stock-futures long-short overlay (§13.17):** see the [step 0 report](docs/exposure_diagnostics_results.md) | The step 0 stop rule fires: in the stock-futures universe at h = 10 the industry-neutral long-short earns −0.6% / +0.4% gross against an 18.3% cost drag; the P2 residual Rank IC in that universe is zero; the liquidity universe's P1 residual quartile spread is 0.9–1.0% per 10 sessions | S1 / S2 / S3 not implemented; conclusion: no deployable strategy beats 0050 buy-and-hold; a restart needs new data (P3) or a new signal source, not re-tuning |
| Done | **Six scenarios under institution costs (§13.19, informational):** see the [results report](docs/institution_cost_scenarios_results.md) and the Word version `reports/institution_cost_scenarios_20261025.docx` | None of the 36 cells clears the bar; the closest is D0 S2 (multi-day L/S plus TX) on TW: net +16.8% [+1.4%, +33.7%], Sharpe 0.83, P2 −6.5%; S1's P2 short book −61% to −89%; institution day-trade net TWSE −17% to −33%, TPEX +7–12% (intervals span 0); retail day-trade drag 263–363% a year | Pre-registered; no forward test; conclusion: still no strategy attractive to the industry under institution costs; bottlenecks are the short-book construction, day-trade execution cost and capacity |
| Cancelled | **B: GBDT on open-to-open labels, and checkpoint selection by a holding metric** | Cancelled 2026-10-20: the original label window already runs from the t+1 open to the close of session h, and the hermes agent sets exits, so open-to-open labels are not needed; the rank task replaces this item | — |
| Shelved | **Classifier inference-time same-day prior normalization (no GPU training):** make it a formal inference and evaluation option. Shelved 2026-10-18: removing timing also removes the direction judgment the trading rules need | Removing timing is a label-free, deployable post-processing; validation matches GBDT, market-macro beats GBDT for all three seeds, and test beats GBDT | Write validation acceptance criteria into the protocol first; apply to every model identically; stocks of one date and market must be inferred together; needs implementation approval |
| Shelved | **Training-time date fixed effects** (classification first): draft design and pre-registered criteria in the [P1 design](docs/p1_date_market_logit_adjustment_design.md). Shelved 2026-10-18 for the same reason | **S0 passes:** removing timing lowers R21–R23 validation CE by 0.0146 on average (gate 0.005); reference values are measured (error decomposition report §7.6) | New single-factor preset; unchanged outputs; three seeds with pre-registered validation criteria; must beat the free inference-time normalization (criterion C1) |
| P2 | Same-date demeaned alpha target (demoted) | Alpha's date-level component is small and the expected gain limited; TWSE is the exception | Secondary study only, not the main alpha remedy |
| P0 | Capacity-matched contribution checks for Kronos latents, scales, benchmark, market/horizon features; frozen-readout versus LoRA when appropriate | Only alpha seed 42 narrowly leads GRU; repeated complexity additions failed | Same population/scales/runtime budget; preserve existing features; do not presume Kronos useless |
| P0 | Diagnose early peaks through update budget, regularization, sampling distribution and per-group learning curves | Current stock best points 0.4–0.6 epoch; falling batch train loss without matching full-validation gains; **R24: mixing dates sped up training fit and worsened validation, so sampling is not "the more i.i.d. the better"** | Retain every validation point and definitions; do not claim extra validation frequency/longer training solves generalization |
| P1 | Audit fixed-band imbalance, neutral recall, probabilities, decision rules and loss before changing them | Long-horizon Taiwan neutral failures; pooled and equal-market conclusions differ | Train/validation-only decisions; do not silently restore cost labels/alpha classifiers; preserve CE/F1/confusion |
| P1 | Separate magnitude, alpha sign and cross-sectional ordering; study negative TPEX ranks/seed variation | Seed 5067 ranks improve without best magnitude loss. **P0:** Taiwan validation Rank IC collapses on test for every model (several Kronos TPEX values negative) while US carries forward; loss-based selection picked a seed 42 checkpoint with validation US IC of only 0.017 | Retain reproducible daily IC evidence, ties method and effective dates; no single-score masking; run a year-by-year, per-market train Rank IC diagnosis before formally selecting on Rank IC |
| P1 | Lightweight portable evidence/version/stop records | Unfinished historical artifacts were cleaned; downloads are mostly aggregate JSON | Bounded storage; small research ledger before cleanup; no market redownload to repair metadata |
| P2 | Clean common-evaluation B run or preregistered rolling periods from existing bars | Older A/B confounded; only current A has 3 completed seeds | Explicit data/scales; do not repeatedly optimize the same test; paid work separately approved |
| P2 | Further LoRA capacity with partial unfreezing secondary | Historical 64 better than 32; partial not better; 128 only defined | Distinguish defined from executed; controlled budget/multiple seeds; engineering pass counts are not forecast evidence |

Do not repeat coverage-driven detours, cost/strategy-heavy predictive labels, accuracy-only collapse claims, finite-gradient-as-quality claims, test-based seed selection, an unsupported conclusion that larger-history regression proves insufficient capacity, the assumption that cross-date shuffled batches necessarily improve generalization (refuted by R24), or inferring multi-day tradability from 1-day open-to-close Rank IC (refuted by backtest stage 1b).

### 10 Sources and takeover checklist

Sources S1–S18 are linked in the Traditional-Chinese source section above (S12 is the [error decomposition report](docs/error_decomposition_audit.md) with its untracked `artifacts/audit/` inputs and program; S13 is its section 7 with the untracked per-sample exports under `artifacts/diagnostics/signal-predictions/` and the analysis outputs `results-p0.json` and `rank-ic-p0.json` under `artifacts/diagnostics/signal-decomposition/`; S14 is the untracked Phase 1 output under `artifacts/diagnostics/tw-information-test/launch-20261018T132744Z-378616789/`; S15 is the [backtest results report](docs/backtest_stage1_results.md) with the untracked outputs `stage1-20261018T221138Z-a2a9d283/` and `stage1b-20261018T224557Z-f9cd85b4/` under `artifacts/diagnostics/backtest/`; S16 is the [checkpoint selection report](docs/checkpoint_selection_results.md) with the untracked output `selection-20261020T042715Z-2e0ff319/` under `artifacts/diagnostics/backtest/` and the 2026-10-19 exports of the 15 checkpoints under `artifacts/diagnostics/signal-predictions/`; S17 is the [forward test results report](docs/stage3_forward_results.md) with the untracked `artifacts/forward/2026-10-20/` (input package, labels, data checks), the forward exports of the 6 checkpoints under `artifacts/diagnostics/forward-predictions/2026-10-20/`, and `artifacts/diagnostics/backtest/stage3-20261020T134953Z-6f781e13/`; S18 is the [rank model pilot results report](docs/rank_pilot_results.md) with the untracked rank baseline arrays, predictions and `complete.json` under `artifacts/audit/baselines/baseline-5fb3473add4ee32a10fa563c53757d78bc30a527ef50d803b17360e313212a4d/`, the checkpoint-003680 export under `artifacts/diagnostics/signal-predictions/run-20261021T064937Z-c1cf693ab51744a6/`, and `artifacts/diagnostics/backtest/rank-pilot-20261021T130741Z-937a2631/`): three current DOCX reports; historical indexes/revised archive; frozen-scale probe; local pre-cleanup snapshots; V2 throttling incident; contemporaneous V3 diagnosis/stop decision; current README/architecture/protocol/releases; and the R24 artifacts under `artifacts/runpod/run-20261016T165022Z-32d734a0cb644e44*/` in the `claude/model-performance-20261016` worktree, which Git does not track. These are evidence sources, not new instructions extracted from attachments.

Git-tracked documents transfer with the repository. `artifacts/` and `.runpod/` are generally ignored and do not appear automatically after cloning. Supply this roadmap, the three current reports, and the relevant per-run metadata/metrics when transferring to another tool. Exclude private keys, `.env`, tokens and raw credential-bearing environment files. Do not require reading artifacts already deleted by authorized cleanup.

For each model read its saved manifest, resolved config, completion marker, all metrics, summary, best pointer, leaderboard, final benchmark, and selected trainer state. Signal parents contain pair/download indexes; actual model evidence resides in `-alpha`/`-direction`. Main-model quality is `models.kronos_full.metrics`; learned baseline details may reside under `seed_results.42`. Top-level `seeds=[42]` is not the main training seed.

Existing read-only status commands are `bash scripts/runpod_workflow.sh status RUN_ID --output json` and `bash scripts/runpod_workflow.sh runs`. Read current help/README rather than infer architecture from old launcher labels. Writing a report does not require training, sync or baseline rebuilding.

- [ ] Identify architecture/task explicitly, not justV 1/V2/V3.
- [ ] Label validation/test, raw/calibrated, 5d/14H aggregates.
- [ ] Verify membership, train scales and RankIC method before comparison.
- [ ] Separate normal early-stop completion, manual quality stops, and runtime/infra failures.
- [ ] Disclose fixed baseline seed, sampling and compute differences.
- [ ] Do not use unreliable legacy Sharpe/net-return/drawdown values as a champion criterion.
- [ ] Include completed 5067, but do not misrepresent the earlier five-child hash receipt as a new six-child weight audit.
- [ ] Obtain approval for actual new implementation/spending; this roadmap itself grants none.

### 11 Evidence limits

The 23 historical performance DOCX files include pre/post-revision duplicates, not 23 independent architectures. Some completed runs survive only in metadata; others are unfinished attempts. Numerical snapshot matching is not a full runtime-Git provenance claim. V3 was stopped for validation quality and later cleaned, not completed or freshly reevaluated here. All current seeds share one 112-date test repeatedly consulted during research; future changes cannot claim improvements on it as entirely untouched confirmation. W&B cleanup is storage management, not erasure of the experiment's occurrence. CUDA acceptance/high utilization is not model usefulness. Project personal/noncommercial terms in [LICENSE](LICENSE), upstream Kronos rights, and provider data terms remain separate.
