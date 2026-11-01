# 模型登錄表 / Model registry

## 中文

這個目錄記錄每個 walk-forward 模型**目前的正本 launch**、每一折的預測存放在哪個 launch、哪些折帶有儲存的模型（2026-11-02 規則）。
檔案由本機的 `bash scripts/runpod_workflow.sh download-lgbm <launch-id>` 與 `download-xs <launch-id>` 依 launch 的 `lineage.json`
自動更新，隨程式碼一起提交；不要手動編輯。

- 路徑：`registry/<kind>/<key>.json`，`kind` 為 `lgbm` 或 `xs`；LightGBM 的 key 是 `<模型名>__<特徵規格>`（模型名已含標籤變體，
  例如 `t-lgbm-alpha-residual_v1a__alpha_chip_tw_v1`），跨股模型的 key 是 `<實驗>__<標籤變體或 standard>__<特徵規格>`。
- `current`：正本 launch、排程、`data_end`、P3 區間、它延伸自哪個 launch（`extended_from`）。
- `folds`：每一折的名稱、預測區間、`predictions_launch`（真正持有該折 partial 檔的 launch；正本以符號連結指向它）、`how`
  （`trained`、`linked`、`predicted`、`resumed`）、`models`（有模型檔的折給 `models/fold-XX`）。
- `model_folds`：P3 需要、因此帶有模型的折；`history`：被取代的舊正本與接替它的 launch。

下一次補 P3 折時，以 `current.launch` 作 `--extendLaunch`；舊折的預測到 `folds[i].predictions_launch` 的目錄查。
被連結的 launch 目錄不能從 network volume 刪除（`scripts/runpod_volume_reset.py` 會跳過它們）。

## English

This directory records, for every walk-forward model, its **current launch**, the launch that holds each fold's
predictions and which folds carry saved models (the 2026-11-02 rule). The files are updated by the local
`bash scripts/runpod_workflow.sh download-lgbm <launch-id>` and `download-xs <launch-id>` from the launch's
`lineage.json` and are committed with the code; do not edit them by hand.

- Path: `registry/<kind>/<key>.json`, `kind` = `lgbm` or `xs`; a LightGBM key is `<model name>__<feature spec>` (the
  model name already carries the label variant, e.g. `t-lgbm-alpha-residual_v1a__alpha_chip_tw_v1`), a cross-stock
  key is `<experiment>__<label variant or standard>__<feature spec>`.
- `current`: the launch, schedule, `data_end`, P3 window and the launch it extended (`extended_from`).
- `folds`: per fold its name, prediction window, `predictions_launch` (the launch that really holds the fold's
  partial file; the current launch links to it), `how` (`trained`, `linked`, `predicted`, `resumed`) and `models`
  (`models/fold-XX` for folds with model files).
- `model_folds`: the folds P3 needs and therefore the ones with models; `history`: superseded launches and their
  successors.

The next P3 extension passes `current.launch` as `--extendLaunch`; an older fold's predictions are found under
`folds[i].predictions_launch`. A linked launch directory must never be deleted from the network volume
(`scripts/runpod_volume_reset.py` skips them).
