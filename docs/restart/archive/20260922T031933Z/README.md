# 2026-09-22 restart：完整成果與刪機備份

本輪已 **COMPLETE / NO_GO_TESTED_REGIME**。保留已驗證 runtime，停止本輪
Qwen3 BF16、m=32、d=8 的 window subset 配置。Pseudo 尚未證明優於 history，
正式判定為 INCONCLUSIVE。這個結論不外推到其他模型或保留原路由的預取系統。

- [最終報告](FINAL_REPORT.md)、[正式判定](DECISION.json)、[修改紀錄](MODIFICATIONS.md)
- [逐方法結果](policy_summary.csv)、[完整統計](aggregate.json)、[換窗延遲分析](profiles/PROFILE_REVIEW.md)
- [correctness v2 的適用範圍](correctness_report.md)、[報告修補與原始失敗保存](P6_REPORTING_FAILURE.json)
- [先前作品的加速機制](LITERATURE_CONTEXT.md)、[原執行驗收](FINAL_ACCEPTANCE.json)

Runtime 的 VALIDATED 以已記錄的 correctness v2 為條件，不能宣稱原始 full-prefix
0.01 門檻通過。原始失敗、native/vendor 對照、修補前程式、數值協議與所有量測均保留。
原執行記錄 35 個 restart regression tests 通過；另有 6 個歷史測試因舊 token-row
fixtures 缺失而失敗，並未刪除或改稱通過。

| 方法 | GSM8K 答對 | 相對 exact 的固定工作量吞吐 |
| --- | --- | --- |
| Exact | 250/256（97.66%） | 1.000 |
| History subset | 195/256（76.17%） | 0.899 |
| Static subset | 141/256（55.08%） | 1.065 |

## 保存內容

`archive_manifest.json` 列出原始檔案與壓縮分片的 SHA-256。兩個
`evidence.tar.gz.part-*` 是同一個 gzip tar 的連續分片，包含完整 run 的 1,566 個檔案：
1,564 個原封存清單項目、原封存清單本身、以及已不持鎖的 run.lock metadata。
其中有全部 1,261 筆實測列、三個完整 profiler JSON、數值診斷、失敗嘗試、測試 log、
凍結協議、樣本識別與模型 revision/checksum、原始及最終 source patches。
約 1.056 GB 原始內容壓縮為約 75.0 MB；每個 Git blob 不超過 40 MiB。

此目錄直接可讀的報告為原始位元組副本；完整 archive 中另有全部 supporting evidence。
`external_run_metadata/` 保留原 workspace 的啟動及結果指標。舊報告的絕對路徑是
歷史 provenance，換機後請使用以下相對路徑命令。Artifact 內的 source patch 是原執行
快照；此 Git branch 已包含最終程式，**不要再次套用 patch**。

模型／資料下載快取和虛擬環境不進 Git；指定 revision、checksum、鎖定環境及取得入口
已保存。Evidence 含測試 prompt/答案與生成結果，不是 checkpoint 或完整下載資料庫。

## 從 Git bundle 取回

Git commit 和 bundle 都是本機資料。刪機前須把 bundle 複製到其他機器，或另行授權
推送這個 branch。原規格 `restart_defaults.yaml` 設定 `allow_push: false`，本次封存
不自動變更此設定，也不自動推送。Bundle 含完整 branch 歷史，不依賴原機器或 origin。

```bash
git clone -b restart/20260922T031933Z /path/to/moe_subset_restart_20260922.bundle moe_subset
cd moe_subset
```

## 還原完整結果，不需模型、GPU 或第三方 Python 套件

以 Python 3.10 或更新版執行。預設還原至 `artifacts/restart_v1/20260922T031933Z`，
需要約 1.2 GB 可用空間。目的目錄已存在時會拒絕覆寫。每個還原檔案都會比對 SHA-256。

```bash
python3 scripts/restart/restore_archive.py --verify-only
python3 scripts/restart/restore_archive.py
```

也可加上 `--output /new/empty-parent/restored-run` 指定尚不存在的目的目錄。
還原的 `run.lock` 只是原始 metadata，不代表有程序執行或 advisory lock 被持有。

## 重新計算判定與另起實驗

完整科學核驗使用專案鎖定的環境。建立環境可能需要下載套件；下列 verify 本身
不載入模型、不下載資料，也不重新執行 benchmark。Python 3.12 基底可用
`RESTART_PYTHON_BASE` 指定。實際量測環境完整列於 `configs/restart/environment.lock.txt`。

```bash
RESTART_PYTHON_BASE=/path/to/python3.12 bash scripts/restart/bootstrap.sh --config configs/restart/restart_v1.yaml
USE_HUB_KERNELS=0 .venv/bin/python -m pseudoroute.restart.verify --run-dir artifacts/restart_v1/20260922T031933Z
```

已完成 run 的 `run_all --resume` 只做離線核驗。要重跑實驗，使用新 run 目錄，
模型與資料取得仍遵守原規格授權和上限：

```bash
RESTART_NEW_RUN="artifacts/restart_v1/$(date -u +%Y%m%dT%H%M%SZ)"
USE_HUB_KERNELS=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pseudoroute.restart.run_all --config configs/restart/restart_v1.yaml --run-dir "$RESTART_NEW_RUN" --resume
```

本次刪機封存只驗證還原與離線核驗，沒有再跑一套 GPU benchmark。
