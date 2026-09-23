# MoE Subset 重啟執行包

版本：2026-09-22 / restart-v1

這份文件包交給**能讀寫檔案、執行 shell、安裝隔離環境及使用本機 GPU 的 code agent**。目的不是再寫一份計畫，而是在新機器上完成 clone、修正、實測與有限範圍的研究判定。

這是待實作的規格，不是已修改完成的 repository，也不包含模型 weights。編寫時已透過公開來源核對主要程式與設定；沒有在你的新機器上執行 GPU 測試。

## 使用方式

把本資料夾解壓到工作目錄，在該目錄開啟 code agent，貼上 `00_MASTER_PROMPT.md` 全文。Agent 應自行讀完其餘文件，逐階段實作、執行、驗證，不需要使用者逐份貼 prompt。

預設 repository 會放在工作目錄的 `moe_subset/`。已存在 repository 時，不覆寫；由 agent 建立自己的 branch/worktree。模型與套件快取放在版本控制之外。

`restart_defaults.yaml` 是本次新實驗的資源授權與預設值。它允許在容量檢查後下載**指定的一個公開 checkpoint**及必要資料；不允許付費雲端、sudo、驅動修改、任意模型下載、push 或破壞既有工作。GPU 48 小時與 wall 72 小時是**執行上限，不是完成時間預估**。Code-agent 自身的工具與工作階段上限仍可能先到。

## 閱讀順序

| 文件 | 職責 |
|---|---|
| `00_MASTER_PROMPT.md` | 一次交付給 agent 的主指令 |
| `01_EXECUTION_CONTRACT.md` | 範圍、授權、資訊邊界、何時停止 |
| `02_BOOTSTRAP_AND_AUDIT.md` | Clone、硬體、版本、資料、CPU-first loader |
| `03_CORRECTNESS_AND_HARNESS.md` | 已知問題、回歸測試、cache/RNG/數值 gate |
| `04_GPU_MOE_BACKEND.md` | Slot-aware GPU MoE、現成 backend、微基準 |
| `05_SUBSET_POLICIES.md` | Fixed/history/pseudo/cost-aware 策略與因果性 |
| `06_EXPERIMENT_PROTOCOL.md` | 公平性能、品質、樣本與統計規範 |
| `07_DECISION_RULES.md` | 可執行判定規則，不把 blocker 當研究失敗 |
| `08_AUTOMATION_AND_DELIVERABLES.md` | Runner、resume、驗收與最後交付 |
| `09_SOURCE_MAP.md` | 核對來源與 repository 導航 |

另附 `decision.schema.json`、`ALL_IN_ONE.md` 與 `SHA256SUMS.txt`。`ALL_IN_ONE.md` 只整合 Markdown；機器可讀設定與 schema 仍以各自原檔為準。

## 最後應拿到什麼

Repository 中的 `artifacts/restart_v1/<run_id>/FINAL_REPORT.md`、`DECISION.json`、實際命令與環境鎖檔、測試／profile／逐樣本結果、程式修改、可重跑入口與 resume 狀態。

判定分開回答：runtime 是否修好、window subset 是否值得繼續、pseudo predictor 是否值得繼續、目前機器與樣本範圍能支持多強的結論。不能只回報「測試通過」，也不能因安裝或資源失敗宣稱方法不可行。
