# 本輪修改紀錄

基底 commit：`9f182729ee33a1737e1203e2693fb903d11e62cf`。所有工作在隔離 clone 的 `restart/20260922T031933Z` 分支；沒有 push，也沒有覆寫歷史協定或 artifacts。完整可套用差異為本目錄的 `source_changes.patch`，包含 tracked 修改與新增檔案。

## Runtime 與正確性

- `restart/stopping.py`：每個 request 建立一次停止條件，保留 prompt 加 continuation 的原生 stop 語意；原 helper 留作歷史重現。
- `restart/model.py`：meta/sharded safetensors CPU-first loader；expert 權重完整留在 CPU RAM，GPU 只有 dense weights 與每層 32 slots。
- `restart/backend.py`、`residency.py`：exact/subset 共用官方 vLLM BF16 fused experts；保留 active-only Python reference。Slot map、generation、compute-completion/copy-ready events 共同保護生命週期，禁止 token-by-token selected-weight materialization。
- `restart/inference.py`：真正自己的 greedy trajectory 與 KV；pseudo 只 commit bridge，planning RNG 隔離；必要的 bootstrap、H2D、等待和末尾 drain 都進入 decode wall。Metrics formatting 在 wall 結束後。
- 既有 `benchmark/qwen_real_offload_speed.py`：修正 wall end 與 metrics 格式化順序。既有 `benchmark/subset_closed_loop.py`：CPU 測試路徑不再無條件碰 CUDA RNG。`config.py`：CUDA 要求不可靜默退回 CPU。

## Selector 與實驗工具

- `restart/policies.py`：獨立定義 residency window d、pseudo 計算 G、legacy 內容跨度、core anchors；H/HC 與 legacy P4/P8、PW4/PM4 分別命名，不偷換 legacy core 優先語意。
- `restart/microbench.py`、`diagnostics.py`：resident 單層、真 checkpoint 數值與同步對照、G4/G8 語意、同機 compute ablation 與 profile。
- `restart/data.py`：固定 GSM8K revision、seeded development/confirmation IDs、歷史開發題排除、prompt fixtures；gold 不進 online selector。
- `restart/state.py`、`prepare.py`、`run_all.py`：owned run root、PID/start-time locks、原子 rows/checksum、source/config/env identity、budget 與自動階段推進。
- `restart/decision.py`、`report.py`、`evidence.py`、`verify.py`：純函式 gate、保守 paired quality CI、request-cluster speed CI、schema/report/row 離線核驗。
- `scripts/restart/`、`configs/restart/`：隔離環境 bootstrap、硬體與容量探測、授權模型下載、鎖定版本與新協定。`tests/restart/` 保留 CPU/GPU/CLI/decision regressions。

## 已保留的修補與校準紀錄

初始環境相容性失敗、resident reference 掃描 inactive experts 的不公平 timing、真 checkpoint 原 full-prefix NRMSE gate 失敗、數值定位與重測，全部保留於 `tests/`、`attempts/attempt1/` 及 `numerical_diagnosis*.json`。數值 v2 的例外依據、凍結時點與接受範圍見 `docs/restart/numerical_protocol_v2.md`（repository 路徑）與本目錄 `correctness_protocol.json`；它不是原 1% 門檻通過的聲稱。

正式量測期間，155 個已凍結 source/config/test 檔案保持不變。後續文獻說明、進度 checksum 與從既有 rows 計算的描述性邊界延遲分析，僅寫在此 run 的 artifacts 內，沒有修改執行來源或預先註冊 gate。

最終測試狀態以 `test_inventory.json`、`correctness_report.md`、`verification.json` 與 `FINAL_REPORT.md` 為準；舊資料缺失的測試不會被改寫成 PASS。

## P6 報告修補

全部模型 rows 完成後，報告的三個 NumPy 布林欄位造成 JSON serialization 失敗。僅修改 `report.py` 的布林輸出轉換及 `verify.py` 的報告來源核驗，新增 `tests/restart/test_reporting.py` 兩項回歸。原 155 個來源中的另 153 個檔案未改；兩個原始檔案、失敗日志與原始 aggregate 保存在 `attempts/p6_reporting_failure/`。修補前後全部統計及 decision 逐項相同，35 項最終回歸通過。`reporting_repair.json` 記錄 old/new/archive hashes；不允許 runtime/selector/backend/decision 檔案利用此例外通過 source 檢查。`execution_source_changes.patch` 保留測量版本，最終 `source_changes.patch` 含報告修補並已於乾淨 checkout 套用核對 156 個檔案。
