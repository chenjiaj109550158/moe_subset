# 08 — 自主執行、可恢復runner與最後驗收

## 必須真的跑，不只留下指令

Agent讀完規格後先實作本階段需要的最小工具與修正，執行tests，查結果，通過就自動下一階段。一般錯誤自行修正，超過bounded retry或遇外部blocker才轉terminal。不要停在「環境好了」「kernel接好了」「建議下一步跑benchmark」。

Runner負責可重現執行，不是假設shell腳本可以自己思考改bug。Code agent在同工作階段監控runner、解釋失敗、改程式後重跑；在工具可等待時持續等到結果。不要啟動nohup後立即回覆完成；若工作階段被平台終止，留下INTERRUPTED與可resume狀態。

## 新增的重跑介面

以下是**要新增的interface，不保證目前repo已有**：

```bash
# 環境建立命令由實際相容版本寫進此腳本，無sudo且可重複執行
bash scripts/restart/bootstrap.sh --config configs/restart/restart_v1.yaml

# 已建立環境後，完整重新執行或接續
python -m pseudoroute.restart.run_all \
  --config configs/restart/restart_v1.yaml \
  --run-dir artifacts/restart_v1/<run_id> \
  --resume

# 不重跑模型，核對證據、重新計算aggregate/decision/report
python -m pseudoroute.restart.verify \
  --run-dir artifacts/restart_v1/<run_id>
```

Bootstrap script 不能替父 shell 永久 activate 環境。它必須輸出環境 interpreter 的絕對路徑；後續測試與執行都使用該 interpreter（上面的 `python` 是示意），REPRODUCE/RESUME 要寫出實際路徑或可移植的 env 啟動命令。

舊 runner 可能硬編碼 PILOT_ID、artifact root、config hashes 與授權 gate。不能只換 YAML 就執行舊 runner；新 runner 的所有輸出必須限制於本 run root，並對寫入歷史 protected paths 作 fail-fast 檢查。共享純 helper 可以，但不能靠 monkeypatch 舊全域輸出路徑冒充完整隔離。

實際新CLI可重用repo現有CLI與run-directory元件，但最後報告中的命令必須真的可執行。`--help`與`--resume`至少有測試；不能文件裡宣稱有指令但未實作。

## State與resume

每個run保存 `STATUS.json`（原子寫入）與 `STATUS.md`。包括phase/step、attempts、source/config/model/env/sample hashes、開始／完成時間、已用GPU/wall budget、outputs、last error、下一個實際command、terminal state。

逐sample/policy/repetition原子寫row，key包含checkpoint/config/code/manifest/backend/hardware identity；相同key且checksum有效才可skip。中斷的temp row不算完成；source change使所有受影響rows失效，不能拼出偽完整aggregate。

Lock不可只靠PID判斷存活；包含host/process start/run ID，只有確定屬自己且已死的lock才清理。Shared cache下載使用library安全鎖。保存log但避免在inner loop逐token刷檔。

持久化first/second freeze、source diff、每次修補test結果。至少保留一個可直接重跑單個失敗case的command。

## 自動budget與恢復策略

在P0、P2微基準後、P4 dev後各估一次成本。估算是本地排程資料，不是把未來結果當測量。追蹤實際device worker elapsed×GPU數，不能用repeated tests的想像速度消耗預算。

Resource降級只按02/06預定規則且在看confirmation結果前決定。GPU不足→CPU tests與小型GPU可行項目；pinned不足→明示共同host模式；主模型RAM不足→BLOCKED主模型；剩餘時間不足→較小預先允許N或停止並report。不可擅自換量化/模型/fake data「跑完」。

科學負結果可正常exit 0且state COMPLETE；程序錯誤/blocked/budget interrupt要另外machine-readable。保留所有結果，使agent不因非GO標籤陷入無限修參數循環。

## 必要artifact tree

```text
artifacts/restart_v1/<run_id>/
  STATUS.json / STATUS.md
  source_manifest.json / source_changes.patch
  hardware.json / environment_manifest.json / environment.lock.txt
  resource_resolution.json / model_manifest.json
  protocol_frozen_1.yaml / protocol_frozen_2.yaml
  development_plan.json / samples_manifest.json / artifact_manifest.json
  AUDIT_FINDINGS.md / correctness_report.md
  tests/                         # logs, junit, exact commands
  backend_candidates.json / backend_selection.json
  backend_correctness.json / moe_microbench.csv
  profiles/                      # trace files + interpretation
  runs/                          # atomic per-row measured results
  aggregate.json / comparison.csv
  DECISION.json / FINAL_REPORT.md
  REPRODUCE.md / RESUME.md
```

可依legacy infra調整，但每一類資訊必須有唯一可導航位置。未執行的類別用狀態/原因，不建立假CSV數據。外部clone failure時先在WORK_ROOT外部run folder寫相同終端報告。

## 最後的自動核驗

重新跑受影響CPU/GPU tests、selector causality/cache-race smoke、當前env `pip check`、新module lint/type checks；主要驗證不僅import。舊不相關失敗列明，不能全標PASS。

由rows重算totals/CI/decision並核對report：任何報告數字都能回到row集合。核驗相同memory/precision/backend、decode分母沒有pseudo tokens、stageH2D沒有混prefill、無零值代替缺資料、沒有新future/label leakage、required comparisons完整。

本次改動不能意外修改historical frozen config/artifact；檢查git diff與原manifest。保留licenses。Verify可離線、不重新載模型地驗source/config/row hashes和統計結果；模型checksum檢查可另標快速/完整模式。

測試decision.schema.json可接受最後DECISION。Evidence paths必須存在；不可引用預計要生成但不存在的檔案。Report的GO/NO_GO和machine result必須相同。

## Agent最後回覆格式

用繁體中文，先列overall與三個dispositions，再給最關鍵的一張測量表、已修正的主要原因、目前限制、FINAL_REPORT/DECISION/reproduce檔案位置。說明哪些真的在本GPU跑過。

沒有主模型實測，就明說沒有；已完成完整測試但沒收益，也直接說。本輪已授權的動作做完或有可驗證終止原因後才交付，不能只提出下一輪計畫。
