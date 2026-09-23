# MoE subset 重啟研究報告

執行狀態：**COMPLETE**。整體判定：**NO_GO_TESTED_REGIME**。

本輪在 RTX PRO 6000 Blackwell 實際完成 1,261 列模型測量，包含 256 題 × E/S/H 的 768 列正式品質，以及 8 題 × 5 repetitions × 3 方法的 120 列正式性能。

1. **Runtime：VALIDATED（correctness v2 範圍內）**。共用 GPU backend、CPU-first loading、停止條件、KV/RNG/slot/event 生命週期已修正並驗證。
2. **Window subset：STOP_TESTED_REGIME**。選定 H 同時失去品質與速度；停止本輪配置，保留 runtime 成果。
3. **Pseudo：INCONCLUSIVE**。P4 未晉級，沒有比便宜 history 更值得投入的確認證據；本輪不擴大此分支。

**數值條件必須一起閱讀：** 原 full-prefix NRMSE ≤ 0.01 未通過（0.0179627）。依規格 03/C08 的 native/vendor 例外，在任何方法輸出與 confirmation 前保存失敗、以 unchanged native 對照定位 BF16 累積差異，另凍結 v2 的跨實作 full-prefix ≤ 0.03。單層 ≤ 0.01、cosine ≥ 0.999 不變；同 backend 同步／event ordering 必須 bitwise 一致。三個預列 prefix NRMSE 為 0.017963、0.019470、0.014101；48 層單層最差 0.007925，同 backend 同步／async 差異皆為 0。這不是原門檻通過或任意 prefix 的數學誤差保證。詳見 `correctness_protocol.json`、`correctness_report.md` 與 repository 的 `docs/restart/numerical_protocol_v2.md`。

## 正式結果

| 方法 | 正確題數／256 | 準確率 | L1 forwards/s | 相對 E 配對速度 [95% CI] | L1 expert MiB/forward | L1 p95 token-ready ms |
|---|---:|---:|---:|---|---:|---:|
| E | 250/256 | 97.66% | 11.506 | 1.000 (reference) | 1024.29 | 100.22 |
| H | 195/256 | 76.17% | 10.351 | 0.899 [0.885, 0.915] | 628.87 | 193.35 |
| S | 141/256 | 55.08% | 12.252 | 1.065 [1.044, 1.086] | 62.23 | 91.60 |

H/E 的 paired gains/losses/ties = **1/56/199**；accuracy delta = **−21.484 pp**，multiplicity-adjusted 保守 CI **[-29.403, -12.525] pp**。S/E = 1/110/145，delta −42.578 pp，CI [-51.399, -31.872] pp。H 相對 S 仍有 +21.094 pp（80/26/150），CI [7.713, 33.609] pp，所以 D 通過；Q 與 S 明確失敗。速度 CI 上界也低於 1.05，負結論不依賴『沒有顯著差異』。

L1 固定 128 個 post-prefill production forwards，greedy 自身 trajectory、忽略 EOS，只用於固定工作量；第一個 prefill sample 與 pseudo positions 不算分母。全部 bootstrap、probe、H2D、等待與 drain 計入 decode。品質使用相同 optimized E 的真正自由生成，沒有 future replay。CI 方法與 J=3 在第二次 freeze 已固定。

## 真正自由生成的成本

| 方法 | 平均 request 秒 | 平均生成 tokens | TTFT 秒 | aggregate TPOT ms | 截斷／256 | invalid／empty | 重複 proxy 題數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E | 28.15 | 289.77 | 3.010 | 87.05 | 5 | 0/0 | 0 |
| H | 39.15 | 387.34 | 3.017 | 93.53 | 42 | 0/0 | 29 |
| S | 47.54 | 550.18 | 3.028 | 81.05 | 97 | 0/0 | 90 |

所有方法 cap=1024；development 的 exact 512-cap 截斷 2/32 (>5%)，因此在第二次 freeze 前統一提升並重跑，舊 512 結果保留。正式品質 E/H/S 截斷率為 1.95%／16.41%／37.89%。沒有缺列或 generation timeout。重複 proxy 是「同一 16-token 片段出現至少 8 次」的事後描述，不能當作預先註冊顯著性 gate；逐樣本記錄見 `CONFIRMATION_DETAILS.json`。主要 Q 結論本身已足夠。

正式 free-generation CUDA allocated 峰值約 17.12 GiB，reserved 約 19.75 GiB，RSS 約 80.29 GiB；每層 32 expert slots 共 13.5 GiB，CPU pinned expert store 54 GiB。所有方法同 dtype、slots、dense/KV 與總 envelope；96 GiB 實體 VRAM 不代表主比較允許全部 experts 常駐。數值校準的 128-slot reference-only 診斷峰值約 71 GiB，未用於主性能比較。

## 為何減少搬運仍沒有加速

H 的 decode expert bytes 減少 38.60%，但每層依然計算 top-8 experts。一般位置 H 平均 82.41 ms、E 86.57 ms；換窗後首個位置 H 182.61 ms、E 87.76 ms；含 bootstrap 的首個 forward H 395.58 ms、E 103.15 ms。每個 128-forward request 的一般位置共省約 0.475 秒，起始與更新位置多花約 1.715 秒，實際 decode wall 多約 1.241 秒。

本版 H/HC 等完整 boundary forward 結束才遍歷各層更新 subset，因此可重疊的提前量不足。上述位置分解支持更新成本抵銷一般位置收益；selector、host scheduling、H2D 與同步的個別因果份額仍未完整分離。原 boundary flag 在舊 window 末 token，等待在下一 token；primary pooled p95 沒有漏計。獨立 profiler 含 prefill 和 16 個 forwards，不能把 CUDA self-time 比例當成 decode critical-path 比例。完整來源與數字見 `profiles/PROFILE_REVIEW.md`、`profiles/confirmation_boundary_analysis.json` 及 raw traces。

共用 backend 的工程收益確實存在：同機單題 16-forward ablation，E 的 Python reference 5.780 → fused 12.229 forwards/s（約 2.12×）；S 7.065 → 12.342（約 1.75×）。這是修正 harness 的 compute ablation，trajectory 可因 BF16 分歧，不能當作整套歷史 runner 或固定路由純 kernel 因果倍數。Resident 單層有 240 列公平 timing，全部必要數值案例通過。停止條件 CPU component 的 5 個平衡 repetitions，128 次檢查中位數 85.487 → 0.674 秒；它不包含整個模型推論。

## Pseudo 與文獻對照

P4 前 8 題 development 為 7/8，最佳 H/HC 為 8/8，未通過預列晉級 gate；PW4/PM4 跳過，沒有把缺測寫成支配或普遍無效。P8/P4 只做小型同條件診斷。G4 保留前四個 anchor IDs、prefix、positions 與 RNG，但 BF16 shape 差異使 bootstrap/joint route ID agreement 約 89.91%／82.49%，完全相同 subset 的層數為 42/48、27/48；不能聲稱 G8→4 完整 bitwise 等價。

過去作品可由不同機制加速：ProMoE／MoE-Infinity 提早預測與預取真正會用的 expert；Fiddler 讓部分未駐留 expert 在 CPU 計算、傳 activation；其他 offloading 工作結合低 bit 量化。模型、精度、硬體、baseline 與是否限制 routing 都影響可比性。本輪沒有實作所有這些機制。原始論文連結與版本限定見 [LITERATURE_CONTEXT.md](LITERATURE_CONTEXT.md)。

## 稽核、修補與資源

35 項最終 CPU/GPU/CLI/decision/reporting regressions 全部通過，pip check、ruff、17-module mypy 皆通過。完整 checkpoint 有 55 個數值／同步檢查與六種 policy 的 19-forward smokes。舊 suite 原先 276 pass／8 fail／10 skip，兩項修正後 278 pass／6 fail／10 skip；剩餘六項是缺少歷史 v17 外部 token-row files，沒有標成 PASS（`tests/cpu_after_fixes.log`）。compute-sanitizer 不可用，已列驗證範圍。

P6 首次報告因 NumPy bool 序列化失敗，main process exit 1，即使 STATUS 曾先寫 COMPLETE，當時也不算通過最後核驗。原始狀態、來源、日志與彙總保存在 `attempts/p6_reporting_failure/`。只修補 report.py 的布林輸出與 verify.py 的嚴格報告來源核驗，新增兩項回歸；153 個其他 frozen source 檔未改，所有 1,261 列原始資料保留。原／修補彙總逐項相等，閾值、公式與 labels 不變；修補後離線 finalize/verify 正常 exit 0。詳見 `reporting_repair.json`、`P6_REPORTING_FAILURE.json`。

`execution_source_changes.patch` 是量測時的 source；`source_changes.patch` 是含報告修補的最終版本。後者已在乾淨基底 checkout 套用，156 個 source/test 檔逐一 checksum 相同。修改紀錄見 [MODIFICATIONS.md](MODIFICATIONS.md)，C01–C08、loader、backend 的狀態／修補／remaining scope 見 [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) 與 [correctness_report.md](correctness_report.md)。舊 protected artifacts/configs 的 git diff 為空；所有失敗、環境候選與 trials 均保留。

截至交付審閱，calendar wall 13.882 小時；main GPU worker elapsed 11.613 小時。另將前置檢查 1.860 小時與報告修補回歸 0.048 小時作保守 GPU 預算上界扣帳，總計 13.521 小時，低於 48 GPU-hour／72 wall-hour 上限及 10% 收尾保留。這些上界不是 GPU busy-time 實測；詳見 `BUDGET_LEDGER.json`。

## 研究動作與可重現性

停止本輪 m=32,d=8 的已測 window selector 配置，保留共用 runtime。現在沒有值得擴大 pseudo 投入的確認證據。若日後另啟研究，最具決定性的缺口是：相同品質與記憶體下，能否把換窗等待移出關鍵路徑；目前沒有此項實測正證據。本輪沒有追加 post-hoc d/m/beta 搜尋或重跑全部 oracle。EP、custom kernels、其他 m/d 與 gated-out pseudo 分支均未作正式確認，不外推到所有模型或預取方法。

本機主模型、微基準、四格、development、120 列 performance、768 列 quality、修補與回歸都已實際執行。精確執行／失敗命令在 `tests/*.log` 和 trial inventory。正式狀態與判定在 [DECISION.json](DECISION.json)，重現／續跑入口在 [REPRODUCE.md](REPRODUCE.md)、[RESUME.md](RESUME.md)。

## 自動產生的完整測量表與設定


本輪判定：**NO_GO_TESTED_REGIME**；執行狀態：**COMPLETE**。

1. Runtime：**VALIDATED**。
2. Window subset：**STOP_TESTED_REGIME**。
3. Pseudo predictor：**INCONCLUSIVE**。

本機實際完成 1261 列模型執行，正式品質列 768、正式性能列 120。未完成的主比較沒有填造 accuracy、speedup 或信賴區間。GPU resident / tiny 結果不能替代主模型結論。

## 主要測量

| Policy | paired quality n | correct | controlled forwards/s |
|---|---:|---:|---:|
| E | 256 | 250 | 11.505726918352234 |
| H | 256 | 195 | 10.351272024813277 |
| HC | 0 | None | None |
| P4 | 0 | None | None |
| P8 | 0 | None | None |
| S | 256 | 141 | 12.251713972945945 |

完整配對 g/l/t、CI、bytes、p95、長度、truncation、invalid/empty 與記憶體請見 `aggregate.json`、`comparison.csv`、`policy_summary.csv`、`runs/`。Controlled workload 固定 128 個 post-prefill production forwards，第一個 prefill sample 和 pseudo positions 不算 production 分母。

## 配對確認與 gate

| 配對 | n / g / l / tie | accuracy delta [CI] | decode speedup [CI] | bytes 減少 | p95 比 |
|---|---|---|---|---|---|
| H vs E | 256 / 1 / 56 / 199 | -0.21484375 [np.float64(-0.29403112629866446), np.float64(-0.12524787496028597)] | 0.8994662177858891 [0.8845513832679271, 0.9145872742470152] | 0.38604439639268584 | 1.9292135244723876 |
| S vs E | 256 / 1 / 110 / 145 | -0.42578125 [np.float64(-0.5139861218255837), np.float64(-0.31872020211812274)] | 1.0647806455222573 [1.0444421175551957, 1.0862376837817305] | 0.9392488480448941 | 0.9139904767510436 |
| H vs S | 256 / 80 / 26 / 150 | 0.2109375 [np.float64(0.07713490049476471), np.float64(0.33608675761742)] | 0.8447432075032841 [0.8333661488994025, 0.8562638914754466] | -9.106073446327684 | 2.1107588903225243 |
| S vs H | 256 / 26 / 80 / 150 | -0.2109375 [np.float64(-0.33608675761742), np.float64(-0.07713490049476471)] | 1.183791702753777 [1.167864264691696, 1.1999527474559768] | 0.9010496009839135 | 0.47376325386325857 |

Accuracy delta 與 bytes reduction 使用 fraction；乘 100 才是百分點／百分比。Gate 狀態：`{'V': 'PASS', 'Q': 'FAIL', 'S': 'FAIL', 'D': 'PASS', 'P': 'INCONCLUSIVE'}`。

研究動作：停止本輪 m=32,d=8 的已測 selector 配置，保留通過驗證的共用 runtime。

## 同機 runtime 四格

| backend / policy | forwards/s | decode wall s | expert bytes/forward |
|---|---:|---:|---:|
| python_reference / E | 5.779579546117733 | 2.7683674690051703 | 1159004160.0 |
| vllm_fused / E | 12.22909403887421 | 1.3083553000033135 | 1149566976.0 |
| vllm_fused / H | 10.834782070672503 | 1.4767255949991522 | 857014272.0 |
| python_reference / S | 7.065248760151996 | 2.264605329997721 | 524353536.0 |
| vllm_fused / S | 12.341961377430987 | 1.2963903799973195 | 523173888.0 |

這是同一修正 harness 的 expert-compute ablation，僅用於歸因；它不代表整套歷史 runner 的端到端倍數。

## 已完成工程與限制

- C01：request stop state 一次建構，保留 prompt+continuation native stop 語意。
- C02：audit/profile/performance 共用數值路徑；必要 device drain 在 wall end 前，metrics formatting 在外。尚未測的 event timing 為 null。
- C03/C04：d/G/content horizon 分離；先建 legacy H=8 內容再取前四；legacy core 硬優先保留，HC 為獨立 policy。
- C05：native FP32 softmax/top-k/normalization，hard reroute 與 pseudo zero-missing 分離；invalid 非零 slot 觸發錯誤。
- C06：bridge-only KV commit、CPU/GPU RNG、兩次 window transition 與 cache prefix invariants；詳細範圍以 tests 為準。
- C07：slot generation、整組 kernel completion、copy-ready 依賴與 churn/m=k tests。
- C08：BF16 對同已量化 weights 的 FP32 參考，固定路由完整 prefix；未通過時禁止方法結論。
- CPU-first loader：sharded safetensors 直接建立 CPU expert store；dense 在 GPU，每層 32 slots；禁止 full-model CUDA staging。
- 共用 GPU backend：vLLM 0.11.0 官方 fused experts；所有 E/S/H/HC/P 使用相同 backend、精度與 slot 預算。

各項狀態及剩餘缺口由 `STATUS.json`、`AUDIT_FINDINGS.md`、`correctness_report.md` 和測試 XML 列明；上列實作描述不等同全部 gate 已通過。

## 環境與來源

Repository SHA `9f182729ee33a1737e1203e2693fb903d11e62cf`；模型 revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`；BF16；RTX PRO 6000 Blackwell 96 GiB，與舊 A100 並非同機比較。CPU、RAM、cgroup、driver/runtime、NUMA/PCIe 與 pinned smoke 見 `hardware.json`。依賴及下載來源見 `environment.lock.txt`、`environment_manifest.json`、`model_manifest.json`。資源與主 m/d 在任何新方法输出前解析。

## 判定範圍與歸因

同機舊 Python / 新 backend 四格列在 `runs/runtime_grid/`；不存在的列表示未測。Resident 微基準與 profiles 是工程診斷，不能把 kernel 倍數、bytes 下降或舊 oracle 8/8 換算成端到端改善。未完成 profile 的 wall time 歸因保持 unresolved。舊 artifacts 僅 provenance，沒有用作本機性能或品質 reference；既有 oracle 不重跑。

## 完整性、預算與恢復

目前 wall 49335.1 秒；GPU worker elapsed 41805.1 秒。前置 GPU 檢查未完整集中記錄 worker 起訖，因此另外以整段前置 wall 6696.8 秒作保守 GPU 預算扣帳上界，不冒充實測 device elapsed。總上限 72 wall-hours / 48 GPU-hours，保留 10% 收尾。每個 row 有 identity/checksum，失敗與缺列不被替換為其他樣本。模型輸出以自己的 KV/trajectory 生成，gold 只進 evaluator。

限制：

- Conclusions apply only to Qwen3-30B-A3B-Instruct-2507 BF16, this RTX PRO 6000, m=32,d=8.
- Exact baseline scope is demand LRU; EP was not implemented.
- Historical v17 external token-row files absent from this checkout; legacy protocol tests report this without rewriting history.
- Resident synthetic timing is not end-to-end or model-quality evidence.
- Correctness v2 revises full-prefix cross-implementation NRMSE from 0.01 to 0.03 after unchanged native controls also failed v1; single-layer 0.01 and cosine 0.999 remain, and sync/event parity must be bitwise.
- No full literature novelty claim; historical eight-row oracle is nondeployable background only.

## 本次命令與正式動作

實際執行與失敗日志位於 `tests/`；未跑或被 gate 阻止的階段見 `STATUS.json`。重現及精確續跑命令見 `REPRODUCE.md` / `RESUME.md`。

決定性證據是同一修正 runtime、相同 m/精度/記憶體預算的完整 E/S/window 自由生成與 fixed-work 配對。沒有這個證據時，不能判方法不可行，也不能主張 pseudo 比 history 值得。

## 分階段性能與自由生成

每階段的實測 rows、整 request、decode、生成長度、truncation、invalid/empty、bootstrap、prefill 與 decode bytes、p50/p95（含 boundary/nonboundary）以及 GPU/RSS 峰值，完整列於 `policy_summary.csv`。Development、diagnostic 與 confirmation 分開，不合併作獨立品質樣本。

## 測試與 trial

`correctness_report.md` 列出每個 JUnit 的 pass/failure/skip，`trial_inventory.json` 保留所有環境、修補、基準與失敗紀錄；`source_changes.patch` 可在記錄的基底 commit 套用。第一版 resident reference 掃描 inactive experts 的 timing 被保留但已失效，不用於最終加速聲稱。
