# 給 code agent 的主 Prompt

你現在要在這台新機器上**實際重啟並完成一輪 MoE subset 專案修正與判定**，不是只分析或產生 TODO。

Repository：`https://github.com/chenjiaj109550158/moe_subset`

先定位本 prompt 所屬的 `moe_subset_restart_20260922/` 規格目錄；通常就在目前工作目錄。只搜尋目前工作目錄及其附近，不掃描整個 home。按序讀完 `01_EXECUTION_CONTRACT.md` 到 `09_SOURCE_MAP.md`，以及 `restart_defaults.yaml`、`decision.schema.json`。`ALL_IN_ONE.md` 是同內容的替代閱讀入口，不必重複讀。

接著在本機連續完成：

1. 安全 clone／建立獨立 worktree、偵測硬體、建立隔離環境、鎖定 repository/model/dependency revisions，檢查與取得必要資料。
2. 重新核對舊問題，新增 regression tests，修正停止條件熱路徑、instrumentation、CPU-first offload 初始化、routing/slot/cache/RNG/非同步生命週期與 joint 數值問題。
3. 建立 exact offload 與 subset 共用的 slot-aware GPU MoE backend，先接現成 BF16 fused/grouped expert kernels；不要直接展開無限期 custom-kernel 專案。必要時做有預算的小 token Triton 路徑。
4. 把 residency window `d`、pseudo 計算數 `G`、pseudo 內容生成跨度分開，驗證 G=8→4 是否保留前四個 anchors 的真正語意；保留 legacy 方法作對照，實作便宜 history 與 bounded cost-aware 對照。
5. 在新機器、新協定下執行 correctness、resident 微基準、受控性能與真正自由生成品質測試。Exact 與 subset 必須共用修正後 backend、精度、slot 與總記憶體預算。舊 artifacts 只供 provenance，不當新機器實測。
6. 依預先凍結規則產生 `FINAL_REPORT.md` 和可驗證的 `DECISION.json`，回答問題在哪裡，以及 runtime、window subset、pseudo predictor 各自是否值得繼續。

本次 prompt 授權在設定上限內建立隔離環境、下載指定公開模型／資料、建立新的 protocol/config、執行列出的 bounded experiments 及保留程式修改。它不是要求修改或覆寫歷史 frozen protocols。不要讓舊 `NEXT_CODEX_PROMPT` 的「只做某一階段／等確認」取代本次任務；但必須保留舊協定與結果的歷史語意。

**執行到底的意思是 run-to-terminal，不是保證正結果或無限運算。** 不要每完成一個階段就詢問是否繼續，也不要只寫完 spec/runner 就停止。先讀現有程式再改；每個階段必須有實際測試或明確 blocker 證據。可以跳過 gate 不允許的昂貴分支，但不能跳過報告與判定。

不要為了通過而放寬已凍結 gate、刪除失敗樣本、引用 future tokens、把模擬當實測、偷偷換模型／精度，或只優化 subset 而保留慢的 exact baseline。不要重新從頭跑過去全部 oracle/pseudo 探索；既有 hard oracle 只作小樣本診斷，新的研究價值以本次公平實測決定。

遇到一般程式錯誤，依規格自行修正並重試；遇到真正的 GPU／RAM／權限／網路／預算限制，完成仍可執行的工作，再給 `BLOCKED` 或 `INCONCLUSIVE`。不可將受阻寫成方法不成立。遇到 code-agent session 上限，留下可驗證 checkpoint 和精確 resume command；同一 prompt 再執行時從未完成步驟接續，不能假裝背景工作仍會自行完成。

你應主動執行工具並等候已啟動工作得到結果，直到正式 terminal state。最後用繁體中文說明結論、數字與證據檔案路徑；不要以「接下來可以做……」代替本次已授權的執行。
