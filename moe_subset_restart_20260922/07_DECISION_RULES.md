# 07 — 結論規則：不要把工程、方法與資源混成一件事

實作可測試的純函式 `evaluate_decision(metrics, protocol)`，由實際aggregate產生判定；不準只讓LLM讀表後自由挑標籤。文字討論可以更細，但不能與機器結果衝突。以下門檻是本次實驗預先指定的研究投資標準，不是普遍定理。

## 判定前的有效性 gate V

所有主結論需要：同模型/revision/dtype、正確routing/slot/KV/RNG、同一鎖定backend、相同記憶體限制、measured而非simulated資料、完整必要配對、未洩漏future/labels、pre-registered選擇與統計、可追溯code/config/sample hashes。

無GPU/模型/RAM/權限/時間導致未完成→BLOCKED或INCONCLUSIVE；已知correctness bug未解→PIVOT_RUNTIME/NEEDS_WORK；不能判方法NO_GO。

## 可量化條件

**Q：相對optimized exact的品質保留**

- Paired accuracy delta的預先指定有效95%下界 ≥ -0.02（2個百分點）。
- 無新出現的顯著崩潰／空答案問題；所有timeouts/截斷完整列報。
- Bootstrap全tie得到[0,0]不能作證；用06的非退化區間。

**S：有實用的decode改善**

- Controlled fixed-work paired speedup點估計≥1.10。
- 其預先指定95%下界>1.00。
- Profile沒有顯示排除predictor/bootstrap/H2D/等待等作弊，L2自由生成的整request與TPOT亦完整報告。
- p95 token-ready latency 目標≤exact的1.20倍；若不滿足，不能給無條件GO，改PILOT_PROMISING並明確標示邊界延遲代價。

decode-related expert bytes/token下降30%是supporting target，沿用作診斷，不是替代S的gate，也不是若只有29%就抹掉真實速度收益。記錄locality/cache headroom，別把傳輸量倍數說成端到端倍數。

**D：動態相對static有額外用途**

以下至少一項有配對結果與明確信賴範圍支持：

- Window candidate相對static有正的quality delta，其有效下界>0；或
- Static相對exact的quality delta有效上界<-0.02，而window通過Q；或
- 在預先列出的不同m設定，window以更小m達到同一Q/S目標，static在該m不能達標（不能post-hoc搜尋）。

只有window比exact快、但static也一樣準且更快，不能宣稱定期更新有必要。點估計較好但D不確定可以是PILOT_PROMISING，不必硬造NO_GO或GO。

**P：pseudo比便宜history值得**

在同一d/m與共同預算下，與H/HC的Pareto比較有增量：相近品質更快，或相近速度更準，且没有被便宜history在品質與速度上支配。正式強聲稱依06多重比較處理；資料不足寫INCONCLUSIVE。只是route mass更高、或P4比P8快，不足以證明應保留pseudo。

## Top-level結論

| overall | 使用條件與含義 |
|---|---|
| GO_WINDOW_AND_PSEUDO | V/Q/S/D成立，pseudo額外價值P也有可靠確認；值得繼續，仍非跨模型普遍證明 |
| GO_WINDOW_HISTORY_ONLY | V/Q/S/D成立，便宜history有效；當前pseudo未達額外效益或被支配 |
| PILOT_PROMISING | 有一致實測信號，但CI、N、tail latency、strong baseline或D/P證據不足 |
| STATIC_SUFFICIENT_TESTED_SCOPE | Static已達Q/S，且對window的品質非劣與速度優勢有足夠配對支持；僅此範圍無需動態更新 |
| PIVOT_RUNTIME | 程式、backend、同步或量測有效性未解；不能判subset本身 |
| PIVOT_SELECTOR | 有效runtime與固定resident/既有oracle顯示潛在空間，但可部署selectors不能同時保品質與速度；不等於oracle可部署 |
| NO_GO_TESTED_REGIME | 有效且充分的指定範圍測試，預列方法未達目標；可停止這個配置/實作，而非宣稱任何d/m/模型皆不可能 |
| INCONCLUSIVE | CI跨門檻、樣本不完整、預算用完、欠必要對照或因素無法分離 |
| BLOCKED | 無法開始主實測的外部資源/權限/模型/GPU blocker；保留已完成CPU/單層工作 |

不要用「統計不顯著」自動當NO_GO。要作負結論，需描述究竟排除了什麼：例如在tested point的speedup上界仍<1.05，或quality delta上界仍低於-2pp，且不是bug或缺樣本造成。若只是不足以證明GO，標INCONCLUSIVE/PILOT。

V優先於其他gate。Kernel快2倍、bytes少30%、舊oracle8/8都不能繞過V/Q/S/D。

## 三個獨立 disposition

除overall外，務必各自輸出：

- Runtime：VALIDATED / NEEDS_WORK / BLOCKED。
- Window subset：CONTINUE / PROMISING / STATIC_SUFFICIENT / STOP_TESTED_REGIME / INCONCLUSIVE / BLOCKED。
- Pseudo：CONTINUE / STOP_CURRENT_VARIANT / INCONCLUSIVE / NOT_TESTED / BLOCKED。

Window可以值得做而pseudo值得停；runtime可成功而研究方法沒價值；小樣本無法確認也不是全盤失敗。

## FINAL_REPORT 必備內容

1. 開頭直接回答「哪些是程式問題、哪些是方法限制、值得繼續哪一部分」，以測得數字而非推測說明。
2. 環境/硬體/checkpoint/commit/protocol/shape/m/d/G/host-mode，與舊A100測試的差別。不能把新機器更快歸因為自己的patch。
3. 每項C01–C08＋CPU-first loader＋GPU backend：狀態、證據、修補與remaining issue。
4. 舊Python vs新backend的同機四格對照；S/H/P vsoptimized E/EP的主結果，含wall、bytes/token、品質g/l/t、CI、tail與memory。
5. Hot-path profile與ablation支持的歸因；無法解釋的wall time明寫unresolved，不能全部歸給某個kernel。
6. 既有oracle的範圍與為何不重跑全套；本次是否真正看到了可部署的speed-quality交集。
7. 所有trial、failed/invalid/incomplete rows、統計限制、sample selection、budget使用與legacy artifacts未被改寫的稽核。
8. 清楚列哪些命令真的跑過、哪些未跑、何因跳過。附reproduce/resume命令與檔案路徑。
9. 最後給一個明確研究動作：繼續window/history、繼續pseudo、轉向runtime/selector、停止本配置，或尚無足夠證據。不要用泛泛的「需要更多實驗」代替已知判斷；仍應說出唯一最具決定性的缺失證據。

如主模型受阻，報告仍必須生成，含已完成修正、實際CPU/微基準結果與精確blocking command/error；不得填造模型準度或speedup。

## Decision unit tests

至少包含：全tie小n不自動Q-pass、速度CI跨1不GO、僅bytes下降不GO、GPU缺失不NO_GO、correctness failure不方法NO_GO、static支配動態、history有效pseudo無增益、缺配對不強判、smoke兩題不推母體、設定hash改變不能resume舊rows。
