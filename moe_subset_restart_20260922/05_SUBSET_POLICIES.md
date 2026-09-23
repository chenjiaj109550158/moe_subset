# 05 — 策略、窗口與有界方法對照

## 必須先畫出時間軸

對每層定義 `S_j`，連續 d 個 production forwards 中 routing 只能在它裡面 top-k。完成第 d 個 forward 後更新下一窗口；最後一個 production forward 可以與 pseudo 合併，但實際 bridge 仍用 S_j，新 S_(j+1) 不能反向影響 bridge。

同 layer 的 next weights 只能在该層舊權重最後一次讀取完成後覆寫；可以與後面 layers 計算重疊，不能假設 m slots 同時容納兩組互斥 subsets。相鄰集合只載入差集。

不同 policies 的 first-window 初始化可以不同，但必須命名、記錄、計費。便宜方法從 natural prefill 統計初始化；legacy pseudo bootstrap 保留其真實成本。不能讓 pseudo 免費 full-model lookahead，再只計後續成本。

## 線上統計

保留 natural 全 router scoring，令 `r_t[e]` 是當前 policy hidden state 的未遮罩 top-k routing mass（其餘0）。Prompt 初始化用平均 natural mass；GPU FP32 更新：

`h_t = (1-alpha)*h_(t-1) + alpha*r_t`, alpha=0.2（開發前凍結）。

只記已完成真實 tokens；pseudo stats 與 production history 分開。統計不能因 subset 外未被執行而永遠為0，也不能讀 reference 未來 routing。現成一維 E=128 的 GPU 累加可用 tensor ops 或小 kernel，不需一開始自寫所有 router kernels。

## 策略清單

| ID | 語意 | 此輪角色 |
|---|---|---|
| E | 全E natural top-k、same m-slot deterministic LRU、exact demand loads | 必要主 baseline |
| EP | 因果預取＋必要時 exact fallback，same slots/copy budget | 支援時的較強 exact baseline |
| S | Prefill mass 選一次 top-m，整段 decode 固定 | 必要靜態比較 |
| H | 每d步按 h 選 top-m，沒有 pseudo forward | 必要便宜動態方法 |
| HC | H 加明確換入成本 | 有界效用／搬運對照 |
| P8 | Legacy first-four-core/history，G=8 | 只供歷史機制與成本對照 |
| P4 | 同 legacy first-four 語意，G=4 | 冗餘修正對照 |
| PW4 | G=4，probability+history 整體排名，再加換入成本 | 有界新 selector |
| PM4 | 與P4同內容/selector，只把pseudo計算改用production hard reroute | 有預算時的語意 ablation |

EP 先查 repo 已有 lossless causal prefetch。可在相同 manager 上用上一 token 的下一層 natural top-k 作便宜提示，確保只有早已觀測的 IDs、額外載入與eviction計費；一層 lookahead 後仍缺就 exact demand load。此 baseline 可能變慢，可由 development timing 在 E/EP 中選較強者並先凍結。EP 無法可靠實作時不要偽裝為已比較強 offloading；可繼續對 E 給 `baseline_scope=exact_demand_LRU_only` 的有範圍結果。

不要把全新昂貴 predictor 架構、oracle-assisted prefetch、未來 token/route cache 加進這輪。

## HC/PW4 的明確目標

令 U 為正規化後每 token 的預期 routing 效用，b_e 為該 expert 實際 bytes，b_bar 為該層平均 expert bytes：

`score[e] = d*U[e] - beta*(b_e/b_bar)*1[e not in current_subset]`

選 score top-m；beta=0.05 是本輪預先指定的實驗常數，不是理論最佳值。HC 用 U=h/sum(h)。PW4 用 U=0.5*h_normalized + 0.5*mean(first_four_pseudo_softmaxes)（兩項皆 sum=1）。數值零和要有決定性 fallback。

這個 cost 是換入代價 proxy，不是假裝準確模擬所有 overlap/交互影響。記錄效用與成本兩項，以及當窗口實際替換數和 bytes。不要只改參數名稱卻繼續硬保留 union core。

Subset tie-breaking 由 score、logical ID 決定；實現方式不可用添加會改變非 tie 排名的 epsilon。

## 有界實驗順序

先 main m/d 的 S/H/E runtime；接著 HC；再以1–2個 diagnostic prompts 跑 P8/P4 真實成本與 first-four equivalence。P4 有成本／品質理由才進 PW4/PM4，不能執著所有 pseudo 版本都完整跑 held-out。

Development 最多10個 candidate configurations，不含 E/EP baselines。已列方法與候選總數須在 `development_plan.json` 列清楚。可把純 history 的 d=4/16 作兩個附加設定；不能把所有 m×d×G×beta 無限制交叉。任何 m 改變都需相同 m 的 exact/static baseline。

P8 的用途是解釋先前失敗，不作新方法主力。H/HC 已有良好速度品質但 pseudo 沒增益時，應保留 window 方法、停止目前 pseudo 分支，不強迫結論一起好或一起壞。

## Future information audit

Inference/selector 只收到 request prompt、已生成 token、已完成 hidden/router summaries、resident state、hardware measurements。Answer labels、reference continuation、future oracle routes 都放 evaluator namespace，不傳入 selector。

Token-replay harness 在每 step 只交付「本 step 已知 token」，不能讓 selector拿整段 replay list。它僅是 fixed-input 診斷，不是 free generation evidence。Cached vanilla 可以供離線對照但不可供線上預測。

除非修 bug 需要1–2個 fixture，不重跑整套 oracle feasibility。舊八題 hard oracle 是非部署的小樣本 ceiling；不是新機器的 speed baseline，也不包含決定未來 subset 的成本。

## 需要輸出的選集資訊

每 policy：初始化、時點、d/G/內容跨度、routing weighting、history dtype/EMA、switch cost、已知資訊集合、是否 training-free、第一窗口成本、pseudo positions/真實 forwards、替換數與 stage bytes。Audit 可保存完整 routes；performance 至少保存必要 GPU counters 的批次摘要。

Routing coverage、mass、worst-window demand coverage 與換入 bytes 是診斷；只有最終自由生成 correctness 可回答品質問題。Restricted routing 的100%resident hit 是策略約束，不是研究成功證據。
