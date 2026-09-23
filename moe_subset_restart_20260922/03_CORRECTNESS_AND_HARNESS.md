# 03 — 修正清單、回歸測試與數值 gate

本階段不預設所有聊天診斷都正確。每列要記錄 source path/symbol、最小重現、修補、test ID、結果和性能影響是否已實測。已有修正則保留證據，不重複改動。

## C01 停止條件生命週期

起點：`benchmark/subset_closed_loop.py::_finished`。將 `StopStringCriteria`、EOS 配置與 request token buffer 放進 request state，一個 request 建立一次，不每 token 建構。保持目前 prompt+continuation 的停止語意，不擅自改成只看 continuation 或固定文字長度；處理跨 token、重疊 stop strings、EOS/list、空 stop、batch=1 的布林結果。

測試 native 舊停止位置與新停止位置相同，constructor 次數是每 request 一次，非每 token 一次。允許品質生成必要的 host termination check，但不可宣稱它是完全 asynchronous。StopStringCriteria 的內部 cache 不代表 constructor 零成本；改善幅度以 profile 為準。

## C02 計時與 instrumentation

提供 `audit`、`profile`、`performance` 三模式。計算語意、線上必需的 stats/selector/scheduler 相同；performance 只移除不必要 dumps/hashes/diagnostic hooks 與 Python metrics aggregation，不移除 predictor 本身或必要等待。

修改 `qwen_real_offload_speed.py` 等 runner：先完成必要 device work，取得 end timestamp，才格式化 metrics/JSON/CPU 報告。Outstanding H2D、真正 decode 等待及 bootstrap 不能躲在計時外。不可把 `finish_metrics` 的 Python 遍歷算推論；device completion 則必須計入。事件計時和跨 stream completion 要有單測或小型 integration test。

Route hooks 在 model/request 初始化時裝一次，以 mode/policy 切換，避免每 token 安裝／拆除。History 所需的 natural top-k stats 保留 GPU FP32；可在 window boundary 有限 D2H 供 scheduler，須計費。Performance 模式不得每層把所有 logits/weights/IDs dump CPU。端到端包括 Python host overhead，不能只回報 CUDA events。

事件生命週期也要納入 profile：不要以移除逐 expert GEMM 為名，仍每個 expert/token 建立大量計時物件。必要 completion events 可在安全的生命週期內復用；不能覆寫仍被 waiter 或 elapsed-time 計算引用的 event。詳細逐 transfer timing 可在 profile 模式量，performance 模式保留必要依賴與真實 bytes counters；未測的 timing 填 null，不能填 0 或從別次 run 冒充。

稽核 audit/performance 在固定 inputs/routing/seed 下得到等價 state 與輸出；有差異先修，不拿 performance 跑品質作弊。

## C03 d、G 與內容跨度解耦

起點：`qwen_penultimate_joint.py::first_four_history_subset` 及 related residual/pseudo helpers。原選集讀前四個 anchors，G=8 後四個可能只增加計算；先寫後四個數值任意改變時選集不變的測試。

新增分開的 `residency_window_tokens=d`、`pseudo_compute_tokens=G`、`pseudo_content_horizon`、`selector_core_anchors`。Legacy 路徑仍 G=8、core=4；新 P4 只計算相同的前四個位置。

**不可直接把所有 horizon=8 改成 4。** 最近內容取樣、unigram retrieval、fallback 序列、位置編碼、RNG draw 次數或 history 更新可能依 horizon 改變。先生成／定義與 legacy 相同的內容序列，再截取前四個給 forward；保持 first-four token IDs、positions、prefix、權重、cache 同義。檢查 bootstrap 與後續 joint 兩條路徑，不只改後者。

標準 causal attention 下尾端 positions 不影響前端的數學依賴；batch shape/算術不同仍可能改變數值。比較 G8 與 G4 的 first-four logits、routes、subset、bridge 和閉迴路答案，不能保證 bitwise 相同或預設速度減半。

## C04 選集硬優先級

原 first-four core union 在 k=8,m=32 時上限就是32，因此 core 內的重要性排序通常無法淘汰其中任何 expert。新增「四組 top-8 完全不重疊，history 其他 expert 極高仍進不來」的 regression test，將它標示為 legacy policy 行為而非擅自修掉。

新 weighted/cost-aware selector 是獨立 policy，不可冒充只是 kernel/refactor。GPU FP32 history 取代原 CPU double 也可能改變接近 tie 的排名，記錄差異；以同政策同 backend 的 routing/state fixtures 驗證，不能偷偷改 tie rule。

## C05 routing 語意

保留完整 router；對 Qwen 核對 pinned implementation 的 FP32 softmax、top-k、`norm_topk_prob`、mask placement。top-k logits 不是最後 mixture weights。不可為 fusion 把 weight 移到 nonlinear MLP input 前。

Natural route 統計是本 policy 當前 hidden state 的未遮罩偏好，不是另一條 full-model trajectory 的真值。生產 hard subset 的 mask/rerank/normalization 與 pseudo `natural_topk_intersection_zero_missing` 是不同策略，必須各自命名與測試；之後可做單獨 ablation，不能靜默混用。

對每個 token：k 個唯一合法 logical IDs；非零 contribution 對應已 ready 的 slot。零 weight 的 invalid placeholder 必須在任何 pointer dereference 前 mask；不可讓 -1 索引最後一個 slot，也不可用 backend 的 missing-expert 零輸出功能掩蓋 production miss。

## C06 cache/RNG/position 與 bridge

固定 prefix 上先測 sequential 與 joint。每次 joint 提交 production KV 恰好一個 bridge position，pseudo KV 不提交；不修改舊 prefix tensors、不重複 forward 真 bridge當加速成果、不把 oracle reference cache 代入 candidate。對 seq_len、cache layer objects、position IDs、RoPE offsets、attention masks、cumulative length（適用時）、fork/commit/reuse 做測試。

Generation RNG 與 planning RNG 隔離，即使 greedy 主測也測 stochastic fixture。只允許當時已生成 token 作為 bridge/內容；準備中的下一個真實 token 不可被讀取來預測自己。

新增至少兩個窗口轉換、第一次窗口、最後不足 d tokens、prompt<d、EOS 在 boundary、stop 在 boundary、G=0/4/8 合法配置測試。某組不受支援就明確 reject，不半支持。

## C07 async slot 生命週期

先同步 exact path 再 async。Slot mapping 要有版本或 generation ID；舊 compute 所有讀取完成後才能覆寫，copy ready 後才能供新 compute 讀取。重用同 slot 的多次 H2D 也要排序。

GPU mapping 不能在舊 kernel 尚未讀完時被 next subset 覆寫。可用分版本 mapping 或嚴格 stream/event 順序。每層 grouped kernel 完成事件可覆蓋整組 slots，但不能在 kernel 之前 record。Producer/consumer 必要依賴保留；移除多餘 host sync 不等於移除 synchronization correctness。

以高 replacement/churn、m=k、重複與空 expert、交錯 streams、多輪 cache reset、joint layer-prefetch stress test 驗證。保存 slot→expert version trace（audit only）。GPU 可用時，available compute-sanitizer 可加做；工具不存在不阻止已充分測試的路徑，但要列限制。

## C08 三層數值驗證

A. **固定路由的單層 MLP**：同 inputs、weights、top-k IDs/weights，Python/reference vs GPU backend。FP32 參考關閉 TF32/不受控 reduced precision；BF16 weights 的高精度參考用相同已量化數值轉 FP32。測試真實及 variance-scaled synthetic tensors，保留 max abs、RMSE/NRMSE、cosine、worst token/layer，零 reference 單獨驗證。

B. **固定 prefix 的完整 forward**：同 route/policy 比較無 offload／同步 offload／async offload；full-resident 真模型不適合硬體時，以已驗證 CPU-first sync path 加單層 native checks 作 reference，標清覆蓋範圍。不能因無 full-resident VRAM 就退回全模型 GPU 初始化。

C. **自由生成**：允許 BF16 同義实現因數值差異產生 trajectory divergence，但這不是 correctness 的豁免。要求 A/B 通過、routing 語意精確、差異定位與新 exact baseline 品質對照；不能僅因 sampled token 偶爾相同就 gate pass。

數值 acceptance 在 `correctness_protocol.json` 先凍結：預設 FP32 `atol=1e-5, rtol=1e-4`；BF16 要求 finite、全局 NRMSE≤0.01、cosine≥0.999（非零 case），並完整回報 worst absolute error。不同 layout/累積方式若失敗，先找最初出錯的層並測高精度，而非事後擴 tolerance。若這些門檻對已知正確 vendor/native 對照都不適用，先保留失敗，建立有來源與理由的**新 correctness protocol**，在任何 held-out／性能 winner 確認前凍結並全重測；不得只為自己的 kernel 放寬。

對 router exact ties，驗證 native 合法 top-k 集合及權重；不以加 epsilon 改 logits 取得表面可重現。Legacy 近 tie 排名差異列報。

## P1 gate

核心 loader/routing/cache/slot/backend-reference 任一未解 correctness 失敗，就禁止其正式品質／性能結論。獨立不依賴失敗模組的便宜 history 路徑仍應繼續。輸出 `correctness_report.md`、JUnit/測試日志、`AUDIT_FINDINGS.md` 與每項修正 diff。
