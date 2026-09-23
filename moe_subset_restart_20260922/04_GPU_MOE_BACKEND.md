# 04 — 共用 slot-aware GPU MoE backend

## 目標與非目標

移除逐 active expert 的 Python compute loop、CUDA `nonzero/where` 造成的 host synchronization、反覆 one-hot/gather/index_add 與不必要 CPU route dumps。不是要求把整層壓成單一 kernel，也不強制實體 permutation。

Prefill/較大 T 可使用分組 dispatch；T=1/5/9 的低延遲路徑可只建立 assignment 索引，讓 GEMM 直接 gather。先重用目前相容、可核對的 BF16 fused/grouped backend，再按 profiling 做小範圍 Triton。不可把本任務變成從零重寫 serving engine。

`T` 是此 forward 的 input token 數，`R=T*k` 是 assignments，`M_e` 才是 expert e 真正的 GEMM rows。Batch=1,k=8 是八個不同 weights 的 M_e=1，不是同 weights 的 M=8。d=8 不會讓普通 autoregressive decode 八步並行。

## 模組介面

可調整實際命名，但職責必須分離：

```python
class MoEComputeBackend:
    def forward(
        self,
        x,                    # CUDA [T,H]
        slot_ids,             # CUDA integer [T,k]
        routing_weights,      # CUDA [T,k], original weighting semantics
        gate_up_slots,        # CUDA [m,2*I,H]
        down_slots,           # CUDA [m,H,I]
        workspace,
    ): ...                    # CUDA [T,H]
```

另有 `ResidencyManager` 處理 CPU store、logical IDs→slot IDs、cache replacement、H2D、events 與 phase accounting。Compute backend 不自行讀 CPU checkpoint 或改 routing。

`python_reference` 保留作 correctness/舊執行結構診斷。新的 GPU backend 由 exact/static/history/pseudo 全共用，不能在方法名上特化數值或偷選更快 precision。

## ID 與 weights layout

保留 GPU `expert_to_slot[E]`，以 logical top-k IDs 查表成 slot IDs。Subset 視窗內 mapping 不變；selector/window update 才更新。Exact 路徑仍需取得缺失資訊以供 CPU loader，但應傳 compact IDs/metadata，避免每層複製 full logits。

不要 `all_slot_weights[slot_ids]` materialize 選中 expert matrices；kernel 直接以 slot 索引原 buffers。可在初始化／slot replacement 做必要 layout packing，但記錄其時間、bytes 與額外 memory。

prefill active union>m 時分批載入與累加完整 natural contributions；每批只執行自己 assignments，不能 normalize 殘存 weights。Group accumulation 的 dtype/order 是 correctness 協定的一部分。

## Backend 整合順序

1. 查已安裝與官方可用 vLLM Triton fused-experts API，核對 callable signature、BF16、activation、unquantized config、strides、expert_map、compiled custom-op dependencies。固定可用 release/commit。保留 licenses/NOTICE；不能複製一個 Python 檔就假設依賴消失。
2. 先用 explicit pre-mapped valid slot IDs，使本地 expert 範圍為 m。只有驗證 API 語意後才用 expert_map=None；不可謊報 expert 數以強制觸發某個 fast path。
3. 小張量與真實 H/I 測試通過後，接 resident-only backend，再接 sync exact、async exact、subset。Import 或 kernel failure 不許吞掉後回退 Python 卻標示 optimized。
4. 最多兩個官方 backend/env 候選。若不相容，可做小型自有 Triton expert kernels，仍需相同介面與 tests；或記 `BLOCKED_BACKEND`，繼續 reference diagnostics。不是任何機器都必須安裝某個特定版本。
5. 自有 Triton 調整最多 YAML 規定的配置／profile cycles。優先 gate-up、SiLU×up、down、weighted combine 整體；不對 router GEMM 先做沒有 profiling 的微調。此輪不引入量化。

Grouped GEMM 的 padding、permutation、workspace、launches、combine 全算成本。可為 T buckets 選不同 configurations，但 dispatcher 只能根據公開形狀／硬體，不能根據方法標籤、答案或 held-out timing 特化。

## correctness gate

讀 03 的數值規範，固定 routes 後比較 output。測試 slot permutation、m=k、m=16/32（合法時）、inactive experts、零 routing weight、全部 token 偏向同一組 experts、T=0（支援或明確 reject）、非 contiguous x（拒絕或轉換後計費）、尾端非整 tile、out-of-range nonzero ID 拒絕／device error flag。

嚴格檢查 zero-missing pseudo 的 valid mask：零值不等於可安全讀錯 pointer。GPU device-side error flags 可在 audit/批次結束檢查；performance 不必每層 `.item()` 做斷言，但也不能省去資料正確性設計。

同步／非同步 offload 固定 routes 應給數值等價 output。GPU 被覆寫 race、NaN/Inf、未初始化 padding、重複 contribution 是 hard failures。

## P2 resident 微基準

使用 checkpoint 真實 H/I/k/dtype，T∈{1,5,9,16,32,128}，m 為本輪合法預算。Resident 測試的 assignments 必須在已載入 slots 內；不要用128個 logical IDs配只有32個weights。Synthetic 未代表模型品質；可加幾組真實 layer weights/input fixtures。

每 shape 測：較均勻、hot-set/skew、實際 trace（可取得時）。編譯與 allocator warmup 分開保存，不計 steady-state；至少5個獨立 timing blocks，block 內重複到合理計時粒度，次數由先導計時決定。

量：完整 forward host wall、GPU elapsed、dispatch/MLP/combine profiling、kernel launches、host synchronizations、workspace 峰值。Micro resident weights 若被 L2 長期缓存會高估真實收益，因此同時報 hot-cache 與 rotating layer/working-set 診斷；不能只選最好 cache 情境。

採 PyTorch profiler；若已有 Nsight Systems 可額外使用，不要求 sudo 安裝或 privileged counters。Profile runs 與正式 timing 分開，必須標記 instrumentation overhead。

輸出 `backend_candidates.json`、`backend_correctness.json`、`moe_microbench.csv`、`profiles/`、`backend_selection.json`。基於完整 pipeline 與 T=1 的結果選 backend，不只選大 T GEMM 最快者。新 backend 未比 reference 快時也保留負結果；「使用 grouped GEMM」不等於完成加速。

## P3 runtime 診斷

先使用新 backend 在相同資源下跑 exact、prefill-static、純 history 三組小規模 fixed-work 測試。原 Python execution 只再跑少量相同 prompts 作四格對照，無需把整個 held-out 全部重跑慢版本。

確認 wall time 裡剩下的是 attention/dense、routing、expert compute、host dispatch、必要 offload 等待、selector 或 metrics。跨 streams 的累計 kernel/H2D 時間不可直接相加當 wall；報 overlap 與 unresolved time，而非把所有未解釋時間寫成 compute。

可作 transfer-disabled、preloaded/replay 的樂觀診斷，但超預算常駐、已知未來、不同 routing trajectory 都必須標示。它們不是已部署速度，也不是普遍可證的 upper bound。

若 static resident path 都沒有實際收益，先檢查 runtime 與此硬體的 transfer headroom，再決定是否值得花 pseudo 預算。不能僅憑單層 microbench 不快就判定 subset 普遍不可行。
