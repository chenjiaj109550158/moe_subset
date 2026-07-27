# PseudoRoute-MoE 專案規格說明

這套 Markdown 文件是把前面完整討論整理成一個可直接交給 Codex 的「從零建構專案」規格。

## 專案目標

在 HBM/VRAM 無法容納大型 MoE 全部 experts 的情況下，把 expert weights 放在 CPU DRAM，並在 decode 過程中以 window 為單位決定：

\[
i+1,\ldots,i+t
\]

這段期間，每一個 MoE layer 可以使用的 expert subset。Window 內盡量不重新從 DRAM 載入 expert，以減少每-token 不規則的 PCIe 傳輸與同步等待。

研究核心不是單純預測下一個 expert，而是建立一種 **future-position-conditioned pseudo routing state**，估計未來一段 window 中每個 expert 的累積 utility，接著在 HBM budget 與載入成本下做 subset selection。

## Codex 應從哪裡開始

先把整個資料夾交給 Codex，並要求它從 [`00_CODEX_START_HERE.md`](00_CODEX_START_HERE.md) 開始讀。若 Codex 一次只能吃一份文件，可使用 [`PROJECT_SPEC_ALL_IN_ONE.md`](PROJECT_SPEC_ALL_IN_ONE.md)。

## 這套規格已經替你固定的初始決策

- 先做 batch size 1。
- 先做普通 autoregressive decoding，不把 speculative decoding 當核心依賴。
- 每層有自己的 subset，但初版所有 layers 共用 window boundary。
- 先做 training-free 版本，再加 RF／linear／MLP predictor baseline。
- 先建立 route trace、oracle、closed-loop constrained generation，再做真正 offloading runtime。
- DapQ 式 position hypothesis 必須透過 content × position factorial experiment 驗證。
- 必須區分 offline oracle 與 online deployable predictor，禁止 future leakage。
- 必須比較：
  - lossless predictor + fallback；
  - one-step no-fallback；
  - multi-step persistent subset；
  - oracle subset。
- 最終目標是 quality、VRAM、H2D bytes、TPOT 的 Pareto frontier，而不是只報 prediction accuracy。

## 最重要的 go/no-go 判斷

1. **Oracle feasibility**：即使知道未來 routing，固定小 subset 是否仍能保持 closed-loop generation 品質？
2. **Position hypothesis**：正確 future position 是否真的比正確 semantic content 更能恢復未來 router output？
3. **System benefit**：pseudo routing probe 的計算成本是否小於它省下的 exposed expert-loading stall？

若任一項不成立，文件中已經定義對應 pivot：
- 從 position-centric 改成 context／routing-history-centric；
- 從 hard commitment 改成 adaptive termination 或 emergency fallback；
- 從真實 probe 改成直接 low-rank/linear predictor；
- 從 per-request subset 改成 static + dynamic residency。
