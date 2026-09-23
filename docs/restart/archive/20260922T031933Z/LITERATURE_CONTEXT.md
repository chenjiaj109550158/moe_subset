# 文獻機制對照（使用者於執行途中提問）

查閱日期：2026-09-22。此文件不改變凍結協議、候選、樣本或判定門檻，不作本機已測性能證據，也不宣稱完整文獻新穎性。

- [ProMoE v1](https://arxiv.org/html/2410.22134v1)：利用中間狀態預測多個後續層，重視 prediction accuracy 與 lead time，搭配 chunked prefetch、early preemption、reordered inference，把傳輸移出需要等待的執行路徑。真正 miss 仍 demand-load。這個跨層 prefetch window 不等於本輪跨 8 個 production forwards 的 hard routing subset。2024 v1 對手工調整 caching baseline 的平均 decode speedup 為 1.36x，對既有框架 offloading 為 2.84x；不同版本摘要數字有變化，因此這裡固定引用 v1。
- [MoE-Infinity v1](https://arxiv.org/html/2401.14361v1)：sequence/request activation tracing、activation-aware caching 與預取；缺失的真正 experts 仍補載。v1 的模型與部署設定並非本輪 Qwen3，也不能把 trace-level 或 bandwidth simulation 直接當本機自由生成證據。
- [Fiddler v1](https://arxiv.org/html/2402.07033v1)：小 batch 時，部分未常駐 expert 在 CPU 計算，比搬整個 expert 權重至 GPU 更划算，只傳小 activation。其 Mixtral-8x7B 評估使用 16-bit，並為公平比較擴充 Mixtral-Offloading 的原精度路徑；不能把所有文獻收益都歸因量化或不公平 baseline。本輪 CPU 僅存放 expert weights，expert 計算仍在 GPU，因此未實作或驗證此異質計算方案。
- [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/pdf/2312.17238)：LRU/pre-loading 搭配 HQQ 混合量化；系統評估使用 2/3-bit experts 與 4-bit attention。精度、cache 容量與搬運成本均不同於本輪 BF16。

本輪確定測得：H/E fixed-work paired speedup 0.899466，95% CI [0.884551, 0.914587]，decode-related expert bytes 減少 38.6044%，p95 token-ready latency 比 1.929214。S/E speedup 1.064781，bytes 減少 93.9249%。H/S/E 共用同一 optimized backend。

由文獻與本輪測量作出的研究推論：若另開下一輪，較有針對性的實驗是 cheap causal exact prefetch（含 miss fallback、額外流量與 eviction 計費），以及 host/selector/換窗同步成本的獨立 ablation。這些目前均不是已證實的新機器加速，也不在本輪確認途中追加，以保持原實驗完整性。

Profile 限制：現有 trace 包含 prefill；其 CUDA self-duration 比例不是 decode wall 分解，重疊事件不能相加當 critical path。H p95 增高與換窗成本相符，但個別 selector、CPU launch、copy wait 的 wall 比重仍需獨立 ablation，不能把所有差額指定給單一元件。
