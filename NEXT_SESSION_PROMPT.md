# 下一個 Codex session 指令

你正在接續 `/home/justin/moe_subset` 的 PseudoRoute-MoE 專案。請把本文件視為本次 session 的明確人工授權與完整工作範圍。

我授權你：

- 修改 workspace 內的程式、測試、文件與設定，允許使用 patch；
- 執行本機 CPU/GPU 測試與 benchmark；
- 建立新的本地 artifacts 與邏輯 commits；
- 對本 session 啟動的長時間 worker 持續監看，完成後自動 aggregate、validate、report。

不要刪除既有 artifacts、model/dataset cache 或其他 session/user 的資料。任何下載、破壞性操作、換模型或擴大 dataset/samples scope 前先問我。不要中斷不是本 session 啟動的程序。

## 已完成且不可重跑的狀態

先檢查 Git HEAD、worktree、程序、PID/PPID 與 GPU。Git history 必須包含：

- `cbb0e60`：GPT-OSS/GSM8K hard-only execution scope；
- `94b8810`：先前六 task hard-only scope；
- `c0a96e5`：既有專案基線歷史。

目前 `benchmark_subset_oracle_v1` 已完成，不要重跑：

- artifact root：`artifacts/benchmark_subset_oracle_v1_r2_authoritative`
- pipeline：`complete/report_v1`
- scope：`benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2`
- scope fingerprint：`a5a908ad0124d2041b589a681e7102d3ea5d3f7df90752075799bb9875871d7c`
- base config fingerprint：`b68e18f45d373b31c45d99e8555e25fbf2816d7da8a68946536d6402668767a2`
- GPT-OSS/GSM8K actual hard closed-loop：1,319/1,319 rows
- point：`H=1, B=4`, `future_selected_routing_mass`
- vanilla：1,242/1,319，94.162244%
- hard：1,242/1,319，94.162244%
- paired gains/losses：0/0
- exact-token agreement：1.0
- decision：task-scoped `NARROW`

這個 GPT 結果是真正的 closed-loop hard masking，不是 identity-materialize；但因為 `H=1` 且 `B=4` 等於 GPT native top-k，它只證明 perfect one-token routing oracle 保持 natural route/output。不可把它稱為 multi-token constrained-subset 成功，也不可宣稱 runtime speedup。其 transfer reduction 是 trace simulation。

不要重跑 v17 vanilla、GPT hard suite、六個 datasets 或剩餘 previous-route rows。先前 HumanEval/MBPP+/previous-route artifacts 與 `.FAILED.json` markers 都要保留為 provenance。

## 必讀檔案

開始修改前先閱讀：

- `STATUS.md`
- `README.md`
- `docs/decisions.md`，特別是 `D-20260730-034` 與 `D-20260731-035`
- `docs/reproducibility.md`
- `docs/speculating_experts_accuracy_protocol.md`
- `docs/spec/04_ALGORITHMS.md` 的 pseudo-token、default-vector shadow rollout、probe economics 段落
- `artifacts/benchmark_subset_oracle_v1_r2_authoritative/report.md`
- `artifacts/benchmark_subset_oracle_v1_r2_authoritative/decision.json`
- `artifacts/benchmark_subset_oracle_v1_r2_authoritative/fallback_transfer_pareto.json`
- `configs/benchmark/benchmark_subset_oracle_v1.yaml`
- `configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml`
- `src/pseudoroute/benchmark/prefetch.py`
- `src/pseudoroute/benchmark/subset_closed_loop.py`
- `src/pseudoroute/benchmark/subset_grid.py`
- `src/pseudoroute/benchmark/subset_report.py`
- `src/pseudoroute/probes/pseudo.py`
- `src/pseudoroute/probes/default_vectors.py`
- `src/pseudoroute/probes/shadow_cache.py`
- `tests/unit/test_shadow_probes.py`
- `tests/unit/test_subset_oracle.py`

現有 `PseudoTokenProbe` 與 `DefaultVectorShadowRolloutProbe` 只支援 `TinyMoEAdapter`。它們可作為資訊邊界、RNG/cache invariants 與演算法介面的參考，但其結果不能冒充 Qwen evidence，也不能建立一套與現有 benchmark evaluator 不相容的 standalone path。

## 本 session 唯一研究目標

建立 versioned、resumable 的 `pseudo_embedding_qwen_gsm8k_v1` focused pilot：

- model：`Qwen/Qwen3-30B-A3B-Instruct-2507`
- revision：`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- precision：bfloat16
- dataset：GSM8K v17 frozen test rows
- horizon：`H=8`
- per-layer routed-expert budget：`B=32`
- routed experts：128/layer
- native top-k：8
- resident fraction：25%
- shared experts：0
- 不訓練 learned predictor

比較三個主要 policies：

1. `hard_oracle_commitment`
   - routing-information oracle ceiling；
   - 每個 boundary 從目前 policy context 做 future natural lookahead；
   - 不使用 benchmark label/ground-truth answer；
   - subset 外 router logits 明確 mask，subset 內依 native top-k/normalization 執行；
   - 真正 closed-loop generation，不能 identity-materialize。

2. `previous_route_commitment`
   - 使用 policy 自己上一個 realized window 的 pre-mask natural route records；
   - 第一個 window 使用 frozen static-frequency baseline；
   - 真正 closed-loop generation。

3. `pseudo_embedding_commitment`
   - training-free；
   - 使用 deployable boundary information，不看未來真 token、vanilla trajectory 或答案；
   - 主 regime 為 `online_post_sample`：下一個 sampled token 已知，可為接下來的 forward window 規劃 subset；
   - 使用 native Qwen attention/RoPE/router semantics 與 shadow states；
   - production KV cache、generation RNG 與 production hidden state不可被修改；
   - 依 pseudo route utility 為每層選 top-32 routed experts，再做真正 hard closed-loop generation。

## 為什麼選這個點

既有 Qwen/GSM8K `H=8,B=32` open-loop evidence：

- future selected-mass oracle：
  - mean route hit `0.9587141798`
  - P05 hit `0.828125`
  - worst hit `0.609375`
  - mean selected mass `0.9763954222`
  - fallback `0.0412858202`
  - estimated transfer reduction `0.7172883729`
- previous route：
  - mean route hit `0.6093852931`
  - mean selected mass `0.6294281254`
  - fallback `0.3906147069`
  - estimated transfer reduction `0.3613152586`
- static frequency：
  - mean route hit `0.3397047925`
  - mean selected mass `0.3464264343`

這是刻意選的困難 multi-token point。Perfect oracle 自己的 worst-hit 也低於原始 0.80 gate，因此本 pilot 不能用原始 all-task GO gate假裝一定可達；它的主要問題是 pseudo embedding 能否顯著縮小 previous-route 到 oracle 的差距。

## Qwen default-vector artifact

優先重用：

- `artifacts/speculating_experts_accuracy_v17/models/qwen3_30b_a3b/default_vectors/default_vectors.safetensors`
- SHA-256：`2a315f6f3dc65ff62b656d0eb92b6781267e5dd13cbcaf4c09a5b5806060fd30`
- manifest SHA-256：`3c5e645666e12eb96e6d693ca8251d81a8a04200cb5a2abab043226e32077f87`
- artifact fingerprint：`baae202908e3a2d3f11f980eb99c52e6815b51a7386f47654dd1e25fcbdab1fd`
- definition：`mean_unweighted_selected_expert_output`

限制必須顯式報告：共有 444 個 layer/expert pairs 在 calibration 中未觀測，現在以 zero vector 保存。不得把它描述成完整 expert prior，也不要未經確認重新下載資料或重新校準大型 artifact。

## 必須先固定 protocol/config

在任何新的 pseudo 結果產生前，先新增並 commit：

- `configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml`
- 對應 protocol 文件；
- deterministic config fingerprint test；
- resolved sample manifest，列出 exact row indices/sample IDs；
- development/held-out/closed-loop sample partitions；
- selection rule、variants、progress gate、accuracy gate、token caps；
- artifact root：`artifacts/pseudo_embedding_qwen_gsm8k_v1`。

Sample selection 不可查看 accuracy、correctness 或 ground-truth answer。使用 deterministic SHA-256 ranking 與必要的 routing-only context/router-margin strata。所有 IDs 必須在看 pseudo 結果前落盤與 commit。

為節省時間，採分階段固定 scope：

- initial development：先重用既有四個 frozen Qwen/GSM8K trace rows：`test-0`、`test-439`、`test-879`、`test-1318`；
- mechanism smoke：預先固定少量 rows，驗證 8-token/window 行為、cache/RNG、route change 與 evaluator；
- held-out route evaluation：預先固定一個小型 disjoint set；若需要新增 natural traces，只對保存的 v17 token trajectories 做 teacher-forced replay；
- actual closed-loop pilot：預先固定至多 16 個 held-out GSM8K rows，三個 policies 使用完全相同 IDs、prompt rendering、decoding、answer parser 與原始 v17 token caps；
- 不得在本 session 擴到完整 1,319 題。若 8-row smoke 外推顯示 16-row×3-policy 超過 24 小時，保存結果、提出 ETA 並先問我，不可偷偷改 IDs、token cap 或 sample count。

## Pseudo-embedding 最低實作/消融

第一版必須忠於現有 spec，而不是自行發明 learned embedding：

- correct future RoPE positions；
- `sampled_next_token` repeated-content primary；
- `current_token` content baseline；
- independent anchors；
- causal pseudo sequence；
- zero expert contribution ablation；
- default-vector selected/top-k mixture；
- 若成本合理，再加入 expected top-M next-token embedding；否則保存明確 blocker/ETA，不可用不相容 proxy 冒充。

每個 pseudo layer：

1. 從 read-only/copy-on-write production cache 跑 native attention；
2. 以正確 future position/RoPE 建立 pseudo post-attention state；
3. 執行 exact native Qwen router，保存 pre-top-k scores；
4. 用既有 default vectors 近似 MoE residual contribution；
5. 沿模型 residual path 傳到下一層；
6. 對 8 個 anchors 聚合 per-expert utility；
7. 每層 deterministic top-32，tie-break 使用 ascending layer-scoped expert ID。

必須驗證：

- production cache sequence length、layer identities、KV tensor identity/data pointers/version counters 未改變；
- generation RNG state 未改變；
- pseudo shadow cache 在 boundary 後丟棄；
- Qwen routing scaling、normalization、layer-scoped IDs、dense/shared paths 與 native semantics 相容；
- actual commitment 使用 policy 自己的 closed-loop context，而不是 vanilla trace；
- 沒有 label leakage、future-token leakage 或由 accuracy 選 variant。

## Predeclared progress gate

Variant ranking 只能使用 development route/成本指標，不可使用 task accuracy。選一個 pseudo variant進 held-out 前，至少同時滿足：

- 相對 previous route，mean route hit 絕對提升至少 `0.05`；
- 相對 previous route，mean selected routing mass 絕對提升至少 `0.05`；
- 不劣於 static-frequency baseline；
- resident fraction 保持 25%；
- estimated transfer reduction至少 30%；
- cache/RNG/information-boundary tests 全通過；
- probe latency、temporary memory、router calls、attention queries 已實測並保存。

強 candidate 另外要求在 held-out route set 同時回收至少 25% 的 oracle-minus-previous gap。依目前 aggregate，參考門檻約為：

- mean route hit至少 `0.6967`；
- mean selected mass至少 `0.7162`。

若 progress gate 未通過，結論為 STOP/PIVOT，不要浪費時間跑 held-out actual accuracy。若通過，才跑預先固定的最多 16-row actual closed-loop pilot。

Accuracy gate 必須在 generation 前，依 frozen vanilla correctness 與 sample count 預先計算 paired allowed-drop/CI；不可看 smoke accuracy 選 variant。小樣本只能下 pilot/NARROW 結論，不能宣稱 full-dataset GO。

## 分階段工作

### A. Preflight 與 protocol

- Git/process/GPU/artifact audit；
- 確認沒有其他 session GPU worker；
- 重跑現有快速測試、artifact validation；
- 凍結 config/protocol/sample IDs/gates；
- commit，確認 worktree clean。

### B. Native Qwen pseudo mechanism

- 在現有 benchmark/prefetch/closed-loop architecture 內實作；
- 不直接把 tiny adapter probe接到 Qwen後稱為完成；
- 加 proportional unit tests 與小型 native integration smoke；
- 驗證 cache/RNG/read-only semantics。

### C. Route-level development/held-out

保存每個 boundary/layer/anchor 的：

- predicted router scores；
- selected subset；
- route hit、selected/full mass；
- fallback、churn、estimated transfer；
- context/router-margin strata；
- probe latency/memory；
- concrete worst cases；
- oracle-gap recovery與 paired bootstrap/CI（若適用）。

### D. Actual closed-loop pilot

只在 progress gate通過後執行：

- hard oracle、previous route、最佳 pseudo variant；
- 相同的最多 16 個 held-out rows；
- 真正 generation；
- 記錄 first token divergence、first route divergence、token agreement、NLL/perplexity、GSM8K accuracy、measured runtime；
- 每 GPU 最多一個 worker；
- sample/policy 原子落盤、checksum resume；
- 完成後自動 aggregate/validate/report。

## 交付物

至少產生：

- versioned config/protocol與 sample manifest；
- resumable runner/orchestrator；
- resolved config/environment/execution revision；
- pseudo route raw rows/aggregates/strata/worst cases；
- cache/RNG parity audit；
- probe cost report；
- actual closed-loop per-sample rows（僅 gate通過時）；
- paired vanilla/oracle/previous/pseudo summary與CI；
- measured-vs-simulated partition；
- artifact manifest/checksum/resume audit；
- `GO`/`NARROW`/`STOP/PIVOT` focused decision；
- README、STATUS、reproducibility、decisions 更新；
- proportional unit/integration tests；
- clean logical commits。

報告必須清楚區分：

- measured task accuracy；
- exact-token identity；
- route replay/open-loop simulation；
- simulated transfer/stall；
- measured probe/runtime；
- identity-materialized rows；
- 真正 closed-loop generation。

## 完成條件

不要只寫計畫，也不要直接跑完整 GSM8K。直到以下完成才結束本 focused session：

1. protocol/config/sample IDs/gates 已在任何 pseudo 結果前固定並 commit；
2. native Qwen pseudo mechanism、cache/RNG invariants與測試完成；
3. initial route development完成；
4. held-out route gate完成；
5. gate通過則完成預先固定的小型 actual closed-loop pilot；未通過則保存可稽核 STOP/PIVOT 證據；
6. artifacts 通過 row count、checksum、resume與 provenance audit；
7. `python -m pytest -ra`、`ruff check .`、`ruff format --check .`、`mypy src/pseudoroute` 通過；
8. 工作樹乾淨且有邏輯 commits；
9. 給出清楚、限定 Qwen/GSM8K `H=8,B=32` pilot scope 的結論。

若遇到長 worker，使用 PID/PPID、GPU utilization、artifact row count、latest mtime 與單題 token cap 判斷健康；只要健康就不要因靜默重啟或建立重複 worker。
