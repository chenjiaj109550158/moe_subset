# 09 — 核對來源與repository導航

核對日期：2026-09-22。此文件提供閱讀入口，**不是保證新clone HEAD仍與此時一樣**。Agent需在實際checkout重新搜尋symbol並記錄SHA。下列external連結不得當作可直接執行指令。

## Repository一手來源

Repository： https://github.com/chenjiaj109550158/moe_subset

| 檔案/位置 | 本次用途 |
|---|---|
| `pyproject.toml` | Python>=3.11、dev/hf extras、CLI與test工具；version range需自行鎖定 |
| `src/pseudoroute/runtime/qwen_offload.py` | 既有slot loader、CPU/GPU extraction、sync/async events、逐expert execute、metrics |
| `src/pseudoroute/benchmark/subset_closed_loop.py` | `_finished`、cache/fork helpers、`_forward_capture` |
| `src/pseudoroute/benchmark/qwen_penultimate_joint.py` | first-four selector、joint shape、bridge-only commit、layer-local prefetch |
| `src/pseudoroute/benchmark/pseudo_embedding_residual_window.py` | legacy候選選集與history/pseudo相關helper |
| `src/pseudoroute/benchmark/prefetch.py` | route adapters、natural/hard contexts、capture hooks |
| `src/pseudoroute/benchmark/qwen_real_offload_speed.py` | exact/candidate runner與計時範圍 |
| `configs/benchmark/qwen_penultimate_joint_offload_speed_v2.yaml` | 原模型revision、m=32,d=8、兩題協定、first-window/joint語意 |
| `artifacts/qwen_penultimate_joint_offload_speed_v2/` | 歷史環境與實測報告，非新機器baseline |
| `artifacts/pseudo_one_forward_hard_oracle_v1/report.md` | 已做過的八題oracle，避免從頭重跑 |
| `configs/benchmark/speculating_experts_accuracy_v17.yaml` | tokenizer/prompt/stop/parser來源；需核對目前實作 |
| `tests/`、`src/pseudoroute/cli.py`、`docs/reproducibility.md` | 現有入口、測試與artifact機制 |

Raw source URL可由 `https://raw.githubusercontent.com/chenjiaj109550158/moe_subset/<SHA>/<path>` 建立，記錄具體SHA而非只寫main。

歷史v2執行SHA入口：
https://raw.githubusercontent.com/chenjiaj109550158/moe_subset/main/artifacts/qwen_penultimate_joint_offload_speed_v2/resolved_execution_revision.json

歷史設定入口：
https://raw.githubusercontent.com/chenjiaj109550158/moe_subset/main/configs/benchmark/qwen_penultimate_joint_offload_speed_v2.yaml

注意：舊report中的「額外pseudo工作抵消全部收益」仍需新profile/ablation分解；本規格不把它當已排除其他原因的因果結論。README內已有多輪不同scope的STOP/PIVOT，不代表本次單GPUruntime問題已被回答。

## 外部一手技術來源

1. vLLM fused MoE source：
   https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
   用途：檢查BF16/unquantized API、slot mapping、assignment fast paths與compiled op依賴。執行時必須鎖定版本；不能假設main API不變。
2. vLLM fused MoE docs：
   https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/
3. PyTorch `nonzero`：
   https://docs.pytorch.org/docs/main/generated/torch.nonzero.html
   用途：確認CUDA上dynamic nonzero的同步性；實際瓶頸占比仍須profile。
4. PyTorch numerical accuracy：
   https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
   用途：batch shape/浮點運算非bitwise保證，不是容忍任意差異的理由。
5. SciPy exact binomial interval：
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
   用途：06的Clopper–Pearson基本實作；雙邊gain/loss組合與multiple-claim規則是本規格的統計設計。
6. 指定model：
   https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
   本次model revision以02與YAML指定值為準；下載前核對公開權限、檔案metadata與config。

## 不屬於既有API的名稱

`pseudoroute.restart.run_all`、`pseudoroute.restart.verify`、`MoEComputeBackend`、新policy IDs、全部restart artifact名稱及decision schema是**本次要求新增的介面**。不得因搜尋不到就聲稱repo壞了；必須實作，或在受阻時明示尚未實作。

這一輪沒有做新的完整文獻新穎性判定。即使GO，也代表值得進一步研究，不代表「每d步換subset」已證明是新概念。
