# 02 — 新機器 bootstrap、來源鎖定與模型載入

## P0：先建立可信工作區

1. 解析 `SPEC_DIR` 和 `WORK_ROOT`；目標 repo 為 `WORK_ROOT/moe_subset`。不存在才 clone 指定 URL；已存在先驗 remote/HEAD/dirty state，不覆寫。不匹配的同名資料夾不可刪除，改用有 run ID 的自有目錄。
2. 建立 `restart/<run_id>` branch 或獨立 worktree。記錄 `git HEAD`、遠端 URL、UTC、dirty diff、規格 hashes。當前 main 可能已更新；鎖定 clone 得到的 SHA，之後不自動 pull。
3. 建立外部 run log，使 repo clone 失敗也有 `FINAL_REPORT.md`/`DECISION.json` 可交付。Clone 失敗時檢查 DNS/代理/存取一次，不無限重試，不捏造檔案內容。
4. 讀 README、STATUS、pyproject、現有 tests、指定源碼與 artifact。用 `rg`/AST/CLI help 確定入口，不能假設 `scripts/run_*.sh` 存在。既有 spec 是背景與 invariants，不授權舊實驗重跑。

## 硬體與環境

輸出 `hardware.json`：OS/arch、CPU、RAM available/total、disk free、可見 GPU UUID/name/compute capability/VRAM/free、driver、PyTorch runtime CUDA、NUMA/PCIe 拓樸（可讀時）、pinned allocation 支援、其他 GPU activity。區分 driver CUDA compatibility、實際 torch runtime 與 nvcc；不能用 `nvidia-smi` 的 CUDA 字樣代替 wheel 相容性測試。

使用 Python 3.11 或 3.12 的獨立 venv/conda，依目前可安裝的官方相容組合解依賴。Repo 宣告 Python>=3.11，但舊 artifact 的 Python/Torch/Transformers 版本不代表新 GPU 必須照抄。先做 CUDA tensor、BF16 matmul、stream/event、pinned H2D 與 tiny model smoke，再安裝其餘依賴。

現有 editable install 是 `python -m pip install -e '.[dev,hf]'`；必須先建立 constraints，不讓它暗中換掉已驗證 torch。最多兩組完整 dependency 候選，記錄 resolver output、`pip check`、`pip freeze --all`、wheel/source provenance。取得 backend 相容組合後鎖定整個 run，不能在 A/B 中途升級套件。

vLLM 可選，不把它當作任意版本皆可 import 的單一 Python 檔。候選 env 可分離，但最後兩個 policy 必須在同一個已鎖定 measurement env。

新增精確裝置解析：主實驗 requested CUDA 沒有成功就 fail/block；tiny CPU tests 必須標示為 CPU，不能 silent fallback。

## 指定模型與來源

主 checkpoint：

- ID：`Qwen/Qwen3-30B-A3B-Instruct-2507`
- Revision：`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- BF16；舊設定描述 48 routed layers、128 experts/layer、native top-k=8、hidden=2048、expert intermediate=768。

以 pinned config/checkpoint tensor metadata 再核對這些值；不符時停止此 adapter，不能套用硬編碼。

先查 HF cache/使用者已提供的 checkpoint。缺少時，在 YAML 授權內查檔案 metadata、估算 shards/tokenizer/config 總量，確認 disk 包含 checkpoint、暫存、env、artifacts 與 headroom，才下載指定 revision。使用 safetensors，不啟用不必要的 `trust_remote_code`。只有 shard hashes、來源與大小進 git，weights 留 cache。

模型 unavailable 時，不把少量 random weights 的數字包裝成 Qwen 實測。

## 必須新增／驗證 CPU-first expert loader

原 `QwenExpertOffloadEngine` 先要求模型完整在 CUDA，再抽 expert tensors 到 pinned CPU。這是 loader 的 staging 限制，不是 method 的 VRAM 必要條件；新 loader 不可沿用全模型 `.cuda()` 再拆的路徑。

可採 meta 初始化加 sharded safetensors streaming：dense/attention/router/norm/embedding/head 逐 tensor 放 CUDA；expert tensors 直接建立 CPU backing store，GPU 僅配置 m slots/layer。處理 Transformers packed/unpacked expert layout 差異，gate/up 拼接順序由 pinned model 實作驗證，不可猜。必要的轉置／打包在載入階段或每次 slot replacement 時執行並計費，不每 token 複製全部 weights。

加 manifest 比對：每層 expert ID、shape、dtype、來源 key、checksum／抽樣數值。偵測未載入的 meta tensors、遺漏的 tied weights、保留在 GPU 的 full expert tensors。Checkpoint 解讀錯誤不得用 `strict=False` 靜默略過。

Host mode：優先 full-pinned CPU store。若 pinned 配額不足但完整 weights 可留 RAM，可明確切到 `cpu_resident_with_bounded_pinned_staging`，兩個 policy 共用且包含 staging copy 成本。不能把實際磁碟 paging 稱為 DRAM offloading；RAM 無法容納完整主模型 backing store 時 block 主實驗，僅繼續較小的 diagnostics。不得自動提高 ulimit/鎖頁限額。

計算並驗證以下 envelope：

`GPU = dense + max_KV + m-slot experts + dispatch/MLP workspaces + pseudo/cache buffers + allocator headroom`

`Host = expert store + 最大同時存活的 shard/tensor 暫存 + pinned staging + tokenizer/runtime + OS headroom`

RAM 估算必須採可實作的 streaming loader 峰值，而不是假設完整第二份 checkpoint 常駐；可用逐 tensor/slice 讀取降低暫存。不能僅因舊全模型載入方式失敗，就判定新 loader 也無法在此 RAM 執行。

先以主 m=32、d=8 檢查。硬體不夠時，只能在看到方法結果之前，按預先次序改採 m=16 或 m=8，寫 `resource_resolution.json`，結論需標明偏離原設定。連 m=k 都放不下則 block 主模型；不把 layers offload/quantization 偷偷加進來。

## 不縮減 experts 的 prefill

Prefill 是 natural top-k over all E，不是一次把全部 E weights 放 GPU。其 union 可能超過 m：stream experts/assignments 分組，保持同一 token 的原始 weights、累積所有 contributions。可以 chunk prompt 但保持 causal attention/KV 與 routing 語意，所有方法用相同 chunking。

Exact baseline 的多 token prefill／微基準也可能有 active union>m；不能只載一次 m 後丟掉剩餘 experts，或錯把每組 contribution 重新 normalize。

## P0 交付與 gate

`hardware.json`、`environment.lock.txt`、`environment_manifest.json`、`resource_resolution.json`、`source_manifest.json`、`model_manifest.json`、新 protocol draft、base tests 原始結果與 `AUDIT_FINDINGS.md`。

先跑現有 CPU tests/ruff/mypy，將既有失敗與新增失敗分開。基線已有無關 lint failures 不需為此重構全 repo，但新增程式與受影響測試必須清楚通過。

能完成 tiny/CPU 不代表 P0 的 real-model readiness 通過；後續 runner 必須讀取 readiness flags。
