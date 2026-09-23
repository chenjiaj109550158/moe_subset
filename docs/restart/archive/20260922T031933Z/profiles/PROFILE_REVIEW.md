# 速度歸因與適用範圍

本節由已保存的正式性能 rows 與獨立 profile 產生，不新增 GPU 執行、不改預先註冊 gates。

## 正式性能的延遲位置

H 在 production forward 8、16、… 輸出後更新 subset，因此更新等待出現在 9、17、… 的 token-ready latency。原 row 的 `boundary=true` 標在前一個 window 的末 token；主 pooled p95 已包含全部延遲，沒有漏計。下表 E/S 使用相同位置作對照，它們沒有動態 window 更新。

| 方法 | 首個 forward，含 bootstrap (ms) | 更新後首個 forward／相同位置 (ms) | 其他一般 forward (ms) |
|---|---:|---:|---:|
| E | 103.151 | 87.757 | 86.567 |
| H | 395.577 | 182.611 | 82.412 |
| S | 393.076 | 79.467 | 79.121 |

每個固定 128-forward request，H 相對 E：起始位置多 0.2924 秒，15 個更新後首位置共多 1.4228 秒，其餘位置共省 0.4745 秒；實際 decode wall 平均多 1.2407 秒。這能定位時間差出現在哪裡；不同 policy 有自己的 routing trajectory，所以不是把每段差值當作單一元件的因果消融。選擇、host scheduling、H2D 與同步各自的份額仍未被完整拆開。

## 獨立 profiler 的範圍

獨立 profiler 包含 prefill 加 16 個 production forwards，會引入額外開銷；不可把這裡的 wall 當作正式 128-forward throughput，也不可把含 prefill 的 CUDA self-time 比例當成 decode 關鍵路徑比例。跨 stream 的 duration 可能重疊。

| 方法 | prefill wall (s) | decode wall (s) | production 前自然 top-k 駐留比例 | 自然 top-k mass 駐留比例 |
|---|---:|---:|---:|---:|
| E | 1.5340 | 2.4646 | 0.6828 | 0.7056 |
| S | 1.5098 | 2.1985 | 0.6393 | 0.6750 |
| H | 1.5180 | 2.3032 | 0.6183 | 0.6542 |

自然 routing 統計是各 policy 自己當下 hidden state 的需求，不能當作 exact trajectory 的 counterfactual coverage。Raw traces 為 `runtime_E.json`、`runtime_H.json`、`runtime_S.json`；算子摘要為同名 `.txt`。

## 能支持與不能支持的解釋

Exact 與 subset 都只計算 top-8 expert；m=32 subset 不等於把原先每層 128 個 expert 的計算量砍成 32 個。Static 大幅減少 decode 搬運，仍只小幅加速，表示此實作的剩餘計算、host dispatch、同步等成本不可忽略；這項對照不能單獨量出各成本份額。History 額外維護 routing 統計並定期更新 resident set，更新位置的實測延遲足以抵銷一般位置的時間節省。

過去作品的預取重疊、CPU expert 計算與量化有不同收益來源；本輪沒有把那些能力都實作進來。主要來源與可比性見 `../LITERATURE_CONTEXT.md`。

## 本版 history 更新時點

`restart/inference.py` 在完整 boundary forward 取出真實 token 後才呼叫 `PolicyRuntime.after_boundary()`；`restart/model.py` 的 H/HC 路徑接著遍歷所有層，選擇並排入下一個 subset 的搬運。因此本版 history 缺少在該 forward 尚未結束時就逐層提前更新的 lead time。Pseudo 的 joint 路徑另有逐層 prefetch，但它還要支付 pseudo 計算成本，且本輪小樣本沒有通過晉級 gate。這是來源碼與延遲位置相符的排程解釋，不代表已測過另一種 history 排程或證明那種排程一定會更快。
