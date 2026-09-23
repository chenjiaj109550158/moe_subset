# 06 — 公平性能、品質與統計

本文件的數字是本次**研究決策門檻與資源選擇**，不是既有論文或硬體保證。先凍結 protocol+hash，之後才解盲確認資料。

## 資料隔離與兩次 freeze

**第一次 freeze**：P0/P1後，在看新的 candidate outputs 前鎖定硬體解析、主 checkpoint、精度、m/d、候選清單、資料抽樣規則、quality/parser/stop/token cap、統計與門檻。以舊 manifests 確認過去用於 selector development 的題目，排除出新 confirmation。不能只挑 baseline 正確的題目，也不能按新 policy 表現挑樣本。

Development 優先用 GSM8K train 中排除固定 few-shot demonstration IDs 後 seeded 抽樣的32題（其中前8題可作 cheap screening）；舊兩題／八題只作 debug，不與新樣本混為獨立觀測。主 confirm 從 GSM8K test 固定 seeded 排序抽樣，排除已知 policy-development IDs。Baseline-only 歷史評估不等於同樣的 predictor tuning，但需標明歷史使用情況；無法證明未被開發使用時稱「confirmation split」，不要稱全新未見資料。

Prompt rendering、few-shot examples、tokenizer/chat template、parser、stop strings以 pinned v17來源為起點。Re-render時驗 saved prompts fixtures，gold答案只給離線 evaluator。

**第二次 freeze**：development 結束，最多選一個主 window candidate、可選一個額外 pseudo comparator、exact baseline E/EP，以及static S；鎖定 final code hash、環境、configs、樣本IDs、N與performance順序。不得讀 held-out correctness 後換方法、beta、d/G或backend。

使用目前 run 的 measured dev throughput/平均及尾端 token 長度估算資源。目標 N=256；在解盲前按剩餘預算選 256、128、64 中最大的可完成 N，保留報告/重測預算。這是 sample-size 決定，不可依 held-out 結果擴增到「剛好通過」。N<256 可能無法支持2pp non-inferiority，屆時明確給pilot/inconclusive。

所有方法 max_new_tokens 初始512；若 development 顯示 exact truncation率>5%，可在第二次freeze前統一改1024並重估預算。Held-out後不能改cap補救個別錯題。若仍大量截斷，結論限定cap，不宣稱完整任務能力。

## 三種性能層次，不混用

### L0 單層 resident compute

依04，所有必要weights已resident。用來決定backend，不當端到端offloading speedup。

### L1 controlled fixed-work decode

主要性能 workload：每 request 固定128個 post-prefill production forwards，greedy自身閉迴路；忽略EOS僅為固定工作量，此模式須標成 `controlled_fixed_length`，不是正式task答案或一般服務時間。Prefill產生的第一個sample與後續F個production forwards分別計數；pseudo positions不加入production throughput分母。

Development performance使用8題×3repetitions；confirmation性能至少8個預先固定且未用於timing調參的prompts×5repetitions。先warmup每種shape/backend，randomized/block-balanced方法順序，一張GPU一次一個worker。同prompt/repetition是配對單位；不能把每token當獨立request。

另可保存fixed-token replay用來分離route/trajectory差異，但把它列diagnostic；selector不能看到future replay content。

### L2 true request free generation

正式品質生成使用原EOS/stop/cap，沒有forced future tokens。可以同時量request latency、TTFT、TPOT、生成長度與bytes/token。回答長度不同時，報每token與整request兩種結果，不能用總wall/總bytes的下降直接宣稱單token改善。

L1的performance reference與L2的quality reference須是同一個鎖定optimized exact實作。舊reference只供驗證／背景。

## 計時邊界

每run至少分：model load/setup、cold compile、prefill、first-subset/bootstrap、decode production、selector/probe、boundary H2D/wait、request drain、metrics/reporting。

主 controlled decode wall 包含從prefill完成後開始的一切必要planning/bootstrap、真實production、H2D、staging、必需同步、window-boundary spike與末尾已發出的工作。要記載first sampled token的處理位置。不能讓任何policy在計時前免費preload為decode選好的subset。

同一prefill執行規則與cache-initialization policy。Exact保留其由prefill自然留下的cache，subset需付把它改為指定S的成本；這是實際差別，必須透明，不能故意清空baseline而讓candidate暖啟動。

開始前清理上一request狀態與必要同步；結束前等待compute和其必要／已提交transfers完成，再取wall timestamp；CPU metrics formatting在外。若額外預取最後沒用到，仍計它的bytes與drain，不事後扣掉。編譯/warmup/setup另外報，不當steady-state；同時提供cold-start數字（可取得時）。

## 同資源比較

E/EP/S/H/P均使用相同 m slots/layer、precision、checkpoint、KV上限、GPU總envelope、dense placement、host模式與compute backend。記錄每policy實際峰值allocated/reserved與RSS/pinned，不只logical expert數。

不能預取再暗藏另一組m weights；若使用staging buffer，必須納入共同cap。同硬體但workspace不同可接受，前提是都在同一上限、沒有額外expert capacity；報所有差異。

動態exact可能必須CPU參與miss scheduling；這是真實成本，但不能保留可消除的逐expertPython GEMM作弱baseline。比較同樣優化的runtime；E/EP的選擇在development凍結。沒有EP時結論寫清baseline範圍。

## 指標定義

- `decode_speedup = median_or_aggregate_TPS(candidate) / same_stat_TPS(exact)`；主統計採每request配對log(TPS_candidate/TPS_exact)的平均再exp，另報aggregate forwards/s。
- `expert_bytes_per_production_forward` 分stage；decode相關的bootstrap/probe/prefetch/staging/production bytes全部計入，prefill另外列。GPU內部weight packing bytes與CPU staging bytes另列。
- `Q_delta = accuracy(candidate)-accuracy(exact)`；同樣本配對，單位percentage points需清楚換算。
- `quality_vs_static`、gains/losses/ties、truncation、invalid/empty answer、重複／崩潰、輸出長度。
- `p50/p95 token-ready latency` 與 `boundary vs nonboundary`；測定義必須是token可用的時間，不能將host enqueue時間叫token latency。精細events或readiness probing在獨立profile run量其干擾，不為正式throughput新增每層sync。
- `cache misses/transfers/bytes`、natural demand coverage/mass、subset replacements、selector latency、pseudo input positions、backend launches、CUDA waits、CPU overhead、memory peaks。

Stage或event ranges可能重疊；不能將所有CUDA event durations相加成critical-path時間。No-speedup歸因至少需一項profile或ablation支持。

## 品質置信區間：避免8/8得到[0,0]

每個sample得到D∈{-1,0,1}。令g為candidate對/exact錯的數量，l反之，n為完整paired samples。delta=(g-l)/n。

可報paired bootstrap作輔助，但主non-inferiority採不退化的保守區間：對p_gain、p_loss各做97.5% two-sided Clopper–Pearson interval（SciPy `binomtest(...).proportion_ci(confidence_level=0.975, method='exact')`），再用

`delta_CI = [L_gain-U_loss, U_gain-L_loss] ∩ [-1,1]`。

由union bound，此區間至少有95% coverage；不需要gain/loss獨立。所有tie也有非零寬度，不準寫成證明母體無損。n=0回null/block。這個保守方法可能判不出小幅差異，正確反應是inconclusive，而不是換較寬鬆gate。

多candidate/多quality claims的正式GO要控制multiplicity：第二次freeze列primary claims數J，對每個claim的gain/loss interval confidence設為 `1 - 0.05/(2*J)`；或將正式判定只保留一個預先指定primary candidate-vs-exact claim，其餘exploratory。選定J後不能以結果更換。若GO額外主張改善static，該claim也需計入J。

## 性能置信區間

使用request-level paired cluster bootstrap，重抽prompts並在其內保留/重抽配對repetition；不把幾千個tokens當n幾千。報95% CI、各prompt speedup、全部repetitions、median/aggregate與環境noise。候選多重確認時使用預先指定的primary candidate或相應alpha調整。

確認資料只能使用freeze後同一code/environment的rows。修bug造成結果失效，要標記舊rows invalidated並以新revision重跑完整受影響比較；不能混不同code取最好值。不能用held-out結果繼續調方法再稱confirmation。

## 資源不足與缺列

Budget開始接近上限時先不啟動下一昂貴row，保留已完成配對資料與完整缺失原因。中断/timeout/錯誤rows不直接當回答錯，也不丟掉換新樣本；統計列出intended/completed/excluded counts與可能偏差。

未完成凍結N、缺配對、或只剩很少題時，結果降為pilot/inconclusive；不能因sample小而自行把2pp容忍改成一題、或把一個repetition當穩定速度。

## 最低對照集合

正式確認至少包含optimized exact、static、選定的便宜或pseudo window candidate。當pseudo值得保留時，最好同時有H/HC；預算不足就不能聲稱pseudo勝過便宜history，只能標not-established。

最後結論分開：runtime修正收益、subset相對exact收益、dynamic相對static的必要性、pseudo相對便宜history的額外價值。這四件事不能用一個speedup數字替代。
