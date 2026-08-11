# Matched-eight Qwen/GSM8K hard routing-oracle pilot

True hard closed-loop generation at H=8 and B=32 on the exact eight parent rows.

- Hard oracle accuracy (measured): 8/8 (100.00%)
- Frozen same-row vanilla accuracy (pre-existing measured rows): 8/8 (100.00%)
- Paired gains/losses/ties versus vanilla: 0/0/8
- Paired accuracy difference bootstrap 95% CI: [0.0000, 0.0000]
- Exact-token agreement (weighted): 0.524826
- Route hit (weighted): 0.946483
- Selected routing mass (weighted): 0.966458
- Total measured runtime: 7706.46 s
- Simulated transfer reduction: 0.766503

The oracle looks ahead from its own closed-loop context and truly masks outside each 
layer's top-32 subset. Accuracy/runtime are measured; transfer is simulated. 
N=8 only supports a PILOT_NARROW diagnostic and does not alter the parent decision.
