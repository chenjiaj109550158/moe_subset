# Pseudo-executed previous-subset mechanism smoke v1

Decision: **MECHANISM_FEASIBLE**.

This is a two-row, 16-token teacher-forced mechanism smoke. It is not held-out evaluation, task accuracy, closed-loop free generation, or a runtime-speedup result.

## Mechanism

At boundary 0 the shadow rollout can execute its native top-8 experts because the prefill assumption exposes all experts. At later boundaries it recomputes each MoE residual on the pseudo hidden state while restricting execution to the policy's own previous realized B=32 subset. The next subset is selected only after the complete eight-anchor shadow rollout.

## Descriptive route results

| Policy | Route hit | Selected mass | Simulated transfer reduction | Mean probe s | Expert calls |
|---|---:|---:|---:|---:|---:|
| pseudo_executed_previous_subset | 0.737386 | 0.755643 | 0.400228 | 2.0376 | 1536 |
| provided_previous_residual_control | 0.697835 | 0.718989 | 0.346436 | 0.8149 | 0 |
| previous_route_commitment_reference | 0.680990 | 0.698562 | 0.332194 | 0.0000 | 0 |

These route metrics replay the saved v17 token trajectory on each policy's own hard-subset state. Transfer is simulated; probe/replay time and CUDA memory are measured.

## Feasibility audit

- test-786: pass=True; next-subset escape=582 expert slots across 48 layers; fresh/control mean absolute residual delta=0.116218; cosine=0.491286.
- test-394: pass=True; next-subset escape=489 expert slots across 48 layers; fresh/control mean absolute residual delta=0.119755; cosine=0.395733.

All production cache identity/data-pointer/version/length and RNG invariants are checked at each pseudo boundary. Full 128-expert pre-mask scores are retained; later expert IDs are checked against the preceding layer-local subset; shadow caches are discarded.

## Interpretation

The proposed residual path is mechanically viable and can now be compared in a larger predeclared route experiment.
This smoke does not establish that it improves route quality; the route numbers are descriptive and were not used as a selection gate.
