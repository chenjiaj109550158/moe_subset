# Dependency Plan

## M0 runtime

- **PyTorch:** tensor operations and the deterministic tiny causal MoE.
- **Pydantic:** strict typed configuration with unknown-key rejection.
- **PyYAML:** human-readable configuration loading.
- **NumPy:** PyTorch interoperability; constrained below 2.4 for the Python 3.11 mypy target.

## M0 development

- **pytest:** device-agnostic tests, including accelerator execution when available
  and CPU compatibility in CI.
- **Ruff:** one formatter/linter instead of overlapping tools.
- **mypy** and **types-PyYAML:** strict package type checking.
- **Hatchling:** build backend only, not a runtime dependency.

## Download and cache policy

Codex agents may download models and datasets required by the active milestone when disk space permits. Estimate download, expanded cache, and artifact sizes first; preserve headroom for traces; reuse a configurable cache; pin revisions; and record identifiers, revisions, cache paths, and fingerprints. Downloads must not be committed to Git, and large external tests remain optional in CI.

## Deferred

Transformers, Accelerate, Datasets, Safetensors, Zarr/PyArrow, SciPy, scikit-learn, plotting packages, and experiment tracking are intentionally absent. Add each when its milestone uses it; installing these dependencies and downloading required research assets are both permitted under the policy above. M0 itself does not require an external model or dataset and uses standard PyTorch device dispatch without CUDA-specific kernels.
