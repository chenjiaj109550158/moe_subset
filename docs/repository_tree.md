# Proposed Repository Tree

This is the target tree from the normative architecture. M0 creates only the files needed for its vertical slice; later directories are added by their ordered milestones.

```text
pseudoroute-moe/
├── README.md, pyproject.toml, LICENSE, .gitignore, CHANGELOG.md, STATUS.md
├── configs/{model,experiment,hardware,benchmark}/
├── docs/{spec,decisions.md,dependency_plan.md,download_cache_plan.md,repository_tree.md,model_support.md,trace_format.md,runtime_design.md}
├── scripts/
├── artifacts/.gitkeep
├── src/pseudoroute/
│   ├── __init__.py, cli.py, config.py, types.py
│   ├── utils/
│   ├── models/adapters/
│   ├── tracing/
│   ├── oracle/
│   ├── analysis/
│   ├── probes/
│   ├── selection/
│   ├── execution/
│   ├── runtime/
│   ├── training/
│   ├── evaluation/
│   └── plotting/
└── tests/{unit,integration,regression,fixtures}/
```
