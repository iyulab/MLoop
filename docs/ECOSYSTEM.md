# The MLoop ecosystem

MLoop trains, evaluates and serves ML.NET models from the command line, with every piece of state in
files. It is one layer of a small set of open-source packages, each with one job.

```
UInsight (Rust statistics engine, .NET bindings)
    └─→ DataLens ─────────────┐   exploratory analysis: profiles, correlations, outliers, time-series primitives
FilePrepper ──┬─→ DataLens    │   reading CSV / Excel / Parquet / JSON, preprocessing steps
              └─→ MLoop.Core ←┘
                    ├─ MLoop.Search        trainer search: candidates, tuner, folds
                    └─ ML.NET              trainers, model.zip, ONNX export
MLoop CLI (`mloop`) and API (`mloop serve`) over MLoop.Core
    ↑
mloop-mcp — MCP server that runs the `mloop` CLI for AI agents
HoneAI — provenance-bearing predictions; calls MLoop over HTTP for its ML tier
```

## The packages

| Package | Job | Reaches MLoop through |
|---------|-----|-----------------------|
| [FilePrepper](https://github.com/iyulab/FilePrepper) | Read tabular files of several formats; preprocessing (`mloop prep`) | NuGet, inside MLoop.Core |
| [DataLens](https://github.com/iyulab/DataLens) | Exploratory analysis behind `mloop info --analyze`, `mloop analyze`, `mloop detect` | NuGet, inside the CLI |
| [UInsight](https://github.com/iyulab/u-insight) | The statistics engine DataLens wraps | DataLens only — MLoop never references it |
| **MLoop** | Training, trainer search, evaluation, promotion, prediction, serving | — |
| [mloop-mcp](https://github.com/iyulab/mloop-mcp) | One MCP tool per `mloop` command, for AI agents | Runs the CLI as a subprocess |
| [HoneAI](https://github.com/iyulab/HoneAI) | Tiers of judges (ML, rules, LLMs, people) whose answers carry their source | `mloop serve` over HTTP |

## Which surface to use

| You are | Use |
|---------|-----|
| A person at a terminal, a script, a CI job | The `mloop` CLI (`--json` for machine-readable output) |
| A long-running service | `mloop serve` — REST endpoints, JWT auth ([GUIDE.md](GUIDE.md)) |
| An AI agent | mloop-mcp, or the CLI ([AI-AGENTS.md](AI-AGENTS.md)) |
| A .NET application that must train or predict in process | The `MLoop.Core` package ([EMBEDDING.md](EMBEDDING.md)) |

The process boundaries (CLI, MCP, HTTP) are the recommended ones: they keep an application free of ML.NET's
dependencies and let MLoop be upgraded on its own. `MLoop.Core` is published, with `MLoop.Extensibility` and
`MLoop.Search`, at the CLI's version for applications that need it in process.

## Dependency direction

Dependencies point one way, toward the data libraries:

- MLoop depends on FilePrepper and DataLens as packages; they know nothing of MLoop.
- `MLoop.Search` depends on ML.NET only; `MLoop.Core` depends on it, never the reverse.
- mloop-mcp and HoneAI depend on MLoop's process surfaces, not on its assemblies.

A concept that belongs to one consuming application does not move down into a library it uses.
