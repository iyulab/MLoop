# Using MLoop from AI agents

MLoop has no AI dependency of its own. An agent drives it from outside, through one of three
process boundaries — each runs the same commands a person runs.

| Surface | Use it when | What the agent sees |
|---------|-------------|---------------------|
| **MCP** — [mloop-mcp](https://github.com/iyulab/mloop-mcp) | The agent runs in an MCP client (Claude Desktop, Cursor, an agent framework with MCP support) | One tool per command: `mloop_train`, `mloop_predict`, `mloop_evaluate`, `mloop_promote`, `mloop_drift`, … |
| **CLI** | The agent can run shell commands | The `mloop` commands, with `--json` for a machine-readable answer |
| **HTTP** — `mloop serve` | The agent or application is a long-running service | The REST endpoints listed in [GUIDE.md](GUIDE.md) |

## Operated by an agent, not written by one

An agent can reach machine learning two ways: write a training script and run it, or operate a fixed set of
commands. MLoop is the second.

- **The agent decides what, the tool decides how.** Which data, label and task, when to evaluate, whether to promote — the
  agent's calls. How a model is searched, validated, calibrated and gated is MLoop's, and is the same for an agent and a person.
- **No generated code runs.** Nothing the agent writes is compiled or executed to train or serve a model. (Scripts under
  `.mloop/scripts/` are an optional extension you write and place yourself; an agent runs them only if you put them there.)
- **Every run leaves files.** `metadata.json`, `metrics.json`, `config.json`, `trials.ndjson` and the leaderboard sit beside the
  model, so a result is inspected and compared (`mloop compare`) rather than reconstructed from a transcript.
- **The safeguards stay in the tool.** The quality gate, promotion rules and refusals answer the same way whoever asks, and
  `--json` returns them as data.
- **No LLM key, and for tabular tasks no network or GPU, is needed to run it.** The search is time-budgeted, so two runs of one command can end on
  different trials; the record says which trials ran.

## MCP

mloop-mcp calls the `mloop` CLI, so install the CLI first, then add the server to the client:

```bash
dotnet tool install -g mloop
```

```json
{
  "mcpServers": {
    "mloop": {
      "command": "npx",
      "args": ["@iyulab/mloop-mcp"]
    }
  }
}
```

## CLI

Every command that reports something takes `--json`, writes one JSON document to stdout and keeps
human-readable progress on stderr. An agent reads stdout and the exit code (`0` done, `1` could not
do what was asked — the reason is in `{"error": "..."}`):

```bash
mloop predict datasets/predict.csv --json
mloop evaluate --json
```

`mloop train --json` is the one stream: a JSON object per line as the run goes — `phase`, `trial`, `warning`, then
`result` (the experiment and its metrics) and `promotion` (`promoted`, and `keptInStagingReason` when it stayed in
staging), or `error`. A run that only checks the data (`--analyze-data`) trains nothing and so has no `result`: it
answers with one `analysis` event (`issues`: type, severity, column, description, suggestedFix — empty when nothing
was found), and `--generate-script` adds a `script` event with the path written:

```bash
mloop train datasets/train.csv --label Churn --time 300 --json
```

Through mloop-mcp the training tools turn that stream into MCP progress notifications, and cancelling a tool call
stops the command it runs.

## See also

- [ECOSYSTEM.md](ECOSYSTEM.md) — how MLoop, mloop-mcp and the data libraries fit together
- [ARCHITECTURE.md](ARCHITECTURE.md) — technical architecture
