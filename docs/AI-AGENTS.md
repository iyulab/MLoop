# Using MLoop from AI agents

MLoop has no AI dependency of its own. An agent drives it from outside, through one of three
process boundaries — each runs the same commands a person runs.

| Surface | Use it when | What the agent sees |
|---------|-------------|---------------------|
| **MCP** — [mloop-mcp](https://github.com/iyulab/mloop-mcp) | The agent runs in an MCP client (Claude Desktop, Cursor, an agent framework with MCP support) | One tool per command: `mloop_train`, `mloop_predict`, `mloop_evaluate`, `mloop_promote`, `mloop_drift`, … |
| **CLI** | The agent can run shell commands | The `mloop` commands, with `--json` for a machine-readable answer |
| **HTTP** — `mloop serve` | The agent or application is a long-running service | The REST endpoints listed in [GUIDE.md](GUIDE.md) |

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
mloop train datasets/train.csv --label Churn --time 300 --json
mloop predict datasets/predict.csv --json
mloop evaluate --json
```

## See also

- [ECOSYSTEM.md](ECOSYSTEM.md) — how MLoop, mloop-mcp and the data libraries fit together
- [ARCHITECTURE.md](ARCHITECTURE.md) — technical architecture
