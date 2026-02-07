# AI Agents

> ⚠️ **Notice**: The `mloop agent` command was removed as part of the "Zero AI Dependency" refactoring.
>
> AI integration is now provided through **mloop-mcp**, a separate MCP (Model Context Protocol) server.

---

## Current Architecture

```
Before (removed):                 Current:
──────────────                    ──────────────
MLoop.AIAgent (embedded)    →     mloop-mcp (external)

"MLoop이 AI를 품는다"        →     "AI가 MLoop을 사용한다"
```

## Using AI with MLoop

### Option 1: mloop-mcp (Recommended)

MLoop provides an MCP server that exposes CLI functionality to AI clients:

**Repository**: https://github.com/iyulab/mloop-mcp

```bash
# Install mloop-mcp
npm install -g @iyulab/mloop-mcp

# Configure in Claude Desktop, Cursor, etc.
```

**Available Tools**:
- `train`: Train ML models using AutoML
- `predict`: Make predictions with trained models
- `list`: List experiments and models
- `promote`: Promote models to production

### Option 2: Direct CLI Integration

Any AI agent can use MLoop through standard CLI:

```bash
# AI agents can invoke MLoop commands directly
mloop train datasets/train.csv --label Churn --time 300
mloop predict --name churn --input new-data.csv
```

---

## Why This Change?

| Before (MLoop.AIAgent) | After (mloop-mcp) |
|------------------------|-------------------|
| AI embedded in CLI | AI accesses CLI externally |
| Tight coupling | Loose coupling |
| MLoop depends on AI providers | MLoop has zero AI dependencies |
| Complex CLI | Simple, focused CLI |

See [ECOSYSTEM.md](./ECOSYSTEM.md) for the full ecosystem architecture.

---

## Migration Guide

If you were using `mloop agent` commands:

```bash
# OLD (removed)
mloop agent "Analyze my dataset" --agent data-analyst

# CURRENT
# 1. Install mloop-mcp
# 2. Configure your AI client (Claude, Cursor, etc.)
# 3. The AI will use MLoop through MCP protocol
```

---

**See Also**:
- [mloop-mcp Repository](https://github.com/iyulab/mloop-mcp)
- [ECOSYSTEM.md](./ECOSYSTEM.md) - Full ecosystem overview
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical architecture

---

**Last Updated**: February 2026
**Version**: Deprecated (archived)
