# AI Agent Usage Guide

> ⚠️ **Deprecated in v1.2.0**: This document describes functionality that has been removed.
>
> The `mloop agent` and `mloop orchestrate` commands are no longer available.
> AI integration is now provided through [mloop-mcp](https://github.com/iyulab/mloop-mcp).

---

## Migration to mloop-mcp

### What Changed

| v1.1.x (Removed) | v1.2.0+ (Current) |
|------------------|-------------------|
| `mloop agent "query"` | Use Claude/Cursor with mloop-mcp |
| `mloop orchestrate` | Use AI client's native orchestration |
| Built-in agent definitions | Prompts in mloop-mcp |
| MLoop.AIAgent package | Deleted |

### New Workflow

```bash
# 1. Install mloop-mcp (MCP Server)
npm install -g @iyulab/mloop-mcp

# 2. Configure your AI client
#    - Claude Desktop: Add to claude_desktop_config.json
#    - Cursor: Add to MCP settings
#    - Other MCP-compatible clients

# 3. The AI will now have access to MLoop tools:
#    - train, predict, list, promote, etc.
```

---

## Why This Change?

**"AI가 MLoop을 사용한다" (AI uses MLoop)**

Instead of embedding AI inside MLoop, we expose MLoop to AI through MCP protocol:

- ✅ MLoop stays simple (zero AI dependencies)
- ✅ Any MCP-compatible AI can use MLoop
- ✅ AI providers are user's choice, not MLoop's
- ✅ Better separation of concerns

---

## See Also

- [AI-AGENTS.md](./AI-AGENTS.md) - Overview
- [mloop-mcp Repository](https://github.com/iyulab/mloop-mcp)
- [ECOSYSTEM.md](./ECOSYSTEM.md) - Full ecosystem architecture

---

**Last Updated**: January 2026
**Version**: v1.2.0
