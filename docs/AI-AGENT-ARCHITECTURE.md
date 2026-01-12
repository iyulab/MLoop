# AI Agent Architecture

> ⚠️ **Deprecated in v1.2.0**: This document describes architecture that has been removed.
>
> The `MLoop.AIAgent` project and all related components were removed as part of the "Zero AI Dependency" refactoring.
> AI integration is now provided through [mloop-mcp](https://github.com/iyulab/mloop-mcp).

---

## Migration to mloop-mcp

### Architectural Change

```
Before v1.2.0:                    After v1.2.0:
──────────────                    ──────────────
MLoop.AIAgent (embedded)    →     mloop-mcp (external MCP server)
Ironbees.AgentMode          →     MCP Protocol
Microsoft.Extensions.AI     →     MCP-compatible AI clients

"MLoop이 AI를 품는다"        →     "AI가 MLoop을 사용한다"
(MLoop contains AI)               (AI uses MLoop)
```

### Why This Change?

| Old Architecture | New Architecture |
|------------------|------------------|
| AI dependencies in MLoop | Zero AI dependencies |
| Tight coupling | Loose coupling via MCP |
| Complex CLI with AI commands | Simple CLI, AI through MCP |
| Provider lock-in | Any MCP-compatible AI client |
| MLoop-specific agents | Generic CLI tools for any AI |

### New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Client Layer                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Claude       │  │ Cursor       │  │ Other MCP    │      │
│  │ Desktop      │  │ AI           │  │ Clients      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    mloop-mcp                                 │
│                                                              │
│  Tools: train, predict, list, promote, info, serve          │
│  Repository: https://github.com/iyulab/mloop-mcp           │
└────────────────────────┬────────────────────────────────────┘
                         │ CLI Invocation
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MLoop CLI                                 │
│                                                              │
│  Commands: train, predict, list, promote, info, serve       │
│  Pure ML functionality, zero AI dependencies                │
└─────────────────────────────────────────────────────────────┘
```

---

## See Also

- [AI-AGENTS.md](./AI-AGENTS.md) - Overview of new AI integration
- [mloop-mcp Repository](https://github.com/iyulab/mloop-mcp) - MCP server implementation
- [ECOSYSTEM.md](./ECOSYSTEM.md) - Full ecosystem architecture

---

## Historical Reference

The content below is preserved for historical reference and migration assistance.
This architecture was used in MLoop v1.0.0-v1.1.x before the "Zero AI Dependency" refactoring.

<details>
<summary>Click to expand legacy architecture documentation</summary>

## Legacy Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Agent Hierarchy](#3-agent-hierarchy)
4. [Core Components](#4-core-components)
5. [LLM Provider Integration](#5-llm-provider-integration)
6. [Agent Implementations](#6-agent-implementations)
7. [Data Flow](#7-data-flow)
8. [Extensibility](#8-extensibility)
9. [Testing Strategy](#9-testing-strategy)
10. [Advanced Features](#10-advanced-features)
11. [Performance Considerations](#11-performance-considerations)

---

## 1. Overview

### 1.1 Purpose

MLoop's AI Agent system provided intelligent ML workflow assistance through specialized conversational agents. Each agent was designed for specific tasks in the ML lifecycle, from data analysis to model deployment.

### 1.2 Design Goals

- **Specialized Expertise**: Each agent focuses on a specific domain
- **Provider Agnostic**: Support multiple LLM providers (GPUStack, Anthropic, OpenAI, Azure)
- **Stateless Operation**: Align with MLoop's multi-process casual design
- **Extensible Architecture**: Easy to add new agents and capabilities
- **Production Ready**: Robust error handling and graceful degradation

### 1.3 Key Technologies (Legacy)

| Technology | Version | Purpose |
|------------|---------|---------|
| Ironbees Agent Mode | 0.4.1 | Agent framework with YAML-based templates |
| Microsoft.Extensions.AI | 9.3.0+ | LLM provider abstraction |
| .NET 10.0 | 10.0 | Runtime platform |
| ML.NET | 5.0.0 | ML operations and model management |

---

## 2. System Architecture

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (Commands)                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AgentCommand │  │ ChatCommand  │  │ StreamCommand│      │
│  │   (Router)   │  │  (Single)    │  │  (Realtime)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Agent Selection & Initialization
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Agent Layer                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ DataAnalyst  │  │ Preprocessing│  │   Model      │      │
│  │    Agent     │  │   Expert     │  │  Architect   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐                                           │
│  │   MLOps      │  ← Orchestrates other agents              │
│  │   Manager    │                                           │
│  └──────────────┘                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Core Operations
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Core Layer                                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  DataLoader  │  │MLoopProject  │  │  FilePrepper │      │
│  │              │  │   Manager    │  │   Impl       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │              LLM Provider Factory                 │      │
│  │  GPUStack | Anthropic | Azure OpenAI | OpenAI    │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│  MLoop.CLI                                                   │
│  ├── Commands/AgentCommand.cs (Entry point)                 │
│  └── [User Interaction, Output Formatting]                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MLoop.AIAgent                                               │
│  ├── Agents/                                                │
│  │   ├── DataAnalystAgent.cs                                │
│  │   ├── PreprocessingExpertAgent.cs                        │
│  │   ├── ModelArchitectAgent.cs                             │
│  │   └── MLOpsManagerAgent.cs                               │
│  │                                                          │
│  ├── Core/                                                  │
│  │   ├── LlmProviderFactory.cs                              │
│  │   ├── MLoopProjectManager.cs                             │
│  │   └── Models/                                            │
│  │       └── MLoopProjectInfo.cs                            │
│  │                                                          │
│  └── [Ironbees.AgentMode Reference]                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MLoop.Core                                                  │
│  ├── Preprocessing/FilePrepperImpl.cs                       │
│  ├── Data/DataLoaderFactory.cs                              │
│  └── [ML.NET Integration]                                   │
└─────────────────────────────────────────────────────────────┘
```

---

*[Remaining legacy documentation preserved for historical reference]*

</details>

---

**Last Updated**: January 2026
**Version**: v1.2.0

