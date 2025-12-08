# AI Agent Architecture

Comprehensive technical documentation for MLoop's AI Agent system built on Ironbees Agent Mode.

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Agent Hierarchy](#3-agent-hierarchy)
4. [Core Components](#4-core-components)
5. [LLM Provider Integration](#5-llm-provider-integration)
6. [Agent Implementations](#6-agent-implementations)
7. [Data Flow](#7-data-flow)
8. [Extensibility](#8-extensibility)
9. [Testing Strategy](#9-testing-strategy)
10. [Performance Considerations](#10-performance-considerations)

---

## 1. Overview

### 1.1 Purpose

MLoop's AI Agent system provides intelligent ML workflow assistance through specialized conversational agents. Each agent is designed for specific tasks in the ML lifecycle, from data analysis to model deployment.

### 1.2 Design Goals

- **Specialized Expertise**: Each agent focuses on a specific domain
- **Provider Agnostic**: Support multiple LLM providers (GPUStack, Anthropic, OpenAI, Azure)
- **Stateless Operation**: Align with MLoop's multi-process casual design
- **Extensible Architecture**: Easy to add new agents and capabilities
- **Production Ready**: Robust error handling and graceful degradation

### 1.3 Key Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Ironbees Agent Mode | 0.1.5+ | Agent framework with ConversationalAgent |
| Microsoft.Extensions.AI | 9.3.0 | LLM provider abstraction |
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

## 3. Agent Hierarchy

### 3.1 Base Class: ConversationalAgent

All agents extend `Ironbees.AgentMode.ConversationalAgent`:

```csharp
public abstract class ConversationalAgent
{
    protected IChatClient ChatClient { get; }
    protected string SystemPrompt { get; }

    public virtual async Task<string> ChatAsync(
        string userMessage,
        CancellationToken cancellationToken = default);

    public virtual IAsyncEnumerable<string> StreamAsync(
        string userMessage,
        CancellationToken cancellationToken = default);
}
```

### 3.2 Agent Inheritance Pattern

```
ConversationalAgent (Ironbees)
    │
    ├── DataAnalystAgent
    │   └── + DataLoaderFactory integration
    │   └── + Statistical analysis methods
    │
    ├── PreprocessingExpertAgent
    │   └── + FilePrepperImpl integration
    │   └── + Script generation methods
    │
    ├── ModelArchitectAgent
    │   └── + ML.NET configuration generation
    │   └── + AutoML recommendation methods
    │
    └── MLOpsManagerAgent
        └── + MLoopProjectManager integration
        └── + Workflow orchestration methods
        └── + Multi-agent coordination
```

### 3.3 Agent Responsibilities

| Agent | Primary Role | Engine Integration |
|-------|--------------|-------------------|
| DataAnalystAgent | Dataset analysis & ML readiness | DataLoaderFactory |
| PreprocessingExpertAgent | C# preprocessing script generation | FilePrepperImpl |
| ModelArchitectAgent | Problem classification & AutoML config | ML.NET AutoML |
| MLOpsManagerAgent | End-to-end workflow orchestration | MLoopProjectManager |

---

## 4. Core Components

### 4.1 LlmProviderFactory

Central factory for creating LLM client instances:

```csharp
namespace MLoop.AIAgent.Core;

public static class LlmProviderFactory
{
    public static IChatClient? CreateChatClient()
    {
        // Priority order:
        // 1. GPUStack (local, cost-effective)
        // 2. Anthropic (production quality)
        // 3. Azure OpenAI (enterprise)
        // 4. OpenAI (development)

        return TryCreateGpuStackClient()
            ?? TryCreateAnthropicClient()
            ?? TryCreateAzureOpenAIClient()
            ?? TryCreateOpenAIClient();
    }
}
```

**Environment Variables**:
```bash
# GPUStack
GPUSTACK_ENDPOINT, GPUSTACK_API_KEY, GPUSTACK_MODEL

# Anthropic
ANTHROPIC_API_KEY, ANTHROPIC_MODEL

# Azure OpenAI
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL

# OpenAI
OPENAI_API_KEY, OPENAI_MODEL
```

### 4.2 MLoopProjectManager

CLI wrapper for MLoop operations:

```csharp
namespace MLoop.AIAgent.Core;

public class MLoopProjectManager
{
    private readonly string _cliPath;
    private readonly string _dotnetPath;

    // Project lifecycle
    public Task<MLoopOperationResult> InitializeProjectAsync(MLoopProjectConfig config);

    // Training operations
    public Task<MLoopOperationResult> TrainModelAsync(MLoopTrainingConfig config);
    public Task<MLoopOperationResult> EvaluateModelAsync(string? experimentId = null);

    // Prediction
    public Task<MLoopOperationResult> PredictAsync(string inputPath, string outputPath);

    // Data operations
    public Task<MLoopOperationResult> PreprocessDataAsync(string inputPath);
    public Task<MLoopOperationResult> GetDatasetInfoAsync(string dataPath);

    // Experiment management
    public Task<List<MLoopExperiment>> ListExperimentsAsync();
    public Task<MLoopOperationResult> PromoteExperimentAsync(string experimentId);
}
```

### 4.3 Data Models

```csharp
namespace MLoop.AIAgent.Core.Models;

public class MLoopProjectConfig
{
    public required string ProjectName { get; set; }
    public required string DataPath { get; set; }
    public required string LabelColumn { get; set; }
    public required string TaskType { get; set; }  // binary-classification, multiclass, regression
    public string? ProjectDirectory { get; set; }
}

public class MLoopTrainingConfig
{
    public int TimeSeconds { get; set; } = 120;
    public string Metric { get; set; } = "Accuracy";
    public double TestSplit { get; set; } = 0.2;
    public string? DataPath { get; set; }
    public string? ExperimentName { get; set; }
}

public class MLoopExperiment
{
    public required string Id { get; set; }
    public required string Name { get; set; }
    public DateTime Timestamp { get; set; }
    public string? Trainer { get; set; }
    public double MetricValue { get; set; }
    public string? MetricName { get; set; }
    public bool IsProduction { get; set; }
}

public class MLoopOperationResult
{
    public bool Success { get; set; }
    public int ExitCode { get; set; }
    public string Output { get; set; } = string.Empty;
    public string? Error { get; set; }
    public Dictionary<string, object> Data { get; set; } = [];
}
```

---

## 5. LLM Provider Integration

### 5.1 Provider Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Microsoft.Extensions.AI                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    IChatClient                        │  │
│  │  Task<ChatResponse> SendAsync(ChatRequest)           │  │
│  │  IAsyncEnumerable<StreamingUpdate> StreamAsync()     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┐
         │               │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │GPUStack │    │Anthropic│    │Azure    │    │ OpenAI  │
    │ Client  │    │ Client  │    │OpenAI   │    │ Client  │
    └─────────┘    └─────────┘    │ Client  │    └─────────┘
                                  └─────────┘
```

### 5.2 Provider Selection Logic

```csharp
public static IChatClient? CreateChatClient()
{
    // 1. Check GPUStack (local deployment)
    var gpuStackEndpoint = Environment.GetEnvironmentVariable("GPUSTACK_ENDPOINT");
    var gpuStackKey = Environment.GetEnvironmentVariable("GPUSTACK_API_KEY");
    var gpuStackModel = Environment.GetEnvironmentVariable("GPUSTACK_MODEL");

    if (!string.IsNullOrEmpty(gpuStackEndpoint) &&
        !string.IsNullOrEmpty(gpuStackKey) &&
        !string.IsNullOrEmpty(gpuStackModel))
    {
        return new OpenAIClient(new ApiKeyCredential(gpuStackKey), new()
        {
            Endpoint = new Uri(gpuStackEndpoint)
        }).AsChatClient(gpuStackModel);
    }

    // 2. Check Anthropic
    var anthropicKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
    var anthropicModel = Environment.GetEnvironmentVariable("ANTHROPIC_MODEL");

    if (!string.IsNullOrEmpty(anthropicKey) && !string.IsNullOrEmpty(anthropicModel))
    {
        return new AnthropicClient(anthropicKey).AsChatClient(anthropicModel);
    }

    // 3. Check Azure OpenAI
    // 4. Check OpenAI
    // ... similar pattern

    return null;
}
```

### 5.3 Provider Capabilities Matrix

| Feature | GPUStack | Anthropic | Azure OpenAI | OpenAI |
|---------|----------|-----------|--------------|--------|
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Tool Use | ⚠️ Model-dependent | ✅ | ✅ | ✅ |
| Vision | ⚠️ Model-dependent | ✅ | ✅ | ✅ |
| Context Window | Model-dependent | 200K | 128K | 128K |
| Local Deployment | ✅ | ❌ | ❌ | ❌ |

---

## 6. Agent Implementations

### 6.1 DataAnalystAgent

**Purpose**: Dataset analysis, ML readiness assessment, statistical insights

```csharp
public class DataAnalystAgent : ConversationalAgent
{
    private const string SystemPrompt = """
        You are an expert data analyst specializing in ML/AI data preparation.

        ## Core Responsibilities
        1. Analyze dataset structure, types, and statistics
        2. Identify data quality issues (missing, outliers, imbalance)
        3. Assess ML readiness and recommend preprocessing
        4. Provide clear, actionable insights

        ## Output Format
        - Dataset Overview: rows, columns, memory
        - Column Analysis: type, missing %, unique values
        - Statistical Summary: mean, std, quartiles
        - ML Readiness Assessment: score and recommendations
        """;

    private readonly DataLoaderFactory _dataLoader;

    public async Task<string> AnalyzeDatasetAsync(string dataPath);
    public async Task<string> AnalyzeDatasetForLLMAsync(string dataPath);
}
```

**Key Methods**:
- `AnalyzeDatasetAsync`: Direct analysis with structured output
- `AnalyzeDatasetForLLMAsync`: LLM-enhanced analysis with recommendations
- `GetDatasetSummaryAsync`: Quick statistical overview

### 6.2 PreprocessingExpertAgent

**Purpose**: Generate C# preprocessing scripts for MLoop extensibility system

```csharp
public class PreprocessingExpertAgent : ConversationalAgent
{
    private const string SystemPrompt = """
        You are an expert in data preprocessing for ML.NET and MLoop.

        ## Core Responsibilities
        1. Analyze preprocessing requirements
        2. Generate complete, executable C# scripts
        3. Follow MLoop IPreprocessingScript interface
        4. Use sequential naming (01_*, 02_*, 03_*)

        ## Script Interface
        public interface IPreprocessingScript
        {
            Task<string> ExecuteAsync(PreprocessContext context);
        }

        ## Output Format
        Complete C# code ready to save to .mloop/scripts/preprocess/
        """;

    private readonly FilePrepperImpl _filePrepper;

    public async Task<string> GenerateScriptAsync(string requirement);
    public async Task<string> GetAvailableTransformsAsync();
}
```

**Script Generation Pattern**:
```csharp
// Generated output example
// .mloop/scripts/preprocess/01_handle_missing.cs

using MLoop.Extensibility;

public class HandleMissingValues : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        // Implementation generated by agent
    }
}
```

### 6.3 ModelArchitectAgent

**Purpose**: ML problem classification, AutoML configuration recommendation

```csharp
public class ModelArchitectAgent : ConversationalAgent
{
    private const string SystemPrompt = """
        You are an ML architect specializing in AutoML configuration.

        ## Core Responsibilities
        1. Classify ML problem type (classification, regression)
        2. Recommend optimal metrics and time budgets
        3. Suggest feature engineering approaches
        4. Generate mloop train commands

        ## ML.NET Trainers Knowledge
        - Binary: LightGbm, FastTree, SdcaLogistic, AveragedPerceptron
        - Multiclass: LightGbm, FastTree, SdcaMaximumEntropy
        - Regression: LightGbm, FastTree, Sdca, Ols

        ## Output Format
        - Problem Classification
        - Recommended Configuration
        - Complete CLI Command
        """;

    public async Task<string> RecommendConfigurationAsync(string problemDescription);
    public async Task<string> ClassifyProblemAsync(string datasetInfo);
}
```

### 6.4 MLOpsManagerAgent

**Purpose**: End-to-end workflow orchestration with MLoop CLI integration

```csharp
public class MLOpsManagerAgent : ConversationalAgent
{
    private const string SystemPrompt = """
        You are an MLOps expert managing the complete ML lifecycle.

        ## Core Responsibilities
        1. Orchestrate ML workflows (init → train → evaluate → deploy)
        2. Execute MLoop CLI commands
        3. Monitor experiments and metrics
        4. Coordinate with other agents

        ## Available Operations
        - InitializeProject: Create new MLoop project
        - TrainModel: Execute AutoML training
        - EvaluateModel: Assess model performance
        - Predict: Generate batch predictions
        - PromoteExperiment: Deploy model to production
        """;

    private readonly MLoopProjectManager _projectManager;

    // LLM-integrated methods (conversation + operation)
    public async Task<string> InitializeProjectAsync(MLoopProjectConfig config, ...);
    public async Task<string> TrainModelAsync(MLoopTrainingConfig config, ...);
    public async Task<string> EvaluateModelAsync(string? experimentId, ...);
    public async Task<string> PredictAsync(string inputPath, string outputPath, ...);
    public async Task<string> ListExperimentsAsync(...);
    public async Task<string> PromoteExperimentAsync(string experimentId, ...);

    // Raw result methods (direct operation)
    public async Task<MLoopOperationResult> GetInitResultAsync(MLoopProjectConfig config);
    public async Task<MLoopOperationResult> GetTrainResultAsync(MLoopTrainingConfig config);
    public async Task<MLoopOperationResult> GetEvaluateResultAsync(string? experimentId);
    public async Task<MLoopOperationResult> GetPredictResultAsync(string input, string output);
    public async Task<List<MLoopExperiment>> GetExperimentsAsync();
    public async Task<MLoopOperationResult> GetPromoteResultAsync(string experimentId);
}
```

**Dual Interface Pattern**:

```csharp
// Pattern 1: LLM-integrated (conversational)
public async Task<string> TrainModelAsync(
    MLoopTrainingConfig config,
    CancellationToken cancellationToken = default)
{
    var result = await _projectManager.TrainModelAsync(config);
    var formattedResult = FormatTrainResultForLLM(result);
    return await ChatAsync(formattedResult, cancellationToken);
}

// Pattern 2: Raw result (programmatic)
public async Task<MLoopOperationResult> GetTrainResultAsync(
    MLoopTrainingConfig config)
{
    return await _projectManager.TrainModelAsync(config);
}

// Helper: Format for LLM context
private static string FormatTrainResultForLLM(MLoopOperationResult result)
{
    var sb = new StringBuilder();
    sb.AppendLine("## Training Operation Result");
    sb.AppendLine($"Status: {(result.Success ? "SUCCESS" : "FAILED")}");
    // ... detailed formatting
    return sb.ToString();
}
```

---

## 7. Data Flow

### 7.1 Agent Chat Flow

```
User Input
    │
    ▼
┌─────────────────┐
│  AgentCommand   │
│  (CLI Router)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent Selection │
│ (by agent name) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LlmProviderFactory │
│ CreateChatClient() │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Selected Agent  │
│ Constructor     │
│ (IChatClient)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent.ChatAsync │
│ or StreamAsync  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM API Call    │
│ (Provider)      │
└────────┬────────┘
         │
         ▼
Response to User
```

### 7.2 MLOps Workflow Flow

```
User: "Train model on data.csv"
    │
    ▼
┌─────────────────────┐
│ MLOpsManagerAgent   │
│ ChatAsync()         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Parse User Intent   │
│ (via LLM)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ MLoopProjectManager │
│ TrainModelAsync()   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Execute CLI:        │
│ dotnet run -- train │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FormatResultForLLM  │
│ (Structure output)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LLM Enhancement     │
│ (Explain result)    │
└──────────┬──────────┘
           │
           ▼
Natural Language Response
```

---

## 8. Extensibility

### 8.1 Adding New Agents

**Step 1**: Create agent class

```csharp
namespace MLoop.AIAgent.Agents;

public class CustomAgent : ConversationalAgent
{
    private const string SystemPrompt = """
        You are a [role description].

        ## Core Responsibilities
        1. [Responsibility 1]
        2. [Responsibility 2]

        ## Output Format
        [Describe expected output]
        """;

    public CustomAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
    }

    // Add specialized methods
    public async Task<string> CustomOperationAsync(string input)
    {
        var context = PrepareContext(input);
        return await ChatAsync(context);
    }
}
```

**Step 2**: Register in AgentCommand

```csharp
// In AgentCommand.cs
private static ConversationalAgent CreateAgent(string agentName, IChatClient chatClient)
{
    return agentName.ToLowerInvariant() switch
    {
        "data-analyst" => new DataAnalystAgent(chatClient),
        "preprocessing-expert" => new PreprocessingExpertAgent(chatClient),
        "model-architect" => new ModelArchitectAgent(chatClient),
        "mlops-manager" => new MLOpsManagerAgent(chatClient),
        "custom" => new CustomAgent(chatClient),  // Add here
        _ => throw new ArgumentException($"Unknown agent: {agentName}")
    };
}
```

### 8.2 Adding New LLM Providers

**Step 1**: Implement provider creation

```csharp
// In LlmProviderFactory.cs
private static IChatClient? TryCreateCustomProvider()
{
    var endpoint = Environment.GetEnvironmentVariable("CUSTOM_ENDPOINT");
    var apiKey = Environment.GetEnvironmentVariable("CUSTOM_API_KEY");
    var model = Environment.GetEnvironmentVariable("CUSTOM_MODEL");

    if (string.IsNullOrEmpty(endpoint) ||
        string.IsNullOrEmpty(apiKey) ||
        string.IsNullOrEmpty(model))
    {
        return null;
    }

    return new CustomClient(apiKey, endpoint).AsChatClient(model);
}
```

**Step 2**: Add to priority chain

```csharp
public static IChatClient? CreateChatClient()
{
    return TryCreateGpuStackClient()
        ?? TryCreateAnthropicClient()
        ?? TryCreateCustomProvider()  // Add in priority order
        ?? TryCreateAzureOpenAIClient()
        ?? TryCreateOpenAIClient();
}
```

### 8.3 Adding Engine Integrations

**Pattern**: Engine + Agent + Format methods

```csharp
public class EnhancedAgent : ConversationalAgent
{
    private readonly CustomEngine _engine;

    public EnhancedAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _engine = new CustomEngine();
    }

    // LLM-integrated method
    public async Task<string> OperationAsync(OperationConfig config, CancellationToken ct)
    {
        var result = await _engine.ExecuteAsync(config);
        var formatted = FormatResultForLLM(result);
        return await ChatAsync(formatted, ct);
    }

    // Raw result method
    public async Task<OperationResult> GetOperationResultAsync(OperationConfig config)
    {
        return await _engine.ExecuteAsync(config);
    }

    // LLM context formatter
    private static string FormatResultForLLM(OperationResult result)
    {
        // Structure result for LLM understanding
    }
}
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Agent Model Tests**:
```csharp
[Fact]
public void MLoopProjectConfig_CanBeInitialized()
{
    var config = new MLoopProjectConfig
    {
        ProjectName = "test-project",
        DataPath = "/data/test.csv",
        LabelColumn = "target",
        TaskType = "binary-classification"
    };

    Assert.Equal("test-project", config.ProjectName);
}
```

**Manager Instantiation Tests**:
```csharp
[Fact]
public void MLoopProjectManager_DefaultConstructor_CreatesInstance()
{
    var manager = new MLoopProjectManager();
    Assert.NotNull(manager);
}
```

### 9.2 Integration Tests

**CLI Integration Tests** (Skip when CLI not built):
```csharp
[Fact(Skip = "Integration test - requires MLoop CLI to be built")]
public async Task TrainModelAsync_ValidConfig_ReturnsResult()
{
    var config = new MLoopTrainingConfig
    {
        TimeSeconds = 30,
        Metric = "Accuracy"
    };

    var result = await _projectManager.TrainModelAsync(config);
    Assert.NotNull(result);
}
```

### 9.3 Test Categories

| Category | Scope | Execution |
|----------|-------|-----------|
| Model Tests | Data model initialization | Always run |
| Manager Tests | Class instantiation | Always run |
| Integration Tests | CLI execution | Skip unless CLI available |
| E2E Tests | Full workflow | Manual execution |

---

## 10. Performance Considerations

### 10.1 LLM API Optimization

- **Streaming**: Use `StreamAsync` for real-time feedback on long responses
- **Context Management**: Keep system prompts focused and concise
- **Caching**: Consider caching common analysis results

### 10.2 CLI Execution

- **Process Isolation**: Each MLoop CLI call runs in separate process
- **Timeout Handling**: Implement appropriate timeouts for long operations
- **Output Parsing**: Efficient parsing of CLI output

### 10.3 Memory Management

```csharp
// Dispose pattern for agents
public class AgentCommand : Command
{
    private async Task<int> ExecuteAsync(string agentName, string query)
    {
        using var chatClient = LlmProviderFactory.CreateChatClient();
        // ... use agent
    }
}
```

### 10.4 Metrics

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Agent Creation | < 100ms | Includes LLM client setup |
| Simple Query | 1-5s | Depends on LLM provider |
| Streaming Response | Real-time | Progressive output |
| CLI Operation | 1s - 10min | Depends on operation (train vs predict) |

---

## Appendix A: Project Structure

```
src/MLoop.AIAgent/
├── MLoop.AIAgent.csproj
│
├── Agents/
│   ├── DataAnalystAgent.cs
│   ├── PreprocessingExpertAgent.cs
│   ├── ModelArchitectAgent.cs
│   └── MLOpsManagerAgent.cs
│
└── Core/
    ├── LlmProviderFactory.cs
    ├── MLoopProjectManager.cs
    └── Models/
        └── MLoopProjectInfo.cs

tests/MLoop.AIAgent.Tests/
├── MLoop.AIAgent.Tests.csproj
├── DataAnalystAgentTests.cs
├── PreprocessingExpertAgentTests.cs
├── ModelArchitectAgentTests.cs
└── MLOpsManagerAgentTests.cs
```

## Appendix B: Configuration Reference

### Environment Variables

```bash
# Required: At least one provider
# GPUStack (Local)
GPUSTACK_ENDPOINT=http://localhost:8080/v1
GPUSTACK_API_KEY=your-key
GPUSTACK_MODEL=llama-3.1-8b

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_MODEL=gpt-4o

# OpenAI
OPENAI_API_KEY=sk-proj-your-key
OPENAI_MODEL=gpt-4o-mini
```

### .env.example Template

```bash
# MLoop AI Agent Configuration
# Copy to .env and configure your preferred provider

# Option 1: GPUStack (Local - Recommended for development)
# GPUSTACK_ENDPOINT=http://localhost:8080/v1
# GPUSTACK_API_KEY=your-gpustack-key
# GPUSTACK_MODEL=llama-3.1-8b

# Option 2: Anthropic (Recommended for production)
# ANTHROPIC_API_KEY=sk-ant-your-key
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Option 3: Azure OpenAI (Enterprise)
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
# AZURE_OPENAI_KEY=your-azure-key
# AZURE_OPENAI_MODEL=gpt-4o

# Option 4: OpenAI (Development)
# OPENAI_API_KEY=sk-proj-your-key
# OPENAI_MODEL=gpt-4o-mini
```

---

**Version**: 1.0.0
**Last Updated**: 2024-12-08
**Status**: Production Ready
