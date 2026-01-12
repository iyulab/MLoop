# MLoop Architecture Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architectural Principles](#2-architectural-principles)
3. [Process Model](#3-process-model)
4. [System Architecture](#4-system-architecture)
5. [Layer Design](#5-layer-design)
6. [Project Structure](#6-project-structure)
7. [Data Models](#7-data-models)
8. [Core Workflows](#8-core-workflows)
9. [Configuration Management](#9-configuration-management)
10. [Git Integration](#10-git-integration)
11. [Technology Stack](#11-technology-stack)
12. [Testing Strategy](#12-testing-strategy)
13. [Long-Running Tasks](#13-long-running-tasks)
14. [Extensibility System](#14-extensibility-system)
15. [Future Extensibility](#15-future-extensibility)

---

## 1. Overview

MLoop is a lightweight MLOps platform built on ML.NET, designed with a **filesystem-first** and **multi-process casual** approach that emphasizes simplicity, transparency, and Git compatibility.

### 1.1 Core Mission

**"Excellent MLOps with Minimum Cost"**

MLoop enables anyone to achieve production-quality ML models with minimal coding and ML expertise, while maintaining flexibility for advanced customization. This is accomplished through:

1. **Minimal Development Cost**: 3-command workflow (`init` → `train` → `predict`) vs traditional multi-week ML projects
2. **Minimal Knowledge Cost**: AutoML eliminates need for ML expertise (AI assistance via mloop-mcp)
3. **Minimal Operational Cost**: Filesystem-based MLOps, no infrastructure complexity
4. **Maximum Value**: Production-ready models with optional extensibility for expert users

### 1.2 Design Philosophy

**Convention Over Configuration**
- Filesystem-based contracts: Drop CSV in `datasets/`, get trained model
- Zero configuration required for 90% of use cases
- Git-friendly MLOps: All state as files, no databases
- Intelligent defaults that work immediately

**AutoML-First, Minimal Coding**
- One command trains production-ready models
- Automatic algorithm selection via ML.NET AutoML
- Optional FilePrepper integration for complex preprocessing
- No manual feature engineering unless user chooses to customize

**AI Integration via MCP (External)**
- AI agents access MLoop through mloop-mcp (separate repository)
- "AI uses MLoop" philosophy (not "MLoop contains AI")
- See docs/ECOSYSTEM.md for MLoop ecosystem architecture

**Extensibility Through Dynamic Scripting**
- Optional C# scripts for custom logic (hooks, metrics, preprocessing)
- Automatic compilation and caching for performance
- Zero overhead when extensions aren't used (<1ms impact)
- Full IDE support (IntelliSense, debugging, type safety)

### 1.3 Technical Design Principles

- **Filesystem-First**: All state managed as files, perfect Git integration
- **Multi-Process Casual**: Each command runs independently, no daemon required
- **Zero Configuration**: Usable immediately with minimal setup
- **Layer Separation**: Clear separation between CLI, Core, and Storage
- **Lightweight**: Independent operation without complex dependencies
- **AutoML-Driven**: Automatic model selection over manual tuning

### 1.4 Target Use Cases

**✅ Suitable For:**
- Medium datasets (< 1GB)
- Standard ML problems (classification, regression)
- Fast prototyping and iteration
- .NET ecosystem projects
- Intermittent ML workloads (train → exit, predict → exit)

**❌ Not Suitable For:**
- Complex feature engineering needs
- State-of-the-art performance requirements
- Large-scale datasets (> 10GB)
- Real-time learning/retraining
- Always-on service requirements

---

## 2. Architectural Principles

### 2.1 SOLID Principles

- **Single Responsibility**: Each component has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes substitutable for base classes
- **Interface Segregation**: No dependencies on unused interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

### 2.2 Design Philosophy

#### Lightweight First
- User handles data preprocessing
- Accept clean, preprocessed data only
- Focus on AutoML capabilities
- Minimal dependency footprint
- **No daemon or background service management**

#### Usability Over Everything
- AutoML-centric workflow
- Speed over precision for prototyping
- 3-step workflow: `init → train → predict`
- Sensible defaults for everything
- **Simple process lifecycle: Start → Execute → Exit**

#### Filesystem-First MLOps
- All state persisted as files
- Git-friendly structure
- No external databases or services
- Human-readable formats (JSON, YAML)
- **Natural multi-process isolation via filesystem**

---

## 3. Process Model

### 3.1 Multi-Process Casual Design

**Core Concept**: Each MLoop command runs as an **independent process** that exits when complete.

```
┌─────────────────────────────────────────────────┐
│  Traditional Daemon Model (NOT MLoop)           │
├─────────────────────────────────────────────────┤
│  $ mloop daemon start      ← Start service     │
│  $ mloop train data.csv    ← Submit to daemon  │
│  $ mloop daemon stop       ← Stop service      │
│                                                 │
│  ❌ Complex: Port management, PID files, etc.  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  MLoop Multi-Process Model (Phase 1)            │
├─────────────────────────────────────────────────┤
│  $ mloop train data.csv    ← Start process     │
│      [Training... 5 min]                        │
│      ✅ Complete, exit                          │
│                                                 │
│  $ mloop predict model.zip data.csv             │
│      [Predicting... 10 sec]                     │
│      ✅ Complete, exit                          │
│                                                 │
│  ✅ Simple: No daemon, no management            │
└─────────────────────────────────────────────────┘
```

### 3.2 Why Multi-Process Casual?

#### Perfect for MLoop's Usage Pattern

**MLoop workloads are intermittent, not continuous:**
```bash
# Typical workflow
$ mloop train data.csv --label target
  → Train for 10 minutes → Exit

$ mloop evaluate exp-001/model.zip test.csv
  → Evaluate for 30 seconds → Exit

$ mloop predict exp-001/model.zip new-data.csv
  → Predict for 10 seconds → Exit
```

**vs Docker (continuous service):**
```bash
$ docker run -d my-service  # Daemon mode
$ docker ps                 # Persistent service
$ docker stop my-service    # Explicit stop
```

MLoop doesn't need to stay running between operations.

#### Advantages

| Aspect | Multi-Process | Background Daemon |
|--------|---------------|-------------------|
| **Simplicity** | ⭐⭐⭐⭐⭐ Minimal | ⭐⭐ Complex |
| **Management** | ✅ None required | ❌ start/stop/status/restart |
| **Isolation** | ✅ Perfect (per-process) | ⚠️ Shared daemon |
| **Concurrent Jobs** | ✅ Natural support | ⚠️ Requires job queue |
| **Debugging** | ✅ Direct terminal output | ❌ Log file inspection |
| **Port Conflicts** | ✅ N/A | ❌ Possible |
| **Process Cleanup** | ✅ OS handles | ❌ Manual management |
| **Usage Pattern Fit** | ✅ Perfect match | ❌ Over-engineering |

#### Disadvantages (and Mitigation)

**Concern**: "What about long-running training jobs?"

**Solution**: Use standard Unix tools (see [Section 13](#13-long-running-tasks))
```bash
# Option 1: nohup
$ nohup mloop train data.csv --label target --time 3600 &

# Option 2: screen/tmux
$ screen -S ml-training
$ mloop train data.csv --label target --time 3600
# Ctrl+A, D to detach
```

### 3.3 Exception: mloop serve

The `mloop serve` command is the **only exception** that maintains a server process:

```bash
$ mloop serve models/production/model.zip --port 5000
🚀 MLoop API Server
   Listening on: http://localhost:5000
   Swagger UI: http://localhost:5000/swagger

   Press Ctrl+C to stop

# Server process stays alive
# Standard web server pattern
# User explicitly controls lifecycle
```

**Why this is different:**
- API serving requires persistent HTTP listener
- User explicitly requests long-running service
- Standard web server behavior (like `dotnet run`)
- Simple lifecycle: Start with command, stop with Ctrl+C

**No daemon management needed:**
- No background service
- No `mloop serve start/stop/status`
- Just run in terminal or use `nohup` if needed

### 3.4 Concurrent Execution

**Natural isolation via filesystem:**

```bash
# Terminal 1: Project A
$ cd ~/projects/sentiment-analyzer
$ mloop train data.csv --label sentiment
  → experiments/exp-001/

# Terminal 2: Project B (concurrent!)
$ cd ~/projects/price-predictor
$ mloop train data.csv --label price
  → experiments/exp-001/

# No conflicts!
# Each project has independent .mloop/ directory
# Filesystem provides natural isolation
```

**Same project, different experiments:**
```bash
# Terminal 1
$ mloop train data.csv --label target --time 300
  → experiments/exp-001/

# Terminal 2 (while #1 is running)
$ mloop train data.csv --label target --time 600 --metric f1
  → experiments/exp-002/

# Works perfectly!
# Each gets unique experiment ID
# No shared state, no locks needed
```

---

## 4. System Architecture

### 4.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              User Interface Layer (CLI)                      │
│                                                              │
│  Each command execution = New process instance              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Command    │  │  Validation  │  │   Progress   │      │
│  │   Parsing    │  │   & Error    │  │   Reporter   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ In-Process Method Calls
                         │ (No IPC, No gRPC)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Core Engine Layer                         │
│                                                              │
│  Instantiated per command execution                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │  Prediction  │  │  Evaluation  │      │
│  │   Engine     │  │   Engine     │  │   Engine     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   AutoML     │  │    Model     │  │  Experiment  │      │
│  │   Runner     │  │   Registry   │  │    Store     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │         FileSystem Manager                       │        │
│  │         (Storage Abstraction)                    │        │
│  └─────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ File I/O Operations
                         │ (Stateless, thread-safe)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Storage Layer                             │
│                                                              │
│  Filesystem Structure (Provides isolation)                   │
│  ├── .mloop/          (Project metadata)                    │
│  ├── experiments/     (Training results)                    │
│  ├── models/          (Promoted models)                     │
│  ├── data/            (User data)                           │
│  └── outputs/         (Predictions, reports)                │
└─────────────────────────────────────────────────────────────┘

Process Lifecycle:
1. User runs command → New process starts
2. CLI parses → Core executes → Storage persists
3. Command completes → Process exits
4. No background service, no daemon
```

### 4.2 Process Lifecycle

```
┌──────────────────────────────────────────────────┐
│  mloop train data.csv --label target             │
└────────────────┬─────────────────────────────────┘
                 │
         ┌───────▼───────┐
         │ Process Start │
         └───────┬───────┘
                 │
         ┌───────▼────────┐
         │  Load Config   │
         │  Find Project  │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Execute Core  │
         │  (Training)    │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Save Results  │
         │  to Filesystem │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │ Process Exit   │
         │  (Return code) │
         └────────────────┘

No state preserved between executions
Next command starts fresh process
```

### 4.3 State Management

**All state lives in filesystem, not in memory:**

```csharp
// ❌ ANTI-PATTERN: In-memory state (daemon model)
public class MLoopDaemon
{
    private Dictionary<string, TrainingJob> _activeJobs;
    private ModelCache _modelCache;
    // State lost on process exit!
}

// ✅ MLoop Pattern: Filesystem state
public class TrainingEngine
{
    public async Task<TrainingResult> TrainAsync(TrainingConfig config)
    {
        // 1. Read config from filesystem
        var projectConfig = await _fileSystem.LoadConfigAsync();

        // 2. Execute training
        var result = await _autoML.TrainAsync(config);

        // 3. Persist everything to filesystem
        await _experimentStore.SaveAsync(result);

        // 4. Process exits, state preserved in files
        return result;
    }
}
```

**Benefits:**
- ✅ No state synchronization between processes
- ✅ Crash-safe (filesystem is durable)
- ✅ Easy to inspect state (`cat experiments/exp-001/metadata.json`)
- ✅ Natural versioning with Git

---

## 5. Layer Design

### 5.1 User Interface Layer (CLI)

**Responsibilities:**
- Parse and validate user commands
- Instantiate Core Engine components
- Provide real-time user feedback (progress bars, logs)
- Format output (tables, JSON, text)
- Exit with appropriate return code

**Key Components:**
- **Command Parser**: System.CommandLine for parsing
- **Validator**: Input validation and error handling
- **Progress Reporter**: Spectre.Console for visual feedback
- **Output Formatter**: Multiple output format support

**Design Guidelines:**
- Keep layer thin (no business logic)
- Direct method calls to Core Engine (no IPC)
- Focus on user experience optimization
- Synchronous execution (await completion)

**Example Implementation:**
```csharp
public class TrainCommand : Command
{
    public TrainCommand() : base("train", "Train a model using AutoML")
    {
        var dataFileArg = new Argument<string>("data-file");
        var labelOption = new Option<string>("--label");

        AddArgument(dataFileArg);
        AddOption(labelOption);

        this.SetHandler(ExecuteAsync, dataFileArg, labelOption);
    }

    private async Task<int> ExecuteAsync(string dataFile, string label)
    {
        try
        {
            // 1. Validate inputs
            if (!File.Exists(dataFile))
                return Error("Data file not found");

            // 2. Find project root
            var projectRoot = ProjectDiscovery.FindRoot();

            // 3. Instantiate Core Engine (in-process)
            var engine = new TrainingEngine(projectRoot);

            // 4. Execute training
            var result = await engine.TrainAsync(new TrainingConfig
            {
                DataFile = dataFile,
                LabelColumn = label
            });

            // 5. Display results
            DisplayResults(result);

            // 6. Exit with success
            return 0;
        }
        catch (Exception ex)
        {
            Error(ex.Message);
            return 1;
        }
        // Process exits after return
    }
}
```

### 5.2 Core Engine Layer

**Responsibilities:**
- All ML-related business logic
- Filesystem state management
- ML.NET and AutoML orchestration
- Experiment tracking and model registry

**Key Components:**

#### Training Engine
```csharp
namespace MLoop.Core.AutoML;

public interface ITrainingEngine
{
    Task<TrainingResult> TrainAsync(
        TrainingConfig config,
        IProgress<TrainingProgress> progress,
        CancellationToken cancellationToken);
}

public class TrainingEngine : ITrainingEngine
{
    private readonly IFileSystemManager _fileSystem;
    private readonly IAutoMLRunner _autoML;
    private readonly IExperimentStore _experiments;

    public async Task<TrainingResult> TrainAsync(
        TrainingConfig config,
        IProgress<TrainingProgress> progress,
        CancellationToken cancellationToken)
    {
        // 1. Generate experiment ID
        var experimentId = await _experiments.GenerateIdAsync();

        // 2. Run AutoML
        var mlResult = await _autoML.RunAsync(config, progress, cancellationToken);

        // 3. Save results to filesystem
        await _experiments.SaveAsync(experimentId, mlResult);

        // 4. Return (process will exit)
        return new TrainingResult
        {
            ExperimentId = experimentId,
            BestTrainer = mlResult.BestTrainer,
            Metrics = mlResult.Metrics
        };
    }
}
```

#### Prediction Engine
```csharp
namespace MLoop.Core.Models;

public interface IPredictionEngine
{
    Task<PredictionResult> PredictAsync(
        string modelPath,
        IDataSource dataSource,
        CancellationToken cancellationToken);
}
```

#### Evaluation Engine
```csharp
namespace MLoop.Core.Evaluation;

public interface IEvaluator
{
    Task<EvaluationResult> EvaluateAsync(
        string modelPath,
        IDataSource testData,
        CancellationToken cancellationToken);
}
```

**Thread Safety:**
- Each process instance is single-threaded for commands
- No shared state between processes
- Filesystem operations use OS-level locking
- No need for distributed locks or semaphores

### 5.3 Storage Layer (Filesystem)

**Responsibilities:**
- Persist all project data as files
- Provide Git-friendly structure
- Use human-readable formats (JSON, YAML)
- Enable natural multi-process isolation

**Thread-Safe Operations:**
```csharp
public class FileSystemManager : IFileSystemManager
{
    // Atomic operations via OS
    public async Task<string> GenerateExperimentIdAsync()
    {
        // Read-Modify-Write with retry
        for (int retry = 0; retry < 3; retry++)
        {
            try
            {
                var index = await ReadIndexAsync();
                var newId = $"exp-{index.NextId:D3}";
                index.NextId++;
                await WriteIndexAsync(index);
                return newId;
            }
            catch (IOException)
            {
                // Another process modified file, retry
                await Task.Delay(100);
            }
        }
        throw new ConcurrencyException("Failed to generate ID");
    }
}
```

---

## 6. Project Structure

### 6.1 Development Project Structure

MLoop is organized into **6 separate projects** with clear separation of concerns:

```
MLoop/
├── src/
│   ├── MLoop.sln                        # .NET 10 Solution
│   │
│   ├── MLoop.Extensibility/             # Extension interfaces (no dependencies)
│   │   ├── Preprocessing/
│   │   │   └── IPreprocessingScript.cs  # Custom preprocessing interface
│   │   ├── Hooks/
│   │   │   └── IHook.cs                 # Lifecycle hook interface
│   │   └── Metrics/
│   │       └── ICustomMetric.cs         # Custom metric interface
│   │
│   ├── MLoop.Core/                      # Core ML engine
│   │   ├── AutoML/                      # ML.NET AutoML wrapper
│   │   │   ├── TrainingEngine.cs
│   │   │   └── TrainingConfig.cs
│   │   ├── Data/                        # Data loading, encoding detection
│   │   │   ├── DataLoaderFactory.cs
│   │   │   └── CsvHelperImpl.cs
│   │   ├── Preprocessing/               # FilePrepper integration
│   │   ├── Scripting/                   # C# script compilation/execution
│   │   ├── Hooks/                       # Hook execution
│   │   ├── Metrics/                     # Metric processing
│   │   └── Models/                      # Domain models (Experiment, etc.)
│   │
│   ├── MLoop.CLI/                       # Command-line interface
│   │   ├── Commands/                    # CLI commands
│   │   │   ├── InitCommand.cs          # mloop init
│   │   │   ├── TrainCommand.cs         # mloop train
│   │   │   ├── PredictCommand.cs       # mloop predict
│   │   │   ├── EvaluateCommand.cs      # mloop evaluate
│   │   │   ├── ListCommand.cs          # mloop list
│   │   │   ├── PromoteCommand.cs       # mloop promote
│   │   │   ├── ServeCommand.cs         # mloop serve (launches API)
│   │   │   ├── DockerCommand.cs        # mloop docker
│   │   │   └── InfoCommand.cs          # mloop info
│   │   ├── Infrastructure/              # Console output, DI setup
│   │   └── Templates/                   # Dockerfile templates
│   │
│   ├── MLoop.API/                       # REST API server (ASP.NET Core)
│   │   ├── Program.cs                   # Minimal API endpoints
│   │   └── appsettings.json             # API configuration
│   │
│   ├── MLoop.DataStore/                 # Prediction logging & feedback (MLOps)
│   │   └── Interfaces/
│   │       ├── IPredictionLogger.cs     # Prediction logging interface
│   │       ├── IFeedbackCollector.cs    # Ground truth feedback collection
│   │       └── IDataSampler.cs          # Production data sampling
│   │
│   └── MLoop.Ops/                       # MLOps automation
│       └── Interfaces/
│           ├── IRetrainingTrigger.cs    # Retraining condition evaluation
│           ├── IModelComparer.cs        # Model performance comparison
│           └── IPromotionManager.cs     # Automated model promotion
│
├── tests/
│   ├── MLoop.Core.Tests/
│   ├── MLoop.API.Tests/
│   └── MLoop.Pipeline.Tests/
│
├── examples/                            # Example projects
│   ├── customer-churn/
│   ├── equipment-anomaly-detection/
│   └── tutorials/
│
├── docs/                                # Documentation
│   ├── ARCHITECTURE.md                  # This file
│   ├── GUIDE.md                         # User guide
│   └── ECOSYSTEM.md                     # MLoop ecosystem overview
│
├── mcp/                                 # [Submodule] mloop-mcp (MCP Server)
│   └── (https://github.com/iyulab/mloop-mcp)
│
├── studio/                              # [Submodule] mloop-studio (Web Platform)
│   └── (https://github.com/iyulab/mloop-studio)
│
├── Directory.Build.props                # Central package management
├── Directory.Packages.props             # Package versions
└── .gitignore
```

### 6.2 Project Dependencies

```
MLoop.Extensibility  ← (interfaces only, no dependencies)
        ↑
    MLoop.Core       ← ML.NET, FilePrepper
        ↑
    ┌───┼───────────┐
    │   │           │
    │   │   ┌───────┴───────┐
    │   │   │               │
MLoop.CLI  MLoop.API  MLoop.DataStore  MLoop.Ops
    │                       │              │
    │                       └──────────────┘
    └─── (CLI launches API via ServeCommand)

Note: DataStore/Ops are MLOps extensions (separate from core CLI)
```

| Project | Role | Key Dependencies |
|---------|------|------------------|
| MLoop.Extensibility | Extension interfaces | None |
| MLoop.Core | ML engine | ML.NET, FilePrepper |
| MLoop.CLI | CLI tool (`mloop`) | System.CommandLine, Spectre.Console |
| MLoop.API | REST API server | ASP.NET Core, Serilog |
| MLoop.DataStore | Prediction logging & feedback | MLoop.Core |
| MLoop.Ops | MLOps automation | MLoop.Core |

### 6.3 User Project Structure (Multi-Model)

MLoop v0.2.0+ supports **multiple models** within a single project. When users run `mloop init my-project --task binary-classification`:

```
my-project/
├── .mloop/                              # Internal (Git ignored)
│   ├── config.json                      # Project settings
│   └── models.json                      # Model name registry
│
├── mloop.yaml                           # User config (Git)
├── .gitignore                           # MLoop gitignore
├── README.md                            # Project guide
│
├── datasets/                            # Training data (Git)
│   ├── train.csv
│   ├── test.csv
│   └── predict.csv
│
├── models/                              # Per-model directories
│   ├── default/                         # Default model (--name omitted)
│   │   ├── staging/                     # Experiments
│   │   │   ├── exp-001/
│   │   │   │   ├── model.zip           # Trained model (ignored)
│   │   │   │   ├── experiment.json     # Experiment metadata (Git)
│   │   │   │   └── training.log        # Training log (ignored)
│   │   │   └── exp-002/
│   │   ├── production/                  # Promoted model
│   │   │   ├── model.zip               # (ignored)
│   │   │   └── metadata.json           # (Git)
│   │   └── registry.json               # Model-specific registry
│   │
│   ├── churn/                           # Named model example
│   │   ├── staging/
│   │   ├── production/
│   │   └── registry.json
│   │
│   └── revenue/                         # Another named model
│       ├── staging/
│       ├── production/
│       └── registry.json
│
└── predictions/                         # Prediction outputs
    ├── default/
    ├── churn/
    └── revenue/
```

### 6.4 Multi-Model Configuration

#### mloop.yaml Schema

```yaml
# Project-level settings
project: customer-analytics

# Model definitions
models:
  default:                      # Required: default model
    task: binary-classification
    label: Churn
    description: Customer churn prediction
    training:
      time_limit_seconds: 300
      metric: F1Score
      test_split: 0.2

  revenue:                      # Optional: named models
    task: regression
    label: Revenue
    description: Revenue prediction model
    training:
      time_limit_seconds: 600
      metric: RSquared

# Shared data settings
data:
  train: datasets/train.csv
  test: datasets/test.csv
```

### 6.5 Multi-Model CLI Usage

```bash
# Default model (--name omitted)
mloop train datasets/train.csv Churn --time 60
mloop predict
mloop promote exp-001

# Named model
mloop train datasets/train.csv Revenue --name revenue --time 60
mloop predict --name revenue
mloop promote exp-001 --name revenue

# List experiments across models
mloop list                    # All models
mloop list --name default     # Specific model

# Multi-model serving
mloop serve                   # Serves all production models
# Routes: /predict?name=default, /predict?name=revenue
```

---

## 7. Data Models

### 7.1 Internal Management Files

#### .mloop/config.json
```json
{
  "project": "my-ml-project",
  "version": "0.2.0",
  "created_at": "2025-12-08T10:00:00Z",
  "mloop_version": "0.2.0"
}
```

#### .mloop/models.json (Model Registry Index)
```json
{
  "models": {
    "default": {
      "created_at": "2025-12-08T10:00:00Z",
      "task": "binary-classification",
      "label": "Churn",
      "experiment_count": 5,
      "production_experiment": "exp-003"
    },
    "revenue": {
      "created_at": "2025-12-08T11:00:00Z",
      "task": "regression",
      "label": "Revenue",
      "experiment_count": 3,
      "production_experiment": "exp-002"
    }
  }
}
```

#### models/{name}/registry.json (Per-Model Registry)
```json
{
  "next_id": 6,
  "production": {
    "experiment_id": "exp-003",
    "promoted_at": "2025-12-08T14:00:00Z",
    "metrics": {
      "F1Score": 0.897,
      "Accuracy": 0.913
    }
  }
}
```

### 7.2 Experiment Files

#### models/{name}/staging/exp-XXX/experiment.json
```json
{
  "model_name": "default",
  "experiment_id": "exp-001",
  "timestamp": "2025-12-08T12:00:00Z",
  "status": "completed",
  "task": "binary-classification",
  "data": {
    "train_file": "datasets/train.csv",
    "rows": 10000,
    "features": 15,
    "label": "Churn"
  },
  "config": {
    "time_limit_seconds": 300,
    "metric": "F1Score",
    "test_split": 0.2
  },
  "result": {
    "best_trainer": "LightGbmBinary",
    "training_time_seconds": 287,
    "metrics": {
      "F1Score": 0.897,
      "Accuracy": 0.913,
      "AUC": 0.945
    }
  },
  "versions": {
    "mlnet": "5.0.0",
    "mloop": "0.2.0"
  }
}
```

### 7.3 User Configuration File

#### mloop.yaml (Multi-Model Format)
```yaml
# MLoop Project Configuration (v0.2.0+)
project: customer-analytics

# Model definitions
models:
  default:
    task: binary-classification
    label: Churn
    description: Customer churn prediction model
    training:
      time_limit_seconds: 300
      metric: F1Score
      test_split: 0.2

  revenue:
    task: regression
    label: Revenue
    description: Revenue prediction model
    training:
      time_limit_seconds: 600
      metric: RSquared

# Shared data paths
data:
  train: datasets/train.csv
  test: datasets/test.csv
```

---

## 8. Core Workflows

### 8.1 Project Initialization Workflow

```
User: mloop init my-project --task binary-classification
  │
  ├─> Process Start
  │
  ├─> CLI: Parse and validate command
  │
  ├─> Core: InitCommand.ExecuteAsync()
  │   │
  │   ├─> Create project directory structure
  │   ├─> Initialize .mloop/ directory
  │   ├─> Generate user files (mloop.yaml, .gitignore, README.md)
  │   └─> Save initial config
  │
  ├─> CLI: Display success message
  │
  └─> Process Exit (return 0)
```

### 8.2 Training Workflow

```
User: mloop train data.csv --label target --time 600
  │
  ├─> Process Start
  │
  ├─> CLI: Parse command and validate inputs
  │
  ├─> Core: TrainCommand.ExecuteAsync()
  │   │
  │   ├─> ProjectDiscovery: Find project root (.mloop/)
  │   │
  │   ├─> ExperimentStore: Generate new experiment ID
  │   │   └─> Atomic update: experiment-index.json
  │   │
  │   ├─> TrainingEngine: Execute AutoML
  │   │   ├─> Load and split data
  │   │   ├─> Run AutoML trials
  │   │   ├─> Stream progress to CLI (real-time)
  │   │   └─> Select best model
  │   │
  │   └─> ExperimentStore: Save results
  │       ├─> Create: experiments/exp-XXX/
  │       ├─> Save: model.zip
  │       ├─> Save: metadata.json
  │       ├─> Save: metrics.json
  │       └─> Save: config.json
  │
  ├─> CLI: Format and display results
  │
  └─> Process Exit (return 0)

# Process exited, all state in filesystem
# Next command starts fresh process
```

### 8.3 Prediction Workflow

```
User: mloop predict models/production/model.zip data.csv
  │
  ├─> Process Start
  │
  ├─> CLI: Parse command
  │
  ├─> Core: PredictCommand.ExecuteAsync()
  │   │
  │   └─> PredictionEngine:
  │       ├─> Load model.zip
  │       ├─> Parse input data (CSV/JSON)
  │       ├─> Execute batch predictions
  │       └─> Generate results
  │
  ├─> CLI: Output results (console or file)
  │
  └─> Process Exit (return 0)
```

### 8.4 Concurrent Execution Example

```
Terminal 1                     Terminal 2
─────────────────────────────  ─────────────────────────────
$ mloop train data.csv
  Process 1234 starts
  └─> exp-001/
      [Training 40%...]
                               $ mloop train data.csv --metric f1
                                 Process 5678 starts
                                 └─> exp-002/
                                     [Training 20%...]
      [Training 80%...]
                                     [Training 60%...]
  ✅ Complete, exit
  Process 1234 exits
                                     [Training 90%...]
                               ✅ Complete, exit
                               Process 5678 exits

Both experiments saved independently
No conflicts, no shared state
```

---

## 9. Configuration Management

### 9.1 Configuration Hierarchy

MLoop merges configuration from multiple sources with the following priority:

1. **CLI Arguments** (Highest Priority)
   - `--label`, `--time`, `--metric`, etc.

2. **mloop.yaml** (Project Level)
   - Project-specific defaults

3. **.mloop/config.json** (Created at init)
   - Initial project settings

4. **Hard-coded Defaults** (Lowest Priority)
   - Built-in default values

**Example Merge:**
```
CLI:           --time 600
mloop.yaml:    time_limit_seconds: 300, metric: f1
config.json:   task: binary-classification
defaults:      test_split: 0.2

Result (single process):
  time_limit_seconds: 600     (from CLI)
  metric: f1                  (from mloop.yaml)
  task: binary-classification (from config.json)
  test_split: 0.2            (from defaults)
```

---

## 10. Git Integration

### 10.1 Version Control Strategy

**Tracked (Committed to Git):**
- `.mloop/config.json` - Project configuration
- `.mloop/registry.json` - Model registry (metadata only)
- `experiments/*/metadata.json` - Experiment metadata
- `experiments/*/metrics.json` - Performance metrics
- `experiments/*/config.json` - Training configuration
- `models/*/metadata.json` - Promoted model metadata
- `mloop.yaml` - User configuration

**Ignored (Not committed):**
- `.mloop/cache/` - Temporary cache
- `experiments/*/model.zip` - Model binaries (large files)
- `experiments/*/training.log` - Detailed logs
- `models/*/model.zip` - Promoted model binaries
- `outputs/` - All output files

### 10.2 Collaboration Workflow

**Developer A (Run Experiment):**
```bash
$ mloop train data.csv --label price
$ git add experiments/exp-005/
$ git commit -m "Experiment 005: LightGBM, Acc=0.92"
$ git push
```

**Developer B (Review Experiment):**
```bash
$ git pull
$ mloop experiment show exp-005  # Read metadata.json, metrics.json
# Model binary can be retrained locally or shared separately
```

---

## 11. Technology Stack

### 11.1 Framework and Runtime

- **.NET 10.0**: Latest LTS version
- **C# 13**: Latest language features
- **Nullable Reference Types**: Enabled
- **Implicit Usings**: Enabled

### 11.2 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Microsoft.ML | 4.0.0 | ML.NET core framework |
| Microsoft.ML.AutoML | 0.21.1 | AutoML engine |
| System.CommandLine | 2.0.0-beta4 | CLI framework |
| YamlDotNet | 16.2.0 | YAML configuration |
| Spectre.Console | 0.49.1 | Rich CLI UI |

---

## 12. Testing Strategy

### 12.1 Unit Testing

**Scope:**
- Individual components in isolation
- Mock filesystem operations
- Interface-based dependency injection

### 12.2 Integration Testing

**Scope:**
- Multi-component interactions
- Real filesystem operations
- Concurrent execution scenarios

**Example: Concurrent Training Test**
```csharp
[Fact]
public async Task ConcurrentTraining_ShouldGenerateUniqueExperimentIds()
{
    using var tempDir = new TempDirectory();
    await InitProject(tempDir.Path);

    // Run two training jobs concurrently
    var task1 = RunTrainAsync(tempDir.Path, "data1.csv");
    var task2 = RunTrainAsync(tempDir.Path, "data2.csv");

    var results = await Task.WhenAll(task1, task2);

    // Both should succeed with different IDs
    Assert.NotEqual(results[0].ExperimentId, results[1].ExperimentId);
}
```

### 12.3 E2E Testing

**Scope:**
- Full CLI command execution
- Process lifecycle verification
- Multi-terminal concurrent execution

---

## 13. Long-Running Tasks

### 13.1 The Challenge

ML training can take minutes to hours. Users may need to:
- Close their terminal
- Disconnect from remote servers
- Run multiple training jobs
- Monitor progress remotely

### 13.2 Solution: Unix Tools (Not Daemon)

**Why not build a daemon?**
- ❌ Over-engineering for intermittent workloads
- ❌ Additional complexity (process management, ports, etc.)
- ❌ Standard Unix tools already solve this perfectly

**Recommended approaches:**

#### Option 1: nohup (Simple)

```bash
# Start training in background
$ nohup mloop train data.csv --label target --time 3600 > training.log 2>&1 &
[1] 12345

# Close terminal, training continues

# Check progress later
$ tail -f training.log

# Or check experiment files
$ cat experiments/exp-001/metadata.json
```

#### Option 2: screen (Interactive)

```bash
# Start screen session
$ screen -S ml-training

# Run training inside screen
$ mloop train data.csv --label target --time 3600
[Training...]

# Detach: Ctrl+A, then D
[detached from 12345.ml-training]

# Close terminal, training continues

# Reattach later
$ screen -r ml-training
[Training 80%...]
```

#### Option 3: tmux (Advanced)

```bash
# Create tmux session
$ tmux new -s ml-training

# Run training
$ mloop train data.csv --label target --time 3600

# Detach: Ctrl+B, then D

# Reattach later
$ tmux attach -t ml-training
```

### 13.3 Monitoring Progress

**Active monitoring (process running):**
```bash
# Real-time progress in terminal
$ mloop train data.csv --label target
🔍 AutoML Progress:
   [████████████░░░░░░░░] 60% | 180s elapsed
```

**Post-execution monitoring:**
```bash
# Check experiment metadata
$ mloop experiment show exp-001

# View training log
$ tail -f experiments/exp-001/training.log

# Check metrics
$ cat experiments/exp-001/metrics.json | jq
```

### 13.4 Future: Optional --detach Flag (Phase 2)

If users really want built-in backgrounding:

```bash
# MLoop handles nohup internally
$ mloop train data.csv --label target --detach
✅ Training started in background: exp-005
   PID: 12345
   Log: experiments/exp-005/training.log

# Check status
$ mloop experiment show exp-005
Status: in_progress (60% complete)

# Still just a process, not a daemon
# No mloop daemon start/stop/status
```

**Implementation:**
```csharp
if (detach)
{
    // Fork process with nohup
    var process = Process.Start(new ProcessStartInfo
    {
        FileName = "nohup",
        Arguments = $"mloop train {args}",
        RedirectStandardOutput = true,
        RedirectStandardError = true
    });

    Console.WriteLine($"Training started: PID {process.Id}");
    return 0; // Parent exits immediately
}
```

---

## 14. Extensibility System

### 14.1 Overview

**Design Philosophy**: Optional code-based customization while maintaining AutoML simplicity.

MLoop v0.2.0+ includes an **optional extensibility system** that allows users to enhance AutoML with domain knowledge through C# scripts, without sacrificing the simplicity of the base workflow.

**Key Principles:**
- **Completely Optional**: Extensions never required for basic operation
- **Zero-Overhead**: < 1ms performance impact when not used
- **Graceful Degradation**: Extension failures don't break AutoML
- **Type-Safe**: Full C# type system with IDE support
- **Convention-Based**: Automatic discovery via filesystem

**⚠️ PRIORITY REVISION (2025-11-09):**

Analysis of Datasets 004-006 revealed **Phase 0 (Preprocessing Scripts) is now P0 CRITICAL**, taking precedence over Phase 1 (Hooks & Metrics).

**Finding**: Current MLoop/FilePrepper handles only **50% of datasets (3/6)**. Critical gaps:
- Multi-file join (Dataset 004)
- Wide-to-Long transformation (Dataset 006)
- Feature engineering (Dataset 005)

**Revised Timeline:**

| Phase | Duration | Target | Priority |
|-------|----------|--------|----------|
| **Phase 0: Preprocessing** | Week 1-2 | Nov 11-22 | **P0 CRITICAL** |
| Phase 1: Hooks & Metrics | Week 3-4 | Nov 25-Dec 6 | P1 HIGH |

### 14.2 Extension Types

#### Phase 0: Preprocessing Scripts (P0 CRITICAL - NEW)

**Purpose**: Data transformation before AutoML training

**Critical Use Cases** (from Dataset Analysis):
- Multi-file operations (join, merge, concat)
- Wide-to-Long transformations (unpivot)
- Feature engineering (computed columns)
- Complex data cleaning beyond FilePrepper

**Interface:**
```csharp
public interface IPreprocessingScript
{
    /// <summary>
    /// Executes preprocessing logic.
    /// Scripts run sequentially: 01_*.cs → 02_*.cs → 03_*.cs
    /// </summary>
    /// <param name="context">Execution context with input/output paths</param>
    /// <returns>Path to output CSV (becomes next script's input)</returns>
    Task<string> ExecuteAsync(PreprocessContext context);
}
```

**Example - Multi-file Join** (Dataset 004):
```csharp
// .mloop/scripts/preprocess/01_join_files.cs
public class JoinMachineAndOrder : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        var machines = await ctx.Csv.ReadAsync("datasets/raw/machine_info.csv");
        var orders = await ctx.Csv.ReadAsync("datasets/raw/order_info.csv");

        var joined = from m in machines
                     join o in orders on m["item"] equals o["중산도면"]
                     select new Dictionary<string, string>
                     {
                         ["설비명"] = m["설비명"],
                         ["item"] = m["item"],
                         ["재고"] = o["재고"],
                         ["생산필요량"] = o["생산필요량"]
                     };

        return await ctx.Csv.WriteAsync(
            Path.Combine(ctx.OutputDirectory, "01_joined.csv"),
            joined.ToList());
    }
}
```

**Example - Wide-to-Long** (Dataset 006):
```csharp
// .mloop/scripts/preprocess/01_unpivot_shipments.cs
public class UnpivotShipments : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);
        var longData = new List<Dictionary<string, string>>();

        foreach (var row in data)
        {
            for (int i = 1; i <= 10; i++)
            {
                var dateCol = $"{i}차 출고날짜";
                var qtyCol = $"{i}차 출고량";

                if (!string.IsNullOrEmpty(row[dateCol]))
                {
                    longData.Add(new Dictionary<string, string>
                    {
                        ["작업지시번호"] = row["작업지시번호"],
                        ["출고차수"] = i.ToString(),
                        ["출고날짜"] = row[dateCol],
                        ["출고량"] = row[qtyCol]
                    });
                }
            }
        }

        return await ctx.Csv.WriteAsync(
            Path.Combine(ctx.OutputDirectory, "01_unpivoted.csv"),
            longData);
    }
}
```

**Execution Flow:**
```
User: mloop train datasets/raw.csv --label "quantity"
  ↓
1. CLI detects .mloop/scripts/preprocess/*.cs
  ↓
2. PreprocessingEngine executes sequentially:
   - 01_join.cs: raw.csv → .mloop/temp/01_joined.csv
   - 02_features.cs: 01_joined.csv → .mloop/temp/02_features.csv
   - 03_datetime.cs: 02_features.csv → datasets/train.csv
  ↓
3. TrainingEngine trains on final output: datasets/train.csv
```

**PreprocessContext:**
```csharp
public class PreprocessContext
{
    public string InputPath { get; set; }          // Input CSV path
    public string OutputDirectory { get; set; }     // Temp directory
    public string ProjectRoot { get; set; }         // Project root
    public CsvHelper Csv { get; set; }              // CSV helper
    public IFilePrepper FilePrepper { get; set; }   // FilePrepper integration
    public ILogger Logger { get; set; }             // Logger
}
```

**CLI Commands:**
```bash
# Auto-preprocessing (recommended)
mloop train datasets/raw.csv --label "quantity" --time 120
# → Auto-detects scripts → Runs preprocessing → Trains

# Manual preprocessing (debugging)
mloop preprocess --input datasets/raw.csv --output datasets/train.csv

# Validate scripts
mloop validate --scripts
```

#### Phase 1: Hooks (Lifecycle Extensions)

**Purpose**: Execute custom logic at specific pipeline points

```
mloop train data.csv
    ↓
[pre-train hook]  ← Data validation, preprocessing checks
    ↓
AutoML Training
    ↓
[post-train hook] ← Model validation, deployment, logging
    ↓
Save Results
```

**Hook Points:**
- `pre-train`: Before AutoML training (data validation)
- `post-train`: After AutoML training (model validation, deployment)
- `pre-predict`: Before batch prediction (input validation)
- `post-evaluate`: After model evaluation (reporting, analysis)

**Use Cases:**
- Data quality validation
- MLflow/W&B integration
- Model performance gates
- Automated deployment triggers

#### Custom Metrics (Business-Aligned Evaluation)

**Purpose**: Define business-specific optimization objectives for AutoML

**Standard Metrics** (Built-in):
```bash
mloop train data.csv --label target --metric accuracy
# Uses: Accuracy, F1, AUC, Precision, Recall
```

**Custom Business Metrics**:
```bash
mloop train data.csv --label target --metric profit-metric.cs
# AutoML optimizes for: Expected Profit, Churn Cost, ROI, etc.
```

### 14.3 Architecture Integration

#### Directory Structure

```
.mloop/
├── scripts/                     # Extension scripts (Phase 1+)
│   ├── hooks/                   # Lifecycle hooks
│   │   ├── pre-train.cs
│   │   ├── post-train.cs
│   │   ├── pre-predict.cs
│   │   └── post-evaluate.cs
│   └── metrics/                 # Custom metrics
│       ├── profit-metric.cs
│       └── churn-cost.cs
├── .cache/                      # Compiled DLLs (auto-generated)
│   └── scripts/
│       ├── hooks.pre-train.dll
│       └── metrics.profit-metric.dll
└── config.json
```

#### Component Architecture

```
┌─────────────────────────────────────────────────────┐
│  MLoop.Extensibility (New NuGet Package)            │
│  ├─ Interfaces (IMLoopHook, IMLoopMetric)          │
│  ├─ Context Classes (HookContext, MetricContext)   │
│  └─ Result Classes (HookResult)                    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  MLoop.Core (Enhanced)                              │
│  ├─ Scripting/                                      │
│  │  ├─ ScriptLoader.cs (Hybrid compilation)        │
│  │  ├─ ScriptDiscovery.cs (Auto-discovery)         │
│  │  └─ ScriptCompiler.cs (Roslyn wrapper)          │
│  └─ AutoML/                                         │
│     └─ TrainingEngine.cs (Hook integration)        │
└─────────────────────────────────────────────────────┘
```

### 14.4 Extension Discovery Flow

```
1. User runs: mloop train data.csv --label target
       ↓
2. Extension Check:
   - .mloop/scripts/ exists? → Yes/No
   - Overhead if No: < 1ms (directory check only)
       ↓
3. If Yes, Script Discovery:
   - Scan .mloop/scripts/hooks/*.cs
   - Scan .mloop/scripts/metrics/*.cs
       ↓
4. Hybrid Compilation:
   - Check .cache/*.dll (cached?)
   - If cached & up-to-date → Load DLL (fast: ~50ms)
   - If not → Compile .cs → Cache DLL (first time: ~500ms)
       ↓
5. Validation:
   - Implements required interface?
   - No compilation errors?
   - On failure → Warning + Continue with AutoML
       ↓
6. Execution:
   - Hook: Execute at lifecycle point
   - Metric: Pass to AutoML optimizer
       ↓
7. AutoML Training (always runs)
```

### 14.5 Hybrid Compilation Strategy

**Challenge**: Balance flexibility (runtime .cs loading) with performance (pre-compiled DLLs)

**Solution**: Hybrid approach combining Roslyn scripting with DLL caching

```csharp
// ScriptLoader implementation
public async Task<T?> LoadScriptAsync<T>(string scriptPath)
{
    var dllPath = GetCachedDllPath(scriptPath);

    // Fast path: Load cached DLL if up-to-date
    if (IsCacheValid(scriptPath, dllPath))
    {
        return LoadFromDll<T>(dllPath);  // ~50ms
    }

    // Slow path: Compile .cs → Cache DLL
    var assembly = await CompileScriptAsync(scriptPath);  // ~500ms
    await SaveAssemblyAsync(assembly, dllPath);

    return LoadFromDll<T>(dllPath);
}

private bool IsCacheValid(string scriptPath, string dllPath)
{
    return File.Exists(dllPath) &&
           File.GetLastWriteTime(dllPath) >= File.GetLastWriteTime(scriptPath);
}
```

**Benefits:**
- ✅ Development: Edit .cs files with full IDE support (IntelliSense, debugging)
- ✅ First Run: Automatic compilation and caching
- ✅ Subsequent Runs: Fast DLL loading
- ✅ Deployment: Pre-compiled DLLs can be included

**Performance:**
```
Extension Check (no scripts):  < 1ms
First Run (compile + cache):   ~500ms
Cached Runs (load DLL):        ~50ms
AutoML Training:               ~300s (unchanged)
```

### 14.6 Graceful Degradation

**Design Goal**: Extension failures never break AutoML

**Error Handling Strategy:**

```csharp
public async Task<IEnumerable<IMLoopHook>> DiscoverHooksAsync()
{
    var hooks = new List<IMLoopHook>();

    if (!Directory.Exists(".mloop/scripts/hooks"))
    {
        // No hooks directory → No error, empty list
        return hooks;
    }

    foreach (var scriptFile in Directory.GetFiles(scriptsDir, "*.cs"))
    {
        try
        {
            var hook = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptFile);
            if (hook != null)
                hooks.Add(hook);
        }
        catch (CompilationException ex)
        {
            _logger.Warning($"⚠️  Compilation failed: {scriptFile}");
            _logger.Warning(ex.Message);
            // Continue with other scripts
        }
        catch (Exception ex)
        {
            _logger.Error($"❌ Unexpected error: {ex.Message}");
            // Continue with other scripts
        }
    }

    return hooks;  // Return whatever loaded successfully
}
```

**User Experience:**
```bash
$ mloop train data.csv --label target

🔍 Discovering extensions...
   ⚠️  Compilation failed: pre-train.cs
       Line 15: Syntax error
   ✅ Loaded hook: post-train.cs (MLflow Logging)

⚠️  Warning: Some extensions failed to load
    Continuing with AutoML...

🚀 Training started (AutoML only)
✅ Training completed
```

### 14.7 Multi-Process Compatibility

**Extensions work seamlessly with multi-process model:**

```
Terminal 1                     Terminal 2
─────────────────────────────  ─────────────────────────────
$ mloop train data.csv         $ mloop train data.csv
  Process 1234 starts            Process 5678 starts

  Load extensions (in-process)   Load extensions (in-process)
  ├─ Compile/load hooks          ├─ Load from cache (shared .dll)
  └─ Execute hooks               └─ Execute hooks

  AutoML training                AutoML training
  [exp-001]                      [exp-002]

  Process 1234 exits             Process 5678 exits
```

**Key Points:**
- Each process loads extensions independently
- DLL cache is shared (filesystem-based)
- No inter-process communication needed
- Natural isolation via process boundaries

### 14.8 Example: Data Validation Hook

```csharp
// .mloop/scripts/hooks/pre-train.cs
using MLoop.Extensibility;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Quality Check";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var preview = ctx.DataView.Preview(maxRows: 1000);
        var rowCount = preview.RowView.Length;

        // Minimum row check
        if (rowCount < 100)
        {
            return HookResult.Abort(
                $"Insufficient data: {rowCount} < 100 rows");
        }

        // Class imbalance check
        var labelCol = ctx.Metadata["LabelColumn"] as string;
        var distribution = AnalyzeClassBalance(ctx.DataView, labelCol);

        if (distribution.ImbalanceRatio > 20)
        {
            ctx.Logger.Warning(
                $"⚠️  Severe class imbalance: {distribution.ImbalanceRatio:F1}:1");
        }

        ctx.Logger.Info($"✅ Validation passed: {rowCount} rows");
        return HookResult.Continue();
    }
}
```

**Usage:**
```bash
$ mloop train data.csv --label target

📊 Executing hook: Data Quality Check
   ✅ Validation passed: 1,234 rows

🚀 AutoML training...
```

### 14.9 Example: Custom Business Metric

```csharp
// .mloop/scripts/metrics/profit-metric.cs
using MLoop.Extensibility;

public class ProfitMetric : IMLoopMetric
{
    public string Name => "Expected Profit";
    public bool HigherIsBetter => true;

    private const double PROFIT_PER_TP = 100.0;
    private const double LOSS_PER_FP = -50.0;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification
            .Evaluate(ctx.Predictions);

        return (metrics.PositiveRecall * PROFIT_PER_TP) +
               (metrics.FalsePositiveRate * LOSS_PER_FP);
    }
}
```

**Usage:**
```bash
$ mloop train data.csv --label target --metric profit-metric.cs

🎯 Optimization metric: Expected Profit (higher is better)

⏱️  AutoML searching...
   Trial 1: LightGbm → $45.32
   Trial 2: FastTree → $48.91 ⭐
   Trial 3: SdcaLogistic → $43.17

✅ Best model: FastTree ($48.91 expected profit)
```

### 14.10 CLI Commands

```bash
# Create new extension
mloop new hook --name DataValidation --type pre-train
mloop new metric --name ProfitMetric

# Validate extension
mloop validate .mloop/scripts/hooks/pre-train.cs
# ✅ Compilation successful
# ✅ Implements IMLoopHook

# List extensions
mloop extensions list
# Hooks:
#   ✅ pre-train.cs (Data Validation)
#   ✅ post-train.cs (MLflow Logging)
# Metrics:
#   ✅ profit-metric.cs (Expected Profit)

# Clean cache
mloop extensions clean
# Removed 5 cached DLLs
```

### 14.11 Backward Compatibility

**Guarantee**: All existing workflows continue to work unchanged

```bash
# ✅ Still works perfectly (no extensions)
$ mloop train data.csv --label target

# ✅ Extensions auto-discovered if .mloop/scripts/ exists
$ mloop train data.csv --label target

# ✅ Force disable extensions
$ mloop train data.csv --label target --no-extensions
```

**Version Policy:**
- v0.1.x: Pure AutoML (no extensions)
- v0.2.x: Hooks & Metrics (opt-in, zero breaking changes)
- v0.3.x: Transforms & Pipelines (opt-in, compatible with v0.2.x)

### 14.12 Implementation Roadmap

**Timeline** (Revised 2025-11-09):

| Phase | Duration | Target | Priority | Status |
|-------|----------|--------|----------|--------|
| **Phase 0.1** | Week 1 | Nov 11-15 | **P0 CRITICAL** | 📋 Planned |
| **Phase 0.2** | Week 2 | Nov 18-22 | **P0 CRITICAL** | 📋 Planned |
| **Phase 1.1** | Week 3 | Nov 25-29 | P1 HIGH | 📋 Planned |
| **Phase 1.2** | Week 4 | Dec 2-6 | P1 HIGH | 📋 Planned |
| **Release** | - | Dec 6 | - | 🎯 Target |

**Phase 0: Preprocessing Scripts** (P0 CRITICAL - Weeks 1-2):
- **Week 1**: Core infrastructure (IPreprocessingScript, ScriptCompiler, PreprocessingEngine)
- **Week 2**: CLI integration (`mloop preprocess`, auto-run in `mloop train`)
- **Goal**: Achieve 100% dataset coverage (6/6 datasets)

**Phase 1: Hooks & Metrics** (P1 HIGH - Weeks 3-4):
- **Week 3**: Hooks infrastructure (IMLoopHook, HookContext, TrainingEngine integration)
- **Week 4**: Custom metrics (IMLoopMetric, AutoML integration)
- **Goal**: Enable business-aligned optimization

**Success Criteria**:
- [ ] Dataset coverage: 100% (6/6 datasets trainable)
- [ ] Backward compatibility: 100% (no breaking changes)
- [ ] Extension overhead: < 1ms when not used
- [ ] Test coverage: >90% for new code
- [ ] Documentation complete with examples

---

## 15. Future Extensibility

### 15.1 Why NOT Background Service (Phase 2)

**Original concern**: "Long training blocks CLI"

**Reality**:
- ✅ Unix tools (nohup, screen, tmux) solve this perfectly
- ✅ Users already familiar with these tools
- ✅ No need to reinvent process management
- ❌ Background daemon adds complexity
- ❌ MLoop workloads are intermittent, not continuous

**Decision**: Stick with multi-process model indefinitely

**If absolutely needed:** Add `--detach` flag (Phase 2), which internally uses `nohup`, not a daemon

### 15.2 Advanced Extensions (Phase 2)

**Potential additions beyond Hooks & Metrics:**
- Custom Transforms (feature engineering scripts)
- Full Pipelines (complete workflow control)
- Dependency management (NuGet references in scripts)

**Note**: These will build on Phase 1 infrastructure (ScriptLoader, discovery, etc.)

### 15.3 Plugin System (Future)

**Potential Plugin Types:**
- Custom Trainers
- Data Loaders
- Storage Providers
- Metric Calculators

**Each plugin runs in-process:**
```csharp
public interface IPlugin
{
    string Name { get; }
    string Version { get; }
    void Initialize(IServiceProvider services);
}

// Loaded per-process, no shared state
var plugins = PluginLoader.Load(".mloop/plugins/");
foreach (var plugin in plugins)
{
    plugin.Initialize(services);
}
```

### 15.4 Remote Storage Support (Future)

**Filesystem abstraction enables remote storage:**

```csharp
public interface IStorageProvider
{
    Task<Stream> ReadAsync(string path);
    Task WriteAsync(string path, Stream content);
}

// Each process connects independently
var storage = config.StorageProvider switch
{
    "local" => new LocalStorageProvider(),
    "azure-blob" => new AzureBlobStorageProvider(),
    _ => throw new NotSupportedException()
};
```

**Multi-process still works:**
- Each process instance connects to storage
- No shared in-memory state
- Filesystem-like operations over network

---

## 15. Performance and Constraints

### 15.1 Target Scale

**Suitable:**
- Dataset size: < 1GB
- Training time: < 1 hour
- Concurrent experiments: Unlimited (filesystem isolated)
- Concurrent processes: Limited by system resources

**Process overhead:**
- Startup time: < 100ms
- Memory per process: ~50-200MB (ML.NET models)
- Acceptable for intermittent workloads

### 15.2 Filesystem Considerations

**Multi-process file locking:**
```csharp
// Atomic ID generation with retry
public async Task<string> GenerateExperimentIdAsync()
{
    for (int retry = 0; retry < 3; retry++)
    {
        try
        {
            // OS-level file locking
            using var fs = new FileStream(
                indexPath,
                FileMode.Open,
                FileAccess.ReadWrite,
                FileShare.None); // Exclusive lock

            var index = await ReadIndexAsync(fs);
            index.NextId++;
            await WriteIndexAsync(fs, index);
            return $"exp-{index.NextId:D3}";
        }
        catch (IOException)
        {
            // Another process has lock, retry
            await Task.Delay(100);
        }
    }
    throw new ConcurrencyException();
}
```

---

## 16. Security Considerations

### 16.1 Process Isolation

**Advantages:**
- ✅ Each command runs in separate process
- ✅ No shared memory between executions
- ✅ OS-level security boundaries
- ✅ Failed process doesn't affect others

### 16.2 Data Security

- Sensitive data in `.gitignore`
- Model encryption (future, if needed)
- No network exposure (except `mloop serve`)

---

## 17. Conclusion

MLoop's **multi-process casual design** perfectly matches its usage pattern and philosophy:

### 17.1 Core Principles

1. **Simplicity**: Each command = Independent process
2. **Transparency**: All state in filesystem, not daemon memory
3. **Isolation**: Natural process and filesystem boundaries
4. **Practicality**: Unix tools handle long-running tasks
5. **No Over-Engineering**: No daemon for intermittent workloads

### 17.2 Key Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-Process** | ML workloads are intermittent (train → exit) |
| **No Daemon** | Unix tools (nohup, screen) solve long-running needs |
| **Filesystem State** | Enables natural isolation and Git integration |
| **In-Process Core** | Simpler than IPC, adequate for single-command execution |
| **Exception: serve** | Only API serving needs persistent process |

### 17.3 Future-Proof

- ✅ Plugin system: In-process loading
- ✅ Remote storage: Per-process connections
- ✅ Optional --detach: Internal nohup, not daemon
- ❌ Background service: Rejected as over-engineering

---

## Appendix A: Quick Reference

### A.1 Process Lifecycle

```
Command Invocation
    ↓
Process Start
    ↓
Load Configuration (from filesystem)
    ↓
Execute Core Logic
    ↓
Save Results (to filesystem)
    ↓
Process Exit
    ↓
No residual state in memory
```

### A.2 Key Commands (Phase 1)

| Command | Process Behavior |
|---------|------------------|
| `mloop init` | Start → Create structure → Exit |
| `mloop train` | Start → Train → Save → Exit |
| `mloop predict` | Start → Load → Predict → Exit |
| `mloop evaluate` | Start → Load → Evaluate → Exit |
| `mloop serve` | Start → Listen → (User Ctrl+C) → Exit |

### A.3 Concurrent Execution Patterns

```bash
# Same project, different experiments
Terminal 1: mloop train data.csv --time 300  → exp-001
Terminal 2: mloop train data.csv --time 600  → exp-002

# Different projects, any command
Terminal 1: cd project-A && mloop train ...
Terminal 2: cd project-B && mloop train ...

# All work perfectly, no conflicts
```

---

**Version**: 1.2.0
**Last Updated**: 2026-01-12
**Status**: Living Document
**Process Model**: Multi-Process Casual
**Multi-Model Support**: Yes (v0.2.0+)
**Project Count**: 6 (Core, CLI, API, Extensibility, DataStore, Ops)
