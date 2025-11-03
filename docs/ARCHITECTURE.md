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
14. [Future Extensibility](#14-future-extensibility)

---

## 1. Overview

MLoop is a lightweight MLOps platform built on ML.NET, designed with a **filesystem-first** and **multi-process casual** approach that emphasizes simplicity, transparency, and Git compatibility.

### 1.1 Core Mission

**"Clean Data In, Trained Model Out - That's It."**

MLoop fills the gap left by Microsoft's discontinued ML.NET CLI, providing .NET developers with a modern, production-ready tool for the complete ML lifecycle.

### 1.2 Design Principles

- **Filesystem-First**: All state managed as files, perfect Git integration
- **Multi-Process Casual**: Each command runs independently, no daemon required
- **Zero Configuration**: Usable immediately with minimal setup
- **Layer Separation**: Clear separation between CLI, Core, and Storage
- **Lightweight**: Independent operation without complex dependencies
- **AutoML-Driven**: Automatic model selection over manual tuning

### 1.3 Target Use Cases

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

```
MLoop/
├── src/
│   ├── MLoop.sln                        # .NET 9 Solution
│   └── MLoop/                           # Main CLI project
│       ├── MLoop.csproj                 # Global tool configuration
│       ├── Program.cs                   # Entry point
│       │
│       ├── Commands/                    # CLI command handlers
│       │   ├── InitCommand.cs          # mloop init
│       │   ├── TrainCommand.cs         # mloop train
│       │   ├── PredictCommand.cs       # mloop predict
│       │   ├── EvaluateCommand.cs      # mloop evaluate
│       │   ├── ExperimentCommand.cs    # mloop experiment
│       │   ├── ModelCommand.cs         # mloop model
│       │   └── ServeCommand.cs         # mloop serve (Phase 2)
│       │
│       ├── Core/                        # Core business logic
│       │   ├── AutoML/                  # AutoML engine
│       │   │   ├── ITrainingEngine.cs
│       │   │   ├── TrainingEngine.cs
│       │   │   ├── AutoMLRunner.cs
│       │   │   ├── TrainingConfig.cs
│       │   │   └── TrainingResult.cs
│       │   │
│       │   ├── Data/                    # Data loaders
│       │   │   ├── IDataLoader.cs
│       │   │   ├── CsvDataLoader.cs
│       │   │   └── JsonDataLoader.cs
│       │   │
│       │   ├── Models/                  # Model management
│       │   │   ├── IPredictionEngine.cs
│       │   │   ├── PredictionEngine.cs
│       │   │   ├── ModelLoader.cs
│       │   │   └── ModelSaver.cs
│       │   │
│       │   └── Evaluation/              # Evaluation
│       │       ├── IEvaluator.cs
│       │       ├── ClassificationEvaluator.cs
│       │       └── RegressionEvaluator.cs
│       │
│       ├── Infrastructure/              # Infrastructure
│       │   ├── FileSystem/              # Filesystem operations
│       │   │   ├── IFileSystemManager.cs
│       │   │   ├── FileSystemManager.cs
│       │   │   ├── IProjectDiscovery.cs
│       │   │   ├── ProjectDiscovery.cs
│       │   │   ├── IExperimentStore.cs
│       │   │   ├── ExperimentStore.cs
│       │   │   ├── IModelRegistry.cs
│       │   │   └── ModelRegistry.cs
│       │   │
│       │   ├── Configuration/           # Config management
│       │   │   ├── MLoopConfig.cs
│       │   │   ├── ConfigLoader.cs
│       │   │   └── ConfigMerger.cs
│       │   │
│       │   └── Logging/                 # Logging and progress
│       │       ├── IProgressReporter.cs
│       │       └── SpectreProgressReporter.cs
│       │
│       └── Templates/                   # Project templates
│           ├── binary-classification.yaml
│           ├── multiclass-classification.yaml
│           └── regression.yaml
│
├── tests/
│   ├── MLoop.Tests/                     # Unit tests
│   │   ├── Commands/
│   │   ├── Core/
│   │   └── Infrastructure/
│   │
│   └── MLoop.IntegrationTests/          # Integration tests
│       └── EndToEndTests.cs
│
├── examples/                            # Example projects
│   ├── sentiment-analysis/
│   ├── iris-classification/
│   └── housing-prices/
│
├── docs/                                # Documentation
│   ├── ARCHITECTURE.md                  # This file
│   ├── getting-started.md
│   ├── cli-reference.md
│   └── long-running-tasks.md           # nohup, screen guide
│
├── Directory.Build.props                # Common build properties
├── .editorconfig                        # Code style
├── nuget.config                         # NuGet configuration
└── .gitignore                           # Git ignore rules
```

### 6.2 User Project Structure

When users run `mloop init my-project --task binary-classification`:

```
my-project/
├── .mloop/                              # Internal (Git ignored)
│   ├── config.json                      # Project settings
│   ├── registry.json                    # Model registry index
│   └── experiment-index.json            # Experiment index
│
├── mloop.yaml                           # User config (optional, Git)
├── .gitignore                           # MLoop gitignore
├── README.md                            # Project guide
│
├── data/                                # User data (Git)
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   └── predictions/                     # Prediction outputs
│
├── experiments/                         # Experiment results
│   ├── exp-001/
│   │   ├── model.zip                    # Trained model (ignored)
│   │   ├── metadata.json                # Experiment metadata (Git)
│   │   ├── metrics.json                 # Performance metrics (Git)
│   │   ├── config.json                  # Training config (Git)
│   │   └── training.log                 # Training log (ignored)
│   ├── exp-002/
│   └── exp-003/
│
└── models/                              # Promoted models
    ├── staging/
    │   ├── model.zip                    # (ignored)
    │   └── metadata.json                # (Git)
    └── production/
        ├── model.zip                    # (ignored)
        └── metadata.json                # (Git)
```

---

## 7. Data Models

### 7.1 Internal Management Files

#### .mloop/config.json
```json
{
  "project_name": "my-ml-project",
  "version": "0.1.0",
  "task": "binary-classification",
  "label_column": "Sentiment",
  "created_at": "2024-11-03T20:00:00Z",
  "mloop_version": "0.1.0-alpha"
}
```

#### .mloop/experiment-index.json
```json
{
  "next_id": 7,
  "experiments": [
    {
      "id": "exp-001",
      "timestamp": "2024-11-03T10:00:00Z",
      "status": "completed",
      "best_metric": 0.85
    },
    {
      "id": "exp-002",
      "timestamp": "2024-11-03T11:00:00Z",
      "status": "completed",
      "best_metric": 0.89
    }
  ]
}
```

#### .mloop/registry.json
```json
{
  "production": {
    "experiment_id": "exp-005",
    "promoted_at": "2024-11-03T21:00:00Z",
    "metrics": {
      "accuracy": 0.913,
      "f1_score": 0.897
    }
  },
  "staging": {
    "experiment_id": "exp-006",
    "promoted_at": "2024-11-03T22:00:00Z"
  }
}
```

### 7.2 Experiment Files

#### experiments/exp-XXX/metadata.json
```json
{
  "experiment_id": "exp-001",
  "timestamp": "2024-11-03T12:00:00Z",
  "status": "completed",
  "task": "binary-classification",
  "data": {
    "train_file": "data/processed/train.csv",
    "rows": 10000,
    "features": 15,
    "label": "Sentiment"
  },
  "config": {
    "time_limit_seconds": 300,
    "metric": "accuracy",
    "test_split": 0.2
  },
  "result": {
    "best_trainer": "LightGbmBinary",
    "training_time_seconds": 287
  },
  "versions": {
    "mlnet": "4.0.0",
    "mloop": "0.1.0-alpha"
  }
}
```

#### experiments/exp-XXX/metrics.json
```json
{
  "accuracy": 0.913,
  "f1_score": 0.897,
  "auc": 0.945,
  "precision": 0.901,
  "recall": 0.893
}
```

### 7.3 User Configuration File

#### mloop.yaml (optional)
```yaml
# MLoop Project Configuration
project_name: sentiment-analyzer
task: binary-classification
label_column: Sentiment

# Training settings (optional - defaults provided)
training:
  time_limit_seconds: 300
  metric: accuracy
  test_split: 0.2

# Data paths (optional)
data:
  train: data/processed/train.csv
  test: data/processed/test.csv

# Model output (optional)
model:
  output_dir: models/staging
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

- **.NET 9.0**: Latest LTS version
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

## 14. Future Extensibility

### 14.1 Why NOT Background Service (Phase 2)

**Original concern**: "Long training blocks CLI"

**Reality**:
- ✅ Unix tools (nohup, screen, tmux) solve this perfectly
- ✅ Users already familiar with these tools
- ✅ No need to reinvent process management
- ❌ Background daemon adds complexity
- ❌ MLoop workloads are intermittent, not continuous

**Decision**: Stick with multi-process model indefinitely

**If absolutely needed:** Add `--detach` flag (Phase 2), which internally uses `nohup`, not a daemon

### 14.2 Plugin System (Future)

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

### 14.3 Remote Storage Support (Future)

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

**Version**: 0.1.0-alpha
**Last Updated**: 2024-11-03
**Status**: Living Document
**Process Model**: Multi-Process Casual (Phase 1+)
