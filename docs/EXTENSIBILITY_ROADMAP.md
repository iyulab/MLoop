# MLoop Extensibility Implementation Roadmap

**Version**: 0.2.0-alpha
**Status**: Implementation Plan
**Target**: Phase 1 (Hooks & Metrics)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1: Core Extensibility](#2-phase-1-core-extensibility)
3. [Implementation Tasks](#3-implementation-tasks)
4. [Technical Specifications](#4-technical-specifications)
5. [Testing Strategy](#5-testing-strategy)
6. [Success Criteria](#6-success-criteria)
7. [Future Phases](#7-future-phases)

---

## 1. Overview

### 1.1 Goals

**Primary Objectives:**
1. Enable optional code-based customization of AutoML pipeline
2. Maintain 100% backward compatibility (no breaking changes)
3. Zero-overhead when extensions not used (< 1ms)
4. Type-safe, IDE-supported extension development

**Non-Goals (Out of Scope):**
- Marketplace or plugin distribution (Phase 3+)
- Complex dependency management (Phase 2+)
- Custom transforms/pipelines (Phase 2)

### 1.2 Timeline

| Phase | Duration | Target Date | Deliverables |
|-------|----------|-------------|--------------|
| **Phase 1.1** | Week 1 | Week of Jan 15 | Infrastructure (NuGet, ScriptLoader) |
| **Phase 1.2** | Week 2 | Week of Jan 22 | Integration (CLI, AutoML) |
| **Phase 1.3** | Week 3 | Week of Jan 29 | Polish (Templates, Docs, Tests) |
| **Release** | - | Feb 1 | v0.2.0-alpha |

---

## 2. Phase 1: Core Extensibility

### 2.1 Scope

**Included:**
- ✅ Hook system (4 hook points: pre-train, post-train, pre-predict, post-evaluate)
- ✅ Custom Metrics (business-aligned optimization)
- ✅ Hybrid compilation (Roslyn + DLL caching)
- ✅ Automatic script discovery
- ✅ Graceful error handling
- ✅ CLI commands (new, validate, extensions list)
- ✅ Documentation and examples

**Excluded (Future Phases):**
- ❌ Custom Transforms (Phase 2)
- ❌ Custom Pipelines (Phase 2)
- ❌ Marketplace (Phase 3+)
- ❌ Dependency management beyond core ML.NET

### 2.2 Architecture Components

```
┌─────────────────────────────────────────────────────┐
│  MLoop.Extensibility (New NuGet Package)            │
│  ├─ Interfaces (IMLoopHook, IMLoopMetric)          │
│  ├─ Context Classes (HookContext, MetricContext)   │
│  └─ Result Classes (HookResult)                    │
└─────────────────────────────────────────────────────┘
                      ▲
                      │
┌─────────────────────┴───────────────────────────────┐
│  MLoop.Core (Enhanced)                              │
│  ├─ Scripting/                                      │
│  │  ├─ ScriptLoader.cs (Hybrid compilation)        │
│  │  ├─ ScriptDiscovery.cs (Auto-discovery)         │
│  │  └─ ScriptCompiler.cs (Roslyn wrapper)          │
│  └─ AutoML/                                         │
│     └─ TrainingEngine.cs (Hook integration)        │
└─────────────────────────────────────────────────────┘
                      ▲
                      │
┌─────────────────────┴───────────────────────────────┐
│  MLoop.CLI (Enhanced)                               │
│  ├─ Commands/                                       │
│  │  ├─ NewCommand.cs (Generate templates)          │
│  │  ├─ ValidateCommand.cs (Validate scripts)       │
│  │  └─ ExtensionsCommand.cs (Manage extensions)    │
│  └─ TrainCommand.cs (Extension loading)            │
└─────────────────────────────────────────────────────┘
```

---

## 3. Implementation Tasks

### 3.1 Week 1: Infrastructure (Jan 15-19)

#### Task 1.1: Create MLoop.Extensibility NuGet Package
**Duration**: 2 days
**Owner**: Core Team

**Deliverables:**
- [ ] `MLoop.Extensibility.csproj` project file
- [ ] `IMLoopHook` interface
- [ ] `IMLoopMetric` interface
- [ ] `HookContext` class
- [ ] `MetricContext` class
- [ ] `HookResult` class
- [ ] XML documentation for all public APIs
- [ ] Package metadata (README, license, icon)

**Implementation:**
```csharp
// src/MLoop.Extensibility/MLoop.Extensibility.csproj
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <GeneratePackageOnBuild>true</GeneratePackageOnBuild>
    <PackageId>MLoop.Extensibility</PackageId>
    <Version>1.0.0-alpha</Version>
    <Authors>MLoop Team</Authors>
    <Description>Extensibility API for MLoop AutoML platform</Description>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.ML" Version="4.0.0" />
  </ItemGroup>
</Project>
```

```csharp
// src/MLoop.Extensibility/IMLoopHook.cs
namespace MLoop.Extensibility;

/// <summary>
/// Defines a lifecycle hook for custom logic at specific pipeline points.
/// </summary>
public interface IMLoopHook
{
    /// <summary>
    /// Gets the display name of this hook.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Executes the hook logic.
    /// </summary>
    /// <param name="context">The execution context with data and metadata.</param>
    /// <returns>A result indicating whether to continue or abort training.</returns>
    Task<HookResult> ExecuteAsync(HookContext context);
}
```

**Success Criteria:**
- [ ] Package builds successfully
- [ ] All interfaces have XML documentation
- [ ] Package can be referenced in test project
- [ ] No external dependencies beyond Microsoft.ML

---

#### Task 1.2: Implement ScriptLoader (Hybrid Compilation)
**Duration**: 3 days
**Owner**: Core Team

**Deliverables:**
- [ ] `ScriptLoader.cs` with Roslyn compilation
- [ ] DLL caching mechanism
- [ ] File change detection
- [ ] Error handling and logging
- [ ] Unit tests for compilation

**Implementation:**
```csharp
// src/MLoop/Core/Scripting/ScriptLoader.cs
public class ScriptLoader : IScriptLoader
{
    private readonly ILogger _logger;
    private readonly ConcurrentDictionary<string, Assembly> _assemblyCache = new();

    public async Task<T?> LoadScriptAsync<T>(string scriptPath) where T : class
    {
        try
        {
            var dllPath = GetCachedDllPath(scriptPath);

            // Check cache validity
            if (IsCacheValid(scriptPath, dllPath))
            {
                _logger.Debug($"Loading from cache: {dllPath}");
                return LoadFromAssembly<T>(dllPath);
            }

            // Compile .cs → .dll
            _logger.Debug($"Compiling: {scriptPath}");
            var assembly = await CompileScriptAsync(scriptPath);

            // Save to cache
            await SaveAssemblyAsync(assembly, dllPath);
            _logger.Debug($"Cached to: {dllPath}");

            return LoadFromAssembly<T>(dllPath);
        }
        catch (CompilationException ex)
        {
            _logger.Error($"Compilation failed: {scriptPath}");
            _logger.Error(ex.Message);
            return null;  // Graceful degradation
        }
        catch (Exception ex)
        {
            _logger.Error($"Unexpected error loading script: {ex.Message}");
            return null;
        }
    }

    private bool IsCacheValid(string scriptPath, string dllPath)
    {
        if (!File.Exists(dllPath))
            return false;

        var scriptTime = File.GetLastWriteTimeUtc(scriptPath);
        var dllTime = File.GetLastWriteTimeUtc(dllPath);

        return dllTime >= scriptTime;
    }

    private string GetCachedDllPath(string scriptPath)
    {
        // .mloop/scripts/hooks/pre-train.cs
        // → .mloop/.cache/scripts/hooks.pre-train.dll

        var relativePath = Path.GetRelativePath(".mloop/scripts", scriptPath);
        var cacheKey = relativePath
            .Replace(Path.DirectorySeparatorChar, '.')
            .Replace(".cs", ".dll");

        return Path.Combine(".mloop/.cache/scripts", cacheKey);
    }

    private async Task<Assembly> CompileScriptAsync(string scriptPath)
    {
        var sourceCode = await File.ReadAllTextAsync(scriptPath);

        var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode, path: scriptPath);
        var references = GetMetadataReferences();

        var compilation = CSharpCompilation.Create(
            assemblyName: Path.GetFileNameWithoutExtension(scriptPath),
            syntaxTrees: new[] { syntaxTree },
            references: references,
            options: new CSharpCompilationOptions(
                OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: OptimizationLevel.Release,
                allowUnsafe: false
            )
        );

        using var ms = new MemoryStream();
        var result = compilation.Emit(ms);

        if (!result.Success)
        {
            var errors = result.Diagnostics
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                .Select(d => $"{d.Location.GetLineSpan().StartLinePosition}: {d.GetMessage()}");

            throw new CompilationException(
                $"Compilation failed:\n{string.Join("\n", errors)}");
        }

        ms.Seek(0, SeekOrigin.Begin);
        return Assembly.Load(ms.ToArray());
    }

    private MetadataReference[] GetMetadataReferences()
    {
        return new[]
        {
            MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(Console).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(MLContext).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(IDataView).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(IMLoopHook).Assembly.Location),
            // Add more as needed
        };
    }

    private T? LoadFromAssembly<T>(string dllPath) where T : class
    {
        var assembly = _assemblyCache.GetOrAdd(dllPath, path =>
        {
            var bytes = File.ReadAllBytes(path);
            return Assembly.Load(bytes);
        });

        var type = assembly.GetTypes()
            .FirstOrDefault(t => typeof(T).IsAssignableFrom(t) && !t.IsInterface);

        if (type == null)
        {
            _logger.Warning($"No type implementing {typeof(T).Name} found in {dllPath}");
            return null;
        }

        return Activator.CreateInstance(type) as T;
    }

    private async Task SaveAssemblyAsync(Assembly assembly, string dllPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(dllPath)!);

        // Assembly is already in memory, need to extract bytes
        // This is handled during compilation (emit to stream)
        // For now, simplified - in real implementation, save from MemoryStream
    }
}
```

**Success Criteria:**
- [ ] Compiles valid C# scripts
- [ ] Caches compiled DLLs
- [ ] Detects file changes and recompiles
- [ ] Handles compilation errors gracefully
- [ ] Unit tests: 90%+ coverage

---

#### Task 1.3: Implement ScriptDiscovery
**Duration**: 1 day
**Owner**: Core Team

**Deliverables:**
- [ ] `ScriptDiscovery.cs` with directory scanning
- [ ] Convention-based discovery (.mloop/scripts/hooks/*.cs)
- [ ] Filtering and validation
- [ ] Unit tests

**Implementation:**
```csharp
// src/MLoop/Core/Scripting/ScriptDiscovery.cs
public class ScriptDiscovery : IScriptDiscovery
{
    private readonly IScriptLoader _scriptLoader;
    private readonly ILogger _logger;

    public async Task<IEnumerable<IMLoopHook>> DiscoverHooksAsync()
    {
        var hooks = new List<IMLoopHook>();
        var scriptsDir = ".mloop/scripts/hooks";

        if (!Directory.Exists(scriptsDir))
        {
            _logger.Debug("No hooks directory found");
            return hooks;
        }

        var scriptFiles = Directory.GetFiles(scriptsDir, "*.cs");
        _logger.Info($"Found {scriptFiles.Length} hook script(s)");

        foreach (var scriptFile in scriptFiles)
        {
            try
            {
                var hook = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptFile);

                if (hook != null)
                {
                    hooks.Add(hook);
                    _logger.Info($"✅ Loaded hook: {hook.Name}");
                }
            }
            catch (Exception ex)
            {
                _logger.Warning($"⚠️  Failed to load {Path.GetFileName(scriptFile)}: {ex.Message}");
            }
        }

        return hooks;
    }

    public async Task<IEnumerable<IMLoopMetric>> DiscoverMetricsAsync()
    {
        var metrics = new List<IMLoopMetric>();
        var scriptsDir = ".mloop/scripts/metrics";

        if (!Directory.Exists(scriptsDir))
        {
            _logger.Debug("No metrics directory found");
            return metrics;
        }

        var scriptFiles = Directory.GetFiles(scriptsDir, "*.cs");
        _logger.Info($"Found {scriptFiles.Length} metric script(s)");

        foreach (var scriptFile in scriptFiles)
        {
            try
            {
                var metric = await _scriptLoader.LoadScriptAsync<IMLoopMetric>(scriptFile);

                if (metric != null)
                {
                    metrics.Add(metric);
                    _logger.Info($"✅ Loaded metric: {metric.Name}");
                }
            }
            catch (Exception ex)
            {
                _logger.Warning($"⚠️  Failed to load {Path.GetFileName(scriptFile)}: {ex.Message}");
            }
        }

        return metrics;
    }
}
```

**Success Criteria:**
- [ ] Discovers all .cs files in hooks/ directory
- [ ] Discovers all .cs files in metrics/ directory
- [ ] Returns empty list if directories don't exist (no error)
- [ ] Logs warnings for failed scripts but continues
- [ ] Unit tests with mock filesystem

---

### 3.2 Week 2: Integration (Jan 22-26)

#### Task 2.1: Integrate Hooks into TrainingEngine
**Duration**: 2 days
**Owner**: Core Team

**Deliverables:**
- [ ] Hook execution points in `TrainingEngine`
- [ ] HookContext population
- [ ] Abort handling (hook returns false)
- [ ] Integration tests

**Implementation:**
```csharp
// src/MLoop/Core/AutoML/TrainingEngine.cs (Enhanced)
public class TrainingEngine : ITrainingEngine
{
    private readonly IScriptDiscovery _scriptDiscovery;
    private IEnumerable<IMLoopHook>? _hooks;

    public TrainingEngine WithHooks(IEnumerable<IMLoopHook> hooks)
    {
        _hooks = hooks;
        return this;
    }

    public async Task<TrainingResult> TrainAsync(TrainingConfig config)
    {
        // 1. Pre-train hooks
        if (_hooks?.Any() == true)
        {
            var preTrainHooks = _hooks.Where(h => h.GetType().Name.Contains("PreTrain"));

            foreach (var hook in preTrainHooks)
            {
                _logger.Info($"📊 Executing hook: {hook.Name}");

                var context = new HookContext
                {
                    MLContext = _mlContext,
                    DataView = dataView,
                    Logger = _logger,
                    Metadata = new Dictionary<string, object>
                    {
                        ["LabelColumn"] = config.LabelColumn,
                        ["ExperimentId"] = experimentId
                    }
                };

                var result = await hook.ExecuteAsync(context);

                if (!result.ShouldContinue)
                {
                    _logger.Error($"❌ Training aborted by hook: {hook.Name}");
                    _logger.Error($"   Reason: {result.Message}");

                    return new TrainingResult
                    {
                        Success = false,
                        Message = $"Aborted by hook: {result.Message}"
                    };
                }
            }
        }

        // 2. Run AutoML (unchanged)
        var mlResult = await _autoML.RunAsync(config, progress, cancellationToken);

        // 3. Post-train hooks
        if (_hooks?.Any() == true)
        {
            var postTrainHooks = _hooks.Where(h => h.GetType().Name.Contains("PostTrain"));

            foreach (var hook in postTrainHooks)
            {
                var context = new HookContext
                {
                    MLContext = _mlContext,
                    DataView = predictions,
                    Logger = _logger,
                    Metadata = new Dictionary<string, object>
                    {
                        ["ExperimentId"] = experimentId,
                        ["Metrics"] = mlResult.Metrics,
                        ["BestTrainer"] = mlResult.BestTrainer,
                        ["ModelPath"] = modelPath
                    }
                };

                await hook.ExecuteAsync(context);  // Post-train hooks don't abort
            }
        }

        return new TrainingResult { Success = true, ... };
    }
}
```

**Success Criteria:**
- [ ] Pre-train hooks execute before AutoML
- [ ] Post-train hooks execute after AutoML
- [ ] Hook abort stops training with clear message
- [ ] Hook exceptions don't crash training (graceful degradation)
- [ ] Integration tests verify hook execution order

---

#### Task 2.2: Integrate Custom Metrics into AutoML
**Duration**: 2 days
**Owner**: Core Team

**Deliverables:**
- [ ] Custom metric wrapper for AutoML
- [ ] Metric optimization in AutoMLExperiment
- [ ] Metric context population
- [ ] Integration tests

**Implementation:**
```csharp
// src/MLoop/Core/AutoML/AutoMLRunner.cs (Enhanced)
public class AutoMLRunner
{
    private IMLoopMetric? _customMetric;

    public AutoMLRunner WithCustomMetric(IMLoopMetric metric)
    {
        _customMetric = metric;
        return this;
    }

    public async Task<AutoMLResult> RunAsync(TrainingConfig config)
    {
        // Create AutoML experiment
        var experiment = _mlContext.Auto()
            .CreateBinaryClassificationExperiment(config.TimeLimit);

        // Use custom metric if provided
        if (_customMetric != null)
        {
            experiment.SetEvaluationMetric(
                WrapCustomMetric(_customMetric),
                _customMetric.HigherIsBetter
                    ? BinaryClassificationMetric.PositivePrecision
                    : BinaryClassificationMetric.Accuracy
            );
        }

        var result = experiment.Execute(trainData, validationData);

        return new AutoMLResult { ... };
    }

    private Func<BinaryClassificationMetrics, double> WrapCustomMetric(
        IMLoopMetric customMetric)
    {
        return (mlMetrics) =>
        {
            // Convert ML.NET metrics to custom metric
            var context = new MetricContext
            {
                MLContext = _mlContext,
                Predictions = predictions,
                LabelColumn = config.LabelColumn,
                ScoreColumn = "Score",
                Logger = _logger
            };

            return customMetric.CalculateAsync(context).Result;
        };
    }
}
```

**Success Criteria:**
- [ ] Custom metrics integrated with AutoML optimization
- [ ] AutoML selects best model based on custom metric
- [ ] Metric logging shows custom metric values
- [ ] Integration tests verify metric optimization

---

#### Task 2.3: Enhance CLI Commands
**Duration**: 1 day
**Owner**: Core Team

**Deliverables:**
- [ ] `mloop new hook` command
- [ ] `mloop new metric` command
- [ ] `mloop validate` command
- [ ] `mloop extensions list` command
- [ ] Template files for hooks/metrics

**Implementation:**
```csharp
// src/MLoop/Commands/NewCommand.cs
public class NewCommand : Command
{
    public NewCommand() : base("new", "Create new extension from template")
    {
        var hookCommand = new Command("hook", "Create new hook");
        var nameOption = new Option<string>("--name", "Hook name");
        var typeOption = new Option<string>("--type", "Hook type (pre-train, post-train, etc.)");

        hookCommand.AddOption(nameOption);
        hookCommand.AddOption(typeOption);
        hookCommand.SetHandler(CreateHookAsync, nameOption, typeOption);

        AddCommand(hookCommand);

        var metricCommand = new Command("metric", "Create new metric");
        metricCommand.AddOption(new Option<string>("--name", "Metric name"));
        metricCommand.SetHandler(CreateMetricAsync, ...);

        AddCommand(metricCommand);
    }

    private async Task<int> CreateHookAsync(string name, string type)
    {
        var template = GetHookTemplate(name, type);
        var filePath = $".mloop/scripts/hooks/{type}.cs";

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        await File.WriteAllTextAsync(filePath, template);

        Console.WriteLine($"✅ Created: {filePath}");
        return 0;
    }

    private string GetHookTemplate(string name, string type)
    {
        return $@"using MLoop.Extensibility;

public class {name}Hook : IMLoopHook
{{
    public string Name => ""{name}"";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        // TODO: Implement your {type} hook logic here

        ctx.Logger.Info(""Hook executed: {name}"");

        return HookResult.Continue();
    }}
}}
";
    }
}
```

**Success Criteria:**
- [ ] `mloop new hook` generates valid C# file
- [ ] `mloop new metric` generates valid C# file
- [ ] Generated files compile successfully
- [ ] Templates include helpful TODOs and comments

---

### 3.3 Week 3: Polish (Jan 29 - Feb 2)

#### Task 3.1: Documentation
**Duration**: 2 days
**Owner**: Documentation Team

**Deliverables:**
- [ ] EXTENSIBILITY.md (comprehensive guide)
- [ ] EXTENSIBILITY_ROADMAP.md (this document)
- [ ] API reference docs (XML → Markdown)
- [ ] Quick start tutorial
- [ ] Real-world examples (3-5 scenarios)

**Success Criteria:**
- [ ] New users can create first hook in < 10 minutes
- [ ] All public APIs documented with examples
- [ ] Troubleshooting guide included

---

#### Task 3.2: Testing
**Duration**: 2 days
**Owner**: QA Team

**Deliverables:**
- [ ] Unit tests for all new components (>90% coverage)
- [ ] Integration tests for extension loading
- [ ] E2E tests for CLI commands
- [ ] Performance benchmarks

**Test Cases:**
```csharp
// Unit Tests
- ScriptLoader: Compilation success/failure
- ScriptLoader: Caching mechanism
- ScriptDiscovery: Directory scanning
- HookContext: Metadata management

// Integration Tests
- Full training with hooks
- Full training with custom metrics
- Extension compilation errors (graceful degradation)
- Concurrent training with extensions

// E2E Tests
- `mloop new hook` → Edit → `mloop train`
- `mloop new metric` → Edit → `mloop train --metric`
- Extension validation workflow
```

**Success Criteria:**
- [ ] All tests passing
- [ ] Code coverage >90% for new code
- [ ] Performance benchmarks meet targets (< 1ms overhead)

---

#### Task 3.3: Example Extensions
**Duration**: 1 day
**Owner**: Developer Advocate

**Deliverables:**
- [ ] `examples/extensions/` directory
- [ ] DataValidationHook example
- [ ] MLflowLoggingHook example
- [ ] ProfitMetric example
- [ ] ChurnCostMetric example
- [ ] README with usage instructions

**Success Criteria:**
- [ ] All examples compile and run successfully
- [ ] Examples cover common use cases
- [ ] Well-documented with inline comments

---

## 4. Technical Specifications

### 4.1 NuGet Packages

**MLoop.Extensibility v1.0.0-alpha**
```xml
<PackageReference Include="Microsoft.ML" Version="4.0.0" />
```

**MLoop.Core (Enhanced)**
```xml
<PackageReference Include="Microsoft.CodeAnalysis.CSharp" Version="4.8.0" />
<PackageReference Include="Microsoft.CodeAnalysis.CSharp.Scripting" Version="4.8.0" />
```

### 4.2 File System Structure

```
.mloop/
├── scripts/
│   ├── hooks/
│   │   ├── pre-train.cs
│   │   ├── post-train.cs
│   │   ├── pre-predict.cs
│   │   └── post-evaluate.cs
│   └── metrics/
│       └── *.cs
├── .cache/
│   └── scripts/
│       ├── hooks.pre-train.dll
│       └── metrics.*.dll
└── config.json
```

### 4.3 Performance Targets

| Metric | Target | Actual (Measured) |
|--------|--------|-------------------|
| Extension check (no scripts) | < 1ms | TBD |
| Script compilation (first run) | < 500ms | TBD |
| DLL loading (cached) | < 50ms | TBD |
| Hook execution overhead | < 100ms | TBD |

---

## 5. Testing Strategy

### 5.1 Test Pyramid

```
      /\
     /E2E\       10%  - Full CLI workflows
    /──────\
   /Integration\    30%  - Extension loading, Hook execution
  /────────────\
 /  Unit Tests  \  60%  - ScriptLoader, Discovery, Compilation
/────────────────\
```

### 5.2 Coverage Targets

| Component | Coverage Target | Critical Paths |
|-----------|----------------|----------------|
| MLoop.Extensibility | 100% | All interfaces |
| ScriptLoader | 95% | Compilation, Caching |
| ScriptDiscovery | 90% | Directory scanning |
| Hook Integration | 90% | Execution, Abort handling |
| Metric Integration | 85% | AutoML optimization |

### 5.3 Test Environments

- **Local Development**: Visual Studio, dotnet test
- **CI/CD**: GitHub Actions
  - Ubuntu 22.04
  - Windows Server 2022
  - macOS 13
- **Performance**: BenchmarkDotNet on dedicated hardware

---

## 6. Success Criteria

### 6.1 Functional Requirements

- [ ] Users can create hooks without errors
- [ ] Hooks execute at correct lifecycle points
- [ ] Custom metrics optimize AutoML correctly
- [ ] Extension compilation errors don't break training
- [ ] All CLI commands work as documented

### 6.2 Non-Functional Requirements

- [ ] Extension check overhead < 1ms (no scripts)
- [ ] Compilation time < 500ms (first run)
- [ ] DLL loading < 50ms (cached)
- [ ] Zero breaking changes to existing workflows
- [ ] Documentation complete and accurate

### 6.3 Release Checklist

- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] All E2E tests passing
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Examples working and documented
- [ ] NuGet packages published
- [ ] Release notes prepared
- [ ] Migration guide for users (if needed)

---

## 7. Future Phases

### 7.1 Phase 2: Advanced Extensions (v0.3.0)

**Target**: Q2 2025
**Duration**: 4-6 weeks

**Features:**
- Custom Transforms (feature engineering)
- Custom Pipelines (full workflow control)
- Dependency management (NuGet references in scripts)
- Advanced CLI commands (run pipeline.cs)

### 7.2 Phase 3: Ecosystem (v1.0.0)

**Target**: Q3 2025
**Duration**: 8-12 weeks

**Features:**
- Extension marketplace/registry
- Versioning and compatibility
- Community contributions
- Official extension library

---

## Appendix A: Risk Mitigation

### A.1 Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Roslyn compilation slow | Medium | High | Hybrid strategy with DLL caching |
| Extension breaks training | Medium | High | Graceful degradation, try-catch |
| Complex API learning curve | High | Medium | Good docs, examples, templates |
| Performance regression | Low | High | Benchmarking, performance tests |
| Breaking changes | Low | High | Comprehensive testing, backward compat |

### A.2 Contingency Plans

**If compilation too slow:**
- Fallback: Pre-compiled assemblies only (lose flexibility)
- Optimization: Background compilation

**If adoption low:**
- Gather user feedback
- Improve documentation
- Add more examples

---

## Appendix B: Dependencies

### B.1 External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Microsoft.ML | 4.0.0 | ML.NET core |
| Microsoft.CodeAnalysis.CSharp | 4.8.0 | Roslyn compilation |
| System.CommandLine | 2.0.0-beta4 | CLI framework |

### B.2 Internal Dependencies

- MLoop.Core (base implementation)
- MLoop.CLI (command handlers)
- MLoop.Tests (testing infrastructure)

---

**Version**: 0.2.0-alpha
**Last Updated**: 2025-01-09
**Status**: Implementation Roadmap (Phase 1)
**Next Review**: Jan 22, 2025
