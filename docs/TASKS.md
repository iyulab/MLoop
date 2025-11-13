# MLoop Task Breakdown
**Derived from Philosophy**: Excellent MLOps with Minimum Cost

This document provides detailed, actionable tasks aligned with the MLoop roadmap and core mission.

---

## Current Sprint: Phase 0 - Data Preparation Excellence
**Duration**: 2 weeks (Nov 11-22, 2025)
**Goal**: Enable 100% dataset coverage with preprocessing scripts

---

### Week 1: Core Infrastructure (Nov 11-15)

#### Task 0.1: Design Preprocessing Interface
**Priority**: P0 CRITICAL
**Estimated Effort**: 4 hours
**Assignee**: TBD

**Description**:
Design the `IPreprocessingScript` interface and execution model for custom data preprocessing.

**Acceptance Criteria**:
- [ ] Interface defined with clear contract
- [ ] Sequential execution model documented (01_*.cs → 02_*.cs)
- [ ] PreprocessContext API designed with required functionality
- [ ] Error handling strategy defined

**Implementation Details**:
```csharp
namespace MLoop.Extensibility.Preprocessing;

public interface IPreprocessingScript
{
    /// <summary>
    /// Executes preprocessing logic on input CSV.
    /// </summary>
    /// <param name="context">Execution context with I/O and helpers</param>
    /// <returns>Path to output CSV file</returns>
    Task<string> ExecuteAsync(PreprocessContext context);
}

public class PreprocessContext
{
    // Input/Output
    public string InputPath { get; init; }
    public string OutputDirectory { get; init; }
    public string ProjectRoot { get; init; }

    // Helpers
    public ICsvHelper Csv { get; init; }
    public IFilePrepper FilePrepper { get; init; }
    public ILogger Logger { get; init; }

    // Metadata
    public int ScriptIndex { get; init; }
    public string ScriptName { get; init; }
}
```

**Files to Create**:
- `src/MLoop.Extensibility/Preprocessing/IPreprocessingScript.cs`
- `src/MLoop.Extensibility/Preprocessing/PreprocessContext.cs`
- `src/MLoop.Extensibility/Preprocessing/CsvHelper.cs`

**Dependencies**: None

**Tests**:
- Interface contract validation
- PreprocessContext property access

---

#### Task 0.2: Implement ScriptCompiler with Roslyn
**Priority**: P0 CRITICAL
**Estimated Effort**: 8 hours
**Assignee**: TBD

**Description**:
Implement Roslyn-based compiler for C# preprocessing scripts with DLL caching.

**Acceptance Criteria**:
- [ ] Compiles .cs files with full ML.NET and MLoop.Core references
- [ ] Caches compiled DLLs in `.mloop/.cache/`
- [ ] Cache invalidation on source file changes
- [ ] Cached DLL load time <50ms
- [ ] Graceful error handling with helpful messages
- [ ] Compilation errors show line numbers and messages

**Implementation Details**:
```csharp
namespace MLoop.Core.Scripting;

public class ScriptCompiler
{
    public async Task<T?> LoadScriptAsync<T>(string scriptPath)
        where T : class
    {
        var dllPath = GetCachedDllPath(scriptPath);

        // Fast path: Load cached DLL if up-to-date
        if (IsCacheValid(scriptPath, dllPath))
        {
            return LoadFromDll<T>(dllPath);
        }

        // Slow path: Compile and cache
        var assembly = await CompileScriptAsync(scriptPath);
        await SaveAssemblyAsync(assembly, dllPath);

        return LoadFromDll<T>(dllPath);
    }

    private async Task<Assembly> CompileScriptAsync(string scriptPath)
    {
        var sourceCode = await File.ReadAllTextAsync(scriptPath);

        var compilation = CSharpCompilation.Create(
            assemblyName: Path.GetFileNameWithoutExtension(scriptPath),
            syntaxTrees: new[] { CSharpSyntaxTree.ParseText(sourceCode) },
            references: GetRequiredReferences(),
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        using var ms = new MemoryStream();
        var result = compilation.Emit(ms);

        if (!result.Success)
        {
            throw new CompilationException(result.Diagnostics);
        }

        ms.Seek(0, SeekOrigin.Begin);
        return Assembly.Load(ms.ToArray());
    }

    private IEnumerable<MetadataReference> GetRequiredReferences()
    {
        return new[]
        {
            MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(MLContext).Assembly.Location),
            MetadataReference.CreateFromFile(typeof(IPreprocessingScript).Assembly.Location),
            // ... other required assemblies
        };
    }
}
```

**Files to Create**:
- `src/MLoop.Core/Scripting/ScriptCompiler.cs`
- `src/MLoop.Core/Scripting/CompilationException.cs`
- `src/MLoop.Core/Scripting/AssemblyCache.cs`

**Dependencies**:
- Microsoft.CodeAnalysis.CSharp (4.13.0) - already in project
- Task 0.1 (IPreprocessingScript interface)

**Tests**:
- Successful compilation and caching
- Cache hit performance (<50ms)
- Cache invalidation on file changes
- Compilation error handling with clear messages
- Interface implementation validation

---

#### Task 0.3: Build PreprocessingEngine
**Priority**: P0 CRITICAL
**Estimated Effort**: 6 hours
**Assignee**: TBD

**Description**:
Orchestrate preprocessing script discovery, ordering, and execution.

**Acceptance Criteria**:
- [ ] Discovers scripts in `.mloop/scripts/preprocess/`
- [ ] Executes in sequential order (01, 02, 03, ...)
- [ ] Manages temporary files in `.mloop/temp/`
- [ ] Chains outputs: Script N output → Script N+1 input
- [ ] Progress reporting for each script
- [ ] Cleanup of temporary files on success
- [ ] Error handling with script context

**Implementation Details**:
```csharp
namespace MLoop.Core.Preprocessing;

public class PreprocessingEngine
{
    private readonly ScriptCompiler _compiler;
    private readonly ILogger _logger;

    public async Task<string> ExecuteAsync(
        string inputPath,
        string projectRoot,
        IProgress<PreprocessingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var scriptsDir = Path.Combine(projectRoot, ".mloop/scripts/preprocess");

        if (!Directory.Exists(scriptsDir))
        {
            _logger.Info("No preprocessing scripts found");
            return inputPath; // Return input unchanged
        }

        var scripts = DiscoverScripts(scriptsDir);
        if (scripts.Count == 0)
        {
            return inputPath;
        }

        var tempDir = Path.Combine(projectRoot, ".mloop/temp");
        Directory.CreateDirectory(tempDir);

        var currentInput = inputPath;

        for (int i = 0; i < scripts.Count; i++)
        {
            var script = scripts[i];

            progress?.Report(new PreprocessingProgress
            {
                CurrentScript = i + 1,
                TotalScripts = scripts.Count,
                ScriptName = Path.GetFileName(script)
            });

            _logger.Info($"Executing: {Path.GetFileName(script)}");

            var scriptInstance = await _compiler.LoadScriptAsync<IPreprocessingScript>(script);

            if (scriptInstance == null)
            {
                throw new PreprocessingException($"Failed to load script: {script}");
            }

            var context = new PreprocessContext
            {
                InputPath = currentInput,
                OutputDirectory = tempDir,
                ProjectRoot = projectRoot,
                ScriptIndex = i,
                ScriptName = Path.GetFileName(script),
                Csv = new CsvHelper(),
                FilePrepper = _filePrepper,
                Logger = _logger
            };

            currentInput = await scriptInstance.ExecuteAsync(context);

            if (!File.Exists(currentInput))
            {
                throw new PreprocessingException(
                    $"Script {script} did not produce output file: {currentInput}");
            }
        }

        return currentInput;
    }

    private List<string> DiscoverScripts(string scriptsDir)
    {
        return Directory.GetFiles(scriptsDir, "*.cs")
            .OrderBy(f => Path.GetFileName(f)) // 01_*.cs, 02_*.cs, ...
            .ToList();
    }
}
```

**Files to Create**:
- `src/MLoop.Core/Preprocessing/PreprocessingEngine.cs`
- `src/MLoop.Core/Preprocessing/PreprocessingProgress.cs`
- `src/MLoop.Core/Preprocessing/PreprocessingException.cs`

**Dependencies**:
- Task 0.1 (IPreprocessingScript)
- Task 0.2 (ScriptCompiler)

**Tests**:
- Script discovery and ordering
- Sequential execution with chaining
- Temporary file management
- Error handling for missing output
- Progress reporting

---

#### Task 0.4: Unit Tests for Preprocessing System
**Priority**: P0 CRITICAL
**Estimated Effort**: 6 hours
**Assignee**: TBD

**Description**:
Comprehensive unit tests for preprocessing system (>90% coverage).

**Acceptance Criteria**:
- [ ] Test coverage >90% for all preprocessing components
- [ ] All edge cases covered
- [ ] Mock filesystem for fast tests
- [ ] Integration tests with real scripts
- [ ] Performance tests for caching

**Test Cases**:
1. **ScriptCompiler Tests**:
   - Successful compilation and type loading
   - Cache hit on unchanged file
   - Cache invalidation on file modification
   - Compilation error handling
   - Missing interface implementation error

2. **PreprocessingEngine Tests**:
   - No scripts directory (returns input unchanged)
   - Single script execution
   - Multiple scripts with chaining
   - Script execution failure handling
   - Missing output file error

3. **Integration Tests**:
   - Real script compilation and execution
   - Multi-file join example
   - Wide-to-long transformation example
   - Error propagation from script

**Files to Create**:
- `tests/MLoop.Core.Tests/Preprocessing/ScriptCompilerTests.cs`
- `tests/MLoop.Core.Tests/Preprocessing/PreprocessingEngineTests.cs`
- `tests/MLoop.Core.Tests/Preprocessing/IntegrationTests.cs`

**Dependencies**: Tasks 0.1, 0.2, 0.3

---

### Week 2: CLI Integration & Examples (Nov 18-22)

#### Task 0.5: CLI Command - mloop preprocess
**Priority**: P1 HIGH
**Estimated Effort**: 4 hours
**Assignee**: TBD

**Description**:
Add `mloop preprocess` command for manual preprocessing execution.

**Acceptance Criteria**:
- [ ] Command syntax: `mloop preprocess --input <csv> --output <csv>`
- [ ] Validation mode: `mloop preprocess --validate`
- [ ] Verbose logging option: `--verbose`
- [ ] Progress indication during execution
- [ ] Helpful error messages

**Implementation Details**:
```csharp
public class PreprocessCommand : Command
{
    public PreprocessCommand() : base("preprocess", "Execute preprocessing scripts")
    {
        var inputOption = new Option<string>(
            "--input",
            "Input CSV file path");
        var outputOption = new Option<string>(
            "--output",
            "Output CSV file path");
        var validateOption = new Option<bool>(
            "--validate",
            "Validate scripts without executing");

        AddOption(inputOption);
        AddOption(outputOption);
        AddOption(validateOption);

        this.SetHandler(ExecuteAsync, inputOption, outputOption, validateOption);
    }

    private async Task<int> ExecuteAsync(
        string inputPath,
        string outputPath,
        bool validate)
    {
        try
        {
            var projectRoot = ProjectDiscovery.FindRoot();
            var engine = new PreprocessingEngine(_compiler, _logger);

            if (validate)
            {
                return await ValidateScriptsAsync(projectRoot);
            }

            var progress = new Progress<PreprocessingProgress>(p =>
            {
                Console.WriteLine($"[{p.CurrentScript}/{p.TotalScripts}] {p.ScriptName}");
            });

            var result = await engine.ExecuteAsync(
                inputPath,
                projectRoot,
                progress);

            if (!string.IsNullOrEmpty(outputPath))
            {
                File.Copy(result, outputPath, overwrite: true);
                Console.WriteLine($"✅ Output saved: {outputPath}");
            }
            else
            {
                Console.WriteLine($"✅ Preprocessing complete: {result}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
            return 1;
        }
    }
}
```

**Files to Create**:
- `src/MLoop.CLI/Commands/PreprocessCommand.cs`

**Dependencies**: Task 0.3 (PreprocessingEngine)

**Tests**:
- Command execution with valid input
- Validation mode
- Error handling for missing files
- Output file creation

---

#### Task 0.6: Auto-preprocessing in mloop train
**Priority**: P0 CRITICAL
**Estimated Effort**: 4 hours
**Assignee**: TBD

**Description**:
Integrate preprocessing into `mloop train` command automatically.

**Acceptance Criteria**:
- [ ] Auto-detects `.mloop/scripts/preprocess/` directory
- [ ] Runs preprocessing before AutoML training
- [ ] Transparent to user (no new flags required)
- [ ] Option to disable: `--no-preprocess`
- [ ] Clear logging of preprocessing steps

**Implementation Details**:
```csharp
// In TrainCommand.cs
public async Task<int> ExecuteAsync(
    string dataFile,
    string label,
    int timeSeconds,
    bool noPreprocess)
{
    try
    {
        var projectRoot = ProjectDiscovery.FindRoot();

        // Auto-preprocessing if scripts exist
        if (!noPreprocess)
        {
            var preprocessingEngine = new PreprocessingEngine(_compiler, _logger);
            var scriptsExist = Directory.Exists(
                Path.Combine(projectRoot, ".mloop/scripts/preprocess"));

            if (scriptsExist)
            {
                Console.WriteLine("📊 Running preprocessing scripts...");

                var preprocessedData = await preprocessingEngine.ExecuteAsync(
                    dataFile,
                    projectRoot,
                    new Progress<PreprocessingProgress>(p =>
                    {
                        Console.WriteLine($"   [{p.CurrentScript}/{p.TotalScripts}] {p.ScriptName}");
                    }));

                Console.WriteLine($"✅ Preprocessing complete");
                dataFile = preprocessedData; // Use preprocessed data for training
            }
        }

        // Continue with normal training flow
        var trainingEngine = new TrainingEngine(/*...*/);
        var result = await trainingEngine.TrainAsync(/*...*/);

        // ... rest of training logic
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Error: {ex.Message}");
        return 1;
    }
}
```

**Files to Modify**:
- `src/MLoop.CLI/Commands/TrainCommand.cs`

**Dependencies**: Task 0.3 (PreprocessingEngine)

**Tests**:
- Training with preprocessing scripts
- Training without preprocessing scripts
- --no-preprocess flag disables preprocessing
- Preprocessing errors handled gracefully

---

#### Task 0.7: Example Preprocessing Scripts
**Priority**: P1 HIGH
**Estimated Effort**: 6 hours
**Assignee**: TBD

**Description**:
Create comprehensive examples for common preprocessing scenarios.

**Acceptance Criteria**:
- [ ] Multi-file join example (Dataset 004 pattern)
- [ ] Wide-to-Long unpivot example (Dataset 006 pattern)
- [ ] Feature engineering example (Dataset 005 pattern)
- [ ] DateTime extraction and encoding example
- [ ] All examples well-commented and tested
- [ ] README for each example

**Examples to Create**:

1. **Multi-File Join** (`examples/preprocessing/multi-file-join/`)
   ```csharp
   // 01_join_machine_orders.cs
   public class JoinMachineOrders : IPreprocessingScript
   {
       public async Task<string> ExecuteAsync(PreprocessContext ctx)
       {
           var machines = await ctx.Csv.ReadAsync("datasets/raw/machines.csv");
           var orders = await ctx.Csv.ReadAsync("datasets/raw/orders.csv");

           var joined = from m in machines
                        join o in orders on m["item_id"] equals o["item_id"]
                        select new Dictionary<string, string>
                        {
                            ["machine_name"] = m["machine_name"],
                            ["item_id"] = m["item_id"],
                            ["quantity"] = o["quantity"],
                            ["order_date"] = o["order_date"]
                        };

           return await ctx.Csv.WriteAsync(
               Path.Combine(ctx.OutputDirectory, "01_joined.csv"),
               joined.ToList());
       }
   }
   ```

2. **Wide-to-Long Unpivot** (`examples/preprocessing/wide-to-long/`)
   ```csharp
   // 01_unpivot_shipments.cs
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
                   var dateCol = $"shipment_{i}_date";
                   var qtyCol = $"shipment_{i}_qty";

                   if (!string.IsNullOrEmpty(row.GetValueOrDefault(dateCol)))
                   {
                       longData.Add(new Dictionary<string, string>
                       {
                           ["order_id"] = row["order_id"],
                           ["shipment_number"] = i.ToString(),
                           ["shipment_date"] = row[dateCol],
                           ["shipment_qty"] = row[qtyCol]
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

3. **Feature Engineering** (`examples/preprocessing/feature-engineering/`)
   ```csharp
   // 01_compute_features.cs
   public class ComputeFeatures : IPreprocessingScript
   {
       public async Task<string> ExecuteAsync(PreprocessContext ctx)
       {
           var data = await ctx.Csv.ReadAsync(ctx.InputPath);

           foreach (var row in data)
           {
               // Compute derived features
               var price = double.Parse(row["price"]);
               var cost = double.Parse(row["cost"]);
               var quantity = double.Parse(row["quantity"]);

               row["profit_margin"] = ((price - cost) / price * 100).ToString("F2");
               row["total_revenue"] = (price * quantity).ToString("F2");
               row["price_per_unit"] = (price / quantity).ToString("F2");
           }

           return await ctx.Csv.WriteAsync(
               Path.Combine(ctx.OutputDirectory, "01_features.csv"),
               data);
       }
   }
   ```

4. **DateTime Processing** (`examples/preprocessing/datetime/`)
   ```csharp
   // 01_extract_datetime_features.cs
   public class ExtractDateTimeFeatures : IPreprocessingScript
   {
       public async Task<string> ExecuteAsync(PreprocessContext ctx)
       {
           var data = await ctx.Csv.ReadAsync(ctx.InputPath);

           foreach (var row in data)
           {
               if (DateTime.TryParse(row["order_date"], out var date))
               {
                   row["order_year"] = date.Year.ToString();
                   row["order_month"] = date.Month.ToString();
                   row["order_day"] = date.Day.ToString();
                   row["order_day_of_week"] = ((int)date.DayOfWeek).ToString();
                   row["order_quarter"] = ((date.Month - 1) / 3 + 1).ToString();
                   row["order_is_weekend"] = (date.DayOfWeek == DayOfWeek.Saturday ||
                                              date.DayOfWeek == DayOfWeek.Sunday) ? "1" : "0";
               }
           }

           return await ctx.Csv.WriteAsync(
               Path.Combine(ctx.OutputDirectory, "01_datetime.csv"),
               data);
       }
   }
   ```

**Files to Create**:
- `examples/preprocessing/multi-file-join/01_join_machine_orders.cs`
- `examples/preprocessing/multi-file-join/README.md`
- `examples/preprocessing/multi-file-join/datasets/` (sample data)
- `examples/preprocessing/wide-to-long/01_unpivot_shipments.cs`
- `examples/preprocessing/wide-to-long/README.md`
- `examples/preprocessing/feature-engineering/01_compute_features.cs`
- `examples/preprocessing/datetime/01_extract_datetime_features.cs`

**Dependencies**: Tasks 0.1-0.6 (Full preprocessing system)

---

#### Task 0.8: Documentation - Preprocessing Guide
**Priority**: P1 HIGH
**Estimated Effort**: 4 hours
**Assignee**: TBD

**Description**:
Comprehensive user guide for preprocessing scripts.

**Acceptance Criteria**:
- [ ] "Getting Started" tutorial (15-minute completion)
- [ ] API reference for `IPreprocessingScript` and `PreprocessContext`
- [ ] Common patterns and recipes
- [ ] Troubleshooting guide
- [ ] Performance best practices

**Document Structure**:
```markdown
# MLoop Preprocessing Guide

## Overview
What preprocessing scripts are and when to use them

## Quick Start (15 minutes)
Step-by-step tutorial creating first script

## Writing Preprocessing Scripts
### Script Structure
### PreprocessContext API
### CSV Helper Methods
### Error Handling

## Common Patterns
### Multi-File Operations
- Joining datasets
- Concatenating files
- Merging on keys

### Data Transformations
- Wide-to-Long unpivot
- Long-to-Wide pivot
- Column renaming and selection

### Feature Engineering
- Computed columns
- DateTime extraction
- Categorical encoding
- Normalization/scaling

## Examples
Links to examples/ directory

## API Reference
Complete PreprocessContext documentation

## Troubleshooting
Common errors and solutions

## Performance Tips
Caching, batch operations, memory management
```

**Files to Create**:
- `docs/PREPROCESSING.md`

**Dependencies**: Task 0.7 (Examples for reference)

---

## Sprint Success Metrics

**Phase 0 Completion Criteria**:
- [ ] All 8 tasks complete and tested
- [ ] Unit test coverage >90%
- [ ] Integration tests passing with example scripts
- [ ] Documentation complete and reviewed
- [ ] Zero breaking changes to existing workflows
- [ ] Performance: <1ms overhead when no scripts present
- [ ] Dataset coverage: 100% (6/6 datasets trainable)

**Definition of Done**:
1. Code reviewed and approved
2. Tests passing (>90% coverage)
3. Documentation complete
4. CHANGELOG.md updated
5. Merged to main branch

---

## Next Sprint Preview: Phase 1 - Hooks & Metrics
**Timeline**: Weeks 3-4 (Nov 25 - Dec 6)

**Key Tasks**:
- Design `IMLoopHook` interface (pre-train, post-train hooks)
- Implement hook discovery and execution
- Example hooks (data validation, MLflow logging)
- Design `IMLoopMetric` for custom business metrics
- AutoML integration for metric optimization
- Documentation and examples

---

**Last Updated**: November 13, 2025
**Sprint**: Phase 0 Week 1-2
**Next Review**: November 15, 2025
