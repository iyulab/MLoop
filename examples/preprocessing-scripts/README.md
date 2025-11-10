# MLoop Preprocessing Scripts - Examples

Example preprocessing scripts demonstrating the three most common data transformation scenarios for ML datasets.

## 📋 Overview

These scripts demonstrate MLoop's Phase 0 preprocessing capabilities, which enable complex data transformations before AutoML training. They are based on real-world scenarios from ML-Resource datasets 004-006.

## 🎯 Use Cases

| Script | Pattern | Use Case | Real Dataset |
|--------|---------|----------|--------------|
| `01_datetime_features.cs` | DateTime Parsing & Feature Extraction | Parse Korean datetime formats and extract temporal features (Year, Month, Hour, DayOfWeek, Quarter, etc.) | Dataset 005 (열처리) |
| `02_unpivot_shipments.cs` | Wide-to-Long Transform (Unpivot) | Convert multiple date/quantity column pairs into rows | Dataset 006 (표면처리) |
| `03_feature_engineering.cs` | Sequential Feature Engineering | Compute derived features from existing columns | Any dataset with production/inventory data |

## 🔄 Sequential Execution Pattern

Preprocessing scripts execute in numeric order (01 → 02 → 03), with each script's output becoming the next script's input:

```
Raw Data → 01_datetime_features.cs → 02_unpivot_shipments.cs → 03_feature_engineering.cs → Final CSV → AutoML Training
```

### Metadata Propagation

Scripts can access execution metadata:

```csharp
var scriptSequence = ctx.GetMetadata<int>("ScriptSequence");  // 1, 2, 3, ...
var totalScripts = ctx.GetMetadata<int>("TotalScripts");      // Total count
var labelColumn = ctx.GetMetadata<string>("LabelColumn");     // If specified
```

## 📝 Script Structure

Every preprocessing script follows this pattern:

```csharp
using MLoop.Extensibility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class MyPreprocessingScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        // 1. Read input data
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);

        // 2. Transform data (LINQ, loops, custom logic)
        var transformed = data.Select(row =>
        {
            var newRow = new Dictionary<string, string>(row);
            // ... transformations ...
            return newRow;
        }).ToList();

        // 3. Save output
        var outputPath = Path.Combine(ctx.OutputDirectory, "01_output.csv");
        await ctx.Csv.WriteAsync(outputPath, transformed);

        // 4. Return output path for next script
        return outputPath;
    }
}
```

## 🚀 How to Use

### Option 1: Manual Execution (for testing)

```bash
# Copy scripts to your project
cp examples/preprocessing-scripts/*.cs my-project/.mloop/scripts/preprocess/

# Run preprocessing manually
cd my-project
mloop preprocess datasets/train.csv --output datasets/preprocessed.csv
```

### Option 2: Automatic Execution (during training)

```bash
# Preprocessing runs automatically before training
mloop train datasets/train.csv --label YourLabelColumn

# Output:
# 🔄 Running preprocessing scripts...
#   📝 [1/3] 01_datetime_features.cs
#     ✅ Output: 01_with_datetime_features.csv
#   📝 [2/3] 02_unpivot_shipments.cs
#     ✅ Output: 02_unpivoted_shipments.csv
#   📝 [3/3] 03_feature_engineering.cs
#     ✅ Output: 03_engineered_features.csv
# ✅ Preprocessing complete
# [Training begins with final preprocessed data]
```

## 📦 Available Context APIs

### PreprocessContext

| Property | Type | Description |
|----------|------|-------------|
| `InputPath` | string | Absolute path to input CSV file |
| `OutputDirectory` | string | Directory where output files should be written |
| `ProjectRoot` | string | Project root directory path |
| `Csv` | ICsvHelper | High-performance CSV reader/writer |
| `Logger` | ILogger | Progress logging (Info, Warning, Error, Debug) |
| `Metadata` | IReadOnlyDictionary | Execution metadata (ScriptSequence, LabelColumn, etc.) |

### ICsvHelper

```csharp
// Read CSV to list of dictionaries (column name → value)
var data = await ctx.Csv.ReadAsync("path/to/file.csv");

// Write list of dictionaries to CSV
await ctx.Csv.WriteAsync("path/to/output.csv", data);

// Custom delimiter and header control
await ctx.Csv.WriteAsync("path.tsv", data, delimiter: '\t', includeHeader: true);
```

### ILogger

```csharp
ctx.Logger.Info("✅ Processing complete");       // Normal info
ctx.Logger.Warning("⚠️ Missing values found");   // Warnings
ctx.Logger.Error("❌ Validation failed");        // Errors
ctx.Logger.Debug("🔍 Detailed debugging info");  // Debug (verbose)
```

## 🎓 Example Scenarios

### Scenario 1: DateTime Feature Extraction (Dataset 005)

**Raw Data:**
```csv
시간,품목,생산필요량,재고
2022.3.1 8:00,GEAR_21,100,100
2022.3.1 16:30,GEAR_42,400,250
```

**After `01_datetime_features.cs`:**
```csv
시간,품목,생산필요량,재고,Year,Month,Day,Hour,DayOfWeek,Quarter,IsWeekend
2022.3.1 8:00,GEAR_21,100,100,2022,3,1,8,2,1,0
2022.3.1 16:30,GEAR_42,400,250,2022,3,1,16,2,1,0
```

**Impact:** Temporal features enable AutoML to learn time-based patterns (seasonality, day-of-week effects, etc.)

### Scenario 2: Wide-to-Long Unpivot (Dataset 006)

**Raw Data (Wide Format):**
```csv
작업지시번호,제품코드,1차 출고날짜,1차 출고량,2차 출고날짜,2차 출고량,3차 출고날짜,3차 출고량
W001,M21,2021-01-25,2000,2021-01-26,1000,2021-01-19,9550
```

**After `02_unpivot_shipments.cs` (Long Format):**
```csv
작업지시번호,제품코드,출고순번,출고날짜,출고량
W001,M21,1,2021-01-25,2000
W001,M21,2,2021-01-26,1000
W001,M21,3,2021-01-19,9550
```

**Impact:** Enables AutoML to learn shipment patterns without wide feature explosion

### Scenario 3: Feature Engineering (Any Dataset)

**Input (from previous script):**
```csv
생산량,생산필요량,재고,수주량,Hour
80,100,150,200,9
```

**After `03_feature_engineering.cs`:**
```csv
생산량,생산필요량,재고,수주량,Hour,생산효율(%),생산미달,재고충분도(%),재고부족,작업교대조,피크시간
80,100,150,200,9,80.00,1,75.00,1,1,1
```

**Impact:** Domain-specific features improve model performance and interpretability

## ⚡ Performance Characteristics

| Aspect | Details |
|--------|---------|
| **Zero Overhead** | < 1ms if no scripts exist (directory check only) |
| **Graceful Degradation** | Training continues even if preprocessing fails (logged as error) |
| **Sequential Execution** | Scripts run in order (01→02→03), each using previous output |
| **Temporary Files** | Outputs saved to `.mloop/temp/preprocess/` (auto-cleaned) |
| **Hybrid Compilation** | C# scripts compiled on first run, cached as DLL for subsequent runs |

## 🔧 Advanced Patterns

### Multi-File Join (Dataset 004)

See `ML-Resource/004-생산계획 최적화/.mloop/scripts/preprocess/01_join_machine_order.cs` for:
- Reading multiple CSV files from project
- LINQ inner/left joins
- Handling missing columns with defaults

### Error Handling

```csharp
public async Task<string> ExecuteAsync(PreprocessContext ctx)
{
    try
    {
        // ... transformation logic ...
    }
    catch (Exception ex)
    {
        ctx.Logger.Error($"Preprocessing failed: {ex.Message}");
        throw new InvalidOperationException($"Script failed: {ex.Message}", ex);
    }
}
```

### Conditional Processing

```csharp
// Skip processing if already done
if (ctx.Metadata.ContainsKey("AlreadyProcessed"))
{
    ctx.Logger.Info("Skipping - already processed");
    return ctx.InputPath;  // Return input as-is
}

// Process differently based on label column
var labelColumn = ctx.GetMetadata<string>("LabelColumn");
if (labelColumn == "생산량")
{
    // Production-specific logic
}
```

## 📊 Testing Scripts

### Unit Testing Pattern

```csharp
// Create test context
var context = new PreprocessContext
{
    InputPath = "test-data/input.csv",
    OutputDirectory = "test-data/output",
    ProjectRoot = ".",
    Csv = new CsvHelperImpl(),
    Logger = new TestLogger()
};

// Execute script
var script = new DateTimeFeatureExtraction();
var outputPath = await script.ExecuteAsync(context);

// Validate output
var result = await context.Csv.ReadAsync(outputPath);
Assert.True(result.Count > 0);
Assert.True(result[0].ContainsKey("Year"));
```

## 📁 File Organization

```
my-project/
├── .mloop/
│   ├── scripts/
│   │   └── preprocess/
│   │       ├── 01_first_step.cs      # Runs first
│   │       ├── 02_second_step.cs     # Runs second
│   │       └── 03_final_step.cs      # Runs last
│   └── temp/
│       └── preprocess/                # Temporary outputs (auto-created)
│           ├── 01_first_output.csv
│           ├── 02_second_output.csv
│           └── 03_final_output.csv
└── datasets/
    ├── train.csv                      # Original data
    └── [preprocessed data used for training]
```

## 🎯 Best Practices

1. **Naming Convention**: Use `01_`, `02_`, `03_` prefixes for execution order
2. **Single Responsibility**: Each script should do one transformation type
3. **Descriptive Names**: `01_join_machines.cs` vs `01_script.cs`
4. **Logging**: Use `ctx.Logger` for progress feedback (users see this)
5. **Error Messages**: Provide clear error messages with context
6. **Null Safety**: Check for missing/empty values before processing
7. **Performance**: Use LINQ for readability, loops for complex logic
8. **Output Path**: Always use `Path.Combine(ctx.OutputDirectory, "filename.csv")`

## 🐛 Debugging Tips

```csharp
// Log intermediate results
ctx.Logger.Debug($"Loaded {data.Count} rows");
ctx.Logger.Debug($"First row keys: {string.Join(", ", data[0].Keys)}");

// Validate data shape
if (data.Count == 0)
{
    throw new InvalidOperationException("No data to process");
}

// Sample output for inspection
var sample = data.Take(5).ToList();
ctx.Logger.Info($"Sample rows: {sample.Count}");
foreach (var row in sample)
{
    ctx.Logger.Debug($"  {string.Join(", ", row.Values)}");
}
```

## 📚 Additional Resources

- **Architecture**: `docs/ARCHITECTURE.md` - Section 14.2 (Preprocessing Scripts)
- **Real Examples**: `ML-Resource/*/. mloop/scripts/preprocess/` - Production scripts
- **API Reference**: `src/MLoop.Extensibility/IPreprocessingScript.cs` - Interface definition
- **Testing**: Run `mloop preprocess` manually before `mloop train` to verify

## ✅ Validation Checklist

Before using preprocessing scripts in production:

- [ ] Scripts compile without errors (checked on first run)
- [ ] Input/output CSV files have correct structure
- [ ] Logging provides useful progress information
- [ ] Error handling prevents training from breaking
- [ ] Output file exists and is valid CSV
- [ ] Manual test with `mloop preprocess` succeeds
- [ ] Integration test with `mloop train` succeeds
- [ ] Performance acceptable (< 10s for typical datasets)

---

**Phase 0 Status**: ✅ Complete (Core + CLI Integration + Examples)
**Next**: E2E testing with real ML-Resource datasets 004-006
