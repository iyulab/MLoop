# MLoop Preprocessing Pipeline

## Overview

MLoop provides a flexible preprocessing pipeline that executes C# scripts sequentially before model training. This enables complex data transformations while maintaining type safety and performance.

## How It Works

### Automatic Script Discovery

MLoop automatically discovers and executes preprocessing scripts in your project:

```
your-project/
└── .mloop/
    └── scripts/
        └── preprocess/
            ├── 01_join_data.cs      ← Runs first
            ├── 02_features.cs       ← Runs second
            └── 03_normalize.cs      ← Runs third
```

**Execution Rules:**
- Scripts execute in **alphabetical order** (use numeric prefixes: `01_`, `02_`, `03_`)
- Each script's **output** becomes the next script's **input**
- Final script's output is used for model training
- All scripts share the same `PreprocessContext`

### Sequential Chaining

```
Raw Data (train.csv)
    ↓
01_join_data.cs → 01_joined.csv
    ↓
02_features.cs → 02_with_features.csv
    ↓
03_normalize.cs → 03_normalized.csv
    ↓
Training (final output used)
```

## Script Template

Every preprocessing script implements `IPreprocessingScript`:

```csharp
using MLoop.Extensibility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Example preprocessing script: Feature engineering
/// </summary>
public class FeatureEngineering : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("=== Feature Engineering ===");

        // 1. Read input CSV
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);
        ctx.Logger.Info($"Loaded: {data.Count:N0} rows");

        // 2. Transform data
        var transformed = data.Select(row =>
        {
            var newRow = new Dictionary<string, string>(row);

            // Add new features
            newRow["feature_x"] = CalculateFeature(row);

            return newRow;
        }).ToList();

        // 3. Write output CSV
        var outputPath = Path.Combine(ctx.OutputDirectory, "02_features.csv");
        await ctx.Csv.WriteAsync(outputPath, transformed);

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;  // Important: return path for next script
    }

    private string CalculateFeature(Dictionary<string, string> row)
    {
        // Your feature calculation logic
        return "calculated_value";
    }
}
```

## Context API

The `PreprocessContext` provides all necessary tools:

### ctx.InputPath
```csharp
// For 01_*.cs: path to raw train.csv
// For 02_*.cs: path to 01_output.csv
// For 03_*.cs: path to 02_output.csv
var data = await ctx.Csv.ReadAsync(ctx.InputPath);
```

### ctx.OutputDirectory
```csharp
// Directory where preprocessed files are stored
var outputPath = Path.Combine(ctx.OutputDirectory, "01_result.csv");
```

### ctx.ProjectRoot
```csharp
// Access other project files
var lookupFile = Path.Combine(ctx.ProjectRoot, "Dataset/data/lookup.csv");
var lookup = await ctx.Csv.ReadAsync(lookupFile);
```

### ctx.Csv
```csharp
// Read CSV → List<Dictionary<string, string>>
var data = await ctx.Csv.ReadAsync(filePath);

// Write CSV
await ctx.Csv.WriteAsync(outputPath, data);

// Note: Comma-formatted numbers ("2,000") are automatically cleaned to "2000"
```

### ctx.Logger
```csharp
ctx.Logger.Info("Processing started");
ctx.Logger.Warning("Potential issue detected");
ctx.Logger.Error("Fatal error occurred");
```

### ctx.Metadata
```csharp
// Access label column from mloop.yaml
var labelCol = ctx.GetMetadata<string>("LabelColumn");

// Store data for next scripts
ctx.SetMetadata("RowCount", data.Count);
```

## Common Patterns

### 1. Multi-File Join

```csharp
public async Task<string> ExecuteAsync(PreprocessContext ctx)
{
    // Read multiple files
    var orders = await ctx.Csv.ReadAsync(Path.Combine(ctx.ProjectRoot, "data/orders.csv"));
    var customers = await ctx.Csv.ReadAsync(Path.Combine(ctx.ProjectRoot, "data/customers.csv"));

    // Join with LINQ
    var joined = from o in orders
                 join c in customers on o["customer_id"] equals c["id"]
                 select new Dictionary<string, string>
                 {
                     ["order_id"] = o["id"],
                     ["customer_name"] = c["name"],
                     ["amount"] = o["amount"]
                 };

    var outputPath = Path.Combine(ctx.OutputDirectory, "01_joined.csv");
    await ctx.Csv.WriteAsync(outputPath, joined.ToList());
    return outputPath;
}
```

### 2. DateTime Feature Extraction

```csharp
public async Task<string> ExecuteAsync(PreprocessContext ctx)
{
    var data = await ctx.Csv.ReadAsync(ctx.InputPath);

    var processed = data.Select(row =>
    {
        var newRow = new Dictionary<string, string>(row);

        // Parse datetime (handles Korean format "2022.3.1 8:00")
        if (row.ContainsKey("timestamp") && DateTime.TryParse(row["timestamp"], out var dt))
        {
            newRow["Year"] = dt.Year.ToString();
            newRow["Month"] = dt.Month.ToString();
            newRow["Day"] = dt.Day.ToString();
            newRow["Hour"] = dt.Hour.ToString();
            newRow["DayOfWeek"] = ((int)dt.DayOfWeek).ToString();
            newRow["IsWeekend"] = (dt.DayOfWeek == DayOfWeek.Saturday ||
                                  dt.DayOfWeek == DayOfWeek.Sunday) ? "1" : "0";
        }

        return newRow;
    }).ToList();

    var outputPath = Path.Combine(ctx.OutputDirectory, "02_datetime.csv");
    await ctx.Csv.WriteAsync(outputPath, processed);
    return outputPath;
}
```

### 3. Wide-to-Long Unpivot

```csharp
public async Task<string> ExecuteAsync(PreprocessContext ctx)
{
    var data = await ctx.Csv.ReadAsync(ctx.InputPath);
    var longData = new List<Dictionary<string, string>>();

    foreach (var row in data)
    {
        // Base columns (common to all rows)
        var baseColumns = new Dictionary<string, string>
        {
            ["id"] = row["id"],
            ["name"] = row["name"]
        };

        // Unpivot: month1, month2, ... → month_number, value
        for (int i = 1; i <= 12; i++)
        {
            var monthCol = $"month{i}";
            if (row.ContainsKey(monthCol) && !string.IsNullOrEmpty(row[monthCol]))
            {
                var longRow = new Dictionary<string, string>(baseColumns)
                {
                    ["month"] = i.ToString(),
                    ["value"] = row[monthCol]
                };
                longData.Add(longRow);
            }
        }
    }

    ctx.Logger.Info($"Unpivoted: {data.Count:N0} wide rows → {longData.Count:N0} long rows");

    var outputPath = Path.Combine(ctx.OutputDirectory, "01_unpivoted.csv");
    await ctx.Csv.WriteAsync(outputPath, longData);
    return outputPath;
}
```

### 4. Safe Column Access (with Better Error Messages)

```csharp
using MLoop.Extensibility;  // For DictionaryExtensions

public async Task<string> ExecuteAsync(PreprocessContext ctx)
{
    var data = await ctx.Csv.ReadAsync(ctx.InputPath);

    var processed = data.Select(row =>
    {
        // Use GetValueOrThrow for helpful error messages
        var value = row.GetValueOrThrow("column_name", ctx.Logger);
        // If "column_name" doesn't exist, throws KeyNotFoundException with:
        // "Column 'column_name' not found in CSV row.
        //  Available columns: col1, col2, col3
        //  Did you mean 'column_name2'?"

        return new Dictionary<string, string>
        {
            ["processed"] = value.ToUpper()
        };
    }).ToList();

    var outputPath = Path.Combine(ctx.OutputDirectory, "processed.csv");
    await ctx.Csv.WriteAsync(outputPath, processed);
    return outputPath;
}
```

## Best Practices

### 1. Use Numeric Prefixes

```
✅ Good:
01_join.cs
02_features.cs
03_normalize.cs

❌ Bad:
join.cs
features.cs
normalize.cs
```

### 2. Keep Scripts Focused

Each script should perform **one logical transformation**:
- `01_join_data.cs` - Only joining
- `02_add_features.cs` - Only feature engineering
- `03_normalize.cs` - Only normalization

### 3. Log Progress

```csharp
ctx.Logger.Info("=== Step: Data Cleaning ===");
ctx.Logger.Info($"Input: {data.Count:N0} rows");
// ... transformation ...
ctx.Logger.Info($"Output: {cleaned.Count:N0} rows ({removed:N0} removed)");
ctx.Logger.Info($"✅ Saved: {outputPath}");
```

### 4. Handle Missing Data

```csharp
// Safe access with default values
var value = row.ContainsKey("optional_col") ? row["optional_col"] : "default";

// Skip rows with missing critical data
if (string.IsNullOrEmpty(row["critical_col"]))
{
    ctx.Logger.Warning($"Skipping row: missing critical_col");
    continue;
}
```

### 5. Validate Output

```csharp
if (outputData.Count == 0)
{
    throw new InvalidOperationException("No data after transformation!");
}

if (!outputData[0].ContainsKey("required_column"))
{
    throw new InvalidOperationException("Missing required_column in output!");
}
```

## Automatic Features

### Comma-Formatted Numbers

MLoop **automatically cleans** Korean/international number formats:

```csv
# Input CSV:
생산량,단가,금액
"2,000","1,500","3,000,000"

# After ctx.Csv.ReadAsync():
data[0]["생산량"] == "2000"      // Comma removed
data[0]["단가"] == "1500"        // Comma removed
data[0]["금액"] == "3000000"     // Comma removed
```

**No manual cleaning needed:**
```csharp
// ❌ Old way (manual):
var value = row["amount"].Replace(",", "").Trim();

// ✅ New way (automatic):
var value = row["amount"];  // Already cleaned!
```

## Debugging

### View Intermediate Outputs

Preprocessed files are saved in `.mloop/temp/preprocess/`:

```bash
.mloop/temp/preprocess/
├── 01_joined.csv          # Output from script 1
├── 02_features.csv        # Output from script 2
└── 03_normalized.csv      # Final output (used for training)
```

### Test Scripts Independently

```bash
# Run only preprocessing (skip training)
mloop preprocess datasets/train.csv
```

### Common Errors

**Error: Column 'X' not found**
```
Solution: Check column names with:
var columns = data.FirstOrDefault()?.Keys.ToArray() ?? new string[0];
ctx.Logger.Info($"Available columns: {string.Join(", ", columns)}");
```

**Error: Output file not found**
```
Solution: Always return the output path:
return outputPath;  // ← Must return for next script
```

**Error: Schema mismatch**
```
Solution: Check numeric values are cleaned:
- Use ctx.Csv.ReadAsync() (auto-cleans commas)
- Verify with: mloop info processed.csv
```

## Integration with Training

```bash
# Preprocessing runs automatically before training
mloop train

# Process flow:
# 1. mloop train detects .mloop/scripts/preprocess/
# 2. Executes: 01_*.cs → 02_*.cs → 03_*.cs
# 3. Uses final output for training
# 4. Shows: "📊 Training Configuration" with preprocessed data info
```

## Advanced Topics

### Metadata Sharing

```csharp
// Script 01: Store metadata
ctx.SetMetadata("OriginalRowCount", data.Count);

// Script 02: Read metadata
var originalCount = ctx.GetMetadata<int>("OriginalRowCount");
ctx.Logger.Info($"Original: {originalCount}, Current: {data.Count}");
```

### Custom Validation

```csharp
// Validate label column exists (automatic in MLoop, shown for reference)
var labelCol = ctx.GetMetadata<string>("LabelColumn");
if (!data[0].ContainsKey(labelCol))
{
    throw new ArgumentException($"Label column '{labelCol}' not found!");
}
```

### Performance Tips

```csharp
// Use LINQ for efficient transformations
var processed = data
    .Where(row => !string.IsNullOrEmpty(row["key"]))
    .Select(row => TransformRow(row))
    .ToList();

// Avoid reading files multiple times
var lookup = await ctx.Csv.ReadAsync(lookupPath);
var lookupDict = lookup.ToDictionary(r => r["id"], r => r["value"]);
```

## Examples from Real Datasets

### Dataset 004: Production Planning

```csharp
// Multi-file join: 21,151 machines + 127 orders → 986 joined rows
var machines = await ctx.Csv.ReadAsync(Path.Combine(ctx.ProjectRoot, "Dataset/data/machine_info.csv"));
var orders = await ctx.Csv.ReadAsync(Path.Combine(ctx.ProjectRoot, "Dataset/data/order_info.csv"));

var joined = from m in machines
             join o in orders on m["item"] equals o["중산도면"]
             select new Dictionary<string, string>
             {
                 ["item"] = m["item"],
                 ["machine"] = m["machine"],
                 ["capacity"] = m["capacity"],
                 ["영업납기"] = o["영업납기"],
                 ["수량"] = o["수량"]
             };
```

### Dataset 005: Supply Chain Optimization

```csharp
// DateTime parsing: "2022.3.1 8:00" → Year, Month, Day, Hour features
// Result: R² = 1.0000 (perfect prediction!)
if (DateTime.TryParse(row["시간"].Replace(".", "-"), out var dt))
{
    newRow["Year"] = dt.Year.ToString();
    newRow["Month"] = dt.Month.ToString();
    newRow["Hour"] = dt.Hour.ToString();
    newRow["IsWeekend"] = (dt.DayOfWeek == DayOfWeek.Saturday ||
                          dt.DayOfWeek == DayOfWeek.Sunday) ? "1" : "0";
}
```

### Dataset 006: Surface Treatment Logistics

```csharp
// Unpivot: 177 wide rows → 655 long rows (10 shipment pairs)
for (int i = 1; i <= 10; i++)
{
    var dateCol = $"{i}차 출고날짜";
    var qtyCol = $"{i}차 출고량";

    if (!row.ContainsKey(dateCol) || string.IsNullOrEmpty(row[dateCol]))
        continue;

    var longRow = new Dictionary<string, string>(baseColumns)
    {
        ["출고차수"] = i.ToString(),
        ["출고날짜"] = row[dateCol],
        ["출고량"] = row[qtyCol]  // Auto-cleaned: "2,000" → "2000"
    };
    longData.Add(longRow);
}
```

## Summary

**Key Takeaways:**
1. Scripts execute **alphabetically** with **sequential chaining**
2. Use `ctx.Csv.ReadAsync()` for **automatic comma cleaning**
3. **One transformation per script** for maintainability
4. **Log progress** for debugging and transparency
5. **Always return output path** for next script
6. Use `.GetValueOrThrow()` for **better error messages**
7. Test with `mloop preprocess` before full training

**Next Steps:**
- See example scripts in `.mloop/scripts/preprocess/`
- Run `mloop preprocess datasets/train.csv` to test
- View outputs in `.mloop/temp/preprocess/`
- Check column types with `mloop info preprocessed.csv`
