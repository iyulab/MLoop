# Data Tools Integration Guide

## Overview

MLoop integrates with powerful external data processing tools to create a complete ML data pipeline:

- **FileFlux**: PDF extraction and document processing
- **FilePrepper**: ML-focused CSV preprocessing and normalization
- **MLoop**: ML.NET training, prediction, and MLOps

**Complete Pipeline**:
```
PDF Documents → FileFlux (extract) → JSON/CSV → FilePrepper (preprocess) → MLoop (train/predict)
```

---

## 1. FileFlux CLI Integration

### 1.1 Overview

FileFlux is a CLI tool for extracting structured content from PDF documents. Essential for processing PDF-based datasets, documentation, and research materials.

**Capabilities**:
- Extract text content from PDFs
- Automatic chunking for large documents
- JSON output with metadata
- Statistics and metrics reporting

### 1.2 Installation

**NuGet Global Tool**:
```bash
# Install globally
dotnet tool install -g fileflux.cli

# Verify installation
fileflux --version
# Output: 0.3.3+789883179e410c6c3b30ea11b78e9a1bf63338aa

# Check if already installed
dotnet tool search fileflux
# Package ID: fileflux.cli
# Latest Version: 0.3.3
# Authors: iyulab
```

### 1.3 Usage

**Command Syntax**:
```bash
fileflux extract <input> [options]

Arguments:
  <input>  Input file path

Options:
  -o, --output <output>  Output file path (default: input.extracted.json)
  -f, --format <format>  Output format (json, jsonl, markdown) [default: json]
  -q, --quiet            Minimal output
  -?, -h, --help         Show help and usage information
```

**Basic Example**:
```bash
# Extract PDF to JSON (default format)
fileflux extract "document.pdf"

# Output
# ✓ Extracted 1 chunks
# ✓ Saved to: document.extracted.json
# ┌────────────────────┬────────┐
# │ Metric             │ Value  │
# ├────────────────────┼────────┤
# │ Total chunks       │ 1      │
# │ Total characters   │ 62,721 │
# │ Average chunk size │ 62,721 │
# └────────────────────┴────────┘
```

**Advanced Examples**:
```bash
# Specify custom output path
fileflux extract "document.pdf" -o "extracted_content.json"

# Output as JSONL (JSON Lines)
fileflux extract "document.pdf" -f jsonl

# Output as Markdown
fileflux extract "document.pdf" -f markdown

# Quiet mode (minimal output)
fileflux extract "document.pdf" -q
```

**Real-World Example (Korean Supply Chain Dataset)**:
```bash
PS D:\data\MLoop\ML-Resource\001-공급망 최적화> fileflux extract "Guidebook_공급망 최적화 AI 데이터셋.pdf"

FileFlux CLI - Extract
  Input:  .\Guidebook_공급망 최적화 AI 데이터셋.pdf
  Output: .\Guidebook_공급망 최적화 AI 데이터셋.extracted.json
  Format: json

✓ Extracted 1 chunks
✓ Saved to: .\Guidebook_공급망 최적화 AI 데이터셋.extracted.json

┌────────────────────┬────────┐
│ Metric             │ Value  │
├────────────────────┼────────┤
│ Total chunks       │ 1      │
│ Total characters   │ 62,721 │
│ Average chunk size │ 62,721 │
└────────────────────┴────────┘
```

### 1.4 Output Formats

**Supported Formats**:
- `json` (default): Single JSON file with chunks array
- `jsonl`: JSON Lines format (one JSON object per line)
- `markdown`: Markdown formatted text

### 1.5 JSON Output Structure

**JSON Structure**:
```json
{
  "chunks": [
    {
      "content": "Extracted text content from PDF...",
      "page": 1,
      "chunk_id": 0,
      "metadata": {
        "start_char": 0,
        "end_char": 62721
      }
    }
  ],
  "statistics": {
    "total_chunks": 1,
    "total_characters": 62721,
    "average_chunk_size": 62721,
    "source_file": "document.pdf"
  }
}
```

### 1.6 MLoop Integration Patterns

#### Pattern 1: Dataset Documentation Extraction

Extract ML dataset guidebooks and documentation before training:

```bash
# 1. Extract dataset guidebook
cd ML-Resource/001-공급망 최적화
fileflux extract "Guidebook_공급망 최적화 AI 데이터셋.pdf"

# 2. Review extracted documentation
cat "Guidebook_공급망 최적화 AI 데이터셋.extracted.json" | jq '.chunks[0].content'

# 3. Use insights to configure MLoop training
mloop init supply-chain-optimization --task regression
mloop train datasets/train.csv --label target --time 600
```

#### Pattern 2: Automated Documentation Processing Hook

**.mloop/scripts/hooks/DocumentationExtractor.cs**:
```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using MLoop.Extensibility;

/// <summary>
/// Extracts PDF documentation using FileFlux CLI before training.
/// Useful for processing dataset guidebooks and research papers.
/// </summary>
public class DocumentationExtractor : IMLoopHook
{
    public string Name => "FileFlux PDF Documentation Extractor";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            var docsDir = Path.Combine(ctx.ProjectRoot, "docs");
            if (!Directory.Exists(docsDir))
            {
                ctx.Logger.Info("No docs/ directory found, skipping PDF extraction");
                return HookResult.Continue();
            }

            var pdfFiles = Directory.GetFiles(docsDir, "*.pdf");
            if (pdfFiles.Length == 0)
            {
                ctx.Logger.Info("No PDF files found in docs/");
                return HookResult.Continue();
            }

            ctx.Logger.Info($"Found {pdfFiles.Length} PDF file(s) to extract");

            foreach (var pdfPath in pdfFiles)
            {
                var jsonPath = Path.ChangeExtension(pdfPath, ".extracted.json");

                // Skip if already extracted and up-to-date
                if (File.Exists(jsonPath) &&
                    File.GetLastWriteTime(jsonPath) >= File.GetLastWriteTime(pdfPath))
                {
                    ctx.Logger.Info($"Skipping {Path.GetFileName(pdfPath)} (already extracted)");
                    continue;
                }

                ctx.Logger.Info($"Extracting: {Path.GetFileName(pdfPath)}");

                // Execute FileFlux CLI
                var result = await RunFileFluxAsync(pdfPath);

                if (result.ExitCode == 0)
                {
                    ctx.Logger.Info($"✅ Extracted to: {Path.GetFileName(jsonPath)}");

                    // Parse statistics
                    var stats = ParseStatistics(jsonPath);
                    if (stats != null)
                    {
                        ctx.Logger.Info($"   Chunks: {stats.TotalChunks}, Characters: {stats.TotalCharacters:N0}");
                    }
                }
                else
                {
                    ctx.Logger.Warning($"⚠️  FileFlux extraction failed: {result.Error}");
                }
            }

            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Documentation extraction error: {ex.Message}");
            // Non-critical failure, continue with training
            return HookResult.Continue();
        }
    }

    private async Task<(int ExitCode, string Output, string Error)> RunFileFluxAsync(string pdfPath)
    {
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "fileflux",
                Arguments = $"extract \"{pdfPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };

        process.Start();

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        return (process.ExitCode, output, error);
    }

    private Statistics? ParseStatistics(string jsonPath)
    {
        try
        {
            var json = File.ReadAllText(jsonPath);
            var doc = JsonDocument.Parse(json);

            if (doc.RootElement.TryGetProperty("statistics", out var stats))
            {
                return new Statistics
                {
                    TotalChunks = stats.GetProperty("total_chunks").GetInt32(),
                    TotalCharacters = stats.GetProperty("total_characters").GetInt32(),
                    AverageChunkSize = stats.GetProperty("average_chunk_size").GetInt32()
                };
            }
        }
        catch { }

        return null;
    }

    private class Statistics
    {
        public int TotalChunks { get; set; }
        public int TotalCharacters { get; set; }
        public int AverageChunkSize { get; set; }
    }
}
```

**Usage**:
```bash
# Place PDF guidebooks in docs/
cp "Guidebook_공급망 최적화.pdf" docs/

# Train - hook automatically extracts PDFs
mloop train datasets/train.csv --label target

# Output:
# 🔧 Executing hook: FileFlux PDF Documentation Extractor
# ℹ️  Found 1 PDF file(s) to extract
# ℹ️  Extracting: Guidebook_공급망 최적화.pdf
# ✅ Extracted to: Guidebook_공급망 최적화.extracted.json
#    Chunks: 1, Characters: 62,721
```

#### Pattern 3: Research Paper Processing Pipeline

Complete workflow for processing ML research papers:

```bash
# 1. Organize research materials
mkdir -p ML-Resource/research-papers
cd ML-Resource/research-papers

# 2. Extract multiple papers
for pdf in *.pdf; do
    fileflux extract "$pdf"
done

# 3. Aggregate insights (custom script)
python aggregate_research.py *.extracted.json > research_summary.md

# 4. Use insights for MLoop configuration
mloop init experiment --task multiclass-classification
# Configure based on research insights
```

### 1.7 Environment & Availability

**Global Tool Installation**:
- **Package**: `fileflux.cli` (NuGet global tool)
- **Version**: 0.3.3
- **Author**: iyulab
- **Command**: `fileflux` (available globally after installation)

**Installation Verification**:
```bash
# Check if FileFlux is installed
dotnet tool list -g | grep fileflux

# Or use where/which command
where fileflux       # Windows
which fileflux       # Linux/macOS

# Verify version
fileflux --version
# Expected: 0.3.3+<commit-hash>

# Test extraction
fileflux extract sample.pdf
```

**Update to Latest Version**:
```bash
# Update to latest version
dotnet tool update -g fileflux.cli

# Uninstall if needed
dotnet tool uninstall -g fileflux.cli
```

---

## 2. FilePrepper Integration

### 2.1 Overview

FilePrepper provides high-performance ML-focused CSV preprocessing with a Pipeline API.

**Performance**: 67-90% reduction in file I/O operations through in-memory processing.

**Key Operations**:
- Normalization (scaling to [0,1] or [-1,1])
- Missing value handling (mean, median, mode, forward/backward fill)
- Row filtering and validation
- Type conversion
- Column operations (drop, rename)

### 2.2 Development Configuration

MLoop uses **conditional references** for FilePrepper:

```xml
<!-- Debug Build: Local project reference -->
<ItemGroup Condition="'$(Configuration)' == 'Debug'">
  <ProjectReference Include="..\..\..\FilePrepper\src\FilePrepper\FilePrepper.csproj" />
</ItemGroup>

<!-- Release Build: NuGet package reference -->
<ItemGroup Condition="'$(Configuration)' == 'Release'">
  <PackageReference Include="FilePrepper" Version="0.4.0" />
</ItemGroup>
```

**Build Commands**:
```bash
# Debug (uses local FilePrepper project for active development)
dotnet build --configuration Debug

# Release (uses stable NuGet package)
dotnet build --configuration Release
```

### 2.3 Pipeline API Basics

```csharp
using FilePrepper;

// Create preprocessing pipeline
var pipeline = new Pipeline()
    .Normalize(column: "Amount", min: 0, max: 1)
    .FillMissing(column: "Age", strategy: FillStrategy.Mean)
    .FilterRows(predicate: row => row["Balance"] > 0)
    .Convert(column: "Date", targetType: typeof(DateTime))
    .Save("preprocessed.csv");

// Execute pipeline
await pipeline.ExecuteAsync("raw_data.csv");
```

### 2.4 MLoop Hook Integration

**.mloop/scripts/hooks/DataPreprocessing.cs**:
```csharp
using System;
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility;
using FilePrepper;

public class DataPreprocessing : IMLoopHook
{
    public string Name => "FilePrepper Data Preprocessing";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            var dataPath = ctx.Metadata["DatasetPath"].ToString();
            var outputPath = Path.Combine(
                Path.GetDirectoryName(dataPath),
                $"preprocessed_{Path.GetFileName(dataPath)}"
            );

            ctx.Logger.Info("Preprocessing data with FilePrepper...");

            var pipeline = new Pipeline()
                // Normalize numerical features
                .Normalize("Feature1", min: 0, max: 1)
                .Normalize("Feature2", min: 0, max: 1)

                // Handle missing values
                .FillMissing("Feature3", FillStrategy.Mean)
                .FillMissing("Category", FillStrategy.Mode)

                // Remove invalid rows
                .FilterRows(row => !string.IsNullOrWhiteSpace(row["ID"]))

                .Save(outputPath);

            await pipeline.ExecuteAsync(dataPath);

            var rowCount = File.ReadLines(outputPath).Count() - 1;
            ctx.Logger.Info($"✅ Preprocessed {rowCount} rows");

            if (rowCount < 10)
            {
                return HookResult.Abort($"Insufficient data: {rowCount} rows");
            }

            ctx.Metadata["DatasetPath"] = outputPath;
            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Preprocessing failed: {ex.Message}");
            return HookResult.Abort(ex.Message);
        }
    }
}
```

**For complete FilePrepper documentation, see**: [`FILEPREPPER_INTEGRATION.md`](FILEPREPPER_INTEGRATION.md)

---

## 3. Complete Data Pipeline

### 3.1 Full Workflow: PDF → Preprocessed CSV → ML Model

**Scenario**: Process supply chain optimization dataset with PDF guidebook

```bash
# Step 1: Extract PDF documentation
cd ML-Resource/001-공급망 최적화
fileflux extract "Guidebook_공급망 최적화 AI 데이터셋.pdf"

# Step 2: Review extracted documentation
cat "Guidebook_공급망 최적화 AI 데이터셋.extracted.json" | jq '.chunks[0].content' | head -50

# Step 3: Initialize MLoop project
mloop init supply-chain --task regression

# Step 4: Create preprocessing hook (if raw CSV needs cleaning)
# .mloop/scripts/hooks/SupplyChainPreprocessing.cs
# (Use FilePrepper to normalize, fill missing values, etc.)

# Step 5: Train with automatic preprocessing
mloop train datasets/train.csv --label optimization_score --time 600

# Output:
# 🔧 Executing hook: FilePrepper Data Preprocessing
# ℹ️  Preprocessing data with FilePrepper...
# ✅ Preprocessed 8,547 rows
#
# 🚀 AutoML training...
# ✅ Training complete: exp-001 (Accuracy: 0.913)
```

### 3.2 Automated Pipeline Hook

Complete hook combining FileFlux and FilePrepper:

**.mloop/scripts/hooks/CompleteDataPipeline.cs**:
```csharp
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility;
using FilePrepper;

/// <summary>
/// Complete data pipeline: PDF extraction → CSV preprocessing → AutoML training.
/// </summary>
public class CompleteDataPipeline : IMLoopHook
{
    public string Name => "Complete Data Pipeline (FileFlux + FilePrepper)";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            // Phase 1: Extract PDF documentation with FileFlux
            await ExtractDocumentationAsync(ctx);

            // Phase 2: Preprocess CSV data with FilePrepper
            await PreprocessDataAsync(ctx);

            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Pipeline failed: {ex.Message}");
            return HookResult.Abort(ex.Message);
        }
    }

    private async Task ExtractDocumentationAsync(HookContext ctx)
    {
        var docsDir = Path.Combine(ctx.ProjectRoot, "docs");
        if (!Directory.Exists(docsDir))
        {
            ctx.Logger.Info("📄 No docs/ directory, skipping PDF extraction");
            return;
        }

        var pdfFiles = Directory.GetFiles(docsDir, "*.pdf");
        if (pdfFiles.Length == 0)
        {
            ctx.Logger.Info("📄 No PDF files found");
            return;
        }

        ctx.Logger.Info($"📄 Extracting {pdfFiles.Length} PDF file(s)...");

        foreach (var pdfPath in pdfFiles)
        {
            var jsonPath = Path.ChangeExtension(pdfPath, ".extracted.json");

            if (File.Exists(jsonPath) &&
                File.GetLastWriteTime(jsonPath) >= File.GetLastWriteTime(pdfPath))
            {
                continue; // Already extracted
            }

            var process = Process.Start(new ProcessStartInfo
            {
                FileName = "fileflux",
                Arguments = $"extract \"{pdfPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            });

            await process.WaitForExitAsync();

            if (process.ExitCode == 0)
            {
                ctx.Logger.Info($"   ✅ {Path.GetFileName(pdfPath)}");
            }
        }
    }

    private async Task PreprocessDataAsync(HookContext ctx)
    {
        var dataPath = ctx.Metadata["DatasetPath"].ToString();
        var outputPath = Path.Combine(
            Path.GetDirectoryName(dataPath),
            $"preprocessed_{Path.GetFileName(dataPath)}"
        );

        ctx.Logger.Info("🔧 Preprocessing data with FilePrepper...");

        var pipeline = new Pipeline()
            .Normalize("Amount", min: 0, max: 1)
            .Normalize("Balance", min: 0, max: 1)
            .FillMissing("Age", FillStrategy.Mean)
            .FillMissing("Category", FillStrategy.Mode)
            .FilterRows(row => !string.IsNullOrWhiteSpace(row["ID"]))
            .Save(outputPath);

        await pipeline.ExecuteAsync(dataPath);

        var rowCount = File.ReadLines(outputPath).Count() - 1;
        ctx.Logger.Info($"✅ Preprocessed {rowCount} rows");

        if (rowCount < 10)
        {
            throw new InvalidOperationException($"Insufficient data: {rowCount} rows");
        }

        ctx.Metadata["DatasetPath"] = outputPath;
    }
}
```

### 3.3 Project Structure for Complete Pipeline

```
supply-chain-optimization/
├── .mloop/
│   ├── scripts/
│   │   └── hooks/
│   │       └── CompleteDataPipeline.cs    # PDF + CSV pipeline
│   └── config.json
│
├── docs/                                    # PDF documentation
│   ├── Guidebook_공급망 최적화.pdf
│   └── Guidebook_공급망 최적화.extracted.json  # FileFlux output
│
├── datasets/
│   ├── raw_train.csv                       # Original data
│   ├── preprocessed_raw_train.csv          # FilePrepper output
│   └── test.csv
│
├── models/
│   ├── staging/
│   │   └── exp-001/
│   └── production/
│
└── mloop.yaml
```

---

## 4. Tool Comparison & Selection

### 4.1 When to Use Each Tool

| Tool | Purpose | Input | Output | Use When |
|------|---------|-------|--------|----------|
| **FileFlux** | PDF extraction | PDF files | JSON (structured text) | Processing documentation, research papers, PDF datasets |
| **FilePrepper** | CSV preprocessing | CSV files | CSV (cleaned) | Normalizing, filling missing values, filtering data |
| **MLoop** | ML training | CSV files | ML models | Training models, making predictions, MLOps |

### 4.2 Decision Tree

```
Input: PDF document?
├─ Yes → FileFlux extract → JSON
│   └─ Convert to CSV → Continue
└─ No → Input: CSV file?
    ├─ Yes → Needs preprocessing?
    │   ├─ Yes → FilePrepper pipeline → Clean CSV
    │   └─ No → Direct to MLoop
    └─ Direct to MLoop

MLoop train → AutoML → Model
```

### 4.3 Integration Strategies

**Strategy 1: Manual Pipeline**
```bash
# User executes each step manually
fileflux extract docs/guidebook.pdf
# Review extracted.json, configure preprocessing
fileprepper normalize data.csv --output preprocessed.csv
mloop train preprocessed.csv --label target
```

**Strategy 2: Hook-Based Automation**
```bash
# Hooks handle FileFlux + FilePrepper automatically
mloop train raw_data.csv --label target

# Hook executes:
# 1. FileFlux PDF extraction (if PDFs in docs/)
# 2. FilePrepper CSV preprocessing
# 3. AutoML training with clean data
```

**Strategy 3: External Scripts**
```bash
# Custom preprocessing script
python preprocess_pipeline.py \
    --pdf-docs docs/*.pdf \
    --csv-data datasets/raw.csv \
    --output datasets/preprocessed.csv

# Then train with MLoop
mloop train datasets/preprocessed.csv --label target
```

---

## 5. Best Practices

### 5.1 FileFlux Best Practices

1. **Extract Once**: Cache extracted JSON files, re-extract only when PDF changes
2. **Review Extracted Content**: Verify extraction quality before using in pipelines
3. **Organize by Project**: Store PDFs in project-specific `docs/` directories
4. **Version Control**: Commit extracted JSON for reproducibility (small files)

### 5.2 FilePrepper Best Practices

1. **Validate Output**: Always check row counts before/after preprocessing
2. **Incremental Pipelines**: Build pipelines step-by-step, test each operation
3. **Log Statistics**: Report preprocessing metrics (rows removed, missing values filled)
4. **Extend When Needed**: Add missing features to FilePrepper (active development model)

### 5.3 Combined Pipeline Best Practices

1. **Separate Concerns**: FileFlux for PDFs, FilePrepper for CSVs, MLoop for ML
2. **Cache Intermediate Results**: Save extracted JSON and preprocessed CSV
3. **Handle Failures Gracefully**: Continue training even if PDF extraction fails
4. **Document Assumptions**: Record preprocessing decisions in project README
5. **Test with Sample Data**: Validate pipeline with small datasets first

---

## 6. Troubleshooting

### 6.1 FileFlux Issues

**Problem**: `fileflux` command not found

**Solution**:
```bash
# Verify FileFlux is in PATH
where fileflux

# If not found, check installation or use absolute path
/path/to/fileflux extract document.pdf
```

**Problem**: Extraction produces empty/corrupted JSON

**Solution**:
- Verify PDF is not password-protected
- Check PDF is not corrupted
- Try extracting manually to verify content

### 6.2 FilePrepper Issues

**Problem**: FilePrepper project reference not found in Debug mode

**Solution**:
```bash
# Verify FilePrepper project exists
ls D:\data\FilePrepper\src\FilePrepper\FilePrepper.csproj

# Or use Release build (NuGet package)
dotnet build --configuration Release
```

**Problem**: Out of memory with large CSV files

**Solution**:
- Process in chunks (feature needed in FilePrepper)
- Use streaming/batch processing
- Increase system memory or use smaller datasets

### 6.3 Integration Issues

**Problem**: Hook fails to execute FileFlux CLI

**Solution**:
```csharp
// Check FileFlux is available before executing
var filefluxPath = FindExecutable("fileflux");
if (filefluxPath == null)
{
    ctx.Logger.Warning("FileFlux not found, skipping PDF extraction");
    return HookResult.Continue();
}
```

---

## 7. Performance Considerations

### 7.1 FileFlux Performance

| PDF Size | Extraction Time | JSON Size | Notes |
|----------|-----------------|-----------|-------|
| 1 MB     | ~1-2 sec       | ~500 KB   | Single document |
| 10 MB    | ~5-10 sec      | ~5 MB     | Multi-page report |
| 100 MB   | ~30-60 sec     | ~50 MB    | Large academic paper |

**Optimization**:
- Cache extracted JSON files
- Extract in parallel for multiple PDFs
- Use selective extraction for large documents

### 7.2 FilePrepper Performance

| Dataset Rows | Manual Processing | FilePrepper | Improvement |
|--------------|-------------------|-------------|-------------|
| 1,000        | 45ms             | 12ms        | 73% faster  |
| 10,000       | 380ms            | 95ms        | 75% faster  |
| 100,000      | 3,200ms          | 890ms       | 72% faster  |
| 1,000,000    | 28,500ms         | 8,100ms     | 72% faster  |

**Key Features**:
- In-memory processing (no intermediate I/O)
- Parallel column operations
- Single-pass execution

---

## 8. Related Documentation

- **FilePrepper Integration**: [`FILEPREPPER_INTEGRATION.md`](FILEPREPPER_INTEGRATION.md)
- **MLoop Extensibility**: [`EXTENSIBILITY.md`](EXTENSIBILITY.md)
- **Architecture**: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **CLI Reference**: [`CLI.md`](CLI.md)

---

## 9. Summary

**Complete ML Data Pipeline**:

1. **FileFlux**: PDF → JSON extraction for documentation and research
2. **FilePrepper**: CSV → Preprocessed CSV for ML-ready data
3. **MLoop**: CSV → Trained models with AutoML

**Key Benefits**:
- ✅ End-to-end data processing automation
- ✅ High-performance preprocessing (67-90% I/O reduction)
- ✅ Hook-based integration for reusable workflows
- ✅ Filesystem-first approach (Git-friendly)
- ✅ Active development model (extend tools as needed)

**Quick Start**:
```bash
# Extract PDF documentation
fileflux extract docs/guidebook.pdf

# Train with automatic preprocessing
mloop train datasets/raw.csv --label target --time 600

# Hooks handle FilePrepper preprocessing automatically
# AutoML trains on clean, normalized data
```
