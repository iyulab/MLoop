using System.Text;
using Ironbees.AgentMode.Agents;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// Preprocessing expert agent specializing in C# data preprocessing script generation.
/// Creates production-ready IPreprocessingScript implementations for MLoop.
/// </summary>
public class PreprocessingExpertAgent : ConversationalAgent
{
    private readonly PreprocessingScriptGenerator _scriptGenerator;
    private readonly DataAnalyzer _dataAnalyzer;

    private new const string SystemPrompt = @"# Preprocessing Expert Agent - System Prompt

You are an expert in data preprocessing with deep knowledge of C# programming and the MLoop preprocessing pipeline. Your role is to generate production-ready C# scripts that implement `IPreprocessingScript` interface.

## Core Responsibilities

1. **Identify Preprocessing Needs**
   - Analyze data quality issues from data-analyst findings
   - Determine appropriate preprocessing strategies
   - Prioritize preprocessing steps for optimal results

2. **Generate C# Preprocessing Scripts**
   - Create valid C# code implementing `IPreprocessingScript`
   - Follow MLoop preprocessing conventions (`.mloop/scripts/preprocess/`)
   - Use sequential naming (01_handle_missing.cs, 02_encode_categorical.cs)
   - Ensure scripts compile and execute correctly

3. **Preprocessing Patterns**
   - **Missing Values**: mean/median/mode imputation, removal
   - **Categorical Encoding**: one-hot, label encoding
   - **Numeric Scaling**: normalization, standardization
   - **Feature Engineering**: derived features, transformations
   - **Data Cleaning**: outlier handling, duplicate removal

4. **Quality Assurance**
   - Generate compilable, testable C# code
   - Include error handling and validation
   - Add clear comments explaining logic
   - Follow C# best practices and naming conventions

## IPreprocessingScript Interface

All generated scripts must implement:

```csharp
public interface IPreprocessingScript
{
    string Name { get; }
    string Description { get; }
    Task<string> ExecuteAsync(PreprocessContext context);
}
```

**PreprocessContext** provides:
- `InputFilePath`: Path to input CSV/JSON
- `OutputFilePath`: Path to save preprocessed data
- `CsvHelper`: Helper for CSV operations
- `Logger`: For logging progress

## Script Template

```csharp
using MLoop.Extensibility;
using MLoop.Core.Data;
using CsvHelper;
using CsvHelper.Configuration;

namespace MLoop.Preprocessing;

public class [ScriptName] : IPreprocessingScript
{
    public string Name => ""[descriptive-name]"";
    public string Description => ""[what this script does]"";

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.LogInformation($""Executing {Name}..."");

        // 1. Read input data
        var records = await context.CsvHelper.ReadRecordsAsync<dynamic>(
            context.InputFilePath);

        // 2. Apply preprocessing logic
        var processed = records.Select(record => {
            // Your transformation logic here
            return record;
        }).ToList();

        // 3. Write output data
        await context.CsvHelper.WriteRecordsAsync(
            context.OutputFilePath, processed);

        context.Logger.LogInformation($""{Name} completed successfully"");
        return context.OutputFilePath;
    }
}
```

## Communication Style

- **Code-First**: Provide complete, working C# code
- **Explanatory**: Add comments explaining complex logic
- **Sequential**: Name scripts with order prefix (01_, 02_, 03_)
- **Modular**: One concern per script (separation of concerns)

## Output Format

When generating preprocessing scripts:

```
🔧 Preprocessing Plan

I'll create [N] preprocessing scripts:

1. **01_handle_missing.cs** - Handle missing values
   - Age: Fill with median
   - Income: Fill with mean
   - Category: Fill with mode

2. **02_encode_categorical.cs** - Encode categorical variables
   - Gender: One-hot encoding
   - Region: Label encoding

3. **03_scale_numeric.cs** - Scale numeric features
   - StandardScaler for all numeric columns

📝 Script: 01_handle_missing.cs

```csharp
[Complete C# code]
```

💡 **Usage**:
Scripts will be saved to `.mloop/scripts/preprocess/` and executed sequentially during `mloop train`.
```

## Key Principles

1. **Production Quality**: Generate code ready for real-world use
2. **Error Handling**: Include try-catch and validation
3. **Testability**: Code should be unit-testable
4. **Performance**: Efficient algorithms for large datasets
5. **Maintainability**: Clear, documented, well-structured code

## Integration with MLoop

- Scripts are discovered by `PreprocessingEngine`
- Executed in alphabetical/numeric order
- Use `PreprocessContext` for file I/O
- Support hybrid compilation + DLL caching

When users ask for preprocessing help, generate complete scripts they can immediately use in their MLoop projects.";

    /// <summary>
    /// Initializes a new instance of the PreprocessingExpertAgent.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions with multi-provider support.</param>
    public PreprocessingExpertAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _scriptGenerator = new PreprocessingScriptGenerator();
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Initializes a new instance of the PreprocessingExpertAgent with custom prompt.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="customSystemPrompt">Custom system prompt for specialized preprocessing scenarios.</param>
    public PreprocessingExpertAgent(IChatClient chatClient, string customSystemPrompt)
        : base(chatClient, customSystemPrompt)
    {
        _scriptGenerator = new PreprocessingScriptGenerator();
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Analyzes a data file and generates preprocessing scripts with LLM-enhanced explanation.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file to analyze.</param>
    /// <param name="outputDirectory">Optional directory to save generated scripts.</param>
    /// <param name="userQuery">Optional user query about the preprocessing.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Preprocessing scripts with LLM-enhanced explanation.</returns>
    public async Task<string> GenerateScriptsAsync(
        string filePath,
        string? outputDirectory = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Perform data analysis first
        var analysisReport = await _dataAnalyzer.AnalyzeAsync(filePath);

        // Generate preprocessing scripts
        var result = _scriptGenerator.GenerateScripts(analysisReport, outputDirectory);

        // Format for LLM context
        var scriptsContext = FormatScriptsForLLM(result, analysisReport);

        // Create the message with scripts context
        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Based on the following preprocessing scripts I generated, provide your expert analysis and explain the preprocessing strategy:\n\n{scriptsContext}"
            : $"Based on these generated preprocessing scripts:\n\n{scriptsContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Generates preprocessing scripts from an existing data analysis report.
    /// </summary>
    /// <param name="analysisReport">Pre-computed data analysis report.</param>
    /// <param name="outputDirectory">Optional directory to save generated scripts.</param>
    /// <param name="userQuery">Optional user query about the preprocessing.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Preprocessing scripts with LLM-enhanced explanation.</returns>
    public async Task<string> GenerateScriptsFromReportAsync(
        DataAnalysisReport analysisReport,
        string? outputDirectory = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Generate preprocessing scripts
        var result = _scriptGenerator.GenerateScripts(analysisReport, outputDirectory);

        // Format for LLM context
        var scriptsContext = FormatScriptsForLLM(result, analysisReport);

        // Create the message with scripts context
        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Based on the following preprocessing scripts I generated, provide your expert analysis and explain the preprocessing strategy:\n\n{scriptsContext}"
            : $"Based on these generated preprocessing scripts:\n\n{scriptsContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw preprocessing script generation result for a data file.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file.</param>
    /// <param name="outputDirectory">Optional directory to save generated scripts.</param>
    /// <returns>The preprocessing script generation result.</returns>
    public async Task<PreprocessingScriptGenerationResult> GetScriptsAsync(
        string filePath,
        string? outputDirectory = null)
    {
        var analysisReport = await _dataAnalyzer.AnalyzeAsync(filePath);
        return _scriptGenerator.GenerateScripts(analysisReport, outputDirectory);
    }

    /// <summary>
    /// Gets the raw preprocessing script generation result from an existing analysis report.
    /// </summary>
    /// <param name="analysisReport">Pre-computed data analysis report.</param>
    /// <param name="outputDirectory">Optional directory to save generated scripts.</param>
    /// <returns>The preprocessing script generation result.</returns>
    public PreprocessingScriptGenerationResult GetScripts(
        DataAnalysisReport analysisReport,
        string? outputDirectory = null)
    {
        return _scriptGenerator.GenerateScripts(analysisReport, outputDirectory);
    }

    /// <summary>
    /// Saves generated scripts to disk.
    /// </summary>
    /// <param name="result">The preprocessing script generation result.</param>
    /// <param name="outputDirectory">Directory to save scripts.</param>
    /// <returns>Task representing the async operation.</returns>
    public async Task SaveScriptsAsync(PreprocessingScriptGenerationResult result, string outputDirectory)
    {
        await _scriptGenerator.SaveScriptsAsync(result, outputDirectory);
    }

    /// <summary>
    /// Formats a PreprocessingScriptGenerationResult into a string suitable for LLM context.
    /// </summary>
    private static string FormatScriptsForLLM(PreprocessingScriptGenerationResult result, DataAnalysisReport analysisReport)
    {
        var sb = new StringBuilder();

        // Dataset Overview
        sb.AppendLine("## 📊 Dataset Overview");
        sb.AppendLine();
        sb.AppendLine($"- **File**: {Path.GetFileName(analysisReport.FilePath)}");
        sb.AppendLine($"- **Rows**: {analysisReport.RowCount:N0}");
        sb.AppendLine($"- **Columns**: {analysisReport.ColumnCount}");
        sb.AppendLine();

        // Data Quality Issues Addressed
        sb.AppendLine("## ⚠️ Data Quality Issues Addressed");
        sb.AppendLine();
        if (analysisReport.QualityIssues.DuplicateRowCount > 0)
        {
            sb.AppendLine($"- **Duplicate Rows**: {analysisReport.QualityIssues.DuplicateRowCount}");
        }
        if (analysisReport.QualityIssues.ConstantColumns.Count > 0)
        {
            sb.AppendLine($"- **Constant Columns**: {string.Join(", ", analysisReport.QualityIssues.ConstantColumns)}");
        }
        if (analysisReport.QualityIssues.ColumnsWithMissingValues.Count > 0)
        {
            sb.AppendLine($"- **Missing Values**: {string.Join(", ", analysisReport.QualityIssues.ColumnsWithMissingValues)}");
        }
        if (analysisReport.QualityIssues.ColumnsWithOutliers.Count > 0)
        {
            sb.AppendLine($"- **Outliers**: {string.Join(", ", analysisReport.QualityIssues.ColumnsWithOutliers)}");
        }
        if (analysisReport.QualityIssues.HighCardinalityColumns.Count > 0)
        {
            sb.AppendLine($"- **High Cardinality**: {string.Join(", ", analysisReport.QualityIssues.HighCardinalityColumns)}");
        }
        sb.AppendLine();

        // Generated Scripts Summary
        sb.AppendLine("## 🔧 Generated Preprocessing Scripts");
        sb.AppendLine();
        sb.AppendLine($"**Total Scripts**: {result.Scripts.Count}");
        sb.AppendLine();

        foreach (var script in result.Scripts)
        {
            sb.AppendLine($"### {script.Sequence}. {script.FileName}");
            sb.AppendLine($"**Name**: {script.Name}");
            sb.AppendLine($"**Description**: {script.Description}");
            sb.AppendLine();
            sb.AppendLine("```csharp");
            // Include first 50 lines of source code to keep context manageable
            var lines = script.SourceCode.Split('\n');
            var previewLines = lines.Take(50);
            sb.AppendLine(string.Join("\n", previewLines));
            if (lines.Length > 50)
            {
                sb.AppendLine($"// ... ({lines.Length - 50} more lines)");
            }
            sb.AppendLine("```");
            sb.AppendLine();
        }

        // Execution Summary
        sb.AppendLine("## 📋 Execution Summary");
        sb.AppendLine();
        sb.AppendLine(result.Summary);
        sb.AppendLine();

        // Usage Instructions
        sb.AppendLine("## 💡 Usage");
        sb.AppendLine();
        sb.AppendLine("Scripts will be saved to `.mloop/scripts/preprocess/` and executed sequentially during `mloop train`.");
        sb.AppendLine();
        sb.AppendLine("```bash");
        sb.AppendLine("# Train with preprocessing");
        sb.AppendLine($"mloop train {Path.GetFileName(analysisReport.FilePath)} --label {analysisReport.RecommendedTarget?.ColumnName ?? "<target>"}");
        sb.AppendLine("```");

        return sb.ToString();
    }
}
