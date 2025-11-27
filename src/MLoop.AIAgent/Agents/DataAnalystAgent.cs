using System.Text;
using Ironbees.AgentMode.Agents;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// Data analyst agent specializing in ML dataset analysis and statistical evaluation.
/// Provides insights on data structure, quality, and ML readiness.
/// Integrates with DataAnalyzer for automated data analysis.
/// </summary>
public class DataAnalystAgent : ConversationalAgent
{
    private readonly DataAnalyzer _dataAnalyzer;

    private new const string SystemPrompt = @"# Data Analyst Agent - System Prompt

You are an expert data analyst specializing in machine learning dataset analysis. Your role is to help users understand their data and prepare it for ML model training.

## Core Responsibilities

1. **Data Structure Analysis**
   - Identify columns and their types (numeric, categorical, text, datetime)
   - Determine dataset dimensions (rows, columns)
   - Assess data quality and completeness

2. **Statistical Summary**
   - Calculate descriptive statistics (mean, median, variance, distribution)
   - Identify data ranges and patterns
   - Detect correlations between features

3. **Data Quality Assessment**
   - Detect missing values and their patterns
   - Identify outliers using IQR and Z-score methods
   - Flag potential data quality issues

4. **ML Readiness Evaluation**
   - Recommend target variables for prediction
   - Suggest feature engineering opportunities
   - Identify potential challenges (imbalance, high cardinality, etc.)

## Communication Style

- **Clear and Concise**: Provide actionable insights without overwhelming detail
- **Visual Format**: Use tables, bullet points, and structured output
- **Educational**: Explain why certain patterns matter for ML
- **Proactive**: Suggest next steps based on findings

## Output Format

When analyzing data, structure your response as:

```
📊 Dataset Overview
- Rows: [count]
- Columns: [count]
- File Size: [size]

📋 Column Analysis
[Table of columns with types and statistics]

⚠️ Data Quality Issues
- Missing Values: [details]
- Outliers: [details]
- Potential Problems: [list]

🎯 ML Readiness
- Recommended Target: [column name]
- Problem Type: [classification/regression]
- Key Features: [list]

💡 Next Steps
[Numbered list of recommendations]
```

## Key Principles

1. **Data-Driven**: Base all conclusions on actual data analysis
2. **Practical**: Focus on actionable insights for ML pipeline
3. **Transparent**: Explain your reasoning and assumptions
4. **User-Centric**: Adapt recommendations to user's ML goals

## Integration with MLoop

You work alongside other specialized agents:
- **preprocessing-expert**: Handles data cleaning based on your findings
- **model-architect**: Uses your analysis to recommend models
- **mlops-manager**: Executes the ML pipeline you help design

When you identify issues (missing values, outliers, encoding needs), suggest that the preprocessing-expert can generate appropriate scripts.";

    /// <summary>
    /// Initializes a new instance of the DataAnalystAgent.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions with multi-provider support.</param>
    public DataAnalystAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Initializes a new instance of the DataAnalystAgent with custom prompt.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="customSystemPrompt">Custom system prompt for specialized analysis scenarios.</param>
    public DataAnalystAgent(IChatClient chatClient, string customSystemPrompt)
        : base(chatClient, customSystemPrompt)
    {
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Analyzes a data file and returns insights.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file to analyze.</param>
    /// <param name="userQuery">Optional user query about the data.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Analysis insights and recommendations.</returns>
    public async Task<string> AnalyzeFileAsync(
        string filePath,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Perform automated data analysis
        var report = await _dataAnalyzer.AnalyzeAsync(filePath);

        // Format the analysis report for the LLM
        var analysisContext = FormatReportForLLM(report);

        // Create the message with analysis context
        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this dataset and provide insights:\n\n{analysisContext}"
            : $"Based on this dataset analysis:\n\n{analysisContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw analysis report for a data file.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file.</param>
    /// <returns>The detailed analysis report.</returns>
    public async Task<DataAnalysisReport> GetAnalysisReportAsync(string filePath)
    {
        return await _dataAnalyzer.AnalyzeAsync(filePath);
    }

    /// <summary>
    /// Formats a DataAnalysisReport into a string suitable for LLM context.
    /// </summary>
    private static string FormatReportForLLM(DataAnalysisReport report)
    {
        var sb = new StringBuilder();

        // Dataset Overview
        sb.AppendLine("## Dataset Overview");
        sb.AppendLine($"- File: {Path.GetFileName(report.FilePath)}");
        sb.AppendLine($"- Rows: {report.RowCount:N0}");
        sb.AppendLine($"- Columns: {report.ColumnCount}");
        sb.AppendLine($"- File Size: {FormatFileSize(report.FileSizeBytes)}");
        sb.AppendLine();

        // Column Analysis
        sb.AppendLine("## Column Analysis");
        sb.AppendLine("| Column | Type | Non-Null | Missing % | Unique | Sample Values |");
        sb.AppendLine("|--------|------|----------|-----------|--------|---------------|");

        foreach (var col in report.Columns)
        {
            var samples = col.SampleValues.Take(3);
            var sampleStr = string.Join(", ", samples.Select(s => s.Length > 20 ? s[..17] + "..." : s));
            sb.AppendLine($"| {col.Name} | {col.InferredType} | {col.NonNullCount:N0} | {col.MissingPercentage:F1}% | {col.UniqueCount:N0} | {sampleStr} |");
        }
        sb.AppendLine();

        // Numeric Statistics
        var numericColumns = report.Columns.Where(c => c.NumericStats != null).ToList();
        if (numericColumns.Count > 0)
        {
            sb.AppendLine("## Numeric Statistics");
            sb.AppendLine("| Column | Mean | Median | Std Dev | Min | Max | Outliers |");
            sb.AppendLine("|--------|------|--------|---------|-----|-----|----------|");

            foreach (var col in numericColumns)
            {
                var stats = col.NumericStats!;
                sb.AppendLine($"| {col.Name} | {stats.Mean:F2} | {stats.Median:F2} | {stats.StandardDeviation:F2} | {stats.Min:F2} | {stats.Max:F2} | {stats.OutlierCount} |");
            }
            sb.AppendLine();
        }

        // Data Quality Issues
        if (report.QualityIssues.HasIssues)
        {
            sb.AppendLine("## Data Quality Issues");

            if (report.QualityIssues.ColumnsWithMissingValues.Count > 0)
            {
                sb.AppendLine($"- **Missing Values**: {string.Join(", ", report.QualityIssues.ColumnsWithMissingValues)}");
            }

            if (report.QualityIssues.ColumnsWithOutliers.Count > 0)
            {
                sb.AppendLine($"- **Outliers Detected**: {string.Join(", ", report.QualityIssues.ColumnsWithOutliers)}");
            }

            if (report.QualityIssues.HighCardinalityColumns.Count > 0)
            {
                sb.AppendLine($"- **High Cardinality**: {string.Join(", ", report.QualityIssues.HighCardinalityColumns)}");
            }

            if (report.QualityIssues.ConstantColumns.Count > 0)
            {
                sb.AppendLine($"- **Constant Columns**: {string.Join(", ", report.QualityIssues.ConstantColumns)}");
            }

            if (report.QualityIssues.DuplicateRowCount > 0)
            {
                sb.AppendLine($"- **Duplicate Rows**: {report.QualityIssues.DuplicateRowCount}");
            }
            sb.AppendLine();
        }

        // Target Recommendation
        if (report.RecommendedTarget != null)
        {
            sb.AppendLine("## Target Variable Recommendation");
            sb.AppendLine($"- **Recommended Target**: {report.RecommendedTarget.ColumnName}");
            sb.AppendLine($"- **Problem Type**: {report.RecommendedTarget.ProblemType}");
            sb.AppendLine($"- **Confidence**: {report.RecommendedTarget.Confidence:P0}");
            sb.AppendLine($"- **Reason**: {report.RecommendedTarget.Reason}");

            if (report.RecommendedTarget.AlternativeTargets.Count > 0)
            {
                sb.AppendLine($"- **Alternatives**: {string.Join(", ", report.RecommendedTarget.AlternativeTargets)}");
            }
            sb.AppendLine();
        }

        // ML Readiness
        sb.AppendLine("## ML Readiness Assessment");
        sb.AppendLine($"- **Ready for Training**: {(report.MLReadiness.IsReady ? "Yes ✓" : "No ✗")}");
        sb.AppendLine($"- **Readiness Score**: {report.MLReadiness.ReadinessScore:P0}");

        if (report.MLReadiness.BlockingIssues.Count > 0)
        {
            sb.AppendLine($"- **Blocking Issues**: {string.Join("; ", report.MLReadiness.BlockingIssues)}");
        }

        if (report.MLReadiness.Warnings.Count > 0)
        {
            sb.AppendLine($"- **Warnings**: {string.Join("; ", report.MLReadiness.Warnings)}");
        }

        if (report.MLReadiness.Recommendations.Count > 0)
        {
            sb.AppendLine("- **Recommendations**:");
            foreach (var rec in report.MLReadiness.Recommendations)
            {
                sb.AppendLine($"  - {rec}");
            }
        }

        return sb.ToString();
    }

    private static string FormatFileSize(long bytes)
    {
        string[] sizes = ["B", "KB", "MB", "GB"];
        int order = 0;
        double size = bytes;

        while (size >= 1024 && order < sizes.Length - 1)
        {
            order++;
            size /= 1024;
        }

        return $"{size:F1} {sizes[order]}";
    }
}
