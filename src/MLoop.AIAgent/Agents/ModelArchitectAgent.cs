using System.Text;
using Ironbees.AgentMode.Agents;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// Model architect agent specializing in ML problem classification and AutoML configuration.
/// Recommends optimal training configurations based on data characteristics.
/// Integrates with ModelRecommender for automated model recommendations.
/// </summary>
public class ModelArchitectAgent : ConversationalAgent
{
    private readonly ModelRecommender _modelRecommender;
    private readonly DataAnalyzer _dataAnalyzer;

    private new const string SystemPrompt = @"# Model Architect Agent - System Prompt

You are an expert ML architect with deep knowledge of ML.NET, AutoML, and machine learning best practices. Your role is to classify ML problems and recommend optimal configurations for MLoop training.

## Core Responsibilities

1. **ML Problem Classification**
   - Identify problem type: Binary Classification, Multiclass Classification, or Regression
   - Determine based on target variable characteristics
   - Consider dataset size and complexity

2. **Model Recommendation**
   - Recommend ML.NET trainers suitable for the problem
   - Consider data characteristics (size, features, balance)
   - Suggest AutoML time limits based on dataset complexity

3. **AutoML Configuration**
   - Recommend optimal time_limit (seconds)
   - Select appropriate performance metric
   - Suggest test_split ratio
   - Configure AutoML search strategy

4. **Performance Metric Selection**
   - **Binary Classification**: Accuracy, F1-Score, AUC, Precision, Recall
   - **Multiclass Classification**: Accuracy, MacroAccuracy, MicroAccuracy
   - **Regression**: R-Squared, RMSE, MAE

## Decision Framework

### Problem Type Detection

**Binary Classification**:
- Target has exactly 2 unique values
- Values like {0, 1}, {True, False}, {""Yes"", ""No""}
- Example: Churn prediction, fraud detection

**Multiclass Classification**:
- Target has 3+ unique categorical values
- Values like {""Low"", ""Medium"", ""High""}
- Example: Sentiment analysis (positive/negative/neutral)

**Regression**:
- Target is continuous numeric
- Predicting quantities, prices, scores
- Example: House price prediction, sales forecasting

### Time Limit Recommendations

| Dataset Size | Complexity | Recommended Time |
|-------------|-----------|------------------|
| < 1,000 rows | Simple | 60-120 seconds |
| 1,000-10,000 | Medium | 180-300 seconds |
| 10,000-100,000 | Large | 300-600 seconds |
| > 100,000 | Very Large | 600-1200 seconds |

### Metric Selection

**Binary Classification**:
- **Balanced data**: Accuracy
- **Imbalanced data**: F1-Score or AUC
- **Cost-sensitive**: Precision (minimize false positives) or Recall (minimize false negatives)

**Multiclass Classification**:
- **Balanced classes**: Accuracy
- **Imbalanced classes**: MacroAccuracy

**Regression**:
- **General**: R-Squared
- **Error magnitude matters**: RMSE
- **Outlier-robust**: MAE

## Communication Style

- **Recommendation-Driven**: Provide clear, specific recommendations
- **Evidence-Based**: Explain reasoning based on data characteristics
- **Configurable**: Offer alternatives if user has specific preferences
- **Actionable**: Translate recommendations into MLoop commands

## Output Format

When recommending model configuration:

```
🎯 ML Problem Analysis

**Problem Type**: [Binary Classification / Multiclass Classification / Regression]

**Reasoning**:
- Target Variable: [column name]
- Unique Values: [count or range]
- Data Characteristics: [key observations]

📊 Recommended Configuration

**AutoML Settings**:
- Time Limit: [seconds] (reasoning: [dataset size/complexity])
- Metric: [metric name] (reasoning: [data balance/business goal])
- Test Split: [ratio] (default: 0.2)

**Expected Trainers**:
- Primary: [trainer name] - [why it fits]
- Alternatives: [other suitable trainers]

⚙️ MLoop Command

```bash
mloop train [data.csv] \
  --label [target_column] \
  --time [seconds] \
  --metric [metric] \
  --test-split [ratio]
```

💡 **Next Steps**:
1. [Specific actionable step]
2. [Another actionable step]
```

## Key Principles

1. **Data-Driven**: Base recommendations on actual data characteristics
2. **ML.NET Aware**: Leverage ML.NET's AutoML capabilities
3. **Practical**: Focus on configurations that work in real scenarios
4. **Transparent**: Explain trade-offs and alternatives
5. **User-Centric**: Adapt to user's goals and constraints

## Integration with MLoop

You work with:
- **data-analyst**: Uses their analysis to classify problems
- **preprocessing-expert**: Considers preprocessing impact on model selection
- **mlops-manager**: Provides configurations for actual training execution

When making recommendations:
- Consider preprocessing applied by preprocessing-expert
- Account for data quality issues identified by data-analyst
- Provide configurations ready for mlops-manager to execute

## Advanced Considerations

- **Feature Count**: High-dimensional data may need more time
- **Class Imbalance**: Affects metric and trainer selection
- **Data Quality**: Poor quality may require conservative settings
- **Business Context**: Align metrics with business goals (e.g., minimize false negatives in medical diagnosis)

Always explain your reasoning and provide alternatives when multiple valid approaches exist.";

    /// <summary>
    /// Initializes a new instance of the ModelArchitectAgent.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions with multi-provider support.</param>
    public ModelArchitectAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _modelRecommender = new ModelRecommender();
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Initializes a new instance of the ModelArchitectAgent with custom prompt.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="customSystemPrompt">Custom system prompt for specialized architecture scenarios.</param>
    public ModelArchitectAgent(IChatClient chatClient, string customSystemPrompt)
        : base(chatClient, customSystemPrompt)
    {
        _modelRecommender = new ModelRecommender();
        _dataAnalyzer = new DataAnalyzer();
    }

    /// <summary>
    /// Analyzes a data file and generates model recommendations.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file to analyze.</param>
    /// <param name="userQuery">Optional user query about the model configuration.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Model recommendation with LLM-enhanced explanation.</returns>
    public async Task<string> RecommendModelAsync(
        string filePath,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Perform data analysis first
        var analysisReport = await _dataAnalyzer.AnalyzeAsync(filePath);

        // Generate model recommendation
        var recommendation = _modelRecommender.RecommendModel(analysisReport);

        // Format the recommendation for the LLM
        var recommendationContext = FormatRecommendationForLLM(recommendation, analysisReport);

        // Create the message with recommendation context
        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Based on the following model recommendation, provide your expert analysis and additional insights:\n\n{recommendationContext}"
            : $"Based on this model recommendation:\n\n{recommendationContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Generates model recommendation from an existing data analysis report.
    /// </summary>
    /// <param name="analysisReport">Pre-computed data analysis report.</param>
    /// <param name="userQuery">Optional user query about the model configuration.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Model recommendation with LLM-enhanced explanation.</returns>
    public async Task<string> RecommendModelFromReportAsync(
        DataAnalysisReport analysisReport,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Generate model recommendation
        var recommendation = _modelRecommender.RecommendModel(analysisReport);

        // Format the recommendation for the LLM
        var recommendationContext = FormatRecommendationForLLM(recommendation, analysisReport);

        // Create the message with recommendation context
        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Based on the following model recommendation, provide your expert analysis and additional insights:\n\n{recommendationContext}"
            : $"Based on this model recommendation:\n\n{recommendationContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw model recommendation for a data file.
    /// </summary>
    /// <param name="filePath">Path to the CSV or JSON file.</param>
    /// <returns>The detailed model recommendation.</returns>
    public async Task<ModelRecommendation> GetModelRecommendationAsync(string filePath)
    {
        var analysisReport = await _dataAnalyzer.AnalyzeAsync(filePath);
        return _modelRecommender.RecommendModel(analysisReport);
    }

    /// <summary>
    /// Gets the raw model recommendation from an existing analysis report.
    /// </summary>
    /// <param name="analysisReport">Pre-computed data analysis report.</param>
    /// <returns>The detailed model recommendation.</returns>
    public ModelRecommendation GetModelRecommendation(DataAnalysisReport analysisReport)
    {
        return _modelRecommender.RecommendModel(analysisReport);
    }

    /// <summary>
    /// Formats a ModelRecommendation into a string suitable for LLM context.
    /// </summary>
    private static string FormatRecommendationForLLM(ModelRecommendation recommendation, DataAnalysisReport analysisReport)
    {
        var sb = new StringBuilder();

        // Problem Analysis
        sb.AppendLine("## 🎯 ML Problem Analysis");
        sb.AppendLine();
        sb.AppendLine($"**Problem Type**: {FormatProblemType(recommendation.ProblemType)}");
        sb.AppendLine($"**Target Column**: {recommendation.TargetColumn}");
        sb.AppendLine($"**Feature Columns**: {recommendation.FeatureColumns.Count} columns");
        sb.AppendLine($"**Confidence**: {recommendation.Confidence:P0}");
        sb.AppendLine();
        sb.AppendLine($"**Reasoning**: {recommendation.Reasoning}");
        sb.AppendLine();

        // Dataset Overview
        sb.AppendLine("## 📊 Dataset Overview");
        sb.AppendLine();
        sb.AppendLine($"- **Rows**: {analysisReport.RowCount:N0}");
        sb.AppendLine($"- **Columns**: {analysisReport.ColumnCount}");
        sb.AppendLine($"- **Feature Columns**: {string.Join(", ", recommendation.FeatureColumns.Take(5))}{(recommendation.FeatureColumns.Count > 5 ? "..." : "")}");
        sb.AppendLine();

        // Recommended Configuration
        sb.AppendLine("## ⚙️ Recommended Configuration");
        sb.AppendLine();
        sb.AppendLine("**AutoML Settings**:");
        sb.AppendLine($"- Time Limit: {recommendation.RecommendedTrainingTimeSeconds} seconds");
        sb.AppendLine($"- Primary Metric: {recommendation.PrimaryMetric}");
        sb.AppendLine($"- Optimization Metric: {recommendation.OptimizationMetric}");
        if (recommendation.AdditionalMetrics.Count > 0)
        {
            sb.AppendLine($"- Additional Metrics: {string.Join(", ", recommendation.AdditionalMetrics)}");
        }
        sb.AppendLine();

        // Recommended Trainers
        if (recommendation.RecommendedTrainers.Count > 0)
        {
            sb.AppendLine("**Recommended Trainers**:");
            foreach (var trainer in recommendation.RecommendedTrainers)
            {
                sb.AppendLine($"- {trainer}");
            }
            sb.AppendLine();
        }

        // MLoop Command
        sb.AppendLine("## 🚀 MLoop Command");
        sb.AppendLine();
        sb.AppendLine("```bash");
        sb.AppendLine($"mloop train {Path.GetFileName(analysisReport.FilePath)} \\");
        sb.AppendLine($"  --label {recommendation.TargetColumn} \\");
        sb.AppendLine($"  --time {recommendation.RecommendedTrainingTimeSeconds} \\");
        sb.AppendLine($"  --metric {recommendation.OptimizationMetric} \\");
        sb.AppendLine("  --test-split 0.2");
        sb.AppendLine("```");
        sb.AppendLine();

        // Warnings
        if (recommendation.Warnings.Count > 0)
        {
            sb.AppendLine("## ⚠️ Warnings");
            sb.AppendLine();
            foreach (var warning in recommendation.Warnings)
            {
                sb.AppendLine($"- {warning}");
            }
            sb.AppendLine();
        }

        // Preprocessing Recommendations
        if (recommendation.PreprocessingRecommendations.Count > 0)
        {
            sb.AppendLine("## 🔧 Preprocessing Recommendations");
            sb.AppendLine();
            foreach (var rec in recommendation.PreprocessingRecommendations)
            {
                sb.AppendLine($"- {rec}");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    private static string FormatProblemType(MLProblemType problemType)
    {
        return problemType switch
        {
            MLProblemType.BinaryClassification => "Binary Classification",
            MLProblemType.MulticlassClassification => "Multiclass Classification",
            MLProblemType.Regression => "Regression",
            _ => problemType.ToString()
        };
    }
}
