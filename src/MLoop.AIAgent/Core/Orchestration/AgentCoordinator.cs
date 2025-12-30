// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Ironbees.Core.Goals;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Agents;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Coordinates agent execution and routes tasks to appropriate specialized agents.
/// Manages the 5 agent types: DataAnalyst, ModelArchitect, PreprocessingExpert, MLOpsManager, IncrementalPreprocessing
/// </summary>
public class AgentCoordinator
{
    private readonly IChatClient _chatClient;
    private readonly ILogger<AgentCoordinator>? _logger;

    /// <summary>
    /// Threshold for using incremental preprocessing (rows).
    /// </summary>
    public int LargeDatasetThreshold { get; set; } = 100_000;

    public AgentCoordinator(IChatClient chatClient, ILogger<AgentCoordinator>? logger = null)
    {
        _chatClient = chatClient;
        _logger = logger;
    }

    /// <summary>
    /// Executes the data analysis phase using DataAnalystAgent.
    /// </summary>
    public async Task<DataAnalysisResult> ExecuteDataAnalysisAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Starting data analysis for {DataFilePath}", context.DataFilePath);

        var agent = new DataAnalystAgent(_chatClient);

        // Use the agent's built-in analysis capability
        var analysisReport = await agent.GetAnalysisReportAsync(context.DataFilePath);

        // Convert report to our result format
        var result = ConvertToDataAnalysisResult(analysisReport, context);

        _logger?.LogInformation("Data analysis completed. Rows: {Rows}, Columns: {Cols}, Target: {Target}",
            result.RowCount, result.ColumnCount, result.DetectedTargetColumn);

        return result;
    }

    /// <summary>
    /// Executes the model recommendation phase using ModelArchitectAgent.
    /// </summary>
    public async Task<ModelRecommendationResult> ExecuteModelRecommendationAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Starting model recommendation");

        var agent = new ModelArchitectAgent(_chatClient);

        var analysis = context.DataAnalysis!;
        var request = $"""
            Based on the following data analysis, recommend models and AutoML configuration:

            Dataset Summary:
            - Rows: {analysis.RowCount:N0}
            - Columns: {analysis.ColumnCount}
            - Target: {analysis.DetectedTargetColumn}
            - Task Type: {analysis.InferredTaskType}
            - Data Quality: {analysis.DataQualityScore:P0}

            Column Details:
            {string.Join("\n", analysis.Columns.Select(c => $"- {c.Name}: {c.DataType} ({(c.IsCategorical ? "Categorical" : "Numeric")}, {c.MissingPercentage:P1} missing)"))}

            Please provide:
            1. Recommended ML task type confirmation
            2. Primary evaluation metric
            3. Top 3-5 trainer/algorithm recommendations with rationale
            4. AutoML configuration (training time, max models, etc.)
            5. Any special considerations

            {(context.Options.MaxTrainingTimeSeconds.HasValue ? $"Note: Maximum training time is {context.Options.MaxTrainingTimeSeconds}s" : "")}
            """;

        var response = await agent.RespondAsync(request, cancellationToken: cancellationToken);

        var result = ParseModelRecommendationResponse(response, analysis);

        _logger?.LogInformation("Model recommendation completed. Task: {TaskType}, Metric: {Metric}, Trainers: {Trainers}",
            result.TaskType, result.PrimaryMetric, string.Join(", ", result.RecommendedTrainers.Select(t => t.TrainerName)));

        return result;
    }

    /// <summary>
    /// Executes the preprocessing phase using PreprocessingExpertAgent or IncrementalPreprocessingAgent.
    /// </summary>
    public async Task<PreprocessingResult> ExecutePreprocessingAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        var analysis = context.DataAnalysis!;
        var useIncremental = analysis.RowCount >= LargeDatasetThreshold;

        _logger?.LogInformation("Starting preprocessing. UseIncremental: {UseIncremental}", useIncremental);

        if (useIncremental)
        {
            return await ExecuteIncrementalPreprocessingAsync(context, cancellationToken);
        }

        return await ExecuteStandardPreprocessingAsync(context, cancellationToken);
    }

    private async Task<PreprocessingResult> ExecuteStandardPreprocessingAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken)
    {
        var agent = new PreprocessingExpertAgent(_chatClient);

        var analysis = context.DataAnalysis!;
        var request = $"""
            Create a preprocessing pipeline for the dataset:

            Dataset: {context.DataFilePath}
            Target: {analysis.DetectedTargetColumn}
            Task Type: {analysis.InferredTaskType}

            Issues to Address:
            {string.Join("\n", analysis.Issues.Select(i => $"- {i}"))}

            Recommendations:
            {string.Join("\n", analysis.Recommendations.Select(r => $"- {r}"))}

            Column Information:
            {string.Join("\n", analysis.Columns.Select(c => $"- {c.Name}: {c.DataType}, Missing: {c.MissingPercentage:P1}"))}

            Please provide:
            1. Step-by-step preprocessing pipeline
            2. Handling for missing values
            3. Feature encoding strategies
            4. Feature scaling if needed
            5. Output file path suggestion
            """;

        var response = await agent.RespondAsync(request, cancellationToken: cancellationToken);

        return ParsePreprocessingResponse(response, context, useIncremental: false);
    }

    private async Task<PreprocessingResult> ExecuteIncrementalPreprocessingAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken)
    {
        var analysis = context.DataAnalysis!;

        // Use ironbees AgenticSettings if available, otherwise create default
        var agenticSettings = context.AgenticSettings ?? new AgenticSettings
        {
            Sampling = new SamplingSettings
            {
                Strategy = SamplingStrategy.Progressive,
                InitialBatchSize = 1000,
                GrowthFactor = 2,
                MaxSamples = 50000
            },
            Confidence = new ConfidenceSettings
            {
                Threshold = 0.95,
                StabilityWindow = 3
            }
        };

        var agent = new IncrementalPreprocessingAgent(_chatClient, agenticSettings);

        var request = $"""
            Process this large dataset incrementally:

            Dataset: {context.DataFilePath}
            Rows: {analysis.RowCount:N0}
            Target: {analysis.DetectedTargetColumn}

            Use progressive sampling to discover preprocessing rules efficiently.
            Apply discovered rules to the full dataset after validation.
            """;

        var response = await agent.RespondAsync(request, cancellationToken: cancellationToken);

        return ParsePreprocessingResponse(response, context, useIncremental: true);
    }

    /// <summary>
    /// Creates the MLOps manager agent for deployment tasks.
    /// </summary>
    public MLOpsManagerAgent CreateMLOpsManager()
    {
        return new MLOpsManagerAgent(_chatClient);
    }

    // ========================================
    // Response Parsing Helpers
    // ========================================

    private static DataAnalysisResult ConvertToDataAnalysisResult(DataAnalysisReport report, OrchestrationContext context)
    {
        // Determine target column from options or recommended target
        var targetColumn = context.Options.TargetColumn
            ?? report.RecommendedTarget?.ColumnName
            ?? report.Columns.FirstOrDefault(c =>
                c.Name.Contains("target", StringComparison.OrdinalIgnoreCase) ||
                c.Name.Contains("label", StringComparison.OrdinalIgnoreCase) ||
                c.Name.Equals("y", StringComparison.OrdinalIgnoreCase))?.Name;

        // Infer task type from target column characteristics or recommended target
        var taskType = context.Options.TaskType
            ?? report.RecommendedTarget?.ProblemType.ToString()
            ?? InferTaskType(report, targetColumn);

        // Convert columns
        var columns = report.Columns.Select(c => new ColumnInfo
        {
            Name = c.Name,
            DataType = c.InferredType.ToString(),
            MissingCount = c.NullCount,
            MissingPercentage = c.MissingPercentage / 100.0, // Convert from percentage to ratio
            UniqueCount = c.UniqueCount,
            IsCategorical = c.CategoricalStats != null || c.InferredType == Models.DataType.Categorical,
            IsNumeric = c.NumericStats != null || c.InferredType == Models.DataType.Numeric,
            IsTarget = c.Name == targetColumn,
            Role = c.Name == targetColumn ? "Label" : "Feature"
        }).ToList();

        // Calculate data quality score from MLReadiness
        var qualityScore = report.MLReadiness.ReadinessScore;
        var totalMissing = report.Columns.Sum(c => c.NullCount);
        var totalCells = report.RowCount * report.Columns.Count;
        var missingPercentage = totalCells > 0 ? (double)totalMissing / totalCells : 0;

        // Collect issues from quality issues
        var issues = new List<string>();
        issues.AddRange(report.QualityIssues.ColumnsWithMissingValues.Select(c => $"Missing values in '{c}'"));
        issues.AddRange(report.QualityIssues.ColumnsWithOutliers.Select(c => $"Outliers detected in '{c}'"));
        issues.AddRange(report.MLReadiness.BlockingIssues);
        issues.AddRange(report.MLReadiness.Warnings);

        // Collect recommendations
        var recommendations = report.MLReadiness.Recommendations.ToList();
        if (report.QualityIssues.HighCardinalityColumns.Count > 0)
            recommendations.Add($"High cardinality columns detected: {string.Join(", ", report.QualityIssues.HighCardinalityColumns)}");

        return new DataAnalysisResult
        {
            RowCount = report.RowCount,
            ColumnCount = report.Columns.Count,
            DetectedTargetColumn = targetColumn,
            InferredTaskType = taskType,
            Columns = columns,
            DataQualityScore = qualityScore,
            MissingValuePercentage = missingPercentage,
            Issues = issues,
            Recommendations = recommendations,
            Confidence = report.RecommendedTarget?.Confidence ?? (targetColumn != null ? 0.85 : 0.60),
            RawAnalysis = null
        };
    }

    private static string? InferTaskType(DataAnalysisReport report, string? targetColumn)
    {
        if (targetColumn == null) return null;

        var targetCol = report.Columns.FirstOrDefault(c => c.Name == targetColumn);
        if (targetCol == null) return null;

        // Check if it's categorical based on stats or inferred type
        var isCategorical = targetCol.CategoricalStats != null || targetCol.InferredType == Models.DataType.Categorical;

        if (isCategorical)
        {
            return targetCol.UniqueCount == 2 ? "BinaryClassification" : "MulticlassClassification";
        }

        return "Regression";
    }

    private ModelRecommendationResult ParseModelRecommendationResponse(string response, DataAnalysisResult analysis)
    {
        // Default ML.NET trainers based on task type
        var taskType = analysis.InferredTaskType ?? "BinaryClassification";
        var trainers = GetDefaultTrainers(taskType);
        var metric = GetDefaultMetric(taskType);

        return new ModelRecommendationResult
        {
            TaskType = taskType,
            PrimaryMetric = metric,
            RecommendedTrainers = trainers,
            AutoMLConfig = new AutoMLConfiguration
            {
                MaxTrainingTimeSeconds = 300,
                MaxModels = 10
            },
            Rationale = response,
            Confidence = 0.80
        };
    }

    private static List<TrainerRecommendation> GetDefaultTrainers(string taskType)
    {
        return taskType.ToLowerInvariant() switch
        {
            "binaryclassification" => [
                new TrainerRecommendation { TrainerName = "FastForest", Reason = "Good for imbalanced data", Priority = 1 },
                new TrainerRecommendation { TrainerName = "LightGbm", Reason = "Fast and accurate", Priority = 2 },
                new TrainerRecommendation { TrainerName = "FastTree", Reason = "Reliable baseline", Priority = 3 }
            ],
            "multiclassclassification" => [
                new TrainerRecommendation { TrainerName = "LightGbm", Reason = "Handles multiclass well", Priority = 1 },
                new TrainerRecommendation { TrainerName = "FastForest", Reason = "Robust to noise", Priority = 2 },
                new TrainerRecommendation { TrainerName = "SdcaMaximumEntropy", Reason = "Linear baseline", Priority = 3 }
            ],
            "regression" => [
                new TrainerRecommendation { TrainerName = "FastTree", Reason = "Handles non-linear patterns", Priority = 1 },
                new TrainerRecommendation { TrainerName = "LightGbm", Reason = "State-of-the-art", Priority = 2 },
                new TrainerRecommendation { TrainerName = "Sdca", Reason = "Fast linear baseline", Priority = 3 }
            ],
            _ => [
                new TrainerRecommendation { TrainerName = "LightGbm", Reason = "General purpose", Priority = 1 }
            ]
        };
    }

    private static string GetDefaultMetric(string taskType)
    {
        return taskType.ToLowerInvariant() switch
        {
            "binaryclassification" => "AUC",
            "multiclassclassification" => "MicroAccuracy",
            "regression" => "RSquared",
            _ => "Accuracy"
        };
    }

    private PreprocessingResult ParsePreprocessingResponse(string response, OrchestrationContext context, bool useIncremental)
    {
        var analysis = context.DataAnalysis!;

        return new PreprocessingResult
        {
            OutputFilePath = GetPreprocessedOutputPath(context),
            Steps = [
                new PreprocessingStep { Name = "Missing Value Handling", Description = "Impute or remove missing values" },
                new PreprocessingStep { Name = "Feature Encoding", Description = "Encode categorical variables" },
                new PreprocessingStep { Name = "Feature Scaling", Description = "Normalize numeric features" }
            ],
            RowsBefore = analysis.RowCount,
            RowsAfter = analysis.RowCount, // May change after preprocessing
            ColumnsBefore = analysis.ColumnCount,
            ColumnsAfter = analysis.ColumnCount, // May change
            UsedIncrementalProcessing = useIncremental,
            Confidence = 0.80,
            PipelineDefinition = response
        };
    }

    private static string GetPreprocessedOutputPath(OrchestrationContext context)
    {
        var dir = Path.GetDirectoryName(context.DataFilePath) ?? ".";
        var name = Path.GetFileNameWithoutExtension(context.DataFilePath);
        var ext = Path.GetExtension(context.DataFilePath);
        return Path.Combine(dir, $"{name}_preprocessed{ext}");
    }
}
