// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Ironbees.Core.Goals;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
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
    /// Executes the data analysis phase using DataAnalyzer service.
    /// </summary>
    public async Task<DataAnalysisResult> ExecuteDataAnalysisAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Starting data analysis for {DataFilePath}", context.DataFilePath);

        // Use DataAnalyzer service directly (no longer using DataAnalystAgent wrapper)
        var dataAnalyzer = new DataAnalyzer();
        var analysisReport = await dataAnalyzer.AnalyzeAsync(context.DataFilePath);

        // Convert report to our result format
        var result = ConvertToDataAnalysisResult(analysisReport, context);

        _logger?.LogInformation("Data analysis completed. Rows: {Rows}, Columns: {Cols}, Target: {Target}",
            result.RowCount, result.ColumnCount, result.DetectedTargetColumn);

        return result;
    }

    /// <summary>
    /// Executes the model recommendation phase using ModelRecommender service.
    /// </summary>
    public async Task<ModelRecommendationResult> ExecuteModelRecommendationAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Starting model recommendation");

        // Use ModelRecommender service directly (no longer using ModelArchitectAgent wrapper)
        var modelRecommender = new ModelRecommender();

        var analysis = context.DataAnalysis!;

        // Convert DataAnalysisResult back to DataAnalysisReport for ModelRecommender
        var analysisReport = ConvertToDataAnalysisReport(analysis);

        // Get model recommendation from the service
        var recommendation = modelRecommender.RecommendModel(analysisReport);

        // Convert to our result format
        var result = ConvertToModelRecommendationResult(recommendation, analysis);

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
        // Use PreprocessingScriptGenerator service directly (no longer using PreprocessingExpertAgent wrapper)
        var scriptGenerator = new PreprocessingScriptGenerator();

        var analysis = context.DataAnalysis!;

        // Convert DataAnalysisResult back to DataAnalysisReport for PreprocessingScriptGenerator
        var analysisReport = ConvertToDataAnalysisReport(analysis);

        // Generate preprocessing scripts from the service
        var generationResult = scriptGenerator.GenerateScripts(analysisReport);

        // Convert to our result format
        var result = ConvertToPreprocessingResult(generationResult, context, useIncremental: false);

        _logger?.LogInformation("Preprocessing pipeline generated. Scripts: {ScriptCount}",
            generationResult.Scripts.Count);

        return await Task.FromResult(result);
    }

    private async Task<PreprocessingResult> ExecuteIncrementalPreprocessingAsync(
        OrchestrationContext context,
        CancellationToken cancellationToken)
    {
        // TODO: Implement incremental preprocessing using IncrementalPreprocessingAgent.yaml
        // For now, use standard preprocessing
        // In future: Invoke via IronbeesOrchestrator.StreamAsync("incremental-preprocessing", ...)

        _logger?.LogWarning("Incremental preprocessing temporarily using standard preprocessing. Full implementation pending.");
        return await ExecuteStandardPreprocessingAsync(context, cancellationToken);
    }

    /// <summary>
    /// Gets the chat client for creating custom agent interactions.
    /// </summary>
    /// <remarks>
    /// MLOps manager functionality should now be invoked via IronbeesOrchestrator.StreamAsync("mlops-manager", ...)
    /// using the mlops-manager agent.yaml template.
    /// </remarks>
    public IChatClient GetChatClient()
    {
        return _chatClient;
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

    // ========================================
    // Conversion Helper Methods
    // ========================================

    private static DataAnalysisReport ConvertToDataAnalysisReport(DataAnalysisResult result)
    {
        // Convert ColumnInfo back to ColumnAnalysis
        var columns = result.Columns.Select(c => new ColumnAnalysis
        {
            Name = c.Name,
            InferredType = c.DataType switch
            {
                "Categorical" => Models.DataType.Categorical,
                "Numeric" => Models.DataType.Numeric,
                "Boolean" => Models.DataType.Boolean,
                "DateTime" => Models.DataType.DateTime,
                "Text" => Models.DataType.Text,
                _ => Models.DataType.Unknown
            },
            NonNullCount = result.RowCount - c.MissingCount,
            NullCount = c.MissingCount,
            UniqueCount = c.UniqueCount,
            NumericStats = c.IsNumeric ? new NumericStatistics
            {
                Min = 0, Max = 0, Mean = 0, Median = 0,
                StandardDeviation = 0, Variance = 0,
                Q1 = 0, Q3 = 0
            } : null,
            CategoricalStats = c.IsCategorical ? new CategoricalStatistics
            {
                MostFrequentValue = "Unknown",
                MostFrequentCount = 0,
                ValueCounts = new Dictionary<string, int>()
            } : null
        }).ToList();

        // Extract quality issues from result.Issues
        var qualityIssues = new DataQualityIssues
        {
            ColumnsWithMissingValues = result.Issues
                .Where(i => i.Contains("Missing values"))
                .Select(i => i.Replace("Missing values in '", "").Replace("'", ""))
                .ToList(),
            ColumnsWithOutliers = result.Issues
                .Where(i => i.Contains("Outliers"))
                .Select(i => i.Replace("Outliers detected in '", "").Replace("'", ""))
                .ToList(),
            HighCardinalityColumns = result.Recommendations
                .Where(r => r.Contains("High cardinality"))
                .SelectMany(r => r.Split(':').Skip(1).FirstOrDefault()?.Split(',') ?? [])
                .Select(s => s.Trim())
                .ToList(),
            DuplicateRowCount = 0,
            ConstantColumns = new List<string>()
        };

        // Create target recommendation if target column is detected
        TargetRecommendation? targetRecommendation = null;
        if (!string.IsNullOrEmpty(result.DetectedTargetColumn))
        {
            targetRecommendation = new TargetRecommendation
            {
                ColumnName = result.DetectedTargetColumn,
                ProblemType = result.InferredTaskType switch
                {
                    "BinaryClassification" => MLProblemType.BinaryClassification,
                    "MulticlassClassification" => MLProblemType.MulticlassClassification,
                    "Regression" => MLProblemType.Regression,
                    _ => MLProblemType.BinaryClassification
                },
                Confidence = result.Confidence,
                Reason = $"Inferred as {result.InferredTaskType} based on column characteristics"
            };
        }

        return new DataAnalysisReport
        {
            FilePath = string.Empty, // Not available in DataAnalysisResult
            RowCount = result.RowCount,
            ColumnCount = result.ColumnCount,
            FileSizeBytes = 0, // Not available
            Columns = columns,
            RecommendedTarget = targetRecommendation,
            QualityIssues = qualityIssues,
            MLReadiness = new MLReadinessAssessment
            {
                IsReady = result.DataQualityScore >= 0.6,
                ReadinessScore = result.DataQualityScore,
                BlockingIssues = result.Issues.Where(i => i.Contains("missing") || i.Contains("required")).ToList(),
                Warnings = result.Issues.Except(result.Issues.Where(i => i.Contains("missing") || i.Contains("required"))).ToList(),
                Recommendations = result.Recommendations
            },
            AnalyzedAt = DateTime.UtcNow
        };
    }

    private static ModelRecommendationResult ConvertToModelRecommendationResult(
        ModelRecommendation recommendation,
        DataAnalysisResult analysis)
    {
        // Convert string list of trainers to TrainerRecommendation objects for orchestration
        var trainers = recommendation.RecommendedTrainers.Select((trainerName, index) => new TrainerRecommendation
        {
            TrainerName = trainerName,
            Reason = $"Recommended for {recommendation.ProblemType}",
            Priority = index + 1
        }).ToList();

        return new ModelRecommendationResult
        {
            TaskType = recommendation.ProblemType.ToString(),
            PrimaryMetric = recommendation.PrimaryMetric,
            RecommendedTrainers = trainers,
            AutoMLConfig = new AutoMLConfiguration
            {
                MaxTrainingTimeSeconds = recommendation.RecommendedTrainingTimeSeconds,
                MaxModels = recommendation.RecommendedTrainers.Count
            },
            Rationale = recommendation.Reasoning,
            Confidence = recommendation.Confidence
        };
    }

    private static PreprocessingResult ConvertToPreprocessingResult(
        PreprocessingScriptGenerationResult generationResult,
        OrchestrationContext context,
        bool useIncremental)
    {
        var analysis = context.DataAnalysis!;

        // Convert PreprocessingScriptInfo to PreprocessingStep
        var steps = generationResult.Scripts.Select(script => new PreprocessingStep
        {
            Name = script.Name,
            Description = script.Description
        }).ToList();

        return new PreprocessingResult
        {
            OutputFilePath = GetPreprocessedOutputPath(context),
            Steps = steps,
            RowsBefore = analysis.RowCount,
            RowsAfter = analysis.RowCount, // May change after preprocessing execution
            ColumnsBefore = analysis.ColumnCount,
            ColumnsAfter = analysis.ColumnCount, // May change after preprocessing execution
            UsedIncrementalProcessing = useIncremental,
            Confidence = 0.90, // High confidence for rule-based script generation
            PipelineDefinition = generationResult.Summary
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
