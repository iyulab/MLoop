using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Recommends ML model configurations based on data analysis
/// </summary>
public class ModelRecommender
{
    private const int MinSamplesForDeepLearning = 10000;
    private const int MinSamplesForComplexModels = 1000;
    private const double ImbalanceThreshold = 3.0; // Majority/minority ratio

    /// <summary>
    /// Generate model recommendation from data analysis report
    /// </summary>
    public ModelRecommendation RecommendModel(DataAnalysisReport analysisReport)
    {
        if (analysisReport.RecommendedTarget == null)
        {
            throw new InvalidOperationException(
                "Cannot recommend model: no target column identified in analysis report");
        }

        var targetColumn = analysisReport.Columns
            .First(c => c.Name == analysisReport.RecommendedTarget.ColumnName);

        var featureColumns = analysisReport.Columns
            .Where(c => c.Name != analysisReport.RecommendedTarget.ColumnName)
            .Select(c => c.Name)
            .ToList();

        var characteristics = ExtractDatasetCharacteristics(analysisReport);
        var problemType = analysisReport.RecommendedTarget.ProblemType;

        var trainingTime = RecommendTrainingTime(characteristics);
        var (primaryMetric, additionalMetrics, optimizationMetric) =
            RecommendMetrics(problemType, characteristics);
        var trainers = RecommendTrainers(problemType, characteristics);
        var warnings = GenerateWarnings(analysisReport, characteristics);
        var preprocessing = GeneratePreprocessingRecommendations(analysisReport);

        var reasoning = GenerateReasoning(
            problemType,
            characteristics,
            targetColumn,
            trainingTime);

        return new ModelRecommendation
        {
            ProblemType = problemType,
            TargetColumn = analysisReport.RecommendedTarget.ColumnName,
            FeatureColumns = featureColumns,
            RecommendedTrainingTimeSeconds = trainingTime,
            PrimaryMetric = primaryMetric,
            AdditionalMetrics = additionalMetrics,
            OptimizationMetric = optimizationMetric,
            RecommendedTrainers = trainers.Select(t => t.TrainerName).ToList(),
            Confidence = CalculateConfidence(analysisReport, characteristics),
            Reasoning = reasoning,
            Warnings = warnings,
            PreprocessingRecommendations = preprocessing
        };
    }

    private DatasetCharacteristics ExtractDatasetCharacteristics(DataAnalysisReport report)
    {
        var targetColumn = report.Columns
            .First(c => c.Name == report.RecommendedTarget!.ColumnName);

        var categoricalCount = report.Columns
            .Count(c => c.InferredType == DataType.Categorical ||
                       c.InferredType == DataType.Boolean);

        var numericCount = report.Columns
            .Count(c => c.InferredType == DataType.Numeric);

        var hasHighCardinality = report.QualityIssues.HighCardinalityColumns.Count > 0;

        var totalMissingValues = report.Columns
            .Sum(c => c.NullCount);
        var totalValues = report.Columns
            .Sum(c => c.NonNullCount + c.NullCount);
        var missingPercentage = totalValues > 0
            ? (totalMissingValues / (double)totalValues) * 100
            : 0;

        // Check for class imbalance
        bool isImbalanced = false;
        double? imbalanceRatio = null;

        if (targetColumn.CategoricalStats != null)
        {
            var valueCounts = targetColumn.CategoricalStats.ValueCounts.Values.ToList();
            if (valueCounts.Count > 0)
            {
                var maxCount = valueCounts.Max();
                var minCount = valueCounts.Min();
                if (minCount > 0)
                {
                    imbalanceRatio = maxCount / (double)minCount;
                    isImbalanced = imbalanceRatio > ImbalanceThreshold;
                }
            }
        }

        return new DatasetCharacteristics
        {
            SampleCount = report.RowCount,
            FeatureCount = report.ColumnCount - 1, // Exclude target
            TargetCardinality = targetColumn.UniqueCount,
            IsImbalanced = isImbalanced,
            ImbalanceRatio = imbalanceRatio,
            MissingValuePercentage = missingPercentage,
            CategoricalFeatureCount = categoricalCount,
            NumericFeatureCount = numericCount,
            HasHighCardinalityFeatures = hasHighCardinality
        };
    }

    private int RecommendTrainingTime(DatasetCharacteristics characteristics)
    {
        // Base time on dataset size and complexity
        int baseTime = 30; // seconds

        // Adjust for sample count
        if (characteristics.SampleCount > 100000)
            baseTime = 600; // 10 minutes
        else if (characteristics.SampleCount > 10000)
            baseTime = 300; // 5 minutes
        else if (characteristics.SampleCount > 1000)
            baseTime = 120; // 2 minutes
        else if (characteristics.SampleCount > 100)
            baseTime = 60; // 1 minute

        // Adjust for feature count
        if (characteristics.FeatureCount > 100)
            baseTime = (int)(baseTime * 1.5);
        else if (characteristics.FeatureCount > 50)
            baseTime = (int)(baseTime * 1.2);

        return baseTime;
    }

    private (string primary, List<string> additional, string optimization) RecommendMetrics(
        MLProblemType problemType,
        DatasetCharacteristics characteristics)
    {
        return problemType switch
        {
            MLProblemType.BinaryClassification => characteristics.IsImbalanced
                ? ("F1Score", new List<string> { "AUC", "Accuracy", "Precision", "Recall" }, "F1Score")
                : ("Accuracy", new List<string> { "AUC", "F1Score", "Precision", "Recall" }, "Accuracy"),

            MLProblemType.MulticlassClassification => characteristics.IsImbalanced
                ? ("MacroF1Score", new List<string> { "MicroAccuracy", "MacroAccuracy", "LogLoss" }, "MacroF1Score")
                : ("MicroAccuracy", new List<string> { "MacroAccuracy", "LogLoss", "MacroF1Score" }, "MicroAccuracy"),

            MLProblemType.Regression =>
                ("R2Score", new List<string> { "RMSE", "MAE", "RSquared" }, "RSquared"),

            _ => ("Accuracy", new List<string>(), "Accuracy")
        };
    }

    private List<TrainerRecommendation> RecommendTrainers(
        MLProblemType problemType,
        DatasetCharacteristics characteristics)
    {
        var recommendations = new List<TrainerRecommendation>();

        switch (problemType)
        {
            case MLProblemType.BinaryClassification:
                // LightGBM - best overall for medium to large datasets
                if (characteristics.SampleCount >= 100)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "LightGbm",
                        ExpectedPerformance = 0.9,
                        TrainingSpeed = 0.8,
                        Rationale = "Excellent performance on structured data with good speed"
                    });
                }

                // FastTree - fast and effective
                recommendations.Add(new TrainerRecommendation
                {
                    TrainerName = "FastTree",
                    ExpectedPerformance = 0.85,
                    TrainingSpeed = 0.9,
                    Rationale = "Fast gradient boosting with good accuracy"
                });

                // SDCA - for smaller datasets
                if (characteristics.SampleCount < 10000)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "SdcaLogisticRegression",
                        ExpectedPerformance = 0.75,
                        TrainingSpeed = 0.95,
                        Rationale = "Very fast linear model for smaller datasets"
                    });
                }
                break;

            case MLProblemType.MulticlassClassification:
                if (characteristics.SampleCount >= 100)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "LightGbm",
                        ExpectedPerformance = 0.9,
                        TrainingSpeed = 0.8,
                        Rationale = "Handles multiclass problems efficiently"
                    });
                }

                recommendations.Add(new TrainerRecommendation
                {
                    TrainerName = "FastTree",
                    ExpectedPerformance = 0.85,
                    TrainingSpeed = 0.9,
                    Rationale = "Fast and effective for multiclass classification"
                });

                if (characteristics.SampleCount < 10000)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "SdcaMaximumEntropy",
                        ExpectedPerformance = 0.75,
                        TrainingSpeed = 0.95,
                        Rationale = "Fast linear model for multiclass problems"
                    });
                }
                break;

            case MLProblemType.Regression:
                if (characteristics.SampleCount >= 100)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "LightGbm",
                        ExpectedPerformance = 0.9,
                        TrainingSpeed = 0.8,
                        Rationale = "Best overall for regression tasks"
                    });
                }

                recommendations.Add(new TrainerRecommendation
                {
                    TrainerName = "FastTree",
                    ExpectedPerformance = 0.85,
                    TrainingSpeed = 0.9,
                    Rationale = "Fast gradient boosting for regression"
                });

                if (characteristics.SampleCount < 10000)
                {
                    recommendations.Add(new TrainerRecommendation
                    {
                        TrainerName = "Sdca",
                        ExpectedPerformance = 0.75,
                        TrainingSpeed = 0.95,
                        Rationale = "Fast linear regression for smaller datasets"
                    });
                }
                break;
        }

        return recommendations.OrderByDescending(r => r.ExpectedPerformance).ToList();
    }

    private List<string> GenerateWarnings(
        DataAnalysisReport report,
        DatasetCharacteristics characteristics)
    {
        var warnings = new List<string>();

        if (characteristics.SampleCount < 100)
        {
            warnings.Add("Dataset is very small (<100 samples). Results may not be reliable.");
        }

        if (characteristics.IsImbalanced && characteristics.ImbalanceRatio.HasValue)
        {
            warnings.Add($"Dataset is imbalanced (ratio: {characteristics.ImbalanceRatio.Value:F2}). " +
                        "Consider using stratified sampling or class weighting.");
        }

        if (characteristics.MissingValuePercentage > 10)
        {
            warnings.Add($"{characteristics.MissingValuePercentage:F1}% missing values detected. " +
                        "Preprocessing will be required.");
        }

        if (characteristics.HasHighCardinalityFeatures)
        {
            warnings.Add("High-cardinality categorical features detected. " +
                        "May need feature engineering or encoding strategies.");
        }

        if (characteristics.FeatureCount > characteristics.SampleCount)
        {
            warnings.Add("More features than samples. Risk of overfitting. " +
                        "Consider feature selection or dimensionality reduction.");
        }

        return warnings;
    }

    private List<string> GeneratePreprocessingRecommendations(DataAnalysisReport report)
    {
        var recommendations = new List<string>();

        if (report.QualityIssues.ColumnsWithMissingValues.Count > 0)
        {
            recommendations.Add("Handle missing values through imputation or removal");
        }

        if (report.QualityIssues.ColumnsWithOutliers.Count > 0)
        {
            recommendations.Add("Review and handle outliers (capping, transformation, or removal)");
        }

        if (report.QualityIssues.HighCardinalityColumns.Count > 0)
        {
            recommendations.Add("Encode high-cardinality categorical features (target encoding, hashing)");
        }

        if (report.QualityIssues.ConstantColumns.Count > 0)
        {
            recommendations.Add($"Remove {report.QualityIssues.ConstantColumns.Count} constant columns");
        }

        if (report.QualityIssues.DuplicateRowCount > 0)
        {
            recommendations.Add($"Remove {report.QualityIssues.DuplicateRowCount} duplicate rows");
        }

        var categoricalColumns = report.Columns
            .Where(c => c.InferredType == DataType.Categorical && c.Name != report.RecommendedTarget?.ColumnName)
            .ToList();

        if (categoricalColumns.Count > 0)
        {
            recommendations.Add($"Encode {categoricalColumns.Count} categorical feature columns (one-hot or label encoding)");
        }

        var numericColumns = report.Columns
            .Where(c => c.InferredType == DataType.Numeric && c.Name != report.RecommendedTarget?.ColumnName)
            .ToList();

        if (numericColumns.Count > 0)
        {
            recommendations.Add("Consider normalizing/scaling numeric features for better model performance");
        }

        return recommendations;
    }

    private string GenerateReasoning(
        MLProblemType problemType,
        DatasetCharacteristics characteristics,
        ColumnAnalysis targetColumn,
        int trainingTime)
    {
        var problemTypeStr = problemType switch
        {
            MLProblemType.BinaryClassification => "binary classification",
            MLProblemType.MulticlassClassification => "multiclass classification",
            MLProblemType.Regression => "regression",
            _ => "unknown"
        };

        var sizeDescription = characteristics.SampleCount switch
        {
            < 100 => "very small",
            < 1000 => "small",
            < 10000 => "medium",
            < 100000 => "large",
            _ => "very large"
        };

        var reasoning = $"Detected {problemTypeStr} problem with {characteristics.TargetCardinality} " +
                       $"target classes/values. Dataset is {sizeDescription} ({characteristics.SampleCount} samples, " +
                       $"{characteristics.FeatureCount} features). ";

        if (characteristics.IsImbalanced)
        {
            reasoning += $"Dataset is imbalanced (ratio: {characteristics.ImbalanceRatio:F2}), recommending " +
                        "F1Score as primary metric. ";
        }

        reasoning += $"Recommended training time: {trainingTime}s based on dataset size and complexity.";

        return reasoning;
    }

    private double CalculateConfidence(
        DataAnalysisReport report,
        DatasetCharacteristics characteristics)
    {
        double confidence = 1.0;

        // Reduce confidence for small datasets
        if (characteristics.SampleCount < 100)
            confidence -= 0.3;
        else if (characteristics.SampleCount < 1000)
            confidence -= 0.1;

        // Reduce confidence for missing data
        if (characteristics.MissingValuePercentage > 20)
            confidence -= 0.2;
        else if (characteristics.MissingValuePercentage > 10)
            confidence -= 0.1;

        // Reduce confidence for quality issues
        if (report.QualityIssues.HasIssues)
            confidence -= 0.1;

        // Reduce confidence if more features than samples
        if (characteristics.FeatureCount > characteristics.SampleCount)
            confidence -= 0.2;

        return Math.Max(0.3, Math.Min(1.0, confidence));
    }
}
