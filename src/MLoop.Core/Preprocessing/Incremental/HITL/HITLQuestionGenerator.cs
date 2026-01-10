using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Generates HITL questions from preprocessing rules.
/// </summary>
public sealed class HITLQuestionGenerator : IHITLQuestionGenerator
{
    private readonly ContextBuilder _contextBuilder;
    private readonly RecommendationEngine _recommendationEngine;
    private readonly ILogger<HITLQuestionGenerator> _logger;

    public HITLQuestionGenerator(ILogger<HITLQuestionGenerator> logger)
    {
        _logger = logger;
        _contextBuilder = new ContextBuilder();
        _recommendationEngine = new RecommendationEngine();
    }

    /// <inheritdoc />
    public HITLQuestion GenerateQuestion(
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        if (!rule.RequiresHITL)
        {
            throw new InvalidOperationException(
                $"Rule {rule.Id} does not require HITL. Type: {rule.Type}");
        }

        var questionId = $"HITL_{rule.Id}_{DateTime.UtcNow:yyyyMMddHHmmss}";

        return rule.Type switch
        {
            PreprocessingRuleType.MissingValueStrategy =>
                GenerateMissingValueQuestion(questionId, rule, sample, analysis),
            PreprocessingRuleType.OutlierHandling =>
                GenerateOutlierQuestion(questionId, rule, sample, analysis),
            PreprocessingRuleType.CategoryMapping =>
                GenerateCategoryQuestion(questionId, rule, sample, analysis),
            PreprocessingRuleType.TypeConversion =>
                GenerateTypeConversionQuestion(questionId, rule, sample, analysis),
            PreprocessingRuleType.BusinessLogicDecision =>
                GenerateBusinessLogicQuestion(questionId, rule, sample, analysis),
            _ => throw new NotSupportedException($"Rule type {rule.Type} not supported for HITL")
        };
    }

    /// <inheritdoc />
    public IReadOnlyList<HITLQuestion> GenerateAllQuestions(
        IReadOnlyList<PreprocessingRule> rules,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var questions = new List<HITLQuestion>();

        foreach (var rule in rules.Where(r => r.RequiresHITL))
        {
            try
            {
                var question = GenerateQuestion(rule, sample, analysis);
                questions.Add(question);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Failed to generate HITL question for rule {RuleId}. Skipping.",
                    rule.Id);
            }
        }

        _logger.LogInformation(
            "Generated {QuestionCount} HITL questions from {RuleCount} rules",
            questions.Count, rules.Count);

        return questions;
    }

    private HITLQuestion GenerateMissingValueQuestion(
        string questionId,
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var column = sample.Columns[rule.ColumnNames[0]];
        var columnName = rule.ColumnNames[0];

        // Calculate statistics
        var (mean, median) = CalculateStatistics(column);

        var options = new List<HITLOption>
        {
            new()
            {
                Key = "A",
                Label = "Delete records with missing values",
                Description = $"Remove {rule.AffectedRows:N0} records from dataset",
                Action = ActionType.Delete,
                IsRecommended = false
            },
            new()
            {
                Key = "B",
                Label = mean.HasValue ? $"Impute with mean ({mean.Value:F1})" : "Impute with mean",
                Description = "Replace missing values with column mean (best for normal distribution)",
                Action = ActionType.ImputeMean,
                IsRecommended = true
            },
            new()
            {
                Key = "C",
                Label = median.HasValue ? $"Impute with median ({median.Value:F1})" : "Impute with median",
                Description = "Replace missing values with median (robust to outliers)",
                Action = ActionType.ImputeMedian,
                IsRecommended = false
            },
            new()
            {
                Key = "D",
                Label = "Replace with custom default value",
                Description = "Specify a custom value to replace missing data",
                Action = ActionType.ImputeCustom,
                IsRecommended = false
            }
        };

        var context = _contextBuilder.BuildContext(rule, sample, analysis);
        var recommendedOption = _recommendationEngine.GetRecommendedOption(rule, sample);
        var reason = _recommendationEngine.GetRecommendationReason(rule, sample, recommendedOption);

        // Update recommendation flags
        foreach (var option in options)
        {
            option.IsRecommended = option.Key == recommendedOption;
        }

        return new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = context,
            Question = $"How should I handle missing values in the '{columnName}' column?",
            Options = options,
            RecommendedOption = recommendedOption,
            RecommendationReason = reason,
            RelatedRule = rule
        };
    }

    private HITLQuestion GenerateOutlierQuestion(
        string questionId,
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var columnName = rule.ColumnNames[0];

        var options = new List<HITLOption>
        {
            new()
            {
                Key = "A",
                Label = "Keep all values",
                Description = "Outliers may represent legitimate edge cases (e.g., executives, special events)",
                Action = ActionType.KeepAsIs,
                IsRecommended = false
            },
            new()
            {
                Key = "B",
                Label = "Remove outliers completely",
                Description = $"Delete {rule.AffectedRows:N0} records with outlier values",
                Action = ActionType.RemoveOutliers,
                IsRecommended = false
            },
            new()
            {
                Key = "C",
                Label = "Cap at percentile threshold",
                Description = "Cap outliers at 99th percentile (preserves all records)",
                Action = ActionType.CapOutliers,
                IsRecommended = false
            },
            new()
            {
                Key = "D",
                Label = "Flag for manual review",
                Description = "Mark outliers for later review without modifying data",
                Action = ActionType.FlagForReview,
                IsRecommended = false
            }
        };

        var context = _contextBuilder.BuildContext(rule, sample, analysis);
        var recommendedOption = _recommendationEngine.GetRecommendedOption(rule, sample);
        var reason = _recommendationEngine.GetRecommendationReason(rule, sample, recommendedOption);

        foreach (var option in options)
        {
            option.IsRecommended = option.Key == recommendedOption;
        }

        return new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = context,
            Question = $"How should I handle outliers in the '{columnName}' column?",
            Options = options,
            RecommendedOption = recommendedOption,
            RecommendationReason = reason,
            RelatedRule = rule
        };
    }

    private HITLQuestion GenerateCategoryQuestion(
        string questionId,
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var columnName = rule.ColumnNames[0];

        var options = new List<HITLOption>
        {
            new()
            {
                Key = "A",
                Label = "Merge all variations",
                Description = "Standardize to a single category value (improves consistency)",
                Action = ActionType.MergeCategories,
                IsRecommended = true
            },
            new()
            {
                Key = "B",
                Label = "Keep as separate categories",
                Description = "Preserve all variations as distinct categories",
                Action = ActionType.KeepAsIs,
                IsRecommended = false
            },
            new()
            {
                Key = "C",
                Label = "Merge but preserve original",
                Description = "Merge variations but store original values in metadata",
                Action = ActionType.MergeCategories,
                IsRecommended = false
            }
        };

        var context = _contextBuilder.BuildContext(rule, sample, analysis);
        var reason = _recommendationEngine.GetRecommendationReason(rule, sample, "A");

        return new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = context,
            Question = $"Should I merge category variations in the '{columnName}' column?",
            Options = options,
            RecommendedOption = "A",
            RecommendationReason = reason,
            RelatedRule = rule
        };
    }

    private HITLQuestion GenerateTypeConversionQuestion(
        string questionId,
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var columnName = rule.ColumnNames[0];

        var options = new List<HITLOption>
        {
            new()
            {
                Key = "A",
                Label = "Convert to most common type",
                Description = "Convert all values to the predominant type",
                Action = ActionType.ConvertType,
                IsRecommended = true
            },
            new()
            {
                Key = "B",
                Label = "Convert to string (preserve all)",
                Description = "Convert to string to preserve all value variations",
                Action = ActionType.ConvertType,
                IsRecommended = false
            },
            new()
            {
                Key = "C",
                Label = "Delete incompatible records",
                Description = $"Remove {rule.AffectedRows:N0} records with incompatible types",
                Action = ActionType.Delete,
                IsRecommended = false
            }
        };

        var context = _contextBuilder.BuildContext(rule, sample, analysis);
        var reason = _recommendationEngine.GetRecommendationReason(rule, sample, "A");

        return new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = context,
            Question = $"How should I handle type inconsistencies in the '{columnName}' column?",
            Options = options,
            RecommendedOption = "A",
            RecommendationReason = reason,
            RelatedRule = rule
        };
    }

    private HITLQuestion GenerateBusinessLogicQuestion(
        string questionId,
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var columnName = rule.ColumnNames[0];

        var options = new List<HITLOption>
        {
            new()
            {
                Key = "A",
                Label = "Apply suggested transformation",
                Description = rule.SuggestedAction ?? "Apply the suggested preprocessing action",
                Action = ActionType.CustomLogic,
                IsRecommended = false
            },
            new()
            {
                Key = "B",
                Label = "Delete affected records",
                Description = $"Remove {rule.AffectedRows:N0} records",
                Action = ActionType.Delete,
                IsRecommended = false
            },
            new()
            {
                Key = "C",
                Label = "Keep as-is (no action)",
                Description = "Preserve current values without modification",
                Action = ActionType.KeepAsIs,
                IsRecommended = true
            },
            new()
            {
                Key = "D",
                Label = "Specify custom action",
                Description = "Define a custom business logic rule",
                Action = ActionType.CustomLogic,
                IsRecommended = false
            }
        };

        var context = _contextBuilder.BuildContext(rule, sample, analysis);
        var reason = "Keeping as-is is safest until business logic requirements are clarified";

        return new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = context,
            Question = $"How should I handle the business logic pattern in the '{columnName}' column?",
            Options = options,
            RecommendedOption = "C",
            RecommendationReason = reason,
            RelatedRule = rule
        };
    }

    private static (double? mean, double? median) CalculateStatistics(DataFrameColumn column)
    {
        if (column is PrimitiveDataFrameColumn<double> doubleCol)
        {
            var mean = CalculateMean(doubleCol);
            var median = CalculateMedian(doubleCol);
            return (mean, median);
        }

        if (column is PrimitiveDataFrameColumn<int> intCol)
        {
            var mean = CalculateMeanInt(intCol);
            var median = CalculateMedianInt(intCol);
            return (mean, median);
        }

        return (null, null);
    }

    private static double CalculateMean(PrimitiveDataFrameColumn<double> column)
    {
        double sum = 0;
        int count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue && !double.IsNaN(value.Value) && !double.IsInfinity(value.Value))
            {
                sum += value.Value;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    private static double CalculateMeanInt(PrimitiveDataFrameColumn<int> column)
    {
        long sum = 0;
        int count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue)
            {
                sum += value.Value;
                count++;
            }
        }

        return count > 0 ? (double)sum / count : 0;
    }

    private static double CalculateMedian(PrimitiveDataFrameColumn<double> column)
    {
        var values = new List<double>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue && !double.IsNaN(value.Value) && !double.IsInfinity(value.Value))
            {
                values.Add(value.Value);
            }
        }

        if (values.Count == 0)
            return 0;

        values.Sort();
        var mid = values.Count / 2;

        return values.Count % 2 == 0
            ? (values[mid - 1] + values[mid]) / 2
            : values[mid];
    }

    private static double CalculateMedianInt(PrimitiveDataFrameColumn<int> column)
    {
        var values = new List<int>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue)
            {
                values.Add(value.Value);
            }
        }

        if (values.Count == 0)
            return 0;

        values.Sort();
        var mid = values.Count / 2;

        return values.Count % 2 == 0
            ? (values[mid - 1] + values[mid]) / 2.0
            : values[mid];
    }
}
