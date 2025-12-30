using System.Text;

namespace MLoop.AIAgent.Core.HITL;

/// <summary>
/// Formats HITL questions as interactive prompts for CLI/UI display.
/// </summary>
public class InteractivePromptBuilder
{
    /// <summary>
    /// Builds a formatted prompt string for a HITL question.
    /// </summary>
    public string BuildPrompt(HITLQuestion question)
    {
        var sb = new StringBuilder();

        // Header with question ID and priority
        sb.AppendLine();
        sb.AppendLine($"╔══════════════════════════════════════════════════════════════╗");
        sb.AppendLine($"║  📋 HITL Decision Required [{question.Id}]                          ║");
        sb.AppendLine($"╚══════════════════════════════════════════════════════════════╝");
        sb.AppendLine();

        // Context section
        sb.AppendLine("📊 Context:");
        sb.AppendLine($"   {question.Context}");
        sb.AppendLine();

        // Question
        sb.AppendLine($"❓ {question.Question}");
        sb.AppendLine();

        // Options based on question type
        switch (question.Type)
        {
            case HITLQuestionType.MultipleChoice:
                BuildMultipleChoicePrompt(sb, question);
                break;
            case HITLQuestionType.YesNo:
                BuildYesNoPrompt(sb, question);
                break;
            case HITLQuestionType.NumericInput:
                BuildNumericInputPrompt(sb, question);
                break;
            case HITLQuestionType.TextInput:
                BuildTextInputPrompt(sb, question);
                break;
            case HITLQuestionType.Confirmation:
                BuildConfirmationPrompt(sb, question);
                break;
        }

        // Recommendation if available
        if (!string.IsNullOrEmpty(question.RecommendedOptionKey))
        {
            sb.AppendLine();
            sb.AppendLine($"💡 Recommendation: {question.RecommendedOptionKey}");
            if (!string.IsNullOrEmpty(question.RecommendationReason))
            {
                sb.AppendLine($"   Reason: {question.RecommendationReason}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("─────────────────────────────────────────────────────────────────");

        return sb.ToString();
    }

    /// <summary>
    /// Builds a summary of all pending HITL questions.
    /// </summary>
    public string BuildQuestionSummary(List<HITLQuestion> questions)
    {
        var sb = new StringBuilder();

        sb.AppendLine();
        sb.AppendLine("╔══════════════════════════════════════════════════════════════╗");
        sb.AppendLine($"║  📋 Pending HITL Decisions: {questions.Count,-31} ║");
        sb.AppendLine("╚══════════════════════════════════════════════════════════════╝");
        sb.AppendLine();

        foreach (var question in questions.OrderByDescending(q => q.Priority))
        {
            var priority = question.Priority switch
            {
                >= 3 => "🔴 HIGH",
                2 => "🟡 MEDIUM",
                _ => "🟢 LOW"
            };

            sb.AppendLine($"  [{question.Id}] {priority} - {question.Question}");
            if (!string.IsNullOrEmpty(question.RelatedRuleId))
            {
                sb.AppendLine($"           Rule: {question.RelatedRuleId}");
            }
        }

        sb.AppendLine();
        return sb.ToString();
    }

    /// <summary>
    /// Builds a decision record summary.
    /// </summary>
    public string BuildDecisionSummary(List<HITLDecision> decisions)
    {
        var sb = new StringBuilder();

        sb.AppendLine();
        sb.AppendLine("╔══════════════════════════════════════════════════════════════╗");
        sb.AppendLine($"║  ✅ HITL Decisions Made: {decisions.Count,-35} ║");
        sb.AppendLine("╚══════════════════════════════════════════════════════════════╝");
        sb.AppendLine();

        foreach (var decision in decisions)
        {
            var answer = GetAnswerSummary(decision.Answer, decision.Question.Type);
            sb.AppendLine($"  [{decision.Question.Id}] {decision.Question.Question}");
            sb.AppendLine($"           → {answer}");
            if (!string.IsNullOrEmpty(decision.ResultingAction))
            {
                sb.AppendLine($"           Action: {decision.ResultingAction}");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Creates a HITL question for missing value strategy.
    /// </summary>
    public static HITLQuestion CreateMissingValueQuestion(
        string column,
        int missingCount,
        double percentage,
        string? recommendedStrategy = null)
    {
        return new HITLQuestion
        {
            Type = HITLQuestionType.MultipleChoice,
            Context = $"Column '{column}' has {missingCount} missing values ({percentage:P1} of records).",
            Question = $"How should missing values in '{column}' be handled?",
            Options =
            [
                new HITLOption { Key = "A", Label = "Fill with mean", Description = "Replace with column mean (numeric only)" },
                new HITLOption { Key = "B", Label = "Fill with median", Description = "Replace with column median (numeric only)" },
                new HITLOption { Key = "C", Label = "Fill with mode", Description = "Replace with most frequent value" },
                new HITLOption { Key = "D", Label = "Fill with constant", Description = "Replace with a specific value (e.g., 0, 'Unknown')" },
                new HITLOption { Key = "E", Label = "Remove rows", Description = "Delete rows with missing values" },
                new HITLOption { Key = "F", Label = "Keep as-is", Description = "Leave missing values unchanged" }
            ],
            RecommendedOptionKey = recommendedStrategy,
            RecommendationReason = recommendedStrategy != null
                ? $"Based on data distribution and column type"
                : null,
            Priority = percentage > 0.1 ? 3 : (percentage > 0.05 ? 2 : 1)
        };
    }

    /// <summary>
    /// Creates a HITL question for outlier handling.
    /// </summary>
    public static HITLQuestion CreateOutlierQuestion(
        string column,
        int outlierCount,
        double minValue,
        double maxValue)
    {
        return new HITLQuestion
        {
            Type = HITLQuestionType.MultipleChoice,
            Context = $"Column '{column}' has {outlierCount} outliers (range: [{minValue:F2}, {maxValue:F2}]).",
            Question = $"How should outliers in '{column}' be handled?",
            Options =
            [
                new HITLOption { Key = "A", Label = "Remove", Description = "Delete rows containing outliers" },
                new HITLOption { Key = "B", Label = "Cap/Winsorize", Description = "Cap values at specified percentiles (e.g., 1st/99th)" },
                new HITLOption { Key = "C", Label = "Transform", Description = "Apply log or square root transformation" },
                new HITLOption { Key = "D", Label = "Keep as-is", Description = "Preserve outliers (they may be valid extreme values)" }
            ],
            RecommendedOptionKey = "D",
            RecommendationReason = "Outliers may represent valid data points requiring domain review",
            Priority = 2
        };
    }

    /// <summary>
    /// Creates a HITL question for type inconsistency.
    /// </summary>
    public static HITLQuestion CreateTypeInconsistencyQuestion(
        string column,
        int numericCount,
        int textCount)
    {
        return new HITLQuestion
        {
            Type = HITLQuestionType.MultipleChoice,
            Context = $"Column '{column}' has mixed types: {numericCount} numeric, {textCount} text values.",
            Question = $"What should be the target type for '{column}'?",
            Options =
            [
                new HITLOption { Key = "A", Label = "Convert to numeric", Description = "Parse as numbers, convert text to NaN" },
                new HITLOption { Key = "B", Label = "Convert to text", Description = "Treat all values as categorical strings" },
                new HITLOption { Key = "C", Label = "Split column", Description = "Create separate columns for numeric and text values" },
                new HITLOption { Key = "D", Label = "Keep mixed", Description = "Preserve current mixed types" }
            ],
            RecommendedOptionKey = numericCount > textCount ? "A" : "B",
            RecommendationReason = $"Majority of values are {(numericCount > textCount ? "numeric" : "text")}",
            Priority = 3
        };
    }

    /// <summary>
    /// Creates a confirmation question for applying rules.
    /// </summary>
    public static HITLQuestion CreateBulkProcessingConfirmation(
        int ruleCount,
        int recordCount,
        double overallConfidence)
    {
        return new HITLQuestion
        {
            Type = HITLQuestionType.Confirmation,
            Context = $"Ready to apply {ruleCount} preprocessing rules to {recordCount:N0} records. " +
                      $"Overall confidence: {overallConfidence:P1}",
            Question = "Proceed with bulk processing?",
            Priority = 3
        };
    }

    private void BuildMultipleChoicePrompt(StringBuilder sb, HITLQuestion question)
    {
        sb.AppendLine("   Options:");
        foreach (var option in question.Options)
        {
            var recommended = option.Key == question.RecommendedOptionKey ? " ⭐" : "";
            sb.AppendLine($"   [{option.Key}] {option.Label}{recommended}");
            if (!string.IsNullOrEmpty(option.Description))
            {
                sb.AppendLine($"       {option.Description}");
            }
        }
        sb.AppendLine();
        sb.AppendLine($"   Enter your choice ({string.Join("/", question.Options.Select(o => o.Key))}): ");
    }

    private static void BuildYesNoPrompt(StringBuilder sb, HITLQuestion question)
    {
        sb.AppendLine("   [Y] Yes");
        sb.AppendLine("   [N] No");
        sb.AppendLine();
        sb.AppendLine("   Enter your choice (Y/N): ");
    }

    private static void BuildNumericInputPrompt(StringBuilder sb, HITLQuestion question)
    {
        sb.AppendLine("   Enter a numeric value: ");
    }

    private static void BuildTextInputPrompt(StringBuilder sb, HITLQuestion question)
    {
        sb.AppendLine("   Enter your value: ");
    }

    private static void BuildConfirmationPrompt(StringBuilder sb, HITLQuestion question)
    {
        sb.AppendLine("   [Y] Approve and proceed");
        sb.AppendLine("   [N] Cancel");
        sb.AppendLine();
        sb.AppendLine("   Enter your choice (Y/N): ");
    }

    private static string GetAnswerSummary(HITLAnswer answer, HITLQuestionType type)
    {
        return type switch
        {
            HITLQuestionType.MultipleChoice => $"Selected: {answer.SelectedOptionKey}",
            HITLQuestionType.YesNo => answer.BooleanValue == true ? "Yes" : "No",
            HITLQuestionType.NumericInput => $"Value: {answer.NumericValue}",
            HITLQuestionType.TextInput => $"Text: {answer.TextValue}",
            HITLQuestionType.Confirmation => answer.BooleanValue == true ? "Approved" : "Rejected",
            _ => "Unknown"
        };
    }
}
