namespace MLoop.AIAgent.Core.HITL;

/// <summary>
/// Type of HITL question.
/// </summary>
public enum HITLQuestionType
{
    /// <summary>Select one from multiple options (A, B, C, D).</summary>
    MultipleChoice,

    /// <summary>Simple yes/no decision.</summary>
    YesNo,

    /// <summary>Input a numeric value (threshold, percentage).</summary>
    NumericInput,

    /// <summary>Input custom text value.</summary>
    TextInput,

    /// <summary>Approve or reject a proposal.</summary>
    Confirmation
}

/// <summary>
/// A single option in a HITL question.
/// </summary>
public class HITLOption
{
    /// <summary>Option key (A, B, C, D or 1, 2, 3, 4).</summary>
    public required string Key { get; set; }

    /// <summary>Display label for the option.</summary>
    public required string Label { get; set; }

    /// <summary>Detailed description of what this option does.</summary>
    public string? Description { get; set; }

    /// <summary>Whether this is the recommended option.</summary>
    public bool IsRecommended { get; set; }
}

/// <summary>
/// A question requiring human decision.
/// </summary>
public class HITLQuestion
{
    /// <summary>Unique identifier for this question.</summary>
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];

    /// <summary>Type of question.</summary>
    public HITLQuestionType Type { get; set; }

    /// <summary>Context explaining what the agent found.</summary>
    public required string Context { get; set; }

    /// <summary>The actual question to ask.</summary>
    public required string Question { get; set; }

    /// <summary>Available options (for MultipleChoice).</summary>
    public List<HITLOption> Options { get; set; } = [];

    /// <summary>Key of the recommended option.</summary>
    public string? RecommendedOptionKey { get; set; }

    /// <summary>Reason for the recommendation.</summary>
    public string? RecommendationReason { get; set; }

    /// <summary>Related rule ID if applicable.</summary>
    public string? RelatedRuleId { get; set; }

    /// <summary>Priority/urgency of this question.</summary>
    public int Priority { get; set; } = 1;

    /// <summary>Timestamp when question was created.</summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// User's answer to a HITL question.
/// </summary>
public class HITLAnswer
{
    /// <summary>ID of the question being answered.</summary>
    public required string QuestionId { get; set; }

    /// <summary>Selected option key (for MultipleChoice).</summary>
    public string? SelectedOptionKey { get; set; }

    /// <summary>Text input value (for TextInput).</summary>
    public string? TextValue { get; set; }

    /// <summary>Numeric input value (for NumericInput).</summary>
    public double? NumericValue { get; set; }

    /// <summary>Boolean value (for YesNo/Confirmation).</summary>
    public bool? BooleanValue { get; set; }

    /// <summary>Additional notes from user.</summary>
    public string? Notes { get; set; }

    /// <summary>User identifier.</summary>
    public string? AnsweredBy { get; set; }

    /// <summary>Timestamp when answered.</summary>
    public DateTime AnsweredAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Record of a HITL decision with full context.
/// </summary>
public class HITLDecision
{
    /// <summary>The question that was asked.</summary>
    public required HITLQuestion Question { get; set; }

    /// <summary>The answer provided.</summary>
    public required HITLAnswer Answer { get; set; }

    /// <summary>The resulting action taken.</summary>
    public string? ResultingAction { get; set; }

    /// <summary>Related preprocessing rule.</summary>
    public string? RuleId { get; set; }
}
