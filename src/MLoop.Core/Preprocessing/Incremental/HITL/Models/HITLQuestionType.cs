namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Defines the type of question to ask the user during HITL interaction.
/// </summary>
public enum HITLQuestionType
{
    /// <summary>
    /// Multiple choice question with A, B, C, D options.
    /// </summary>
    MultipleChoice,

    /// <summary>
    /// Simple yes/no binary decision.
    /// </summary>
    YesNo,

    /// <summary>
    /// Numeric input (threshold, percentage, count).
    /// </summary>
    NumericInput,

    /// <summary>
    /// Text input for custom value or expression.
    /// </summary>
    TextInput,

    /// <summary>
    /// Confirmation to approve or reject with review.
    /// </summary>
    Confirmation
}
