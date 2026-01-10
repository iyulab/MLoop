using MLoop.Core.Preprocessing.Incremental.HITL.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL.Contracts;

/// <summary>
/// Builds interactive prompts and collects user responses for HITL questions.
/// </summary>
public interface IHITLPromptBuilder
{
    /// <summary>
    /// Builds and displays a formatted prompt for a HITL question.
    /// </summary>
    /// <param name="question">The question to display.</param>
    /// <returns>The question ID for tracking.</returns>
    string BuildPrompt(HITLQuestion question);

    /// <summary>
    /// Collects the user's answer to a HITL question.
    /// </summary>
    /// <param name="question">The question to collect answer for.</param>
    /// <returns>The user's answer with validation.</returns>
    HITLAnswer CollectAnswer(HITLQuestion question);

    /// <summary>
    /// Displays a confirmation message for a recorded answer.
    /// </summary>
    /// <param name="answer">The answer to confirm.</param>
    void DisplayConfirmation(HITLAnswer answer);
}
