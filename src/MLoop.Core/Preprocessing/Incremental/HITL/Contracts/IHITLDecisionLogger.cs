using MLoop.Core.Preprocessing.Incremental.HITL.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL.Contracts;

/// <summary>
/// Logs HITL decisions for audit trail and compliance.
/// </summary>
public interface IHITLDecisionLogger
{
    /// <summary>
    /// Logs a single HITL decision with question, answer, and rule context.
    /// </summary>
    /// <param name="log">The decision log to record.</param>
    /// <returns>Task representing the async operation.</returns>
    Task LogDecisionAsync(HITLDecisionLog log);

    /// <summary>
    /// Retrieves all decision logs for a specific rule.
    /// </summary>
    /// <param name="ruleId">The rule identifier.</param>
    /// <returns>List of decision logs for the rule.</returns>
    Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByRuleAsync(string ruleId);

    /// <summary>
    /// Retrieves all decisions made within a time range.
    /// </summary>
    /// <param name="startTime">Start of the time range.</param>
    /// <param name="endTime">End of the time range.</param>
    /// <returns>List of decision logs within the range.</returns>
    Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByTimeRangeAsync(
        DateTime startTime,
        DateTime endTime);

    /// <summary>
    /// Gets a summary of all decisions for reporting.
    /// </summary>
    /// <returns>Decision summary statistics.</returns>
    Task<HITLDecisionSummary> GetDecisionSummaryAsync();
}
