using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL.Contracts;

/// <summary>
/// Generates HITL questions from preprocessing rules.
/// </summary>
public interface IHITLQuestionGenerator
{
    /// <summary>
    /// Generates a HITL question for a single rule.
    /// </summary>
    /// <param name="rule">The preprocessing rule requiring HITL.</param>
    /// <param name="sample">The data sample being analyzed.</param>
    /// <param name="analysis">Analysis results for the sample.</param>
    /// <returns>A HITL question with context and options.</returns>
    HITLQuestion GenerateQuestion(
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis);

    /// <summary>
    /// Generates HITL questions for all rules requiring human input.
    /// </summary>
    /// <param name="rules">List of preprocessing rules.</param>
    /// <param name="sample">The data sample being analyzed.</param>
    /// <param name="analysis">Analysis results for the sample.</param>
    /// <returns>List of HITL questions.</returns>
    IReadOnlyList<HITLQuestion> GenerateAllQuestions(
        IReadOnlyList<PreprocessingRule> rules,
        DataFrame sample,
        SampleAnalysis analysis);
}
