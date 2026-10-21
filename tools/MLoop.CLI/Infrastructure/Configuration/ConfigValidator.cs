using MLoop.Core.Preprocessing;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Prep-step structural validation, shared by <c>PrepRunCommand</c>.
/// </summary>
/// <remarks>
/// This class once also held a full <c>Validate(MLoopConfig)</c> / <c>ValidateLabelInCsv</c> config
/// validator "extracted for testability", but the production path (<c>mloop validate</c>) never called
/// it — <see cref="MLoop.CLI.Commands.ValidateCommand"/> is the live, filesystem-aware validator. The
/// orphaned copy had drifted (its task list lost the three NLP tasks; its unsupervised set lost
/// time-series-anomaly) and carried the same bugs since fixed in the live path. Its only
/// unique coverage — required-field checks for forecasting/ranking/recommendation — was ported into
/// <c>ValidateCommand</c>, and the dead methods + their tests were removed. Only the
/// genuinely live <see cref="ValidatePrepSteps"/> remains here.
/// </remarks>
public static class ConfigValidator
{
    /// <summary>
    /// Validates a list of prep steps and returns errors.
    /// </summary>
    public static List<string> ValidatePrepSteps(List<PrepStep> steps)
    {
        var errors = new List<string>();

        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];
            var prefix = $"prep[{i}]";

            if (string.IsNullOrWhiteSpace(step.Type))
            {
                errors.Add($"{prefix}: Step type is required");
                continue;
            }

            if (PrepStepRules.Canonical(step.Type) is null)
            {
                errors.Add($"{prefix}: {PrepStepRules.UnknownType(step.Type)}");
                continue;
            }

            errors.AddRange(PrepStepRules.Problems(step).Select(p => $"{prefix}: {p}"));
        }

        return errors;
    }
}
