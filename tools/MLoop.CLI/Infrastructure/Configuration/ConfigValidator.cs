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
/// time-series-anomaly) and carried the same F-17/F-19 bugs since fixed in the live path. Its only
/// unique coverage — required-field checks for forecasting/ranking/recommendation — was ported into
/// <c>ValidateCommand</c> (F-20), and the dead methods + their tests were removed (Cycle 51). Only the
/// genuinely live <see cref="ValidatePrepSteps"/> remains here.
/// </remarks>
public static class ConfigValidator
{
    private static readonly HashSet<string> ValidPrepStepTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "fill-missing", "fill_missing",
        "normalize", "scale",
        "remove-columns", "remove_columns",
        "rename-columns", "rename_columns",
        "drop-duplicates", "drop_duplicates",
        "extract-date", "extract_date",
        "parse-datetime", "parse_datetime",
        "filter-rows", "filter_rows",
        "add-column", "add_column",
        "parse-korean-time", "parse_korean_time",
        "parse-excel-date", "parse_excel_date",
        "rolling",
        "resample",
        "sample", "data-sampling", "data_sampling"
    };

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

            if (!ValidPrepStepTypes.Contains(step.Type))
            {
                errors.Add($"{prefix}: Unknown prep step type '{step.Type}'");
                continue;
            }

            var normalizedType = step.Type.ToLowerInvariant().Replace('_', '-');
            switch (normalizedType)
            {
                case "fill-missing":
                case "normalize":
                case "scale":
                case "remove-columns":
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: '{step.Type}' requires 'columns' or 'column'");
                    break;

                case "rename-columns":
                    if (step.Mapping == null || step.Mapping.Count == 0)
                        errors.Add($"{prefix}: 'rename-columns' requires 'mapping'");
                    break;

                case "extract-date":
                case "parse-korean-time":
                case "parse-excel-date":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: '{step.Type}' requires 'column'");
                    break;

                case "parse-datetime":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'parse-datetime' requires 'column'");
                    if (string.IsNullOrEmpty(step.Format))
                        errors.Add($"{prefix}: 'parse-datetime' requires 'format'");
                    break;

                case "filter-rows":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'filter-rows' requires 'column'");
                    if (string.IsNullOrEmpty(step.Operator))
                        errors.Add($"{prefix}: 'filter-rows' requires 'operator'");
                    if (string.IsNullOrEmpty(step.Value))
                        errors.Add($"{prefix}: 'filter-rows' requires 'value'");
                    break;

                case "add-column":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'add-column' requires 'column'");
                    if (string.IsNullOrEmpty(step.Value) && string.IsNullOrEmpty(step.Expression))
                        errors.Add($"{prefix}: 'add-column' requires 'value' or 'expression'");
                    break;

                case "rolling":
                    if (step.WindowSize < 1)
                        errors.Add($"{prefix}: 'rolling' requires 'window_size' > 0");
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'rolling' requires 'columns'");
                    break;

                case "resample":
                    if (string.IsNullOrEmpty(step.TimeColumn))
                        errors.Add($"{prefix}: 'resample' requires 'time_column'");
                    if (string.IsNullOrEmpty(step.Window))
                        errors.Add($"{prefix}: 'resample' requires 'window'");
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'resample' requires 'columns'");
                    break;

                case "sample":
                case "data-sampling":
                case "data_sampling":
                    if (step.Count <= 0)
                        errors.Add($"{prefix}: 'sample' requires 'count' > 0");
                    var sampleMethod = (step.Method ?? "random").ToLowerInvariant();
                    if (sampleMethod != "random" && sampleMethod != "stratified")
                        errors.Add($"{prefix}: 'sample' method must be 'random' or 'stratified'");
                    if (sampleMethod == "stratified" && string.IsNullOrEmpty(step.Column))
                        errors.Add($"{prefix}: 'sample' with stratified method requires 'column'");
                    break;
            }
        }

        return errors;
    }
}
