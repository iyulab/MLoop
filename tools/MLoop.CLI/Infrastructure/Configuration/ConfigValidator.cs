using MLoop.Core.Preprocessing;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Validates MLoop configuration for correctness.
/// Extracted from ValidateCommand for testability.
/// </summary>
public static class ConfigValidator
{
    private static readonly HashSet<string> ValidTaskTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "regression",
        "binary-classification",
        "multiclass-classification"
    };

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
        "resample"
    };

    public record ValidationResult(List<string> Errors, List<string> Warnings);

    /// <summary>
    /// Validates an MLoopConfig and returns all errors and warnings.
    /// </summary>
    public static ValidationResult Validate(MLoopConfig config)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        if (string.IsNullOrWhiteSpace(config.Project))
            warnings.Add("Project name is not specified");

        if (config.Models == null || config.Models.Count == 0)
        {
            errors.Add("No models defined. At least one model is required.");
            return new ValidationResult(errors, warnings);
        }

        foreach (var (modelName, modelDef) in config.Models)
        {
            ValidateModel(modelName, modelDef, errors, warnings);
        }

        return new ValidationResult(errors, warnings);
    }

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
            }
        }

        return errors;
    }

    /// <summary>
    /// Validates label columns exist in CSV headers.
    /// </summary>
    public static List<string> ValidateLabelInCsv(
        string[] csvHeaders,
        Dictionary<string, ModelDefinition> models)
    {
        var errors = new List<string>();
        var headerSet = new HashSet<string>(csvHeaders, StringComparer.OrdinalIgnoreCase);

        foreach (var (modelName, modelDef) in models)
        {
            if (!string.IsNullOrWhiteSpace(modelDef.Label) && !headerSet.Contains(modelDef.Label))
            {
                errors.Add(
                    $"models.{modelName}.label: Column '{modelDef.Label}' not found in CSV. " +
                    $"Available: {string.Join(", ", csvHeaders.Take(10))}{(csvHeaders.Length > 10 ? "..." : "")}");
            }
        }

        return errors;
    }

    private static void ValidateModel(
        string modelName,
        ModelDefinition model,
        List<string> errors,
        List<string> warnings)
    {
        if (string.IsNullOrWhiteSpace(model.Task))
            errors.Add($"models.{modelName}.task: Task type is required");
        else if (!ValidTaskTypes.Contains(model.Task))
            errors.Add($"models.{modelName}.task: Invalid task type '{model.Task}'");

        if (string.IsNullOrWhiteSpace(model.Label))
            errors.Add($"models.{modelName}.label: Label column is required");

        if (model.Prep is { Count: > 0 })
        {
            var prepErrors = ValidatePrepSteps(model.Prep);
            foreach (var err in prepErrors)
            {
                errors.Add($"models.{modelName}.{err}");
            }
        }
    }
}
