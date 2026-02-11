using System.CommandLine;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop validate - Validate project configuration
/// </summary>
public static class ValidateCommand
{
    private static readonly HashSet<string> ValidTaskTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "regression",
        "binary-classification",
        "multiclass-classification"
    };

    private static readonly HashSet<string> ValidMetrics = new(StringComparer.OrdinalIgnoreCase)
    {
        "auto",
        "accuracy", "auc", "f1", "recall", "precision", "log-loss",
        "r2", "rmse", "mae", "mse", "rSquared"
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

    public static Command Create()
    {
        var verboseOption = new Option<bool>("--verbose", "-v")
        {
            Description = "Show detailed validation results",
            DefaultValueFactory = _ => false
        };

        var command = new Command("validate", "Validate project configuration (mloop.yaml)");
        command.Options.Add(verboseOption);

        command.SetAction((parseResult) =>
        {
            var verbose = parseResult.GetValue(verboseOption);
            return ExecuteAsync(verbose);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(bool verbose)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var errors = new List<ValidationError>();
            var warnings = new List<ValidationWarning>();

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold]MLoop Configuration Validation[/]").LeftJustified());
            AnsiConsole.WriteLine();

            // Check mloop.yaml exists
            var yamlPath = ctx.FileSystem.CombinePath(ctx.ProjectRoot, "mloop.yaml");
            if (!ctx.FileSystem.FileExists(yamlPath))
            {
                errors.Add(new ValidationError("mloop.yaml", "Configuration file not found"));
                DisplayResults(errors, warnings, verbose);
                return 1;
            }

            if (verbose)
            {
                AnsiConsole.MarkupLine($"[grey]Config file: {yamlPath}[/]");
                AnsiConsole.WriteLine();
            }

            // Load and validate config
            MLoopConfig config;
            try
            {
                config = await ctx.ConfigLoader.LoadUserConfigAsync();
            }
            catch (Exception ex)
            {
                errors.Add(new ValidationError("mloop.yaml", $"Failed to parse YAML: {ex.Message}"));
                DisplayResults(errors, warnings, verbose);
                return 1;
            }

            // Validate project name
            if (string.IsNullOrWhiteSpace(config.Project))
            {
                warnings.Add(new ValidationWarning("project", "Project name is not specified"));
            }

            // Validate models section
            if (config.Models == null || config.Models.Count == 0)
            {
                errors.Add(new ValidationError("models", "No models defined. At least one model is required."));
            }
            else
            {
                foreach (var (modelName, modelDef) in config.Models)
                {
                    ValidateModel(modelName, modelDef, errors, warnings);
                }
            }

            // Validate data paths and label column
            if (config.Data != null)
            {
                ValidateDataPaths(config.Data, ctx.ProjectRoot, ctx.FileSystem, errors, warnings);

                // Validate label columns exist in training data
                if (!string.IsNullOrWhiteSpace(config.Data.Train) && config.Models != null)
                {
                    var trainPath = ctx.FileSystem.CombinePath(ctx.ProjectRoot, config.Data.Train);
                    if (ctx.FileSystem.FileExists(trainPath))
                    {
                        ValidateLabelColumnsInCsv(trainPath, config.Models, errors, warnings);
                    }
                }
            }

            // Display results
            DisplayResults(errors, warnings, verbose);

            // Return appropriate exit code
            if (errors.Count > 0)
            {
                AnsiConsole.MarkupLine("[red]Validation failed.[/] Please fix the errors above.");
                return 1;
            }

            if (warnings.Count > 0)
            {
                AnsiConsole.MarkupLine("[yellow]Validation passed with warnings.[/]");
                return 0;
            }

            AnsiConsole.MarkupLine("[green]Validation successful![/] Configuration is valid.");
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);
            return 1;
        }
    }

    private static void ValidateModel(
        string modelName,
        ModelDefinition model,
        List<ValidationError> errors,
        List<ValidationWarning> warnings)
    {
        var prefix = $"models.{modelName}";

        // Validate model name
        if (string.IsNullOrWhiteSpace(modelName))
        {
            errors.Add(new ValidationError(prefix, "Model name cannot be empty"));
            return;
        }

        if (!IsValidModelName(modelName))
        {
            errors.Add(new ValidationError(prefix, "Model name must be lowercase alphanumeric with hyphens only"));
        }

        // Validate task type
        if (string.IsNullOrWhiteSpace(model.Task))
        {
            errors.Add(new ValidationError($"{prefix}.task", "Task type is required"));
        }
        else if (!ValidTaskTypes.Contains(model.Task))
        {
            errors.Add(new ValidationError($"{prefix}.task",
                $"Invalid task type '{model.Task}'. Valid values: {string.Join(", ", ValidTaskTypes)}"));
        }

        // Validate label
        if (string.IsNullOrWhiteSpace(model.Label))
        {
            errors.Add(new ValidationError($"{prefix}.label", "Label column is required"));
        }

        // Validate training settings
        if (model.Training != null)
        {
            ValidateTrainingSettings($"{prefix}.training", model.Training, errors, warnings);
        }

        // Validate prep steps
        if (model.Prep is { Count: > 0 })
        {
            ValidatePrepSteps($"{prefix}.prep", model.Prep, errors, warnings);
        }
    }

    private static void ValidateTrainingSettings(
        string prefix,
        TrainingSettings training,
        List<ValidationError> errors,
        List<ValidationWarning> warnings)
    {
        // Validate time limit
        if (training.TimeLimitSeconds.HasValue)
        {
            if (training.TimeLimitSeconds.Value <= 0)
            {
                errors.Add(new ValidationError($"{prefix}.time_limit_seconds",
                    "Time limit must be greater than 0"));
            }
            else if (training.TimeLimitSeconds.Value < 30)
            {
                warnings.Add(new ValidationWarning($"{prefix}.time_limit_seconds",
                    "Time limit less than 30 seconds may not produce good results"));
            }
            else if (training.TimeLimitSeconds.Value > 3600)
            {
                warnings.Add(new ValidationWarning($"{prefix}.time_limit_seconds",
                    "Time limit over 1 hour may be unnecessarily long for most datasets"));
            }
        }

        // Validate metric
        if (!string.IsNullOrWhiteSpace(training.Metric) && !ValidMetrics.Contains(training.Metric))
        {
            warnings.Add(new ValidationWarning($"{prefix}.metric",
                $"Unknown metric '{training.Metric}'. Common values: {string.Join(", ", ValidMetrics.Take(6))}"));
        }

        // Validate test split
        if (training.TestSplit.HasValue)
        {
            if (training.TestSplit.Value <= 0 || training.TestSplit.Value >= 1)
            {
                errors.Add(new ValidationError($"{prefix}.test_split",
                    "Test split must be between 0 and 1 (exclusive)"));
            }
            else if (training.TestSplit.Value < 0.1)
            {
                warnings.Add(new ValidationWarning($"{prefix}.test_split",
                    "Test split less than 0.1 may not provide reliable evaluation"));
            }
            else if (training.TestSplit.Value > 0.5)
            {
                warnings.Add(new ValidationWarning($"{prefix}.test_split",
                    "Test split greater than 0.5 leaves less data for training"));
            }
        }
    }

    private static void ValidateDataPaths(
        DataSettings data,
        string projectRoot,
        IFileSystemManager fileSystem,
        List<ValidationError> errors,
        List<ValidationWarning> warnings)
    {
        if (!string.IsNullOrWhiteSpace(data.Train))
        {
            var trainPath = fileSystem.CombinePath(projectRoot, data.Train);
            if (!fileSystem.FileExists(trainPath))
            {
                warnings.Add(new ValidationWarning("data.train",
                    $"Training data file not found: {data.Train}"));
            }
        }

        if (!string.IsNullOrWhiteSpace(data.Test))
        {
            var testPath = fileSystem.CombinePath(projectRoot, data.Test);
            if (!fileSystem.FileExists(testPath))
            {
                warnings.Add(new ValidationWarning("data.test",
                    $"Test data file not found: {data.Test}"));
            }
        }

        if (!string.IsNullOrWhiteSpace(data.Predict))
        {
            var predictPath = fileSystem.CombinePath(projectRoot, data.Predict);
            if (!fileSystem.FileExists(predictPath))
            {
                warnings.Add(new ValidationWarning("data.predict",
                    $"Prediction data file not found: {data.Predict}"));
            }
        }
    }

    private static void ValidatePrepSteps(
        string prefix,
        List<MLoop.Core.Preprocessing.PrepStep> steps,
        List<ValidationError> errors,
        List<ValidationWarning> warnings)
    {
        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];
            var stepPrefix = $"{prefix}[{i}]";

            // Validate step type
            if (string.IsNullOrWhiteSpace(step.Type))
            {
                errors.Add(new ValidationError(stepPrefix, "Step type is required"));
                continue;
            }

            if (!ValidPrepStepTypes.Contains(step.Type))
            {
                errors.Add(new ValidationError($"{stepPrefix}.type",
                    $"Unknown prep step type '{step.Type}'. Valid types: fill-missing, normalize, remove-columns, rename-columns, drop-duplicates, extract-date, parse-datetime, filter-rows, add-column, parse-korean-time, parse-excel-date, rolling, resample"));
                continue;
            }

            // Validate required parameters per step type
            var normalizedType = step.Type.ToLowerInvariant().Replace('_', '-');
            switch (normalizedType)
            {
                case "fill-missing":
                case "normalize":
                case "scale":
                case "remove-columns":
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                    {
                        errors.Add(new ValidationError(stepPrefix,
                            $"'{step.Type}' requires 'columns' or 'column' parameter"));
                    }
                    break;

                case "rename-columns":
                    if (step.Mapping == null || step.Mapping.Count == 0)
                    {
                        errors.Add(new ValidationError(stepPrefix,
                            "'rename-columns' requires 'mapping' parameter"));
                    }
                    break;

                case "extract-date":
                case "parse-datetime":
                case "parse-korean-time":
                case "parse-excel-date":
                    if (string.IsNullOrEmpty(step.Column))
                    {
                        errors.Add(new ValidationError(stepPrefix,
                            $"'{step.Type}' requires 'column' parameter"));
                    }
                    if (normalizedType == "parse-datetime" && string.IsNullOrEmpty(step.Format))
                    {
                        errors.Add(new ValidationError(stepPrefix,
                            "'parse-datetime' requires 'format' parameter"));
                    }
                    break;

                case "filter-rows":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add(new ValidationError(stepPrefix, "'filter-rows' requires 'column' parameter"));
                    if (string.IsNullOrEmpty(step.Operator))
                        errors.Add(new ValidationError(stepPrefix, "'filter-rows' requires 'operator' parameter"));
                    if (string.IsNullOrEmpty(step.Value))
                        errors.Add(new ValidationError(stepPrefix, "'filter-rows' requires 'value' parameter"));
                    break;

                case "add-column":
                    if (string.IsNullOrEmpty(step.Column))
                        errors.Add(new ValidationError(stepPrefix, "'add-column' requires 'column' parameter"));
                    if (string.IsNullOrEmpty(step.Value) && string.IsNullOrEmpty(step.Expression))
                        errors.Add(new ValidationError(stepPrefix,
                            "'add-column' requires 'value' or 'expression' parameter"));
                    break;

                case "rolling":
                    if (step.WindowSize < 1)
                        errors.Add(new ValidationError(stepPrefix, "'rolling' requires 'window_size' > 0"));
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                        errors.Add(new ValidationError(stepPrefix, "'rolling' requires 'columns' parameter"));
                    break;

                case "resample":
                    if (string.IsNullOrEmpty(step.TimeColumn))
                        errors.Add(new ValidationError(stepPrefix, "'resample' requires 'time_column' parameter"));
                    if (string.IsNullOrEmpty(step.Window))
                        errors.Add(new ValidationError(stepPrefix, "'resample' requires 'window' parameter"));
                    if ((step.Columns == null || step.Columns.Count == 0) && string.IsNullOrEmpty(step.Column))
                        errors.Add(new ValidationError(stepPrefix, "'resample' requires 'columns' parameter"));
                    break;
            }
        }
    }

    private static void ValidateLabelColumnsInCsv(
        string csvPath,
        Dictionary<string, ModelDefinition> models,
        List<ValidationError> errors,
        List<ValidationWarning> warnings)
    {
        try
        {
            var csvConfig = new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                MissingFieldFound = null,
                BadDataFound = null
            };

            using var reader = new StreamReader(csvPath);
            using var csv = new CsvReader(reader, csvConfig);
            csv.Read();
            csv.ReadHeader();
            var headers = csv.HeaderRecord;

            if (headers == null || headers.Length == 0)
            {
                warnings.Add(new ValidationWarning("data.train", "Training CSV has no header row"));
                return;
            }

            var headerSet = new HashSet<string>(headers, StringComparer.OrdinalIgnoreCase);

            foreach (var (modelName, modelDef) in models)
            {
                if (!string.IsNullOrWhiteSpace(modelDef.Label) && !headerSet.Contains(modelDef.Label))
                {
                    errors.Add(new ValidationError($"models.{modelName}.label",
                        $"Label column '{modelDef.Label}' not found in training CSV. Available columns: {string.Join(", ", headers.Take(10))}{(headers.Length > 10 ? "..." : "")}"));
                }
            }
        }
        catch (Exception ex)
        {
            warnings.Add(new ValidationWarning("data.train",
                $"Could not read training CSV headers: {ex.Message}"));
        }
    }

    private static bool IsValidModelName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        foreach (var c in name)
        {
            if (!char.IsLetterOrDigit(c) && c != '-' && c != '_')
                return false;
        }

        return char.IsLetter(name[0]) || name[0] == '_';
    }

    private static void DisplayResults(
        List<ValidationError> errors,
        List<ValidationWarning> warnings,
        bool verbose)
    {
        if (errors.Count > 0)
        {
            AnsiConsole.MarkupLine($"[red bold]Errors ({errors.Count}):[/]");
            foreach (var error in errors)
            {
                AnsiConsole.MarkupLine($"  [red]✗[/] [cyan]{error.Path}[/]: {error.Message}");
            }
            AnsiConsole.WriteLine();
        }

        if (warnings.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow bold]Warnings ({warnings.Count}):[/]");
            foreach (var warning in warnings)
            {
                AnsiConsole.MarkupLine($"  [yellow]⚠[/] [cyan]{warning.Path}[/]: {warning.Message}");
            }
            AnsiConsole.WriteLine();
        }

        if (verbose && errors.Count == 0 && warnings.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No issues found.[/]");
            AnsiConsole.WriteLine();
        }

        // Summary
        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .AddColumn("Check")
            .AddColumn("Status");

        table.AddRow("Config file exists", "[green]✓[/]");
        table.AddRow("YAML syntax valid", errors.Any(e => e.Path == "mloop.yaml") ? "[red]✗[/]" : "[green]✓[/]");
        table.AddRow("Models defined", errors.Any(e => e.Path == "models") ? "[red]✗[/]" : "[green]✓[/]");
        table.AddRow("Task types valid", errors.Any(e => e.Path.Contains(".task")) ? "[red]✗[/]" : "[green]✓[/]");
        table.AddRow("Labels specified", errors.Any(e => e.Path.Contains(".label")) ? "[red]✗[/]" : "[green]✓[/]");
        table.AddRow("Training settings", errors.Any(e => e.Path.Contains(".training")) ? "[red]✗[/]" :
            warnings.Any(w => w.Path.Contains(".training")) ? "[yellow]⚠[/]" : "[green]✓[/]");
        table.AddRow("Data paths", warnings.Any(w => w.Path.StartsWith("data.")) ? "[yellow]⚠[/]" : "[green]✓[/]");
        table.AddRow("Prep steps", errors.Any(e => e.Path.Contains(".prep")) ? "[red]✗[/]" :
            warnings.Any(w => w.Path.Contains(".prep")) ? "[yellow]⚠[/]" : "[green]✓[/]");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private record ValidationError(string Path, string Message);
    private record ValidationWarning(string Path, string Message);
}
