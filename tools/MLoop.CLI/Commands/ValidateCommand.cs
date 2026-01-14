using System.CommandLine;
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

            // Validate data paths
            if (config.Data != null)
            {
                ValidateDataPaths(config.Data, ctx.ProjectRoot, ctx.FileSystem, errors, warnings);
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

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private record ValidationError(string Path, string Message);
    private record ValidationWarning(string Path, string Message);
}
