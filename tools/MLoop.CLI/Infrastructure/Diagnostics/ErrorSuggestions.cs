using System.Reflection;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// Provides actionable error suggestions based on exception types and messages.
/// Helps users understand and resolve common issues quickly.
/// </summary>
public static class ErrorSuggestions
{
    /// <summary>
    /// Displays an error with actionable suggestions based on context.
    /// </summary>
    public static void DisplayError(Exception ex, string context = "")
    {
        // stderr, not stdout: every command's top-level catch funnels here before returning a
        // non-zero exit code, so this one routing decision is what makes "exit != 0 ⇒ stderr has a
        // cause" true CLI-wide. See ErrorConsole for the full rationale.
        var err = ErrorConsole.Out;

        err.Markup("[red]Error:[/] ");
        err.WriteLine(ex.Message);

        if (ex.InnerException != null && AddsInformation(ex.Message, ex.InnerException.Message))
        {
            err.MarkupLine($"[grey]  Inner: {Markup.Escape(ex.InnerException.Message)}[/]");
        }

        // Get and display suggestions
        var suggestions = GetSuggestions(ex, context);
        if (suggestions.Count > 0)
        {
            err.WriteLine();
            err.MarkupLine("[yellow]Suggestions:[/]");
            foreach (var suggestion in suggestions)
            {
                err.MarkupLine($"  [blue]>[/] {suggestion}");
            }
        }

        // Always show version for diagnostics
        err.WriteLine();
        err.MarkupLine($"[grey]mloop v{GetVersion()}[/]");
    }

    /// <summary>
    /// Whether the inner message is worth printing, i.e. the outer message does not already carry it.
    /// </summary>
    /// <remarks>
    /// Wrapping sites commonly build the outer message as "{context}: {inner.Message}" (see
    /// TrainingEngine's "Training failed for experiment {id}: …"). Each layer is reasonable alone, but
    /// together they print the same diagnosis twice — harmless when the message was one line, glaring
    /// once diagnostics grew to a class-distribution table plus remediation options. The display layer
    /// is the right place to resolve it: its job is to add information, and repeating text the reader
    /// has already seen adds none.
    /// </remarks>
    internal static bool AddsInformation(string outerMessage, string innerMessage)
    {
        var inner = innerMessage?.Trim();
        if (string.IsNullOrEmpty(inner)) return false;
        return !(outerMessage ?? string.Empty).Contains(inner, StringComparison.Ordinal);
    }

    /// <summary>
    /// Gets actionable suggestions for the given exception.
    /// </summary>
    public static List<string> GetSuggestions(Exception ex, string context = "")
    {
        var suggestions = new List<string>();
        var message = ex.Message.ToLowerInvariant();
        var typeName = ex.GetType().Name;

        // File and path errors
        if (ex is FileNotFoundException || message.Contains("file not found") || message.Contains("could not find file"))
        {
            suggestions.Add("Check if the file path is correct");
            suggestions.Add("Ensure the file exists and is not locked by another process");
            if (context == "training")
            {
                suggestions.Add("Create datasets/train.csv or specify a file: [cyan]mloop train <data-file>[/]");
            }
        }

        // Directory errors
        if (ex is DirectoryNotFoundException || message.Contains("directory not found"))
        {
            suggestions.Add("Verify the directory path exists");
            suggestions.Add("Create the directory structure if needed");
        }

        // Permission errors
        if (ex is UnauthorizedAccessException || message.Contains("access denied") || message.Contains("permission"))
        {
            suggestions.Add("Check file/folder permissions");
            suggestions.Add("Ensure no other application is using the file");
            suggestions.Add("Try running as administrator (Windows) or with sudo (Linux/Mac)");
        }

        // ML.NET specific errors
        if (message.Contains("feature vector dimension mismatch") || message.Contains("onehotencoding"))
        {
            suggestions.Add("Prediction/test data columns don't match the model's training schema");
            suggestions.Add("Check for missing, renamed, or extra feature columns versus the training data");
            suggestions.Add("Ensure each column's type matches training (e.g. numeric vs. text)");
            suggestions.Add("Try evaluating with a test split from the same dataset used for training");
        }
        else if (message.Contains("schema") || message.Contains("column"))
        {
            suggestions.Add("Verify the data file has the expected columns");
            suggestions.Add("Check if the label column name matches (case-sensitive)");
            suggestions.Add("Run [cyan]mloop analyze[/] to inspect your data schema");
        }

        if (message.Contains("label") && (message.Contains("not found") || message.Contains("missing")))
        {
            suggestions.Add("Update the label in mloop.yaml: [cyan]label: YourLabelColumnName[/]");
            suggestions.Add("Or specify via CLI: [cyan]mloop train --label YourLabelColumnName[/]");
        }

        if (message.Contains("task") && (message.Contains("invalid") || message.Contains("not supported")))
        {
            suggestions.Add("Valid tasks: [cyan]binary-classification[/], [cyan]multiclass-classification[/], [cyan]regression[/]");
            suggestions.Add("Update in mloop.yaml or use: [cyan]mloop train --task <task-type>[/]");
        }

        // Memory and resource errors
        if (ex is OutOfMemoryException || message.Contains("out of memory") || message.Contains("memory"))
        {
            suggestions.Add("Try reducing the dataset size with sampling");
            suggestions.Add("Close other applications to free memory");
            suggestions.Add("Consider using incremental training or data chunking");
        }

        // Timeout errors
        if (message.Contains("timeout") || message.Contains("timed out"))
        {
            suggestions.Add("Increase the time limit: [cyan]--time <seconds>[/]");
            suggestions.Add("Try a smaller dataset for faster iteration");
        }

        // Data format errors
        if (message.Contains("csv") || message.Contains("parse") || message.Contains("format"))
        {
            suggestions.Add("Verify your CSV file is properly formatted");
            suggestions.Add("Check for special characters or encoding issues (use UTF-8)");
            suggestions.Add("Ensure consistent column separators (comma)");
        }

        // Model errors
        if (message.Contains("model") && (message.Contains("not found") || message.Contains("load")))
        {
            suggestions.Add("Run [cyan]mloop experiments list[/] to see available models");
            suggestions.Add("Train a model first: [cyan]mloop train[/]");
            suggestions.Add("Promote an experiment: [cyan]mloop experiments promote <experiment-id>[/]");
        }

        // Configuration errors
        if (message.Contains("config") || message.Contains("yaml") || message.Contains("mloop.yaml"))
        {
            suggestions.Add("Check mloop.yaml syntax (YAML is space-sensitive)");
            suggestions.Add("Run [cyan]mloop init[/] to create a default configuration");
            suggestions.Add("Validate your YAML at https://www.yamllint.com/");
        }

        // Network errors
        if (message.Contains("network") || message.Contains("connection") || message.Contains("socket"))
        {
            suggestions.Add("Check your network connection");
            suggestions.Add("Verify firewall settings");
            suggestions.Add("Retry the operation after a moment");
        }

        // Training-specific context
        if (context == "training")
        {
            if (!suggestions.Any())
            {
                // Generic training suggestions if no specific match
                suggestions.Add("Run [cyan]mloop analyze <data-file>[/] to check data quality");
                suggestions.Add("Try with a smaller time limit first: [cyan]mloop train --time 30[/]");
                suggestions.Add("Check the GUIDE.md for common solutions");
            }
        }

        // Prediction-specific context
        if (context == "prediction")
        {
            if (message.Contains("schema") || message.Contains("column"))
            {
                suggestions.Add("Ensure prediction data has the same columns as training data");
                suggestions.Add("The label column can be empty but must exist in the schema");
            }
        }

        // Promote-specific context
        if (context == "promote")
        {
            if (message.Contains("not found") || message.Contains("does not exist"))
            {
                suggestions.Add("Run [cyan]mloop list --all[/] to see available experiments");
                suggestions.Add("Verify the experiment ID is correct");
            }
        }

        // Validate-specific context
        if (context == "validate")
        {
            if (!suggestions.Any())
            {
                suggestions.Add("Check mloop.yaml syntax: [cyan]mloop validate[/]");
                suggestions.Add("Validate YAML format at https://www.yamllint.com/");
            }
        }

        // Preprocessing-specific context
        if (context == "preprocessing")
        {
            if (!suggestions.Any())
            {
                suggestions.Add("Check prep step configuration in mloop.yaml");
                suggestions.Add("Run [cyan]mloop prep run --dry-run[/] to preview pipeline");
                suggestions.Add("Verify input CSV exists and is well-formatted");
            }
        }

        // Evaluate-specific context
        if (context == "evaluate")
        {
            if (!suggestions.Any())
            {
                suggestions.Add("Ensure a production model exists: [cyan]mloop list[/]");
                suggestions.Add("Check test data has same schema as training data");
                suggestions.Add("Create datasets/test.csv or specify: [cyan]mloop evaluate <exp-id> <test-file>[/]");
            }
        }

        // Serve-specific context
        if (context == "serve")
        {
            if (!suggestions.Any())
            {
                suggestions.Add("Ensure MLoop.API is available (check MLOOP_API_PATH env var)");
                suggestions.Add("Try specifying a different port: [cyan]mloop serve --port 5001[/]");
                suggestions.Add("Check if another process is using the port");
            }
        }

        // Update-specific context
        if (context == "update")
        {
            if (!suggestions.Any())
            {
                suggestions.Add("Check your internet connection");
                suggestions.Add("Download manually from https://github.com/iyulab/MLoop/releases");
            }
        }

        // Init-specific context
        if (context == "init")
        {
            if (message.Contains("already exists") || message.Contains("directory"))
            {
                suggestions.Add("Choose a different project name");
                suggestions.Add("Delete the existing directory first if you want to start fresh");
            }
        }

        // Fallback generic suggestions
        if (!suggestions.Any())
        {
            suggestions.Add("Review the full error message above for details");
            suggestions.Add("Check the GUIDE.md or README.md for documentation");
            suggestions.Add("Report issues at: https://github.com/iyulab/MLoop/issues");
        }

        return suggestions;
    }

    /// <summary>
    /// Gets the current mloop CLI version string (semver without commit hash).
    /// </summary>
    private static string GetVersion()
    {
        var attr = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>();

        if (attr is null) return "unknown";

        var version = attr.InformationalVersion;
        var plusIndex = version.IndexOf('+');
        return plusIndex >= 0 ? version[..plusIndex] : version;
    }

    /// <summary>
    /// Displays a training-specific error with enhanced diagnostics.
    /// </summary>
    public static void DisplayTrainingError(Exception ex, string modelName, string? dataFile = null)
    {
        var err = ErrorConsole.Out;

        err.WriteLine();
        err.MarkupLine($"[red]Training failed for model '[cyan]{Markup.Escape(modelName)}[/]'[/]");
        err.WriteLine();

        err.MarkupLine("[red]Error:[/]");
        err.WriteLine($"  {ex.Message}");

        if (ex.InnerException != null)
        {
            err.WriteLine();
            err.MarkupLine("[grey]Inner exception:[/]");
            err.WriteLine($"  {ex.InnerException.Message}");
        }

        var suggestions = GetSuggestions(ex, "training");
        if (suggestions.Count > 0)
        {
            err.WriteLine();
            err.MarkupLine("[yellow]What you can try:[/]");
            foreach (var suggestion in suggestions)
            {
                err.MarkupLine($"  [blue]1.[/] {suggestion}");
            }
        }

        // Quick diagnostic commands
        err.WriteLine();
        err.MarkupLine("[grey]Diagnostic commands:[/]");
        if (!string.IsNullOrEmpty(dataFile))
        {
            err.MarkupLine($"  [cyan]mloop analyze {Markup.Escape(dataFile)}[/]  - Analyze your data");
        }
        err.MarkupLine("  [cyan]mloop status[/]                    - Check project status");
    }
}
