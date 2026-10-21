using System.CommandLine;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Hooks;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop new hook - Generates hook template scripts
/// </summary>
public static class NewHookCommand
{
    public static Command Create()
    {
        var nameArg = new Argument<string>("name")
        {
            Description = "Hook name (e.g., 'data-validation', 'mlflow-logging')"
        };

        var typeArg = new Argument<string>("type")
        {
            Description = "Hook type: pre-train, post-train, pre-predict, post-evaluate"
        };

        var templateOption = new Option<string?>("--template")
        {
            Description = "Template: basic, validation, logging, performance-gate, deploy " +
                          "(default: validation for pre-train and pre-predict, logging for post-train and post-evaluate)"
        };

        var command = new Command("hook", "Generate a new hook script");
        command.Arguments.Add(nameArg);
        command.Arguments.Add(typeArg);
        command.Options.Add(templateOption);

        command.SetAction((parseResult) =>
        {
            var name = parseResult.GetValue(nameArg)!;
            var type = parseResult.GetValue(typeArg)!;
            var template = parseResult.GetValue(templateOption);
            return ExecuteAsync(name, type, template);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(string name, string hookType, string? template)
    {
        try
        {
            // Validate hook type
            var hookTypeEnum = ValidateAndParseHookType(hookType);
            if (hookTypeEnum == null)
            {
                return 1;
            }

            // Find project root
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
                return 1;
            }

            // Initialize hook engine
            var hookEngine = new HookEngine(projectRoot, new NullLogger());
            var hookDir = hookEngine.GetHookDirectory(hookTypeEnum.Value);

            // Ensure directory exists
            Directory.CreateDirectory(hookDir);

            // Generate file name
            var fileName = $"{name.Trim()}.cs";
            var filePath = Path.Combine(hookDir, fileName);

            if (File.Exists(filePath))
            {
                ErrorConsole.Error($"Hook already exists: {fileName}", $"Path: {filePath}");
                return 1;
            }

            // Select template
            if (!string.IsNullOrEmpty(template) && !Templates.Contains(template, StringComparer.OrdinalIgnoreCase))
            {
                ErrorConsole.Error(
                    $"Unknown template '{template}'.",
                    $"Use one of: {string.Join(", ", Templates)} — or leave --template out for the hook type's default.");
                return 1;
            }
            var templateType = SelectTemplate(template, hookTypeEnum.Value);

            // Generate hook content
            var hookContent = GenerateHookContent(name, hookTypeEnum.Value, templateType);

            // Write file
            await File.WriteAllTextAsync(filePath, hookContent);

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]✅ Hook created successfully![/]");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[cyan]Name:[/] {name}");
            AnsiConsole.MarkupLine($"[cyan]Type:[/] {hookType}");
            AnsiConsole.MarkupLine($"[cyan]Template:[/] {templateType}");
            ValueLine.Write("[cyan]Path:[/] ", filePath);
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            AnsiConsole.MarkupLine("  1. Edit the hook script to implement your logic");
            AnsiConsole.MarkupLine("  2. Run [blue]mloop train[/] - hook will be auto-discovered");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex);
            return 1;
        }
    }

    internal static HookType? ValidateAndParseHookType(string type)
    {
        var validTypes = new Dictionary<string, HookType>(StringComparer.OrdinalIgnoreCase)
        {
            ["pre-train"] = HookType.PreTrain,
            ["post-train"] = HookType.PostTrain,
            ["pre-predict"] = HookType.PrePredict,
            ["post-evaluate"] = HookType.PostEvaluate
        };

        if (!validTypes.TryGetValue(type, out var hookType))
        {
            ErrorConsole.Error($"Invalid hook type '{type}'", "Valid types: pre-train, post-train, pre-predict, post-evaluate");
            return null;
        }

        return hookType;
    }

    /// <summary>The templates <see cref="GenerateHookContent"/> writes.</summary>
    internal static readonly IReadOnlyList<string> Templates = ["basic", "validation", "logging", "performance-gate", "deploy"];

    internal static string SelectTemplate(string? template, HookType hookType)
    {
        if (!string.IsNullOrEmpty(template))
        {
            return template.ToLowerInvariant();
        }

        // Default templates based on hook type
        return hookType switch
        {
            HookType.PreTrain => "validation",
            HookType.PostTrain => "logging",
            HookType.PrePredict => "validation",
            HookType.PostEvaluate => "logging",
            _ => "basic"
        };
    }

    internal static string GenerateHookContent(string name, HookType hookType, string template)
    {
        var className = ToPascalCase(name) + "Hook";
        var hookName = ToTitleCase(name);

        return template.ToLowerInvariant() switch
        {
            "validation" => GenerateValidationTemplate(className, hookName, hookType),
            "logging" => GenerateLoggingTemplate(className, hookName),
            "performance-gate" => GeneratePerformanceGateTemplate(className, hookName),
            "deploy" => GenerateDeployTemplate(className, hookName),
            _ => GenerateBasicTemplate(className, hookName, hookType)
        };
    }

    // The templates are written for the contexts the engine passes (TrainCommand, TrainingEngine):
    // pre-train and pre-predict may carry no DataView — the data file is in Metadata["DataFile"] —
    // and Metrics, when present, is the run's metric dictionary (name → value).
    private const string Usings = """
        using System;
        using System.Collections.Generic;
        using System.Linq;
        using System.Threading.Tasks;
        using MLoop.Extensibility.Hooks;
        """;

    private static string GenerateBasicTemplate(string className, string hookName, HookType hookType) => $$"""
        {{Usings}}

        public class {{className}} : IMLoopHook
        {
            public string Name => "{{hookName}}";

            public Task<HookResult> ExecuteAsync(HookContext ctx)
            {
                try
                {
                    // TODO: Implement your {{hookType}} logic here
                    var modelName = ctx.GetMetadata<string>("ModelName", "default");
                    ctx.Logger.Info($"{{hookName}}: model {modelName}");

                    return Task.FromResult(HookResult.Continue());
                }
                catch (Exception ex)
                {
                    return Task.FromResult(HookResult.Abort($"Hook error: {ex.Message}"));
                }
            }
        }

        """;

    private static string GenerateValidationTemplate(string className, string hookName, HookType hookType)
    {
        var check = hookType is HookType.PreTrain or HookType.PrePredict
            ? """
                        // The data, as a DataView when the stage has one, otherwise the file it reads.
                        long? rowCount = ctx.DataView?.GetRowCount();
                        if (rowCount is null && ctx.GetMetadata<string>("DataFile", "") is { Length: > 0 } dataFile && File.Exists(dataFile))
                            rowCount = File.ReadLines(dataFile).LongCount() - 1; // minus the header

                        if (rowCount is null)
                        {
                            ctx.Logger.Warning("No data to check");
                            return Task.FromResult(HookResult.Continue());
                        }

                        ctx.Logger.Info($"Dataset size: {rowCount:N0} rows");
                        if (rowCount < MinRows)
                            return Task.FromResult(HookResult.Abort($"Insufficient data: {rowCount} rows (at least {MinRows})"));

                        ctx.Logger.Info("Validation passed");
            """
            : """
                        var modelName = ctx.GetMetadata<string>("ModelName", "default");
                        ctx.Logger.Info($"Validating {modelName}...");
            """;

        return $$"""
            {{Usings}}
            using System.IO;

            public class {{className}} : IMLoopHook
            {
                public string Name => "{{hookName}}";

                private const int MinRows = 100;

                public Task<HookResult> ExecuteAsync(HookContext ctx)
                {
                    try
                    {
            {{check}}

                        return Task.FromResult(HookResult.Continue());
                    }
                    catch (Exception ex)
                    {
                        return Task.FromResult(HookResult.Abort($"Validation error: {ex.Message}"));
                    }
                }
            }

            """;
    }

    private static string GenerateLoggingTemplate(string className, string hookName) => $$"""
        {{Usings}}

        public class {{className}} : IMLoopHook
        {
            public string Name => "{{hookName}}";

            public Task<HookResult> ExecuteAsync(HookContext ctx)
            {
                try
                {
                    var modelName = ctx.GetMetadata<string>("ModelName", "default");
                    ctx.Logger.Info($"Logging {modelName} metrics");

                    if (ctx.Metrics is IReadOnlyDictionary<string, double> metrics)
                    {
                        foreach (var (metricName, metricValue) in metrics)
                            ctx.Logger.Info($"   {metricName}: {metricValue:F4}");
                    }

                    return Task.FromResult(HookResult.Continue());
                }
                catch (Exception ex)
                {
                    ctx.Logger.Warning($"Logging failed: {ex.Message}");
                    return Task.FromResult(HookResult.Continue());
                }
            }
        }

        """;

    private static string GeneratePerformanceGateTemplate(string className, string hookName) => $$"""
        {{Usings}}

        public class {{className}} : IMLoopHook
        {
            public string Name => "{{hookName}}";

            // The metric to hold the model to, and its floor. The names are those the run reports
            // (accuracy, auc, f1_score, r_squared, …); pick one your task produces.
            private const string Metric = "accuracy";
            private const double Floor = 0.70;

            public Task<HookResult> ExecuteAsync(HookContext ctx)
            {
                try
                {
                    if (ctx.Metrics is not IReadOnlyDictionary<string, double> metrics)
                        return Task.FromResult(HookResult.Continue());

                    if (!metrics.TryGetValue(Metric, out var value))
                    {
                        ctx.Logger.Warning($"No '{Metric}' among the metrics ({string.Join(", ", metrics.Keys)})");
                        return Task.FromResult(HookResult.Continue());
                    }

                    ctx.Logger.Info($"{Metric}: {value:F4} (floor {Floor:F4})");
                    if (value < Floor)
                        return Task.FromResult(HookResult.Abort($"{Metric} {value:F4} is below {Floor:F4}"));

                    ctx.Logger.Info("Performance gate passed");
                    return Task.FromResult(HookResult.Continue());
                }
                catch (Exception ex)
                {
                    return Task.FromResult(HookResult.Abort($"Validation error: {ex.Message}"));
                }
            }
        }

        """;

    private static string GenerateDeployTemplate(string className, string hookName) => $$"""
        {{Usings}}

        public class {{className}} : IMLoopHook
        {
            public string Name => "{{hookName}}";

            private const string Metric = "accuracy";
            private const double DeployThreshold = 0.90;

            public Task<HookResult> ExecuteAsync(HookContext ctx)
            {
                try
                {
                    if (ctx.Metrics is IReadOnlyDictionary<string, double> metrics
                        && metrics.TryGetValue(Metric, out var value) && value >= DeployThreshold)
                    {
                        // TODO: Implement deployment logic
                        var modelPath = ctx.GetMetadata<string>("ModelPath", "");
                        ctx.Logger.Info($"Model ready to deploy: {modelPath}");

                        return Task.FromResult(HookResult.ModifyConfig(
                            new Dictionary<string, object> { ["DeploymentTriggered"] = true },
                            "Deployment triggered"));
                    }

                    return Task.FromResult(HookResult.Continue());
                }
                catch (Exception ex)
                {
                    ctx.Logger.Warning($"Deployment failed: {ex.Message}");
                    return Task.FromResult(HookResult.Continue());
                }
            }
        }

        """;

    internal static string ToPascalCase(string text)
    {
        return string.Concat(
            text.Split(new[] { '-', '_', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(word => char.ToUpperInvariant(word[0]) + word.Substring(1).ToLowerInvariant()));
    }

    internal static string ToTitleCase(string text)
    {
        return string.Join(" ",
            text.Split(new[] { '-', '_' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(word => char.ToUpperInvariant(word[0]) + word.Substring(1).ToLowerInvariant()));
    }

    private class NullLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
