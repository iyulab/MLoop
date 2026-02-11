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
            Description = "Template: basic (default), validation, logging, performance-gate, deploy"
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
                AnsiConsole.MarkupLine("[red]Error:[/] Not inside a MLoop project.");
                AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
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
                AnsiConsole.MarkupLine($"[red]Error:[/] Hook already exists: {fileName}");
                AnsiConsole.MarkupLine($"[yellow]Path:[/] {filePath}");
                return 1;
            }

            // Select template
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
            AnsiConsole.MarkupLine($"[cyan]Path:[/] {filePath}");
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

    private static HookType? ValidateAndParseHookType(string type)
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
            AnsiConsole.MarkupLine($"[red]Error:[/] Invalid hook type '{type}'");
            AnsiConsole.MarkupLine("[yellow]Valid types:[/] pre-train, post-train, pre-predict, post-evaluate");
            return null;
        }

        return hookType;
    }

    private static string SelectTemplate(string? template, HookType hookType)
    {
        if (!string.IsNullOrEmpty(template))
        {
            return template;
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

    private static string GenerateHookContent(string name, HookType hookType, string template)
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

    private static string GenerateBasicTemplate(string className, string hookName, HookType hookType)
    {
        var stage = hookType.ToString();

        return $@"using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class {className} : IMLoopHook
{{
    public string Name => ""{hookName}"";

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        try
        {{
            ctx.Logger.Info(""Executing {hookName} hook..."");

            // TODO: Implement your hook logic here
            var modelName = ctx.GetMetadata<string>(""ModelName"", ""default"");
            ctx.Logger.Info($""Processing model: {{modelName}}"");

            return Task.FromResult(HookResult.Continue());
        }}
        catch (Exception ex)
        {{
            return Task.FromResult(HookResult.Abort($""Hook error: {{ex.Message}}""));
        }}
    }}
}}
";
    }

    private static string GenerateValidationTemplate(string className, string hookName, HookType hookType)
    {
        var dataAvailable = hookType == HookType.PreTrain || hookType == HookType.PrePredict;

        return $@"using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class {className} : IMLoopHook
{{
    public string Name => ""{hookName}"";

    private const int MIN_ROWS = 100;

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        try
        {{
{(dataAvailable ? @"            if (ctx.DataView == null)
            {
                return Task.FromResult(HookResult.Abort(""No data provided""));
            }

            var rowCount = ctx.DataView.GetRowCount() ?? 0;
            ctx.Logger.Info($""Dataset size: {{rowCount:N0}} rows"");

            if (rowCount < MIN_ROWS)
            {
                return Task.FromResult(HookResult.Abort($""Insufficient data: {{rowCount}} rows""));
            }

            ctx.Logger.Info(""✅ Validation passed"");" : @"            var modelName = ctx.GetMetadata<string>(""ModelName"", ""default"");
            ctx.Logger.Info($""Validating {{modelName}}..."");")}

            return Task.FromResult(HookResult.Continue());
        }}
        catch (Exception ex)
        {{
            return Task.FromResult(HookResult.Abort($""Validation error: {{ex.Message}}""));
        }}
    }}
}}
";
    }

    private static string GenerateLoggingTemplate(string className, string hookName)
    {
        return $@"using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class {className} : IMLoopHook
{{
    public string Name => ""{hookName}"";

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        try
        {{
            var modelName = ctx.GetMetadata<string>(""ModelName"", ""default"");
            ctx.Logger.Info($""📊 Logging {{modelName}} metrics"");

            if (ctx.Metrics != null)
            {{
                var metricsDict = ctx.Metrics as IDictionary<string, double>;
                if (metricsDict != null)
                {{
                    foreach (var (metricName, metricValue) in metricsDict)
                    {{
                        ctx.Logger.Info($""   {{metricName}}: {{metricValue:F4}}"");
                    }}
                }}
            }}

            return Task.FromResult(HookResult.Continue());
        }}
        catch (Exception ex)
        {{
            ctx.Logger.Warning($""Logging failed: {{ex.Message}}"");
            return Task.FromResult(HookResult.Continue());
        }}
    }}
}}
";
    }

    private static string GeneratePerformanceGateTemplate(string className, string hookName)
    {
        return $@"using System.Threading.Tasks;
using Microsoft.ML.Data;
using MLoop.Extensibility.Hooks;

public class {className} : IMLoopHook
{{
    public string Name => ""{hookName}"";

    private const double MIN_ACCURACY = 0.70;

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        try
        {{
            if (ctx.Metrics == null)
            {{
                return Task.FromResult(HookResult.Continue());
            }}

            double primaryMetric = 0.0;
            if (ctx.Metrics is BinaryClassificationMetrics binaryMetrics)
            {{
                primaryMetric = binaryMetrics.Accuracy;
            }}
            else if (ctx.Metrics is MulticlassClassificationMetrics multiclassMetrics)
            {{
                primaryMetric = multiclassMetrics.MacroAccuracy;
            }}
            else if (ctx.Metrics is RegressionMetrics regressionMetrics)
            {{
                primaryMetric = regressionMetrics.RSquared;
            }}

            ctx.Logger.Info($""Metric: {{primaryMetric:P2}} (threshold: {{MIN_ACCURACY:P2}})"");

            if (primaryMetric < MIN_ACCURACY)
            {{
                return Task.FromResult(HookResult.Abort($""Below threshold: {{primaryMetric:P2}}""));
            }}

            ctx.Logger.Info(""✅ Performance gate passed"");
            return Task.FromResult(HookResult.Continue());
        }}
        catch (Exception ex)
        {{
            return Task.FromResult(HookResult.Abort($""Validation error: {{ex.Message}}""));
        }}
    }}
}}
";
    }

    private static string GenerateDeployTemplate(string className, string hookName)
    {
        return $@"using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class {className} : IMLoopHook
{{
    public string Name => ""{hookName}"";

    private const double DEPLOY_THRESHOLD = 0.90;

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {{
        try
        {{
            var metricsDict = ctx.Metrics as IDictionary<string, double>;
            var primaryMetric = metricsDict?.Values.FirstOrDefault() ?? 0.0;

            if (primaryMetric >= DEPLOY_THRESHOLD)
            {{
                ctx.Logger.Info(""🚀 Triggering deployment..."");

                // TODO: Implement deployment logic
                var modelPath = ctx.GetMetadata<string>(""ModelPath"", """");
                ctx.Logger.Info($""Model ready: {{modelPath}}"");

                return HookResult.ModifyConfig(
                    new Dictionary<string, object> {{ [""DeploymentTriggered""] = true }},
                    ""Deployment triggered"");
            }}

            return HookResult.Continue();
        }}
        catch (Exception ex)
        {{
            ctx.Logger.Warning($""Deployment failed: {{ex.Message}}"");
            return HookResult.Continue();
        }}
    }}
}}
";
    }

    private static string ToPascalCase(string text)
    {
        return string.Concat(
            text.Split(new[] { '-', '_', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(word => char.ToUpperInvariant(word[0]) + word.Substring(1).ToLowerInvariant()));
    }

    private static string ToTitleCase(string text)
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
