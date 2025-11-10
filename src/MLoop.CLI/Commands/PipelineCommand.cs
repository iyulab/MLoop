using System.CommandLine;
using System.Text.Json;
using Microsoft.ML;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Pipeline;
using Spectre.Console;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.CLI.Commands;

/// <summary>
/// Pipeline command to execute ML workflows from YAML
/// </summary>
public class PipelineCommand : Command
{
    public PipelineCommand() : base("pipeline", "Execute ML workflow from YAML pipeline definition")
    {
        var fileArg = new Argument<string>(
            name: "file",
            description: "Path to pipeline YAML file");

        var variablesOption = new Option<string?>(
            name: "--vars",
            description: "Variables in JSON format to override pipeline variables");

        var dryRunOption = new Option<bool>(
            name: "--dry-run",
            description: "Validate pipeline without executing",
            getDefaultValue: () => false);

        var saveResultOption = new Option<string?>(
            name: "--save-result",
            description: "Save pipeline result to JSON file");

        AddArgument(fileArg);
        AddOption(variablesOption);
        AddOption(dryRunOption);
        AddOption(saveResultOption);

        this.SetHandler(async (file, vars, dryRun, saveResult) =>
        {
            // Initialize services
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);
            var mlContext = new MLContext(seed: 42);

            await ExecuteAsync(file, vars, dryRun, saveResult, projectDiscovery, mlContext);
        }, fileArg, variablesOption, dryRunOption, saveResultOption);
    }

    private static async Task ExecuteAsync(
        string pipelineFile,
        string? variablesJson,
        bool dryRun,
        string? saveResultPath,
        IProjectDiscovery projectDiscovery,
        MLContext mlContext)
    {
        try
        {
            // Verify we're in an MLoop project
            var projectRoot = projectDiscovery.FindRoot();

            // Read and parse pipeline YAML
            if (!File.Exists(pipelineFile))
            {
                AnsiConsole.MarkupLine($"[red]❌ Pipeline file not found: {pipelineFile}[/]");
                return;
            }

            var yamlContent = await File.ReadAllTextAsync(pipelineFile);

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();

            var pipeline = deserializer.Deserialize<PipelineDefinition>(yamlContent);

            if (pipeline == null)
            {
                AnsiConsole.MarkupLine("[red]❌ Failed to parse pipeline file[/]");
                return;
            }

            // Override variables if provided
            if (!string.IsNullOrEmpty(variablesJson))
            {
                var vars = JsonSerializer.Deserialize<Dictionary<string, object>>(variablesJson);
                if (vars != null && pipeline.Variables != null)
                {
                    foreach (var (key, value) in vars)
                    {
                        pipeline.Variables[key] = value;
                    }
                }
            }

            // Display pipeline info
            AnsiConsole.Write(new Rule($"[blue]{pipeline.Name}[/]").LeftJustified());

            if (!string.IsNullOrEmpty(pipeline.Description))
            {
                AnsiConsole.MarkupLine($"[grey]{pipeline.Description}[/]");
            }

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]📂 Project: {projectRoot}[/]");
            AnsiConsole.MarkupLine($"[grey]📄 Pipeline: {pipelineFile}[/]");
            AnsiConsole.MarkupLine($"[grey]📦 Steps: {pipeline.Steps.Count}[/]");
            AnsiConsole.WriteLine();

            if (dryRun)
            {
                AnsiConsole.MarkupLine("[yellow]🔍 DRY RUN MODE - Pipeline validation only[/]");
                DisplayPipelineSteps(pipeline);
                AnsiConsole.MarkupLine("[green]✅ Pipeline validation successful[/]");
                return;
            }

            // Execute pipeline
            var executor = new PipelineExecutor(mlContext);

            var result = await executor.ExecuteAsync(
                pipeline,
                logger: msg => AnsiConsole.MarkupLine($"[grey]{msg}[/]"));

            // Display results
            DisplayPipelineResults(result);

            // Save result if requested
            if (!string.IsNullOrEmpty(saveResultPath))
            {
                var resultJson = JsonSerializer.Serialize(result, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                await File.WriteAllTextAsync(saveResultPath, resultJson);
                AnsiConsole.MarkupLine($"\n[grey]💾 Result saved to: {saveResultPath}[/]");
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]❌ Pipeline execution failed: {ex.Message}[/]");
            throw;
        }
    }

    private static void DisplayPipelineSteps(PipelineDefinition pipeline)
    {
        var table = new Table();
        table.AddColumn("Step");
        table.AddColumn("Type");
        table.AddColumn("Parameters");

        for (int i = 0; i < pipeline.Steps.Count; i++)
        {
            var step = pipeline.Steps[i];
            var paramsStr = string.Join(", ", step.Parameters.Select(p => $"{p.Key}={p.Value}"));
            table.AddRow(
                $"{i + 1}. {step.Name}",
                $"[blue]{step.Type}[/]",
                $"[grey]{paramsStr}[/]"
            );
        }

        AnsiConsole.Write(table);
    }

    private static void DisplayPipelineResults(PipelineResult result)
    {
        var table = new Table();
        table.Border = TableBorder.Rounded;
        table.AddColumn("Step");
        table.AddColumn("Type");
        table.AddColumn("Duration");
        table.AddColumn("Status");

        foreach (var step in result.StepResults)
        {
            var statusMarkup = step.Status switch
            {
                StepStatus.Completed => "[green]✅ Completed[/]",
                StepStatus.Failed => "[red]❌ Failed[/]",
                StepStatus.Skipped => "[yellow]⏭️  Skipped[/]",
                _ => "[grey]Unknown[/]"
            };

            table.AddRow(
                step.StepName,
                $"[blue]{step.StepType}[/]",
                $"[grey]{step.Duration.TotalSeconds:F2}s[/]",
                statusMarkup
            );
        }

        AnsiConsole.WriteLine();
        AnsiConsole.Write(table);

        // Summary
        var statusColor = result.Status switch
        {
            PipelineStatus.Completed => "green",
            PipelineStatus.Failed => "red",
            PipelineStatus.PartiallyCompleted => "yellow",
            _ => "grey"
        };

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[{statusColor}]{result.Status}[/]").LeftJustified());
        AnsiConsole.MarkupLine($"[grey]⏱️  Duration: {result.Duration.TotalSeconds:F2}s[/]");

        if (!string.IsNullOrEmpty(result.Error))
        {
            AnsiConsole.MarkupLine($"[red]Error: {result.Error}[/]");
        }
    }
}
