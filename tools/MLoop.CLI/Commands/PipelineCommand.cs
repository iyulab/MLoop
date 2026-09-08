using System.CommandLine;
using System.Text.Json;
using Microsoft.ML;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Diagnostics;
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
        var fileArg = new Argument<string>("file")
        {
            Description = "Path to pipeline YAML file"
        };

        var variablesOption = new Option<string?>("--vars", "-v")
        {
            Description = "Variables in JSON format to override pipeline variables"
        };

        var dryRunOption = new Option<bool>("--dry-run")
        {
            Description = "Validate pipeline without executing",
            DefaultValueFactory = _ => false
        };

        var saveResultOption = new Option<string?>("--save-result", "-s")
        {
            Description = "Save pipeline result to JSON file"
        };

        this.Arguments.Add(fileArg);
        this.Options.Add(variablesOption);
        this.Options.Add(dryRunOption);
        this.Options.Add(saveResultOption);

        this.SetAction((parseResult) =>
        {
            var file = parseResult.GetValue(fileArg)!;
            var vars = parseResult.GetValue(variablesOption);
            var dryRun = parseResult.GetValue(dryRunOption);
            var saveResult = parseResult.GetValue(saveResultOption);

            // Initialize services
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);
            var mlContext = new MLContext(seed: 42);

            return ExecuteAsync(file, vars, dryRun, saveResult, projectDiscovery, mlContext);
        });
    }

    /// <summary>
    /// Returns the process exit code: <c>0</c> when the pipeline ran to completion, non-zero when
    /// it could not be read, could not be parsed, or ended in a failed or partially-completed run.
    /// </summary>
    /// <remarks>
    /// Every one of those failures printed a diagnostic and then exited <c>0</c> — the command's
    /// handler returned a bare <c>Task</c>, so no path could report otherwise. A pipeline is the
    /// command most likely to be run unattended by a scheduler, which is exactly the caller that
    /// reads nothing but the exit code.
    /// </remarks>
    private static async Task<int> ExecuteAsync(
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
                ErrorConsole.Error($"Pipeline file not found: {pipelineFile}");
                return 1;
            }

            var yamlContent = await File.ReadAllTextAsync(pipelineFile);

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();

            var pipeline = deserializer.Deserialize<PipelineDefinition>(yamlContent);

            if (pipeline == null)
            {
                ErrorConsole.Error(
                    $"Failed to parse pipeline file: {pipelineFile}",
                    "The file parsed as empty — check that it is a pipeline definition and not an empty document.");
                return 1;
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
                return 0;
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

            // A run that ended Failed or PartiallyCompleted said so in the table and then exited 0.
            // The table is for a person watching; the exit code is for everything else.
            if (result.Status != PipelineStatus.Completed)
            {
                ErrorConsole.Error(
                    $"Pipeline '{result.PipelineName}' ended {result.Status}.",
                    result.Error ?? "The step table above shows which step did not complete.");
                return 1;
            }

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "pipeline");
            return 1;
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
