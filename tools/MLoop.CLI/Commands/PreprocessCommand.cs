using System.CommandLine;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Preprocessing;
using MLoop.Core.Preprocessing.Incremental;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Deliverables;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleApplication;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration;
using MLoop.Extensibility;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop preprocess - Executes preprocessing scripts on data
/// </summary>
public static class PreprocessCommand
{
    public static Command Create()
    {
        var inputFileArg = new Argument<string?>("input-file")
        {
            Description = "Path to input data file (defaults to datasets/train.csv if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output path for preprocessed data (defaults to .mloop/temp/preprocess/)"
        };

        var labelOption = new Option<string?>("--label", "-l")
        {
            Description = "Label column name (optional, passes to preprocessing context)"
        };

        var incrementalOption = new Option<bool>("--incremental")
        {
            Description = "Use incremental workflow with HITL validation"
        };

        var skipHitlOption = new Option<bool>("--skip-hitl")
        {
            Description = "Skip HITL prompts and use recommended defaults (incremental mode only)"
        };

        var autoApproveOption = new Option<bool>("--auto-approve")
        {
            Description = "Automatically approve rules at confidence checkpoint (incremental mode only)"
        };

        var command = new Command("preprocess", "Execute preprocessing scripts on data");
        command.Arguments.Add(inputFileArg);
        command.Options.Add(outputOption);
        command.Options.Add(labelOption);
        command.Options.Add(incrementalOption);
        command.Options.Add(skipHitlOption);
        command.Options.Add(autoApproveOption);

        command.SetAction((parseResult) =>
        {
            var inputFile = parseResult.GetValue(inputFileArg);
            var output = parseResult.GetValue(outputOption);
            var labelColumn = parseResult.GetValue(labelOption);
            var incremental = parseResult.GetValue(incrementalOption);
            var skipHitl = parseResult.GetValue(skipHitlOption);
            var autoApprove = parseResult.GetValue(autoApproveOption);
            return ExecuteAsync(inputFile, output, labelColumn, incremental, skipHitl, autoApprove);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? inputFile,
        string? output,
        string? labelColumn,
        bool incremental,
        bool skipHitl,
        bool autoApprove)
    {
        try
        {
            // Initialize components
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            // Find project root
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

            // Route to incremental or traditional workflow
            if (incremental)
            {
                return await ExecuteIncrementalAsync(projectRoot, inputFile, output, labelColumn, skipHitl, autoApprove);
            }
            else
            {
                return await ExecuteTraditionalAsync(projectRoot, inputFile, output, labelColumn);
            }
        }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.MarkupLine("[red]Error:[/]");
            AnsiConsole.WriteLine(ex.Message);
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "preprocessing");
            return 1;
        }
    }

    private static async Task<int> ExecuteTraditionalAsync(
        string projectRoot,
        string? inputFile,
        string? output,
        string? labelColumn)
    {
        // Initialize preprocessing engine
        var logger = new SpectreLogger();
        var preprocessingEngine = new PreprocessingEngine(projectRoot, logger);

        // Check if preprocessing scripts exist
        if (!preprocessingEngine.HasPreprocessingScripts())
        {
            AnsiConsole.MarkupLine("[yellow]No preprocessing scripts found[/]");
            AnsiConsole.MarkupLine($"[grey]Create scripts in: {preprocessingEngine.GetPreprocessingDirectory()}[/]");
            AnsiConsole.MarkupLine("[grey]Scripts should be named: 01_*.cs, 02_*.cs, etc.[/]");
            return 0;
        }

        // Resolve input file path (Convention: datasets/train.csv)
        var fileSystem = new FileSystemManager();
        string resolvedInputFile;

        if (string.IsNullOrEmpty(inputFile))
        {
            // Auto-discover datasets/train.csv
            var datasetDiscovery = new DatasetDiscovery(fileSystem);
            var datasets = datasetDiscovery.FindDatasets(projectRoot);

            if (datasets?.TrainPath == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] No input file specified and datasets/train.csv not found.");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop preprocess <input-file>");
                return 1;
            }

            resolvedInputFile = datasets.TrainPath;
            AnsiConsole.MarkupLine($"[green]✓[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedInputFile)}[/]");
        }
        else
        {
            // Validate explicit input file
            resolvedInputFile = Path.IsPathRooted(inputFile)
                ? inputFile
                : Path.Combine(projectRoot, inputFile);

            if (!File.Exists(resolvedInputFile))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Input file not found: {resolvedInputFile}");
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]✓[/] Using input file: [cyan]{Path.GetRelativePath(projectRoot, resolvedInputFile)}[/]");
        }

        AnsiConsole.WriteLine();

        // Execute preprocessing with progress
        string outputPath = string.Empty;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync("[yellow]Running preprocessing scripts...[/]", async ctx =>
            {
                outputPath = await preprocessingEngine.ExecuteAsync(resolvedInputFile, labelColumn);
                ctx.Status("[green]Preprocessing complete![/]");
            });

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[green]✓[/] Preprocessing complete!");
        AnsiConsole.MarkupLine($"[green]✓[/] Output: [cyan]{Path.GetRelativePath(projectRoot, outputPath)}[/]");

        // Copy to specified output location if provided
        if (!string.IsNullOrEmpty(output))
        {
            var resolvedOutput = Path.IsPathRooted(output)
                ? output
                : Path.Combine(projectRoot, output);

            var outputDir = Path.GetDirectoryName(resolvedOutput);
            if (!string.IsNullOrEmpty(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            File.Copy(outputPath, resolvedOutput, overwrite: true);
            AnsiConsole.MarkupLine($"[green]✓[/] Copied to: [cyan]{Path.GetRelativePath(projectRoot, resolvedOutput)}[/]");
        }

        AnsiConsole.WriteLine();

        return 0;
    }

    private static async Task<int> ExecuteIncrementalAsync(
        string projectRoot,
        string? inputFile,
        string? output,
        string? labelColumn,
        bool skipHitl,
        bool autoApprove)
    {
        // Resolve input file
        var fileSystem = new FileSystemManager();
        string resolvedInputFile;

        if (string.IsNullOrEmpty(inputFile))
        {
            // Auto-discover datasets/train.csv
            var datasetDiscovery = new DatasetDiscovery(fileSystem);
            var datasets = datasetDiscovery.FindDatasets(projectRoot);

            if (datasets?.TrainPath == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] No input file specified and datasets/train.csv not found.");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop preprocess <input-file>");
                return 1;
            }

            resolvedInputFile = datasets.TrainPath;
            AnsiConsole.MarkupLine($"[green]✓[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedInputFile)}[/]");
        }
        else
        {
            // Validate explicit input file
            resolvedInputFile = Path.IsPathRooted(inputFile)
                ? inputFile
                : Path.Combine(projectRoot, inputFile);

            if (!File.Exists(resolvedInputFile))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Input file not found: {resolvedInputFile}");
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]✓[/] Using input file: [cyan]{Path.GetRelativePath(projectRoot, resolvedInputFile)}[/]");
        }

        // Configure incremental workflow
        var outputDir = !string.IsNullOrEmpty(output)
            ? (Path.IsPathRooted(output) ? output : Path.Combine(projectRoot, output))
            : Path.Combine(projectRoot, "cleaned");

        var config = new IncrementalWorkflowConfig
        {
            SkipHITL = skipHitl,
            EnableAutoApproval = autoApprove,
            OutputDirectory = outputDir,
            GenerateScripts = true,
            GenerateReport = true
        };

        // Initialize workflow components
        var samplingEngine = new SamplingEngine();
        var sampleAnalyzer = new SampleAnalyzer();
        var ruleDiscoveryEngine = new RuleDiscoveryEngine(new SpectreGenericLogger<RuleDiscoveryEngine>());
        var hitlWorkflowService = new HITLWorkflowService(
            new HITLQuestionGenerator(new SpectreGenericLogger<HITLQuestionGenerator>()),
            new HITLPromptBuilder(new SpectreGenericLogger<HITLPromptBuilder>()),
            new HITLDecisionLogger(outputDir, new SpectreGenericLogger<HITLDecisionLogger>()),
            new SpectreGenericLogger<HITLWorkflowService>());

        var ruleApplier = new RuleApplier(new SpectreGenericLogger<RuleApplier>());
        var scriptGenerator = new ScriptGenerator(new SpectreGenericLogger<ScriptGenerator>());
        var reportGenerator = new ReportGenerator(new SpectreGenericLogger<ReportGenerator>());
        var deliverableGenerator = new DeliverableGenerator(
            scriptGenerator,
            reportGenerator,
            new SpectreGenericLogger<DeliverableGenerator>());

        var orchestrator = new IncrementalWorkflowOrchestrator(
            samplingEngine,
            sampleAnalyzer,
            ruleDiscoveryEngine,
            hitlWorkflowService,
            new SpectreGenericLogger<IncrementalWorkflowOrchestrator>(),
            ruleApplier,
            deliverableGenerator);

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[blue]Incremental Preprocessing Workflow[/]");
        AnsiConsole.MarkupLine($"[grey]Output: {Path.GetRelativePath(projectRoot, outputDir)}[/]");
        AnsiConsole.WriteLine();

        // Execute workflow with progress display
        IncrementalWorkflowState? finalState = null;
        string? manifestCleanedPath = null;

        await AnsiConsole.Progress()
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new SpinnerColumn())
            .StartAsync(async ctx =>
            {
                var workflowTask = ctx.AddTask("[yellow]Incremental Workflow[/]", maxValue: 100);

                var progressReporter = new Progress<WorkflowProgress>(progress =>
                {
                    workflowTask.Value = progress.Percentage * 100;
                    workflowTask.Description = $"[yellow]{progress.Stage}[/] - {progress.Message}";
                });

                finalState = await orchestrator.ExecuteWorkflowAsync(resolvedInputFile, config, progressReporter);
                manifestCleanedPath = finalState != null
                    ? Path.Combine(config.OutputDirectory, "cleaned_data.csv")
                    : null;

                workflowTask.Value = 100;
                workflowTask.StopTask();
            });

        if (finalState == null || string.IsNullOrEmpty(manifestCleanedPath))
        {
            AnsiConsole.MarkupLine("[red]✗[/] Workflow did not complete successfully");
            return 1;
        }

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[green]✓[/] Incremental preprocessing complete!");
        AnsiConsole.MarkupLine($"[green]✓[/] Cleaned data: [cyan]{Path.GetRelativePath(projectRoot, manifestCleanedPath)}[/]");
        AnsiConsole.MarkupLine($"[green]✓[/] Discovered rules: {finalState.DiscoveredRules.Count}");
        AnsiConsole.MarkupLine($"[green]✓[/] Approved rules: {finalState.ApprovedRules.Count}");
        AnsiConsole.MarkupLine($"[green]✓[/] Confidence: {finalState.ConfidenceScore:P2}");
        AnsiConsole.MarkupLine($"[green]✓[/] Converged: {(finalState.HasConverged ? "Yes" : "No")}");
        AnsiConsole.WriteLine();

        return 0;
    }
}

/// <summary>
/// Logger implementation using Spectre.Console
/// </summary>
internal class SpectreLogger : MLoop.Extensibility.Preprocessing.ILogger
{
    public void Debug(string message)
    {
        AnsiConsole.MarkupLine($"[grey]{message}[/]");
    }

    public void Info(string message)
    {
        AnsiConsole.WriteLine(message);
    }

    public void Warning(string message)
    {
        AnsiConsole.MarkupLine($"[yellow]{message}[/]");
    }

    public void Error(string message)
    {
        AnsiConsole.MarkupLine($"[red]{message}[/]");
    }

    public void Error(string message, Exception exception)
    {
        AnsiConsole.MarkupLine($"[red]{message}[/]");
        AnsiConsole.WriteException(exception);
    }
}

/// <summary>
/// Generic logger implementation for incremental workflow using Microsoft.Extensions.Logging
/// </summary>
internal class SpectreGenericLogger<T> : Microsoft.Extensions.Logging.ILogger<T>
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        var message = formatter(state, exception);

        switch (logLevel)
        {
            case LogLevel.Trace:
            case LogLevel.Debug:
                AnsiConsole.MarkupLine($"[grey]{message}[/]");
                break;
            case LogLevel.Information:
                AnsiConsole.MarkupLine($"[blue]{message}[/]");
                break;
            case LogLevel.Warning:
                AnsiConsole.MarkupLine($"[yellow]{message}[/]");
                break;
            case LogLevel.Error:
            case LogLevel.Critical:
                AnsiConsole.MarkupLine($"[red]{message}[/]");
                if (exception != null)
                {
                    AnsiConsole.WriteException(exception);
                }
                break;
        }
    }
}
