using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Preprocessing;
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

        var labelOption = new Option<string?>("--label")
        {
            Description = "Label column name (optional, passes to preprocessing context)"
        };

        var command = new Command("preprocess", "Execute preprocessing scripts on data")
        {
            inputFileArg,
            outputOption,
            labelOption
        };

        command.SetHandler(ExecuteAsync, inputFileArg, outputOption, labelOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? inputFile,
        string? output,
        string? labelColumn)
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
        catch (InvalidOperationException ex)
        {
            AnsiConsole.MarkupLine("[red]Error:[/]");
            AnsiConsole.WriteLine(ex.Message);
            return 1;
        }
        catch (Exception ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);

            if (ex.InnerException != null)
            {
                AnsiConsole.Markup("[grey]Details:[/] ");
                AnsiConsole.WriteLine(ex.InnerException.Message);
            }

            return 1;
        }
    }
}

/// <summary>
/// Logger implementation using Spectre.Console
/// </summary>
internal class SpectreLogger : ILogger
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
