using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Preprocessing;
using MLoop.Core.Scripting;
using MLoop.Core.Storage;
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

        var command = new Command(
            "preprocess",
            "Run preprocessing on data: executes the IPreprocessingScript files in "
            + ".mloop/scripts/preprocess/ (alphabetical, chained).");
        command.Arguments.Add(inputFileArg);
        command.Options.Add(outputOption);
        command.Options.Add(labelOption);

        command.SetAction((parseResult) =>
        {
            var inputFile = parseResult.GetValue(inputFileArg);
            var output = parseResult.GetValue(outputOption);
            var labelColumn = parseResult.GetValue(labelOption);
            return ExecuteAsync(inputFile, output, labelColumn);
        });

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
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
                return 1;
            }

            return await ExecuteScriptsAsync(projectRoot, inputFile, output, labelColumn);
        }
        catch (InvalidOperationException ex)
        {
            ErrorConsole.Error(Markup.Escape(ex.Message));
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "preprocessing");
            return 1;
        }
    }

    private static async Task<int> ExecuteScriptsAsync(
        string projectRoot,
        string? inputFile,
        string? output,
        string? labelColumn)
    {
        // Initialize preprocessing engine
        var logger = new ConsoleLogger();
        var preprocessingEngine = new PreprocessingEngine(projectRoot, logger);

        // Check if preprocessing scripts exist
        if (!preprocessingEngine.HasPreprocessingScripts())
        {
            AnsiConsole.MarkupLine("[yellow]No preprocessing scripts found[/]");
            ValueLine.Write("[grey]Create scripts in: [/]", preprocessingEngine.GetPreprocessingDirectory());
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
                ErrorConsole.Error(
                    "No input file specified and datasets/train.csv not found.",
                    "Create datasets/train.csv or specify a file: mloop preprocess <input-file>");
                return 1;
            }

            resolvedInputFile = datasets.TrainPath;
            ValueLine.Write("[green]✓[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedInputFile));
        }
        else
        {
            // Validate explicit input file
            resolvedInputFile = ProjectPaths.Resolve(inputFile, projectRoot);

            if (!File.Exists(resolvedInputFile))
            {
                ErrorConsole.Error($"Input file not found: {Markup.Escape(resolvedInputFile)}", ErrorConsole.PathNotFoundTip(projectRoot));
                return 1;
            }

            ValueLine.Write("[green]✓[/] Using input file: ", Path.GetRelativePath(projectRoot, resolvedInputFile));
        }

        AnsiConsole.WriteLine();

        // The engine narrates each script and the end of the run; a spinner around it only repeated
        // that ("Running preprocessing scripts..." and "Preprocessing complete!" twice over).
        var outputPath = await preprocessingEngine.ExecuteAsync(resolvedInputFile, labelColumn);

        ValueLine.Write("[green]✓[/] Output: ", Path.GetRelativePath(projectRoot, outputPath));

        // Copy to specified output location if provided
        if (!string.IsNullOrEmpty(output))
        {
            var resolvedOutput = ProjectPaths.Resolve(output, projectRoot);

            var outputDir = Path.GetDirectoryName(resolvedOutput);
            if (!string.IsNullOrEmpty(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            File.Copy(outputPath, resolvedOutput, overwrite: true);
            ValueLine.Write("[green]✓[/] Copied to: ", Path.GetRelativePath(projectRoot, resolvedOutput));
        }

        AnsiConsole.WriteLine();

        return 0;
    }
}

