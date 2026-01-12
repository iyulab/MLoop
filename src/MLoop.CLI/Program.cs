using System.CommandLine;
using DotNetEnv;
using MLoop.CLI.Commands;
using Spectre.Console;

namespace MLoop.CLI;

/// <summary>
/// MLoop CLI Entry Point
/// </summary>
internal class Program
{
    static int Main(string[] args)
    {
        // Load .env file from project root (D:\data\MLoop\.env)
        var projectRoot = FindProjectRoot();
        if (projectRoot != null)
        {
            var envPath = Path.Combine(projectRoot, ".env");
            if (File.Exists(envPath))
            {
                Env.Load(envPath);
                // Console.WriteLine($"Loaded environment from {envPath}");
            }
        }
        var rootCommand = new RootCommand("MLoop - A modern CLI tool for ML.NET with filesystem-based MLOps")
        {
            // Phase 1 Commands (MVP)
            InitCommand.Create(),
            TrainCommand.Create(),
            PredictCommand.Create(),
            PreprocessCommand.Create(),
            ListCommand.Create(),
            LogsCommand.Create(),
            FeedbackCommand.Create(),
            CompareCommand.Create(),
            PromoteCommand.Create(),
            InfoCommand.Create(),
            EvaluateCommand.Create(),
            ValidateCommand.Create(),
            ExtensionsCommand.Create(),
            NewCommand.Create(),
            StatusCommand.Create(),

            // Phase 2 Commands
            new ServeCommand(),
            new PipelineCommand(),

            // Phase 4: Production Deployment
            DockerCommand.Create(),
        };

        // Display banner
        if (args.Length == 0)
        {
            DisplayBanner();
        }

        var parseResult = rootCommand.Parse(args);
        return parseResult.Invoke();
    }

    private static void DisplayBanner()
    {
        AnsiConsole.Write(
            new FigletText("MLoop")
                .LeftJustified()
                .Color(Color.Blue));

        AnsiConsole.MarkupLine("[grey]v0.1.0-alpha - ML.NET CLI Tool[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Clean Data In, Trained Model Out - That's It.[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Usage: [blue]mloop[/] [[command]] [[options]]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Commands:");
        AnsiConsole.MarkupLine("  [green]init[/]        Initialize a new ML project");
        AnsiConsole.MarkupLine("  [green]train[/]       Train a model using AutoML");
        AnsiConsole.MarkupLine("  [green]predict[/]     Make predictions with a trained model");
        AnsiConsole.MarkupLine("  [green]preprocess[/]  Execute preprocessing scripts on data");
        AnsiConsole.MarkupLine("  [green]list[/]        List all experiments");
        AnsiConsole.MarkupLine("  [green]logs[/]        View prediction logs");
        AnsiConsole.MarkupLine("  [green]feedback[/]    Manage prediction feedback");
        AnsiConsole.MarkupLine("  [green]compare[/]     Compare experiments side by side");
        AnsiConsole.MarkupLine("  [green]promote[/]     Promote an experiment to production");
        AnsiConsole.MarkupLine("  [green]info[/]        Display dataset profiling information");
        AnsiConsole.MarkupLine("  [green]evaluate[/]    Evaluate model performance on test data");
        AnsiConsole.MarkupLine("  [green]validate[/]    Validate extensibility scripts");
        AnsiConsole.MarkupLine("  [green]extensions[/]  List all hooks and metrics");
        AnsiConsole.MarkupLine("  [green]status[/]      Show project status at a glance");
        AnsiConsole.MarkupLine("  [green]serve[/]       Start REST API for model serving");
        AnsiConsole.MarkupLine("  [green]docker[/]      Generate Docker configuration for deployment");
        AnsiConsole.MarkupLine("  [green]pipeline[/]    Execute ML workflow from YAML");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Use [blue]mloop [[command]] --help[/] for more information about a command.");
    }

    /// <summary>
    /// Find project root by looking for .env or .git directory
    /// </summary>
    private static string? FindProjectRoot()
    {
        var current = Directory.GetCurrentDirectory();
        while (current != null)
        {
            if (File.Exists(Path.Combine(current, ".env")) ||
                Directory.Exists(Path.Combine(current, ".git")))
            {
                return current;
            }

            var parent = Directory.GetParent(current);
            current = parent?.FullName;
        }
        return null;
    }
}
