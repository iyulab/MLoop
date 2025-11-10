using System.CommandLine;
using MLoop.CLI.Commands;
using Spectre.Console;

namespace MLoop.CLI;

/// <summary>
/// MLoop CLI Entry Point
/// </summary>
internal class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("MLoop - A modern CLI tool for ML.NET with filesystem-based MLOps")
        {
            // Phase 1 Commands (MVP)
            InitCommand.Create(),
            TrainCommand.Create(),
            PredictCommand.Create(),
            PreprocessCommand.Create(),
            ListCommand.Create(),
            PromoteCommand.Create(),
            InfoCommand.Create(),
            EvaluateCommand.Create(),
            ValidateCommand.Create(),
            ExtensionsCommand.Create(),
        };

        // Display banner
        if (args.Length == 0)
        {
            DisplayBanner();
        }

        return await rootCommand.InvokeAsync(args);
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
        AnsiConsole.MarkupLine("  [green]promote[/]     Promote an experiment to production");
        AnsiConsole.MarkupLine("  [green]info[/]        Display dataset profiling information");
        AnsiConsole.MarkupLine("  [green]evaluate[/]    Evaluate model performance on test data");
        AnsiConsole.MarkupLine("  [green]validate[/]    Validate extensibility scripts");
        AnsiConsole.MarkupLine("  [green]extensions[/]  List all hooks and metrics");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Use [blue]mloop [[command]] --help[/] for more information about a command.");
    }
}
