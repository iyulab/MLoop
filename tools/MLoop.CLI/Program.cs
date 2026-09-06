using System.CommandLine;
using DotNetEnv;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Update;
using Spectre.Console;

namespace MLoop.CLI;

/// <summary>
/// MLoop CLI Entry Point
/// </summary>
internal class Program
{
    static int Main(string[] args)
    {
        // Register the DeepLearning module for object-detection etc. tasks.
        // Production wiring: only apps that reference + register MLoop.Core.DeepLearning
        // get DL support; tabular-only consumers never pull it in.
        MLoop.Core.AutoML.DeepLearningRegistry.Register(new MLoop.Core.DeepLearning.DeepLearningModule());

        // Cleanup .old binary from previous standalone update (Windows only)
        UpdateChecker.CleanupOldBinary();

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
        var rootCommand = BuildRootCommand();

        // Display banner
        if (args.Length == 0)
        {
            DisplayBanner();
        }

        var exitCode = Execute(rootCommand, args);

        // Lazy update check (skip for update command itself)
        var firstArg = args.Length > 0 ? args[0] : null;
        if (firstArg != "update")
        {
            try
            {
                var checkTask = Task.Run(() => UpdateChecker.CheckForUpdateAsync(forceCheck: false));
                if (checkTask.Wait(TimeSpan.FromSeconds(2)))
                {
                    var info = checkTask.Result;
                    if (info?.UpdateAvailable == true)
                    {
                        // Diagnostics must never pollute stdout: machine-readable outputs
                        // (`mloop token -q`, `mloop predict --json`) are consumed via stdout capture
                        // (e.g. the documented `export MLOOP_TOKEN=$(mloop token -q)`), and an update
                        // notice appended there corrupts the token/JSON. Route to stderr.
                        var stderr = AnsiConsole.Create(new AnsiConsoleSettings
                        {
                            Out = new AnsiConsoleOutput(Console.Error)
                        });
                        stderr.WriteLine();
                        stderr.MarkupLine($"[yellow]A new version (v{info.LatestVersion}) is available. Run 'mloop update' to upgrade.[/]");
                    }
                }
            }
            catch
            {
                // Silently ignore update check failures
            }
        }

        return exitCode;
    }

    /// <summary>
    /// Parses and dispatches, factored out for the same reason <see cref="BuildRootCommand"/> is:
    /// a test that reached for <c>Parse(args).Invoke()</c> directly would be exercising a
    /// hand-rolled copy of this dispatch, and the exit that only exists here — a command line that
    /// fails to parse, where no command action ever runs — would be the half the copy omits.
    /// </summary>
    internal static int Execute(RootCommand rootCommand, string[] args)
    {
        var parseResult = rootCommand.Parse(args);

        if (parseResult.Errors.Count > 0)
            return ParseFailureReport.Report(parseResult, args);

        // A line can parse cleanly and still say something the caller did not write.
        var optionLike = OptionLikeArgumentCheck.Find(parseResult);
        if (optionLike is not null)
            return OptionLikeArgumentCheck.Report(parseResult, args, optionLike);

        return parseResult.Invoke();
    }

    /// <summary>
    /// The full command tree, factored out so tests parse the exact same tree <c>Main</c> runs
    /// against rather than a hand-maintained copy that could drift from it (see
    /// <c>ReadmeCommandContractTests</c>).
    /// </summary>
    internal static RootCommand BuildRootCommand() =>
        new("MLoop - A modern CLI tool for ML.NET with filesystem-based MLOps")
        {
            // Phase 1 Commands (MVP)
            InitCommand.Create(),
            TrainCommand.Create(),
            PredictCommand.Create(),
            PreprocessCommand.Create(),
            ListCommand.Create(),
            LogsCommand.Create(),
            FeedbackCommand.Create(),
            SampleCommand.Create(),
            TriggerCommand.Create(),
            CompareCommand.Create(),
            PromoteCommand.Create(),
            InfoCommand.Create(),
            AnalyzeCommand.Create(),
            DetectCommand.Create(),
            EvaluateCommand.Create(),
            ValidateCommand.Create(),
            ExtensionsCommand.Create(),
            NewCommand.Create(),
            StatusCommand.Create(),

            // Phase 2 Commands
            new ServeCommand(),
            new TokenCommand(),  // Issue JWT bearer tokens for the serve API
            new PipelineCommand(),
            new PrepCommand(),  // Data preprocessing pipeline
            new FeaturesCommand(),  // Feature selection policy (mloop.yaml)

            // Phase 4: Production Deployment
            DockerCommand.Create(),

            // Runtime Management
            RuntimeCommand.Create(),

            // Utility
            UpdateCommand.Create(),
        };

    private static void DisplayBanner()
    {
        AnsiConsole.Write(
            new FigletText("MLoop")
                .LeftJustified()
                .Color(Color.Blue));

        AnsiConsole.MarkupLine($"[grey]v{UpdateChecker.GetCurrentVersion()} - ML.NET CLI Tool[/]");
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
        AnsiConsole.MarkupLine("  [green]sample[/]      Create retraining datasets from predictions");
        AnsiConsole.MarkupLine("  [green]trigger[/]     Evaluate retraining triggers for models");
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
        AnsiConsole.MarkupLine("  [green]prep[/]        Data preprocessing pipeline");
        AnsiConsole.MarkupLine("  [green]update[/]      Check for and install CLI updates");
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
