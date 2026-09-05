using System.CommandLine;
using MLoop.Core.Runtime;
using Spectre.Console;
using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.CLI.Commands;

public static class RuntimeCommand
{
    public static Command Create()
    {
        var command = new Command("runtime", "Manage on-demand ML runtime downloads");

        command.Subcommands.Add(CreateListCommand());
        command.Subcommands.Add(CreateInstallCommand());
        command.Subcommands.Add(CreateRemoveCommand());

        return command;
    }

    private static Command CreateListCommand()
    {
        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the runtime list as JSON to stdout instead of the human table. " +
                          "Progress and narration go to stderr."
        };

        var command = new Command("list", "List available and installed runtimes");
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var jsonOutput = parseResult.GetValue(jsonOption);

            // In --json mode stdout must be pure JSON, so route all human-facing Spectre output to
            // stderr — the same reassignment predict/evaluate/validate/status --json use.
            if (jsonOutput)
                AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
                {
                    Out = new AnsiConsoleOutput(Console.Error)
                });

            var manager = new RuntimeManager();
            var table = new Table();
            table.AddColumn("ID");
            table.AddColumn("Name");
            table.AddColumn("Status");
            table.AddColumn("Size");
            table.AddColumn("Tasks");

            // Collect statuses once so the human table and --json consume the same data, rather
            // than computing GetStatus twice.
            var statuses = RuntimeRegistry.All
                .Select(runtime => (Runtime: runtime, Status: manager.GetStatus(runtime)))
                .ToList();

            foreach (var (runtime, status) in statuses)
            {
                var statusText = status.Installed
                    ? "[green]Installed[/]"
                    : $"[dim]Not installed (~{runtime.ApproximateSizeMB}MB)[/]";

                var sizeText = status.Installed
                    ? $"{status.SizeBytes / (1024 * 1024)}MB"
                    : "-";

                table.AddRow(
                    runtime.Id,
                    runtime.DisplayName,
                    statusText,
                    sizeText,
                    string.Join(", ", runtime.RequiredByTasks));
            }

            AnsiConsole.Write(table);

            if (jsonOutput)
            {
                var payload = new
                {
                    Runtimes = statuses.Select(s => new
                    {
                        s.Runtime.Id,
                        s.Runtime.DisplayName,
                        s.Status.Installed,
                        SizeBytes = s.Status.Installed ? s.Status.SizeBytes : (long?)null,
                        s.Runtime.ApproximateSizeMB,
                        s.Runtime.RequiredByTasks
                    })
                };
                Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(payload, new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
                }));
            }

            return Task.FromResult(0);
        });

        return command;
    }

    private static Command CreateInstallCommand()
    {
        var runtimeArg = new Argument<string>("runtime")
        {
            Description = "Runtime to install (tf, torch)",
            Arity = ArgumentArity.ExactlyOne
        };
        var quietOption = new Option<bool>("--quiet", "-q")
        {
            Description = "Suppress interactive output"
        };

        var command = new Command("install", "Download and install a runtime");
        command.Arguments.Add(runtimeArg);
        command.Options.Add(quietOption);

        command.SetAction(async (parseResult) =>
        {
            var runtimeId = parseResult.GetValue(runtimeArg);
            var quiet = parseResult.GetValue(quietOption);
            var runtime = RuntimeRegistry.GetById(runtimeId!);

            if (runtime == null)
            {
                ErrorConsole.Error(UnknownRuntime(runtimeId));
                return 1;
            }

            var manager = new RuntimeManager();

            if (manager.IsInstalled(runtime))
            {
                AnsiConsole.MarkupLine($"[green]✓[/] {runtime.DisplayName} is already installed");
                return 0;
            }

            if (quiet)
            {
                var progress = new Progress<RuntimeDownloadProgress>(p =>
                {
                    if (p.Phase == DownloadPhase.Complete)
                        Console.WriteLine(p.Message);
                });
                await manager.InstallAsync(runtime, progress);
            }
            else
            {
                await AnsiConsole.Progress()
                    .AutoClear(false)
                    .Columns(
                        new TaskDescriptionColumn(),
                        new ProgressBarColumn(),
                        new PercentageColumn(),
                        new SpinnerColumn())
                    .StartAsync(async ctx =>
                    {
                        var task = ctx.AddTask($"Installing {runtime.DisplayName}", maxValue: 100);
                        var progress = new Progress<RuntimeDownloadProgress>(p =>
                        {
                            task.Description = p.Message;
                            if (p.TotalBytes > 0)
                                task.Value = p.PercentComplete;
                            else if (p.Phase == DownloadPhase.Extracting)
                                task.Value = 90;
                            else if (p.Phase == DownloadPhase.Verifying)
                                task.Value = 95;
                            else if (p.Phase == DownloadPhase.Complete)
                                task.Value = 100;
                        });

                        await manager.InstallAsync(runtime, progress);
                    });

                AnsiConsole.MarkupLine($"[green]✓[/] {runtime.DisplayName} installed successfully");
            }

            return 0;
        });

        return command;
    }

    private static Command CreateRemoveCommand()
    {
        var runtimeArg = new Argument<string>("runtime") { Description = "Runtime to remove (tf, torch)" };
        var command = new Command("remove", "Remove an installed runtime");
        command.Arguments.Add(runtimeArg);

        command.SetAction((parseResult) =>
        {
            var runtimeId = parseResult.GetValue(runtimeArg);
            var runtime = RuntimeRegistry.GetById(runtimeId!);

            if (runtime == null)
            {
                ErrorConsole.Error(UnknownRuntime(runtimeId));
                return Task.FromResult(1);
            }

            var manager = new RuntimeManager();
            if (!manager.IsInstalled(runtime))
            {
                AnsiConsole.MarkupLine($"[dim]{runtime.DisplayName} is not installed[/]");
                return Task.FromResult(0);
            }

            manager.Remove(runtime);
            AnsiConsole.MarkupLine($"[green]✓[/] {runtime.DisplayName} removed");
            return Task.FromResult(0);
        });

        return command;
    }

    /// <summary>
    /// The rejection for an unrecognised runtime id, listing what the user could have typed. Shared
    /// by <c>install</c> and <c>remove</c>: worded per command, only one of them named the valid ids
    /// and the other left the user guessing.
    /// </summary>
    private static string UnknownRuntime(string? runtimeId) =>
        $"Unknown runtime '{runtimeId}'. Available: {string.Join(", ", RuntimeRegistry.All.Select(r => r.Id))}";
}
