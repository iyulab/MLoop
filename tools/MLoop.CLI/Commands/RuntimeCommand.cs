using System.CommandLine;
using MLoop.Core.Runtime;
using Spectre.Console;

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
        var command = new Command("list", "List available and installed runtimes");
        command.SetAction(_ =>
        {
            var manager = new RuntimeManager();
            var table = new Table();
            table.AddColumn("ID");
            table.AddColumn("Name");
            table.AddColumn("Status");
            table.AddColumn("Size");
            table.AddColumn("Tasks");

            foreach (var runtime in RuntimeRegistry.All)
            {
                var status = manager.GetStatus(runtime);
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
                AnsiConsole.MarkupLine($"[red]Error:[/] Unknown runtime '{runtimeId}'. Available: {string.Join(", ", RuntimeRegistry.All.Select(r => r.Id))}");
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
                AnsiConsole.MarkupLine($"[red]Error:[/] Unknown runtime '{runtimeId}'");
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
}
