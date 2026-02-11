using System.CommandLine;
using System.Diagnostics;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Update;
using Spectre.Console;

namespace MLoop.CLI.Commands;

public static class UpdateCommand
{
    public static Command Create()
    {
        var checkOption = new Option<bool>("--check", "-c")
        {
            Description = "Check for updates without installing",
            DefaultValueFactory = _ => false
        };

        var command = new Command("update", "Check for and install MLoop CLI updates");
        command.Options.Add(checkOption);

        command.SetAction(async (parseResult) =>
        {
            var checkOnly = parseResult.GetValue(checkOption);
            await ExecuteAsync(checkOnly);
        });

        return command;
    }

    private static async Task ExecuteAsync(bool checkOnly)
    {
        AnsiConsole.MarkupLine("[blue]Checking for updates...[/]");

        var info = await UpdateChecker.CheckForUpdateAsync(forceCheck: true);

        if (info is null)
        {
            AnsiConsole.MarkupLine("[red]Failed to check for updates. Please check your internet connection.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"  Current version: [grey]{info.CurrentVersion}[/]");
        AnsiConsole.MarkupLine($"  Latest version:  [green]{info.LatestVersion}[/]");

        if (!info.UpdateAvailable)
        {
            AnsiConsole.MarkupLine("[green]You are running the latest version.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"[yellow]A new version (v{info.LatestVersion}) is available![/]");

        if (checkOnly)
        {
            AnsiConsole.MarkupLine("Run [blue]mloop update[/] to install the update.");
            return;
        }

        var installMethod = InstallDetector.Detect();

        switch (installMethod)
        {
            case InstallMethod.DotnetTool:
                await UpdateViaDotnetToolAsync();
                break;

            case InstallMethod.StandaloneBinary:
                await UpdateViaStandaloneBinaryAsync(info.LatestVersion);
                break;
        }
    }

    private static async Task UpdateViaDotnetToolAsync()
    {
        AnsiConsole.MarkupLine("[blue]Updating via dotnet tool...[/]");

        var psi = new ProcessStartInfo("dotnet", "tool update -g mloop")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        var process = Process.Start(psi);
        if (process is null)
        {
            AnsiConsole.MarkupLine("[red]Failed to start dotnet tool update.[/]");
            return;
        }

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode == 0)
        {
            AnsiConsole.MarkupLine("[green]Update completed successfully![/]");
            if (!string.IsNullOrWhiteSpace(output))
                AnsiConsole.WriteLine(output.TrimEnd());
        }
        else
        {
            AnsiConsole.MarkupLine("[red]Update failed.[/]");
            if (!string.IsNullOrWhiteSpace(error))
                AnsiConsole.MarkupLine($"[red]{error.TrimEnd()}[/]");
        }
    }

    private static async Task UpdateViaStandaloneBinaryAsync(string version)
    {
        var rid = InstallDetector.GetRuntimeIdentifier();
        var ext = rid.StartsWith("win") ? ".exe" : "";
        var tempPath = Path.Combine(Path.GetTempPath(), $"mloop-update{ext}");

        try
        {
            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask($"[blue]Downloading mloop-{rid}{ext}[/]");

                    await UpdateChecker.DownloadLatestBinaryAsync(version, rid, tempPath,
                        (downloaded, total) =>
                        {
                            if (total.HasValue && total.Value > 0)
                            {
                                task.MaxValue = total.Value;
                                task.Value = downloaded;
                            }
                        });

                    task.StopTask();
                });

            AnsiConsole.MarkupLine("[blue]Replacing executable...[/]");
            UpdateChecker.ReplaceExecutable(tempPath);
            AnsiConsole.MarkupLine("[green]Update completed successfully![/]");
            AnsiConsole.MarkupLine("[grey]Please restart mloop to use the new version.[/]");
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "update");

            // Cleanup temp file
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch { }
            }
        }
    }
}
