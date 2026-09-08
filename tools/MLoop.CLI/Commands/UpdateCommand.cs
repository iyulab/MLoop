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
            return await ExecuteAsync(checkOnly);
        });

        return command;
    }

    /// <summary>
    /// Returns the process exit code: <c>0</c> when the check or the update succeeded, non-zero
    /// when the version could not be checked or the update did not install.
    /// </summary>
    /// <remarks>
    /// A failed update reported "Update failed." and exited <c>0</c>. `mloop update` is the command
    /// most often written into a provisioning script or a scheduled job, where the exit code is the
    /// only thing read — so a self-update that silently did not happen looked identical to one that
    /// did, and the next run kept the old binary without anyone learning why.
    /// </remarks>
    private static async Task<int> ExecuteAsync(bool checkOnly)
    {
        AnsiConsole.MarkupLine("[blue]Checking for updates...[/]");

        var info = await UpdateChecker.CheckForUpdateAsync(forceCheck: true);

        if (info is null)
        {
            ErrorConsole.Error(
                "Failed to check for updates.",
                "Check the network connection, then run 'mloop update' again.");
            return 1;
        }

        AnsiConsole.MarkupLine($"  Current version: [grey]{info.CurrentVersion}[/]");
        AnsiConsole.MarkupLine($"  Latest version:  [green]{info.LatestVersion}[/]");

        if (!info.UpdateAvailable)
        {
            AnsiConsole.MarkupLine("[green]You are running the latest version.[/]");
            return 0;
        }

        AnsiConsole.MarkupLine($"[yellow]A new version (v{info.LatestVersion}) is available![/]");

        if (checkOnly)
        {
            AnsiConsole.MarkupLine("Run [blue]mloop update[/] to install the update.");
            return 0;
        }

        var installMethod = InstallDetector.Detect();

        return installMethod switch
        {
            InstallMethod.DotnetTool => await UpdateViaDotnetToolAsync(),
            InstallMethod.StandaloneBinary => await UpdateViaStandaloneBinaryAsync(info.LatestVersion),
            // An install this build cannot update is not a success — say which, and exit non-zero
            // so a script does not carry on as though the new version were in place.
            _ => Unsupported(installMethod),
        };
    }

    private static int Unsupported(InstallMethod installMethod)
    {
        ErrorConsole.Error(
            $"Cannot self-update an installation of type '{installMethod}'.",
            "Reinstall with the method you originally used, or install the dotnet tool: "
            + "dotnet tool update -g mloop");
        return 1;
    }

    private static async Task<int> UpdateViaDotnetToolAsync()
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
            ErrorConsole.Error(
                "Failed to start 'dotnet tool update -g mloop'.",
                "Check that the .NET SDK is on PATH.");
            return 1;
        }

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode == 0)
        {
            AnsiConsole.MarkupLine("[green]Update completed successfully![/]");
            if (!string.IsNullOrWhiteSpace(output))
                AnsiConsole.WriteLine(output.TrimEnd());
            return 0;
        }

        ErrorConsole.Error(
            $"Update failed (dotnet tool update exited {process.ExitCode}).",
            string.IsNullOrWhiteSpace(error) ? null : error.TrimEnd());
        return process.ExitCode;
    }

    private static async Task<int> UpdateViaStandaloneBinaryAsync(string version)
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
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "update");

            // Cleanup temp file
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch (IOException) { }
            }

            return 1;
        }
    }
}
