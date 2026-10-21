using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Display;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop rollback - Returns production to the experiment it held before the current one.
/// </summary>
/// <remarks>
/// Promotion already backed up the previous production and recorded it in the promotion history, but
/// going back was a manual copy. The target is decided by <see cref="RollbackTarget"/>, which the REST
/// API reads as well.
/// </remarks>
public static class RollbackCommand
{
    public static Command Create()
    {
        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var toOption = new Option<string?>("--to")
        {
            Description = "Experiment to return to; it must have been in production before (default: the one the current production replaced)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the result as JSON (machine-readable, non-interactive)"
        };

        var command = new Command("rollback", "Return production to the experiment it held before");
        command.Options.Add(nameOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(nameOption),
            parseResult.GetValue(toOption),
            parseResult.GetValue(jsonOption)));

        return command;
    }

    private static async Task<int> ExecuteAsync(string? modelName, string? to, bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                WriteError("Not inside a MLoop project. Run 'mloop init' to create a new project.", jsonOutput);
                return 1;
            }

            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
            var promotionManager = new FilePromotionManager(projectRoot);

            var current = (await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None))?.ExperimentId;
            var history = await promotionManager.GetHistoryAsync(resolvedModelName, CancellationToken.None);
            var (target, error) = RollbackTarget.Resolve(history, current, to?.Trim());
            if (target is null)
            {
                WriteError(error!, jsonOutput);
                return 1;
            }

            if (!experimentStore.ExperimentExists(resolvedModelName, target))
            {
                WriteError($"'{target}' is no longer in staging, so it cannot be restored from there. "
                           + $"Its copy from when it was production is under models/{resolvedModelName}/backups/.", jsonOutput);
                return 1;
            }

            var backupPath = await promotionManager.BackupProductionAsync(resolvedModelName);
            await modelRegistry.PromoteAsync(resolvedModelName, target, MLoop.Ops.Interfaces.PromotionActions.Rollback, cancellationToken: CancellationToken.None);

            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    Model = resolvedModelName,
                    ExperimentId = target,
                    Status = "rolled-back",
                    PreviousProduction = current,
                    BackupPath = backupPath,
                }, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            }
            else
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLineInterpolated($"[green]>[/] Production for [cyan]{resolvedModelName}[/] is back to [cyan]{target}[/] (was [cyan]{current}[/])");
                if (backupPath != null)
                    ValueLine.Write("[green]>[/] Backup: ", backupPath);
                AnsiConsole.WriteLine();
            }

            return 0;
        }
        catch (Exception ex)
        {
            if (jsonOutput)
            {
                WriteError(ex.Message, jsonOutput: true);
                return 1;
            }
            ErrorSuggestions.DisplayError(ex, "rollback");
            return 1;
        }
    }

    private static void WriteError(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            JsonError.Emit(message);
            Console.Error.WriteLine(message);
        }
        else
        {
            ErrorConsole.Out.MarkupLineInterpolated($"[red]Error:[/] {message}");
        }
    }
}
