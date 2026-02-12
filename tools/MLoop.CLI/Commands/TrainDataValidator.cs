using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Handles data file resolution and label validation for train command.
/// Extracted from TrainCommand to reduce file size and improve separation of concerns.
/// </summary>
internal static class TrainDataValidator
{
    /// <summary>
    /// Validates that the label column exists in the data file.
    /// Throws ArgumentException with helpful message if not found.
    /// </summary>
    public static async Task ValidateLabelColumnAsync(string dataFilePath, string labelColumn, string modelName)
    {
        var csvHelper = new CsvHelperImpl();
        var data = await csvHelper.ReadAsync(dataFilePath);

        if (data.Count == 0)
        {
            throw new InvalidOperationException($"Data file is empty: {dataFilePath}");
        }

        var firstRow = data[0];
        var availableColumns = firstRow.Keys.ToArray();

        if (!firstRow.ContainsKey(labelColumn))
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[red]Error:[/] Label column not found in data for model '[cyan]{modelName}[/]'");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"  [yellow]Label specified:[/] '{labelColumn}'");
            AnsiConsole.MarkupLine($"  [yellow]Available columns:[/] {string.Join(", ", availableColumns)}");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[yellow]Tip:[/] Update the label in mloop.yaml or use --label option for --name {modelName}");
            AnsiConsole.WriteLine();

            throw new ArgumentException(
                $"Label column '{labelColumn}' not found in data for model '{modelName}'.\n" +
                $"Available columns: {string.Join(", ", availableColumns)}",
                nameof(labelColumn));
        }
    }

    /// <summary>
    /// Resolves the data file path from various sources (explicit path, config, or auto-discovery).
    /// </summary>
    public static Task<string?> ResolveDataFileAsync(
        string? dataFile,
        MLoopConfig userConfig,
        string projectRoot,
        IDatasetDiscovery datasetDiscovery,
        IFileSystemManager fileSystem)
    {
        if (!string.IsNullOrEmpty(dataFile))
        {
            var resolvedPath = Path.IsPathRooted(dataFile)
                ? dataFile
                : Path.Combine(projectRoot, dataFile);

            return File.Exists(resolvedPath)
                ? Task.FromResult<string?>(resolvedPath)
                : Task.FromResult<string?>(null);
        }

        // Try config data path
        if (!string.IsNullOrEmpty(userConfig.Data?.Train))
        {
            var configPath = Path.IsPathRooted(userConfig.Data.Train)
                ? userConfig.Data.Train
                : Path.Combine(projectRoot, userConfig.Data.Train);

            if (File.Exists(configPath))
            {
                return Task.FromResult<string?>(configPath);
            }
        }

        // Auto-discover datasets/train.csv
        var datasets = datasetDiscovery.FindDatasets(projectRoot);
        return Task.FromResult(datasets?.TrainPath);
    }
}
