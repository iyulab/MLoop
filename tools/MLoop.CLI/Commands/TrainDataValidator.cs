using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Models;
using FilePrepper.Pipeline;
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

    /// <summary>
    /// Applies data sampling if the dataset exceeds maxRows.
    /// Uses task-aware default strategy: stratified for classification, random for others.
    /// Returns the (possibly sampled) data file path.
    /// </summary>
    public static async Task<string> ApplySamplingIfNeededAsync(
        string dataFilePath,
        int maxRows,
        string taskType,
        string? labelColumn,
        string? explicitStrategy,
        int seed,
        string projectRoot)
    {
        // Count rows
        int rowCount = 0;
        using (var reader = new StreamReader(dataFilePath))
        {
            await reader.ReadLineAsync().ConfigureAwait(false); // Skip header
            while (await reader.ReadLineAsync().ConfigureAwait(false) != null)
                rowCount++;
        }

        if (rowCount <= maxRows)
            return dataFilePath;

        // Determine sampling strategy
        var strategy = explicitStrategy?.ToLowerInvariant();
        if (string.IsNullOrEmpty(strategy))
        {
            // Task-aware default: classification → stratified, others → random
            strategy = taskType.ToLowerInvariant() switch
            {
                "binary-classification" or "multiclass-classification"
                    or "binaryclassification" or "multiclassclassification"
                    or "classification" => "stratified",
                _ => "random"
            };
        }

        // Stratified requires a label column; fall back to random if unavailable
        if (strategy == "stratified" && string.IsNullOrEmpty(labelColumn))
            strategy = "random";

        // Use FilePrepper DataPipeline for sampling
        var pipeline = await FilePrepper.Pipeline.DataPipeline.FromCsvAsync(dataFilePath).ConfigureAwait(false);
        var totalRows = pipeline.RowCount;
        var random = new Random(seed);

        FilePrepper.Pipeline.DataPipeline sampled;
        string strategyLabel;

        if (strategy == "stratified" && !string.IsNullOrEmpty(labelColumn))
        {
            // Stratified: proportional sampling per class
            var ratio = (double)maxRows / totalRows;
            var groups = new Dictionary<string, List<int>>();
            int scanIdx = 0;

            pipeline.FilterRows(row =>
            {
                var key = row.GetValueOrDefault(labelColumn, "");
                if (!groups.ContainsKey(key))
                    groups[key] = [];
                groups[key].Add(scanIdx++);
                return true;
            });

            var selected = new HashSet<int>();
            foreach (var group in groups.Values)
            {
                var groupSize = Math.Max(1, (int)(group.Count * ratio));
                foreach (var idx in group.OrderBy(_ => random.NextDouble()).Take(groupSize))
                    selected.Add(idx);
            }

            int filterIdx = 0;
            sampled = pipeline.FilterRows(_ => selected.Contains(filterIdx++));
            strategyLabel = $"stratified by {labelColumn}";
        }
        else
        {
            // Random sampling
            var indices = Enumerable.Range(0, totalRows).ToArray();
            for (int i = totalRows - 1; i > 0 && i >= totalRows - maxRows; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            var selected = new HashSet<int>(indices.Skip(Math.Max(0, totalRows - maxRows)));

            int filterIdx = 0;
            sampled = pipeline.FilterRows(_ => selected.Contains(filterIdx++));
            strategyLabel = "random";
        }

        // Write sampled data
        var outputDir = Path.GetDirectoryName(dataFilePath)!;
        var sampledPath = Path.Combine(outputDir, "train_sampled.csv");
        await sampled.ToCsvAsync(sampledPath).ConfigureAwait(false);

        AnsiConsole.MarkupLine(
            $"[blue]Sampled[/] [cyan]{rowCount:N0}[/] → [cyan]{sampled.RowCount:N0}[/] rows ({strategyLabel}, seed={seed})");

        return sampledPath;
    }
}
