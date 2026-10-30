using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Which input columns a trained model relies on — <c>mloop explain</c>. The model scores a labeled file, each feature column's values are
/// shuffled among the rows in turn (<see cref="PermutationImportance"/>), and the file goes through exactly the path <c>mloop evaluate</c>
/// takes — loader, schema, weights, metric — so a column's importance is the fall of the number <c>evaluate</c> reports.
/// </summary>
public static class ModelExplanation
{
    public const int DefaultRepeats = 3;

    /// <summary>Rows scored at most; a larger file is sampled (seeded) — every column costs <c>repeats</c> evaluations.</summary>
    public const int DefaultRows = 2_000;

    public const int DefaultSeed = 1;

    /// <summary>The tasks whose score is one number a shuffled column can lower.</summary>
    public static bool Supports(string? task)
        => TaskTypes.Canonical(task) is "binary-classification" or "multiclass-classification" or "regression";

    /// <summary>What was measured.</summary>
    /// <param name="Metric">The metric the falls are in.</param>
    /// <param name="HigherIsBetter">Whether a higher <paramref name="Metric"/> is better.</param>
    /// <param name="Baseline">The metric on the unshuffled rows.</param>
    /// <param name="Rows">Rows scored.</param>
    /// <param name="Repeats">Shuffles per column.</param>
    /// <param name="Features">The shuffled columns, most relied-on first.</param>
    /// <param name="NotShuffled">Columns of the file the model does not read as features (excluded or ignored at training) or the file lacks.</param>
    public sealed record Explanation(
        string Metric, bool HigherIsBetter, double Baseline, int Rows, int Repeats,
        IReadOnlyList<FeatureImportance> Features, IReadOnlyList<string> NotShuffled);

    public static async Task<Explanation> ExplainAsync(
        EvaluationEngine engine, string modelPath, string dataFile, ExperimentData experiment, string? weightColumn,
        int repeats = DefaultRepeats, int maxRows = DefaultRows, int seed = DefaultSeed,
        Action<int, int>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(experiment);
        if (!Supports(experiment.Task))
            throw new NotSupportedException(
                $"Permutation importance explains binary-classification, multiclass-classification and regression models; this one is {experiment.Task}.");
        if (repeats < 1)
            throw new ArgumentOutOfRangeException(nameof(repeats), repeats, "At least one shuffle per column.");
        if (maxRows < 2)
            throw new ArgumentOutOfRangeException(nameof(maxRows), maxRows, "At least two rows: one row cannot be shuffled.");

        var (header, rows) = await ReadAsync(dataFile, cancellationToken).ConfigureAwait(false);
        var label = experiment.Config.LabelColumn;
        if (!header.Any(h => h.Equals(label, StringComparison.OrdinalIgnoreCase)))
            throw new ArgumentException($"'{dataFile}' has no label column '{label}' — the score of a model needs the answers. Columns: {string.Join(", ", header)}.");
        if (rows.Count < 2)
            throw new ArgumentException($"'{dataFile}' has {rows.Count} row(s); importance needs a labeled file with at least two.");
        if (rows.Count > maxRows)
            rows = [.. PermutationImportance.Order(rows.Count, seed).Take(maxRows).Order().Select(i => rows[i])];

        var features = FeatureColumns(header, experiment, weightColumn, out var notShuffled);
        var directory = Path.Combine(Path.GetTempPath(), "mloop-explain-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var schema = experiment.Config.InputSchema;
            async Task<Dictionary<string, double>> ScoreAsync(string name, IReadOnlyList<string[]> table)
            {
                var path = Path.Combine(directory, name);
                await WriteAsync(path, header, table, cancellationToken).ConfigureAwait(false);
                var evaluation = await engine.EvaluateAsync(
                    modelPath, path, label, experiment.Task, cancellationToken, schema,
                    experiment.Config.GroupColumn, experiment.Config.UserColumn, experiment.Config.ItemColumn, weightColumn).ConfigureAwait(false);
                return evaluation.Metrics;
            }

            var baselineMetrics = await ScoreAsync("baseline.csv", rows).ConfigureAwait(false);
            var primary = TaskMetadata.ResolvePrimaryMetric(baselineMetrics, experiment.Config.Metric, experiment.Task)
                ?? throw new InvalidOperationException("The evaluation reported no metric to score the shuffles by.");
            var higherIsBetter = !MetricDirection.IsLowerBetter(primary.Name);

            var scores = new List<(string Column, IReadOnlyList<double> Scores)>();
            var done = 0;
            foreach (var (column, index) in features)
            {
                var shuffled = new List<double>(repeats);
                for (var r = 0; r < repeats; r++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var order = PermutationImportance.Order(rows.Count, PermutationImportance.SeedOf(seed, index, r));
                    var table = new List<string[]>(rows.Count);
                    for (var i = 0; i < rows.Count; i++)
                    {
                        var copy = (string[])rows[i].Clone();
                        copy[index] = rows[order[i]][index];
                        table.Add(copy);
                    }
                    var metrics = await ScoreAsync($"shuffle-{index}-{r}.csv", table).ConfigureAwait(false);
                    if (metrics.TryGetValue(primary.Name, out var value) && double.IsFinite(value))
                        shuffled.Add(value);
                    progress?.Invoke(++done, features.Count * repeats);
                }
                scores.Add((column, shuffled));
            }

            return new Explanation(
                primary.Name, higherIsBetter, primary.Value, rows.Count, repeats,
                PermutationImportance.Summarize(primary.Value, higherIsBetter, scores), notShuffled);
        }
        finally
        {
            try { Directory.Delete(directory, recursive: true); } catch { /* temp files; the OS reclaims them */ }
        }
    }

    // The columns of the file to shuffle: those the model read as features. A column the saved schema calls Ignore or Exclude (an
    // identifier, a date, a constant) is not shuffled — the model never saw it — and the label and the weight are not features.
    private static List<(string Column, int Index)> FeatureColumns(
        IReadOnlyList<string> header, ExperimentData experiment, string? weightColumn, out List<string> notShuffled)
    {
        var structural = new[] { experiment.Config.LabelColumn, weightColumn, experiment.Config.GroupColumn, experiment.Config.UserColumn, experiment.Config.ItemColumn }
            .Where(c => !string.IsNullOrWhiteSpace(c)).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var read = experiment.Config.InputSchema?.Columns
            .Where(c => c.Purpose == "Feature").Select(c => c.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var features = new List<(string, int)>();
        notShuffled = [];
        for (var i = 0; i < header.Count; i++)
        {
            var name = header[i].Trim();
            if (structural.Contains(name))
                continue;
            if (read is not null && !read.Contains(name))
                notShuffled.Add(name);
            else
                features.Add((name, i));
        }
        return features;
    }

    private static async Task<(string[] Header, List<string[]> Rows)> ReadAsync(string csvPath, CancellationToken cancellationToken)
    {
        var (readPath, detection) = EncodingDetector.ConvertToUtf8WithBom(csvPath);
        try
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture) { HasHeaderRecord = true, MissingFieldFound = null, BadDataFound = null };
            using var reader = new StreamReader(readPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            using var csv = new CsvReader(reader, config);
            if (!await csv.ReadAsync().ConfigureAwait(false) || !csv.ReadHeader() || csv.HeaderRecord is null)
                throw new ArgumentException($"'{csvPath}' has no header row.");
            var header = csv.HeaderRecord;
            var rows = new List<string[]>();
            while (await csv.ReadAsync().ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var record = csv.Parser.Record ?? [];
                var row = new string[header.Length];
                for (var i = 0; i < header.Length; i++)
                    row[i] = i < record.Length ? record[i] : "";
                rows.Add(row);
            }
            return (header, rows);
        }
        finally
        {
            if (detection.WasConverted && File.Exists(readPath))
                File.Delete(readPath);
        }
    }

    private static async Task WriteAsync(string path, IReadOnlyList<string> header, IReadOnlyList<string[]> rows, CancellationToken cancellationToken)
    {
        await using var writer = new StreamWriter(path, false, new System.Text.UTF8Encoding(false));
        await using var output = new CsvWriter(writer, CultureInfo.InvariantCulture);
        foreach (var h in header)
            output.WriteField(h);
        await output.NextRecordAsync().ConfigureAwait(false);
        foreach (var row in rows)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var field in row)
                output.WriteField(field);
            await output.NextRecordAsync().ConfigureAwait(false);
        }
    }
}
