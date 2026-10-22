using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Search.Candidates;
using MLoop.Tests.Common;
using Xunit.Abstractions;

namespace MLoop.Core.Tests.Search;

/// <summary>Runs only when <c>MLOOP_SEARCH_GATE=1</c>: the measurement takes many minutes and is read, not merely passed.</summary>
public sealed class SearchGateFactAttribute : FactAttribute
{
    public SearchGateFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("MLOOP_SEARCH_GATE") != "1")
            Skip = "The search comparison runs on request: set MLOOP_SEARCH_GATE=1.";
    }
}

/// <summary>
/// The comparison of MLoop's trainer search with AutoML's, through the training path both run on — the
/// same loader, split, column roles and test evaluation. Three measurements:
/// <list type="number">
/// <item><b>Search quality</b> — the trial count held equal, five seeds each: the distribution of the test
/// partition's metric per data set and search.</item>
/// <item><b>Time budget</b> — the same budgets on both: trials that scored, and the time a run took against
/// the time it was given (the refit included).</item>
/// <item><b>Determinism</b> — MLoop's search twice under one seed proposes and scores the same trials.</item>
/// </list>
/// Only the third is asserted; the first two are written as a report (<c>MLOOP_SEARCH_GATE_OUT</c>, a
/// directory, else the test output) whose numbers set the gate's tolerances.
/// </summary>
/// <remarks>
/// Data sets: three generated binary tables, plus every <c>name.csv</c> with a <c>name.label</c> beside it
/// (the label column's name) in <c>MLOOP_SEARCH_GATE_DATA</c>. LightGBM is a candidate only where its native
/// library loads, so a machine without it compares four trainers, not five.
/// </remarks>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public sealed class SearchGateMeasurement : IDisposable
{
    private static readonly int[] Seeds = [1, 2, 3, 4, 5];
    private const int Trials = 20;
    private const int QualityBudgetSeconds = 600;
    private static readonly int[] Budgets = [10, 30];

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_gate_{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _output;

    public SearchGateMeasurement(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private sealed record DataSet(string Name, string Path, string Label);

    private sealed record Run(string DataSet, string Engine, int Seed, int? MaxTrials, int Budget,
        string Metric, double? Value, int Trials, int Scored, double Seconds, string? Error);

    [SearchGateFact]
    public async Task Measure()
    {
        var sets = DataSets();
        var runs = new List<Run>();

        foreach (var set in sets)
        {
            foreach (var seed in Seeds)
            {
                runs.Add(await TrainAsync(set, SearchEngine.AutoML, seed, Trials, QualityBudgetSeconds));
                runs.Add(await TrainAsync(set, SearchEngine.MLoop, seed, Trials, QualityBudgetSeconds));
            }
            foreach (var budget in Budgets)
            {
                runs.Add(await TrainAsync(set, SearchEngine.AutoML, 1, null, budget));
                runs.Add(await TrainAsync(set, SearchEngine.MLoop, 1, null, budget));
            }
        }

        var determinism = new List<(string DataSet, bool Same)>();
        foreach (var set in sets)
        {
            var first = await TrialListAsync(set);
            var second = await TrialListAsync(set);
            determinism.Add((set.Name, first == second));
        }

        Report(runs, determinism);
        Assert.All(determinism, d => Assert.True(d.Same, $"{d.DataSet}: one seed gave two different searches"));
    }

    private IReadOnlyList<DataSet> DataSets()
    {
        var sets = new List<DataSet>
        {
            Generate("numeric-2k", 2000, (rng, i) => { var x = rng.NextDouble(); var z = rng.NextDouble(); return ([F(x), F(z), F(rng.NextDouble())], x + 0.5 * z + rng.NextDouble() * 0.6 > 1.0); }, "x,z,noise"),
            Generate("mixed-2k", 2000, (rng, i) => { string[] c = ["red", "green", "blue", "grey", "black"]; var x = rng.NextDouble(); var colour = c[rng.Next(c.Length)]; return ([F(x), colour, F(rng.NextDouble())], x + (colour is "red" or "blue" ? 0.35 : 0) + rng.NextDouble() * 0.5 > 0.95); }, "x,colour,noise"),
            Generate("imbalanced-5k", 5000, (rng, i) => { var x = rng.NextDouble(); var z = rng.NextDouble(); return ([F(x), F(z), F(rng.NextDouble())], x * z + rng.NextDouble() * 0.15 > 0.8); }, "x,z,noise"),
        };

        if (Environment.GetEnvironmentVariable("MLOOP_SEARCH_GATE_DATA") is { Length: > 0 } dir && Directory.Exists(dir))
        {
            foreach (var csv in Directory.EnumerateFiles(dir, "*.csv").Order(StringComparer.Ordinal))
            {
                var labelFile = System.IO.Path.ChangeExtension(csv, ".label");
                if (File.Exists(labelFile))
                    sets.Add(new DataSet(System.IO.Path.GetFileNameWithoutExtension(csv), csv, File.ReadAllText(labelFile).Trim()));
            }
        }
        return sets;
    }

    private DataSet Generate(string name, int rows, Func<Random, int, (string[] Features, bool Label)> row, string header)
    {
        var rng = new Random(17);
        var lines = new List<string>(rows + 1) { header + ",label" };
        for (var i = 0; i < rows; i++)
        {
            var (features, label) = row(rng, i);
            lines.Add(string.Join(",", features) + "," + (label ? "yes" : "no"));
        }
        var path = System.IO.Path.Combine(_dir, name + ".csv");
        File.WriteAllLines(path, lines);
        return new DataSet(name, path, "label");
    }

    private static string F(double value) => value.ToString("F4", CultureInfo.InvariantCulture);

    private static TrainingConfig Config(DataSet set, SearchEngine engine, int seed, int? maxTrials, int budget) => new()
    {
        ModelName = "gate",
        DataFile = set.Path,
        LabelColumn = set.Label,
        Task = "binary-classification",
        TimeLimitSeconds = budget,
        Metric = "auc",
        Engine = engine,
        SearchSettings = new SearchSettings(Seed: seed, Threads: 1),
        SearchMaxTrials = maxTrials,
    };

    private static async Task<Run> TrainAsync(DataSet set, SearchEngine engine, int seed, int? maxTrials, int budget)
    {
        var ctx = new MLContext(seed: seed);
        var clock = Stopwatch.StartNew();
        try
        {
            var result = await new AutoMLRunner(ctx, new CsvDataLoader(ctx))
                .RunAsync(Config(set, engine, seed, maxTrials, budget), null, CancellationToken.None);
            return new Run(set.Name, engine.ToString(), seed, maxTrials, budget, "auc",
                result.Metrics.TryGetValue("auc", out var auc) ? auc : null,
                result.Trials.Count, result.Trials.Count(t => t.Error is null), clock.Elapsed.TotalSeconds, null);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return new Run(set.Name, engine.ToString(), seed, maxTrials, budget, "auc", null, 0, 0, clock.Elapsed.TotalSeconds, ex.GetBaseException().Message);
        }
    }

    // Everything a trial is but its time: trainer, settings, and every metric, exactly.
    private static async Task<string> TrialListAsync(DataSet set)
    {
        var ctx = new MLContext(seed: 1);
        var result = await new AutoMLRunner(ctx, new CsvDataLoader(ctx))
            .RunAsync(Config(set, SearchEngine.MLoop, 1, 8, QualityBudgetSeconds), null, CancellationToken.None);
        return string.Join("\n", result.Trials.Select(t =>
            $"{t.TrainerName} {string.Join(",", t.Metrics.OrderBy(m => m.Key, StringComparer.Ordinal).Select(m => $"{m.Key}={m.Value:R}"))} {t.Error}"));
    }

    private void Report(IReadOnlyList<Run> runs, IReadOnlyList<(string DataSet, bool Same)> determinism)
    {
        var md = new StringBuilder();
        md.AppendLine("# Trainer search comparison — MLoop vs AutoML (binary)");
        md.AppendLine();
        md.AppendLine(CultureInfo.InvariantCulture, $"Machine: {System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier}, {Environment.ProcessorCount} cores.");
        md.AppendLine();
        md.AppendLine(CultureInfo.InvariantCulture, $"## Search quality — {Trials} trials, seeds {string.Join(" ", Seeds)}, test-partition auc");
        md.AppendLine();
        md.AppendLine("| data | search | mean | min | max | per seed | failed runs |");
        md.AppendLine("|---|---|---|---|---|---|---|");
        foreach (var group in runs.Where(r => r.MaxTrials is not null).GroupBy(r => (r.DataSet, r.Engine)))
        {
            var values = group.Where(r => r.Value is not null).Select(r => r.Value!.Value).ToList();
            md.AppendLine(CultureInfo.InvariantCulture,
                $"| {group.Key.DataSet} | {group.Key.Engine} | {(values.Count > 0 ? values.Average() : double.NaN):F4} | {(values.Count > 0 ? values.Min() : double.NaN):F4} | {(values.Count > 0 ? values.Max() : double.NaN):F4} | {string.Join(" ", group.Select(r => r.Value?.ToString("F4", CultureInfo.InvariantCulture) ?? "—"))} | {group.Count(r => r.Error is not null)} |");
        }
        md.AppendLine();
        md.AppendLine("## Time budget — seed 1, default trial cap");
        md.AppendLine();
        md.AppendLine("| data | search | budget s | took s | trials | scored | auc | error |");
        md.AppendLine("|---|---|---|---|---|---|---|---|");
        foreach (var r in runs.Where(r => r.MaxTrials is null))
        {
            md.AppendLine(CultureInfo.InvariantCulture,
                $"| {r.DataSet} | {r.Engine} | {r.Budget} | {r.Seconds:F1} | {r.Trials} | {r.Scored} | {r.Value?.ToString("F4", CultureInfo.InvariantCulture) ?? "—"} | {r.Error?.Replace("|", "/", StringComparison.Ordinal)} |");
        }
        md.AppendLine();
        md.AppendLine("## Determinism — MLoop, seed 1 twice, 8 trials");
        md.AppendLine();
        foreach (var (set, same) in determinism)
            md.AppendLine(CultureInfo.InvariantCulture, $"- {set}: {(same ? "same trials" : "DIFFERENT")}");

        var text = md.ToString();
        _output.WriteLine(text);
        if (Environment.GetEnvironmentVariable("MLOOP_SEARCH_GATE_OUT") is { Length: > 0 } outDir)
        {
            Directory.CreateDirectory(outDir);
            File.WriteAllText(System.IO.Path.Combine(outDir, "search-gate.md"), text);
            File.WriteAllText(System.IO.Path.Combine(outDir, "search-gate.json"), JsonSerializer.Serialize(new { runs, determinism = determinism.Select(d => new { d.DataSet, d.Same }) }, new JsonSerializerOptions { WriteIndented = true }));
        }
    }
}
