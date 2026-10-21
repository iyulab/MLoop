using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// The baseline a candidate was compared with on the same rows: its metrics there, and the paired
/// per-row comparison when the task has a per-row measure (<c>null</c> otherwise, or when the two
/// models did not answer the same number of rows, with the reason in <paramref name="NotPaired"/>).
/// </summary>
public sealed record Comparison(string Against, Dictionary<string, double> Metrics, PairedResult? Paired, string? NotPaired);

/// <summary>A model resolved to compare with, or why none could be.</summary>
/// <param name="NotFound">The error is that the experiment (or a production model) does not exist.</param>
public sealed record ComparisonBaseline(string? ExperimentId, string? ModelPath, ExperimentData? Data, string? Error, bool NotFound = false);

/// <summary>
/// What <c>mloop evaluate</c> and <c>POST /evaluate</c> both decide before and after scoring: which
/// weight the test rows count by, which model a candidate is compared with, and the comparison itself.
/// One place, so the command and the endpoint cannot weigh the same file differently.
/// </summary>
public static class ModelComparison
{
    /// <summary>
    /// The weight the test rows count by: <paramref name="requested"/> when given, else the column
    /// <paramref name="experiment"/> was trained with when <paramref name="testDataFile"/> carries it.
    /// When the experiment was trained with a weight the file lacks, the weight is <c>null</c> and
    /// <c>Warning</c> says the training and test metrics describe different distributions.
    /// </summary>
    public static (string? WeightColumn, string? Warning) ResolveWeight(string? requested, ExperimentData experiment, string testDataFile)
    {
        ArgumentNullException.ThrowIfNull(experiment);
        if (!string.IsNullOrWhiteSpace(requested))
            return (requested.Trim(), null);
        var trained = experiment.Config.WeightColumn;
        if (string.IsNullOrEmpty(trained))
            return (null, null);
        return HeaderHas(testDataFile, trained)
            ? (trained, null)
            : (null, $"The training metrics are weighted by '{trained}' and this file has no such column, so the test " +
                     "metrics are unweighted — the training and test metrics describe different distributions.");
    }

    /// <summary>
    /// <c>production</c> or an experiment id → the model to compare with. Refused when it is the candidate
    /// itself, or when it was trained for another task or label: the same rows would then be asking the
    /// two models different questions.
    /// </summary>
    public static async Task<ComparisonBaseline> ResolveBaselineAsync(
        string against, string modelName, string candidateId, ExperimentData candidate,
        IExperimentStore store, IModelRegistry registry, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        string id;
        if (against.Equals("production", StringComparison.OrdinalIgnoreCase))
        {
            var production = await registry.GetProductionAsync(modelName, cancellationToken);
            if (production is null)
                return new ComparisonBaseline(null, null, null, $"against production: no model is in production for '{modelName}'.", NotFound: true);
            id = production.ExperimentId;
        }
        else
        {
            id = against;
            if (!store.ExperimentExists(modelName, id))
                return new ComparisonBaseline(null, null, null, $"--against: experiment not found: {id} for model '{modelName}'.", NotFound: true);
        }
        if (string.Equals(id, candidateId, StringComparison.OrdinalIgnoreCase))
            return new ComparisonBaseline(null, null, null, $"against {against}: that is {candidateId}, the experiment being evaluated — name another one.");

        var data = await store.LoadAsync(modelName, id, cancellationToken);
        if (!string.Equals(data.Task, candidate.Task, StringComparison.OrdinalIgnoreCase)
            || !string.Equals(data.Config.LabelColumn, candidate.Config.LabelColumn, StringComparison.OrdinalIgnoreCase))
            return new ComparisonBaseline(null, null, null,
                $"against {id}: it was trained for {data.Task} on '{data.Config.LabelColumn}', {candidateId} for {candidate.Task} on " +
                $"'{candidate.Config.LabelColumn}' — the two cannot be compared on the same rows.");
        var modelPath = Path.Combine(store.GetExperimentPath(modelName, id), ExperimentLayout.ModelFileName);
        return File.Exists(modelPath)
            ? new ComparisonBaseline(id, modelPath, data, null)
            : new ComparisonBaseline(null, null, null, $"against {id}: model file not found: {modelPath}", NotFound: true);
    }

    /// <summary>Scores the baseline on the same rows and weights, and pairs the two per row where the task allows.</summary>
    public static async Task<Comparison> CompareAsync(
        EvaluationEngine engine,
        string candidateId, string candidateModel, ExperimentData candidate,
        ComparisonBaseline baseline, string testDataFile, string? weightColumn, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        if (baseline.Error is not null || baseline.Data is null || baseline.ModelPath is null || baseline.ExperimentId is null)
            throw new ArgumentException("The baseline was not resolved: " + baseline.Error, nameof(baseline));
        var task = candidate.Task;
        var label = candidate.Config.LabelColumn;
        var b = baseline.Data;
        var metrics = await engine.EvaluateAsync(baseline.ModelPath, testDataFile, label, task, cancellationToken,
            b.Config.InputSchema, b.Config.GroupColumn, b.Config.UserColumn, b.Config.ItemColumn, weightColumn);

        if (PairedComparison.MeasureOf(task) is not { } measure)
            return new Comparison(baseline.ExperimentId, metrics, null, $"{task} has no per-row measure to pair — compare the metrics.");
        var cl = await engine.RowLossesAsync(candidateModel, testDataFile, label, task, candidate.Config.InputSchema, weightColumn, cancellationToken);
        var bl = await engine.RowLossesAsync(baseline.ModelPath, testDataFile, label, task, b.Config.InputSchema, weightColumn, cancellationToken);
        if (cl is not { } c || bl is not { } o)
            return new Comparison(baseline.ExperimentId, metrics, null, "The models' outputs do not carry the columns a per-row comparison reads.");
        if (c.Losses.Length != o.Losses.Length)
            return new Comparison(baseline.ExperimentId, metrics, null,
                $"{candidateId} answered {c.Losses.Length} rows and {baseline.ExperimentId} {o.Losses.Length} — they were not read as the same rows.");
        return new Comparison(baseline.ExperimentId, metrics, PairedComparison.Compare(measure, c.Losses, o.Losses, c.Weights), null);
    }

    // Whether the CSV header names the column (case-insensitive, as the loaders match names).
    private static bool HeaderHas(string csvPath, string column)
    {
        try
        {
            using var reader = new StreamReader(csvPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            var header = reader.ReadLine();
            return header is not null && MLoop.Core.Prediction.CsvFieldParser.ParseFields(header)
                .Any(f => string.Equals(f.Trim(), column, StringComparison.OrdinalIgnoreCase));
        }
        catch (IOException)
        {
            return false;
        }
    }
}
