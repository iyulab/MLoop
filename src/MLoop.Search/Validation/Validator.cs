using Microsoft.ML;

namespace MLoop.Search.Validation;

/// <summary>A pipeline's score across the validated folds: each fold's metrics, their mean and spread.</summary>
/// <param name="FoldMetrics">The metrics of each validated fold, in fold order.</param>
/// <param name="Mean">Each metric's mean over the folds where it is defined.</param>
/// <param name="StandardDeviation">Each metric's spread over those folds (0 for one fold).</param>
/// <param name="IsHoldout">Whether the score is one holdout rather than averaged folds.</param>
/// <param name="StandardError">
/// How far each metric's <see cref="Mean"/> could move on other rows: the folds' spread over √folds — or, for a
/// holdout, the spread of <see cref="Validator.HoldoutSlices"/> equal slices of it over √slices. Absent for a metric
/// no two folds or slices defined.
/// </param>
public sealed record ValidationScore(
    IReadOnlyList<IReadOnlyDictionary<string, double>> FoldMetrics,
    IReadOnlyDictionary<string, double> Mean,
    IReadOnlyDictionary<string, double> StandardDeviation,
    bool IsHoldout,
    IReadOnlyDictionary<string, double>? StandardError = null)
{
    /// <summary>How many folds were averaged.</summary>
    public int Folds => FoldMetrics.Count;
}

/// <summary>
/// Scores a pipeline by fitting it on all but one part of a <see cref="FoldPlan"/> and scoring the part
/// left out, for every validated part, and averaging. The score of a trial is the mean —
/// not one fold drawn at random, as AutoML scored — so two trials compare on the same rows.
/// </summary>
public static class Validator
{
    /// <summary>The slices a holdout is cut into to estimate its standard error.</summary>
    public const int HoldoutSlices = 5;

    /// <param name="mlContext">The context the folds are fitted in.</param>
    /// <param name="data">The training partition.</param>
    /// <param name="plan">Its folds.</param>
    /// <param name="pipeline">The pipeline to fit — featurization and trainer together, so no fold sees its own rows in the fit.</param>
    /// <param name="score">The metrics of a scored validation part — the host's metric authority, weights included.</param>
    /// <param name="cancellationToken">Stops the fit in progress (<see cref="FitCancellation"/>) — and with it every later use of <paramref name="mlContext"/>.</param>
    /// <param name="scoredPart">Given each validated part scored by the model fitted without it — what out-of-fold answers are read from.</param>
    public static ValidationScore Validate(
        MLContext mlContext,
        IDataView data,
        FoldPlan plan,
        IEstimator<ITransformer> pipeline,
        Func<IDataView, IReadOnlyDictionary<string, double>> score,
        CancellationToken cancellationToken = default,
        Action<IDataView>? scoredPart = null)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(score);

        using var stop = FitCancellation.Register(mlContext, cancellationToken);
        var folds = new List<IReadOnlyDictionary<string, double>>();
        IReadOnlyList<IReadOnlyDictionary<string, double>>? slices = null;
        foreach (var part in plan.Validated)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var view = new HoldoutView(data, plan.FoldOf, part);
            var train = mlContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, lowerBound: 0, upperBound: 0.5);
            var valid = mlContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, lowerBound: 0.5, upperBound: 1.5);
            var model = pipeline.Fit(train);
            var scored = model.Transform(valid);
            // Read twice when answers are kept — the metrics, then the answers — so the part is cached: scoring it
            // again costs a pass of the whole pipeline per fold, measured as two to three fewer trials in a minute
            // on 87,000 rows, and a different model kept.
            if (scoredPart is not null)
                scored = mlContext.Data.Cache(scored);
            folds.Add(score(scored));
            scoredPart?.Invoke(scored);
            if (plan.IsHoldout)
                slices = Slices(mlContext, scored, plan.HoldoutSliceOf!, score);
        }

        var (mean, std) = Summarize(folds);
        var standardError = slices is not null
            ? Summarize(slices).Spread.Where(kv => slices.Count(s => s.ContainsKey(kv.Key)) > 1)
                .ToDictionary(kv => kv.Key, kv => kv.Value / Math.Sqrt(HoldoutSlices), StringComparer.Ordinal)
            : folds.Count > 1
                ? std.ToDictionary(kv => kv.Key, kv => kv.Value / Math.Sqrt(folds.Count), StringComparer.Ordinal)
                : null;
        return new ValidationScore(folds, mean, std, plan.IsHoldout, standardError);
    }

    // The holdout's metrics on each of its HoldoutSlices slices (FoldPlan.HoldoutSliceOf — stratified random shares of
    // it, by the row's position among the holdout's). A slice its metric cannot be computed on (one class) is left out.
    private static List<IReadOnlyDictionary<string, double>> Slices(
        MLContext mlContext, IDataView scored, IReadOnlyList<int> sliceOf, Func<IDataView, IReadOnlyDictionary<string, double>> score)
    {
        var slices = new List<IReadOnlyDictionary<string, double>>();
        if (sliceOf.Count < HoldoutSlices)
            return slices;
        for (var i = 0; i < HoldoutSlices; i++)
        {
            var slice = mlContext.Data.FilterRowsByColumn(new HoldoutView(scored, sliceOf, i), HoldoutView.ColumnName, lowerBound: 0.5, upperBound: 1.5);
            try { slices.Add(score(slice)); }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException) { }
        }
        return slices;
    }

    private static (Dictionary<string, double> Mean, Dictionary<string, double> Spread) Summarize(
        IReadOnlyList<IReadOnlyDictionary<string, double>> parts)
    {
        var mean = new Dictionary<string, double>(StringComparer.Ordinal);
        var std = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var name in parts.SelectMany(f => f.Keys).Distinct(StringComparer.Ordinal))
        {
            var values = parts.Select(f => f.TryGetValue(name, out var v) ? v : double.NaN).Where(double.IsFinite).ToArray();
            if (values.Length == 0)
                continue;
            var m = values.Average();
            mean[name] = m;
            std[name] = values.Length > 1 ? Math.Sqrt(values.Sum(v => (v - m) * (v - m)) / (values.Length - 1)) : 0;
        }
        return (mean, std);
    }
}
