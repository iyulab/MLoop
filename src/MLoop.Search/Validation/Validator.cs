using Microsoft.ML;

namespace MLoop.Search.Validation;

/// <summary>A pipeline's score across the validated folds: each fold's metrics, their mean and spread.</summary>
/// <param name="FoldMetrics">The metrics of each validated fold, in fold order.</param>
/// <param name="Mean">Each metric's mean over the folds where it is defined.</param>
/// <param name="StandardDeviation">Each metric's spread over those folds (0 for one fold).</param>
/// <param name="IsHoldout">Whether the score is one holdout rather than averaged folds.</param>
public sealed record ValidationScore(
    IReadOnlyList<IReadOnlyDictionary<string, double>> FoldMetrics,
    IReadOnlyDictionary<string, double> Mean,
    IReadOnlyDictionary<string, double> StandardDeviation,
    bool IsHoldout)
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
    /// <param name="mlContext">The context the folds are fitted in.</param>
    /// <param name="data">The training partition.</param>
    /// <param name="plan">Its folds.</param>
    /// <param name="pipeline">The pipeline to fit — featurization and trainer together, so no fold sees its own rows in the fit.</param>
    /// <param name="score">The metrics of a scored validation part — the host's metric authority, weights included.</param>
    /// <param name="cancellationToken">Stops between folds.</param>
    public static ValidationScore Validate(
        MLContext mlContext,
        IDataView data,
        FoldPlan plan,
        IEstimator<ITransformer> pipeline,
        Func<IDataView, IReadOnlyDictionary<string, double>> score,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(score);

        var folds = new List<IReadOnlyDictionary<string, double>>();
        foreach (var part in plan.Validated)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var view = new HoldoutView(data, plan.FoldOf, part);
            var train = mlContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, lowerBound: 0, upperBound: 0.5);
            var valid = mlContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, lowerBound: 0.5, upperBound: 1.5);
            var model = pipeline.Fit(train);
            folds.Add(score(model.Transform(valid)));
        }

        var names = folds.SelectMany(f => f.Keys).Distinct(StringComparer.Ordinal).ToList();
        var mean = new Dictionary<string, double>(StringComparer.Ordinal);
        var std = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var name in names)
        {
            var values = folds.Select(f => f.TryGetValue(name, out var v) ? v : double.NaN).Where(double.IsFinite).ToArray();
            if (values.Length == 0)
                continue;
            var m = values.Average();
            mean[name] = m;
            std[name] = values.Length > 1 ? Math.Sqrt(values.Sum(v => (v - m) * (v - m)) / (values.Length - 1)) : 0;
        }
        return new ValidationScore(folds, mean, std, plan.IsHoldout);
    }
}
