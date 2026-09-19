namespace MLoop.Core.Evaluation;

/// <summary>
/// The single place a metric name a user wrote becomes the canonical name the rest of MLoop
/// speaks — the keys of <c>AutoMLResult.Metrics</c>, the leaderboard's <c>metric</c>, the
/// experiment index's <c>metricName</c>.
/// </summary>
/// <remarks>
/// <para>
/// Before this, three sites each held their own vocabulary and none of them agreed. The optimizer
/// accepted <c>f1_score</c>, <c>f1</c>, <c>r2</c>; <c>validate</c> accepted <c>rSquared</c>,
/// <c>recall</c>, <c>log-loss</c>; the shipped examples wrote <c>F1Score</c>. A name the optimizer
/// did not recognize fell through to the task's default <b>without a word</b>, so
/// <c>metric: F1Score</c> optimized accuracy while every record — the config, the index, the
/// report — kept saying F1. The index's score was correct and its label was not.
/// </para>
/// <para>
/// Matching folds case and separators (<c>F1Score</c>, <c>f1-score</c> and <c>f1_score</c> are one
/// name) and accepts the ML.NET enum spellings and the short forms people actually type. An
/// unknown name comes back <c>null</c> — callers decide what to do about it and say so, rather than
/// substituting a default silently. <see cref="Rejection"/> is the one answer to what a caller does
/// about it: every producer of a training config refuses the name before any work starts.
/// </para>
/// <para>
/// The canonical set is not a second copy of anything: it is the union of what the optimizer can
/// be asked for and what the trial reporter writes, and a round-trip test in the AutoML runner
/// pins that the two agree.
/// </para>
/// </remarks>
public static class MetricNames
{
    /// <summary>The value that asks AutoML to choose — resolved to the task's primary metric by <see cref="Canonical(string?, string?)"/>.</summary>
    public const string Auto = "auto";

    // Canonical names, grouped by the task family whose optimizer accepts them. The families are
    // for readers; matching is over the union.
    private static readonly string[] Binary =
        ["accuracy", "auc", "auprc", "f1_score", "precision", "recall", "negative_precision", "negative_recall"];
    private static readonly string[] Multiclass =
        ["macro_accuracy", "micro_accuracy", "log_loss", "log_loss_reduction", "top_k_accuracy"];
    private static readonly string[] Regression =
        ["r_squared", "rmse", "mae", "mse"];
    private static readonly string[] Other =
        ["average_distance", "davies_bouldin_index", "ndcg", "detection_rate", "mape"];

    /// <summary>Every canonical metric name, in a stable order.</summary>
    public static IReadOnlyList<string> All { get; } = [.. Binary, .. Multiclass, .. Regression, .. Other];

    // Folded spelling → canonical. Every canonical name folds to itself; the rest are the
    // spellings users and ML.NET use for the same thing.
    private static readonly IReadOnlyDictionary<string, string> ByFoldedSpelling = Build();

    private static Dictionary<string, string> Build()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in All)
            map[Fold(name)] = name;

        void Alias(string spelling, string canonical) => map[Fold(spelling)] = canonical;

        Alias("f1", "f1_score");
        Alias("r2", "r_squared");
        Alias("AreaUnderRocCurve", "auc");
        Alias("AreaUnderPrecisionRecallCurve", "auprc");
        Alias("PositivePrecision", "precision");
        Alias("PositiveRecall", "recall");
        Alias("RootMeanSquaredError", "rmse");
        Alias("MeanAbsoluteError", "mae");
        Alias("MeanSquaredError", "mse");
        return map;
    }

    /// <summary>
    /// The canonical name for <paramref name="name"/>, <see cref="Auto"/> for the auto value, or
    /// <c>null</c> when the name is not one MLoop knows.
    /// </summary>
    public static string? Canonical(string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;

        var folded = Fold(name);
        if (folded == Auto)
            return Auto;

        return ByFoldedSpelling.TryGetValue(folded, out var canonical) ? canonical : null;
    }

    /// <summary>
    /// The canonical name for <paramref name="name"/> as the optimizer for <paramref name="task"/>
    /// reads it: <see cref="Auto"/> becomes the task's primary metric, and <c>accuracy</c> under a
    /// multiclass task means <c>macro_accuracy</c>, which is what that optimizer has always done
    /// with it. <c>null</c> when the name is unknown.
    /// </summary>
    public static string? Canonical(string? task, string? name)
    {
        var canonical = Canonical(name);
        if (canonical is null)
            return null;

        if (canonical == Auto)
            return TaskMetadata.PrimaryMetric(task) ?? Auto;

        if (canonical == "accuracy" && IsMulticlassFamily(task))
            return "macro_accuracy";

        return canonical;
    }

    /// <summary>Whether <paramref name="name"/> is a spelling of a metric MLoop knows (or the auto value).</summary>
    public static bool IsKnown(string? name) => Canonical(name) is not null;

    /// <summary>
    /// The canonical names the AutoML optimizer for <paramref name="task"/> can be asked to
    /// optimize, or <c>null</c> for a task whose trainer takes no metric choice (clustering,
    /// ranking, forecasting, the deep-learning tasks …) — there the name is recorded, not optimized.
    /// </summary>
    public static IReadOnlyList<string>? OptimizableFor(string? task) =>
        task?.Trim().ToLowerInvariant() switch
        {
            "binary-classification" => Binary,
            "multiclass-classification" => Multiclass,
            "regression" => Regression,
            _ => null
        };

    /// <summary>
    /// Why <paramref name="name"/> cannot be the metric for <paramref name="task"/>, or <c>null</c>
    /// when it can. A blank name is the default and always can.
    /// </summary>
    /// <remarks>
    /// Two ways to fail, and both refuse rather than substitute. A name MLoop does not know at all is
    /// usually a typo. A name it knows that the task's optimizer cannot take (<c>auc</c> for
    /// regression) is a mismatch. Either way the run would have optimized the task default while
    /// the config, the index and the report recorded the name that was given — the same
    /// record-versus-reality split that let <c>metric: F1Score</c> optimize accuracy.
    /// </remarks>
    public static string? Rejection(string? task, string? name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;

        var accepted = OptimizableFor(task);
        var canonical = Canonical(task, name);

        if (canonical is null)
        {
            var suggestion = Suggest(name, accepted ?? All);
            return $"Unknown metric '{name}'." +
                   (suggestion is null ? "" : $" Did you mean '{suggestion}'?") +
                   $" Known names{(accepted is null ? "" : $" for {task}")}: {Auto}, {string.Join(", ", accepted ?? All)}.";
        }

        if (accepted is not null && canonical != Auto && !accepted.Contains(canonical))
            return $"Metric '{name}' does not apply to {task}. Choose one of: {Auto}, {string.Join(", ", accepted)}.";

        return null;
    }

    // The closest candidate by edit distance over folded spellings, when it is close enough to be a
    // plausible typo (at most two edits, and fewer than half the name).
    private static string? Suggest(string name, IReadOnlyList<string> candidates)
    {
        var folded = Fold(name);
        string? best = null;
        var bestDistance = int.MaxValue;
        foreach (var (spelling, canonical) in ByFoldedSpelling)
        {
            if (!candidates.Contains(canonical))
                continue;
            var distance = EditDistance(folded, spelling);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = canonical;
            }
        }
        return bestDistance <= 2 && bestDistance * 2 < folded.Length ? best : null;
    }

    private static int EditDistance(string a, string b)
    {
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];
        for (var j = 0; j <= b.Length; j++)
            previous[j] = j;
        for (var i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (var j = 1; j <= b.Length; j++)
            {
                var substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                current[j] = Math.Min(substitution, Math.Min(previous[j] + 1, current[j - 1] + 1));
            }
            (previous, current) = (current, previous);
        }
        return previous[b.Length];
    }

    private static bool IsMulticlassFamily(string? task) =>
        task is not null && task.Trim().ToLowerInvariant() is
            "multiclass-classification" or "image-classification" or "text-classification";

    /// <summary>Lower case, letters and digits only — the shape two spellings are compared in.</summary>
    private static string Fold(string name)
    {
        Span<char> buffer = stackalloc char[name.Length];
        var n = 0;
        foreach (var c in name)
        {
            if (char.IsAsciiLetterOrDigit(c))
                buffer[n++] = char.ToLowerInvariant(c);
        }
        return new string(buffer[..n]);
    }
}
