using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Training;

/// <summary>What an <see cref="EnsembleChain"/> blends: a binary classifier's probability or a regression's value.</summary>
public enum EnsembleKind
{
    /// <summary>A binary classifier — the blend is the weighted mean of the members' <c>Probability</c>.</summary>
    Binary,

    /// <summary>A regression — the blend is the weighted mean of the members' <c>Score</c>.</summary>
    Regression,
}

/// <summary>The fit of a blend was stopped between members because the caller's time was up.</summary>
public sealed class EnsembleOverBudgetException(int fitted, int total)
    : Exception($"the time budget ran out after fitting {fitted} of the {total} members")
{
    /// <summary>How many members were fitted.</summary>
    public int Fitted { get; } = fitted;
}

/// <summary>One member of a blend: how to build its pipeline and how often the blend counts it.</summary>
/// <param name="Pipeline">Builds the member's whole pipeline (featurizer and trainer) in a context.</param>
/// <param name="Count">How many times the forward selection added the member — its weight is this over the sum of all counts.</param>
public sealed record EnsembleMember(Func<MLContext, IEstimator<ITransformer>> Pipeline, int Count);

/// <summary>
/// Fits the members of a blend (<c>ForwardSelection</c>'s counts) and joins them in <em>one</em> transformer chain that
/// saves as one <c>model.zip</c> and answers with the weighted mean: members and their answers in turn
/// (<c>CopyColumns</c> keeps each member's answer under its own name), then one <c>Expression</c> over them that writes the
/// answer under the names every scoring path reads (<c>Probability</c> · <c>PredictedLabel</c> · <c>Score</c> — the log-odds
/// of the probability — for a binary classifier, <c>Score</c> for a regression). Public API only — no format of ML.NET's own is relied on.
/// </summary>
/// <remarks>
/// Each member is fitted on the raw training rows and reads the raw columns when the chain scores: the chain keeps the table's
/// columns under other names ahead of the first member and puts them back under their own after every member. A step that
/// rewrites a column in place (a normalization, a one-hot or text featurization under the column's own name) would otherwise
/// leave the next member a vector where it was fitted on a string. A blend is not exportable to ONNX
/// (<c>Expression</c> has no graph) — that refusal is the caller's. The weights go in as integer counts over their sum, so
/// no decimal is ever printed into the expression and no culture can change it.
/// </remarks>
public static class EnsembleChain
{
    private const string RawPrefix = "RawColumn_";

    /// <summary>The most members one blend holds: <c>Expression</c> reads at most five columns, and 5 is as far as the blend was measured.</summary>
    public const int MaxMembers = 5;

    /// <summary>The probability at which a blend of probabilities answers the positive class — what a binary classifier answers at before a calibration.</summary>
    public const float DecisionPoint = 0.5f;

    /// <summary>
    /// Fits the members that count and returns the chain.
    /// </summary>
    /// <param name="mlContext">The context the members are fitted in.</param>
    /// <param name="members">The members; one whose <see cref="EnsembleMember.Count"/> is 0 is left out.</param>
    /// <param name="train">The training partition every member is fitted on.</param>
    /// <param name="kind">What the blend is of.</param>
    /// <param name="giveUp">Asked after each member is fitted; when it says so, the fit stops (<see cref="EnsembleOverBudgetException"/>) — the caller's time is up.</param>
    /// <exception cref="EnsembleOverBudgetException"><paramref name="giveUp"/> said so after a member was fitted.</exception>
    /// <exception cref="ArgumentException">No member counts, or more than <see cref="MaxMembers"/> do.</exception>
    public static TransformerChain<ITransformer> Fit(
        MLContext mlContext, IReadOnlyList<EnsembleMember> members, IDataView train, EnsembleKind kind, Func<bool>? giveUp = null)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(members);
        ArgumentNullException.ThrowIfNull(train);
        var counted = members.Where(m => m.Count > 0).ToArray();
        if (counted.Length == 0)
            throw new ArgumentException("A blend needs at least one member that counts.", nameof(members));
        if (counted.Length > MaxMembers)
            throw new ArgumentException($"A blend holds at most {MaxMembers} members; {counted.Length} count.", nameof(members));

        var answer = kind == EnsembleKind.Binary ? "Probability" : "Score";
        var steps = new List<ITransformer>();
        var names = new List<string>();
        var backups = new List<string>();
        IDataView Seen() => steps.Count == 0 ? train : new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
        void Add(IEstimator<ITransformer> step) => steps.Add(step.Fit(Seen()));
        for (var i = 0; i < counted.Length; i++)
        {
            // Every member reads the raw columns, as it was fitted on them. A step that rewrites a column under its own name (a
            // normalization, a one-hot or text featurization) leaves the next member a vector where it was fitted on a string, so
            // the columns this member rewrites are kept under other names ahead of it and put back under their own after it. A
            // member sees the raw columns either way, so it is fitted on the training rows themselves: fitted on the chain's output
            // every pass of its trainer would run the members before it again.
            var member = counted[i].Pipeline(mlContext).Fit(train);
            if (giveUp?.Invoke() == true)
                throw new EnsembleOverBudgetException(i + 1, counted.Length);
            var rewritten = RewrittenColumns(train.Schema, member.GetOutputSchema(train.Schema));
            var kept = rewritten.Select(name => $"{RawPrefix}{i}_{name}").ToArray();
            for (var c = 0; c < rewritten.Length; c++)
                Add(mlContext.Transforms.CopyColumns(kept[c], rewritten[c]));
            steps.Add(member);
            var name = $"Member{i}";
            names.Add(name);
            Add(mlContext.Transforms.CopyColumns(name, answer));
            for (var c = 0; c < rewritten.Length; c++)
                Add(mlContext.Transforms.CopyColumns(rewritten[c], kept[c]));
            backups.AddRange(kept);
        }

        var counts = counted.Select(m => m.Count).ToArray();
        var withAnswers = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
        steps.Add(mlContext.Transforms.Expression(answer, WeightedMean(counts), [.. names]).Fit(withAnswers));
        if (kind == EnsembleKind.Binary)
        {
            var blended = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(mlContext.Transforms.Expression("PredictedLabel", $"(p) => p >= {DecisionPoint.ToString("R", CultureInfo.InvariantCulture)}", "Probability").Fit(blended));
            blended = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(mlContext.Transforms.Expression("Score", LogitLambda, "Probability").Fit(blended));
        }
        var finished = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
        steps.Add(mlContext.Transforms.DropColumns([.. names, .. backups]).Fit(finished));
        return new TransformerChain<ITransformer>(steps.ToArray());
    }

    /// <summary>
    /// The log-odds of a probability — what a binary classifier's raw <c>Score</c> is for a logistic model, and zero where the
    /// probability is <see cref="DecisionPoint"/>. A probability of exactly 0 or 1 (a mean of members that all say so) is held
    /// off the edge so the score stays finite.
    /// </summary>
    internal const string LogitLambda = "(p) => log(max(p, 0.000001) / max(1 - p, 0.000001))";

    // The visible columns a step left under their own names but made new: the name now points at a later column of the output
    // schema than it did in the input's (the old one is hidden behind it).
    private static string[] RewrittenColumns(DataViewSchema input, DataViewSchema output)
        => [.. input.Where(c => !c.IsHidden)
            .Where(c => output.GetColumnOrNull(c.Name) is { } after && after.Index != c.Index)
            .Select(c => c.Name)];

    /// <summary>The log-odds of <paramref name="probability"/> as <see cref="LogitLambda"/> computes them, for rows built outside the chain.</summary>
    public static float LogOdds(float probability)
        => MathF.Log(Math.Max(probability, 0.000001f) / Math.Max(1 - probability, 0.000001f));

    /// <summary>The lambda of a weighted mean over integer counts: <c>(a, b) =&gt; (2 * a + 3 * b) / 5</c>.</summary>
    public static string WeightedMean(IReadOnlyList<int> counts)
    {
        var args = string.Join(", ", counts.Select((_, i) => $"x{i}"));
        var sum = string.Join(" + ", counts.Select((c, i) => $"{c.ToString(CultureInfo.InvariantCulture)} * x{i}"));
        return $"({args}) => ({sum}) / {counts.Sum().ToString(CultureInfo.InvariantCulture)}";
    }
}
