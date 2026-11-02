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

/// <summary>One member of a blend: how to build its pipeline and how often the blend counts it.</summary>
/// <param name="Pipeline">Builds the member's whole pipeline (featurizer and trainer) in a context.</param>
/// <param name="Count">How many times the forward selection added the member — its weight is this over the sum of all counts.</param>
public sealed record EnsembleMember(Func<MLContext, IEstimator<ITransformer>> Pipeline, int Count);

/// <summary>
/// Fits the members of a blend (<c>ForwardSelection</c>'s counts) and joins them in <em>one</em> transformer chain that
/// saves as one <c>model.zip</c> and answers with the weighted mean: members and their answers in turn
/// (<c>CopyColumns</c> keeps each member's answer under its own name), then one <c>Expression</c> over them that writes the
/// answer under the names every scoring path reads (<c>Probability</c> · <c>PredictedLabel</c> · <c>Score</c> for a binary
/// classifier, <c>Score</c> for a regression). Public API only — no format of ML.NET's own is relied on.
/// </summary>
/// <remarks>
/// Each member is fitted on the training rows <em>as the chain before it transforms them</em>, not on the raw rows: a
/// member whose pipeline begins with a step that rewrites an input column in place (a normalization, a fill of gaps) would
/// otherwise be fitted on the raw column and then, in the saved chain, read the column an earlier member already rewrote.
/// Fitting each on what it will be given keeps training and scoring the same function. A blend is not exportable to ONNX
/// (<c>Expression</c> has no graph) — that refusal is the caller's. The weights go in as integer counts over their sum, so
/// no decimal is ever printed into the expression and no culture can change it.
/// </remarks>
public static class EnsembleChain
{
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
    /// <exception cref="ArgumentException">No member counts, or more than <see cref="MaxMembers"/> do.</exception>
    public static TransformerChain<ITransformer> Fit(
        MLContext mlContext, IReadOnlyList<EnsembleMember> members, IDataView train, EnsembleKind kind)
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
        for (var i = 0; i < counted.Length; i++)
        {
            var seen = steps.Count == 0 ? train : new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(counted[i].Pipeline(mlContext).Fit(seen));
            var name = $"Member{i}";
            names.Add(name);
            var scored = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(mlContext.Transforms.CopyColumns(name, answer).Fit(scored));
        }

        var counts = counted.Select(m => m.Count).ToArray();
        var withAnswers = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
        steps.Add(mlContext.Transforms.Expression(answer, WeightedMean(counts), [.. names]).Fit(withAnswers));
        if (kind == EnsembleKind.Binary)
        {
            var blended = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(mlContext.Transforms.Expression("PredictedLabel", $"(p) => p >= {DecisionPoint.ToString("R", CultureInfo.InvariantCulture)}", "Probability").Fit(blended));
            blended = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
            steps.Add(mlContext.Transforms.Expression("Score", $"(p) => p - {DecisionPoint.ToString("R", CultureInfo.InvariantCulture)}", "Probability").Fit(blended));
        }
        var finished = new TransformerChain<ITransformer>(steps.ToArray()).Transform(train);
        steps.Add(mlContext.Transforms.DropColumns([.. names]).Fit(finished));
        return new TransformerChain<ITransformer>(steps.ToArray());
    }

    /// <summary>The lambda of a weighted mean over integer counts: <c>(a, b) =&gt; (2 * a + 3 * b) / 5</c>.</summary>
    public static string WeightedMean(IReadOnlyList<int> counts)
    {
        var args = string.Join(", ", counts.Select((_, i) => $"x{i}"));
        var sum = string.Join(" + ", counts.Select((c, i) => $"{c.ToString(CultureInfo.InvariantCulture)} * x{i}"));
        return $"({args}) => ({sum}) / {counts.Sum().ToString(CultureInfo.InvariantCulture)}";
    }
}
