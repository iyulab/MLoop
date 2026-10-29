using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The probability from which a binary classifier answers its positive class, chosen on an evaluation
/// set for a target precision — the lowest threshold whose precision reaches the target, which keeps
/// the most recall. It lives beside the model (<see cref="ExperimentLayout.DecisionThresholdFileName"/>)
/// and every path that loads the model applies it through <see cref="ScoringModel.Load(MLContext, string, string?, ScoringLayers)"/>,
/// so predict, serve, evaluate and the gate all read the same answers.
/// </summary>
/// <param name="Threshold">The probability from which the answer is the positive class.</param>
/// <param name="TargetPrecision">The precision it was chosen for.</param>
/// <param name="Precision">The precision of <paramref name="PrecisionOf"/> it reached on <paramref name="ChosenOn"/>.</param>
/// <param name="Recall">The recall of <paramref name="PrecisionOf"/> it kept there.</param>
/// <param name="Rows">Labeled rows it was chosen on.</param>
/// <param name="ChosenOn">The evaluation file's name.</param>
/// <param name="WeightColumn">The column the rows were weighted by, if any.</param>
/// <param name="ChosenAt">When.</param>
/// <param name="PrecisionOf">
/// The class whose precision was targeted, by the name the label used. <c>null</c> is the positive
/// class — every threshold chosen before a class could be named. The answer rule is the same either
/// way (positive from <paramref name="Threshold"/> on); which class the precision belongs to decides
/// only which side of it was asked to be trustworthy.
/// </param>
/// <param name="Answered">
/// How many of the evaluation's rows the threshold answers as the targeted class — the rows
/// <paramref name="Precision"/> is a share of. <c>null</c> for a threshold chosen before this was recorded.
/// </param>
/// <param name="ClassRows">How many rows of the targeted class the evaluation had at all. <c>null</c> when not recorded.</param>
/// <param name="PrecisionLowerBound">
/// What the measured <paramref name="Precision"/> certifies at 95 % confidence (one-sided Wilson bound over the
/// answered rows; weighted rows count as their effective number). The threshold is the lowest one that reached the
/// target on this very file, so the precision it shows is the luckiest reading of a few rows: below the target,
/// the target is not what the file proves. <c>null</c> when not recorded.
/// </param>
/// <param name="Guarantee">
/// Set when the threshold was chosen with <c>--guarantee</c> (<see cref="PrecisionGuarantee"/>): the risk it was certified at and the
/// evidence. <c>null</c> for a threshold chosen the usual way — the lowest one that reached the target on the file.
/// </param>
public sealed record DecisionThreshold(
    double Threshold,
    double TargetPrecision,
    double Precision,
    double Recall,
    int Rows,
    string ChosenOn,
    string? WeightColumn,
    DateTimeOffset ChosenAt,
    string? PrecisionOf = null,
    int? Answered = null,
    int? ClassRows = null,
    double? PrecisionLowerBound = null,
    DecisionGuarantee? Guarantee = null)
{
    /// <summary>The z of a one-sided 95 % bound.</summary>
    private const double OneSided95 = 1.6448536269514722;

    /// <summary>
    /// How far below the target a threshold's certified precision may sit before the file is called too small to
    /// choose it on. The threshold is the lowest one that reached the target <i>on this file</i>, so its measured
    /// precision lands just above the target and its bound lands below it by a margin that shrinks with the rows —
    /// any margin at all is always there. Five points is the shortfall measured on held-out data when a threshold
    /// was chosen on 15–30 positive rows (0.06–0.07 on average); measured on a real binary set it is crossed
    /// below about a hundred answered rows and not at three hundred.
    /// </summary>
    public const double UncertifiedMargin = 0.05;

    /// <summary>Whether the file a threshold was chosen on is too small to stand behind <paramref name="target"/>: the bound sits more than <see cref="UncertifiedMargin"/> below it.</summary>
    public static bool IsUncertified(double target, double lowerBound) => target - lowerBound > UncertifiedMargin;

    /// <summary>
    /// The lowest share of <paramref name="effectiveRows"/> answers that is still consistent, at 95 % confidence, with
    /// <paramref name="observed"/> being the share that were right (Wilson's one-sided bound).
    /// </summary>
    public static double PrecisionBound(double observed, double effectiveRows)
    {
        if (effectiveRows <= 0) return 0;
        var z2 = OneSided95 * OneSided95;
        var centre = observed + z2 / (2 * effectiveRows);
        var spread = OneSided95 * Math.Sqrt(observed * (1 - observed) / effectiveRows + z2 / (4 * effectiveRows * effectiveRows));
        return Math.Clamp((centre - spread) / (1 + z2 / effectiveRows), 0, 1);
    }

    /// <summary>
    /// What a chosen threshold rests on: how many rows it answers as the class, how many rows the class has, and the
    /// bound its precision certifies (<see cref="PrecisionBound"/>). Weighted rows count as their effective number
    /// (Kish): a few heavy rows are a few rows however many there are.
    /// </summary>
    public static (int Answered, int ClassRows, double LowerBound) EvidenceFor(
        IReadOnlyList<(double Probability, bool Positive, double Weight)> rows, double threshold, double precision, bool ofPositive = true)
    {
        var finite = rows.Where(r => double.IsFinite(r.Probability) && r.Weight > 0).ToList();
        var answered = finite.Where(r => ofPositive ? r.Probability >= threshold : r.Probability < threshold).ToList();
        var sum = answered.Sum(r => r.Weight);
        var sumSquares = answered.Sum(r => r.Weight * r.Weight);
        var effective = sumSquares > 0 ? sum * sum / sumSquares : 0;
        return (answered.Count, finite.Count(r => r.Positive == ofPositive), PrecisionBound(precision, effective));
    }

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    /// <summary>The threshold file beside <paramref name="modelPath"/>, or <c>null</c> when there is none.</summary>
    public static DecisionThreshold? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<DecisionThreshold>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes this threshold beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => MLoop.Core.Storage.AtomicFile.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the threshold beside <paramref name="modelPath"/> lives.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.DecisionThresholdFileName);

    /// <summary><paramref name="model"/>, loaded from <paramref name="modelPath"/>, with the threshold beside it applied when there is one.</summary>
    public static ITransformer ApplyBeside(MLContext mlContext, ITransformer model, string modelPath)
        => ReadBeside(modelPath) is { } threshold ? threshold.Apply(mlContext, model) : model;

    /// <summary>
    /// <paramref name="model"/> followed by a step that answers the positive class from
    /// <see cref="Threshold"/> on: <c>PredictedLabel = Probability ≥ Threshold</c>. In memory only — the
    /// saved model is unchanged, and removing the file returns the trainer's own threshold.
    /// </summary>
    public ITransformer Apply(MLContext mlContext, ITransformer model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        var threshold = (float)Threshold;
        var step = mlContext.Transforms.CustomMapping<ProbabilityInput, DecisionOutput>(
                (input, output) => output.PredictedLabel = input.Probability >= threshold, contractName: null)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<ProbabilityInput>()));
        return new TransformerChain<ITransformer>(model, step);
    }

    /// <summary>
    /// The threshold at which the precision of one class of <paramref name="rows"/> reaches
    /// <paramref name="targetPrecision"/> while keeping the most of that class's recall, or <c>null</c>
    /// when none does. Rows are (probability, label, weight).
    /// </summary>
    /// <remarks>
    /// For the positive class that is the <b>lowest</b> such threshold — every row from it upward is
    /// answered positive. For the negative class (<paramref name="ofPositive"/> false) it is the
    /// <b>highest</b>: every row below it is answered negative. A class with no rows has no precision to reach.
    /// <para>
    /// The threshold sits <b>halfway</b> between the last probability answered as the class and the next
    /// one (0 or 1 past the end), never on a row's probability. Models with few distinct probabilities —
    /// trees, a calibration's steps — put many rows on one value, and a threshold on that value leaves them
    /// to the last digit of float arithmetic: an exported ONNX graph, which differs from ML.NET in the
    /// seventh digit, answered 39 of 300 such rows the other way. Halfway, either side keeps its answer.
    /// </para>
    /// </remarks>
    public static (double Threshold, double Precision, double Recall)? Choose(
        IReadOnlyList<(double Probability, bool Positive, double Weight)> rows, double targetPrecision, bool ofPositive = true)
    {
        ArgumentNullException.ThrowIfNull(rows);
        var finite = rows.Where(r => double.IsFinite(r.Probability) && r.Weight > 0);
        // Walk from the rows answered as the class first: down from the top for positive, up from the bottom for negative.
        var usable = (ofPositive ? finite.OrderByDescending(r => r.Probability) : finite.OrderBy(r => r.Probability)).ToList();
        var ofClass = usable.Where(r => r.Positive == ofPositive).Sum(r => r.Weight);
        if (ofClass <= 0)
            return null;

        (double, double, double)? best = null;
        double hit = 0, miss = 0;
        for (var i = 0; i < usable.Count; i++)
        {
            if (usable[i].Positive == ofPositive) hit += usable[i].Weight; else miss += usable[i].Weight;
            // A threshold sits only where the probability changes: rows of one probability are answered alike.
            if (i + 1 < usable.Count && usable[i + 1].Probability == usable[i].Probability)
                continue;
            var precision = hit / (hit + miss);
            if (precision < targetPrecision)
                continue;
            // Halfway to the next probability the walk would reach — the first answered the other way.
            var next = i + 1 < usable.Count ? usable[i + 1].Probability : ofPositive ? 0.0 : 1.0;
            var threshold = (usable[i].Probability + next) / 2;
            best = (threshold, precision, hit / ofClass);
        }
        return best;
    }

    /// <summary>
    /// Each labeled row's (probability, label, weight) from a binary model's output — what
    /// <see cref="Choose"/> reads. <c>null</c> when the output has no probability: a threshold is a
    /// probability, and a model that does not calibrate one cannot be given one.
    /// </summary>
    public static IReadOnlyList<(double Probability, bool Positive, double Weight)>? RowsOf(
        IDataView scored, string labelColumn, IReadOnlyList<double>? weights = null)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var probability = scored.Schema.GetColumnOrNull("Probability");
        if (label is null || probability is null || probability.Value.Type != NumberDataViewType.Single
            || (label.Value.Type != BooleanDataViewType.Instance && label.Value.Type != NumberDataViewType.Single))
            return null;
        var rows = new List<(double, bool, double)>();
        using var cursor = scored.GetRowCursor([label.Value, probability.Value]);
        var getProbability = cursor.GetGetter<float>(probability.Value);
        Func<bool?> truth;
        if (label.Value.Type == BooleanDataViewType.Instance)
        {
            var get = cursor.GetGetter<bool>(label.Value);
            truth = () => { bool v = default; get(ref v); return v; };
        }
        else
        {
            var get = cursor.GetGetter<float>(label.Value);
            truth = () => { float v = default; get(ref v); return float.IsNaN(v) ? null : v > 0; };
        }
        float p = default;
        var i = 0;
        while (cursor.MoveNext())
        {
            getProbability(ref p);
            var w = weights is null ? 1.0 : weights[i];
            i++;
            if (truth() is { } y)
                rows.Add((p, y, w));
        }
        return rows;
    }

    /// <summary>The input a threshold reads.</summary>
    public sealed class ProbabilityInput
    {
        public float Probability { get; set; }
    }

    /// <summary>The answer it writes.</summary>
    public sealed class DecisionOutput
    {
        public bool PredictedLabel { get; set; }
    }
}
