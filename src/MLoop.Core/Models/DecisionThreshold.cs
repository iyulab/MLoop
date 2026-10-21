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
public sealed record DecisionThreshold(
    double Threshold,
    double TargetPrecision,
    double Precision,
    double Recall,
    int Rows,
    string ChosenOn,
    string? WeightColumn,
    DateTimeOffset ChosenAt,
    string? PrecisionOf = null)
{
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
    public void WriteBeside(string modelPath) => File.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

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
    /// <b>highest</b>: every row below it is answered negative, so the threshold sits at the next
    /// probability above the last row answered negative. A class with no rows has no precision to reach.
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
            var threshold = ofPositive
                ? usable[i].Probability
                // The next probability up answers positive; past the last row, the next float does.
                : i + 1 < usable.Count ? usable[i + 1].Probability : MathF.BitIncrement((float)usable[i].Probability);
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
