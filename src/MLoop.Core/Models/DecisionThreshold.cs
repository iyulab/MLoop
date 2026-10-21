using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The probability from which a binary classifier answers its positive class, chosen on an evaluation
/// set for a target precision — the lowest threshold whose precision reaches the target, which keeps
/// the most recall. It lives beside the model (<see cref="ExperimentLayout.DecisionThresholdFileName"/>)
/// and every path that loads the model applies it through <see cref="Load"/>, so predict, serve,
/// evaluate and the gate all read the same answers.
/// </summary>
/// <param name="Threshold">The probability from which the answer is the positive class.</param>
/// <param name="TargetPrecision">The precision it was chosen for.</param>
/// <param name="Precision">The positive precision it reached on <paramref name="ChosenOn"/>.</param>
/// <param name="Recall">The positive recall it kept there.</param>
/// <param name="Rows">Labeled rows it was chosen on.</param>
/// <param name="ChosenOn">The evaluation file's name.</param>
/// <param name="WeightColumn">The column the rows were weighted by, if any.</param>
/// <param name="ChosenAt">When.</param>
public sealed record DecisionThreshold(
    double Threshold,
    double TargetPrecision,
    double Precision,
    double Recall,
    int Rows,
    string ChosenOn,
    string? WeightColumn,
    DateTimeOffset ChosenAt)
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

    /// <summary>
    /// Loads the model at <paramref name="modelPath"/> with its decision threshold applied, when it has
    /// one — the one way a scoring path loads a model, so no path answers at a threshold another does not.
    /// </summary>
    public static ITransformer Load(MLContext mlContext, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        return ApplyBeside(mlContext, mlContext.Model.Load(modelPath, out _), modelPath);
    }

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
    /// The lowest threshold at which the positive precision of <paramref name="rows"/> reaches
    /// <paramref name="targetPrecision"/>, or why none does. Rows are (probability, label, weight).
    /// </summary>
    public static (double Threshold, double Precision, double Recall)? Choose(
        IReadOnlyList<(double Probability, bool Positive, double Weight)> rows, double targetPrecision)
    {
        ArgumentNullException.ThrowIfNull(rows);
        var usable = rows.Where(r => double.IsFinite(r.Probability) && r.Weight > 0).OrderByDescending(r => r.Probability).ToList();
        var positives = usable.Where(r => r.Positive).Sum(r => r.Weight);
        if (positives <= 0)
            return null;

        (double, double, double)? best = null;
        double tp = 0, fp = 0;
        for (var i = 0; i < usable.Count; i++)
        {
            if (usable[i].Positive) tp += usable[i].Weight; else fp += usable[i].Weight;
            // A threshold sits only where the probability changes: rows of one probability are answered alike.
            if (i + 1 < usable.Count && usable[i + 1].Probability == usable[i].Probability)
                continue;
            var precision = tp / (tp + fp);
            if (precision >= targetPrecision)
                best = (usable[i].Probability, precision, tp / positives);
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
