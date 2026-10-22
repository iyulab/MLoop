using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// A classifier's probability calibration, fit on labeled rows without retraining the model: a map
/// from the probability the model gives to one that matches how often it is right. It lives
/// beside the model (<see cref="ExperimentLayout.CalibrationFileName"/>) and
/// <see cref="ScoringModel.Load(MLContext, string, string?, ScoringLayers)"/> applies it, so predict,
/// serve, evaluate (and its ECE) and a threshold chosen afterwards all read calibrated probabilities.
/// </summary>
/// <remarks>
/// Binary: <b>Platt</b> fits <c>σ(a·logit(p) + b)</c> — two parameters, steady on a few hundred rows;
/// <b>isotonic</b> fits a non-decreasing step map by pool-adjacent-violators — no shape assumed, but it
/// needs more rows to be steady. Both keep the order of the probabilities (isotonic may tie
/// neighbours), so the model's ranking — its AUC — is unchanged up to ties. Multiclass:
/// <b>temperature</b> scales every class's log-probability by <c>1/T</c> and renormalizes — one
/// parameter, and each row's order of classes (its answer) is unchanged. Rows carry weights, so a
/// recheck sample drawn with probability π calibrates with weight 1/π.
/// </remarks>
/// <param name="Method"><c>platt</c> or <c>isotonic</c> (binary), <c>temperature</c> (multiclass).</param>
/// <param name="A">Platt slope on the logit.</param>
/// <param name="B">Platt intercept.</param>
/// <param name="Knots">Isotonic: the probabilities at which the map's values are given (ascending).</param>
/// <param name="Values">Isotonic: the calibrated probability at each knot; between knots the map is linear.</param>
/// <param name="Rows">Labeled rows it was fit on.</param>
/// <param name="FitOn">The file's name.</param>
/// <param name="WeightColumn">The column the rows were weighted by, if any.</param>
/// <param name="EceBefore">ECE of the model's own probabilities on those rows.</param>
/// <param name="EceAfter">ECE of the calibrated probabilities on the same rows — in-sample, so optimistic.</param>
/// <param name="FitAt">When.</param>
/// <param name="Temperature">Temperature: <c>T</c>; above 1 softens an overconfident model, below 1 sharpens.</param>
public sealed record ProbabilityCalibration(
    string Method,
    double? A,
    double? B,
    double[]? Knots,
    double[]? Values,
    int Rows,
    string FitOn,
    string? WeightColumn,
    double? EceBefore,
    double? EceAfter,
    DateTimeOffset FitAt,
    double? Temperature = null)
{
    /// <summary>The binary methods, by the name the CLI and the file use; the first is the default.</summary>
    public static IReadOnlyList<string> BinaryMethods { get; } = ["platt", "isotonic"];

    /// <summary>The multiclass methods; the first is the default.</summary>
    public static IReadOnlyList<string> MulticlassMethods { get; } = ["temperature"];

    /// <summary>Whether this calibration maps a multiclass model's class probabilities.</summary>
    public bool IsMulticlass => Method == "temperature";

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    /// <summary>The calibration beside <paramref name="modelPath"/>, or <c>null</c> when there is none.</summary>
    public static ProbabilityCalibration? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<ProbabilityCalibration>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes this calibration beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => File.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the calibration beside <paramref name="modelPath"/> lives.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.CalibrationFileName);

    /// <summary>The calibrated probability for the model's own <paramref name="p"/>.</summary>
    public double Map(double p)
    {
        if (double.IsNaN(p))
            return p;
        p = Math.Clamp(p, 0, 1);
        if (Method == "platt")
            return Sigmoid(A!.Value * Logit(p) + B!.Value);

        var x = Knots!;
        var y = Values!;
        if (p <= x[0]) return y[0];
        if (p >= x[^1]) return y[^1];
        var i = Array.BinarySearch(x, p);
        if (i >= 0) return y[i];
        i = ~i;   // x[i-1] < p < x[i]
        var t = (p - x[i - 1]) / (x[i] - x[i - 1]);
        return y[i - 1] + t * (y[i] - y[i - 1]);
    }

    /// <summary>
    /// <paramref name="model"/> followed by a step that replaces <c>Probability</c> with the calibrated
    /// one and answers <c>PredictedLabel</c> from it at 0.5 — the label and the probability a row
    /// carries then agree. A decision threshold, applied after, answers from the calibrated probability.
    /// </summary>
    public ITransformer Apply(MLContext mlContext, ITransformer model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        if (IsMulticlass)
            return new TemperedModel(mlContext, model, this);
        var step = mlContext.Transforms.CustomMapping<ProbabilityInput, CalibratedOutput>(
                (input, output) =>
                {
                    var calibrated = (float)Map(input.Probability);
                    output.Probability = calibrated;
                    output.PredictedLabel = calibrated >= 0.5f;
                }, contractName: null)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<ProbabilityInput>()));
        return new TransformerChain<ITransformer>(model, step);
    }

    /// <summary><paramref name="model"/> with the calibration beside <paramref name="modelPath"/> applied, when there is one.</summary>
    public static ITransformer ApplyBeside(MLContext mlContext, ITransformer model, string modelPath)
        => ReadBeside(modelPath) is { } calibration ? calibration.Apply(mlContext, model) : model;

    /// <summary>
    /// Fits a calibration to <paramref name="rows"/> — (the model's probability, label, weight) — by
    /// <paramref name="method"/>, or <c>null</c> when the rows cannot carry one: both classes have to
    /// be present.
    /// </summary>
    public static ProbabilityCalibration? Fit(
        IReadOnlyList<(double Probability, bool Positive, double Weight)> rows, string method,
        string fitOn, string? weightColumn, DateTimeOffset fitAt)
    {
        ArgumentNullException.ThrowIfNull(rows);
        var usable = rows.Where(r => double.IsFinite(r.Probability) && r.Weight > 0).ToList();
        if (!usable.Any(r => r.Positive) || !usable.Any(r => !r.Positive))
            return null;

        ProbabilityCalibration calibration = method switch
        {
            "platt" => FitPlatt(usable),
            "isotonic" => FitIsotonic(usable),
            _ => throw new ArgumentException($"Unknown binary calibration method '{method}'. Choose one of: {string.Join(", ", BinaryMethods)}.", nameof(method)),
        };
        return calibration with
        {
            Rows = usable.Count,
            FitOn = fitOn,
            WeightColumn = weightColumn,
            EceBefore = Ece(usable, p => p),
            EceAfter = Ece(usable, calibration.Map),
            FitAt = fitAt,
        };
    }

    // Weighted logistic regression of the label on logit(p), by Newton's method with a small ridge so a
    // separable sample stays finite.
    private static ProbabilityCalibration FitPlatt(List<(double Probability, bool Positive, double Weight)> rows)
    {
        double a = 1, b = 0;
        for (var iteration = 0; iteration < 100; iteration++)
        {
            double ga = 0, gb = 0, haa = 1e-6, hab = 0, hbb = 1e-6;
            foreach (var (p, y, w) in rows)
            {
                var z = Logit(Math.Clamp(p, 0, 1));
                var q = Sigmoid(a * z + b);
                var r = (y ? 1 : 0) - q;
                ga += w * r * z;
                gb += w * r;
                var s = w * q * (1 - q);
                haa += s * z * z;
                hab += s * z;
                hbb += s;
            }
            var det = haa * hbb - hab * hab;
            if (det <= 0) break;
            var da = (hbb * ga - hab * gb) / det;
            var db = (haa * gb - hab * ga) / det;
            a += da;
            b += db;
            if (Math.Abs(da) < 1e-9 && Math.Abs(db) < 1e-9) break;
        }
        return new ProbabilityCalibration("platt", a, b, null, null, 0, "", null, null, null, default);
    }

    // Weighted pool-adjacent-violators over the rows sorted by probability; each pooled block becomes a
    // knot at its weighted mean probability carrying its weighted positive rate.
    private static ProbabilityCalibration FitIsotonic(List<(double Probability, bool Positive, double Weight)> rows)
    {
        var blocks = new List<(double X, double Y, double W)>();
        foreach (var group in rows.GroupBy(r => r.Probability).OrderBy(g => g.Key))
        {
            var w = group.Sum(r => r.Weight);
            var block = (X: group.Key, Y: group.Where(r => r.Positive).Sum(r => r.Weight) / w, W: w);
            while (blocks.Count > 0 && blocks[^1].Y >= block.Y)
            {
                var last = blocks[^1];
                blocks.RemoveAt(blocks.Count - 1);
                var total = last.W + block.W;
                block = ((last.X * last.W + block.X * block.W) / total, (last.Y * last.W + block.Y * block.W) / total, total);
            }
            blocks.Add(block);
        }
        return new ProbabilityCalibration("isotonic", null, null,
            [.. blocks.Select(k => k.X)], [.. blocks.Select(k => k.Y)], 0, "", null, null, null, default);
    }

    // The positive probability against whether the row was positive — the binary ECE evaluate reports.
    private static double? Ece(List<(double Probability, bool Positive, double Weight)> rows, Func<double, double> map)
        => WeightedMetrics.Ece([.. rows.Select(r => (map(r.Probability), r.Positive, r.Weight))]);

    /// <summary>
    /// The class probabilities of one row at <see cref="Temperature"/>: <c>p_i^(1/T)</c> renormalized.
    /// A non-finite or empty row comes back as it went in.
    /// </summary>
    public float[] Temper(ReadOnlySpan<float> probabilities)
    {
        var t = Temperature ?? 1;
        var result = new float[probabilities.Length];
        double total = 0;
        for (var i = 0; i < probabilities.Length; i++)
        {
            var v = Math.Pow(Math.Max(probabilities[i], 1e-12), 1 / t);
            result[i] = (float)v;
            total += v;
        }
        if (!(total > 0) || !double.IsFinite(total))
            return probabilities.ToArray();
        for (var i = 0; i < result.Length; i++)
            result[i] = (float)(result[i] / total);
        return result;
    }

    /// <summary>
    /// Fits a temperature to multiclass <paramref name="rows"/> — (class probabilities, the true
    /// class's slot, weight) — by minimizing the weighted log loss, or <c>null</c> without rows.
    /// </summary>
    public static ProbabilityCalibration? FitTemperature(
        IReadOnlyList<(float[] Probabilities, int Label, double Weight)> rows, string fitOn, string? weightColumn, DateTimeOffset fitAt)
    {
        ArgumentNullException.ThrowIfNull(rows);
        var usable = rows.Where(r => r.Weight > 0 && r.Label >= 0 && r.Label < r.Probabilities.Length
                                     && r.Probabilities.All(float.IsFinite)).ToList();
        if (usable.Count == 0)
            return null;

        double LogLoss(double logT)
        {
            var calibration = Temperatured(Math.Exp(logT));
            return usable.Sum(r => -r.Weight * Math.Log(Math.Max(calibration.Temper(r.Probabilities)[r.Label], 1e-12)));
        }

        // Golden-section search on log T over [1/20, 20]: the log loss is unimodal in T.
        double lo = Math.Log(0.05), hi = Math.Log(20), golden = (Math.Sqrt(5) - 1) / 2;
        double x1 = hi - golden * (hi - lo), x2 = lo + golden * (hi - lo), f1 = LogLoss(x1), f2 = LogLoss(x2);
        for (var i = 0; i < 80 && hi - lo > 1e-6; i++)
        {
            if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - golden * (hi - lo); f1 = LogLoss(x1); }
            else { lo = x1; x1 = x2; f1 = f2; x2 = lo + golden * (hi - lo); f2 = LogLoss(x2); }
        }

        var fitted = Temperatured(Math.Exp((lo + hi) / 2));
        return fitted with
        {
            Rows = usable.Count,
            FitOn = fitOn,
            WeightColumn = weightColumn,
            EceBefore = TopLabelEce(usable, r => r),
            EceAfter = TopLabelEce(usable, p => fitted.Temper(p)),
            FitAt = fitAt,
        };
    }

    private static ProbabilityCalibration Temperatured(double t)
        => new("temperature", null, null, null, null, 0, "", null, null, null, default, t);

    // Top-label: the largest probability against whether its class was right — the multiclass ECE evaluate reports.
    private static double? TopLabelEce(List<(float[] Probabilities, int Label, double Weight)> rows, Func<float[], float[]> map)
        => WeightedMetrics.Ece([.. rows.Select(r =>
        {
            var p = map(r.Probabilities);
            var top = 0;
            for (var i = 1; i < p.Length; i++)
                if (p[i] > p[top]) top = i;
            return ((double)p[top], top == r.Label, r.Weight);
        })]);

    /// <summary>
    /// Each labeled row's (class probabilities, true slot, weight) from a multiclass model's output, or
    /// <c>null</c> when <c>Score</c> is not a vector of the label key's width. Slot i is the class whose key is i + 1.
    /// </summary>
    public static IReadOnlyList<(float[] Probabilities, int Label, double Weight)>? MulticlassRowsOf(
        IDataView scored, string labelColumn, IReadOnlyList<double>? weights = null)
    {
        var score = scored.Schema.GetColumnOrNull("Score");
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        if (score is null || label is null
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType itemType } vector
            || itemType != NumberDataViewType.Single
            || label.Value.Type is not KeyDataViewType key
            || vector.Size != (int)key.Count)
            return null;
        var rows = new List<(float[], int, double)>();
        using var cursor = scored.GetRowCursor([score.Value, label.Value]);
        var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
        var getLabel = cursor.GetGetter<uint>(label.Value);
        VBuffer<float> scores = default;
        uint labelKey = default;
        var i = 0;
        while (cursor.MoveNext())
        {
            getScore(ref scores);
            getLabel(ref labelKey);
            var w = weights is null ? 1.0 : weights[i];
            i++;
            if (labelKey == 0)
                continue;   // missing label
            rows.Add((scores.DenseValues().ToArray(), (int)labelKey - 1, w));
        }
        return rows;
    }

    // The model followed by the tempering step. The step's Score width and annotations come from the
    // model's output on the schema it is actually given — a saved model need not carry its input
    // schema — so the chain is built on first use for a schema and kept for it. Never saved: the
    // calibration lives in its own file, beside the model.
    private sealed class TemperedModel(MLContext mlContext, ITransformer model, ProbabilityCalibration calibration) : ITransformer
    {
        private readonly object _gate = new();
        private (DataViewSchema Schema, ITransformer Chain)? _built;

        public bool IsRowToRowMapper => model.IsRowToRowMapper;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => ChainFor(inputSchema).GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => ChainFor(inputSchema).GetRowToRowMapper(inputSchema);

        public IDataView Transform(IDataView input) => ChainFor(input.Schema).Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) =>
            throw new NotSupportedException("A calibrated model is saved as the model and its calibration file, not as one model.");

        private ITransformer ChainFor(DataViewSchema schema)
        {
            lock (_gate)
            {
                if (_built is { } built && ReferenceEquals(built.Schema, schema))
                    return built.Chain;
                var chain = calibration.ApplyTemperature(mlContext, model, schema);
                _built = (schema, chain);
                return chain;
            }
        }
    }

    // Replaces Score with the tempered probabilities. The new column carries the old one's class-name
    // annotations (SlotNames, TrainingLabelValues): MulticlassScoreVocabulary names the classes from
    // them, and a replacement without them would leave every probability keyed by position.
    private ITransformer ApplyTemperature(MLContext mlContext, ITransformer model, DataViewSchema inputSchema)
    {
        var score = model.GetOutputSchema(inputSchema).GetColumnOrNull("Score");
        if (score?.Type is not VectorDataViewType { Size: > 0 } vector || vector.ItemType != NumberDataViewType.Single)
            return model;
        var width = vector.Size;

        var input = SchemaDefinition.Create(typeof(ScoreVector));
        input["Score"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, width);
        var output = SchemaDefinition.Create(typeof(ScoreVector));
        output["Score"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, width);
        foreach (var annotation in score.Value.Annotations.Schema)
            CopyAnnotation(score.Value, annotation, output["Score"]);

        var step = mlContext.Transforms.CustomMapping<ScoreVector, ScoreVector>(
                (row, tempered) => tempered.Score = Temper(row.Score),
                contractName: null, inputSchemaDefinition: input, outputSchemaDefinition: output)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<ScoreVector>(), input));
        return new TransformerChain<ITransformer>(model, step);
    }

    private static void CopyAnnotation(DataViewSchema.Column source, DataViewSchema.Column annotation, SchemaDefinition.Column target)
    {
        switch (annotation.Type)
        {
            case VectorDataViewType { ItemType: TextDataViewType }:
                {
                    VBuffer<ReadOnlyMemory<char>> value = default;
                    source.Annotations.GetValue(annotation.Name, ref value);
                    target.AddAnnotation(annotation.Name, value, annotation.Type);
                    break;
                }
            case VectorDataViewType { ItemType: NumberDataViewType n } when n == NumberDataViewType.Single:
                {
                    VBuffer<float> value = default;
                    source.Annotations.GetValue(annotation.Name, ref value);
                    target.AddAnnotation(annotation.Name, value, annotation.Type);
                    break;
                }
                // Other annotations (score kind, column set) describe the trainer's output; the vocabulary
                // does not read them, and a scored column without them still scores.
        }
    }

    /// <summary>The class probabilities temperature reads and writes.</summary>
    public sealed class ScoreVector
    {
        public float[] Score { get; set; } = [];
    }

    private static double Logit(double p)
    {
        const double eps = 1e-7;
        p = Math.Clamp(p, eps, 1 - eps);
        return Math.Log(p / (1 - p));
    }

    private static double Sigmoid(double z) => 1 / (1 + Math.Exp(-z));

    /// <summary>The input calibration reads.</summary>
    public sealed class ProbabilityInput
    {
        public float Probability { get; set; }
    }

    /// <summary>What it writes.</summary>
    public sealed class CalibratedOutput
    {
        public float Probability { get; set; }
        public bool PredictedLabel { get; set; }
    }
}
