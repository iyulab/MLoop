using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// What an anomaly-detection table model's <c>Score</c> and <c>PredictedLabel</c> mean: the squared distance of a row from what
/// the kept principal components rebuild, and "anomalous" when that distance is above the value that
/// <see cref="Contamination"/> of the training rows exceed. It lives beside the model
/// (<see cref="ExperimentLayout.AnomalyScoringFileName"/>) and every path that loads the model applies it through
/// <see cref="ScoringModel.Load(MLContext, string, string?, ScoringLayers)"/>, so predict, serve, evaluate and the gate read the
/// same answers; training evaluates through the same layer.
/// </summary>
/// <remarks>
/// ML.NET's <c>RandomizedPca</c> scores a row by its <i>relative</i> rebuild error (the squared distance divided by the row's own
/// squared length) and calls a row anomalous above 0.5, a number that comes from no data. The division makes a row near the
/// centre — the most ordinary kind — score high, and the fixed 0.5 flagged 32–61 % of the rows of public tables whose true
/// anomalies were 1.5–6 %. On six public tables the mean area under the curve was 0.685 for the raw distance and 0.536 for the
/// relative one. The raw distance is computed here from the model's own eigenvectors and mean and the standardized row the model
/// already produces (<c>Features</c>), so the saved model is untouched; a model without this file keeps ML.NET's relative
/// score and its 0.5.
/// </remarks>
/// <param name="Threshold">The score above which a row is answered anomalous.</param>
/// <param name="Contamination">The share of the training rows above <paramref name="Threshold"/> by construction.</param>
/// <param name="Rows">The training rows whose scores the threshold was read from.</param>
/// <param name="Score">The score's definition — <c>reconstruction-error</c> is the only one.</param>
public sealed record AnomalyScoring(double Threshold, double Contamination, int Rows, string Score = AnomalyScoring.ReconstructionErrorScore)
{
    /// <summary>The score this layer writes: the squared distance from the components' rebuild.</summary>
    public const string ReconstructionErrorScore = "reconstruction-error";

    /// <summary>The annotation on the <c>Score</c> column that carries the threshold.</summary>
    public const string ThresholdAnnotation = "AnomalyThreshold";

    /// <summary>
    /// The threshold <paramref name="scored"/>'s <c>Score</c> column was judged against, or <c>null</c> when this layer did not
    /// score it (a model trained before it, or another task).
    /// </summary>
    public static double? ThresholdOf(DataViewSchema scored)
    {
        ArgumentNullException.ThrowIfNull(scored);
        if (scored.GetColumnOrNull("Score") is not { } score
            || score.Annotations.Schema.GetColumnOrNull(ThresholdAnnotation) is not { } annotation
            || annotation.Type != NumberDataViewType.Single)
            return null;
        float value = default;
        score.Annotations.GetValue(ThresholdAnnotation, ref value);
        return float.IsFinite(value) ? value : null;
    }

    /// <summary>The share of the training rows called anomalous when the project does not say.</summary>
    public const double DefaultContamination = 0.05;

    /// <summary>Whether <paramref name="contamination"/> is a share a quantile can be read at: above 0 and below one half.</summary>
    public static bool IsValidContamination(double contamination) => contamination is > 0 and < 0.5;

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    /// <summary>The file beside <paramref name="modelPath"/>, or <c>null</c> when there is none.</summary>
    public static AnomalyScoring? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<AnomalyScoring>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes this record beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => AtomicFile.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the file beside <paramref name="modelPath"/> lives.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.AnomalyScoringFileName);

    /// <summary><paramref name="model"/>, loaded from <paramref name="modelPath"/>, scored as the file beside it says when there is one.</summary>
    public static ITransformer ApplyBeside(MLContext mlContext, ITransformer model, string modelPath)
        => ReadBeside(modelPath) is { } scoring ? scoring.Apply(mlContext, model) : model;

    /// <summary>
    /// The anomaly model of <paramref name="model"/> — the principal components its last step carries — or <c>null</c> when it
    /// is not a RandomizedPca model.
    /// </summary>
    public static PcaModelParameters? PcaOf(ITransformer model)
    {
        // By the parameters the step carries, not by its generic type: a model that was saved and loaded comes back as a
        // prediction transformer of the parameters' base interface, not of PcaModelParameters.
        // A chain of any type (TransformerChain<T> is not covariant: a fitted one is typed by its last step, a loaded one is not).
        var last = model is IEnumerable<ITransformer> chain ? chain.LastOrDefault() : model;
        return last is IPredictionTransformer<object> { Model: PcaModelParameters pca } ? pca : null;
    }

    /// <summary>
    /// <paramref name="model"/> followed by a step that writes <c>Score</c> as the raw rebuild error and <c>PredictedLabel</c> as
    /// "above <see cref="Threshold"/>". In memory only — the saved model is unchanged, and removing the file returns ML.NET's own
    /// score. A model that is not a RandomizedPca one is answered unchanged.
    /// </summary>
    public ITransformer Apply(MLContext mlContext, ITransformer model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        if (PcaOf(model) is not { } pca)
            return model;

        var rebuild = Rebuild.Of(pca);
        var threshold = Threshold;
        var input = SchemaDefinition.Create(typeof(FeaturesIn));
        input[nameof(FeaturesIn.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, rebuild.Width);
        // The threshold rides on the score column, where whoever turns the score into a confidence reads it (ThresholdOf).
        var output = SchemaDefinition.Create(typeof(ScoredOut));
        output[nameof(ScoredOut.Score)].AddAnnotation(ThresholdAnnotation, (float)threshold, NumberDataViewType.Single);
        var step = mlContext.Transforms.CustomMapping<FeaturesIn, ScoredOut>(
                (row, scored) =>
                {
                    var error = rebuild.Error(row.Features);
                    scored.Score = error;
                    scored.PredictedLabel = float.IsFinite(error) && error > threshold;
                }, contractName: null, inputSchemaDefinition: input, outputSchemaDefinition: output)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<FeaturesIn>(), input));
        return new TransformerChain<ITransformer>(model, step);
    }

    /// <summary>
    /// What the training rows say of the score: the record for <paramref name="featurized"/> (rows with the standardized
    /// <c>Features</c> vector the model sees), with the threshold at the point <paramref name="contamination"/> of the rows score
    /// above. Rows whose score is not a number are left out of the quantile.
    /// </summary>
    /// <exception cref="InvalidOperationException">No row has a finite score — there is nothing to read a threshold from.</exception>
    public static AnomalyScoring Fit(IDataView featurized, PcaModelParameters pca, double contamination)
    {
        ArgumentNullException.ThrowIfNull(featurized);
        ArgumentNullException.ThrowIfNull(pca);
        if (!IsValidContamination(contamination))
            throw new ArgumentOutOfRangeException(nameof(contamination), contamination, "contamination must be above 0 and below 0.5");

        var rebuild = Rebuild.Of(pca);
        var scores = new List<double>();
        var features = featurized.Schema["Features"];
        using (var cursor = featurized.GetRowCursor([features]))
        {
            var getter = cursor.GetGetter<VBuffer<float>>(features);
            VBuffer<float> row = default;
            while (cursor.MoveNext())
            {
                getter(ref row);
                var error = rebuild.Error(row.DenseValues().ToArray());
                if (float.IsFinite(error))
                    scores.Add(error);
            }
        }
        if (scores.Count == 0)
            throw new InvalidOperationException("No training row has a finite anomaly score, so there is nothing to place the threshold from.");
        return new AnomalyScoring(QuantileAbove(scores, contamination), contamination, scores.Count);
    }

    /// <summary>
    /// The value that <paramref name="share"/> of <paramref name="scores"/> lie above — the (1 − share) quantile, taken as an
    /// order statistic, so a row at the threshold is not above it and ties cannot push the share over.
    /// </summary>
    public static double QuantileAbove(IReadOnlyList<double> scores, double share)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (scores.Count == 0)
            throw new ArgumentException("There are no scores to take a quantile of.", nameof(scores));
        var sorted = scores.OrderBy(s => s).ToArray();
        var index = (int)Math.Ceiling((1 - share) * sorted.Length) - 1;
        return sorted[Math.Clamp(index, 0, sorted.Length - 1)];
    }

    /// <summary>The squared distance of one standardized row from the components' rebuild.</summary>
    internal sealed class Rebuild
    {
        private readonly float[] _mean;
        private readonly float[][] _axes;

        private Rebuild(float[] mean, float[][] axes) => (_mean, _axes) = (mean, axes);

        /// <summary>How many columns a row has.</summary>
        public int Width => _mean.Length;

        public static Rebuild Of(PcaModelParameters pca)
        {
            VBuffer<float>[] vectors = default!;
            pca.GetEigenVectors(ref vectors, out _);
            VBuffer<float> mean = default;
            pca.GetMean(ref mean);
            return new Rebuild(mean.DenseValues().ToArray(), vectors.Select(v => v.DenseValues().ToArray()).ToArray());
        }

        /// <summary>
        /// ‖r‖² − Σ(axis·r)² with r = row − mean — the part of the centred row the orthonormal axes do not carry. NaN for a row
        /// that has a missing value or is not as wide as the model's.
        /// </summary>
        public float Error(float[]? row)
        {
            if (row is null || row.Length != _mean.Length)
                return float.NaN;
            double norm = 0;
            var centred = new double[row.Length];
            for (var j = 0; j < row.Length; j++)
            {
                centred[j] = (double)row[j] - _mean[j];
                norm += centred[j] * centred[j];
            }
            double carried = 0;
            foreach (var axis in _axes)
            {
                double projection = 0;
                for (var j = 0; j < centred.Length; j++)
                    projection += axis[j] * centred[j];
                carried += projection * projection;
            }
            // Rounding can put the difference a hair below zero for a row the axes carry exactly.
            return (float)Math.Max(0, norm - carried);
        }
    }

    /// <summary>The standardized row the model's step scores.</summary>
    public sealed class FeaturesIn
    {
        public float[] Features { get; set; } = [];
    }

    /// <summary>The answer the step writes over the trainer's.</summary>
    public sealed class ScoredOut
    {
        public float Score { get; set; }
        public bool PredictedLabel { get; set; }
    }
}
