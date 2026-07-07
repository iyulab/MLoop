using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

namespace MLoop.Core.Detection;

/// <summary>Options for one-shot SR-CNN anomaly detection.</summary>
public sealed record OneShotAnomalyOptions
{
    /// <summary>Anomaly decision threshold in [0, 1]; a point is anomalous when its SR score exceeds this.</summary>
    public double Threshold { get; init; } = 0.3;

    /// <summary>Boundary sensitivity in [0, 100] — larger values produce tighter margin bounds.</summary>
    public double Sensitivity { get; init; } = 99.0;

    /// <summary>
    /// Seasonality period (number of points per cycle). <see langword="null"/> auto-detects via
    /// <c>DetectSeasonality</c>; 0 disables seasonal decomposition.
    /// </summary>
    public int? Period { get; init; }
}

/// <summary>One detected point: the SPC-chart triple (expected/upper/lower) plus the anomaly verdict.</summary>
public sealed record OneShotAnomalyPoint
{
    public required int Index { get; init; }
    public required double Value { get; init; }
    public required bool IsAnomaly { get; init; }
    /// <summary>SR (spectral-residual) anomaly score in [0, 1].</summary>
    public required double Score { get; init; }
    public required double ExpectedValue { get; init; }
    public required double UpperBound { get; init; }
    public required double LowerBound { get; init; }
}

/// <summary>Result of a one-shot detection pass over an entire series.</summary>
public sealed record OneShotAnomalyResult
{
    public required IReadOnlyList<OneShotAnomalyPoint> Points { get; init; }

    /// <summary>The seasonality period used (auto-detected or caller-supplied); 0 = non-seasonal.</summary>
    public required int Period { get; init; }

    public int AnomalyCount => Points.Count(p => p.IsAnomaly);
}

/// <summary>
/// One-shot (batch) time-series anomaly detection over an entire series via
/// <c>DetectEntireAnomalyBySrCnn</c> — no train/predict split, no model artifact. This is the only
/// SR-CNN surface that supports <see cref="SrCnnDetectMode.AnomalyAndMargin"/>, i.e. the only way to
/// get per-point ExpectedValue/UpperBound/LowerBound (SPC-chart bounds): the streaming
/// <c>DetectAnomalyBySrCnn</c> transformer used by trained time-series-anomaly models emits only
/// [alert, score, mag] and has no margin mode upstream.
/// </summary>
/// <remarks>
/// Single authority for the 7-slot <c>AnomalyAndMargin</c> output contract — consumers must use the
/// <c>*Slot</c> constants rather than re-hardcoding indices.
/// </remarks>
public static class SrCnnOneShotDetector
{
    /// <summary>SR-CNN requires at least this many points to form its spectral window.</summary>
    public const int MinimumPoints = 12;

    // DetectEntireAnomalyBySrCnn AnomalyAndMargin output slots (VBuffer<double>, length 7).
    public const int IsAnomalySlot = 0;
    public const int RawScoreSlot = 1;
    public const int MagSlot = 2;
    public const int ExpectedValueSlot = 3;
    public const int BoundaryUnitSlot = 4;
    public const int UpperBoundSlot = 5;
    public const int LowerBoundSlot = 6;

    private sealed class SeriesPoint
    {
        public double Value { get; set; }
    }

    private sealed class SrCnnPrediction
    {
        [VectorType]
        public double[] Prediction { get; set; } = [];
    }

    /// <summary>
    /// Runs one-shot detection over <paramref name="series"/> and returns every point with its
    /// anomaly verdict, score, and margin bounds.
    /// </summary>
    /// <exception cref="ArgumentException">Series too short, or options out of range.</exception>
    public static OneShotAnomalyResult Detect(IReadOnlyList<double> series, OneShotAnomalyOptions? options = null)
    {
        options ??= new OneShotAnomalyOptions();

        if (series.Count < MinimumPoints)
            throw new ArgumentException(
                $"One-shot anomaly detection requires at least {MinimumPoints} data points (got {series.Count}).",
                nameof(series));
        if (options.Threshold is < 0.0 or > 1.0)
            throw new ArgumentException($"Threshold must be in [0, 1] (got {options.Threshold}).", nameof(options));
        if (options.Sensitivity is < 0.0 or > 100.0)
            throw new ArgumentException($"Sensitivity must be in [0, 100] (got {options.Sensitivity}).", nameof(options));
        if (options.Period is < 0)
            throw new ArgumentException($"Period must be >= 0 (got {options.Period}).", nameof(options));

        var ml = new MLContext(seed: 42);
        var dataView = ml.Data.LoadFromEnumerable(series.Select(v => new SeriesPoint { Value = v }));

        // DetectSeasonality returns -1 when the series has no significant period.
        var period = options.Period
            ?? Math.Max(0, ml.AnomalyDetection.DetectSeasonality(dataView, nameof(SeriesPoint.Value)));

        var srOptions = new SrCnnEntireAnomalyDetectorOptions
        {
            Threshold = options.Threshold,
            Sensitivity = options.Sensitivity,
            DetectMode = SrCnnDetectMode.AnomalyAndMargin,
            Period = period,
            BatchSize = -1, // one-shot: the entire series is a single batch
        };

        var output = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(
            dataView, nameof(SrCnnPrediction.Prediction), nameof(SeriesPoint.Value), srOptions);

        var points = new List<OneShotAnomalyPoint>(series.Count);
        var index = 0;
        foreach (var row in ml.Data.CreateEnumerable<SrCnnPrediction>(output, reuseRowObject: false))
        {
            var p = row.Prediction;
            if (p.Length <= LowerBoundSlot)
                throw new InvalidOperationException(
                    $"SR-CNN AnomalyAndMargin output has {p.Length} slots (expected {LowerBoundSlot + 1}) — upstream contract changed.");

            points.Add(new OneShotAnomalyPoint
            {
                Index = index,
                Value = series[index],
                IsAnomaly = p[IsAnomalySlot] != 0,
                Score = p[RawScoreSlot],
                ExpectedValue = p[ExpectedValueSlot],
                UpperBound = p[UpperBoundSlot],
                LowerBound = p[LowerBoundSlot],
            });
            index++;
        }

        return new OneShotAnomalyResult { Points = points, Period = period };
    }
}
