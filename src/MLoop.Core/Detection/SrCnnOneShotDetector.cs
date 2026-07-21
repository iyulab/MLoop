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

/// <summary>
/// One detected point: the anomaly verdict plus two distinct bands around the expected value.
/// They answer different questions and must not be conflated — see <see cref="MarginUpper"/> and
/// <see cref="ControlUpper"/>.
/// </summary>
public sealed record OneShotAnomalyPoint
{
    public required int Index { get; init; }
    public required double Value { get; init; }
    public required bool IsAnomaly { get; init; }
    /// <summary>SR (spectral-residual) anomaly score in [0, 1].</summary>
    public required double Score { get; init; }
    public required double ExpectedValue { get; init; }

    /// <summary>
    /// Upper edge of SR-CNN's detection margin — <c>expected + unit * factor(sensitivity)</c>.
    /// This is the gate that produced <see cref="IsAnomaly"/>: upstream clears the flag for any
    /// point inside the margin, so <c>IsAnomaly ⇒ outside</c>. The converse does NOT hold (the
    /// margin never raises a flag), and <c>unit</c> scales with the series' level/trend rather than
    /// its spread — so this is verdict traceability, <b>not</b> a control limit.
    /// </summary>
    public required double MarginUpper { get; init; }

    /// <inheritdoc cref="MarginUpper"/>
    public required double MarginLower { get; init; }

    /// <summary>
    /// Upper control limit (UCL) — <c>expected + 3 * σ̂</c>, where σ̂ is the robust dispersion of the
    /// residuals (see <see cref="OneShotAnomalyResult.ResidualSigma"/>). Shewhart semantics: normal
    /// operation plots inside the band, so this is the band to chart.
    /// </summary>
    public required double ControlUpper { get; init; }

    /// <inheritdoc cref="ControlUpper"/>
    public required double ControlLower { get; init; }
}

/// <summary>Result of a one-shot detection pass over an entire series.</summary>
public sealed record OneShotAnomalyResult
{
    public required IReadOnlyList<OneShotAnomalyPoint> Points { get; init; }

    /// <summary>The seasonality period used (auto-detected or caller-supplied); 0 = non-seasonal.</summary>
    public required int Period { get; init; }

    /// <summary>
    /// Robust dispersion of the residuals (value − expected) that the control limits are built from;
    /// 0 for a series with no residual variation.
    /// </summary>
    public required double ResidualSigma { get; init; }

    public int AnomalyCount => Points.Count(p => p.IsAnomaly);
}

/// <summary>
/// One-shot (batch) time-series anomaly detection over an entire series via
/// <c>DetectEntireAnomalyBySrCnn</c> — no train/predict split, no model artifact. This is the only
/// SR-CNN surface that supports <see cref="SrCnnDetectMode.AnomalyAndMargin"/>, i.e. the only way to
/// get a per-point expected value and detection margin: the streaming <c>DetectAnomalyBySrCnn</c>
/// transformer used by trained time-series-anomaly models emits only [alert, score, mag] and has no
/// margin mode upstream.
/// </summary>
/// <remarks>
/// <para>
/// Single authority for the 7-slot <c>AnomalyAndMargin</c> output contract — consumers must use the
/// <c>*Slot</c> constants rather than re-hardcoding indices.
/// </para>
/// <para>
/// Also the authority for the two-band contract. Upstream's margin is a detection gate scaled by
/// <c>CalculateBoundaryUnit</c>, which measures the series' level/trend and is blind to its spread —
/// it cannot serve as a control limit at any sensitivity. Control limits are therefore derived here
/// from the residual dispersion; see <see cref="OneShotAnomalyPoint.MarginUpper"/> and
/// <see cref="OneShotAnomalyPoint.ControlUpper"/>.
/// </para>
/// </remarks>
public static class SrCnnOneShotDetector
{
    /// <summary>SR-CNN requires at least this many points to form its spectral window.</summary>
    public const int MinimumPoints = 12;

    /// <summary>Shewhart 3-sigma control limits — the textbook default for a control chart.</summary>
    public const double ControlLimitSigmas = 3.0;

    /// <summary>Consistency constant making the median absolute deviation an estimator of sigma
    /// for normally distributed residuals (1 / Φ⁻¹(0.75)).</summary>
    private const double MadToSigma = 1.4826;

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

        var raw = new List<(bool IsAnomaly, double Score, double Expected, double Upper, double Lower)>(series.Count);
        foreach (var row in ml.Data.CreateEnumerable<SrCnnPrediction>(output, reuseRowObject: false))
        {
            var p = row.Prediction;
            if (p.Length <= LowerBoundSlot)
                throw new InvalidOperationException(
                    $"SR-CNN AnomalyAndMargin output has {p.Length} slots (expected {LowerBoundSlot + 1}) — upstream contract changed.");

            raw.Add((p[IsAnomalySlot] != 0, p[RawScoreSlot], p[ExpectedValueSlot], p[UpperBoundSlot], p[LowerBoundSlot]));
        }

        // Control limits need the whole series' residual dispersion, so they can only be built once
        // every point's expected value is known.
        var residualSigma = EstimateResidualSigma(series, raw);
        var halfBand = ControlLimitSigmas * residualSigma;

        var points = new List<OneShotAnomalyPoint>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            var r = raw[i];
            points.Add(new OneShotAnomalyPoint
            {
                Index = i,
                Value = series[i],
                IsAnomaly = r.IsAnomaly,
                Score = r.Score,
                ExpectedValue = r.Expected,
                MarginUpper = r.Upper,
                MarginLower = r.Lower,
                ControlUpper = r.Expected + halfBand,
                ControlLower = r.Expected - halfBand,
            });
        }

        return new OneShotAnomalyResult { Points = points, Period = period, ResidualSigma = residualSigma };
    }

    /// <summary>
    /// Robust scale of the residuals (value − expected), excluding the points SR-CNN flagged so the
    /// anomalies do not inflate the band that is supposed to exclude them. MAD is the primary
    /// estimator; it degenerates to 0 when more than half the residuals coincide, so a standard
    /// deviation stands in before giving up on 0 (a genuinely constant series has no spread).
    /// </summary>
    private static double EstimateResidualSigma(
        IReadOnlyList<double> series,
        IReadOnlyList<(bool IsAnomaly, double Score, double Expected, double Upper, double Lower)> raw)
    {
        var residuals = new List<double>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            if (!raw[i].IsAnomaly)
                residuals.Add(series[i] - raw[i].Expected);
        }

        // Every point flagged: fall back to the full set rather than emitting an undefined band.
        if (residuals.Count == 0)
        {
            for (int i = 0; i < raw.Count; i++)
                residuals.Add(series[i] - raw[i].Expected);
        }

        var median = Median(residuals);
        var mad = Median(residuals.Select(r => Math.Abs(r - median)).ToList());
        if (mad > 0)
            return MadToSigma * mad;

        var mean = residuals.Average();
        var variance = residuals.Sum(r => (r - mean) * (r - mean)) / residuals.Count;
        return Math.Sqrt(variance);
    }

    private static double Median(List<double> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        var mid = sorted.Count / 2;
        return sorted.Count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
}
