using System.Text.Json;
using System.Text.Json.Serialization;
using DataLens.Models;

namespace MLoop.CLI.Commands;

/// <summary>
/// Serializes non-finite doubles (NaN, ±Infinity) as JSON null instead of crashing or
/// emitting the invalid <c>NaN</c>/<c>Infinity</c> tokens. Statistical aspects such as
/// <c>importance</c> can produce an infinite condition number on multicollinear data — the
/// very case the aspect is meant to diagnose — and named-literal output would break strict
/// JSON parsers (e.g. the MCP bridge's <c>JSON.parse</c>). System.Text.Json routes
/// <see cref="Nullable{Double}"/> through this converter too, so both <c>double</c> and
/// <c>double?</c> stay valid, parseable JSON.
/// </summary>
internal sealed class FiniteDoubleConverter : JsonConverter<double>
{
    public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        => reader.GetDouble();

    public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
    {
        if (double.IsFinite(value))
            writer.WriteNumberValue(value);
        else
            writer.WriteNullValue();
    }
}

/// <summary>Structured JSON envelope for `mloop analyze` aspects (LLM-consumable).</summary>
public sealed record AnalyzeEnvelope(
    string Aspect,
    bool Available,
    string Summary,
    object? Data,
    IReadOnlyList<string> Flags);

/// <summary>Maps DataLens reports to the analyze JSON envelope.</summary>
public static class AnalyzeJson
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new FiniteDoubleConverter() }
    };

    public static string Serialize(AnalyzeEnvelope env) => JsonSerializer.Serialize(env, Options);

    public static AnalyzeEnvelope Unavailable(string aspect) => new(
        aspect, false,
        "DataLens is not available; install the DataLens NuGet package to enable analysis.",
        null, Array.Empty<string>());

    public static AnalyzeEnvelope MapProfile(
        ProfileReport? report,
        Dictionary<string, (long MissingCount, int UniqueCount)> stats,
        int rowCount)
    {
        var profileLookup = new Dictionary<string, ColumnProfile>();
        if (report?.Columns != null)
            foreach (var c in report.Columns) profileLookup[c.Name] = c;

        var columns = new List<object>();
        var flags = new List<string>();

        foreach (var kvp in stats)
        {
            var name = kvp.Key;
            var missing = kvp.Value.MissingCount;
            var unique = kvp.Value.UniqueCount;
            var missingPct = rowCount > 0 ? missing / (double)rowCount * 100 : 0;
            var isConstant = unique == 1;

            profileLookup.TryGetValue(name, out var p);
            columns.Add(new
            {
                name,
                dataType = p?.DataType ?? "unknown",
                missingCount = missing,
                missingPercent = Math.Round(missingPct, 2),
                uniqueCount = unique,
                isConstant,
                mean = p?.Mean,
                stdDev = p?.StdDev,
                min = p?.Min,
                max = p?.Max
            });

            if (isConstant) flags.Add($"constant-column: {name}");
            else if (missingPct > 30) flags.Add($"high-null: {name} ({missingPct:F1}%)");
        }

        var summary = $"{columns.Count} column(s); "
            + $"{flags.Count(f => f.StartsWith("constant-column"))} constant, "
            + $"{flags.Count(f => f.StartsWith("high-null"))} high-null.";

        return new AnalyzeEnvelope("profile", true, summary, new { columns }, flags);
    }

    public static AnalyzeEnvelope MapCorrelation(CorrelationReport? report)
    {
        if (report == null)
            return new AnalyzeEnvelope("correlation", true, "No numeric columns to correlate.",
                new { highPairs = Array.Empty<object>(), multicollinearity = false }, Array.Empty<string>());

        var pairs = report.HighCorrelationPairs
            .OrderByDescending(p => p.AbsValue)
            .Select(p => new { column1 = p.Column1, column2 = p.Column2, pearson = Math.Round(p.Value, 4) })
            .ToList();

        bool multicollinearity = report.HighCorrelationPairs.Any(p => p.AbsValue >= 0.8);

        var flags = report.HighCorrelationPairs
            .Where(p => p.AbsValue >= 0.9)
            .OrderByDescending(p => p.AbsValue)
            .Select(p => $"duplicate-feature-candidate: {p.Column1}~{p.Column2} (r={p.Value:F2})")
            .ToList();

        var summary = pairs.Count == 0
            ? "No high-correlation pairs found among numeric columns."
            : $"{pairs.Count} high-correlation pair(s); multicollinearity {(multicollinearity ? "likely" : "unlikely")}.";

        return new AnalyzeEnvelope("correlation", true, summary,
            new { highPairs = pairs, multicollinearity }, flags);
    }

    /// <summary>
    /// Maps DataLens feature analysis into the importance envelope.
    /// <para>
    /// The <c>analyze importance</c> command always carries a label, so it must surface
    /// <em>target-aware</em> predictive importance (permutation for numeric targets,
    /// mutual-info/ANOVA for categorical) rather than the target-agnostic structural
    /// (variance/condition-number) ranking — structural importance over-ranks features that
    /// the model never uses (SEQ026: KWh ranked high structurally yet predictively inert).
    /// </para>
    /// <para>
    /// In every case the label column itself is excluded: DataLens computes importance over
    /// the full numeric matrix (target included), so the target self-references at rank #1
    /// unless filtered here. Structural diagnostics (condition number, low-variance /
    /// high-correlation counts) are still surfaced as supplementary signal.
    /// </para>
    /// </summary>
    /// <summary>One ranked feature: its name and the score under the chosen importance method.</summary>
    public readonly record struct RankedFeature(string Feature, double Score);

    /// <summary>
    /// True when a candidate importance source has at least one finite, non-zero score among the
    /// non-label features — i.e. it actually discriminates. Guards against degenerate all-zero
    /// rankings (see RankImportance) before that source is preferred over structural.
    /// </summary>
    private static bool HasSignal(IEnumerable<(string Name, double Score)>? scores, string? label)
    {
        if (scores == null) return false;
        return scores
            .Where(s => string.IsNullOrEmpty(label) || !string.Equals(s.Name, label, StringComparison.Ordinal))
            .Any(s => double.IsFinite(s.Score) && Math.Abs(s.Score) > 1e-9);
    }

    /// <summary>
    /// Selects the importance ranking from a DataLens feature report: target-aware source first
    /// (permutation → mutual-info → ANOVA), structural variance last; the label column is always
    /// excluded. Shared by the JSON envelope and the console renderer so both stay consistent.
    /// Returns method = null with an empty list when no scores are available.
    /// </summary>
    public static (string? Method, IReadOnlyList<RankedFeature> Ranking) RankImportance(
        FeatureReport? features, string? label)
    {
        var structural = features?.Importance;

        // Choose the ranking source: target-aware first, structural as last resort.
        // A target-aware source is only used when it carries signal among the non-label
        // features — DataLens currently leaves the target column in the feature matrix, so a
        // model can predict the target from itself and leave every real feature at importance 0
        // (degenerate). When that happens we fall back to structural rather than surface a
        // useless all-zero ranking. (See DataLens issue: target not excluded from matrix.)
        List<(string Name, double Score)>? raw = null;
        string? method = null;

        if (HasSignal(features?.Permutation?.Features.Select(f => (f.Name, f.Importance)), label))
        {
            raw = features!.Permutation!.Features.Select(f => (f.Name, f.Importance)).ToList();
            method = "permutation";
        }
        else if (HasSignal(features?.MutualInfo?.Features.Select(f => (f.Name, f.Mi)), label))
        {
            raw = features!.MutualInfo!.Features.Select(f => (f.Name, f.Mi)).ToList();
            method = "mutual-info";
        }
        else if (HasSignal(features?.Anova?.Features.Select(f => (f.Name, f.FStatistic)), label))
        {
            raw = features!.Anova!.Features.Select(f => (f.Name, f.FStatistic)).ToList();
            method = "anova-f";
        }
        else if (structural is { Scores.Count: > 0 })
        {
            raw = structural.Scores.Select(s => (s.Name, s.Score)).ToList();
            method = "structural";
        }

        if (raw == null || raw.Count == 0)
            return (null, Array.Empty<RankedFeature>());

        var ranking = raw
            .Where(r => string.IsNullOrEmpty(label) || !string.Equals(r.Name, label, StringComparison.Ordinal))
            .OrderByDescending(r => r.Score)
            .Select(r => new RankedFeature(r.Name, Math.Round(r.Score, 4)))
            .ToList();

        return (method, ranking);
    }

    public static AnalyzeEnvelope MapImportance(FeatureReport? features, string? label)
    {
        var structural = features?.Importance;
        var (method, rankedFeatures) = RankImportance(features, label);

        if (method == null)
            return new AnalyzeEnvelope("importance", true,
                "No feature importance scores available (requires a label and numeric/categorical features).",
                new { method = (string?)null, ranking = Array.Empty<object>(), lowVarianceCount = 0, highCorrPairsCount = 0, conditionNumber = 0.0 },
                Array.Empty<string>());

        var ranking = rankedFeatures
            .Select(r => new { feature = r.Feature, score = r.Score })
            .ToList();

        var conditionNumber = structural?.ConditionNumber ?? 0.0;
        var lowVariance = structural?.LowVarianceCount ?? 0;
        var highCorrPairs = structural?.HighCorrPairsCount ?? 0;

        var flags = new List<string>();
        if (conditionNumber > 1000)
            flags.Add($"multicollinearity-suspected (condition number {conditionNumber:F0})");
        if (lowVariance > 0)
            flags.Add($"low-variance-features: {lowVariance}");
        if (highCorrPairs > 0)
            flags.Add($"high-correlation-pairs: {highCorrPairs}");

        var top = ranking.Count > 0 ? ranking[0].feature : "(none)";
        var summary = $"{ranking.Count} feature(s) ranked by {method}; top = {top}.";

        return new AnalyzeEnvelope("importance", true, summary, new
        {
            method,
            ranking,
            lowVarianceCount = lowVariance,
            highCorrPairsCount = highCorrPairs,
            conditionNumber = Math.Round(conditionNumber, 2)
        }, flags);
    }

    /// <summary>
    /// A "highly-skewed" flag is only actionable for continuous columns. A low-cardinality column
    /// (e.g. a binary 1/2 rectifier id) has a mathematically-defined skewness that is really just its
    /// class balance — flagging it as skewed misleads a downstream agent into proposing a meaningless
    /// transform (F-02, observed in a live agent run). When per-column cardinality is supplied, the
    /// skew flag is suppressed at/below this distinct-value count.
    /// </summary>
    private const int SkewFlagMinCardinality = 3;

    public static AnalyzeEnvelope MapDistribution(
        DescriptiveReport? desc,
        DistributionReport? dist,
        IReadOnlyDictionary<string, (long MissingCount, int UniqueCount)>? stats)
    {
        var skew = new Dictionary<string, (double? Skewness, double? Kurtosis)>();
        if (desc?.Columns != null)
            foreach (var c in desc.Columns) skew[c.Name] = (c.Skewness, c.Kurtosis);

        var names = new List<string>();
        var byName = new Dictionary<string, ColumnDistribution>();
        if (dist?.Columns != null)
            foreach (var c in dist.Columns) { byName[c.Name] = c; names.Add(c.Name); }
        foreach (var n in skew.Keys) if (!byName.ContainsKey(n)) names.Add(n);

        var columns = new List<object>();
        var flags = new List<string>();
        foreach (var name in names)
        {
            skew.TryGetValue(name, out var s);
            byName.TryGetValue(name, out var d);
            columns.Add(new
            {
                name,
                skewness = s.Skewness,
                kurtosis = s.Kurtosis,
                isNormal = d?.IsNormal,
                sampleSize = d?.SampleSize,
                shapiroWilkP = d?.SwPValue,
                jarqueBeraP = d?.JbPValue,
                andersonDarlingP = d?.AdPValue
            });
            var lowCardinality = stats != null
                && stats.TryGetValue(name, out var st)
                && st.UniqueCount < SkewFlagMinCardinality;
            if (s.Skewness.HasValue && Math.Abs(s.Skewness.Value) > 1 && !lowCardinality)
                flags.Add($"highly-skewed: {name} (skew={s.Skewness.Value:F2})");
        }

        var summary = columns.Count == 0
            ? "No numeric columns to analyze for distribution."
            : $"{columns.Count} numeric column(s); {flags.Count} highly skewed.";

        return new AnalyzeEnvelope("distribution", true, summary, new { columns }, flags);
    }

    public static AnalyzeEnvelope MapOutliers(OutlierReport? report)
    {
        if (report == null)
            return new AnalyzeEnvelope("outliers", true, "No outlier analysis available (requires numeric columns).",
                new { outlierCount = 0, totalRows = 0, outlierPercentage = 0.0, isolationForestThreshold = (double?)null },
                Array.Empty<string>());

        var flags = new List<string>();
        if (report.OutlierPercentage > 10)
            flags.Add($"high-outlier-rate ({report.OutlierPercentage:F1}%)");

        var summary = $"{report.OutlierCount} / {report.TotalRows} rows are outliers ({report.OutlierPercentage:F2}%).";

        return new AnalyzeEnvelope("outliers", true, summary, new
        {
            outlierCount = report.OutlierCount,
            totalRows = report.TotalRows,
            outlierPercentage = Math.Round(report.OutlierPercentage, 2),
            isolationForestThreshold = report.IsolationForest != null
                ? Math.Round(report.IsolationForest.Threshold, 4) : (double?)null
        }, flags);
    }
}
