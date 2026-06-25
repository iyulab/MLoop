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

    public static AnalyzeEnvelope MapImportance(FeatureImportanceSummary? importance)
    {
        if (importance == null || importance.Scores.Count == 0)
            return new AnalyzeEnvelope("importance", true,
                "No feature importance scores available (requires a label and numeric/categorical features).",
                new { ranking = Array.Empty<object>(), lowVarianceCount = 0, highCorrPairsCount = 0, conditionNumber = 0.0 },
                Array.Empty<string>());

        var ranking = importance.Scores
            .OrderByDescending(s => s.Score)
            .Select(s => new { feature = s.Name, score = Math.Round(s.Score, 4) })
            .ToList();

        var flags = new List<string>();
        if (importance.ConditionNumber > 1000)
            flags.Add($"multicollinearity-suspected (condition number {importance.ConditionNumber:F0})");
        if (importance.LowVarianceCount > 0)
            flags.Add($"low-variance-features: {importance.LowVarianceCount}");
        if (importance.HighCorrPairsCount > 0)
            flags.Add($"high-correlation-pairs: {importance.HighCorrPairsCount}");

        var top = ranking[0].feature;
        var summary = $"{ranking.Count} feature(s) ranked; top = {top}.";

        return new AnalyzeEnvelope("importance", true, summary, new
        {
            ranking,
            lowVarianceCount = importance.LowVarianceCount,
            highCorrPairsCount = importance.HighCorrPairsCount,
            conditionNumber = Math.Round(importance.ConditionNumber, 2)
        }, flags);
    }

    public static AnalyzeEnvelope MapDistribution(DescriptiveReport? desc, DistributionReport? dist)
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
            if (s.Skewness.HasValue && Math.Abs(s.Skewness.Value) > 1)
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
