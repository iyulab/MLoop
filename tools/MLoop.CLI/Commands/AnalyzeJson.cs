using System.Text.Json;
using DataLens.Models;

namespace MLoop.CLI.Commands;

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
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
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
}
