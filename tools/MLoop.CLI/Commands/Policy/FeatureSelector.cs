using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.CLI.Commands.Policy;

/// <summary>
/// Pure, filesystem-free feature include/exclude policy editing via ColumnOverride
/// (Type:"ignore" excludes a column from features). D-P2b: --keep computes the
/// complement against the actual train-data header; the label is always kept.
/// </summary>
public static class FeatureSelector
{
    public static void Drop(Dictionary<string, ColumnOverride> columns, IEnumerable<string> drop)
    {
        foreach (var c in drop)
        {
            var existing = columns.TryGetValue(c, out var o) ? o.Description : null;
            columns[c] = new ColumnOverride { Type = "ignore", Description = existing };
        }
    }

    public static IReadOnlyList<string> KeepComplement(
        IEnumerable<string> allColumns, IEnumerable<string> keep, string? label)
    {
        var keepSet = new HashSet<string>(keep, StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrEmpty(label)) keepSet.Add(label);
        return allColumns.Where(c => !keepSet.Contains(c)).ToList();
    }

    public static void ApplyKeep(
        Dictionary<string, ColumnOverride> columns, IReadOnlyList<string> allColumns,
        IEnumerable<string> keep, string? label)
        => Drop(columns, KeepComplement(allColumns, keep, label));

    /// <summary>
    /// Splits requested column names into those present in the data header and those absent
    /// (case-sensitive, matching train-time ColumnOverride key lookup). Lets
    /// <c>features select --drop</c> warn on — and skip — columns that don't exist
    /// (e.g. an agent's hallucinated or typo'd name) instead of silently recording a phantom
    /// ignore override that never takes effect while the real column stays a feature.
    /// </summary>
    public static (IReadOnlyList<string> Known, IReadOnlyList<string> Unknown) PartitionByHeader(
        IEnumerable<string> requested, IReadOnlyCollection<string> header)
    {
        var headerSet = new HashSet<string>(header, StringComparer.Ordinal);
        var known = new List<string>();
        var unknown = new List<string>();
        foreach (var c in requested)
            (headerSet.Contains(c) ? known : unknown).Add(c);
        return (known, unknown);
    }

    public static int Reset(Dictionary<string, ColumnOverride> columns)
    {
        var ignored = columns
            .Where(kv => string.Equals(kv.Value.Type, "ignore", StringComparison.OrdinalIgnoreCase))
            .Select(kv => kv.Key)
            .ToList();
        foreach (var k in ignored) columns.Remove(k);
        return ignored.Count;
    }
}
