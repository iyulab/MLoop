using MLoop.Core.Preprocessing;

namespace MLoop.CLI.Commands.Policy;

/// <summary>
/// Pure, filesystem-free editing of a model's prep step list. Upsert key is
/// (normalized type, column-set) — D-P2a: the same transform may target different
/// column groups, and re-issuing the same (type, columns) is idempotent.
/// </summary>
public static class PrepPlanEditor
{
    /// <summary>Upsert: replace a step with the same (type, column-set), else append.</summary>
    public static void SetStep(List<PrepStep> steps, PrepStep step)
    {
        var idx = steps.FindIndex(s => SameTarget(s, step));
        if (idx >= 0) steps[idx] = step;
        else steps.Add(step);
    }

    /// <summary>
    /// Removes steps matching type (and columns, if provided). Returns removed count.
    /// An empty (non-null) <paramref name="columns"/> list is treated the same as
    /// <c>null</c> — a type-only match (no column filtering).
    /// </summary>
    public static int RemoveStep(List<PrepStep> steps, string type, IReadOnlyList<string>? columns)
    {
        var normType = Norm(type);
        var colKey = columns is { Count: > 0 } ? ColumnKeyOf(columns) : null;
        return steps.RemoveAll(s =>
            string.Equals(Norm(s.Type), normType, StringComparison.Ordinal)
            && (colKey == null || ColumnKey(s) == colKey));
    }

    /// <summary>Parses "type[:method]" — e.g. "normalize:z-score" → ("normalize","z-score").</summary>
    public static (string Type, string? Method) ParseSet(string setArg)
    {
        var i = setArg.IndexOf(':');
        return i < 0
            ? (setArg.Trim(), null)
            : (setArg[..i].Trim(), setArg[(i + 1)..].Trim());
    }

    internal static bool SameTarget(PrepStep a, PrepStep b) =>
        string.Equals(Norm(a.Type), Norm(b.Type), StringComparison.Ordinal)
        && ColumnKey(a) == ColumnKey(b);

    private static string Norm(string s) => s.ToLowerInvariant().Replace('_', '-');

    private static string ColumnKey(PrepStep s) =>
        s.Columns is { Count: > 0 } ? ColumnKeyOf(s.Columns) : (s.Column ?? "");

    private static string ColumnKeyOf(IReadOnlyList<string> cols) =>
        string.Join(",", cols.OrderBy(c => c, StringComparer.Ordinal));
}
