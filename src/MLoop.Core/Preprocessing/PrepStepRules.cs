namespace MLoop.Core.Preprocessing;

/// <summary>
/// The prep step vocabulary and what each step needs — read by the executor that runs the steps
/// and by both validators (<c>mloop validate</c>, <c>mloop prep run</c>).
/// </summary>
/// <remarks>
/// The step names used to live in three lists — the executor's dispatch and a copy in each
/// validator — and they had drifted: <c>mloop validate</c> refused a <c>sample</c> step that
/// <c>mloop prep run</c> accepted and ran.
/// </remarks>
public static class PrepStepRules
{
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["scale"] = "normalize",
        ["data-sampling"] = "sample",
    };

    /// <summary>The step types, by their canonical names.</summary>
    public static IReadOnlyList<string> CanonicalNames { get; } =
    [
        "fill-missing", "normalize", "remove-columns", "rename-columns", "drop-duplicates",
        "extract-date", "parse-datetime", "filter-rows", "add-column", "parse-korean-time",
        "parse-excel-date", "rolling", "resample", "sample",
    ];

    /// <summary>
    /// The canonical name of <paramref name="type"/> — case-insensitive, <c>_</c> read as <c>-</c>,
    /// aliases resolved (<c>scale</c> is <c>normalize</c>, <c>data-sampling</c> is <c>sample</c>) —
    /// or <c>null</c> when it is not a step type.
    /// </summary>
    public static string? Canonical(string? type)
    {
        if (string.IsNullOrWhiteSpace(type))
            return null;
        var name = type.Trim().ToLowerInvariant().Replace('_', '-');
        if (Aliases.TryGetValue(name, out var target))
            name = target;
        return CanonicalNames.Contains(name) ? name : null;
    }

    /// <summary>The message for a step whose type is not one of <see cref="CanonicalNames"/>.</summary>
    public static string UnknownType(string type) =>
        $"Unknown prep step type '{type}'. Valid types: {string.Join(", ", CanonicalNames)}";

    /// <summary>
    /// What <paramref name="step"/> lacks for its type, one message each (empty when it has what it
    /// needs or its type is unknown — <see cref="UnknownType"/> reports that).
    /// </summary>
    public static IEnumerable<string> Problems(PrepStep step)
    {
        var type = Canonical(step.Type);
        var hasColumns = step.Columns is { Count: > 0 } || !string.IsNullOrEmpty(step.Column);
        switch (type)
        {
            case "fill-missing":
            case "normalize":
            case "remove-columns":
                if (!hasColumns)
                    yield return $"'{step.Type}' requires 'columns' or 'column'";
                break;

            case "rename-columns":
                if (step.Mapping is not { Count: > 0 })
                    yield return "'rename-columns' requires 'mapping'";
                break;

            case "extract-date":
            case "parse-korean-time":
            case "parse-excel-date":
                if (string.IsNullOrEmpty(step.Column))
                    yield return $"'{step.Type}' requires 'column'";
                break;

            case "parse-datetime":
                if (string.IsNullOrEmpty(step.Column))
                    yield return "'parse-datetime' requires 'column'";
                if (string.IsNullOrEmpty(step.Format))
                    yield return "'parse-datetime' requires 'format'";
                break;

            case "filter-rows":
                if (string.IsNullOrEmpty(step.Column))
                    yield return "'filter-rows' requires 'column'";
                if (string.IsNullOrEmpty(step.Operator))
                    yield return "'filter-rows' requires 'operator'";
                if (string.IsNullOrEmpty(step.Value))
                    yield return "'filter-rows' requires 'value'";
                break;

            case "add-column":
                if (string.IsNullOrEmpty(step.Column))
                    yield return "'add-column' requires 'column'";
                if (string.IsNullOrEmpty(step.Value) && string.IsNullOrEmpty(step.Expression))
                    yield return "'add-column' requires 'value' or 'expression'";
                break;

            case "rolling":
                if (step.WindowSize < 1)
                    yield return "'rolling' requires 'window_size' > 0";
                if (!hasColumns)
                    yield return "'rolling' requires 'columns'";
                break;

            case "resample":
                if (string.IsNullOrEmpty(step.TimeColumn))
                    yield return "'resample' requires 'time_column'";
                if (string.IsNullOrEmpty(step.Window))
                    yield return "'resample' requires 'window'";
                if (!hasColumns)
                    yield return "'resample' requires 'columns'";
                break;

            case "sample":
                if (step.Count <= 0)
                    yield return "'sample' requires 'count' > 0";
                var method = (step.Method ?? "random").ToLowerInvariant();
                if (method is not ("random" or "stratified"))
                    yield return "'sample' method must be 'random' or 'stratified'";
                if (method == "stratified" && string.IsNullOrEmpty(step.Column))
                    yield return "'sample' with stratified method requires 'column'";
                break;
        }
    }
}
