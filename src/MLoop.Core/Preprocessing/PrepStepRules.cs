using FilePrepper.Tasks.WindowOps;

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

    // The aggregation methods of rolling and resample, by every spelling a config may use: the executor reads them here and both
    // validators check them here, so a method the executor cannot run is refused before it is run.
    private static readonly Dictionary<string, AggregationMethod> AggregationSpellings = new(StringComparer.OrdinalIgnoreCase)
    {
        ["mean"] = AggregationMethod.Mean,
        ["average"] = AggregationMethod.Mean,
        ["avg"] = AggregationMethod.Mean,
        ["min"] = AggregationMethod.Min,
        ["max"] = AggregationMethod.Max,
        ["sum"] = AggregationMethod.Sum,
        ["count"] = AggregationMethod.Count,
        ["std"] = AggregationMethod.Std,
        ["stdev"] = AggregationMethod.Std,
        ["var"] = AggregationMethod.Var,
        ["variance"] = AggregationMethod.Var,
        ["median"] = AggregationMethod.Median,
        ["first"] = AggregationMethod.First,
        ["last"] = AggregationMethod.Last,
    };

    /// <summary>The aggregation methods <c>rolling</c> and <c>resample</c> take (<c>method:</c>), in their canonical spelling.</summary>
    public static IReadOnlyList<string> AggregationMethods { get; } =
        ["mean", "min", "max", "sum", "count", "std", "var", "median", "first", "last"];

    /// <summary>Reads a <c>method:</c> of <c>rolling</c> or <c>resample</c>; false when it is not one of <see cref="AggregationMethods"/> (or an alias).</summary>
    public static bool TryParseAggregation(string? name, out AggregationMethod method)
    {
        method = default;
        return name is not null && AggregationSpellings.TryGetValue(name.Trim(), out method);
    }

    // The filter-rows operators, by every spelling: the first four compare numbers, the rest text.
    private static readonly HashSet<string> NumericFilterOperators = new(StringComparer.OrdinalIgnoreCase)
    {
        "gt", ">", "gte", ">=", "lt", "<", "lte", "<=",
    };

    private static readonly HashSet<string> TextFilterOperators = new(StringComparer.OrdinalIgnoreCase)
    {
        "eq", "==", "neq", "!=", "contains", "not-contains", "not_contains",
    };

    /// <summary>Whether <paramref name="op"/> is a <c>filter-rows</c> operator.</summary>
    public static bool IsFilterOperator(string? op) =>
        op is not null && (NumericFilterOperators.Contains(op.Trim()) || TextFilterOperators.Contains(op.Trim()));

    /// <summary>Whether <paramref name="op"/> compares numbers (<c>gt</c> <c>gte</c> <c>lt</c> <c>lte</c> and their symbols), so its value must be one.</summary>
    public static bool IsNumericFilterOperator(string? op) => op is not null && NumericFilterOperators.Contains(op.Trim());

    /// <summary>The step types, by their canonical names.</summary>
    public static IReadOnlyList<string> CanonicalNames { get; } =
    [
        "fill-missing", "normalize", "remove-columns", "rename-columns", "drop-duplicates",
        "extract-date", "parse-datetime", "filter-rows", "add-column", "parse-korean-time",
        "parse-excel-date", "rolling", "resample", "sample", "explode", "split-list",
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
            case "explode":
            case "split-list":
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
                if (!string.IsNullOrEmpty(step.Operator) && !IsFilterOperator(step.Operator))
                    yield return $"'filter-rows' operator '{step.Operator}' is not one of: gt, gte, lt, lte, eq, neq, contains, not-contains (or > >= < <= == !=)";
                else if (IsNumericFilterOperator(step.Operator) && !string.IsNullOrEmpty(step.Value)
                    && !double.TryParse(step.Value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out _))
                    yield return $"'filter-rows' operator '{step.Operator}' compares numbers, but value '{step.Value}' is not a number";
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
                if (!TryParseAggregation(step.Method ?? "mean", out _))
                    yield return $"'rolling' method '{step.Method}' is not one of: {string.Join(", ", AggregationMethods)}";
                break;

            case "resample":
                if (string.IsNullOrEmpty(step.TimeColumn))
                    yield return "'resample' requires 'time_column'";
                if (string.IsNullOrEmpty(step.Window))
                    yield return "'resample' requires 'window'";
                if (!hasColumns)
                    yield return "'resample' requires 'columns'";
                if (!TryParseAggregation(step.Method ?? "mean", out _))
                    yield return $"'resample' method '{step.Method}' is not one of: {string.Join(", ", AggregationMethods)}";
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
