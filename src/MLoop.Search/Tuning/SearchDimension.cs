namespace MLoop.Search.Tuning;

/// <summary>What a hyperparameter's values are, and so how its unit-space coordinate maps back to one.</summary>
public enum DimensionKind
{
    /// <summary>A whole number — the mapped value is rounded down.</summary>
    Integer,

    /// <summary>A double-precision number.</summary>
    Double,

    /// <summary>A single-precision number — the mapped value is rounded to <see cref="float"/>.</summary>
    Single,

    /// <summary>One of a list of named choices — the value is the choice's index.</summary>
    Choice,
}

/// <summary>
/// One hyperparameter of a trainer's search space: its range (or choices), whether the range is
/// searched on a log scale, and its starting value. The tuner moves in the unit cube; this maps a
/// coordinate in [0, 1] to a value and back.
/// </summary>
/// <remarks>
/// The arithmetic follows ML.NET's search space (<c>Microsoft.ML.SearchSpace</c> <c>UniformNumericOption</c>,
/// <c>ChoiceOption</c>) operation for operation — <c>x / (max − min) − min / (max − min)</c>, not the
/// algebraically equal <c>(x − min) / (max − min)</c> — so a tuner driven through this space proposes
/// exactly what the upstream one proposed (a different last bit floors an integer to its neighbour).
/// </remarks>
public sealed record SearchDimension
{
    private SearchDimension(string name, DimensionKind kind, double min, double max, bool logBase, IReadOnlyList<string>? choices)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (kind != DimensionKind.Choice && !(max > min))
            throw new ArgumentException($"'{name}': max ({max}) must be larger than min ({min}).");
        if (logBase && !(min > 0))
            throw new ArgumentException($"'{name}': a log-scale range needs min > 0.");
        Name = name;
        Kind = kind;
        Min = min;
        Max = max;
        LogBase = logBase;
        Choices = choices;
    }

    /// <summary>The hyperparameter's name — the trainer option it sets.</summary>
    public string Name { get; }

    /// <summary>What its values are.</summary>
    public DimensionKind Kind { get; }

    /// <summary>Lowest value (a choice: 0).</summary>
    public double Min { get; }

    /// <summary>Highest value (a choice: the number of choices).</summary>
    public double Max { get; }

    /// <summary>Whether the range is searched on a log scale.</summary>
    public bool LogBase { get; }

    /// <summary>The choices, for <see cref="DimensionKind.Choice"/>.</summary>
    public IReadOnlyList<string>? Choices { get; }

    /// <summary>The starting value's unit-space coordinates (empty for a single choice).</summary>
    public IReadOnlyList<double> Default { get; private init; } = [];

    /// <summary>How many unit-space coordinates it takes: 1, or 0 for a choice of one.</summary>
    public int FeatureDim => Kind == DimensionKind.Choice && Choices!.Count == 1 ? 0 : 1;

    /// <summary>A whole-number hyperparameter in [<paramref name="min"/>, <paramref name="max"/>].</summary>
    public static SearchDimension Integer(string name, int min, int max, bool logBase = false, int? defaultValue = null)
    {
        var d = new SearchDimension(name, DimensionKind.Integer, min, max, logBase, null);
        return d with { Default = defaultValue is { } v ? [d.UnitOf(v)] : [0.0] };
    }

    /// <summary>A double-precision hyperparameter in [<paramref name="min"/>, <paramref name="max"/>].</summary>
    public static SearchDimension Double(string name, double min, double max, bool logBase = false, double? defaultValue = null)
    {
        var d = new SearchDimension(name, DimensionKind.Double, min, max, logBase, null);
        return d with { Default = defaultValue is { } v ? [d.UnitOf(v)] : [0.0] };
    }

    /// <summary>A single-precision hyperparameter in [<paramref name="min"/>, <paramref name="max"/>].</summary>
    public static SearchDimension Single(string name, float min, float max, bool logBase = false, float? defaultValue = null)
    {
        var d = new SearchDimension(name, DimensionKind.Single, min, max, logBase, null);
        return d with { Default = defaultValue is { } v ? [d.UnitOf(v)] : [0.0] };
    }

    /// <summary>One of <paramref name="choices"/>, starting at <paramref name="defaultChoice"/> (or the first).</summary>
    public static SearchDimension Choice(string name, IReadOnlyList<string> choices, string? defaultChoice = null)
    {
        ArgumentNullException.ThrowIfNull(choices);
        if (choices.Count == 0 || choices.Distinct(StringComparer.Ordinal).Count() != choices.Count)
            throw new ArgumentException($"'{name}': choices must be non-empty and distinct.");
        var d = new SearchDimension(name, DimensionKind.Choice, 0, choices.Count, false, [.. choices]);
        if (d.FeatureDim == 0)
            return d;
        if (defaultChoice is null)
            return d with { Default = [0.0] };
        var index = IndexOfChoice(choices, defaultChoice, name);
        return d with { Default = [d.UnitOf(index)] };
    }

    /// <summary>The unit-space coordinate of <paramref name="value"/> (a choice: its index).</summary>
    public double UnitOf(double value)
    {
        // A choice maps its index as a single-precision number; a single maps its float value.
        if (Kind is DimensionKind.Choice or DimensionKind.Single)
            value = Convert.ToDouble((float)value);
        if (LogBase)
        {
            var logMax = Math.Log(Max);
            var logMin = Math.Log(Min);
            var logX = Math.Log(value);
            return logX / (logMax - logMin) - logMin / (logMax - logMin);
        }
        return value / (Max - Min) - Min / (Max - Min);
    }

    /// <summary>The value at unit-space coordinate <paramref name="unit"/> (a choice: its index).</summary>
    public double ValueAt(double unit)
    {
        var value = LogBase ? Min * Math.Pow(Max / Min, unit) : Min + (Max - Min) * unit;
        switch (Kind)
        {
            case DimensionKind.Integer:
                return Convert.ToInt32(Math.Floor(value));
            case DimensionKind.Single:
                return Convert.ToSingle(value);
            case DimensionKind.Choice:
                var index = Convert.ToInt32(Math.Floor(Convert.ToSingle(value)));
                return Math.Min(index, Choices!.Count - 1);
            default:
                return value;
        }
    }

    /// <summary>The index of <paramref name="choice"/> among <see cref="Choices"/>.</summary>
    public int IndexOf(string choice) => IndexOfChoice(Choices ?? throw new InvalidOperationException($"'{Name}' is not a choice."), choice, Name);

    private static int IndexOfChoice(IReadOnlyList<string> choices, string choice, string name)
    {
        for (var i = 0; i < choices.Count; i++)
            if (string.Equals(choices[i], choice, StringComparison.Ordinal))
                return i;
        throw new ArgumentException($"'{name}': '{choice}' is not one of the choices.");
    }
}
