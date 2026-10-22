using System.Globalization;

namespace MLoop.Search.Tuning;

/// <summary>
/// A trainer's hyperparameters as a unit cube: each dimension one coordinate (a single choice none),
/// in the order of their names. A point in the cube maps to a <see cref="SearchPoint"/> and back.
/// </summary>
/// <remarks>
/// Names are ordered as ML.NET's search space orders them (<c>OrderBy(key)</c> with the default,
/// culture-aware comparer — invariant here, so the order does not move with the machine's culture);
/// the coordinate order decides which random draw moves which hyperparameter.
/// </remarks>
public sealed class SearchSpace
{
    private static readonly StringComparer NameOrder = StringComparer.Create(CultureInfo.InvariantCulture, ignoreCase: false);

    /// <param name="dimensions">The hyperparameters; names must be distinct.</param>
    public SearchSpace(IEnumerable<SearchDimension> dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        Dimensions = [.. dimensions.OrderBy(d => d.Name, NameOrder)];
        if (Dimensions.Select(d => d.Name).Distinct(StringComparer.Ordinal).Count() != Dimensions.Count)
            throw new ArgumentException("Dimension names must be distinct.", nameof(dimensions));
    }

    /// <summary>The dimensions, in coordinate order.</summary>
    public IReadOnlyList<SearchDimension> Dimensions { get; }

    /// <summary>The number of unit-space coordinates.</summary>
    public int FeatureDim => Dimensions.Sum(d => d.FeatureDim);

    /// <summary>The starting point's coordinates.</summary>
    public double[] Default => [.. Dimensions.SelectMany(d => d.Default)];

    /// <summary>The point at <paramref name="unit"/> (one coordinate per <see cref="FeatureDim"/>).</summary>
    public SearchPoint PointAt(IReadOnlyList<double> unit)
    {
        ArgumentNullException.ThrowIfNull(unit);
        if (unit.Count != FeatureDim)
            throw new ArgumentException($"Expected {FeatureDim} coordinates, got {unit.Count}.", nameof(unit));
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        var i = 0;
        foreach (var d in Dimensions)
        {
            if (d.FeatureDim == 0)
            {
                values[d.Name] = 0;
                continue;
            }
            var u = unit[i++];
            if (u < 0 || u > 1)
                throw new ArgumentOutOfRangeException(nameof(unit), $"Coordinate {u} of '{d.Name}' is outside [0, 1].");
            values[d.Name] = d.ValueAt(u);
        }
        return new SearchPoint(this, values);
    }

    /// <summary>The coordinates of <paramref name="point"/>.</summary>
    public double[] UnitOf(SearchPoint point)
    {
        ArgumentNullException.ThrowIfNull(point);
        return [.. Dimensions.Where(d => d.FeatureDim > 0).Select(d => d.UnitOf(point[d.Name]))];
    }
}

/// <summary>
/// A set of hyperparameter values — one per dimension of its <see cref="SearchSpace"/>: whole numbers
/// for integers, the index for a choice.
/// </summary>
public sealed class SearchPoint
{
    private readonly IReadOnlyDictionary<string, double> _values;

    internal SearchPoint(SearchSpace space, IReadOnlyDictionary<string, double> values)
    {
        Space = space;
        _values = values;
    }

    /// <summary>The space the point is in.</summary>
    public SearchSpace Space { get; }

    /// <summary>The value of <paramref name="name"/> (a choice: its index).</summary>
    public double this[string name] => _values[name];

    /// <summary>A whole-number hyperparameter.</summary>
    public int Int(string name) => (int)_values[name];

    /// <summary>A single-precision hyperparameter.</summary>
    public float Single(string name) => (float)_values[name];

    /// <summary>The chosen value of a choice.</summary>
    public string Choice(string name) => Space.Dimensions.First(d => d.Name == name).Choices![(int)_values[name]];

    /// <summary>The values by name, in coordinate order — what a trial records.</summary>
    public IReadOnlyList<KeyValuePair<string, double>> Values => [.. Space.Dimensions.Select(d => new KeyValuePair<string, double>(d.Name, _values[d.Name]))];

    /// <inheritdoc />
    public override string ToString()
        => string.Join(", ", Space.Dimensions.Select(d => d.Kind == DimensionKind.Choice
            ? $"{d.Name}={Choice(d.Name)}"
            : string.Create(CultureInfo.InvariantCulture, $"{d.Name}={_values[d.Name]:R}")));
}
