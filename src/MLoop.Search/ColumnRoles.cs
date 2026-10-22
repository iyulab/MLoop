namespace MLoop.Search;

/// <summary>What a search does with a column it may read.</summary>
public enum ColumnRole
{
    /// <summary>A number, a boolean or a vector of numbers, used as it is.</summary>
    Numeric,

    /// <summary>A string holding categories — encoded one value per slot.</summary>
    Categorical,

    /// <summary>A string holding language — featurized as text.</summary>
    Text,

    /// <summary>Present in the data but not a feature.</summary>
    Ignored,
}

/// <summary>
/// The role of every column a search may read, and the label it learns. One table rather than a
/// list per role: a column has exactly one role, and asking "what is this column?" has one answer.
/// </summary>
/// <remarks>
/// Order is kept, and a column whose role changes moves to the end — the order of each role's
/// columns is the order in which they reach the feature vector, so a role change must not reshuffle
/// the columns that kept theirs.
/// </remarks>
public sealed class ColumnRoles
{
    private readonly List<KeyValuePair<string, ColumnRole>> _columns;

    public ColumnRoles(string label, IEnumerable<KeyValuePair<string, ColumnRole>> columns)
    {
        if (string.IsNullOrWhiteSpace(label))
            throw new ArgumentException("A search needs a label column.", nameof(label));

        Label = label;
        _columns = [];
        foreach (var (name, role) in columns)
            Put(_columns, name, role);
    }

    /// <summary>The column the search learns to predict. It has no role of its own.</summary>
    public string Label { get; }

    /// <summary>Every column with its role, in order.</summary>
    public IReadOnlyList<KeyValuePair<string, ColumnRole>> Columns => _columns;

    /// <summary>The columns with <paramref name="role"/>, in order.</summary>
    public IReadOnlyList<string> Of(ColumnRole role) =>
        [.. _columns.Where(c => c.Value == role).Select(c => c.Key)];

    /// <summary>The role of <paramref name="column"/>, or null when the table does not name it.</summary>
    public ColumnRole? RoleOf(string column)
    {
        foreach (var (name, role) in _columns)
            if (name == column)
                return role;
        return null;
    }

    /// <summary>A copy in which <paramref name="column"/> has <paramref name="role"/>, at the end.</summary>
    public ColumnRoles With(string column, ColumnRole role)
    {
        var copy = new List<KeyValuePair<string, ColumnRole>>(_columns);
        Put(copy, column, role);
        return new ColumnRoles(Label, copy);
    }

    private static void Put(List<KeyValuePair<string, ColumnRole>> columns, string name, ColumnRole role)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A column needs a name.", nameof(name));

        columns.RemoveAll(c => c.Key == name);
        columns.Add(new(name, role));
    }
}
