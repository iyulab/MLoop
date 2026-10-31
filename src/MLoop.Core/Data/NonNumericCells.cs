using System.Globalization;

namespace MLoop.Core.Data;

/// <summary>
/// Where a column the loader reads as numbers holds a cell that is not one — <c>N/A</c>, <c>--</c>, <c>unknown</c> —
/// counted over every row of the file.
/// </summary>
/// <remarks>
/// <para>
/// The loader types a column from the start of the file (its first 10,000 lines), so text further down never changes the
/// type: a numeric column stays numeric and each such cell is loaded as a missing value (<c>MissingRealsAsNaNs</c>).
/// That is a fair reading, and a silent one — a column of 14,000 values with 785 <c>N/A</c> reported no missing cells
/// anywhere and the model trained on 5.6 % less of it than the file shows. This is the count that makes it visible.
/// </para>
/// <para>
/// A cell is a number for this purpose when the loader reads it as one or as a missing value: empty, <c>?</c>,
/// <c>NA</c>, <c>NaN</c>, or anything a float parses from (invariant culture). It does not decide which column is
/// numeric — the layout does, and the caller says which columns to ask about, so a text column's cells are never parsed.
/// </para>
/// </remarks>
public sealed class NonNumericCells
{
    /// <summary>Fewest non-numeric cells that are worth a word whatever the file's size.</summary>
    public const int MinimumCells = 5;

    /// <summary>The share of a column's rows that is worth a word on a small file, where <see cref="MinimumCells"/> is never reached.</summary>
    public const double MinimumShare = 0.005;

    private readonly bool[] _asked;
    private readonly long[] _counts;
    private readonly long[] _nonEmpty;
    private readonly long[] _firstLines;
    private readonly string?[] _examples;
    private long _rows;

    /// <param name="asked">Per column of the file, whether the loader reads it as numbers (only those are counted).</param>
    public NonNumericCells(IReadOnlyList<bool> asked)
    {
        ArgumentNullException.ThrowIfNull(asked);
        _asked = [.. asked];
        _counts = new long[_asked.Length];
        _nonEmpty = new long[_asked.Length];
        _firstLines = new long[_asked.Length];
        _examples = new string?[_asked.Length];
    }

    /// <summary>Whether the loader reads <paramref name="cell"/> as a number or as a missing value.</summary>
    public static bool ReadsAsNumber(string cell)
    {
        var trimmed = cell.Trim();
        return trimmed.Length == 0
            || trimmed is "?" || trimmed.Equals("NA", StringComparison.OrdinalIgnoreCase) || trimmed.Equals("NaN", StringComparison.OrdinalIgnoreCase)
            || double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out _);
    }

    /// <summary>Counts the asked cells of one row of the file, which sits on line <paramref name="line"/> (1 is the header).</summary>
    public void Add(IReadOnlyList<string> fields, long line)
    {
        ArgumentNullException.ThrowIfNull(fields);
        _rows++;
        for (var i = 0; i < Math.Min(fields.Count, _counts.Length); i++)
        {
            if (!_asked[i])
                continue;
            if (!string.IsNullOrWhiteSpace(fields[i]))
                _nonEmpty[i]++;
            if (ReadsAsNumber(fields[i]))
                continue;
            if (_counts[i]++ == 0)
            {
                _firstLines[i] = line;
                _examples[i] = fields[i].Trim();
            }
        }
    }

    /// <summary>
    /// Whether column <paramref name="index"/> is numbers with a few cells that are not: at least <see cref="FormattedNumbers.MinimumValues"/>
    /// non-empty cells, <see cref="FormattedNumbers.MinimumShare"/> of them numbers, and at least one that is not.
    /// </summary>
    public bool MostlyNumbers(int index)
    {
        if (index < 0 || index >= _counts.Length || _counts[index] == 0)
            return false;
        var nonEmpty = _nonEmpty[index];
        return nonEmpty >= FormattedNumbers.MinimumValues && nonEmpty - _counts[index] >= FormattedNumbers.MinimumShare * nonEmpty;
    }

    /// <summary>The rows counted so far.</summary>
    public long Rows => _rows;

    /// <summary>The cells of column <paramref name="index"/> that are not numbers: how many, the first one's line and its text.</summary>
    public (long Count, long FirstLine, string? Example) Of(int index) =>
        index >= 0 && index < _counts.Length ? (_counts[index], _firstLines[index], _examples[index]) : (0, 0, null);

    /// <summary>
    /// The sentence to say of column <paramref name="index"/> named <paramref name="column"/>, or <see langword="null"/> when its
    /// non-numeric cells are too few to matter: at least <see cref="MinimumCells"/> of them, or <see cref="MinimumShare"/> of the rows.
    /// One place, so <c>train</c> and <c>info</c> say the same thing.
    /// </summary>
    public string? Describe(int index, string column)
    {
        var (count, firstLine, example) = Of(index);
        if (count == 0 || (count < MinimumCells && count < MinimumShare * _rows))
            return null;

        var share = _rows == 0 ? 0 : (double)count / _rows;
        return string.Create(CultureInfo.InvariantCulture,
            $"Column '{column}' is read as numbers, but {count:N0} of {_rows:N0} cells ({share:P1}) are not (first: '{example}' on line {firstLine:N0}) - they load as missing values");
    }
}
