namespace MLoop.Core.Data;

/// <summary>
/// Which lines of a delimited file are rows. Training and sampling read a file through a loader that skips a blank line, so a
/// report that counted one would state more rows than anything reads, with a row of missing values to go with each. A file of
/// one column is the exception: a blank line there is a record of one empty value - a gap in a series - the way the series
/// commands read it. Every place that counts the lines of a table asks here; two loops that each decided had disagreed (a
/// table with two blank lines after three rows was 5 rows in <c>info</c>, 3 in <c>sample</c>).
/// </summary>
public static class TableRows
{
    /// <summary>Whether <paramref name="line"/>, a line after the header of a file of <paramref name="columnCount"/> columns, is a row.</summary>
    public static bool IsRow(string line, int columnCount)
    {
        ArgumentNullException.ThrowIfNull(line);
        return columnCount == 1 || !string.IsNullOrWhiteSpace(line);
    }

    /// <summary>
    /// The records of a table file, counted by reading it as CSV: a quoted field that holds line breaks is one record, a blank line
    /// is none. Where the lines are not yet folded (<see cref="IsRow"/> sees lines), a count of lines is more than the table holds.
    /// </summary>
    public static async Task<int> CountAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        var count = 0;
        await foreach (var _ in new CsvHelperImpl().ReadRowsAsync(path, cancellationToken: cancellationToken).ConfigureAwait(false))
            count++;
        return count;
    }
}
